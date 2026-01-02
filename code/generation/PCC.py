import os
import json
import re
import time
from datetime import datetime
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Environment & OpenAI Client
# =========================

# Load .env from project root (adjust the path if needed)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE")  

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE or None
)

# =========================
# Arabic Prompt Template
# =========================

# prompt_template_ar = """
# أنت خبير في تخصيص الخدمات التعليمية. بناءً على صورة الطالب، قدم خدمات مخصصة لتحسين كفاءته في التعلم، وتشمل:

# 1. Learning Path Planning: اقتراح مسار تعلم وترتيب موضوعات أو مهارات مناسبة لمستوى الطالب وأهدافه.
# 2. Personalized Recommendations: توصيات بتمارين أو مواد قراءة تناسب نقاط ضعفه وعاداته الدراسية.

# يرجى توليد صورة طالب مناسبة للمادة: {subject}، ثم تقديم مسار تعلم مقترح وتوصيات مخصصة.

# أعد المخرجات بصيغة JSON فقط:
# "Student Profile": (object)
# "Learning Path Planning": (array: strings or objects)
# "Personalized Recommendations": (object or array)
# """

prompt_template_ar = """
أنت خبير في تخصيص الخدمات التعليمية. بناءً على صورة الطالب في مادة {subject} ومستوى {level}،
أنشئ ملفًا منظمًا يتكوّن من ثلاثة أجزاء رئيسية، مع الحفاظ على أسماء الحقول بالإنجليزية والقيم باللغة العربية فقط.
يجب أن يكون الإخراج بصيغة JSON بالهيكل التالي تمامًا (بدون أي حقول إضافية):
"Student Profile": (object)
"Learning Path Planning": (array: strings or objects)
"Personalized Recommendations": (object or array)
التعليمات المهمة:
- استخدم اللغة العربية في جميع القيم النصية فقط، مع بقاء أسماء الحقول (المفاتيح) بالإنجليزية كما هي.
- املأ جميع الحقول بمحتوى مناسب لصورة طالب حقيقية.
- لا تضف أي حقول أو نصوص خارج JSON المطلوب.
"""



# =========================
# OpenAI Request
# =========================

def send_request(prompt: str) -> Optional[str]:
    """Send a request to the OpenAI API and return the result as raw text."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )

        if response.choices:
            response_text = response.choices[0].message.content
            # Clean up the response by removing possible markdown code blocks
            cleaned_response = re.sub(r'```(json)?\s*|\s*```', '', response_text).strip()
            print(cleaned_response)
            return cleaned_response

        return None

    except Exception as e:
        print(f"API request failed: {e}")
        return None

# =========================
# JSON Fixer
# =========================

def fix_json(response: str) -> Optional[str]:
    """Attempt to fix common JSON formatting errors."""
    try:
        # Try parsing directly
        json.loads(response)
        return response
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}, attempting to fix...")

    try:
        # Replace single quotes with double quotes
        fixed_response = response.replace("'", '"')
        json.loads(fixed_response)
        return fixed_response
    except json.JSONDecodeError:
        pass

    try:
        # Attempt to fix missing commas or colons (very heuristic)
        fixed_response = re.sub(r'(?<=[}\]"\'\w])\s*(?=[{\["\'\w])', ', ', response)
        fixed_response = re.sub(r'(?<=[:])\s*(?=[{\["\'\w])', ' ', fixed_response)
        json.loads(fixed_response)
        return fixed_response
    except json.JSONDecodeError:
        pass

    print("Unable to fix JSON formatting error")
    return None

# =========================
# Validation
# =========================

def validate_response(response: str) -> bool:
    """Validate whether the API response is valid."""
    try:
        # Attempt to fix JSON format
        fixed_response = fix_json(response)
        if not fixed_response:
            return False

        data = json.loads(fixed_response)

        # Check if required fields exist
        required_keys = [
            "Student Profile",
            "Learning Path Planning",
            "Personalized Recommendations",
        ]
        if not all(key in data for key in required_keys):
            print("Validation failed: Missing required fields")
            return False

        # Check if fields are empty
        if any(not data[key] for key in required_keys):
            print("Validation failed: Fields are empty")
            return False

        # Lenient validation: Accept dict, list, or string types
        if not isinstance(data["Student Profile"], (dict, list, str)):
            print(
                f"Validation failed: 'Student Profile' field type not supported - "
                f"{type(data['Student Profile'])}"
            )
            return False

        if not isinstance(data["Learning Path Planning"], (list, dict, str)):
            print(
                f"Validation failed: 'Learning Path Planning' field type not supported - "
                f"{type(data['Learning Path Planning'])}"
            )
            return False

        if not isinstance(data["Personalized Recommendations"], (dict, list, str)):
            print(
                f"Validation failed: 'Personalized Recommendations' field type not supported - "
                f"{type(data['Personalized Recommendations'])}"
            )
            return False

        return True
    except Exception as e:
        print(f"Validation failed: JSON parsing error - {e}")
        return False

# =========================
# Single Generation
# =========================

def get_student_profile(subject: str, level: str) -> Optional[dict]:
    """Generate a student profile with learning plan and recommendations in Arabic."""
    prompt = prompt_template_ar.format(subject=subject, level=level)

    response = send_request(prompt)
    if not response or not validate_response(response):
        print(f"Generation failed: {subject}-{level}-ar")
        return None

    try:
        fixed_response = fix_json(response) or response
        profile_data = json.loads(fixed_response)

        def process_field(field_name: str):
            value = profile_data.get(field_name, "")
            if isinstance(value, (dict, list)):
                return value
            elif isinstance(value, str):
                return value.strip()
            else:
                return str(value).strip()

        return {
            "Subject": subject,
            "Education Level": level,
            "Language": "ar",
            "Student Profile": process_field("Student Profile"),
            "Learning Path Planning": process_field("Learning Path Planning"),
            "Personalized Recommendations": process_field("Personalized Recommendations"),
            "Generation Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    except Exception as e:
        print(f"Parsing failed: Unable to extract data - {e}")
        return None

# =========================
# Batch Processing
# =========================

def process_subjects(subject_list: List[Tuple[str, str]], output_file: str):
    """Process all subject and education levels, generating 3 entries per subject-level pair."""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'a', encoding='utf-8') as outfile:
        for idx, (subject, level) in enumerate(subject_list, 1):
            print(f"Processing [{idx}/{len(subject_list)}] {subject}-{level}-ar")

            successful_attempts = 0
            while successful_attempts < 3:
                result = get_student_profile(subject, level)

                if result:
                    try:
                        result["Generation Index"] = successful_attempts + 1
                        result["Generation Time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                        outfile.flush()

                        successful_attempts += 1
                        print(f"Saved successfully: {subject}-{level}-ar ({successful_attempts}/3)")
                    except Exception as e:
                        print(f"File writing failed: {e}")
                else:
                    print(f"Generation failed: {subject}-{level}-ar")

                time.sleep(1)

# =========================
# Subject List (Arabic)
# =========================

def load_subject_list() -> List[Tuple[str, str]]:
    """إرجاع قائمة بالمواد والمستويات الدراسية باللغة العربية"""
    return [
        # التعليم العام
        *[
            (subj, level)
            for subj in [
                "اللغة العربية",
                "الرياضيات",
                "اللغة الإنجليزية",
                "الفيزياء",
                "الكيمياء",
                "الأحياء",
                "التاريخ",
                "الجغرافيا",
            ]
            for level in [
                "المرحلة الابتدائية",
                "المرحلة المتوسطة",
                "المرحلة الثانوية",
            ]
        ],

        # التعليم العالي
        *[
            (subj, level)
            for subj in [
                "الرياضيات",
                "الفيزياء",
                "الكيمياء",
                "الأحياء",
                "علوم الحاسب",
                "الأتمتة",
                "الاستزراع المائي",
                "علوم المحاصيل",
                "الاقتصاد التطبيقي",
                "الاقتصاد النظري",
                "علم التربية العام",
                "التربية البدنية",
                "القانون",
                "إدارة الأعمال",
                "الإدارة العامة",
                "الطب الأساسي",
                "الطب السريري",
                "علم الاجتماع",
                "الأدب والفنون",
                "علم النفس",
                "التاريخ",
                "العلوم العسكرية",
            ]
            for level in [
                "بكالوريوس",
                "ماجستير",
                "دكتوراه",
            ]
        ],
    ]


# =========================
# Main
# =========================

def main():
    """Main function: Arabic-only student profile & personalization generation."""
    output_dir = os.getenv("OUTPUT_DIR", "./data")
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"PCC_ar_{current_time}.jsonl")

    subject_list = load_subject_list()

    process_subjects(subject_list, output_file)

    print(f"Processing completed, results saved to: {output_file}")


if __name__ == "__main__":
    main()
