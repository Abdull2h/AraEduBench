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
API_BASE = os.getenv("OPENAI_API_BASE", "")  # optional; can be empty

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE or None
)

# =========================
# Arabic Prompt Template
# =========================

prompt_template_ar = """
أنت وكيل ذكي قادر على توليد محتوى أو مهام تعلم مخصّصة للطلاب بناءً على الفروق الفردية بينهم.
يرجى إنشاء صورة طالب يدرس مادة {subject}، ثم توليد محتوى تعلم شخصي وفق النقاط التالية:

One-on-one: تمارين أو مواد قراءة مخصّصة لطالب واحد بحسب صفاته.
Tiered Teaching: أهداف تعلم وطرائق تدريس وتقويم وواجبات مختلفة لثلاثة مستويات من الطلاب.
Other: تصميم بيانات أو مؤشرات تقييم مناسبة للفرد، لمجموعة تعلم، وللفصل كامل.

لا تضف أي محتوى خارج المطلوب.
استخدم اللغة العربية في جميع القيم النصية فقط، مع بقاء أسماء الحقول (المفاتيح) بالإنجليزية كما هي.

المستوى التعليمي: {level}

أعد المخرجات بصيغة JSON فقط:
"Student Profile": (object)
"Personalized Learning Content/Task": (object)
"""


# =========================
# OpenAI Request
# =========================

def send_request(prompt: str) -> Optional[str]:
    """Send a request to the OpenAI API and return the result."""
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
# Validation
# =========================

def validate_response(response: str) -> bool:
    """Validate whether the API response is valid."""
    try:
        data = json.loads(response)

        # Check if required fields exist
        required_keys = ["Student Profile", "Personalized Learning Content/Task"]
        if not all(key in data for key in required_keys):
            print(f"Validation failed: Missing required fields. Required fields: {required_keys}")
            return False

        # Check 'Student Profile' type and non-emptiness
        student_profile = data["Student Profile"]
        if isinstance(student_profile, str):
            if not student_profile.strip():
                print("Validation failed: Field 'Student Profile' is an empty string")
                return False
        elif isinstance(student_profile, dict):
            if not student_profile:
                print("Validation failed: Field 'Student Profile' is an empty dictionary")
                return False
        else:
            print("Validation failed: Field 'Student Profile' has incorrect type, should be a string or dictionary")
            return False

        # Check 'Personalized Learning Content/Task' type and content
        personalized_content = data["Personalized Learning Content/Task"]
        if not isinstance(personalized_content, dict):
            print("Validation failed: Field 'Personalized Learning Content/Task' has incorrect type, should be a dictionary")
            return False

        # Define required nested keys (case-insensitive)
        nested_required_keys = ["One-on-one", "Tiered Teaching", "Other"]

        # Normalize keys to lowercase for comparison
        normalized_personalized_content = {key.lower(): value for key, value in personalized_content.items()}

        # Validate nested fields
        for key in nested_required_keys:
            lower_key = key.lower()
            if lower_key not in normalized_personalized_content:
                print(f"Validation failed: Missing nested field. Required field: {key}")
                return False

            nested_value = normalized_personalized_content[lower_key]
            if not isinstance(nested_value, dict):
                print(f"Validation failed: Field '{key}' has incorrect type, should be a dictionary")
                return False
            if not nested_value:
                print(f"Validation failed: Field '{key}' is empty")
                return False

        return True

    except json.JSONDecodeError as e:
        print(f"Validation failed: JSON parsing error - {e}")
        return False
    except Exception as e:
        print(f"Validation failed: Unknown error - {e}")
        return False

# =========================
# Single Generation
# =========================

def get_question_and_answer(subject: str, level: str) -> Optional[dict]:
    """Generate student profile and personalized learning content in Arabic."""
    prompt = prompt_template_ar.format(subject=subject, level=level)

    response = send_request(prompt)
    if not response or not validate_response(response):
        print(f"Generation failed: {subject}-{level}-ar")
        return None

    # Parse JSON data
    try:
        qa_data = json.loads(response)
        return {
            "Subject": subject,
            "Education Level": level,
            "Language": "ar",
            "Student Profile": qa_data["Student Profile"],
            "Personalized Learning Content/Task": qa_data["Personalized Learning Content/Task"],
            "Generation Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    except Exception as e:
        print(f"Parsing failed: Unable to extract data - {e}")
        return None

# =========================
# Batch Processing
# =========================

def process_subjects(subject_list: List[Tuple[str, str]], output_file: str):
    """Process all subject and difficulty combinations, generating one result per subject-level pair."""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'a', encoding='utf-8') as outfile:
        for idx, (subject, level) in enumerate(subject_list, 1):
            print(f"Processing [{idx}/{len(subject_list)}] {subject}-{level}-ar")

            result = get_question_and_answer(subject, level)
            if result:
                try:
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    outfile.flush()
                    print(f"Saved successfully: {subject}-{level}-ar")
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
    """Main function: Arabic-only personalized learning content/task generation."""
    output_dir = os.getenv("OUTPUT_DIR", "./data")
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"PLS_ar_{current_time}.jsonl")

    subject_list = load_subject_list()

    process_subjects(subject_list, output_file)

    print(f"Processing completed, results saved to: {output_file}")


if __name__ == "__main__":
    main()
