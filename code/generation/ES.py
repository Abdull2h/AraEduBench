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

prompt_template_ar = """
أنت وكيل ذكي قادر على التعرف على الحالة العاطفية للطالب، تحليل أسباب المشكلة، وتقديم نصائح مناسبة.  
يرجى توليد حوار متعدد الأدوار مع طالب يدرس مادة: {subject}، مع تحديد حالته العاطفية وتحليل سبب المشكلة، ثم تقديم نصائح داعمة.  
درجة القلق هي: {question_type}.  
لا تضف أي محتوى خارج المطلوب.

المستوى التعليمي: {level}

أعد النتيجة بصيغة JSON فقط:
"Dialogue with Student": ""
"Emotional State Analysis": ""
"Comfort and Advice": ""
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
# Validation
# =========================

def validate_response(response: str) -> bool:
    """Validate whether the API response is valid JSON with required fields."""
    try:
        data = json.loads(response)

        # Required fields (keys remain in English)
        required_keys = [
            "Dialogue with Student",
            "Emotional State Analysis",
            "Comfort and Advice",
        ]
        if not all(key in data for key in required_keys):
            print(f"Validation failed: Missing required fields. Required fields: {required_keys}")
            return False

        # Ensure content is non-empty after stripping
        for key in required_keys:
            field_content = str(data[key]).strip()
            if not field_content:
                print(f"Validation failed: Field '{key}' is empty or invalid")
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

def get_question_and_answer(subject: str, level: str, anxiety_level: str) -> Optional[dict]:
    """
    Generate emotional support dialogue based on subject, education level, and anxiety level.
    All content is in Arabic, JSON keys remain in English.
    """
    prompt = prompt_template_ar.format(
        subject=subject,
        level=level,
        question_type=anxiety_level,
    )

    response = send_request(prompt)
    if not response or not validate_response(response):
        print(f"Generation failed: {subject}-{level}-{anxiety_level}-ar")
        return None

    try:
        qa_data = json.loads(response)
        return {
            "Subject": subject,
            "Education Level": level,
            "Anxiety Level": anxiety_level,
            "Language": "ar",
            "Dialogue with Student": str(qa_data.get("Dialogue with Student", "") or qa_data.get("与学生的对话", "")),
            "Emotional State Analysis": str(qa_data.get("Emotional State Analysis", "") or qa_data.get("情绪状态分析", "")),
            "Comfort and Advice": str(qa_data.get("Comfort and Advice", "") or qa_data.get("安慰与建议", "")),
            "Generation Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    except Exception as e:
        print(f"Parsing failed: Unable to extract data - {e}")
        return None

# =========================
# Batch Processing
# =========================

def process_subjects(subject_list: List[Tuple[str, str]], output_file: str):
    """
    Process all subject and education levels, generating 4 entries per anxiety level.
    Arabic-only generation.
    """

    # Anxiety levels (question_types) in Arabic (mapping Mild / Moderate / Severe)
    question_types = [
        "قلق خفيف",    # Mild Anxiety
        "قلق متوسط",   # Moderate Anxiety
        "قلق شديد",    # Severe Anxiety
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'a', encoding='utf-8') as outfile:
        for idx, (subject, level) in enumerate(subject_list, 1):
            print(f"Processing [{idx}/{len(subject_list)}] {subject}-{level}-ar")

            for q_type in question_types:
                for repeat in range(4):
                    result = get_question_and_answer(subject, level, q_type)

                    if result:
                        try:
                            result["Generation Index"] = repeat + 1
                            result["Generation Time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                            outfile.flush()

                            print(f"Saved successfully: {subject}-{level}-{q_type}-ar ({repeat + 1}/4)")
                        except Exception as e:
                            print(f"File writing failed: {e}")
                    else:
                        print(f"Generation failed: {subject}-{level}-{q_type}-ar (Attempt {repeat + 1})")

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
    """Main function: Arabic-only emotional support generation."""
    output_dir = os.getenv("OUTPUT_DIR", "./data")
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"ES_ar_{current_time}.jsonl")

    subject_list = load_subject_list()

    process_subjects(subject_list, output_file)

    print(f"Processing completed, results saved to: {output_file}")


if __name__ == "__main__":
    main()
