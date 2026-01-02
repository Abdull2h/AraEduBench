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
يرجى توليد سؤال مناسب للمادة والمستوى المحددين، مع تقديم الإجابة النموذجية. نوع السؤال هو: {question_type}.
إذا كان السؤال قصير الإجابة، يمكن تضمين خطوات حسابية أو أكواد عند الحاجة. لا تضف أي محتوى خارج المطلوب.

المادة: {subject}
المستوى: {level}

أعد النتيجة بصيغة JSON فقط:
"Question": "نص السؤال بالتفصيل (مع الخيارات إن وُجدت)"
"Answer": ""
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
    """Validate whether the API response is valid JSON with required fields."""
    try:
        data = json.loads(response)

        # Check if required fields exist
        required_keys = ["Question", "Answer"]
        if not all(key in data for key in required_keys):
            print(f"Validation failed: Missing required fields. Required fields: {required_keys}")
            return False

        # Check field content is not empty or invalid
        for key in required_keys:
            field_content = str(data[key]).strip()  # Convert to string and remove whitespace
            if not field_content:  # Check if empty
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
# Single Q&A Generation
# =========================

def get_question_and_answer(subject: str, level: str, question_type: str) -> Optional[dict]:
    """Generate a question and its answer in Arabic."""
    prompt = prompt_template_ar.format(subject=subject, level=level, question_type=question_type)

    response = send_request(prompt)
    if not response or not validate_response(response):
        print(f"Generation failed: {subject}-{level}-{question_type}-ar")
        return None

    try:
        qa_data = json.loads(response)

        def process_field(field_name: str):
            value = qa_data.get(field_name, "")
            if isinstance(value, list):
                return " ".join(str(item).strip() for item in value)
            elif isinstance(value, str):
                return value.strip()
            else:
                return str(value).strip()

        result = {
            "Subject": subject,
            "Education Level": level,
            "Question Type": question_type,
            "Language": "ar",
            "Question": process_field("Question"),
            "Answer": process_field("Answer"),
            "Generation Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Handle optional fields
        if "Knowledge Point" in qa_data:
            result["Knowledge Point"] = process_field("Knowledge Point")
        if "Solution Approach" in qa_data:
            result["Solution Approach"] = process_field("Solution Approach")

        return result

    except Exception as e:
        print(f"Parsing failed: Unable to extract data - {e}")
        return None

# =========================
# Batch Processing
# =========================

def process_subjects(subject_list: List[Tuple[str, str]], output_file: str):
    """Process all subject and education levels, generating 5 entries per question type."""
    # Question types in Arabic, preserving original semantics:
    # Single Choice, Multiple Choice, Short Answer
    question_types = [
        "اختيار من متعدد (إجابة واحدة صحيحة)",     # Single Choice
        "اختيار من متعدد (أكثر من إجابة صحيحة)",   # Multiple Choice
        "سؤال قصير الإجابة",                        # Short Answer
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'a', encoding='utf-8') as outfile:
        for idx, (subject, level) in enumerate(subject_list, 1):
            print(f"Processing [{idx}/{len(subject_list)}] {subject}-{level}-ar")

            for q_type in question_types:
                for repeat in range(5):
                    result = get_question_and_answer(subject, level, q_type)

                    if result:
                        try:
                            # Only write non-empty fields
                            valid_result = {k: v for k, v in result.items() if v}
                            outfile.write(json.dumps(valid_result, ensure_ascii=False) + '\n')
                            outfile.flush()

                            print(f"Saved successfully: {subject}-{level}-{q_type}-ar ({repeat + 1}/5)")
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
    """Main function: Arabic-only question & answer generation."""
    output_dir = os.getenv("OUTPUT_DIR", "./data")
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"QA_ar_{current_time}.jsonl")

    subject_list = load_subject_list()

    process_subjects(subject_list, output_file)

    print(f"Processing completed, results saved to: {output_file}")


if __name__ == "__main__":
    main()
