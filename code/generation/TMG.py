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
أنت مسؤول عن مساعدة المعلمين في توليد المواد التعليمية. بناءً على الفصل الدراسي في الكتاب أو نقطة معرفية معينة، قم تلقائيًا بإنشاء خطة درس مُنظَّمة تتضمن الأهداف التعليمية، والنقاط الأساسية والصعوبات، وتصميم الأنشطة الصفية.
يرجى توليد نقطة معرفية مناسبة بحرية للمادة والمستوى المحددين، وإنشاء خطة درس مُنظَّمة تلقائيًا تشمل الأهداف التعليمية، والنقاط الأساسية والصعوبات، وتصميم الأنشطة الصفية. نوع السؤال هو {question_type}.
إذا كان نوع السؤال سؤالاً قصير الإجابة، فيجب تضمين الأكواد أو الحسابات الرياضية عند الحاجة لبعض المواد. لا تضف أي محتوى إضافي.

المادة: {subject}
مستوى الصعوبة: {level}

- استخدم اللغة العربية في جميع القيم، مع بقاء أسماء الحقول (المفاتيح) بالإنجليزية كما هي.
أعد النتيجة بصيغة JSON:
"Knowledge Point": ""
"Teaching Materials": ""
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
        required_keys = ["Knowledge Point", "Teaching Materials"]
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
# Single Generation
# =========================

def get_question_and_answer(subject: str, level: str, question_type: str) -> Optional[dict]:
    """Get a knowledge point and its teaching materials in Arabic."""
    prompt = prompt_template_ar.format(subject=subject, level=level, question_type=question_type)

    response = send_request(prompt)
    if not response or not validate_response(response):
        print(f"Generation failed: {subject}-{level}-{question_type}-ar")
        return None

    try:
        qa_data = json.loads(response)
        return {
            "Subject": subject,
            "Level": level,
            "Question Type": question_type,
            "Language": "ar",
            "Knowledge Point": qa_data.get("Knowledge Point", "") or qa_data.get("知识点", ""),
            "Teaching Materials": qa_data.get("Teaching Materials", "") or qa_data.get("教学素材", ""),
            "Generation Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    except Exception as e:
        print(f"Parsing failed: Unable to extract data - {e}")
        return None

# =========================
# Progress Tracking
# =========================

def load_processed_combinations(output_file: str) -> set:
    """Load already processed combinations from the output file."""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    data = json.loads(line.strip())
                    combination = (data["Subject"], data["Level"], data["Question Type"])
                    processed.add(combination)
                except Exception as e:
                    print(f"Failed to parse line in progress file: {e}")
    return processed

# =========================
# Batch Processing
# =========================

def process_subjects(subject_list: List[Tuple[str, str]], output_file: str):
    """Process all subject and difficulty combinations, generating five results per question type."""
    # Question types in Arabic, preserving original semantics:
    # Single Choice, Multiple Choice, Short Answer
    question_types = [
        "اختيار من متعدد (إجابة واحدة صحيحة)",     # Single Choice
        "اختيار من متعدد (أكثر من إجابة صحيحة)",   # Multiple Choice
        "سؤال قصير الإجابة",                        # Short Answer
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    processed_combinations = load_processed_combinations(output_file)

    with open(output_file, 'a', encoding='utf-8') as outfile:
        for idx, (subject, level) in enumerate(subject_list, 1):
            print(f"Processing [{idx}/{len(subject_list)}] {subject}-{level}-ar")

            for q_type in question_types:
                combination = (subject, level, q_type)

                if combination in processed_combinations:
                    print(f"Skipping already processed: {combination}")
                    continue

                successful_attempts = 0

                while successful_attempts < 5:
                    result = get_question_and_answer(subject, level, q_type)

                    if result:
                        try:
                            result["Generation Index"] = successful_attempts + 1
                            result["Generation Time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                            outfile.flush()

                            successful_attempts += 1
                            print(f"Saved successfully: {subject}-{level}-{q_type}-ar ({successful_attempts}/5)")
                            processed_combinations.add(combination)
                        except Exception as e:
                            print(f"File writing failed: {e}")
                    else:
                        print(f"Generation failed: {subject}-{level}-{q_type}-ar")

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
    """Main function: Arabic-only teaching material generation."""
    output_dir = os.getenv("OUTPUT_DIR", "./data")
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"TMG_ar_{current_time}.jsonl")

    subject_list = load_subject_list()

    process_subjects(subject_list, output_file)

    print(f"Processing completed, results saved to: {output_file}")


if __name__ == "__main__":
    main()
