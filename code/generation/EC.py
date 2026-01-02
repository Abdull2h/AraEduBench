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

# Load .env from project root (adjust path if needed)
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
أنت معلم خبير تقوم بتصحيح إجابات الطلاب.  
يرجى توليد سؤال في مادة: {subject} وبمستوى: {level}، مع إنشاء إجابة خاطئة للطالب، ثم تقديم الإجابة الصحيحة مع شرح التصحيح.  
نوع السؤال هو: {question_type}.  
لا تضف أي محتوى خارج المطلوب. يجب أن تكون الإجابات باللغة العربية.

أعد النتيجة بصيغة JSON فقط:
"Question": ""
"Original Answer": ""
"Corrected Answer": ""
"Correction Explanation": ""
"""


# =========================
# OpenAI Request
# =========================

def send_request(prompt: str) -> Optional[str]:
    """Send a request to the OpenAI API and return the raw JSON string (as text)."""
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

    print("Unable to fix JSON formatting errors")
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
            "Question",
            "Original Answer",
            "Corrected Answer",
            "Correction Explanation",
        ]
        if not all(key in data for key in required_keys):
            print("Validation failed: Missing required fields")
            return False

        # Check if field types are correct
        if not all(isinstance(data[key], str) for key in required_keys):
            print("Validation failed: Field type error")
            return False

        # Check if fields are empty strings
        if any(not data[key] for key in required_keys):
            print("Validation failed: Fields are empty")
            return False

        return True
    except Exception as e:
        print(f"Validation failed: JSON parsing error - {e}")
        return False

# =========================
# Single Q&A Generation
# =========================

def get_question_and_answer(subject: str, level: str, question_type: str) -> Optional[dict]:
    """
    Generate a question, an incorrect student answer, the corrected answer,
    and an explanation in Arabic for a given subject, level, and question type.
    """
    prompt = prompt_template_ar.format(
        subject=subject,
        level=level,
        question_type=question_type,
    )

    response = send_request(prompt)
    if not response or not validate_response(response):
        print(f"Generation failed: {subject}-{level}-{question_type}-ar")
        return None

    try:
        # Reuse fixed JSON if needed
        fixed_response = fix_json(response) or response
        qa_data = json.loads(fixed_response)

        return {
            "Subject": subject,
            "Difficulty Level": level,
            "Question Type": question_type,
            "Language": "ar",
            "Question": qa_data.get("Question", ""),
            "Original Answer": qa_data.get("Original Answer", ""),
            "Corrected Answer": qa_data.get("Corrected Answer", ""),
            "Correction Explanation": qa_data.get("Correction Explanation", ""),
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
    Process all subject and difficulty combinations, generating five questions
    for each question type and saving them in a JSONL file.
    """

    # Arabic question types, matching original semantics:
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

            for question_type in question_types:
                successful_attempts = 0 

                while successful_attempts < 5: 
                    result = get_question_and_answer(subject, level, question_type)

                    if result:
                        try:
                            result["Generation Index"] = successful_attempts + 1
                            result["Generation Time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            # Write to file
                            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                            outfile.flush() 

                            successful_attempts += 1
                            print(
                                f"Saved successfully: {subject}-{level}-{question_type}-ar ({successful_attempts}/5)"
                            )
                        except Exception as e:
                            print(f"File writing failed: {e}")
                    else:
                        print(f"Generation failed: {subject}-{level}-{question_type}-ar")

                    # Delay to avoid rate limits
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
    """Main function: Arabic-only error-correction generation."""
    output_dir = os.getenv("OUTPUT_DIR", "./data")
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"EC_ar_{current_time}.jsonl")

    subject_list = load_subject_list()

    process_subjects(subject_list, output_file)

    print(f"Processing completed, results saved to: {output_file}")


if __name__ == "__main__":
    main()
