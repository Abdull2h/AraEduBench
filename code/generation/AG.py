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
أنت أداة لتقييم إجابات الطلاب، ويجب أن تقوم بما يلي:
1. التقييم الموضوعي: مثل أسئلة الاختيار من متعدد، الصح والخطأ، وأسئلة الفراغ؛ ويمكنك أيضًا تقييم أسئلة الحل التفصيلي من خلال إعطاء تقسيم للدرجات أو مرجع للتصحيح.
2. التقييم الذاتي: تقييم الواجبات الكبيرة، أو التقارير، أو المشروعات العملية بناءً على أبعاد مثل: حجم العمل، درجة الاكتمال، وتطبيق المعرفة.
3. التغذية الراجعة الشخصية: تقديم ملاحظات واضحة وبنّاءة حول إجابة الطالب، تتضمن نقاط القوة، ونقاط الضعف، والفجوات المعرفية، واقتراحات لتحسين التعلم.

يرجى توليد سؤال مناسب باللغة العربية للمادة ومستوى الصعوبة التاليين، ثم توليد إجابة طالب (يمكن أن تحتوي على بعض الأخطاء الواقعية)، ثم تقييم هذه الإجابة. نوع السؤال هو: {question_type}.
إذا كان نوع السؤال "سؤال قصير الإجابة"، فيجب تضمين الأكواد أو الحسابات الرياضية عند الحاجة لبعض المواد. لا تضف أي محتوى إضافي خارج المطلوب.
يجب أن تستخدم اللغة العربية في نص السؤال، وإجابة الطالب، وتفاصيل التصحيح، والتغذية الراجعة.

المادة: {subject}
مستوى الصعوبة: {level}

أعد النتيجة بصيغة JSON بالحقول التالية فقط:
"Question": "نص السؤال باللغة العربية"
"Student's Answer": "إجابة الطالب باللغة العربية"
"Score": "درجة رقمية أو وصفية"
"Scoring Details": "تفاصيل معايير التصحيح والتحليل"
"Personalized Feedback": "تغذية راجعة شخصية باللغة العربية"
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
            # Remove ```json ... ``` wrappers if present
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
    """Validate whether the API response is a valid JSON with the required fields."""
    try:
        data = json.loads(response)

        required_keys = [
            "Question",
            "Student's Answer",
            "Score",
            "Scoring Details",
            "Personalized Feedback",
        ]

        # Check required keys
        if not all(key in data for key in required_keys):
            print(f"Validation failed: Missing required fields. Required: {required_keys}")
            return False

        # Check non-empty values
        for key in required_keys:
            if not str(data[key]).strip():
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
    """
    Generate a question, student's answer, score, and feedback in Arabic
    for a given subject, level, and question type.
    """
    prompt = prompt_template_ar.format(
        subject=subject,
        level=level,
        question_type=question_type,
    )

    response = send_request(prompt)
    if not response or not validate_response(response):
        print(f"Generation failed: {subject}-{level}-{question_type}")
        return None

    try:
        qa_data = json.loads(response)
        return {
            "Subject": subject,
            "Level": level,
            "Question Type": question_type,
            "Language": "ar",
            "Question": qa_data.get("Question", ""),
            "Student's Answer": qa_data.get("Student's Answer", ""),
            "Score": qa_data.get("Score", ""),
            "Scoring Details": qa_data.get("Scoring Details", ""),
            "Personalized Feedback": qa_data.get("Personalized Feedback", ""),
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
    Process all subject and difficulty combinations, generating four results
    per question type and saving them in a JSONL file.
    """
    question_types = [
        "اختيار من متعدد",
        "صح وخطأ",
        "سؤال قصير الإجابة",
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as outfile:
        for idx, (subject, level) in enumerate(subject_list, 1):
            print(f"Processing [{idx}/{len(subject_list)}] {subject} - {level} - ar")

            for q_type in question_types:
                for repeat in range(4):  
                    result = get_question_and_answer(subject, level, q_type)

                    if result:
                        try:
                            result["Generation Index"] = repeat + 1
                            result["Generation Time"] = datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )

                            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                            outfile.flush()

                            print(
                                f"Saved successfully: {subject} - {level} - {q_type} - ar ({repeat + 1}/3)"
                            )
                        except Exception as e:
                            print(f"File writing failed: {e}")
                    else:
                        print(
                            f"Generation failed: {subject} - {level} - {q_type} - ar (Attempt {repeat + 1})"
                        )

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
    """Main function: Arabic-only generation."""
    output_dir = os.getenv("OUTPUT_DIR", "./data")
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"AG_ar_{current_time}.jsonl")

    subject_list = load_subject_list()

    process_subjects(subject_list, output_file)

    print(f"Processing completed, results saved to: {output_file}")


if __name__ == "__main__":
    main()
