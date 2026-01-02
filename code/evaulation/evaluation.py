import os
import sys
import json
import random
from dotenv import load_dotenv
from jsonschema import validate, ValidationError
import pandas as pd
from openai import OpenAI
import re
from datetime import datetime

# ==========================================================
# LOAD ENV
# ==========================================================
load_dotenv()

PROVIDER = "deepseek"  # "deepseek" or "openai"

if (PROVIDER == 'openai'):
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")


if (PROVIDER == 'deepseek'):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_base = os.getenv("DEEPSEEK_API_BASE")


if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError("DEEPSEEK_API_KEY not found in .env file")

client = OpenAI(api_key=api_key, base_url=api_base or None)

# ==========================================================
# CONFIG
# ==========================================================
MODEL_NAME = "gpt-4o" # or "deepseek-chat"
TEMPERATURE = 0.0
EVALUATOR_NAME = MODEL_NAME

# ==========================================================
# METRICS REGISTRY
# ==========================================================
METRICS_REGISTRY = {
    "EC": [
        {"code": "IFTC", "description": "Instruction Following & Task Completion: This sub-metric measures the model's ability to accurately interpret and complete assigned tasks, such as solving problems, correcting errors, or generating questions, while adhering to the required output format and constraints."},
        {"code": "RTC", "description": "Role & Tone Consistency: This dimension evaluates whether the language style, tone, and level of expertise in the response are appropriate for the designated role (e.g., teacher, teaching assistant, peer) and the target learner group (e.g., primary school students, university students)."},
        {"code": "BFA", "description": "Basic Factual Accuracy: This sub-metric examines the accuracy of objective information, including definitions, formulas, factual statements, code syntax, and terminology"},
        {"code": "RPR", "description": "Reasoning Process Rigor: This criterion focuses on the completeness and logical validity of the model's reasoning in tasks that require multi-step derivations, explanations, or justifications"},
        {"code": "EICP", "description": "Error Identification & Correction Precision: In contexts involving diagnostics or feedback, this sub-metric evaluates the model's ability to accurately detect, localize, and correct errors without introducing false positives or negatives"},
        {"code": "CSI", "description": "Clarity, Simplicity & Inspiration: This sub-metric assesses whether the explanation is articulated clearly and accessibly, using appropriate language to promote understanding and stimulate student interest or engagement"},
        {"code": "MGP", "description": "Motivation, Guidance & Positive Feedback: It evaluates the model's ability to encourage learners through constructive feedback and supportive guidance, promoting confidence and independent thinking rather than relying on direct answers alone"},
    ],

    "IP": [
        {"code": "IFTC", "description": "Instruction Following & Task Completion: This sub-metric measures the model's ability to accurately interpret and complete assigned tasks, such as solving problems, correcting errors, or generating questions, while adhering to the required output format and constraints."},
        {"code": "CRSC", "description": "Content Relevance & Scope Control: The response is assessed for its focus on the specified topic or knowledge area, as well as its ability to stay within the intended difficulty level, subject boundaries, and content scope"},
        {"code": "SEI", "description": "Scenario Element Integration: This sub-metric measures the degree to which the model effectively incorporates scenario-specific information, such as prior student responses, individual learning preferences, or stated pedagogical objectives. This is especially important in personalized learning and interactive tutoring contexts"},
        {"code": "BFA", "description": "Basic Factual Accuracy: This sub-metric examines the accuracy of objective information, including definitions, formulas, factual statements, code syntax, and terminology"},
        {"code": "DKA", "description": "Domain Knowledge Accuracy: It assesses the appropriateness and depth of subject-specific knowledge presented in the response, ensuring alignment with disciplinary standards across domains such as mathematics, law, and computer science"},
        {"code": "RPR", "description": "Reasoning Process Rigor: This criterion focuses on the completeness and logical validity of the model's reasoning in tasks that require multi-step derivations, explanations, or justifications"},
        {"code": "CSI", "description": "Clarity, Simplicity & Inspiration: This sub-metric assesses whether the explanation is articulated clearly and accessibly, using appropriate language to promote understanding and stimulate student interest or engagement"},
        {"code": "HOTS", "description": "Higher-Order Thinking & Skill Development: This sub-metric examines whether the response promotes advanced cognitive skills, such as critical thinking, problem-solving, creative reasoning, and the ability to transfer knowledge to new contexts"},
    ],

    "AG": [
        {"code": "IFTC", "description": "Instruction Following & Task Completion: This sub-metric measures the model's ability to accurately interpret and complete assigned tasks, such as solving problems, correcting errors, or generating questions, while adhering to the required output format and constraints."},
        {"code": "CRSC", "description": "Content Relevance & Scope Control: The response is assessed for its focus on the specified topic or knowledge area, as well as its ability to stay within the intended difficulty level, subject boundaries, and content scope"},
    ],

    "QA": [
        {"code": "IFTC", "description": "Instruction Following & Task Completion: This sub-metric measures the model's ability to accurately interpret and complete assigned tasks, such as solving problems, correcting errors, or generating questions, while adhering to the required output format and constraints."},
        {"code": "CRSC", "description": "Content Relevance & Scope Control: The response is assessed for its focus on the specified topic or knowledge area, as well as its ability to stay within the intended difficulty level, subject boundaries, and content scope"},
        {"code": "BFA", "description": "Basic Factual Accuracy: This sub-metric examines the accuracy of objective information, including definitions, formulas, factual statements, code syntax, and terminology"},
        {"code": "RPR", "description": "Reasoning Process Rigor: This criterion focuses on the completeness and logical validity of the model's reasoning in tasks that require multi-step derivations, explanations, or justifications"},
    ],

    "TMG": [
        {"code": "IFTC", "description": "Instruction Following & Task Completion: This sub-metric measures the model's ability to accurately interpret and complete assigned tasks, such as solving problems, correcting errors, or generating questions, while adhering to the required output format and constraints."},
        {"code": "RTC", "description": "Role & Tone Consistency: This dimension evaluates whether the language style, tone, and level of expertise in the response are appropriate for the designated role (e.g., teacher, teaching assistant, peer) and the target learner group (e.g., primary school students, university students)."},
        {"code": "CRSC", "description": "Content Relevance & Scope Control: The response is assessed for its focus on the specified topic or knowledge area, as well as its ability to stay within the intended difficulty level, subject boundaries, and content scope"},
        {"code": "BFA", "description": "Basic Factual Accuracy: This sub-metric examines the accuracy of objective information, including definitions, formulas, factual statements, code syntax, and terminology"},
        {"code": "DKA", "description": "Domain Knowledge Accuracy: It assesses the appropriateness and depth of subject-specific knowledge presented in the response, ensuring alignment with disciplinary standards across domains such as mathematics, law, and computer science"},
        {"code": "CSI", "description": "Clarity, Simplicity & Inspiration: This sub-metric assesses whether the explanation is articulated clearly and accessibly, using appropriate language to promote understanding and stimulate student interest or engagement"},
        {"code": "HOTS", "description": "Higher-Order Thinking & Skill Development: This sub-metric examines whether the response promotes advanced cognitive skills, such as critical thinking, problem-solving, creative reasoning, and the ability to transfer knowledge to new contexts"},
    ],

    "QG": [
        {"code": "IFTC", "description": "Instruction Following & Task Completion: This sub-metric measures the model's ability to accurately interpret and complete assigned tasks, such as solving problems, correcting errors, or generating questions, while adhering to the required output format and constraints."},
        {"code": "CRSC", "description": "Content Relevance & Scope Control: The response is assessed for its focus on the specified topic or knowledge area, as well as its ability to stay within the intended difficulty level, subject boundaries, and content scope"},
        {"code": "BFA", "description": "Basic Factual Accuracy: This sub-metric examines the accuracy of objective information, including definitions, formulas, factual statements, code syntax, and terminology"},
        {"code": "DKA", "description": "Domain Knowledge Accuracy: It assesses the appropriateness and depth of subject-specific knowledge presented in the response, ensuring alignment with disciplinary standards across domains such as mathematics, law, and computer science"},
        {"code": "CSI", "description": "Clarity, Simplicity & Inspiration: This sub-metric assesses whether the explanation is articulated clearly and accessibly, using appropriate language to promote understanding and stimulate student interest or engagement"},
        {"code": "HOTS", "description": "Higher-Order Thinking & Skill Development: This sub-metric examines whether the response promotes advanced cognitive skills, such as critical thinking, problem-solving, creative reasoning, and the ability to transfer knowledge to new contexts"},
    ],

    "ES": [
        {"code": "IFTC", "description": "Instruction Following & Task Completion: This sub-metric measures the model's ability to accurately interpret and complete assigned tasks, such as solving problems, correcting errors, or generating questions, while adhering to the required output format and constraints."},
        {"code": "RTC", "description": "Role & Tone Consistency: This dimension evaluates whether the language style, tone, and level of expertise in the response are appropriate for the designated role (e.g., teacher, teaching assistant, peer) and the target learner group (e.g., primary school students, university students)."},
        {"code": "SEI", "description": "Scenario Element Integration: This sub-metric measures the degree to which the model effectively incorporates scenario-specific information, such as prior student responses, individual learning preferences, or stated pedagogical objectives. This is especially important in personalized learning and interactive tutoring contexts"},
        {"code": "MGP", "description": "Motivation, Guidance & Positive Feedback: It evaluates the model's ability to encourage learners through constructive feedback and supportive guidance, promoting confidence and independent thinking rather than relying on direct answers alone"},
        {"code": "PAS", "description": "Personalization, Adaptation & Learning Support: This criterion measures the response's ability to adapt based on the learner's background, proficiency level, and individual needs, including tailored suggestions, scaffolded prompts, and relevant resource recommendations"},
    ],

    "PCC": [
        {"code": "IFTC", "description": "Instruction Following & Task Completion: This sub-metric measures the model's ability to accurately interpret and complete assigned tasks, such as solving problems, correcting errors, or generating questions, while adhering to the required output format and constraints."},
        {"code": "SEI", "description": "Scenario Element Integration: This sub-metric measures the degree to which the model effectively incorporates scenario-specific information, such as prior student responses, individual learning preferences, or stated pedagogical objectives. This is especially important in personalized learning and interactive tutoring contexts"},
        {"code": "PAS", "description": "Personalization, Adaptation & Learning Support: This criterion measures the response's ability to adapt based on the learner's background, proficiency level, and individual needs, including tailored suggestions, scaffolded prompts, and relevant resource recommendations"},
    ],

    "PLS": [
        {"code": "IFTC", "description": "Instruction Following & Task Completion: This sub-metric measures the model's ability to accurately interpret and complete assigned tasks, such as solving problems, correcting errors, or generating questions, while adhering to the required output format and constraints."},
        {"code": "CRSC", "description": "Content Relevance & Scope Control: The response is assessed for its focus on the specified topic or knowledge area, as well as its ability to stay within the intended difficulty level, subject boundaries, and content scope"},
        {"code": "SEI", "description": "Scenario Element Integration: This sub-metric measures the degree to which the model effectively incorporates scenario-specific information, such as prior student responses, individual learning preferences, or stated pedagogical objectives. This is especially important in personalized learning and interactive tutoring contexts"},
        {"code": "PAS", "description": "Personalization, Adaptation & Learning Support: This criterion measures the response's ability to adapt based on the learner's background, proficiency level, and individual needs, including tailored suggestions, scaffolded prompts, and relevant resource recommendations"},
        {"code": "HOTS", "description": "Higher-Order Thinking & Skill Development: This sub-metric examines whether the response promotes advanced cognitive skills, such as critical thinking, problem-solving, creative reasoning, and the ability to transfer knowledge to new contexts"},
    ],
}

# ==========================================================
# SYSTEM PROMPT
# ==========================================================
SYSTEM_PROMPT = """
You are an impartial evaluation engine.
You will evaluate answers produced by AI models.
Your task is to assign numeric scores only.
Rules:
- Evaluate using ONLY the provided evaluation matrices.
- Ignore model names and do not favor any model.
- Use a 0-10.00 scale for each evaluation matrix.
- Aggregate your judgment across ALL provided records.
- Output NUMBERS ONLY.
- Do NOT include explanations, text, comments, or formatting.
- Output must be valid JSON and strictly follow the requested schema.
""".strip()

# ==========================================================
# UTILITIES
# ==========================================================
def extract_code_from_path(path: str) -> str:
    code = os.path.basename(path).split(".")[0]
    if code not in METRICS_REGISTRY:
        raise ValueError(
            f"Unknown evaluation code '{code}'. "
            f"Available codes: {list(METRICS_REGISTRY.keys())}"
        )
    return code


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def filter_records_by_model(records, model_name: str):
    """
    Keep ONLY the answer for the requested model_name in each record.
    Throws error if any record doesn't contain that model.
    """
    filtered = []
    missing = 0

    for r in records:
        answers = r.get("model_answers", [])
        hit = next((a for a in answers if a.get("model") == model_name), None)
        if not hit:
            missing += 1
            continue

        filtered.append({
            "question_template": r.get("question_template", ""),
            "model_answers": [hit]
        })

    if missing > 0:
        raise ValueError(
            f"Model '{model_name}' not found in {missing} record(s). "
            "Make sure the model name matches exactly."
        )

    if not filtered:
        raise ValueError(f"No records found for model '{model_name}'")

    return filtered


def safe_path_name(name: str) -> str:
    """
    Make a string safe for filesystem paths.
    Replaces '/', spaces, etc. with '_'.
    """
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def anonymize_model_names_for_prompt(records):
    """
    Return a deep-copied version of records where model names are replaced
    with anonymized labels (Answer_1, Answer_2, ...), for prompt only.
    Keeps order stable within a record.
    """
    copied = json.loads(json.dumps(records))  # simple deep copy (JSON-safe)
    for r in copied:
        for i, ans in enumerate(r.get("model_answers", []), start=1):
            ans["model"] = f"Answer_{i}"
    return copied


# ==========================================================
# PROMPT BUILDER
# ==========================================================
def build_user_prompt(records, metrics):
    metrics_text = "\n".join(
        f"- {m['code']}: {m['description']}" for m in metrics
    )

    # anonymize model names ONLY inside the prompt
    prompt_records = anonymize_model_names_for_prompt(records)

    data = [
        {
            "question": r["question_template"],
            "model_answers": r["model_answers"]
        }
        for r in prompt_records
    ]

    metric_codes = [m["code"] for m in metrics]

    schema_hint = {
        "Answer_1": {code: "number" for code in metric_codes} | {"average": "number"}
    }

    return f"""
Evaluation matrices:
{metrics_text}

You are evaluating answers produced by ONE model.
The model is anonymized as "Answer_1" to avoid bias.

Evaluate the model across ALL questions.

Data:
{json.dumps(data, ensure_ascii=False, indent=2)}

Required JSON output format (numbers only):
{json.dumps(schema_hint, indent=2)}
""".strip()


# ==========================================================
# JSON SCHEMA
# ==========================================================
def build_schema(model_name, metric_codes):
    return {
        "type": "object",
        "properties": {
            model_name: {
                "type": "object",
                "properties": {
                    **{m: {"type": "number"} for m in metric_codes},
                    "average": {"type": "number"}
                },
                "required": metric_codes + ["average"],
                "additionalProperties": False
            }
        },
        "required": [model_name],
        "additionalProperties": False
    }

# ==========================================================
# OPENAI CALL
# ==========================================================
def _extract_first_json_object(text: str) -> str:
    if not text:
        raise ValueError("Empty response content from the model.")

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find a JSON object in response:\n{text[:500]}")

    return text[start:end + 1]


def call_judge(prompt: str):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        cleaned = _extract_first_json_object(content)
        return json.loads(cleaned)

# ==========================================================
# MAIN EVALUATION
# ==========================================================
def evaluate_file(jsonl_path: str, target_model: str):
    code = extract_code_from_path(jsonl_path)
    metrics = METRICS_REGISTRY[code]

    records = load_jsonl(jsonl_path)

    records = filter_records_by_model(records, target_model)

    metric_codes = [m["code"] for m in metrics]

    prompt = build_user_prompt(records, metrics)
    result = call_judge(prompt)

    schema = build_schema("Answer_1", metric_codes)


    try:
        validate(instance=result, schema=schema)
    except ValidationError as e:
        raise RuntimeError(f"Schema validation failed:\n{e}")


    result = {target_model: result["Answer_1"]}
    return code, result

# ==========================================================
# CSV EXPORT
# ==========================================================
def export_to_csv(code: str, results: dict):
    """
    Save results to:
    results/{evaluator}/{model}/{code}_{YYYYMMDD_HHMMSS}.csv
    """

    timestamp = current_timestamp()
    evaluator_safe = safe_path_name(EVALUATOR_NAME)

    for model, scores in results.items():
        model_safe = safe_path_name(model)

        # Build directory path
        base_dir = os.path.join(
            "results",
            evaluator_safe,
            model_safe
        )
        os.makedirs(base_dir, exist_ok=True)

        # Build file path
        path = os.path.join(
            base_dir,
            f"{code}_{timestamp}.csv"
        )

        # Write CSV
        row = {"model": model}
        row.update(scores)
        df = pd.DataFrame([row])
        df.to_csv(path, index=False)

        return path  # single-model mode

# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: python evaluation.py <path_to_jsonl> <target_model>")

    jsonl_path = sys.argv[1]
    target_model = sys.argv[2] if len(sys.argv) == 3 else None

    if not target_model:
        raise SystemExit("You must provide a target_model (exact name in model_answers).")

    code, results = evaluate_file(jsonl_path, target_model)

    print(f"\n▶ Evaluation code: {code}")
    print(json.dumps(results, indent=2))

    csv_path = export_to_csv(code, results)
    print(f"\n✅ CSV saved to: {csv_path}")
