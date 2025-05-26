import os
import re
import json
import asyncio
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# Configuration
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-pro-preview-05-06"
REQUEST_DELAY = 1.0  # seconds
INPUT_EXCEL_PATH = r"D:\data_science\final_project\MAS-PRM\evaluation_v3\processed_Rot_eval_output.xlsx"
OUTPUT_EXCEL_PATH = r"D:\data_science\final_project\MAS-PRM\evaluation_v3\LLM-judge-baseline-Rot_evaluation_results.xlsx"

# Fixed evaluation criteria
FIXED_CRITERIA = [
    {
        "name": "Clarity",
        "description": (
            "Is the answer written with precise and unambiguous language, "
            "free from vagueness, redundancy, or logical inconsistency?"
        )
    },
    {
        "name": "Completeness",
        "description": (
            "Does the answer address every component of the question in detail, "
            "demonstrating a comprehensive and in-depth understanding, without omitting any relevant aspect?"
        )
    },
    {
        "name": "Relevance",
        "description": (
            "Is the answer strictly focused on the core of the question, "
            "avoiding any off-topic, generic, or filler content, and supported by appropriate reasoning or evidence?"
        )
    }
]


async def call_gemini_api(prompt: str) -> dict:
    """Call Gemini and return parsed JSON dict, or empty dict on failure."""
    model = genai.GenerativeModel(MODEL_NAME)
    try:
        resp = await model.generate_content_async(
            contents=[prompt],
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        text = getattr(resp, "text", "").strip()
        match = re.search(r"(\{.*\})", text, re.S)
        if match:
            return json.loads(match.group(1))
    except Exception as e:
        print(f"[Warning] API call failed: {e}")
    return {}

def python_check_correctness(question: str, ground_truth: str, generated: str) -> int:
    """Fallback correctness check."""
    if ground_truth.isdigit():
        opts = dict(re.findall(r"(\d+)\.\s*(.+?)(?=(?:\d+\.)|$)", question, re.S))
        low = generated.lower()
        if ground_truth in low or opts.get(ground_truth, "").lower() in low:
            return 1
        return 0
    return 1 if ground_truth.strip().lower() in generated.lower() else 0

async def evaluate_correctness(question: str, ground_truth: str, generated: str) -> int:
    prompt = (
        "You are an expert judge. Determine if the generated answer correctly "
        "includes the reference answer. For multiple-choice questions (options 0–3), "
        "if the generated answer gives the option text without the number, still mark correct.\n\n"
        f"Question:\n{question}\n\n"
        f"Reference Answer:\n{ground_truth}\n\n"
        f"Generated Answer:\n{generated}\n\n"
        "Output only JSON:\n"
        "{\"correctness_score\": 0 or 1}"
    )
    res = await call_gemini_api(prompt)
    score = res.get("correctness_score")
    return score if score in (0, 1) else python_check_correctness(question, ground_truth, generated)

async def evaluate_with_criteria(question: str, generated: str) -> list[dict]:
    """Evaluate generated answer against the three fixed criteria (0–10)."""
    crit_text = "\n".join(
        f"{i+1}. {c['name']}: {c['description']}"
        for i, c in enumerate(FIXED_CRITERIA)
    )
    prompt = (
        "Evaluate the generated answer on each of the following criteria, giving a score 0–10. "
        "Output only JSON:\n"
        "{\n  \"evaluation_results\": [\n"
        "    {\"criterion_name\": \"…\", \"score\": number},\n"
        "    …\n  ]\n}\n\n"
        f"Question:\n{question}\n\n"
        f"Generated Answer:\n{generated}\n\n"
        "Criteria:\n" + crit_text
    )
    res = await call_gemini_api(prompt)
    print(res)
    return res.get("evaluation_results", [])

async def main():
    df = pd.read_excel(INPUT_EXCEL_PATH, engine="openpyxl")
    required = [
        "combined_input_question",
        "ground_truth_answer",
        "overall_best_generated_answer_across_cycles"
    ]
    if not all(col in df.columns for col in required):
        raise RuntimeError(f"Input must have columns {required}")
    print("讀取成功")
    records = []
    for _, row in df.iterrows():
        q   = str(row["combined_input_question"])
        gt  = str(row["ground_truth_answer"])
        gen = str(row["overall_best_generated_answer_across_cycles"])

        # 1. Correctness (0/1)
        corr = await evaluate_correctness(q, gt, gen)
        # await asyncio.sleep(REQUEST_DELAY)
        print("正確")
        # 2. Fixed-criteria evaluation
        evals = await evaluate_with_criteria(q, gen)
        # await asyncio.sleep(REQUEST_DELAY)
        print("標準")
        # accumulate results
        rec = {
            "combined_input_question": q,
            "ground_truth_answer":      gt,
            "overall_best_generated_answer_across_cycles": gen,
            "correctness": corr
        }
        for e in evals:
            rec[e["criterion_name"]] = e["score"]
        records.append(rec)

    # build output DataFrame
    cols = required + ["correctness"] + [c["name"] for c in FIXED_CRITERIA]
    out_df = pd.DataFrame(records).reindex(columns=cols)
    out_df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print(f"Saved evaluation to {OUTPUT_EXCEL_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
