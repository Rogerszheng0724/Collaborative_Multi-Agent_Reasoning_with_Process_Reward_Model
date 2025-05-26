import os
import re
import json
import asyncio
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")
genai.configure(api_key=API_KEY)
print(API_KEY)

MODEL_NAME = "gemini-2.5-pro-preview-05-06"
INPUT_EXCEL_PATH = r"D:\data_science\final_project\MAS-PRM\evaluation_v3\CoT-LLM-judge\processed_Cot_eval_output_Part4.xlsx"
OUTPUT_EXCEL_PATH = r"D:\data_science\final_project\MAS-PRM\evaluation_v3\CoT-LLM-judge\LLM-judge-baseline-Cot_evaluation_results_Part4.xlsx"

# FIXED_CRITERIA = [
#     {
#         "name": "Clarity",
#         "description": (
#             "Is the answer written with precise and unambiguous language, "
#             "free from vagueness, redundancy, or logical inconsistency?"
#         )
#     },
#     {
#         "name": "Completeness",
#         "description": (
#             "Does the answer address every component of the question in detail, "
#             "demonstrating a comprehensive and in-depth understanding, without omitting any relevant aspect?"
#         )
#     },
#     {
#         "name": "Relevance",
#         "description": (
#             "Is the answer strictly focused on the core of the question, "
#             "avoiding any off-topic, generic, or filler content, and supported by appropriate reasoning or evidence?"
#         )
#     }
# ]

FIXED_CRITERIA = [
    {
        "name": "Clarity",
        "description": (
            "• Uses precise terminology and unambiguous phrasing.\n"
            "• Maintains a clear logical flow with no undefined pronouns or jargon.\n"
            "• Avoids redundancy, filler words, and contradictory assertions."
        )
    },
    {
        "name": "Completeness",
        "description": (
            "• Explicitly addresses every sub-question or requirement.\n"
            "• Provides necessary definitions, examples, or data to support each point.\n"
            "• Considers edge cases or counterarguments where relevant, without omitting any critical aspect."
        )
    },
    {
        "name": "Relevance",
        "description": (
            "• Stays focused on the core prompt with no off-topic digressions.\n"
            "• Ties every claim directly to the question and backs it with reasoning or citations.\n"
            "• Omits generic filler and unsupported opinions."
        )
    }
]

# --- Semaphores for concurrency control ---
api_semaphore = asyncio.Semaphore(1)   # Gemini API 最大併發請求數
row_semaphore = asyncio.Semaphore(1)  # 同時處理列的最大數量

# --- Shared model instance ---
model = genai.GenerativeModel(MODEL_NAME)

async def call_gemini_api(prompt: str) -> dict:
    """Call Gemini with api_semaphore to limit API concurrency."""
    async with api_semaphore:
        try:
            resp = await model.generate_content_async(
                contents=[prompt],
                generation_config=genai.types.GenerationConfig(temperature=0.7)
            )
            text = getattr(resp, "text", "").strip()
            m = re.search(r"(\{.*\})", text, re.S)
            if m:
                return json.loads(m.group(1))
        except Exception as e:
            print(f"[Warning] API call failed: {e}")
    return {}

def python_check_correctness(q: str, gt: str, gen: str) -> int:
    """Fallback correctness check."""
    if gt.isdigit():
        opts = dict(re.findall(r"(\d+)\.\s*(.+?)(?=(?:\d+\.)|$)", q, re.S))
        low = gen.lower()
        if gt in low or opts.get(gt, "").lower() in low:
            return 1
        return 0
    return 1 if gt.strip().lower() in gen.lower() else 0

async def evaluate_all(question: str, ground_truth: str, generated: str) -> dict:
    crit_text = "\n".join(
        f"{i+1}. {c['name']}: {c['description']}"
        for i, c in enumerate(FIXED_CRITERIA)
    )
    prompt = (
        "You are an expert judge. Perform the following two evaluations and output ONLY JSON:\n\n"
        "1) correctness_score: 0 or 1, indicating whether the Generated Answer contains the Reference Answer.\n"
        "2) evaluation_results: an array of scores 0–10 for each of these criteria.\n\n"
        f"Question:\n{question}\n\n"
        f"Reference Answer:\n{ground_truth}\n\n"
        f"Generated Answer:\n{generated}\n\n"
        "Criteria:\n" + crit_text + "\n\n"
        "Expected JSON format:\n"
        "{\n"
        '  "correctness_score": 0 or 1,\n'
        '  "evaluation_results": [\n'
        '    {"criterion_name":"Clarity",     "score": number},\n'
        '    {"criterion_name":"Completeness","score": number},\n'
        '    {"criterion_name":"Relevance",   "score": number}\n'
        "  ]\n"
        "}"
    )

    res = await call_gemini_api(prompt)
    corr = res.get("correctness_score")
    if corr not in (0, 1):
        corr = python_check_correctness(question, ground_truth, generated)

    evals = res.get("evaluation_results", [])
    out = {"correctness": corr}
    for e in evals:
        out[e["criterion_name"]] = e["score"]
    return out

async def process_row(idx: int, row: "Row", total: int) -> dict:
    """用 row_semaphore 限制最大同時處理列的數量。"""
    async with row_semaphore:
        q   = str(row.combined_input_question)
        gt  = str(row.ground_truth_answer)
        gen = str(row.overall_best_generated_answer_across_cycles)

        print(f"[{idx}/{total}] Start processing row {idx}")
        scores = await evaluate_all(q, gt, gen)
        print(
            f"[{idx}/{total}] Done: correctness={scores['correctness']}, "
            f"Clarity={scores.get('Clarity','-')}, "
            f"Completeness={scores.get('Completeness','-')}, "
            f"Relevance={scores.get('Relevance','-')}"
        )

        return {
            "combined_input_question": q,
            "ground_truth_answer":      gt,
            "overall_best_generated_answer_across_cycles": gen,
            **scores
        }

async def main():
    df = pd.read_excel(INPUT_EXCEL_PATH, engine="openpyxl")
    required = [
        "combined_input_question",
        "ground_truth_answer",
        "overall_best_generated_answer_across_cycles"
    ]
    if not all(col in df.columns for col in required):
        raise RuntimeError(f"Input must have columns {required}")

    total = len(df)
    # 產生所有處理任務，使用 itertuples 並以屬性存取欄位
    tasks = [
        process_row(i, row, total)
        for i, row in enumerate(df.itertuples(index=False, name="Row"), start=1)
    ]
    # 並行執行
    results = await asyncio.gather(*tasks)

    # 輸出 Excel
    cols = [
        "combined_input_question",
        "ground_truth_answer",
        "overall_best_generated_answer_across_cycles",
        "correctness",
        *[c["name"] for c in FIXED_CRITERIA]
    ]
    out_df = pd.DataFrame(results).reindex(columns=cols)
    out_df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print(f"Saved evaluation results to {OUTPUT_EXCEL_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
