import uvloop
import asyncio
import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm

# use uvloop for faster event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-pro-preview-05-06"
INPUT_EXCEL_PATH  = r"D:\data_science\final_project\MAS-PRM\evaluation_v3\processed_Got_eval_output.xlsx"
OUTPUT_EXCEL_PATH = r"D:\data_science\final_project\MAS-PRM\evaluation_v3\LLM-judge-Got_evaluation_results.xlsx"

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

# build and reuse prompt header
_CRIT_TEXT = "\n".join(f"{i+1}. {c['name']}: {c['description']}"
                       for i, c in enumerate(FIXED_CRITERIA))
_PROMPT_HEADER = (
    "You are an expert judge. Perform the following two evaluations and output ONLY JSON:\n\n"
    "1) correctness_score: 0 or 1, indicating whether the Generated Answer contains the Reference Answer.\n"
    "2) evaluation_results: an array of scores 0–10 for each of these criteria.\n\n"
    "Criteria:\n" + _CRIT_TEXT + "\n\n"
    "Expected JSON format:\n"
    "{\n"
    '  "correctness_score": 0 or 1,\n'
    '  "evaluation_results": [\n'
    '    {"criterion_name":"Clarity",     "score": number},\n'
    '    {"criterion_name":"Completeness","score": number},\n'
    '    {"criterion_name":"Relevance",   "score": number}\n'
    "  ]\n"
    "}\n\n"
)

# semaphores: increased concurrency
api_semaphore = asyncio.Semaphore(10)   # Gemini API 最大併發請求數
row_semaphore = asyncio.Semaphore(20)   # 同時處理的 row 數量

# shared model instance
model = genai.GenerativeModel(MODEL_NAME)

# precompile regex
_JSON_RE = re.compile(r"(\{.*\})", re.S)
_OPT_RE  = re.compile(r"(\d+)\.\s*(.+?)(?=(?:\d+\.)|$)", re.S)

async def call_gemini_api(prompt: str) -> dict:
    async with api_semaphore:
        try:
            resp = await model.generate_content_async(
                contents=[prompt],
                generation_config=genai.types.GenerationConfig(temperature=0.2)
            )
            text = getattr(resp, "text", "").strip()
            m = _JSON_RE.search(text)
            if m:
                return json.loads(m.group(1))
        except Exception as e:
            tqdm.write(f"[Warning] API call failed: {e}")
    return {}

def python_check_correctness(q: str, gt: str, gen: str) -> int:
    if gt.isdigit():
        opts = dict(_OPT_RE.findall(q))
        low = gen.lower()
        if gt in low or opts.get(gt, "").lower() in low:
            return 1
        return 0
    return 1 if gt.strip().lower() in gen.lower() else 0

async def evaluate_all(question: str, ground_truth: str, generated: str) -> dict:
    prompt = (
        _PROMPT_HEADER +
        f"Question:\n{question}\n\n"
        f"Reference Answer:\n{ground_truth}\n\n"
        f"Generated Answer:\n{generated}\n"
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
    async with row_semaphore:
        q   = str(row.combined_input_question)
        gt  = str(row.ground_truth_answer)
        gen = str(row.overall_best_generated_answer_across_cycles)
        scores = await evaluate_all(q, gt, gen)
        return {
            "combined_input_question": q,
            "ground_truth_answer":      gt,
            "overall_best_generated_answer_across_cycles": gen,
            **scores
        }

async def main():
    df = pd.read_excel(INPUT_EXCEL_PATH, engine="openpyxl")
    required = ["combined_input_question", "ground_truth_answer", "overall_best_generated_answer_across_cycles"]
    if not all(col in df.columns for col in required):
        raise RuntimeError(f"Input must have columns {required}")

    total = len(df)
    tasks = [
        process_row(i, row, total)
        for i, row in enumerate(df.itertuples(index=False, name="Row"), start=1)
    ]

    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=total, desc="Evaluating rows"):
        res = await coro
        results.append(res)

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
