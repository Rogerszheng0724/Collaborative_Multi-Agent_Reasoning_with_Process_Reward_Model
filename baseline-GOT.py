# baseline_got_gemini_eval.py

import os
import time
import uuid
import json
import pandas as pd
from dotenv import load_dotenv
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

# NLP 評估指標
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calc

# -----------------------------
#  I/O-bound：Gemini LLM 客戶端
# -----------------------------
import google.generativeai as genai

# class GeminiLLM:
#     def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
#         if not api_key:
#             raise ValueError("必須提供 GEMINI_API_KEY。")
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel(model_name)

#     def generate(self, prompt: str, temperature: float = 0.7) -> str:
#         response = self.model.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(temperature=temperature)
#         )
#         # 聚合可能在 .parts 裡
#         if hasattr(response, "parts") and response.parts:
#             return "".join(part.text for part in response.parts if hasattr(part, "text"))
#         return getattr(response, "text", "")

# ========== I/O-bound: Gemini LLM 包裝器 ==========
class GeminiLLM:
    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        """
        初始化 Gemini LLM 介面。
        Args:
            api_key (str): 您的 Google Generative AI API 金鑰。
            model_name (str): 要使用的 Gemini 模型 (例如 "gemini-pro", "gemini-2.0-flash")。
        """
        if not api_key: # 檢查金鑰是否為空
            raise ValueError("必須提供有效 Gemini API 金鑰。")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"GeminiLLM 已使用模型初始化：{model_name}")

    def generate(self, prompt, temperature=0.7, safety_settings=None):
        # try:
        #     response = self.model.generate_content(
        #         prompt,
        #         generation_config=genai.types.GenerationConfig(temperature=temperature),
        #         safety_settings=[
        #             {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #             {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #             {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #             {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #         ],
        #     )
        #     return getattr(response, "text", None) or "".join([p.text for p in response.parts])
        # except Exception as e:
        #     return f"[ERROR] {e}"
        """
        使用 Gemini API 產生內容。
        Args:
            prompt (str): 要發送給 LLM 的提示。
            temperature (float): 控制隨機性，數值越低越確定。
            safety_settings (list of dict, optional): 自訂安全設定。
        Returns:
            str: LLM 的回應文字。
        """
        print(f"\n--- 正在發送提示到 Gemini ---\n{prompt[:500]}...\n--- Gemini 提示結束 ---") # 縮短打印的提示長度
        try:
            effective_safety_settings = safety_settings or [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                ),
                safety_settings=effective_safety_settings
            )

            llm_response_text = ""
            if hasattr(response, 'parts') and response.parts:
                 llm_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            elif hasattr(response, 'text'):
                 llm_response_text = response.text

            if not llm_response_text and hasattr(response, 'prompt_feedback') and \
               response.prompt_feedback.block_reason != genai.types.BlockReason.BLOCK_REASON_UNSPECIFIED:
                block_reason_str = genai.types.BlockReason(response.prompt_feedback.block_reason).name
                print(f"警告：提示因 {block_reason_str} 被封鎖。安全評級：{response.prompt_feedback.safety_ratings}")
                return f"錯誤：提示因 {block_reason_str} 被封鎖。"

            print(f"--- 收到 Gemini 回應 ---\n{llm_response_text[:500]}...\n--- Gemini 回應結束 ---") # 縮短打印的回應長度
            return llm_response_text if llm_response_text else "錯誤：未產生內容或提示有問題。"
        except Exception as e:
            print(f"Gemini API 調用期間發生錯誤：{e}")
            return f"錯誤：產生內容時發生錯誤：{str(e)}"

def send_prompt_to_llm(llm: GeminiLLM, prompt: str) -> str:
    """I/O-bound: 呼叫 Gemini API"""
    return llm.generate(prompt)

# -----------------------------
#  CPU-bound：Graph of Thoughts
# -----------------------------
class Thought:
    def __init__(self, content: str, parents: Optional[List['Thought']] = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.parents = parents or []
        self.children: List['Thought'] = []
        self.score: Optional[float] = None

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "parents": [p.id for p in self.parents],
            "children": [c.id for c in self.children],
            "score": self.score
        }

class GraphOfThoughts:
    def __init__(self, llm: GeminiLLM):
        self.llm = llm
        self.thoughts: Dict[str, Thought] = {}

    def add_thought(self, content: str, parent_ids: Optional[List[str]] = None) -> Thought:
        parents = [self.thoughts[pid] for pid in parent_ids] if parent_ids else []
        t = Thought(content, parents)
        for p in parents:
            p.children.append(t)
        self.thoughts[t.id] = t
        return t

    def _generate_prompt_for_new_thought(self,
                                        task_description: str,
                                        existing: Optional[List[str]] = None,
                                        num_new: int = 3) -> str:
        prompt = f"Main task: {task_description}\n"
        if existing:
            prompt += "Existing thoughts:\n" + "\n".join(f"- {c}" for c in existing) + "\n"
        prompt += f"Please generate {num_new} distinct ideas/steps to advance the task, each on its own line."
        return prompt
    def _generate_prompt_for_aggregation(self, thoughts: List[Thought]) -> str:
        prompt = "Please aggregate the following ideas into a coherent response:\n"
        for thought in thoughts:
            prompt += f"- {thought.content}\n"
        prompt += "Final answer:"
        return prompt

# CPU-bound：簡單長度打分
def score_thought_cpu(thought: Thought) -> float:
    length = len(thought.content)
    score = min(1.0, length / 200.0)
    thought.score = score
    return score

# CPU-bound：簡單聚合（最短、中間、最長三者串接）
def aggregate_thoughts_cpu(contents: List[str]) -> str:
    sorted_cs = sorted(contents, key=len)
    picks = [sorted_cs[0], sorted_cs[len(sorted_cs)//2], sorted_cs[-1]]
    return " || ".join(picks)

# -----------------------------
#  CPU-bound：Evaluator
# -----------------------------
class Evaluator:
    def __init__(self):
        nltk.download("wordnet", quiet=True)
        self.rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

    def compute(self, pred: str, ref: str) -> Dict[str, float]:
        p = pred.strip()
        r = ref.strip()
        smoother = SmoothingFunction().method1
        bleu = sentence_bleu([r.split()], p.split(), smoothing_function=smoother)
        meteor = single_meteor_score(r.split(), p.split())
        rouge_scores = self.rouge.score(r, p)
        P, R, F1 = bert_score_calc([p], [r], lang="en", rescale_with_baseline=False)
        return {
            "bleu": round(bleu,4),
            "meteor": round(meteor,4),
            "rouge1_f": round(rouge_scores["rouge1"].fmeasure,4),
            "rouge2_f": round(rouge_scores["rouge2"].fmeasure,4),
            "rougeL_f": round(rouge_scores["rougeL"].fmeasure,4),
            "bert_p": round(P[0].item(),4),
            "bert_r": round(R[0].item(),4),
            "bert_f1": round(F1[0].item(),4),
        }

# -----------------------------
#  主流程
# -----------------------------
def run_got_evaluation(input_csv: str,
                       output_csv: str,
                       api_key: str,
                       num_new: int = 3,
                       max_samples: Optional[int] = None):
    df = pd.read_csv(input_csv)
    if max_samples:
        df = df.head(max_samples)

    llm = GeminiLLM(api_key)
    evaluator = Evaluator()
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="GOT Eval"):
        task = row["instruction"]
        context = row.get("context", "")
        reference = row["response"]

        graph = GraphOfThoughts(llm)
        # 1. 建立根節點
        root = graph.add_thought(
            content=f"Task: {task}" + (f" Context: {context}" if pd.notna(context) else "")
        )

        # 2. 產生新思考（I/O-bound）
        prompt = graph._generate_prompt_for_new_thought(task, None, num_new)
        resp = send_prompt_to_llm(llm, prompt)

        # 3. 切分多條思考
        lines = [l.strip("0123456789. )") for l in resp.splitlines() if l.strip()]
        thoughts = []
        for line in lines:
            t = graph.add_thought(line, parent_ids=[root.id])
            thoughts.append(t)

        # 4. 評分（CPU-bound）
        with ProcessPoolExecutor() as cpu_pool:
            for fut in as_completed([cpu_pool.submit(score_thought_cpu, t) for t in thoughts]):
                _ = fut.result()

        # 5. 聚合 top3 並當作最終答案
        top3 = sorted(thoughts, key=lambda t: t.score or 0, reverse=True)[:3]

        # --- NEW: use LLM to aggregate into a single final answer ---
        agg_prompt = graph._generate_prompt_for_aggregation(top3)
        final_answer = send_prompt_to_llm(llm, agg_prompt).strip()
        final_thought = graph.add_thought(final_answer, parent_ids=[t.id for t in top3])

        # 6. 評估最終答案 vs. reference
        metrics = evaluator.compute(final_thought.content, reference)
        results.append({
            "instruction": task,
            "context": context,
            "reference": reference,
            "generated": final_thought.content,
            **metrics
        })

    pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 完成！結果已儲存至 {output_csv}")

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("請先在環境變數設定 GEMINI_API_KEY")

    input_path = r"D:\data_science\final_project\MAS-PRM\dataset\dolly_gsm8k.csv"
    output_path = r"D:\data_science\final_project\MAS-PRM\evaluation\baseline-GOT_eval_output.csv"

    run_got_evaluation(
        input_csv  = input_path,
        output_csv = output_path,
        api_key    = api_key,
        num_new    = 3,
        max_samples=3                 # 若要限制測試筆數，可設 None 跑全量
    )
