import os
import pandas as pd
import google.generativeai as genai
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
import nltk

# 下載必要的 NLTK 資源
nltk.download("wordnet", quiet=True)

# -----------------------------
# I/O-bound: Gemini LLM 包裝器
# -----------------------------
class GeminiLLM:
    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        if not api_key:
            raise ValueError("必須提供有效的 Gemini API key")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"Initialized Gemini LLM: {model_name}")

    def generate(self, prompt, temperature=0.7, safety_settings=None):
        print(f"\n--- Prompt to Gemini ---\n{prompt[:300]}...\n")
        settings = safety_settings or [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
                safety_settings=settings
            )
            # 擷取回傳文字
            if hasattr(resp, "parts"):
                text = "".join([p.text for p in resp.parts if hasattr(p, "text")])
            else:
                text = getattr(resp, "text", "")
            return text or "[ERROR] No content"
        except Exception as e:
            return f"[ERROR] {e}"

# -----------------------------
# CPU-bound: Reversal-of-Thought 推理邏輯
# -----------------------------
class ReversalOfThoughtReasoner:
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def generate_with_rot(self, task, context=None):
        # Step 1: 正向推理 - 初步答案
        prompt_pos = f"Task: {task}\n"
        if context:
            prompt_pos += f"Context: {context}\n"
        prompt_pos += (
            "Generate an initial detailed solution for the task.\n"
            "Answer format: Initial Answer: <your answer>"
        )
        initial = self.llm.generate(prompt_pos)

        # Step 2: 反向推理 - 質疑初步答案，提出替代觀點
        prompt_rev = f"Task: {task}\n"
        if context:
            prompt_rev += f"Context: {context}\n"
        prompt_rev += (
            f"Initial Answer: {initial}\n"
            "Identify potential flaws or assumptions in the initial answer, then propose an alternative approach or correction.\n"
            "Answer format: Reversal Insight: <your insight>"
        )
        reversal = self.llm.generate(prompt_rev)

        # Step 3: 綜合 - 結合正向與反向，給出最終答案
        prompt_syn = f"Task: {task}\n"
        if context:
            prompt_syn += f"Context: {context}\n"
        prompt_syn += (
            f"Initial Answer: {initial}\nReversal Insight: {reversal}\n"
            "Based on the above, synthesize and provide the final, refined answer.\n"
            "Answer format: Final Answer: <your answer>"
        )
        final = self.llm.generate(prompt_syn)
        return final

# -----------------------------
# CPU-bound: 評估指標
# -----------------------------
class Evaluator:
    def compute_metrics(self, pred, ref):
        ref, pred = ref.strip(), pred.strip()
        smoother = SmoothingFunction().method1
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoother)
        meteor = single_meteor_score(word_tokenize(ref), word_tokenize(pred))
        rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True).score(ref, pred)
        P, R, F1 = bert_score([pred], [ref], lang="en", rescale_with_baseline=False)
        return {
            "bleu": round(bleu,4),
            "meteor": round(meteor,4),
            "rouge1": round(rouge["rouge1"].fmeasure,4),
            "rouge2": round(rouge["rouge2"].fmeasure,4),
            "rougeL": round(rouge["rougeL"].fmeasure,4),
            "bert_p": round(P[0].item(),4),
            "bert_r": round(R[0].item(),4),
            "bert_f1": round(F1[0].item(),4),
        }

# -----------------------------
# 主流程函式
# -----------------------------
def run_rot_evaluation(input_csv, output_csv, api_key, num_samples=None):
    df = pd.read_csv(input_csv)
    if num_samples:
        df = df.head(num_samples)

    llm = GeminiLLM(api_key)
    reasoner = ReversalOfThoughtReasoner(llm)
    evaluator = Evaluator()

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ROT Eval"):
        task = row.get("instruction")
        context = row.get("context") if pd.notna(row.get("context","")) else None
        ref = row.get("response")
        gen = reasoner.generate_with_rot(task, context)
        metrics = evaluator.compute_metrics(gen, ref)

        results.append({
            "instruction": task,
            "context": context,
            "ground_truth": ref,
            "generated": gen,
            **metrics
        })

    pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ ROT evaluation done. Results at: {output_csv}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("請先在 .env 設定 GEMINI_API_KEY")
    input_path = r"D:\data_science\final_project\MAS-PRM\dataset\dolly_gsm8k.csv"
    output_path = r"D:\data_science\final_project\MAS-PRM\evaluation\baseline-ROT_eval_output.csv"
    run_rot_evaluation(input_path, output_path, key, num_samples=5)
