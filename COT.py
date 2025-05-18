import os
import time
import pandas as pd
import google.generativeai as genai
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download("wordnet", quiet=True)
# 載入 .env 檔案內容（預設會從目前目錄讀取 .env）
load_dotenv()

# ========== I/O-bound: Gemini LLM 包裝器 ==========
class GeminiLLM:
    def __init__(self, api_key, model_name="gemini-1.5-flash-latest"):
        """
        初始化 Gemini LLM 介面。
        Args:
            api_key (str): 您的 Google Generative AI API 金鑰。
            model_name (str): 要使用的 Gemini 模型 (例如 "gemini-pro", "gemini-1.5-flash-latest")。
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

# ========== CPU-bound: CoT 推理邏輯 ==========
class ChainOfThoughtReasoner:
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def generate_with_cot(self, task, context=None):
        prompt = f"Task: {task}\n"
        if context:
            prompt += f"Context: {context}\n"
        # prompt += "Let's think step by step.\nAnswer format: Final Answer: <your answer>"
        prompt += "請一步一步進行思考。\nAnswer format: 最終答案： <your answer>"
        return self.llm.generate(prompt)

# ========== CPU-bound: 評估指標 ==========
class Evaluator:
    def compute_metrics(self, prediction, reference):
        ref = reference.strip()
        pred = prediction.strip()

        # bleu = sentence_bleu([ref.split()], pred.split())
        smoother = SmoothingFunction().method1
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoother)
        # meteor = single_meteor_score(ref, pred)
        meteor = single_meteor_score(word_tokenize(ref), word_tokenize(pred))
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge = scorer.score(ref, pred)

        P, R, F1 = bert_score([pred], [ref], lang="en", rescale_with_baseline=False)
        return {
            "nlp_bleu": round(bleu, 4),
            "nlp_meteor": round(meteor, 4),
            "nlp_rouge1": round(rouge["rouge1"].fmeasure, 4),
            "nlp_rouge2": round(rouge["rouge2"].fmeasure, 4),
            "nlp_rougeL": round(rouge["rougeL"].fmeasure, 4),
            "nlp_bert_precision": round(P[0].item(), 4),
            "nlp_bert_recall": round(R[0].item(), 4),
            "nlp_bert_f1": round(F1[0].item(), 4),
        }

# ========== 執行主流程 ==========
def run_cot_evaluation(input_csv, output_csv, api_key, num_samples=1):
    df = pd.read_csv(input_csv).head(num_samples)

    llm = GeminiLLM(api_key)
    reasoner = ChainOfThoughtReasoner(llm)
    evaluator = Evaluator()

    all_results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        task = row["instruction"]
        context = row["context"] if pd.notna(row["context"]) else None
        reference = row["response"]

        generated = reasoner.generate_with_cot(task, context)
        metrics = evaluator.compute_metrics(generated, reference)

        result = {
            "instruction": task,
            "context": context,
            "ground_truth": reference,
            "generated_answer": generated,
            **metrics
        }
        all_results.append(result)

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 評估完成！結果已儲存至：{output_csv}")

# ========== 執行區塊 ==========
if __name__ == "__main__":
    # os.environ["GEMINI_API_KEY"] = "your-api-key"  # ← 可改為讀取 .env 或手動貼上
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not set.")
        print("警告：未在環境變數中找到 GEMINI_API_KEY。")

    input_path = r"D:\data_science\final_project\MAS-PRM-main\dataset\dolly_gsm8k.csv"
    output_path = r"D:\data_science\final_project\MAS-PRM-main\evaluation\cot_eval_output.csv"

    run_cot_evaluation(
        input_csv=input_path,
        output_csv=output_path,
        api_key=os.environ["GEMINI_API_KEY"],
        num_samples=3
    )
