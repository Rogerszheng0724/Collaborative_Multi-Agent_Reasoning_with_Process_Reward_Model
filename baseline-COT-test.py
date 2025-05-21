import os
import time
import re
import random
import traceback
import logging
import pandas as pd
import urllib.error
import google.generativeai as genai
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# ===== Setup logging =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ===== Load environment and verify =====
load_dotenv()
cwd = os.getcwd()
api_key = os.getenv('GEMINI_API_KEY')
# logger.info(f"Working dir: {cwd}")
if not api_key:
    logger.error("GEMINI_API_KEY missing in environment.")
    raise SystemExit("Set GEMINI_API_KEY in .env or environment variables.")

# ===== Constants =====
SIMILARITY_THRESHOLD_FOR_LABEL_L = 0.8
beta_prm = 1.0
MAX_RETRIES = 5
INITIAL_DELAY = 30
MAX_DELAY = 300

# ===== Download NLTK resources =====

def download_nltk_resource(resource_name, download_name):
    try:
        nltk.data.find(resource_name)
        logger.info(f"NLTK resource '{resource_name}' already available.")
    except LookupError:
        logger.info(f"NLTK resource '{resource_name}' not found. Downloading '{download_name}'...")
        try:
            nltk.download(download_name, halt_on_error=False)
            nltk.data.find(resource_name)
            logger.info(f"NLTK resource '{resource_name}' verified after download.")
        except Exception as e:
            logger.error(f"Failed to download NLTK resource '{download_name}': {e}\n{traceback.format_exc()}")

# Ensure all needed data
for resource, name in [('corpora/wordnet', 'wordnet'), ('corpora/omw-1.4', 'omw-1.4')]:
    download_nltk_resource(resource, name)

# ===== LLM Call Retry Logic =====
def call_llm_with_retry(llm_generate_fn, prompt, llm_name="LLM",
                        max_retries=MAX_RETRIES, initial_delay=INITIAL_DELAY, max_delay=MAX_DELAY):
    retries = 0
    delay = initial_delay
    while True:
        try:
            return llm_generate_fn(prompt=prompt)
        except Exception as e:
            err = str(e).lower()
            if not any(x in err for x in ["429", "quota", "rate limit", "resourceexhausted"]):
                logger.error(f"{llm_name} error: {e}")
                raise
            retries += 1
            if retries > max_retries:
                logger.error(f"{llm_name} retries exceeded: {e}")
                raise
            wait = min(delay + random.uniform(0, delay * 0.2), max_delay)
            logger.warning(f"{llm_name} rate-limited, retry {retries}/{max_retries} after {wait:.1f}s...")
            time.sleep(wait)
            delay = min(delay * 2, max_delay)

# ===== I/O-bound: Gemini LLM Wrapper =====
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

# ===== CPU-bound: Chain-of-Thought Reasoner =====
class ChainOfThoughtReasoner:
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def generate_with_cot(self, task, context=None):
        parts = [f"Task: {task}"]
        if context:
            parts.append(f"Context: {context}")
        parts.append("Let's think step by step.\nAnswer format: Final Answer: <your answer>")
        prompt = "\n".join(parts)
        return self.llm.generate(prompt)

# ===== CPU-bound: Basic NLP Metrics =====
class Evaluator:
    def compute_metrics(self, prediction, reference):
        ref, pred = reference.strip(), prediction.strip()
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=SmoothingFunction().method1)
        meteor = single_meteor_score(word_tokenize(ref), word_tokenize(pred))
        rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True).score(ref, pred)
        P, R, F1 = bert_score([pred], [ref], lang='en', rescale_with_baseline=False)
        return {
            'nlp_bleu': round(bleu, 4),
            'nlp_meteor': round(meteor, 4),
            'nlp_rouge1': round(rouge['rouge1'].fmeasure, 4),
            'nlp_rouge2': round(rouge['rouge2'].fmeasure, 4),
            'nlp_rougeL': round(rouge['rougeL'].fmeasure, 4),
            'nlp_bert_precision': round(P[0].item(), 4),
            'nlp_bert_recall': round(R[0].item(), 4),
            'nlp_bert_f1': round(F1[0].item(), 4),
        }

# ===== LLM Evaluation Function (Integrated from provided detailed explanation) =====
def evaluate_with_llm(task_description, thoughtflow_summary, generated_answer, ground_truth_answer, llm_interface, logger, beta_prm):
    R_score_val = None
    receval_assessment_text = "RECEVAL assessment placeholder: LLM not called or error."
    label_l = 0
    similarity_score = 0.0

    if not llm_interface or not hasattr(llm_interface, 'generate'):
        logger.warning("evaluate_with_llm", "Evaluation LLM not effectively initialized. Returning placeholder values for R-score, RECEVAL, and similarity.")
        R_score_val = 0.1
        similarity_score = 0.5
        if similarity_score >= SIMILARITY_THRESHOLD_FOR_LABEL_L: label_l = 1
        return R_score_val, receval_assessment_text, label_l, similarity_score

    # Removed BaseDummyLLM check as per request

    str_task_description = str(task_description) if task_description is not None else ""
    str_generated_answer = str(generated_answer) if generated_answer is not None else ""
    str_ground_truth_answer = str(ground_truth_answer) if ground_truth_answer is not None else ""
    str_thoughtflow_summary = str(thoughtflow_summary) if thoughtflow_summary is not None else ""

    r_score_output_str = None
    try:
        logger.info("Requesting R-score from LLM...")
        r_score_prompt = f"""Task: Evaluate the quality of the generated answer below.
Original Question/Task: {str_task_description}

Generated Answer:
\"\"\"
{str_generated_answer}
\"\"\"

Compare this generated answer to an ideal or "gold standard" reference answer (which you should infer or imagine if not explicitly provided).
How good is the generated answer in terms of correctness, completeness, relevance, and clarity for the original question/task?
Provide a numerical R-score. Positive scores indicate better than average quality relative to a baseline, negative scores are poorer.
This score is conceptually aligned with an implicit Process Reward Model's evaluation, $R = \\beta \\log(\\frac{{\\pi_{{policy}}(answer)}}{{\\pi_{{reference}}(answer)}})$, where beta = {beta_prm}.
Focus on the quality of the final answer itself.

Output only the numerical R-score (e.g., 0.75, -0.2):"""
        r_score_output_str = call_llm_with_retry(
            llm_interface.generate,
            prompt=r_score_prompt,
            llm_name="R_Score_Eval_LLM"
        )
        score_match_r = re.search(r"([-+]?\d*\.\d+|\d+)", str(r_score_output_str))
        if score_match_r:
            R_score_val = float(score_match_r.group(1))
            logger.info(f"R-score from LLM: {R_score_val}")
        else:
            logger.warning("evaluate_with_llm_R_Score_Parsing", f"Could not parse R-score from LLM output: '{str(r_score_output_str)}'.Setting R-score to default 0.0.")
            R_score_val = 0.0
    except Exception as e:
        logger.error("evaluate_with_llm_R_Score_Error", f"Error parsing R-score from LLM (after retries): {e}. LLM Output: '{str(r_score_output_str) if r_score_output_str else 'N/A'}'")
        R_score_val = 0.0

    receval_assessment_text_output = None
    try:
        logger.info("Requesting RECEVAL assessment from LLM...")
        receval_prompt = f"""Task: Evaluate the following thoughtflow based on RECEVAL criteria.
Original Question/Task: {str_task_description}

Thoughtflow Summary (e.g., debate, PRM iterations, internal reasoning steps):
\"\"\"
{str_thoughtflow_summary}
\"\"\"

Final Answer generated from this thoughtflow:
\"\"\"
{str_generated_answer}
\"\"\"

RECEVAL Criteria to consider:
1.  **Clarity & Coherence**: Is the reasoning process easy to understand? Are the steps logically connected and well-explained?
2.  **Soundness & Validity**: Are the arguments made during the process sound? Are inferences valid and based on correct premises?
3.  **Sufficiency & Completeness**: Does the reasoning process cover all necessary aspects of the question/task? Are there any significant omissions or unaddressed constraints?
4.  **Relevance**: Are all parts of the reasoning process relevant to answering the question/task? Are there digressions or irrelevant information?
5.  **Efficiency**: Is the reasoning concise and to the point, or does it include unnecessary detours or overly complex steps for the given task?

Provide a qualitative assessment of the overall thoughtflow that led to the final answer. Focus on the quality of the process, not just the final answer.
Your assessment:"""
        receval_assessment_text_output = call_llm_with_retry(
            llm_interface.generate,
            prompt=receval_prompt,
            llm_name="RECEVAL_Eval_LLM"
        )
        if receval_assessment_text_output and not str(receval_assessment_text_output).lower().startswith("error:") and not str(receval_assessment_text_output).lower().startswith("llm dummy response"):
            receval_assessment_text = str(receval_assessment_text_output)
        else:
            logger.warning("evaluate_with_llm_RECEVAL_Output", f"RECEVAL LLM output was problematic. Output: {str(receval_assessment_text_output)[:100]}...")
        logger.info(f"RECEVAL assessment from LLM (length): {len(str(receval_assessment_text))}")
    except Exception as e:
        logger.error("evaluate_with_llm_RECEVAL_Error", f"Error getting RECEVAL assessment from LLM (after retries): {e}")
        receval_assessment_text = f"RECEVAL assessment error (after retries): {e}"

    similarity_output_str = None
    try:
        logger.info("Requesting semantic similarity score from LLM...")
        similarity_prompt = f"""Task: Compare the semantic similarity of the two answers below.
Answer 1 (Generated by the system):
\"\"\"
{str_generated_answer}
\"\"\"

Answer 2 (Ground Truth / Reference Answer):
\"\"\"
{str_ground_truth_answer}
\"\"\"

Provide a similarity score between 0.0 (completely different) and 1.0 (semantically identical).
Focus on whether they convey the same core meaning, information, and address the same key aspects of the original task.
A score of 1.0 means they are essentially saying the same thing, even if worded differently. A score of 0.0 means they are unrelated.

Output only the numerical similarity score (e.g., 0.85):"""
        similarity_output_str = call_llm_with_retry(
            llm_interface.generate,
            prompt=similarity_prompt,
            llm_name="Similarity_Eval_LLM"
        )
        sim_score_match = re.search(r"([0-9.]+)", str(similarity_output_str))
        if sim_score_match:
            similarity_score = float(sim_score_match.group(1))
            similarity_score = max(0.0, min(similarity_score, 1.0))
            logger.info(f"Semantic similarity score from LLM: {similarity_score}")
        else:
            logger.warning("evaluate_with_llm_Similarity_Parsing",f"Could not parse similarity score from LLM output: '{str(similarity_output_str)}'. Setting to default 0.0.")
            similarity_score = 0.0
        if similarity_score >= SIMILARITY_THRESHOLD_FOR_LABEL_L:
            label_l = 1
        else:
            label_l = 0
        logger.info(f"Label l set to {label_l} based on similarity {similarity_score} (threshold {SIMILARITY_THRESHOLD_FOR_LABEL_L})")
    except Exception as e:
        logger.error("evaluate_with_llm_Similarity_Error", f"Error parsing similarity score from LLM (after retries): {e}. LLM Output: '{str(similarity_output_str) if similarity_output_str else 'N/A'}'")
        similarity_score = 0.0
        label_l = 0
    return R_score_val, receval_assessment_text, label_l, similarity_score

# ===== Main Execution =====
def run_cot_evaluation(input_csv, output_csv, num_samples=None):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.read_csv(input_csv)
    if num_samples:
        df = df.head(num_samples)
    llm = GeminiLLM(api_key)
    reasoner = ChainOfThoughtReasoner(llm)
    evaluator = Evaluator()
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        task = row.get('instruction', '')
        ctx = row.get('context', '')
        ref = row.get('response', '')
        # 生成答案，其中可能包含思維流程 (CoT)
        gen = reasoner.generate_with_cot(task, ctx)

        # 從生成的答案中提取最終答案和思維流程。
        # 這裡假設最終答案在 "Final Answer: " 之後，且思維流程是整個生成內容。
        # 您可能需要根據實際的 LLM 輸出格式調整此處的解析邏輯。
        thoughtflow_summary = gen # 暫時將整個生成內容視為思維流程總結
        final_answer_match = re.search(r"Final Answer:\s*(.*)", gen, re.DOTALL)
        extracted_answer = final_answer_match.group(1).strip() if final_answer_match else gen.strip()

        # NLP 指標計算
        met = evaluator.compute_metrics(extracted_answer, ref) # 使用提取的最終答案

        # LLM 評估
        # 調整參數以符合 evaluate_with_llm 的簽名
        R, rec, lbl, sim = evaluate_with_llm(task, thoughtflow_summary, extracted_answer, ref, llm, logger, beta_prm)
        results.append({
            'input_instruction': task,
            'input_context': ctx,
            'ground_truth_answer': ref,
            'generated_answer_final': extracted_answer, # 儲存提取的最終答案
            'thoughtflow_summary': thoughtflow_summary, # 儲存完整的思維流程
            'R_score_final_answer': R,
            'label_l_final_answer': lbl,
            'llm_similarity_final_answer': sim,
            'receval_assessment_thoughtflow_incl_prm': rec,
            **met
        })
    pd.DataFrame(results).to_csv(output_csv, index=False, encoding='utf-8-sig')
    logger.info(f"Evaluation complete. Results saved to: {output_csv}")

if __name__ == "__main__":
    # Paths can be customized or passed as arguments
    input_path = r"D:\data_science\final_project\MAS-PRM\dataset\dolly_gsm8k_MMLU1.csv"
    output_path = r"D:\data_science\final_project\MAS-PRM\evaluation\baseline-Cot_eval_output.csv"
    run_cot_evaluation(input_path, output_path, num_samples=3)