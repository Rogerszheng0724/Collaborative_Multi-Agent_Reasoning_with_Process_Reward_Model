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
        if not api_key:
            raise ValueError("必須提供有效 Gemini API 金鑰。")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"GeminiLLM 已使用模型初始化：{model_name}")

    def generate(self, prompt, temperature=0.7, safety_settings=None):
        print(f"\n--- 正在發送提示到 Gemini ---\n{prompt[:500]}...\n--- Gemini 提示結束 ---")
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

            print(f"--- 收到 Gemini 回應 ---\n{llm_response_text[:500]}...\n--- Gemini 回應結束 ---")
            return llm_response_text if llm_response_text else "錯誤：未產生內容或提示有問題。"
        except Exception as e:
            print(f"Gemini API 調用期間發生錯誤：{e}")
            return f"錯誤：產生內容時發生錯誤：{str(e)}"

# ===== CPU-bound: Graph-of-Thought Reasoner (Replaces ChainOfThoughtReasoner) =====
class GraphOfThoughtReasoner:
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def generate_with_got(self, task, context=None):
        """
        Generates a response using a Graph-of-Thought (GoT) approach.
        This simplified implementation simulates GoT by:
        1. Decomposing the task into sub-questions/nodes.
        2. Generating answers for each node.
        3. Integrating these answers into a final coherent response.

        A true GoT would involve dynamic node creation, parallelism, and more complex
        dependency management. This provides a conceptual demonstration.
        """
        parts = []
        if context:
            parts.append(f"Context: {context}")
        parts.append(f"Task: {task}")
        parts.append("Let's break this problem down into sub-problems and connect them. First, identify the core questions. Then, answer each question. Finally, synthesize these answers into a comprehensive final response.")

        initial_prompt = "\n".join(parts) + "\n\nStep 1: Decompose the task into key sub-questions. List them clearly, one per line. If the task is simple, state 'No decomposition needed, directly answer.'"
        decomposition_response = call_llm_with_retry(self.llm.generate, prompt=initial_prompt, llm_name="GoT_Decomposition_LLM")
        logger.info(f"Decomposition Response: {decomposition_response}")

        sub_questions = [q.strip() for q in decomposition_response.split('\n') if q.strip() and not q.startswith('Step 1:')]
        if "No decomposition needed" in decomposition_response:
            sub_questions = [task] # Treat the original task as the single "sub-question"

        node_responses = {}
        for i, q in enumerate(sub_questions):
            node_prompt = f"""Based on the following overall task and context, answer this specific sub-question:
Overall Task: {task}
Context: {context if context else 'N/A'}
Sub-question {i+1}: {q}

Provide a concise answer to this sub-question.
Answer:"""
            node_response = call_llm_with_retry(self.llm.generate, prompt=node_prompt, llm_name=f"GoT_Node_{i+1}_LLM")
            node_responses[f"Sub-question {i+1}: {q}"] = node_response
            logger.info(f"Node {i+1} Response for '{q}': {node_response[:100]}...")

        synthesis_prompt_parts = [f"Overall Task: {task}"]
        if context:
            synthesis_prompt_parts.append(f"Context: {context}")
        synthesis_prompt_parts.append("\nHere are the sub-questions identified and their respective answers:")
        for q, ans in node_responses.items():
            synthesis_prompt_parts.append(f"- {q}\n  Answer: {ans}")
        synthesis_prompt_parts.append("\nStep 2: Based on the overall task, the context, and the answers to the sub-questions above, synthesize a comprehensive, coherent, and final answer. Ensure the final answer directly addresses the original task. Conclude with 'Final Answer: <your synthesized answer>'.")

        final_synthesis_response = call_llm_with_retry(self.llm.generate, prompt="\n".join(synthesis_prompt_parts), llm_name="GoT_Synthesis_LLM")
        logger.info(f"Final Synthesis Response: {final_synthesis_response}")

        # Combine thought process (decomposition, node responses, synthesis)
        full_thought_process = "Graph of Thought Process:\n"
        full_thought_process += f"1. Task Decomposition:\n{decomposition_response}\n\n"
        full_thought_process += "2. Sub-question Answers (Nodes):\n"
        for q, ans in node_responses.items():
            full_thought_process += f"   - {q}\n     Answer: {ans}\n"
        full_thought_process += f"\n3. Final Synthesis:\n{final_synthesis_response}"

        return full_thought_process # Return the entire GoT process for assessment

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
def run_got_evaluation(input_csv, output_csv, num_samples=None):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.read_csv(input_csv)
    if num_samples:
        df = df.head(num_samples)
    llm = GeminiLLM(api_key)
    # Using GraphOfThoughtReasoner instead of ChainOfThoughtReasoner
    reasoner = GraphOfThoughtReasoner(llm)
    evaluator = Evaluator()
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        task = row.get('instruction', '')
        ctx = row.get('context', '')
        ref = row.get('response', '')

        # Generate answer with Graph of Thought
        gen_full_got_process = reasoner.generate_with_got(task, ctx)

        # Extract the final answer and the thoughtflow summary from the GoT output
        # The final answer is expected to be after "Final Answer:"
        final_answer_match = re.search(r"Final Answer:\s*(.*)", gen_full_got_process, re.DOTALL)
        extracted_answer = final_answer_match.group(1).strip() if final_answer_match else "Could not extract final answer."
        thoughtflow_summary = gen_full_got_process # The entire GoT process serves as the thoughtflow summary

        # NLP Metrics Calculation
        met = evaluator.compute_metrics(extracted_answer, ref)

        # LLM Evaluation
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
    input_path = r"D:\data_science\final_project\MAS-PRM\dataset\dolly_gsm8k_MMLU1.csv"
    output_path = r"D:\data_science\final_project\MAS-PRM\evaluation\baseline-Got_eval_output.csv"
    run_got_evaluation(input_path, output_path, num_samples=None)