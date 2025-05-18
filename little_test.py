# main_evaluator.py
import os
import traceback
import pandas as pd
import math
import numpy as np
import re # Ensure re is imported
from dotenv import load_dotenv
import time # Added: for retry delays and proactive sleeps
import random # Added: for retry jitter

# --- NLP Metrics Libraries ---
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calc
import urllib # For catching network errors during NLTK download
nltk.download("wordnet")
nltk.download("omw-1.4")

# --- Terminal Logger (Define early as it's used by dummies) ---
class TerminalLogger:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def _print(self, tag, stage, message, indent_level=0):
        if self.verbose:
            indent = "  " * indent_level
            print(f"[{tag}][{stage}]{indent} {message}")

    # Renamed for clarity for PRM specific logging
    def prm_iteration_thoughtflow(self, iteration, content, detail_level=0):
        if self.verbose:
            indent = "  " * detail_level
            print(f"\n--- [PRM ROUND {iteration} THOUGHTFLOW (Input to PRM)] ---{indent}\n{content}\n--------------------------------------------------")

    def prm_iteration_answer(self, iteration, score, justification, optimized_artifact, detail_level=0):
        if self.verbose:
            indent = "  " * detail_level
            print(f"\n--- [PRM ROUND {iteration} ANSWER (PRM Eval & Optimized Artifact)] ---")
            print(f"{indent}PRM Score: {score:.3f}")
            print(f"{indent}PRM Justification:\n{justification}")
            print(f"{indent}Optimized Artifact (Output of this round):\n{optimized_artifact}\n--------------------------------------------------")

    def thoughtflow(self, stage, message, detail_level=0): # General thoughtflow logging
        self._print("THOUGHTFLOW", stage, message, detail_level)

    def discussion(self, stage, agent_name, message, detail_level=0):
        indent = "  " * detail_level
        if self.verbose:
            content_str = str(message) if message is not None else "No content"
            print(f"[DISCUSSION][{stage}][{agent_name}]{indent} \n{content_str}")
            print(f"{'-'*70}")

    def answer(self, source_system, content, is_final=False, detail_level=0, iteration=None): # General answer logging
        tag_prefix = "FINAL " if is_final else "INTERMEDIATE "
        stage_info = f"[{source_system}]"
        if iteration is not None:
            stage_info += f"[Iter {iteration}]"

        indent = "  " * detail_level
        if self.verbose:
            content_str = str(content) if content is not None else "No content"
            print(f"[{tag_prefix}SYSTEM_OUTPUT]{stage_info}{indent} \n{content_str}")
            print(f"{'='*70}")

    def section_start(self, section_name):
        if self.verbose:
            print(f"\n{'#'*20} STARTING: {section_name.upper()} {'#'*20}")

    def section_end(self, section_name):
        if self.verbose:
            print(f"{'#'*20} FINISHED: {section_name.upper()} {'#'*20}\n")

    def error(self, stage, message):
        print(f"[ERROR][{stage}] {message}")

    def warning(self, stage, message):
        print(f"[WARNING][{stage}] {message}")

    def info(self, message):
         if self.verbose:
            print(f"[INFO] {message}")

# --- LLM Call Retry Logic ---
def call_llm_with_retry(llm_generate_function, prompt, logger, llm_name="LLM", max_retries=5, initial_delay=30, max_delay=300):
    retries = 0
    current_delay = initial_delay
    while retries < max_retries:
        try:
            if retries > 0 :
                proactive_sleep_duration = random.uniform(1, 5)
                logger.info(f"[{llm_name}_RETRY] Proactively sleeping for {proactive_sleep_duration:.2f}s before retry attempt {retries+1}.")
                time.sleep(proactive_sleep_duration)
            response = llm_generate_function(prompt=prompt)
            if isinstance(response, str) and ("error" in response.lower() or "sorry" in response.lower() and "cannot fulfill" in response.lower()):
                if "rate limit" in response.lower() or "quota" in response.lower() or "429" in response:
                    raise Exception(f"LLM indicated a retryable error in response: {response}")
            return response
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit_error = ("429" in error_str or
                                   "quota" in error_str or
                                   "rate limit" in error_str or
                                   "resourceexhausted" in error_str.replace(" ", "") or
                                   "user ratelimit" in error_str or
                                   "try again later" in error_str)
            if is_rate_limit_error:
                retries += 1
                if retries >= max_retries:
                    logger.error(f"{llm_name}_RETRY", f"Max retries ({max_retries}) reached. Aborting call. Last error: {e}")
                    raise
                suggested_delay_match = re.search(r"(?:retry|wait|delay).*?(\d+)\s*(?:seconds|s)", error_str, re.IGNORECASE)
                if suggested_delay_match:
                    wait_time = int(suggested_delay_match.group(1)) + random.uniform(1, 5)
                else:
                    wait_time = current_delay + random.uniform(0, current_delay * 0.2)
                wait_time = min(wait_time, max_delay)
                logger.warning(f"{llm_name}_RETRY", f"Rate limit or retryable error encountered. Retrying in {wait_time:.2f} seconds (Attempt {retries}/{max_retries})... Error: {e}")
                time.sleep(wait_time)
                current_delay = min(current_delay * 2, max_delay)
            else:
                logger.error(f"{llm_name}_RETRY", f"Encountered non-retryable LLM call error: {e}")
                raise
    logger.error(f"{llm_name}_RETRY", "Failed to get LLM response after retry loop (should not have been reached).")
    raise Exception("LLM call failed after multiple retries without returning or re-raising a specific error.")

# --- Download NLTK resources ---
def download_nltk_resource(resource_name, download_name, logger):
    try:
        nltk.data.find(resource_name)
        logger.info(f"NLTK resource '{resource_name}' found.")
    except LookupError:
        logger.info(f"NLTK resource '{resource_name}' not found. Attempting to download '{download_name}'...")
        try:
            nltk.download(download_name, quiet=True, halt_on_error=False)
            logger.info(f"NLTK download attempt for '{download_name}' completed.")
            nltk.data.find(resource_name)
            logger.info(f"NLTK resource '{resource_name}' successfully verified after download attempt.")
        except LookupError:
            logger.error("NLTK_DOWNLOAD", f"NLTK resource '{resource_name}' still not found after download attempt. Metrics depending on it (e.g., METEOR for wordnet) may be affected.")
            logger.error("NLTK_DOWNLOAD", f"Please try running 'import nltk; nltk.download(\"{download_name}\")' manually in a Python interpreter, and ensure NLTK_DATA environment variable is set if needed.")
        except urllib.error.URLError as e:
            logger.error("NLTK_DOWNLOAD", f"Failed to download '{download_name}' due to network issue (URLError): {e}. Check internet connection and NLTK server status.")
        except Exception as e:
            logger.error("NLTK_DOWNLOAD", f"Unexpected error during NLTK download or verification for '{download_name}': {e}\n{traceback.format_exc()}")
    except Exception as e:
        logger.error("NLTK_CHECK", f"Unexpected error while checking NLTK resource '{resource_name}': {e}")

temp_logger_for_nltk = TerminalLogger(verbose=True)
download_nltk_resource('tokenizers/punkt', 'punkt', temp_logger_for_nltk)
download_nltk_resource('corpora/wordnet', 'wordnet', temp_logger_for_nltk)
download_nltk_resource('corpora/omw-1.4', 'omw-1.4', temp_logger_for_nltk)
del temp_logger_for_nltk

# --- Load environment variables ---
load_dotenv()

# --- Define Base Dummy Classes Globally ---
class BaseDummyLLM:
    def __init__(self, api_key=None, model_name="dummy_model", logger=None):
        self.api_key = api_key
        self.model_name = model_name
        self.logger = logger if logger and hasattr(logger, 'info') else TerminalLogger(verbose=False)
        self.logger.info(f"BaseDummyLLM initialized, model: {self.model_name}")
    def generate(self, prompt, temperature=0.7):
        str_prompt = str(prompt) if prompt is not None else ""
        self.logger.info(f"BaseDummyLLM ({self.model_name}): Generating dummy response for prompt: {str_prompt[:50]}...")
        return f"LLM dummy response (BaseDummyLLM for {self.model_name}): {str_prompt[:50]}..."
    def generate_with_simulated_score(self, prompt, temperature=0.7):
        response = self.generate(prompt, temperature)
        return response, 0.1

class BaseDummyEmbedder:
    def __init__(self, api_key=None, model_name="dummy_embedding_model", logger=None):
        self.api_key = api_key
        self.model_name = model_name
        self.logger = logger if logger and hasattr(logger, 'info') else TerminalLogger(verbose=False)
        self.logger.info(f"BaseDummyEmbedder initialized, model: {self.model_name}")
    def calculate_similarity(self, text1, text2):
        self.logger.info(f"BaseDummyEmbedder: Calculating dummy similarity for texts: '{str(text1)[:20]}...' and '{str(text2)[:20]}...'")
        return 0.5

# --- Define Dummy System Classes Globally ---
class DummyGraphOfThoughts:
    def __init__(self, llm_interface, logger=None):
        self.llm = llm_interface
        self.logger = logger if logger and hasattr(logger, 'info') else TerminalLogger(verbose=False)
        self.logger.info("DummyGraphOfThoughts initialized.")
    def generate_and_evaluate_thoughts(self, task_description, num_thoughts):
        str_task_description = str(task_description) if task_description is not None else ""
        self.logger.info(f"Dummy GOT: Generating {num_thoughts} thoughts for '{str_task_description[:20]}...'")
        class DummyThought:
            def __init__(self, content, score=0.0, prm_justification="Dummy justification"):
                self.content = content
                self.score = score
                self.prm_justification = prm_justification
        return [DummyThought(f"GOT dummy thought for {str_task_description[:20]}", score=0.1)] * num_thoughts
    def print_graph(self): self.logger.info("Dummy GOT: Printing graph (empty).")
    def rank_thoughts(self): return []

class DummyLayerOfThoughts:
    def __init__(self, llm_interface, logger=None, prm_evaluator_llm=None):
        self.llm = llm_interface
        self.prm_evaluator_llm = prm_evaluator_llm or llm_interface
        self.logger = logger if logger and hasattr(logger, 'info') else TerminalLogger(verbose=False)
        self.logger.info("DummyLayerOfThoughts initialized.")
    def run_pipeline(self, conceptual_steps, main_task_description, initial_input=None, min_layer_prm_score_threshold=0.3):
        str_initial_input = str(initial_input) if initial_input is not None else ""
        str_main_task_description = str(main_task_description) if main_task_description is not None else ""
        self.logger.info(f"Dummy LOT: Running pipeline for '{str_initial_input[:20]}...' on task '{str_main_task_description[:20]}...'")
        return f"LOT dummy plan for {str_initial_input[:20]}"

class DummyReversalOfThought:
    def __init__(self, llm_interface, embedding_model_interface, similarity_threshold=0.7, logger=None):
        self.llm = llm_interface
        self.embedder = embedding_model_interface
        self.similarity_threshold = similarity_threshold
        self.logger = logger if logger and hasattr(logger, 'info') else TerminalLogger(verbose=False)
        self.logger.info("DummyReversalOfThought initialized.")
    def preference_guided_reverse_reasoning_warmup(self, demonstrations, main_task_description_for_prm, warm_iterations=1):
        str_main_task_description = str(main_task_description_for_prm) if main_task_description_for_prm is not None else ""
        self.logger.info(f"Dummy ROT: PGRR warmup for task '{str_main_task_description[:20]}...'")
        return "ROT dummy PGRR output"
    def cognitive_preference_manager(self, original_task_prompt_text, llm_taste_prompt_text, main_task_description_for_prm):
        str_main_task_description = str(main_task_description_for_prm) if main_task_description_for_prm is not None else ""
        self.logger.info(f"Dummy ROT: CPM for task '{str_main_task_description[:20]}...'")
        return "ROT dummy CPM output"
    def solve_task_with_final_prompt(self, prompt, instance):
        str_instance = str(instance) if instance is not None else ""
        str_prompt = str(prompt) if prompt is not None else ""
        self.logger.info(f"Dummy ROT: Solving '{str_instance[:20]}...' with prompt '{str_prompt[:20]}...'")
        return f"ROT dummy solution for {str_instance[:20]}"

# --- Initialize Pointers to Dummies, Attempt to Override with Real Imports ---
_RealGotGeminiLLM = None
# ... (rest of the import and dummy class setup remains the same) ...
_RealLotGeminiLLMInterface = None
_RealRotGeminiLLMInterface = None
_RealRotGeminiEmbeddingInterface = None
_RealGraphOfThoughts = None
_RealLayerOfThoughts = None
_RealReversalOfThought = None

GotGeminiLLM_cls = BaseDummyLLM
LotGeminiLLMInterface_cls = BaseDummyLLM
RotGeminiLLMInterface_cls = BaseDummyLLM
RotGeminiEmbeddingInterface_cls = BaseDummyEmbedder

GraphOfThoughts_cls = DummyGraphOfThoughts
LayerOfThoughts_cls = DummyLayerOfThoughts
ReversalOfThought_cls = DummyReversalOfThought

IMPORTS_SUCCESSFUL = False
try:
    from GOT import GraphOfThoughts as TempRealGraphOfThoughts, GeminiLLM as TempRealGotGeminiLLM
    from LOT import LayerOfThoughts as TempRealLayerOfThoughts, GeminiLLMInterface as TempRealLotGeminiLLMInterface
    from ROT import ReversalOfThought as TempRealReversalOfThought, GeminiLLMInterface as TempRealRotGeminiLLMInterface, GeminiEmbeddingInterface as TempRealRotGeminiEmbeddingInterface

    _RealGraphOfThoughts = TempRealGraphOfThoughts
    _RealGotGeminiLLM = TempRealGotGeminiLLM
    _RealLayerOfThoughts = TempRealLayerOfThoughts
    _RealLotGeminiLLMInterface = TempRealLotGeminiLLMInterface
    _RealReversalOfThought = TempRealReversalOfThought
    _RealRotGeminiLLMInterface = TempRealRotGeminiLLMInterface
    _RealRotGeminiEmbeddingInterface = TempRealRotGeminiEmbeddingInterface

    GraphOfThoughts_cls = _RealGraphOfThoughts
    GotGeminiLLM_cls = _RealGotGeminiLLM
    LayerOfThoughts_cls = _RealLayerOfThoughts
    LotGeminiLLMInterface_cls = _RealLotGeminiLLMInterface
    ReversalOfThought_cls = _RealReversalOfThought
    RotGeminiLLMInterface_cls = _RealRotGeminiLLMInterface
    RotGeminiEmbeddingInterface_cls = _RealRotGeminiEmbeddingInterface

    IMPORTS_SUCCESSFUL = True
    print("[INFO] Successfully imported and reassigned GOT, LOT, and ROT modules.")
except ImportError as e:
    print(f"[WARNING] Failed to import real GOT, LOT, or ROT modules: {e}")
    print("[INFO] Continuing with globally defined dummy classes. Full functionality will be limited.")
except Exception as e:
    print(f"[ERROR] Unexpected error during import of real modules: {e}\n{traceback.format_exc()}")
    print("[INFO] Continuing with globally defined dummy classes.")


# --- API Key Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("[WARNING] GEMINI_API_KEY not found in environment variables.")
    if IMPORTS_SUCCESSFUL:
        print("[WARNING] LLM calls may fail if API key for real modules is not set.")
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" and IMPORTS_SUCCESSFUL:
    print("[WARNING] Using placeholder API key. Real LLM calls are likely to fail. Please set a valid GEMINI_API_KEY.")

# --- Configuration ---
BETA_PRM = 0.05
SIMILARITY_THRESHOLD_FOR_LABEL_L = 0.7

# --- Debate Agent ---
class DebateAgent:
    def __init__(self, name, llm_interface, logger):
        self.name = name
        self.llm = llm_interface
        self.logger = logger

    def speak(self, prompt, context_summary=""):
        if not hasattr(self.llm, 'generate') or not callable(self.llm.generate):
            error_msg = f"LLM interface for agent {self.name} is incorrect or not initialized (no 'generate' method)."
            self.logger.error("MAS_DEBATE_AGENT", error_msg)
            return f"Error: {error_msg}"
        str_prompt = str(prompt) if prompt is not None else ""
        str_context_summary = str(context_summary) if context_summary is not None else ""
        full_prompt = f"You are {self.name}, an expert.\nCurrent discussion context: {str_context_summary}\nYour specific task: {str_prompt}\nPlease provide your response. Ensure it is clear, structured, and directly addresses the task."
        # Using general thoughtflow for agent thinking
        self.logger.thoughtflow("MAS_DEBATE_AGENT_THINKING", f"Agent {self.name} is formulating response to: {str_prompt[:100]}...")
        try:
            if isinstance(self.llm, BaseDummyLLM):
                response = self.llm.generate(prompt=full_prompt)
            else:
                 response = call_llm_with_retry(
                    self.llm.generate,
                    prompt=full_prompt,
                    logger=self.logger,
                    llm_name=f"DebateAgent_{self.name}"
                )
        except Exception as e:
            self.logger.error("MAS_DEBATE_AGENT_SPEAK_ERROR", f"Agent {self.name} encountered a critical error during LLM call (after retries): {e}")
            return f"Error: {self.name} had an LLM error during response generation (after retries)."
        response_str = str(response) if response is not None else f"{self.name} failed to generate a response."
        self.logger.discussion("MAS_DEBATE_TURN_OUTPUT", self.name, response_str)
        return response_str

# --- MAS Orchestrator ---
class MASOrchestrator:
    def __init__(self, api_key, logger):
        self.api_key = api_key
        self.logger = logger
        self.got_llm = None
        self.got_system = None
        self.lot_llm = None
        self.lot_system = None
        self.rot_llm = None
        self.rot_embedder = None
        self.rot_system = None
        self.debate_llm = None
        self.synthesis_llm = None
        self.iterative_optimizer_llm = None

        if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY_HERE":
            self.logger.warning("MASOrchestrator_Init", "API key is invalid or a placeholder. MAS may use dummy LLMs if real modules were imported, or fail if they expect a key.")
        try:
            if IMPORTS_SUCCESSFUL and GotGeminiLLM_cls == _RealGotGeminiLLM:
                self.logger.info("Instantiating real GOT.GeminiLLM (from GOT.py).")
                self.got_llm = GotGeminiLLM_cls(api_key=self.api_key)
            else:
                self.logger.info(f"Instantiating GotGeminiLLM_cls as {GotGeminiLLM_cls.__name__} (with logger).")
                self.got_llm = GotGeminiLLM_cls(api_key=self.api_key, logger=self.logger)
            self.got_system = GraphOfThoughts_cls(llm_interface=self.got_llm, logger=self.logger)
            self.logger.info(f"GraphOfThoughts (GOT) system initialized with LLM type: {type(self.got_llm).__name__}.")

            self.lot_llm = LotGeminiLLMInterface_cls(api_key=self.api_key, logger=self.logger)
            self.lot_system = LayerOfThoughts_cls(llm_interface=self.lot_llm, logger=self.logger, prm_evaluator_llm=self.lot_llm)
            self.logger.info(f"LayerOfThoughts (LOT) system initialized with LLM type: {type(self.lot_llm).__name__}.")

            self.rot_llm = RotGeminiLLMInterface_cls(api_key=self.api_key, logger=self.logger)
            self.rot_embedder = RotGeminiEmbeddingInterface_cls(api_key=self.api_key, logger=self.logger)
            self.rot_system = ReversalOfThought_cls(
                llm_interface=self.rot_llm,
                embedding_model_interface=self.rot_embedder,
                logger=self.logger
            )
            self.logger.info(f"ReversalOfThought (ROT) system initialized with LLM type: {type(self.rot_llm).__name__}.")

            if self.got_llm and hasattr(self.got_llm, 'generate') and not isinstance(self.got_llm, BaseDummyLLM):
                self.debate_llm = self.got_llm
                self.synthesis_llm = self.got_llm
                self.iterative_optimizer_llm = self.got_llm
                self.logger.info("Debate, Synthesis, Optimizer/PRM LLMs set to GOT LLM.")
            elif self.rot_llm and hasattr(self.rot_llm, 'generate') and not isinstance(self.rot_llm, BaseDummyLLM):
                self.debate_llm = self.rot_llm
                self.synthesis_llm = self.rot_llm
                self.iterative_optimizer_llm = self.rot_llm
                self.logger.info("Debate, Synthesis, Optimizer/PRM LLMs set to ROT LLM.")
            elif self.lot_llm and hasattr(self.lot_llm, 'generate') and not isinstance(self.lot_llm, BaseDummyLLM):
                self.debate_llm = self.lot_llm
                self.synthesis_llm = self.lot_llm
                self.iterative_optimizer_llm = self.lot_llm
                self.logger.info("Debate, Synthesis, Optimizer/PRM LLMs set to LOT LLM.")
            else:
                self.logger.warning("MASOrchestrator_Init", "No suitable real LLM found from initialized systems for Debate/Synthesis/Optimizer/PRM. Using BaseDummyLLM for these roles.")
                self.debate_llm = BaseDummyLLM(api_key=self.api_key, model_name="debate_dummy_fallback", logger=self.logger)
                self.synthesis_llm = BaseDummyLLM(api_key=self.api_key, model_name="synthesis_dummy_fallback", logger=self.logger)
                self.iterative_optimizer_llm = BaseDummyLLM(api_key=self.api_key, model_name="optimizer_prm_dummy_fallback", logger=self.logger)
        except Exception as e:
            self.logger.error("MASOrchestrator_Init", f"Critical error during subsystem initialization: {e}")
            self.logger.error("MASOrchestrator_Init_Traceback", "Detailed error traceback:")
            traceback.print_exc()
            self.logger.warning("MASOrchestrator_Init_Fallback", "All components will fall back to dummy implementations due to this initialization error.")
            self.got_llm = BaseDummyLLM(api_key=self.api_key, model_name="got_llm_exc_fallback", logger=self.logger)
            self.got_system = DummyGraphOfThoughts(self.got_llm, self.logger)
            self.lot_llm = BaseDummyLLM(api_key=self.api_key, model_name="lot_llm_exc_fallback", logger=self.logger)
            self.lot_system = DummyLayerOfThoughts(self.lot_llm, self.logger, self.lot_llm)
            self.rot_llm = BaseDummyLLM(api_key=self.api_key, model_name="rot_llm_exc_fallback", logger=self.logger)
            self.rot_embedder = BaseDummyEmbedder(api_key=self.api_key, model_name="rot_embed_exc_fallback", logger=self.logger)
            self.rot_system = DummyReversalOfThought(self.rot_llm, self.rot_embedder, logger=self.logger)
            self.debate_llm = BaseDummyLLM(api_key=self.api_key, model_name="debate_llm_exc_fallback", logger=self.logger)
            self.synthesis_llm = BaseDummyLLM(api_key=self.api_key, model_name="synthesis_llm_exc_fallback", logger=self.logger)
            self.iterative_optimizer_llm = BaseDummyLLM(api_key=self.api_key, model_name="optimizer_prm_llm_exc_fallback", logger=self.logger)

    def conduct_mas_debate(self, mission_context, got_idea, lot_plan, max_rounds=3):
        self.logger.section_start(f"MAS Style Debate (Targeting {max_rounds} Rounds)")
        proactive_delay_between_turns = int(os.getenv("MAS_DEBATE_DELAY_S", "5"))
        debate_transcript = []
        str_mission_context = str(mission_context) if mission_context is not None else "N/A"
        str_got_idea = str(got_idea) if got_idea is not None else "N/A"
        str_lot_plan = str(lot_plan) if lot_plan is not None else "N/A"
        discussion_context_summary = f"Mission Context:\n{str_mission_context}\n"
        discussion_context_summary += f"Initial Core Idea from GOT:\n{str_got_idea}\n"
        discussion_context_summary += f"Initial Detailed Plan from LOT:\n{str_lot_plan}\n"
        discussion_context_summary += "The debate will now commence.\n"

        if not self.debate_llm or isinstance(self.debate_llm, BaseDummyLLM) or not hasattr(self.debate_llm, 'generate'):
            self.logger.warning("MAS_DEBATE", "Debate LLM not effectively initialized or is a dummy. Using simulated debate statements.")
            debate_transcript.append({"speaker": "Moderator", "utterance": "Simulated debate starting due to LLM issue."})
            got_sim_statement = self.got_llm.generate("Provide a concise opening statement for GOT's idea.") if self.got_llm else "Simulated GOT statement (LLM unavailable)."
            lot_sim_statement = self.lot_llm.generate("Provide a concise opening statement for LOT's plan.") if self.lot_llm else "Simulated LOT statement (LLM unavailable)."
            temp_critic_llm = self.debate_llm if self.debate_llm and hasattr(self.debate_llm, 'generate') else BaseDummyLLM(logger=self.logger)
            critic_sim_statement = temp_critic_llm.generate("Provide a concise critical analysis.")
            debate_transcript.append({"speaker": "GOT_Representative", "utterance": got_sim_statement})
            debate_transcript.append({"speaker": "LOT_Representative", "utterance": lot_sim_statement})
            debate_transcript.append({"speaker": "Critical_Analyst", "utterance": critic_sim_statement})
            self.logger.section_end(f"MAS Style Debate (Simulated)")
            return debate_transcript

        got_agent = DebateAgent("GOT_Representative", self.debate_llm, self.logger)
        lot_agent = DebateAgent("LOT_Representative", self.debate_llm, self.logger)
        critic_agent = DebateAgent("Critical_Analyst", self.debate_llm, self.logger)
        opening_statement = f"Debate Topic: In-depth discussion based on the following mission context, GOT idea, and LOT plan.\n{discussion_context_summary}"
        debate_transcript.append({"speaker": "Moderator", "utterance": opening_statement})
        current_round = 0
        got_statement_response = str_got_idea
        lot_statement_response = str_lot_plan

        if current_round < max_rounds:
            current_round += 1
            self.logger.info(f"Debate Round {current_round}: GOT Representative")
            prompt_got = f"You are the GOT Representative. Your core idea is: '{got_statement_response[:150]}...'. Considering the overall mission context: '{str_mission_context[:100]}...', please elaborate on the strengths of your idea and how it fundamentally addresses the core problem presented in the mission. Be specific and persuasive."
            got_statement_response = got_agent.speak(prompt_got, discussion_context_summary)
            debate_transcript.append({"speaker": got_agent.name, "utterance": got_statement_response})
            discussion_context_summary += f"\nRound {current_round} - {got_agent.name}:\n{got_statement_response}\n"
            self.logger.info(f"Debate: Proactively sleeping for {proactive_delay_between_turns}s after GOT Rep.")
            time.sleep(proactive_delay_between_turns)
        if current_round < max_rounds:
            current_round += 1
            self.logger.info(f"Debate Round {current_round}: LOT Representative")
            prompt_lot = f"You are the LOT Representative. Your detailed plan is: '{lot_statement_response[:150]}...'. The GOT Representative has just stated: '{got_statement_response[:100]}...'. Explain how your plan effectively implements or expands upon GOT's idea, and how it specifically addresses potential challenges in achieving the mission: '{str_mission_context[:100]}...'. Provide concrete details."
            lot_statement_response = lot_agent.speak(prompt_lot, discussion_context_summary)
            debate_transcript.append({"speaker": lot_agent.name, "utterance": lot_statement_response})
            discussion_context_summary += f"\nRound {current_round} - {lot_agent.name}:\n{lot_statement_response}\n"
            self.logger.info(f"Debate: Proactively sleeping for {proactive_delay_between_turns}s after LOT Rep.")
            time.sleep(proactive_delay_between_turns)
        if current_round < max_rounds:
            current_round += 1
            self.logger.info(f"Debate Round {current_round}: Critical Analyst")
            prompt_critic = f"You are the Critical Analyst. Considering GOT's idea ('{got_statement_response[:100]}...') and LOT's plan ('{lot_statement_response[:100]}...'), critically evaluate their combined approach. Identify potential weaknesses, overlooked aspects, or inconsistencies concerning the overall mission: '{str_mission_context[:100]}...'. Offer specific, constructive suggestions for improvement or alternative considerations."
            critic_statement = critic_agent.speak(prompt_critic, discussion_context_summary)
            debate_transcript.append({"speaker": critic_agent.name, "utterance": critic_statement})
            discussion_context_summary += f"\nRound {current_round} - {critic_agent.name}:\n{critic_statement}\n"
        self.logger.section_end(f"MAS Style Debate (Completed {current_round} Rounds)")
        return debate_transcript

    def _get_prm_feedback_for_reasoning_process(self, task_description, current_reasoning_process, iteration_count):
        if not self.iterative_optimizer_llm or not hasattr(self.iterative_optimizer_llm, 'generate'):
            self.logger.warning("MASOrchestrator._get_prm_feedback", "Iterative optimizer LLM (PRM evaluator) not effectively initialized. Returning dummy feedback.")
            return 0.5, "Dummy PRM feedback: Evaluator LLM not available. Please check logical coherence and completeness.", "DummyPRM_Evaluator"
        if isinstance(self.iterative_optimizer_llm, BaseDummyLLM):
             self.logger.warning("MASOrchestrator._get_prm_feedback", "Iterative optimizer LLM (PRM evaluator) is BaseDummyLLM. Returning dummy feedback.")
             return 0.5, "Dummy PRM feedback (from BaseDummyLLM): Check logic. Ensure all steps are covered.", "BaseDummyPRM_Evaluator"
        str_task_description = str(task_description) if task_description is not None else ""
        str_current_reasoning_process = str(current_reasoning_process) if current_reasoning_process is not None else ""
        prm_feedback_prompt = f"""
        As an advanced Process Reward Model (PRM) evaluator, your task is to assess the following reasoning process/answer for the given task and provide actionable optimization feedback.
        Main Task:
        {str_task_description}

        Current Reasoning Process/Answer (Iteration {iteration_count}):
        \"\"\"
        {str_current_reasoning_process}
        \"\"\"

        Please evaluate this reasoning process/answer based on the following criteria:
        1.  **Correctness**: Is the conclusion accurate? Are there errors in intermediate steps or calculations (if any)?
        2.  **Completeness**: Does it comprehensively address all critical aspects and constraints of the task? Are there significant omissions?
        3.  **Logicality & Coherence**: Are the reasoning steps logical and well-supported? Is the overall presentation coherent and easy to follow?
        4.  **Conciseness & Efficiency**: Is there redundant information or unnecessary complexity? Could the reasoning be more direct?
        5.  **Actionability/Clarity**: If this is a plan, instruction, or explanation, is it clear, unambiguous, and easy to execute or understand?

        Provide an overall score (a float between 0.0 for very poor and 1.0 for excellent) and detailed justification.
        In your justification, clearly point out at least 1-2 specific, actionable points for improvement. For each point, briefly suggest how it could be improved.

        Output Format (Strictly Adhere):
        PRM Score: [A float between 0.0 and 1.0]
        PRM Justification: [Detailed assessment covering the aspects above and specific, actionable improvement suggestions]
        """
        self.logger.info(f"MASOrchestrator: Requesting PRM feedback for reasoning (Iteration {iteration_count})...")
        try:
            llm_response = call_llm_with_retry(
                self.iterative_optimizer_llm.generate,
                prompt=prm_feedback_prompt,
                logger=self.logger,
                llm_name="PRM_Feedback_LLM"
            )
        except Exception as e:
            self.logger.error("MASOrchestrator_Get_PRM_Feedback_Error", f"Critical error during LLM call for PRM feedback (after retries): {e}")
            return 0.1, f"Error obtaining PRM feedback due to LLM call failure (after retries): {e}", "PRM_Feedback_Error"
        str_llm_response = str(llm_response) if llm_response is not None else ""
        score_match = re.search(r"PRM Score:\s*([0-9.]+)", str_llm_response, re.IGNORECASE)
        justification_match = re.search(r"PRM Justification:\s*(.+)", str_llm_response, re.IGNORECASE | re.DOTALL)
        prm_score = float(score_match.group(1)) if score_match else 0.0
        prm_justification = justification_match.group(1).strip() if justification_match else "Could not parse PRM justification from LLM response."
        if not score_match:
            self.logger.warning("MASOrchestrator_Get_PRM_Feedback_Parsing", f"Could not parse PRM score from LLM response. Raw response: '{str_llm_response[:200]}...'")
            if "Could not parse PRM justification" in prm_justification and "PRM Score:" not in str_llm_response:
                 prm_justification = f"PRM Evaluator LLM raw output (format may be incorrect, score parsing failed): {str_llm_response}"
        self.logger.info(f"MASOrchestrator: PRM Evaluation (Iteration {iteration_count}) - Score: {prm_score:.2f}, Justification (start): {prm_justification[:150]}...")
        return prm_score, prm_justification, "ImplicitPRM_LLMEvaluator"

    def run_collaborative_task(self, initial_task_description, rot_demonstrations=None, problem_instance_for_rot_final_solve=None, num_debate_rounds=2, num_prm_iterations=3):
        proactive_delay_between_stages = int(os.getenv("MAS_STAGE_DELAY_S", "5"))
        self.logger.section_start(f"Collaborative Task (with {num_prm_iterations} PRM Iterations)")
        str_initial_task_description = str(initial_task_description) if initial_task_description is not None else "No task description provided."
        self.logger.info(f"Initial task description (raw): {str_initial_task_description[:100]}...")
        refined_task_prompt_for_core_logic = str_initial_task_description

        if rot_demonstrations and hasattr(self.rot_system, 'cognitive_preference_manager') and not isinstance(self.rot_system, DummyReversalOfThought):
            self.logger.info("--- ROT Phase ---")
            try:
                pgrr_output = self.rot_system.preference_guided_reverse_reasoning_warmup(
                    demonstrations=rot_demonstrations,
                    main_task_description_for_prm=str_initial_task_description,
                    warm_iterations=1
                )
                if pgrr_output and "dummy" not in str(pgrr_output).lower() and "error" not in str(pgrr_output).lower() and "failed" not in str(pgrr_output).lower():
                    cpm_output = self.rot_system.cognitive_preference_manager(
                        original_task_prompt_text=str_initial_task_description,
                        llm_taste_prompt_text=str(pgrr_output),
                        main_task_description_for_prm=str_initial_task_description
                    )
                    if cpm_output and "dummy" not in str(cpm_output).lower() and "error" not in str(cpm_output).lower() and "failed" not in str(cpm_output).lower():
                         refined_task_prompt_for_core_logic = str(cpm_output)
                    else:
                         self.logger.warning("MASOrchestrator_ROT_CPM", f"ROT CPM returned non-optimal output ('{str(cpm_output)[:50]}...'), using PGRR output if valid, else initial description.")
                         if pgrr_output and "dummy" not in str(pgrr_output).lower() and "error" not in str(pgrr_output).lower() and "failed" not in str(pgrr_output).lower():
                            refined_task_prompt_for_core_logic = str(pgrr_output)
                else:
                    self.logger.warning("MASOrchestrator_ROT_PGRR", f"ROT PGRR returned non-optimal output ('{str(pgrr_output)[:50]}...'), skipping CPM and using initial task description.")
                self.logger.info(f"ROT Phase output (refined task prompt, partial): {refined_task_prompt_for_core_logic[:100]}...")
                self.logger.info(f"Proactively sleeping for {proactive_delay_between_stages}s after ROT phase.")
                time.sleep(proactive_delay_between_stages)
            except Exception as e_rot:
                self.logger.error("MASOrchestrator_ROT_Phase_Error", f"ROT phase execution error: {e_rot}. Using original task description for core logic.")
        else:
            self.logger.info("Skipping ROT phase as no demonstrations were provided or ROT system is a dummy.")

        self.logger.info("--- GOT Phase ---")
        got_best_idea_content = f"Default GOT core idea for task: {refined_task_prompt_for_core_logic[:50]}..."
        try:
            initial_thoughts = self.got_system.generate_and_evaluate_thoughts(
                task_description=str(refined_task_prompt_for_core_logic),
                num_thoughts=1
            )
            if initial_thoughts and hasattr(initial_thoughts[0], 'content') and initial_thoughts[0].content:
                got_best_idea_content = str(initial_thoughts[0].content)
            else:
                self.logger.warning("MASOrchestrator_GOT_Phase", "GOT generate_and_evaluate_thoughts returned empty or invalid thoughts. Using default idea.")
        except Exception as e_got:
            self.logger.error("MASOrchestrator_GOT_Phase_Error", f"GOT phase execution error: {e_got}. Using default GOT idea.")
        self.logger.info(f"GOT best idea (partial): {got_best_idea_content[:100]}...")
        self.logger.info(f"Proactively sleeping for {proactive_delay_between_stages}s after GOT phase.")
        time.sleep(proactive_delay_between_stages)

        self.logger.info("--- LOT Phase ---")
        lot_detailed_plan_str = f"Default LOT detailed plan, based on GOT idea: {got_best_idea_content[:50]}..."
        try:
            plan_output = self.lot_system.run_pipeline(
                conceptual_steps=["Analyze GOT idea in context of the refined task", "Formulate detailed execution steps", "Finalize the comprehensive plan"],
                main_task_description=str(refined_task_prompt_for_core_logic),
                initial_input=str(got_best_idea_content)
            )
            if plan_output:
                lot_detailed_plan_str = str(plan_output)
            else:
                self.logger.warning("MASOrchestrator_LOT_Phase", "LOT run_pipeline returned no output. Using default plan.")
        except Exception as e_lot:
            self.logger.error("MASOrchestrator_LOT_Phase_Error", f"LOT phase execution error: {e_lot}. Using default LOT plan.")
        self.logger.info(f"LOT detailed plan (partial): {lot_detailed_plan_str[:100]}...")
        self.logger.info(f"Proactively sleeping for {proactive_delay_between_stages}s after LOT phase.")
        time.sleep(proactive_delay_between_stages)

        self.logger.info("--- MAS Debate Phase ---")
        mas_debate_transcript = self.conduct_mas_debate(
            mission_context=str(refined_task_prompt_for_core_logic),
            got_idea=str(got_best_idea_content),
            lot_plan=str(lot_detailed_plan_str),
            max_rounds=num_debate_rounds
        )
        debate_summary_for_synthesis = "Debate Record Summary:\n"
        for entry in mas_debate_transcript:
            speaker = entry.get('speaker', 'UnknownSpeaker')
            utterance = entry.get('utterance', 'No utterance')
            debate_summary_for_synthesis += f"{speaker}: {str(utterance)[:100]}...\n"
        self.logger.info(f"Debate summary captured. Proactively sleeping for {proactive_delay_between_stages}s after MAS Debate.")
        time.sleep(proactive_delay_between_stages)

        original_thoughtflow_summary_pre_prm = (
            f"Initial Task (Raw): {str_initial_task_description[:150]}...\n"
            f"Refined Task (Post-ROT, if applicable): {refined_task_prompt_for_core_logic[:150]}...\n"
            f"GOT Core Idea: {got_best_idea_content[:150]}...\n"
            f"LOT Detailed Plan: {lot_detailed_plan_str[:150]}...\n"
            f"Debate Summary:\n{debate_summary_for_synthesis}\n"
            "--- End of Pre-PRM Thoughtflow Components ---"
        )
        self.logger.info(f"Original thoughtflow (pre-PRM) captured. Length: {len(original_thoughtflow_summary_pre_prm)}")

        self.logger.info("--- Initial Synthesis Phase ---")
        current_reasoning_artifact = f"Default initial synthesis result, for task: {refined_task_prompt_for_core_logic[:50]}..."
        if not self.synthesis_llm or isinstance(self.synthesis_llm, BaseDummyLLM) or not hasattr(self.synthesis_llm, 'generate'):
            self.logger.warning("MASOrchestrator_Initial_Synthesis", "Synthesis LLM not effectively initialized or is a dummy. Using placeholder for initial synthesis.")
        else:
            synthesis_prompt = f"""
            Based on the following task context, outputs from different reasoning modules (GOT, LOT), and a debate summary, generate a comprehensive initial answer or reasoning process.
            Task Context (this is the core problem to solve, possibly refined by ROT):
            {refined_task_prompt_for_core_logic}

            GOT Core Idea (a high-level approach):
            {got_best_idea_content}

            LOT Detailed Plan (a structured plan based on GOT's idea):
            {lot_detailed_plan_str}

            Debate Summary (key points and critiques from discussion):
            {debate_summary_for_synthesis}

            Synthesize these elements into a coherent and comprehensive initial answer/reasoning process that directly addresses the Task Context:
            """
            try:
                synthesis_output = call_llm_with_retry(
                    self.synthesis_llm.generate,
                    prompt=synthesis_prompt,
                    logger=self.logger,
                    llm_name="InitialSynthesis_LLM"
                )
                if synthesis_output and not str(synthesis_output).lower().startswith("error:") and not str(synthesis_output).lower().startswith("llm dummy response"):
                    current_reasoning_artifact = str(synthesis_output)
                else:
                    self.logger.warning("MASOrchestrator_Initial_Synthesis_Output", f"Initial synthesis LLM did not produce valid output or produced error/dummy. Using placeholder. Output: {str(synthesis_output)[:100]}...")
            except Exception as e_synth:
                self.logger.error("MASOrchestrator_Initial_Synthesis_Error", f"Critical error during initial synthesis LLM call (after retries): {e_synth}. Using placeholder.")
        # Log the initial artifact that will go into the PRM loop
        self.logger.answer("INITIAL_REASONING_ARTIFACT_PRE_PRM", current_reasoning_artifact, is_final=False, detail_level=1)
        self.logger.info(f"Proactively sleeping for {proactive_delay_between_stages}s after Initial Synthesis.")
        time.sleep(proactive_delay_between_stages)

        self.logger.section_start(f"Implicit PRM Iterative Optimization ({num_prm_iterations} Iterations)")
        best_artifact_from_prm_iterations = current_reasoning_artifact
        best_prm_score_overall = -1.0
        prm_iteration_log_for_excel = []

        for i in range(num_prm_iterations):
            iteration_num = i + 1
            self.logger.info(f"--- Starting PRM Iteration {iteration_num}/{num_prm_iterations} ---")

            # Log the "thoughtflow" input to this PRM iteration using the new logger method
            self.logger.prm_iteration_thoughtflow(iteration_num, str(current_reasoning_artifact))

            prm_score, prm_justification, prm_evaluator_source = self._get_prm_feedback_for_reasoning_process(
                str(refined_task_prompt_for_core_logic),
                str(current_reasoning_artifact),
                iteration_num
            )

            # Prepare data for Excel log before optimization
            iteration_log_entry = {
                "prm_iteration": iteration_num,
                "thoughtflow_artifact_before_opt": str(current_reasoning_artifact),
                "prm_score": prm_score,
                "prm_justification_feedback": prm_justification,
                "optimized_artifact_after_feedback": "" # Placeholder
            }

            if prm_score > best_prm_score_overall:
                best_prm_score_overall = prm_score
                best_artifact_from_prm_iterations = str(current_reasoning_artifact)
                self.logger.info(f"PRM Iteration {iteration_num}: New best artifact (based on current eval) found with score {prm_score:.3f}")

            if prm_score >= 0.95 and i < num_prm_iterations - 1: # If score is high and not the last iteration
                self.logger.info(f"PRM Iteration {iteration_num}: Score {prm_score:.3f} reached early termination threshold. Stopping PRM iterations.")
                iteration_log_entry["optimized_artifact_after_feedback"] = "Optimization skipped due to early termination based on PRM score."
                prm_iteration_log_for_excel.append(iteration_log_entry) # Log before breaking
                # Log the "answer" for this round (evaluation part only as optimization is skipped) using the new logger method
                self.logger.prm_iteration_answer(iteration_num, prm_score, prm_justification, iteration_log_entry["optimized_artifact_after_feedback"])
                break

            # Proactive sleep before optimization call
            if i < num_prm_iterations - 1: # Only sleep if there's a next optimization step
                 self.logger.info(f"Proactively sleeping for {proactive_delay_between_stages}s before PRM optimization call (for Iteration {iteration_num} output).")
                 time.sleep(proactive_delay_between_stages)

            # Optimization step
            optimized_artifact_for_this_round = str(current_reasoning_artifact) # Default to current if optimizer fails or last iter
            if i < num_prm_iterations -1 : # Only optimize if it's not the last iteration's evaluation
                optimization_llm_prompt = f"""
                Original Task (The goal is to produce an excellent response to this):
                {refined_task_prompt_for_core_logic}

                Current Reasoning/Answer (Version {iteration_num}):
                \"\"\"
                {current_reasoning_artifact}
                \"\"\"

                Process Reward Model (PRM) Evaluation & Improvement Suggestions for Version {iteration_num}:
                PRM Score: {prm_score:.2f}
                PRM Improvement Suggestions: {prm_justification}

                Strictly following the PRM's improvement suggestions, revise and enhance the "Current Reasoning/Answer (Version {iteration_num})" to create an improved "Version {iteration_num + 1}".
                Ensure the new version directly addresses the issues and suggestions made by the PRM. The goal is to achieve a higher PRM score on the next evaluation.
                Focus on improvements in correctness, completeness, logicality, conciseness, and clarity as highlighted by the PRM.

                Improved Reasoning/Answer (Version {iteration_num + 1}):
                """
                self.logger.info(f"PRM Iteration {iteration_num}: Generating optimized version ({iteration_num + 1}) based on PRM feedback...")

                if not self.iterative_optimizer_llm or isinstance(self.iterative_optimizer_llm, BaseDummyLLM) or not hasattr(self.iterative_optimizer_llm, 'generate'):
                    self.logger.warning("MASOrchestrator_PRM_Optimize", f"Iterative optimizer LLM for PRM Version {iteration_num + 1} not effectively initialized or is dummy. Using placeholder.")
                    optimized_artifact_for_this_round = f"Placeholder optimized artifact (Version {iteration_num + 1}) (after PRM feedback for V{iteration_num})"
                else:
                    try:
                        optimized_output = call_llm_with_retry(
                            self.iterative_optimizer_llm.generate,
                            prompt=optimization_llm_prompt,
                            logger=self.logger,
                            llm_name=f"PRM_Optimization_LLM_Iter_{iteration_num + 1}"
                        )
                        if optimized_output and not str(optimized_output).lower().startswith("error:") and not str(optimized_output).lower().startswith("llm dummy response"):
                            optimized_artifact_for_this_round = str(optimized_output)
                        else:
                            self.logger.warning("MASOrchestrator_PRM_Optimize_Output", f"Optimizer LLM for Version {iteration_num + 1} did not produce valid output or produced error/dummy. Output: {str(optimized_output)[:100]}...")
                            # Retain previous artifact if optimization fails to produce valid output
                            optimized_artifact_for_this_round = str(current_reasoning_artifact)
                    except Exception as e_opt:
                        self.logger.error("MASOrchestrator_PRM_Optimization_Error", f"Critical error during optimization LLM call for Version {iteration_num + 1} (after retries): {e_opt}. Retaining previous version.")
                        optimized_artifact_for_this_round = str(current_reasoning_artifact) # Retain current on error

                current_reasoning_artifact = optimized_artifact_for_this_round # Update for next loop
                iteration_log_entry["optimized_artifact_after_feedback"] = str(current_reasoning_artifact)
                self.logger.info(f"PRM Iteration {iteration_num}: Optimized artifact for Version {iteration_num + 1} generated.")
                self.logger.info(f"Proactively sleeping for {proactive_delay_between_stages}s after PRM optimization call (Iter {iteration_num} -> V{iteration_num + 1}).")
                time.sleep(proactive_delay_between_stages)

            else: # This was the last iteration's evaluation
                self.logger.info(f"PRM Iteration {iteration_num}: This was the final PRM evaluation. The artifact used for this evaluation is considered for the best overall.")
                iteration_log_entry["optimized_artifact_after_feedback"] = "Final evaluation round; artifact is the input to this eval, no further optimization in this loop."
                # The `current_reasoning_artifact` at this point is what was evaluated.
                # `best_artifact_from_prm_iterations` will hold the one with highest score.

            # Log the complete "answer" for this round using the new logger method
            self.logger.prm_iteration_answer(iteration_num, prm_score, prm_justification, iteration_log_entry["optimized_artifact_after_feedback"])
            prm_iteration_log_for_excel.append(iteration_log_entry)
            self.logger.info(f"--- Finished PRM Iteration {iteration_num}/{num_prm_iterations} ---")

        self.logger.section_end(f"Implicit PRM Iterative Optimization")

        final_synthesized_plan = best_artifact_from_prm_iterations
        self.logger.info(f"MASOrchestrator: After {num_prm_iterations} PRM iterations, selected artifact (which had PRM score: {best_prm_score_overall:.3f}) is being used as final plan.")

        thoughtflow_summary_incl_prm = f"{original_thoughtflow_summary_pre_prm}\n--- PRM Iteration History (Summary) ---\n"
        for entry in prm_iteration_log_for_excel:
            thoughtflow_summary_incl_prm += (
                f"Iteration {entry['prm_iteration']}: Score={entry['prm_score']:.2f}, "
                f"Justification (start): {str(entry['prm_justification_feedback'])[:80]}..., "
                f"Artifact Before Opt (start): {str(entry['thoughtflow_artifact_before_opt'])[:50]}...\n"
                f"Optimized Artifact (start): {str(entry['optimized_artifact_after_feedback'])[:50]}...\n---\n"
            )
        thoughtflow_summary_incl_prm += "--- PRM Iteration History End ---"

        self.logger.answer("FINAL_SYNTHESIZED_PLAN (POST-PRM)", final_synthesized_plan, is_final=True)
        self.logger.section_end(f"Collaborative Task")

        return {
            "synthesized_final_plan": final_synthesized_plan,
            "original_thoughtflow_summary_pre_prm": original_thoughtflow_summary_pre_prm,
            "thoughtflow_summary_incl_prm_iterations_text": thoughtflow_summary_incl_prm,
            "prm_iteration_details_for_excel": prm_iteration_log_for_excel
        }


# --- Evaluation Functions ---
# ... (evaluate_with_llm, calculate_cross_entropy, calculate_nlp_metrics remain the same) ...
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

    if isinstance(llm_interface, BaseDummyLLM):
        logger.warning("evaluate_with_llm", "Evaluation LLM is BaseDummyLLM. Returning dummy placeholder values.")
        R_score_val = 0.1
        similarity_score = 0.5
        if similarity_score >= SIMILARITY_THRESHOLD_FOR_LABEL_L: label_l = 1
        return R_score_val, "RECEVAL assessment (BaseDummyLLM): Process seems okay.", label_l, similarity_score

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
            logger=logger,
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
            logger=logger,
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
            logger=logger,
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

def calculate_cross_entropy(R_score, label_l, logger):
    if R_score is None:
        logger.warning("calculate_cross_entropy", "R_score is None. Cannot calculate cross-entropy. Returning None.")
        return None
    try:
        R_score_float = float(R_score)
        if R_score_float > 30:
            log_sigma_R = 0.0
            log_one_minus_sigma_R = -R_score_float
        elif R_score_float < -30:
            log_sigma_R = R_score_float
            log_one_minus_sigma_R = 0.0
        else:
            log_sigma_R = -math.log(1 + math.exp(-R_score_float))
            log_one_minus_sigma_R = -math.log(1 + math.exp(R_score_float))
        label_l_int = int(label_l)
        loss = - (label_l_int * log_sigma_R + (1 - label_l_int) * log_one_minus_sigma_R)
        return loss
    except (ValueError, TypeError) as e:
        logger.error("calculate_cross_entropy_ConversionError", f"Error converting R_score or label_l to numeric: {e}. R_score: {R_score}, label_l: {label_l}")
        return None
    except Exception as e:
        logger.error("calculate_cross_entropy_MathError", f"Error calculating cross-entropy: {e}")
        return None

def calculate_nlp_metrics(generated_answer, ground_truth_answer, logger):
    metrics = {
        "bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "meteor": 0.0,
        "bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0
    }
    str_generated_answer = str(generated_answer) if generated_answer is not None else ""
    str_ground_truth_answer = str(ground_truth_answer) if ground_truth_answer is not None else ""
    if not str_generated_answer.strip() or not str_ground_truth_answer.strip():
        logger.warning("calculate_nlp_metrics", "Generated answer or ground truth is empty or whitespace. Most NLP metrics will be 0.")
        return metrics
    try:
        ref_tokens = [word_tokenize(str_ground_truth_answer.lower())]
        gen_tokens = word_tokenize(str_generated_answer.lower())
    except LookupError as le:
        logger.error("calculate_nlp_metrics_Tokenization", f"Tokenization failed due to NLTK resource 'punkt' not found: {le}. NLP metrics will be 0.")
        return metrics
    except Exception as e:
        logger.error("calculate_nlp_metrics_Tokenization_Error", f"Tokenization failed: {e}. NLP metrics will be 0.")
        return metrics
    if not gen_tokens and (not ref_tokens or not ref_tokens[0]):
        logger.info("calculate_nlp_metrics", "Both generated and reference answers are empty after tokenization. NLP metrics will be 0.")
        return metrics
    if not gen_tokens:
        logger.warning("calculate_nlp_metrics", "Generated answer is empty after tokenization. Most NLP metrics will be 0.")
        try:
            rouge_calc_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores_dict = rouge_calc_scorer.score(str_ground_truth_answer, "")
            metrics["rouge1"] = rouge_scores_dict['rouge1'].fmeasure
            metrics["rouge2"] = rouge_scores_dict['rouge2'].fmeasure
            metrics["rougeL"] = rouge_scores_dict['rougeL'].fmeasure
        except Exception as e_rouge_empty_gen:
            logger.error("calculate_nlp_metrics_Rouge_EmptyGen_Error", f"Error calculating ROUGE when generated answer is empty post-tokenization: {e_rouge_empty_gen}")
        return metrics
    try:
        chencherry = SmoothingFunction()
        metrics["bleu"] = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=chencherry.method1)
    except ZeroDivisionError:
        logger.warning("calculate_nlp_metrics_Bleu", "ZeroDivisionError during BLEU calculation. BLEU set to 0.0.")
        metrics["bleu"] = 0.0
    except Exception as e:
        logger.error("calculate_nlp_metrics_Bleu_Error", f"Error calculating BLEU: {e}")
        metrics["bleu"] = 0.0
    try:
        rouge_calc_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores_dict = rouge_calc_scorer.score(
            str_ground_truth_answer if str_ground_truth_answer.strip() else " ",
            str_generated_answer if str_generated_answer.strip() else " "
        )
        metrics["rouge1"] = rouge_scores_dict['rouge1'].fmeasure
        metrics["rouge2"] = rouge_scores_dict['rouge2'].fmeasure
        metrics["rougeL"] = rouge_scores_dict['rougeL'].fmeasure
    except Exception as e:
        logger.error("calculate_nlp_metrics_Rouge_Error", f"Error calculating ROUGE: {e}")
    try:
        metrics["meteor"] = meteor_score(ref_tokens, gen_tokens)
    except LookupError as le:
        logger.error("calculate_nlp_metrics_Meteor_LookupError", f"Error calculating METEOR due to NLTK resource (e.g., 'wordnet') not found: {le}. METEOR set to 0.0.")
        metrics["meteor"] = 0.0
    except Exception as e:
        logger.error("calculate_nlp_metrics_Meteor_Error", f"Error calculating METEOR: {e}")
        metrics["meteor"] = 0.0
    try:
        global torch
        if torch:
            P, R, F1 = bert_score_calc(
                [str_generated_answer if str_generated_answer.strip() else " "],
                [str_ground_truth_answer if str_ground_truth_answer.strip() else " "],
                lang="en", verbose=False, model_type='bert-base-uncased'
            )
            metrics["bert_precision"] = P.mean().item() if not torch.isnan(P.mean()).any() else 0.0
            metrics["bert_recall"] = R.mean().item() if not torch.isnan(R.mean()).any() else 0.0
            metrics["bert_f1"] = F1.mean().item() if not torch.isnan(F1.mean()).any() else 0.0
        else:
            logger.warning("calculate_nlp_metrics_BertScore", "PyTorch (torch) not available. Skipping BERTScore calculation.")
    except Exception as e:
        logger.error("calculate_nlp_metrics_BertScore_Error", f"Error calculating BERTScore: {e}")
    return metrics

# --- Main Processing Logic ---
def main():
    logger = TerminalLogger(verbose=True)
    logger.section_start("Main Evaluation Flow")
    global IMPORTS_SUCCESSFUL, GEMINI_API_KEY, torch
    try:
        import torch as pytorch_module
        torch = pytorch_module
        logger.info("PyTorch successfully imported. BERTScore should work if model files are accessible.")
    except ImportError:
        logger.warning("main_PyTorch_Import", "Failed to import PyTorch (torch). BERTScore calculation will be skipped or may fail. Please install PyTorch if BERTScore is desired.")
        torch = None
    if not IMPORTS_SUCCESSFUL:
        logger.warning("main_Module_Import", "One or more core modules (GOT, LOT, ROT) failed to import. Functionality will heavily rely on dummy classes.")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.warning("main_API_Key", f"GEMINI_API_KEY is invalid ('{str(GEMINI_API_KEY)[:10]}...'). LLM calls will use dummy classes or fail if real modules are loaded.")

    orchestrator = MASOrchestrator(api_key=GEMINI_API_KEY, logger=logger)
    evaluation_llm_interface = orchestrator.synthesis_llm
    if not evaluation_llm_interface or not hasattr(evaluation_llm_interface, 'generate'):
         logger.warning("main_Eval_LLM_Init", "Evaluation LLM (from orchestrator.synthesis_llm) not effectively initialized. Evaluation results using LLM will be placeholders.")
    elif isinstance(evaluation_llm_interface, BaseDummyLLM):
         logger.warning("main_Eval_LLM_Dummy", "Evaluation LLM (from orchestrator.synthesis_llm) is BaseDummyLLM. LLM-based evaluation results will be dummy values.")

    csv_file_path = "dolly_gsm8k.csv"
    logger.info(f"Attempting to load CSV data from: {csv_file_path}")
    if not os.path.exists(csv_file_path):
        logger.error("main_CSV_NotFound", f"CSV file not found: {csv_file_path}")
        try:
            logger.info(f"Attempting to create a sample '{csv_file_path}' file at: {os.path.abspath(csv_file_path)}")
            sample_df = pd.DataFrame(columns=['instruction', 'context', 'response', 'rot_demonstration_input', 'rot_demonstration_output'])
            sample_df.loc[0] = ["What is the capital of France?", "France is a country in Europe.", "The capital of France is Paris.", "France capital?", "Paris"]
            sample_df.loc[1] = ["Explain the theory of relativity.", "", "The theory of relativity, developed by Albert Einstein...", "Relativity meaning", "Space-time curvature"]
            sample_df.to_csv(csv_file_path, index=False, encoding='utf-8')
            logger.info(f"Sample '{csv_file_path}' created with UTF-8 encoding. Please populate it and rerun.")
        except Exception as e_create:
            logger.error("main_CSV_Create_Error", f"Failed to create sample CSV file '{csv_file_path}': {e_create}")
        logger.section_end("Main Evaluation Flow")
        return
    try:
        try:
            dataset_df = pd.read_csv(csv_file_path, encoding='utf-8')
            logger.info(f"Successfully loaded {len(dataset_df)} records from '{csv_file_path}' using UTF-8 encoding.")
        except UnicodeDecodeError:
            logger.warning("main_CSV_Encoding", f"Failed to load '{csv_file_path}' with UTF-8 encoding. Attempting with 'latin1' encoding...")
            dataset_df = pd.read_csv(csv_file_path, encoding='latin1')
            logger.info(f"Successfully loaded {len(dataset_df)} records from '{csv_file_path}' using latin1 encoding.")
        required_columns = ['instruction', 'response']
        missing_cols = [col for col in required_columns if col not in dataset_df.columns]
        if missing_cols:
            logger.error("main_CSV_Columns_Missing", f"CSV file '{csv_file_path}' must contain: {', '.join(required_columns)}. Missing: {', '.join(missing_cols)}")
            logger.section_end("Main Evaluation Flow")
            return
        if "context" not in dataset_df.columns:
            logger.info("CSV file does not contain 'context' column. Context will be empty for all rows.")
            dataset_df["context"] = ""
        if "rot_demonstration_input" not in dataset_df.columns or "rot_demonstration_output" not in dataset_df.columns:
            logger.info("CSV file does not contain 'rot_demonstration_input' or 'rot_demonstration_output'. ROT demos use defaults if missing.")
            if "rot_demonstration_input" not in dataset_df.columns: dataset_df["rot_demonstration_input"] = pd.NA
            if "rot_demonstration_output" not in dataset_df.columns: dataset_df["rot_demonstration_output"] = pd.NA
    except Exception as e:
        logger.error("main_CSV_Load_Error", f"Error loading or processing CSV file '{csv_file_path}': {e}")
        logger.section_end("Main Evaluation Flow")
        return

    all_results_main = []
    all_prm_iteration_data = []
    num_processed = 0
    default_max_items = len(dataset_df)
    max_items_to_process_str = input(f"Dataset has {default_max_items} items. How many to process? (Enter number, or press Enter for all [{default_max_items}]): ")
    try:
        max_items_to_process = int(max_items_to_process_str) if max_items_to_process_str.strip() else default_max_items
        if not (0 < max_items_to_process <= default_max_items):
            logger.warning("main_Process_Items_Invalid", f"Invalid number of items ({max_items_to_process}). Processing all {default_max_items} items.")
            max_items_to_process = default_max_items
    except ValueError:
        logger.warning("main_Process_Items_ValueError", f"Invalid input for number of items. Processing all {default_max_items} items.")
        max_items_to_process = default_max_items
    logger.info(f"Will process a maximum of {max_items_to_process} items.")

    for index, row in dataset_df.iterrows():
        if num_processed >= max_items_to_process:
            logger.info(f"Reached processing limit of {max_items_to_process} items.")
            break
        num_processed += 1
        logger.section_start(f"Processing Item {num_processed}/{max_items_to_process} (CSV Index {index})")
        instruction = str(row["instruction"]) if pd.notna(row["instruction"]) else ""
        context = str(row["context"]) if "context" in row and pd.notna(row["context"]) else ""
        ground_truth_answer = str(row["response"]) if pd.notna(row["response"]) else ""
        if context and context.strip():
            question = f"Instruction: {instruction}\n\nContext: {context}"
        else:
            question = f"Instruction: {instruction}"
        logger.info(f"Input (Instruction & Context combined): {question[:250]}...")
        logger.info(f"Ground Truth Answer (Response from CSV): {ground_truth_answer[:150]}...")
        rot_demonstrations_for_this_item = None
        if ("rot_demonstration_input" in dataset_df.columns and \
            "rot_demonstration_output" in dataset_df.columns and \
            pd.notna(row.get("rot_demonstration_input")) and \
            pd.notna(row.get("rot_demonstration_output"))):
            rot_demo_input = str(row["rot_demonstration_input"])
            rot_demo_output = str(row["rot_demonstration_output"])
            if rot_demo_input.strip() and rot_demo_output.strip():
                rot_demonstrations_for_this_item = [(rot_demo_input, rot_demo_output)]
                logger.info(f"Using ROT demonstration from CSV: Input='{rot_demo_input[:50]}...', Output='{rot_demo_output[:50]}...'")
        if not rot_demonstrations_for_this_item:
             logger.info("No valid ROT demonstration found in CSV for this item, or columns missing.")
        orchestrator_outputs = orchestrator.run_collaborative_task(
            initial_task_description=question,
            rot_demonstrations=rot_demonstrations_for_this_item,
            num_debate_rounds=2,
            num_prm_iterations=3
        )
        generated_answer = orchestrator_outputs.get("synthesized_final_plan", "Failed to generate answer.")
        original_thoughtflow_summary = orchestrator_outputs.get("original_thoughtflow_summary_pre_prm", "Failed to generate original thoughtflow summary.")
        thoughtflow_summary_incl_prm_text = orchestrator_outputs.get("thoughtflow_summary_incl_prm_iterations_text", "Failed to generate thoughtflow summary with PRM.")
        prm_iteration_details_list = orchestrator_outputs.get("prm_iteration_details_for_excel", [])
        logger.info(f"Generated Answer (post-PRM iteration, partial): {str(generated_answer)[:150]}...")
        logger.info(f"Original Thoughtflow Summary (pre-PRM, partial): {str(original_thoughtflow_summary)[:250]}...")
        logger.info(f"Thoughtflow Summary (incl. PRM iterations text, partial): {str(thoughtflow_summary_incl_prm_text)[:250]}...")
        for prm_detail in prm_iteration_details_list:
            prm_detail['csv_index_fk'] = index
            all_prm_iteration_data.append(prm_detail)
        R_score, receval_assessment, label_l, llm_similarity_score = evaluate_with_llm(
            task_description=question,
            thoughtflow_summary=thoughtflow_summary_incl_prm_text,
            generated_answer=generated_answer,
            ground_truth_answer=ground_truth_answer,
            llm_interface=evaluation_llm_interface,
            logger=logger,
            beta_prm=BETA_PRM
        )
        logger.info(f"LLM Evaluation - R-score: {R_score}, Label l: {label_l}, Similarity: {llm_similarity_score}, RECEVAL (length): {len(str(receval_assessment))}")
        ce_loss = calculate_cross_entropy(R_score, label_l, logger)
        logger.info(f"Cross-Entropy Loss: {ce_loss}")
        nlp_metrics = calculate_nlp_metrics(generated_answer, ground_truth_answer, logger)
        logger.info(f"NLP Metrics: {nlp_metrics}")
        current_main_result = {
            "csv_index": index,
            "input_instruction": instruction,
            "input_context": context,
            "combined_input_question": question,
            "ground_truth_answer": ground_truth_answer,
            "generated_answer_final": generated_answer,
            "R_score_final_answer": R_score,
            "label_l_final_answer": label_l,
            "llm_similarity_final_answer": llm_similarity_score,
            "cross_entropy_loss_final_answer": ce_loss,
            "original_thoughtflow_summary_pre_prm": original_thoughtflow_summary,
            "thoughtflow_summary_incl_prm_text": thoughtflow_summary_incl_prm_text,
            "receval_assessment_thoughtflow_incl_prm": receval_assessment,
            **{f"nlp_{k}": v for k, v in nlp_metrics.items()}
        }
        all_results_main.append(current_main_result)
        logger.section_end(f"Finished Item {num_processed}/{max_items_to_process}")
        time.sleep(int(os.getenv("MAS_ITEM_DELAY_S", "2")))

    logger.section_start("Overall Results Summary and Export")
    if not all_results_main:
        logger.info("No items were processed to summarize or save.")
    else:
        results_main_df = pd.DataFrame(all_results_main)
        results_prm_iterations_df = pd.DataFrame(all_prm_iteration_data)
        output_excel_filename1 = "D:\\data_science\\final_project\\MAS-PRM\\evaluation\\evaluation_results_with_prm_iterations.xlsx"
        output_excel_filename2 = "D:\\data_science\\final_project\\MAS-PRM\\evaluation\\PRM_Iteration_Details.xlsx"
        try:
            with pd.ExcelWriter(output_excel_filename1, engine='openpyxl') as writer:
                results_main_df.to_excel(writer, sheet_name='Main_Results', index=False)
                results_prm_iterations_df.to_excel(writer, sheet_name='PRM_Iteration_Details', index=False)
            logger.info(f"Detailed evaluation results and PRM iteration data saved to: {output_excel_filename1}n")
            # results_main_df.to_csv(output_excel_filename1, index=False, encoding="utf-8-sig")
            # results_prm_iterations_df.to_csv(output_excel_filename2, index=False, encoding="utf-8-sig")
            # logger.info(f"Detailed evaluation results and PRM iteration data saved to: \n {output_excel_filename1} \n {output_excel_filename2}\n")
        except Exception as e:
            logger.error("main_Excel_Save_Error", f"Error saving results to Excel: {e}. Please ensure 'openpyxl' is installed ('pip install openpyxl').")
        numeric_cols_for_avg = [
            "R_score_final_answer", "label_l_final_answer", "llm_similarity_final_answer",
            "cross_entropy_loss_final_answer", "nlp_bleu", "nlp_rouge1", "nlp_rouge2",
            "nlp_rougeL", "nlp_meteor", "nlp_bert_precision", "nlp_bert_recall", "nlp_bert_f1"
        ]
        avg_scores_summary = {}
        logger.info("Average scores for processed items (Main Results):")
        for col_name in numeric_cols_for_avg:
            if col_name in results_main_df.columns:
                valid_numeric_series = pd.to_numeric(results_main_df[col_name], errors='coerce').dropna()
                if not valid_numeric_series.empty:
                    avg_scores_summary[f"avg_{col_name}"] = valid_numeric_series.mean()
                else:
                    avg_scores_summary[f"avg_{col_name}"] = "N/A (no valid numeric data)"
            else:
                avg_scores_summary[f"avg_{col_name}"] = "N/A (column not found in results)"
            avg_value_to_print = avg_scores_summary.get(f"avg_{col_name}")
            if isinstance(avg_value_to_print, float):
                 logger.info(f"  {avg_value_to_print:.4f} : {col_name}")
            else:
                 logger.info(f"  {str(avg_value_to_print)} : {col_name}")

    logger.section_end("Overall Results Summary and Export")
    logger.section_end("Main Evaluation Flow")

if __name__ == "__main__":
    main()