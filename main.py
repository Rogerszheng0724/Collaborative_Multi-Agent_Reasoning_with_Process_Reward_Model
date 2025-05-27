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

# --- Terminal Logger (Define early as it's used by dummies) ---
class TerminalLogger:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def _print(self, tag, stage, message, indent_level=0):
        if self.verbose:
            indent = "  " * indent_level
            print(f"[{tag}][{stage}]{indent} {message}")

    def thoughtflow(self, stage, message, detail_level=0):
        self._print("THOUGHTFLOW", stage, message, detail_level)

    def discussion(self, stage, agent_name, message, detail_level=0):
        indent = "  " * detail_level
        if self.verbose:
            content_str = str(message) if message is not None else "No content"
            print(f"[DISCUSSION][{stage}][{agent_name}]{indent} \n{content_str}")
            print(f"{'-'*70}")

    def answer(self, source_system, content, is_final=False, detail_level=0):
        tag_prefix = "FINAL " if is_final else "INTERMEDIATE "
        indent = "  " * detail_level
        if self.verbose:
            content_str = str(content) if content is not None else "No content"
            print(f"[{tag_prefix}ANSWER][{source_system}]{indent} \n{content_str}")
            print(f"{'='*70}")

    def section_start(self, section_name):
        if self.verbose:
            print(f"\n{'#'*20} STARTING: {section_name.upper()} {'#'*20}")

    def section_end(self, section_name):
        if self.verbose:
            print(f"{'#'*20} FINISHED: {section_name.upper()} {'#'*20}\n")

    def error(self, stage, message):
        print(f"[ERROR][{stage}] {message}")
        # traceback.print_exc() # Optional: print full stack trace

    def warning(self, *args): # MODIFIED: Changed to use *args
        """
        記錄警告訊息。
        可以接受兩種參數模式：
        1. warning(message_string): 來自 LOT.py 的調用模式，stage 會設為 "LOT_System"。
        2. warning(stage_string, message_string): new_main.py 內部的標準調用模式。
        """
        if len(args) == 1:
            # 假設是從 LOT.py 調用: logger.warning(message)
            stage = "LOT_System"  # 或者一個更通用的預設 stage，例如 "DefaultStage"
            message = args[0]
            print(f"[WARNING][{stage}] {message}")
        elif len(args) == 2:
            # 假設是從 new_main.py 內部調用: logger.warning(stage, message)
            stage = args[0]
            message = args[1]
            print(f"[WARNING][{stage}] {message}")
        else:
            # 處理未預期的參數數量
            print(f"[WARNING][UnknownContext] Invalid warning call with args: {args}")

    def info(self, message):
         if self.verbose:
            print(f"[INFO] {message}")

# --- LLM Call Retry Logic ---
def call_llm_with_retry(llm_generate_function, prompt, logger, llm_name="LLM", max_retries=5, initial_delay=30, max_delay=300):
    """
    LLM call function with retry logic.
    """
    retries = 0
    current_delay = initial_delay
    while retries < max_retries:
        try:
            response = llm_generate_function(prompt=prompt)
            return response
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit_error = ("429" in error_str or
                                   "quota" in error_str or
                                   "rate limit" in error_str or
                                   "resourceexhausted" in error_str.replace(" ", ""))

            if is_rate_limit_error:
                retries += 1
                if retries >= max_retries:
                    logger.error(f"{llm_name}_RETRY", f"Max retries ({max_retries}) reached. Aborting call. Last error: {e}")
                    raise

                suggested_delay_match = re.search(r"retry_delay.*?seconds:\s*(\d+)", error_str)
                if suggested_delay_match:
                    wait_time = int(suggested_delay_match.group(1)) + random.uniform(1, 5)
                else:
                    wait_time = current_delay + random.uniform(0, current_delay * 0.2)

                wait_time = min(wait_time, max_delay)

                logger.warning(f"{llm_name}_RETRY", f"Rate limit error encountered. Retrying in {wait_time:.2f} seconds (Attempt {retries}/{max_retries})... Error: {e}") # This call should now work
                time.sleep(wait_time)
                current_delay = min(current_delay * 2, max_delay)
            else:
                logger.error(f"{llm_name}_RETRY", f"Encountered non-retryable LLM call error: {e}")
                raise
    logger.error(f"{llm_name}_RETRY", "Failed to get LLM response after retry loop (should not reach here).")
    raise Exception("LLM call failed after multiple retries without returning or re-raising specific error.")


# --- Download NLTK resources ---
def download_nltk_resource(resource_name, download_name, logger):
    """Helper function to download NLTK resources."""
    try:
        nltk.data.find(resource_name)
        logger.info(f"NLTK resource '{resource_name}' found.")
    except LookupError:
        logger.info(f"NLTK resource '{resource_name}' not found. Attempting to download '{download_name}'...")
        try:
            nltk.download(download_name, halt_on_error=False) # Continue even if download reports error
            logger.info(f"NLTK download attempt for '{download_name}' completed.")
            # Try to find it again after download attempt
            nltk.data.find(resource_name)
            logger.info(f"NLTK resource '{resource_name}' verified after download attempt.")
        except LookupError: # Still not found after download attempt
            logger.error("NLTK_DOWNLOAD", f"NLTK resource '{resource_name}' still not found after download attempt. This may cause issues with NLP metrics (e.g., METEOR).")
            logger.error("NLTK_DOWNLOAD", f"Please try running 'import nltk; nltk.download(\"{download_name}\")' manually in a Python interpreter.")
        except urllib.error.URLError as e:
            logger.error("NLTK_DOWNLOAD", f"Failed to download '{download_name}' due to network issue: {e}")
        except Exception as e:
            logger.error("NLTK_DOWNLOAD", f"Unexpected error during NLTK download or verification for '{download_name}': {e}\n{traceback.format_exc()}")
    except Exception as e:
        logger.error("NLTK_CHECK", f"Unexpected error while checking NLTK resource '{resource_name}': {e}")

temp_logger_for_nltk = TerminalLogger(verbose=True)
download_nltk_resource('tokenizers/punkt', 'punkt', temp_logger_for_nltk)
download_nltk_resource('corpora/wordnet', 'wordnet', temp_logger_for_nltk)
download_nltk_resource('corpora/omw-1.4', 'omw-1.4', temp_logger_for_nltk) # Open Multilingual Wordnet, often needed with wordnet
del temp_logger_for_nltk

# --- Load environment variables ---
load_dotenv()

# --- Define Base Dummy Classes Globally ---
class BaseDummyLLM:
    def __init__(self, api_key=None, model_name="dummy_model", logger=None):
        self.api_key = api_key
        self.model_name = model_name
        if logger and hasattr(logger, 'info'):
            self.logger = logger
        else:
            self.logger = TerminalLogger(verbose=False)
        self.logger.info(f"BaseDummyLLM initialized, model: {self.model_name}")

    def generate(self, prompt, temperature=0.7):
        self.logger.info(f"BaseDummyLLM ({self.model_name}): Generating dummy response for prompt: {prompt[:50]}...")
        return f"LLM dummy response (BaseDummyLLM for {self.model_name}): {prompt}..."

    def generate_with_simulated_score(self, prompt, temperature=0.7):
        response = self.generate(prompt, temperature)
        return response, 0.1

class BaseDummyEmbedder:
    def __init__(self, api_key=None, model_name="dummy_embedding_model", logger=None):
        self.api_key = api_key
        self.model_name = model_name
        if logger and hasattr(logger, 'info'):
            self.logger = logger
        else:
            self.logger = TerminalLogger(verbose=False)
        self.logger.info(f"BaseDummyEmbedder initialized, model: {self.model_name}")

    def calculate_similarity(self, text1, text2):
        return 0.5

# --- Define Dummy System Classes Globally ---
class DummyGraphOfThoughts: # Based on version [1]
    def __init__(self, llm_interface, logger=None):
        self.llm = llm_interface
        if logger and hasattr(logger, 'info'):
            self.logger = logger
        else:
            self.logger = TerminalLogger(verbose=False)
        self.logger.info("DummyGraphOfThoughts initialized.")
        self._thoughts = {} # Internal store for dummy thoughts
        self._next_id = 1

    def _create_dummy_thought(self, content, score=0.0, prm_justification="Dummy justification", parents_ids=None):
        thought_id = f"dummy_thought_{self._next_id}"
        self._next_id += 1
        class DummyThought:
            def __init__(self, id, content, score, prm_justification, parents_ids):
                self.id = id
                self.content = content
                self.score = score
                self.prm_justification = prm_justification
                self.parents = parents_ids if parents_ids else [] # Store parent IDs
        thought = DummyThought(thought_id, content, score, prm_justification, parents_ids)
        self._thoughts[thought_id] = thought
        return thought

    def generate_thoughts(self, task_description, num, from_ids=None): # Renamed from generate_and_evaluate_thoughts for clarity and added from_ids
        self.logger.info(f"Dummy GOT: Generating {num} thoughts for '{task_description[:20]}...' from IDs {from_ids if from_ids else 'root'}")
        thoughts = []
        for i in range(num):
            score = random.uniform(0.1, 0.9)
            justification = f"Dummy PRM justification for thought (score: {score:.2f})"
            thought = self._create_dummy_thought(
                f"GOT dummy thought {i+1} for {task_description}...",
                score=score,
                prm_justification=justification,
                parents_ids=from_ids
            )
            thoughts.append(thought)
        return thoughts

    def refine_thought(self, thought_id, task_description, instruction): # Renamed from refine_and_evaluate_thought
        self.logger.info(f"Dummy GOT: Refining thought ID '{thought_id}' for task '{task_description[:20]}...' with instruction '{instruction[:20]}...'")
        if thought_id not in self._thoughts:
            self.logger.error("DummyGOT", f"Thought ID {thought_id} not found for refinement.")
            return None
        original_thought = self._thoughts[thought_id]
        refined_score = min(original_thought.score + random.uniform(0.05, 0.2), 1.0) # Usually higher after refinement
        refined_justification = f"Dummy PRM justification for refined thought (original ID: {thought_id}, score: {refined_score:.2f})"
        refined_thought = self._create_dummy_thought(
            f"Refined GOT dummy thought for {task_description} (original ID: {thought_id}) based on '{instruction}...'",
            score=refined_score,
            prm_justification=refined_justification,
            parents_ids=[thought_id]
        )
        return refined_thought

    def aggregate_thoughts(self, ids, task_description): # MODIFIED: Parameter name changed from thought_ids_to_aggregate to ids to match potential calls
        self.logger.info(f"Dummy GOT: Aggregating thoughts {ids} for task '{task_description[:20]}...'")
        aggregated_content_parts = []
        parent_ids_for_agg = []
        base_score = 0.0
        num_aggregated = 0
        # Ensure 'ids' is iterable, even if it was meant to be 'thought_ids_to_aggregate'
        if not isinstance(ids, list): # Simple check, could be more robust
            self.logger.warning("DummyGOT_Aggregate", f"Parameter 'ids' for aggregation was not a list: {ids}. Attempting to proceed if it's a single ID string.")
            ids = [ids] if isinstance(ids, str) else []


        for tid in ids:
            if tid in self._thoughts:
                thought = self._thoughts[tid]
                aggregated_content_parts.append(thought.content + "...") # Take part of content
                parent_ids_for_agg.append(tid)
                base_score += thought.score
                num_aggregated +=1
            else:
                self.logger.warning("DummyGOT", f"Thought ID {tid} not found for aggregation.")
        if not aggregated_content_parts:
            return None

        final_aggregated_content = "Aggregated: " + " | ".join(aggregated_content_parts)
        aggregated_score = (base_score / num_aggregated) * random.uniform(0.9, 1.1) if num_aggregated > 0 else 0.0
        aggregated_score = min(max(aggregated_score, 0.0), 1.0)
        aggregated_justification = f"Dummy PRM justification for aggregated thought (score: {aggregated_score:.2f})"

        aggregated_thought_obj = self._create_dummy_thought(
            final_aggregated_content,
            score=aggregated_score,
            prm_justification=aggregated_justification,
            parents_ids=parent_ids_for_agg
        )
        return aggregated_thought_obj

    def get_thought(self, thought_id): # Added helper
        return self._thoughts.get(thought_id)

    def print_graph(self): self.logger.info(f"Dummy GOT: Printing graph (contains {len(self._thoughts)} thoughts).") # More informative
    def rank_thoughts(self):
        self.logger.info("Dummy GOT: Returning thoughts sorted by score.")
        if not self._thoughts: return []
        # Ensure all thoughts are actual DummyThought objects if mixed returns happened before
        valid_thoughts = [t for t in self._thoughts.values() if hasattr(t, 'score')]
        return sorted(valid_thoughts, key=lambda t: t.score, reverse=True)


class DummyLayerOfThoughts:
    def __init__(self, llm_interface, logger=None, prm_evaluator_llm=None):
        self.llm = llm_interface
        self.prm_evaluator_llm = prm_evaluator_llm or llm_interface
        if logger and hasattr(logger, 'info'):
            self.logger = logger
        else:
            self.logger = TerminalLogger(verbose=False)
        self.logger.info("DummyLayerOfThoughts initialized.")
    def run_pipeline(self, conceptual_steps, main_task_description, initial_input=None, min_layer_prm_score_threshold=0.3):
        self.logger.info(f"Dummy LOT: Running pipeline for '{str(initial_input)[:20]}...' on task '{main_task_description[:20]}...'")
        return f"LOT dummy plan for {str(initial_input)} based on steps: {conceptual_steps}"

class DummyReversalOfThought:
    def __init__(self, llm_interface, embedding_model_interface, similarity_threshold=0.7, logger=None):
        self.llm = llm_interface
        self.embedder = embedding_model_interface
        self.similarity_threshold = similarity_threshold
        if logger and hasattr(logger, 'info'):
            self.logger = logger
        else:
            self.logger = TerminalLogger(verbose=False)
        self.logger.info("DummyReversalOfThought initialized.")
    def preference_guided_reverse_reasoning_warmup(self, demonstrations, main_task_description_for_prm, warm_iterations=1):
        self.logger.info(f"Dummy ROT: PGRR warmup for task '{main_task_description_for_prm[:20]}...' with {len(demonstrations)} demos, {warm_iterations} iterations.")
        return "ROT dummy PGRR output after warmup"
    def cognitive_preference_manager(self, original_task_prompt_text, llm_taste_prompt_text, main_task_description_for_prm):
        self.logger.info(f"Dummy ROT: CPM for task '{main_task_description_for_prm[:20]}...' using taste '{llm_taste_prompt_text[:20]}...'")
        return f"ROT dummy CPM refined prompt from original: {original_task_prompt_text}..."
    def solve_task_with_final_prompt(self, prompt, instance): # Parameter names changed to match usage in orchestrator
        self.logger.info(f"Dummy ROT: Solving '{str(instance)[:20]}...' with prompt '{str(prompt)[:20]}...'")
        return f"ROT dummy solution for {str(instance)} using prompt {str(prompt)}"


# --- Initialize Pointers to Dummies, Attempt to Override with Real Imports ---
_RealGotGeminiLLM = None
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
    # Ensure your custom modules GOT, LOT, ROT are in PYTHONPATH or same directory
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
    print(f"[ERROR] Unexpected error during import of real modules: {e}")
    traceback.print_exc()
    print("[INFO] Continuing with globally defined dummy classes.")

# --- API Key Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("[WARNING] GEMINI_API_KEY not found in environment variables.")
    if IMPORTS_SUCCESSFUL:
        print("[WARNING] LLM calls may fail if API key for real modules is not set.")
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" and IMPORTS_SUCCESSFUL:
    print("[WARNING] Using placeholder API key. Real LLM calls may fail.")

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
            error_msg = f"LLM interface for agent {self.name} is incorrect or not initialized."
            self.logger.error("MAS_DEBATE_AGENT", error_msg)
            return f"Error: {error_msg}"

        full_prompt = f"You are {self.name}, an expert.\nCurrent discussion context (first 200 chars): {context_summary}\nYour specific task: {prompt}\nPlease provide your response. Ensure it is clear, correct, structured, and directly addresses the task."
        self.logger.thoughtflow("MAS_DEBATE", f"Agent {self.name} is thinking...\nPrompt focus: {prompt[:100]}...")

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
            self.logger.error("MAS_DEBATE_AGENT", f"Agent {self.name} encountered a critical error during LLM call (after retries): {e}")
            return f"Error: {self.name} had an LLM error during response generation (after retries)."

        response_str = str(response) if response is not None else f"{self.name} failed to generate a response."
        self.logger.discussion("MAS_DEBATE_TURN", self.name, response_str)
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
            self.logger.warning("MASOrchestrator", "API key is invalid or a placeholder. MAS may use dummy LLMs if real modules were imported.") # This call should now work

        try:
            if IMPORTS_SUCCESSFUL and GotGeminiLLM_cls == _RealGotGeminiLLM:
                self.logger.info("Instantiating real GOT.GeminiLLM (from GOT.py).")
                self.got_llm = GotGeminiLLM_cls(api_key=self.api_key) # Assuming real LLM might not take logger
            else:
                self.logger.info(f"Instantiating GotGeminiLLM_cls as {GotGeminiLLM_cls.__name__} (with logger).")
                self.got_llm = GotGeminiLLM_cls(api_key=self.api_key, logger=self.logger)
            self.got_system = GraphOfThoughts_cls(llm_interface=self.got_llm, logger=self.logger)
            self.logger.info("GraphOfThoughts (GOT) system initialized with its LLM.")

            if IMPORTS_SUCCESSFUL and LotGeminiLLMInterface_cls == _RealLotGeminiLLMInterface:
                 self.lot_llm = LotGeminiLLMInterface_cls(api_key=self.api_key)
            else:
                self.lot_llm = LotGeminiLLMInterface_cls(api_key=self.api_key, logger=self.logger)
            # Pass the same LLM instance for PRM evaluation if LOT uses one internally, or a dedicated one if available
            prm_eval_llm_for_lot = self.lot_llm # Default to same LLM
            self.lot_system = LayerOfThoughts_cls(llm_interface=self.lot_llm, logger=self.logger, prm_evaluator_llm=prm_eval_llm_for_lot)
            self.logger.info("LayerOfThoughts (LOT) system initialized.")

            if IMPORTS_SUCCESSFUL and RotGeminiLLMInterface_cls == _RealRotGeminiLLMInterface:
                self.rot_llm = RotGeminiLLMInterface_cls(api_key=self.api_key)
            else:
                self.rot_llm = RotGeminiLLMInterface_cls(api_key=self.api_key, logger=self.logger)

            if IMPORTS_SUCCESSFUL and RotGeminiEmbeddingInterface_cls == _RealRotGeminiEmbeddingInterface:
                self.rot_embedder = RotGeminiEmbeddingInterface_cls(api_key=self.api_key)
            else:
                self.rot_embedder = RotGeminiEmbeddingInterface_cls(api_key=self.api_key, logger=self.logger)

            self.rot_system = ReversalOfThought_cls(
                llm_interface=self.rot_llm,
                embedding_model_interface=self.rot_embedder,
                logger=self.logger
            )
            self.logger.info("ReversalOfThought (ROT) system initialized.")

            # LLM for Debate, Synthesis, Optimizer - prefer a real one if available
            candidate_llms = [self.got_llm, self.rot_llm, self.lot_llm]
            chosen_llm_source = "None"
            shared_llm = None

            for llm_candidate in candidate_llms:
                if llm_candidate and hasattr(llm_candidate, 'generate') and not isinstance(llm_candidate, BaseDummyLLM):
                    shared_llm = llm_candidate
                    if llm_candidate == self.got_llm: chosen_llm_source = "GOT LLM"
                    elif llm_candidate == self.rot_llm: chosen_llm_source = "ROT LLM"
                    elif llm_candidate == self.lot_llm: chosen_llm_source = "LOT LLM"
                    break

            if shared_llm:
                self.debate_llm = shared_llm
                self.synthesis_llm = shared_llm
                self.iterative_optimizer_llm = shared_llm
                self.logger.info(f"Debate, Synthesis, Optimizer LLMs set to shared real LLM from {chosen_llm_source}.")
            else: # This call should now work
                self.logger.warning("MASOrchestrator", "No suitable real LLM found from initialized systems for Debate/Synthesis/Optimizer. Using BaseDummyLLM for these roles.")
                dummy_fallback_llm = BaseDummyLLM(api_key=self.api_key, model_name="shared_dummy_fallback", logger=self.logger)
                self.debate_llm = dummy_fallback_llm
                self.synthesis_llm = dummy_fallback_llm
                self.iterative_optimizer_llm = dummy_fallback_llm

        except Exception as e:
            self.logger.error("MASOrchestrator_Init", f"Critical error during subsystem initialization: {e}")
            self.logger.error("MASOrchestrator_Init", "Detailed error traceback:")
            traceback.print_exc() # This call should now work
            self.logger.warning("MASOrchestrator_Init", "All components will fall back to dummy implementations due to initialization error.")

            # Fallback initialization for all components to ensure orchestrator is usable in a degraded state
            self.got_llm = BaseDummyLLM(api_key=self.api_key, model_name="got_llm_exc_fallback", logger=self.logger)
            self.got_system = DummyGraphOfThoughts(self.got_llm, self.logger)
            self.lot_llm = BaseDummyLLM(api_key=self.api_key, model_name="lot_llm_exc_fallback", logger=self.logger)
            self.lot_system = DummyLayerOfThoughts(self.lot_llm, self.logger, self.lot_llm)
            self.rot_llm = BaseDummyLLM(api_key=self.api_key, model_name="rot_llm_exc_fallback", logger=self.logger)
            self.rot_embedder = BaseDummyEmbedder(api_key=self.api_key, model_name="rot_embed_exc_fallback", logger=self.logger)
            self.rot_system = DummyReversalOfThought(self.rot_llm, self.rot_embedder, logger=self.logger)
            self.debate_llm = BaseDummyLLM(api_key=self.api_key, model_name="debate_llm_exc_fallback", logger=self.logger)
            self.synthesis_llm = BaseDummyLLM(api_key=self.api_key, model_name="synthesis_llm_exc_fallback", logger=self.logger)
            self.iterative_optimizer_llm = BaseDummyLLM(api_key=self.api_key, model_name="optimizer_llm_exc_fallback", logger=self.logger)


    def conduct_mas_debate(self, mission_context,rot_idea, got_idea=None, lot_idea=None, max_rounds=3): # Removed prm and prm_prompt, they are not used here
        self.logger.section_start(f"MAS Style Debate (Targeting {max_rounds} Rounds)")
        proactive_delay_between_turns = 5 # As in [1]


        debate_transcript = []
        # Truncate inputs for context summary to avoid overly long summaries
        mission_context_summary_part = str(mission_context)
        rot_idea_summary_part = str(rot_idea)
        # got_idea_summary_part = str(got_idea)
        lot_idea_summary_part = str(lot_idea)

        discussion_context_summary = f"Mission Context (partial):\n{mission_context_summary_part}...\n"
        discussion_context_summary += f"Initial Core Idea from ROT (partial):\n{rot_idea_summary_part}...\n"
        # discussion_context_summary += f"Initial Core Idea from GOT (partial):\n{got_idea_summary_part}...\n"
        discussion_context_summary += f"Initial Detailed Plan from LOT (partial):\n{lot_idea_summary_part}...\n"
        discussion_context_summary += "The debate will now commence focusing on these ideas.\n"


        if not self.debate_llm or isinstance(self.debate_llm, BaseDummyLLM) or not hasattr(self.debate_llm, 'generate'): # This call should now work
            self.logger.warning("MAS_DEBATE", "Debate LLM not effectively initialized or is dummy. Using simulated debate.")
            debate_transcript.append({"speaker": "Moderator", "utterance": "Simulated debate starting due to LLM configuration."})

            # Use the LLMs associated with each system if they are dummies, otherwise a generic dummy
            rot_sim_statement = self.rot_system.llm.generate("ROT opening statement prompt") if hasattr(self.rot_system, 'llm') and isinstance(self.rot_system.llm, BaseDummyLLM) else "Simulated ROT statement."
            # got_sim_statement = self.got_system.llm.generate("GOT opening statement prompt") if hasattr(self.got_system, 'llm') and isinstance(self.got_system.llm, BaseDummyLLM) else "Simulated GOT statement."
            lot_sim_statement = self.lot_system.llm.generate("LOT opening statement prompt") if hasattr(self.lot_system, 'llm') and isinstance(self.lot_system.llm, BaseDummyLLM) else "Simulated LOT statement."

            critic_llm_for_sim = self.debate_llm if (self.debate_llm and hasattr(self.debate_llm, 'generate') and not isinstance(self.debate_llm, BaseDummyLLM)) else BaseDummyLLM(logger=self.logger, model_name="critic_sim_dummy")
            critic_sim_statement = critic_llm_for_sim.generate("Critic statement prompt on simulated ideas.")

            debate_transcript.append({"speaker": "ROT_Representative", "utterance": rot_sim_statement})
            # debate_transcript.append({"speaker": "GOT_Representative", "utterance": got_sim_statement})
            debate_transcript.append({"speaker": "LOT_Representative", "utterance": lot_sim_statement})
            debate_transcript.append({"speaker": "Critical_Analyst", "utterance": critic_sim_statement})
            self.logger.section_end(f"MAS Style Debate (Simulated)")
            return debate_transcript

        rot_agent = DebateAgent("ROT_Representative", self.debate_llm, self.logger)
        # got_agent = DebateAgent("GOT_Representative", self.debate_llm, self.logger)
        lot_agent = DebateAgent("LOT_Representative", self.debate_llm, self.logger)
        critic_agent = DebateAgent("Critical_Analyst", self.debate_llm, self.logger)

        opening_statement = f"Debate Topic: In-depth discussion based on the mission context and ideas from ROT and LOT.\n{discussion_context_summary}" # Already includes partial ideas
        debate_transcript.append({"speaker": "Moderator", "utterance": opening_statement})

        current_round_count = 0 # Tracks how many agents have spoken
        # Define prompts carefully to avoid excessive length by referencing full original ideas
        # but the context_summary passed to speak() will be truncated.
        # The agents receive full ideas in their specific prompts.

        # ROT's turn
        if current_round_count < max_rounds:
            current_round_count += 1
            self.logger.info(f"Debate Round {current_round_count}: ROT Representative")
            prompt_rot = f"""
                As the ROT(Reversal Of Thought) Representative, your core idea is: '{str(rot_idea)}'. The mission is: '{str(mission_context)}'.
                1. Elaborate on how your idea addresses the core problem and highlight its key strengths.
                2. Critically evaluate the lOT idea: '{str(lot_idea)}'.
                For 1 and 2, discuss their potential weaknesses, overlooked aspects, or limitations compared to your ROT idea, and explain why your approach might be preferable.
                """
            rot_statement = rot_agent.speak(prompt_rot, discussion_context_summary) # discussion_context_summary provides overview
            debate_transcript.append({"speaker": rot_agent.name, "utterance": rot_statement})
            discussion_context_summary += f"\nRound {current_round_count} - {rot_agent.name} (statement partial):\n{rot_statement}...\n"
            if proactive_delay_between_turns > 0:
                self.logger.info(f"Debate: Proactively sleeping for {proactive_delay_between_turns}s after {rot_agent.name}.")
                time.sleep(proactive_delay_between_turns)

        # # GOT's turn
        # if current_round_count < max_rounds:
        #     current_round_count += 1
        #     self.logger.info(f"Debate Round {current_round_count}: GOT Representative")
        #     prompt_got = f"""
        #         As the GOT(Graph Of Thoughts) Representative, your core idea is: '{str(got_idea)}'. The mission is: '{str(mission_context)}'.
        #         1. Elaborate on how your idea addresses the core problem and highlight its key strengths.
        #         2. Critically evaluate the ROT idea: '{str(rot_idea)}'.
        #         For 1 and 2, discuss their potential weaknesses, overlooked aspects, or limitations compared to your GOT idea, and explain why your approach might be preferable.
        #         """
        #     got_statement = got_agent.speak(prompt_got, discussion_context_summary)
        #     debate_transcript.append({"speaker": got_agent.name, "utterance": got_statement})
        #     discussion_context_summary += f"\nRound {current_round_count} - {got_agent.name} (statement partial):\n{got_statement}...\n"
        #     if proactive_delay_between_turns > 0:
        #         self.logger.info(f"Debate: Proactively sleeping for {proactive_delay_between_turns}s after {got_agent.name}.")
        #         time.sleep(proactive_delay_between_turns)

        # LOT's turn
        if current_round_count < max_rounds:
            current_round_count += 1
            self.logger.info(f"Debate Round {current_round_count}: LOT Representative")
            prompt_lot = f"""
                As the LOT(Layer Of Thoughts) Representative, your core idea/plan is: '{str(lot_idea)}'. The mission is: '{str(mission_context)}'.
                1. Elaborate on how your detailed plan addresses the core problem and highlight its key strengths and feasibility.
                2. Critically evaluate the ROT idea: '{str(rot_idea)}'.
                For 2 and 3, discuss their potential weaknesses, overlooked aspects, or limitations compared to your LOT plan, and explain why your approach might be preferable.
                """
            lot_statement = lot_agent.speak(prompt_lot, discussion_context_summary)
            debate_transcript.append({"speaker": lot_agent.name, "utterance": lot_statement})
            discussion_context_summary += f"\nRound {current_round_count} - {lot_agent.name} (statement partial):\n{lot_statement}...\n"
            if proactive_delay_between_turns > 0:
                self.logger.info(f"Debate: Proactively sleeping for {proactive_delay_between_turns}s after {lot_agent.name}.")
                time.sleep(proactive_delay_between_turns)

        # Critic's turn (only if there's room in max_rounds and previous statements exist)
        # Get latest statements for critic, fall back to placeholders if not generated
        final_rot_statement_for_critic = next((item['utterance'] for item in reversed(debate_transcript) if item['speaker'] == 'ROT_Representative'), rot_idea)
        # final_got_statement_for_critic = next((item['utterance'] for item in reversed(debate_transcript) if item['speaker'] == 'GOT_Representative'), got_idea)
        final_lot_statement_for_critic = next((item['utterance'] for item in reversed(debate_transcript) if item['speaker'] == 'LOT_Representative'), lot_idea)

        if current_round_count < max_rounds:
            current_round_count += 1
            self.logger.info(f"Debate Round {current_round_count}: Critical Analyst")
            prompt_critic = f"""
                As the Critical_Analyst, your task is to evaluate the ideas presented by ROT and LOT representatives for the mission: '{str(mission_context)}'.
                ROT's latest statement/idea: '{str(rot_idea)}'
                LOT's latest statement/idea: '{str(lot_idea)}'
                Critically evaluate all three. Identify potential weaknesses, overlooked aspects, or inconsistencies in each, relative to the mission.
                Assess the correctness and completeness of each proposed solution.
                Suggest specific improvements or points of caution for each.
                Provide a balanced overall critique.

                At the very end of your response, include this final section:

                ### Synthesized Accurate Answer:
                (Provide a clear, concise answer that integrates the best insights from ROT and LOT. This section must appear last. If it is a multiple-choice question, your answer must exactly match one of the options provided in the question.)
                """
            critic_statement = critic_agent.speak(prompt_critic, discussion_context_summary)
            debate_transcript.append({"speaker": critic_agent.name, "utterance": critic_statement})
            # No need to add critic's statement to discussion_context_summary if it's the last turn

        self.logger.section_end(f"MAS Style Debate (Completed {current_round_count} Rounds)")
        return debate_transcript

    def _get_prm_feedback_for_reasoning_process(self, task_description, current_reasoning_process, iteration_count_label): # Renamed iteration_count to iteration_count_label
        if not self.iterative_optimizer_llm or not hasattr(self.iterative_optimizer_llm, 'generate'): # This call should now work
            self.logger.warning("MASOrchestrator._get_prm_feedback", "Iterative optimizer LLM not effectively initialized. Returning dummy feedback.")
            return 0.5, "Dummy PRM feedback: Please check logical coherence and completeness. (Optimizer LLM not ready)", "DummyPRM_Evaluator_Uninit"
        if isinstance(self.iterative_optimizer_llm, BaseDummyLLM): # This call should now work
             self.logger.warning("MASOrchestrator._get_prm_feedback", "Iterative optimizer LLM is BaseDummyLLM. Returning dummy feedback.")
             return 0.5, "Dummy PRM feedback (from BaseDummyLLM): Check logic and provide specific improvement points.", "BaseDummyPRM_Evaluator"

        prm_feedback_prompt = f"""
        As an advanced Process Reward Model (PRM) evaluator, your task is to assess the following reasoning process/answer for the given task and provide actionable optimization feedback.
        Main Task:
        {task_description}
        Current Reasoning Process/Answer (to be evaluated, from Cycle {iteration_count_label}):
        \"\"\"
        {current_reasoning_process}
        \"\"\"
        Please evaluate this reasoning process/answer based on:
        1.  **Correctness**: Is the conclusion accurate? Are there errors in intermediate steps?
        2.  **Completeness**: Does it cover all critical aspects of the task? Are there omissions?
        3.  **Logicality & Coherence**: Are the reasoning steps logical? Is the overall presentation coherent?
        4.  **Conciseness**: Is there redundant information or unnecessary complexity?
        5.  **Actionability/Clarity**: If this is a plan or instruction, is it clear and easy to execute?
        Provide an overall score (between 0.0 and 1.0, where 1.0 is perfect) and detailed justification.
        In your justification, clearly point out 1-2 specific points most in need of improvement and briefly state how to improve them.
        Output Format (Strictly Adhere):
        PRM Score: [A float between 0.0 and 1.0]
        PRM Justification: [Detailed assessment covering the aspects above and specific, actionable improvement suggestions]
        """
        self.logger.info(f"MASOrchestrator: Getting PRM feedback for reasoning (Cycle {iteration_count_label})...")
        try:
            llm_response = call_llm_with_retry(
                self.iterative_optimizer_llm.generate,
                prompt=prm_feedback_prompt,
                logger=self.logger,
                llm_name="PRM_Feedback_LLM"
            )
        except Exception as e:
            self.logger.error("MASOrchestrator._get_prm_feedback", f"Critical error during LLM call for PRM feedback (after retries) for cycle {iteration_count_label}: {e}")
            return 0.1, f"Error getting PRM feedback (after retries): {e}", "PRM_Feedback_Error"

        # Ensure llm_response is a string before regex search
        llm_response_str = str(llm_response) if llm_response is not None else ""

        score_match = re.search(r"PRM Score:\s*([0-9.]+)", llm_response_str, re.IGNORECASE)
        justification_match = re.search(r"PRM Justification:\s*(.+)", llm_response_str, re.IGNORECASE | re.DOTALL)

        prm_score = float(score_match.group(1)) if score_match else 0.0
        prm_justification = justification_match.group(1).strip() if justification_match else "Could not parse PRM justification."

        if not score_match: # This call should now work
            self.logger.warning("MASOrchestrator._get_prm_feedback", f"Could not parse PRM score from response for cycle {iteration_count_label}. Raw response: '{llm_response_str[:200]}...'")
            if "Could not parse PRM justification." in prm_justification and "PRM Score:" not in llm_response_str : # Avoid overwriting if justification was parsed but score wasn't
                 prm_justification = f"PRM Evaluator LLM raw output (format may be incorrect for cycle {iteration_count_label}): {llm_response_str}"

        self.logger.info(f"MASOrchestrator: Cycle {iteration_count_label} - PRM Score: {prm_score:.2f}, Justification (start): {prm_justification[:150]}...")
        return prm_score, prm_justification, "ImplicitPRM_LLMEvaluator"

    def ROT_phase(self, task_desc, rot_demos, prob_instance_rot_solve, proactive_delay): # Simplified params for [1]
        self.logger.info("--- ROT Phase ---")
        # Default values, to be updated if ROT runs successfully
        current_rot_solution = f"Default ROT solution for task: {task_desc}..."
        current_refined_task_prompt = task_desc # Start with the original task description

        # Check if the ROT system is not a dummy and has the necessary methods
        if hasattr(self.rot_system, 'cognitive_preference_manager') and \
           hasattr(self.rot_system, 'preference_guided_reverse_reasoning_warmup') and \
           hasattr(self.rot_system, 'solve_task_with_final_prompt') and \
           not isinstance(self.rot_system, DummyReversalOfThought):
            try:
                # Use provided demonstrations or a default if none
                demonstrations_for_pgrr = rot_demos if rot_demos else [("Example input: Describe a cat.", "Example output: A cat is a small, furry mammal often kept as a pet.")]
                self.logger.info(f"ROT: Starting PGRR warmup with {len(demonstrations_for_pgrr)} demonstrations.")
                pgrr_output = self.rot_system.preference_guided_reverse_reasoning_warmup(
                    demonstrations=demonstrations_for_pgrr,
                    main_task_description_for_prm=task_desc, # Use current task description for PRM context if needed by ROT's PRM
                    warm_iterations=1 # Standard warm-up iterations
                )

                # Check if PGRR output is valid before proceeding to CPM
                if pgrr_output and "dummy" not in str(pgrr_output).lower() and "error" not in str(pgrr_output).lower():
                    self.logger.info(f"ROT: PGRR warmup successful. Output (partial): {str(pgrr_output)[:100]}...")
                    self.logger.info("ROT: Proceeding to Cognitive Preference Manager (CPM).")
                    cpm_output = self.rot_system.cognitive_preference_manager(
                        original_task_prompt_text=task_desc, # The original task description
                        llm_taste_prompt_text=pgrr_output,     # Output from PGRR, representing "taste"
                        main_task_description_for_prm=task_desc # Context for any PRM within CPM
                    )

                    # Check if CPM output is valid to be used as a refined prompt
                    if cpm_output and "dummy" not in str(cpm_output).lower() and "error" not in str(cpm_output).lower():
                        current_refined_task_prompt = str(cpm_output) # Update refined task prompt
                        self.logger.info(f"ROT: CPM successful. Refined task prompt (partial): {current_refined_task_prompt[:100]}...")
                    else: # This call should now work
                        self.logger.warning("ROT", "CPM output was invalid, dummy or error. Using original task description for final solve.")
                        current_refined_task_prompt = task_desc # Fallback to original task description

                else: # PGRR output was invalid # This call should now work
                    self.logger.warning("ROT", "PGRR output was invalid, dummy or error. Using original task description for final solve.")
                    current_refined_task_prompt = task_desc # Fallback

                # Solve the task using the (potentially refined) prompt
                # Determine instance: use problem_instance_for_rot_final_solve if provided, else task_desc
                instance_for_final_solve = prob_instance_rot_solve if prob_instance_rot_solve else task_desc
                self.logger.info(f"ROT: Solving task with final prompt. Instance (partial): {str(instance_for_final_solve)[:100]}...")

                rot_solution_candidate = self.rot_system.solve_task_with_final_prompt(
                    final_prompt_text=current_refined_task_prompt, # Parameter name in dummy is 'prompt'
                    problem_input=instance_for_final_solve  # Parameter name in dummy is 'instance'
                )


                if rot_solution_candidate and "dummy" not in str(rot_solution_candidate).lower() and "error" not in str(rot_solution_candidate).lower():
                    current_rot_solution = str(rot_solution_candidate) # Update the final ROT solution
                    self.logger.info(f"ROT: Task solved successfully. Solution (partial): {current_rot_solution[:100]}...")
                else: # This call should now work
                    self.logger.warning("ROT", "Final solve step returned invalid, dummy, or error. Default ROT solution will be used.")
                    # current_rot_solution remains the default initialized at the start of the method

            except Exception as e_rot:
                self.logger.error("ROT_PHASE", f"Error during ROT phase execution: {e_rot}")
                traceback.print_exc() # This call should now work
                self.logger.warning("ROT_PHASE", "ROT phase failed. Default ROT solution and original task description will be used.")
                # Ensure defaults are set if an error occurs mid-process
                current_rot_solution = f"Error-fallback ROT solution for task: {task_desc}..."
                current_refined_task_prompt = task_desc
        else:
            self.logger.info("ROT phase skipped (ROT system is dummy or not fully configured). Using defaults.")

        if proactive_delay > 0 :
            self.logger.info(f"Proactively sleeping for {proactive_delay}s after ROT phase.")
            time.sleep(proactive_delay)
        return current_rot_solution, current_refined_task_prompt


    def GOT_phase(self, task_desc, proactive_delay): # Simplified params for [1]
        self.logger.info("--- GOT Phase Start ---")
        aggregated_thought_content = f"Default GOT aggregated idea for task: {task_desc}..." # Default

        try:
            # Step 1: Generate initial thoughts (e.g., 2 diverse ideas)
            self.logger.info("GOT: Generating initial thoughts...")
            initial_thoughts = self.got_system.generate_thoughts(task_description=task_desc, num=2)

            if not initial_thoughts or not all(hasattr(t, 'id') and hasattr(t, 'content') and hasattr(t, 'score') for t in initial_thoughts): # This call should now work
                self.logger.warning("GOT", "Initial thought generation failed or returned invalid thoughts. Using default.")
                if proactive_delay > 0: time.sleep(proactive_delay)
                return aggregated_thought_content

            self.logger.info(f"GOT: Generated {len(initial_thoughts)} initial thoughts.")
            for idea in initial_thoughts:
                self.logger.info(f"  Initial idea ID {idea.id}: '{idea.content[:60]}...' (Score: {idea.score:.2f})")

            # Step 2: Elaborate on the best initial thought (or first one for simplicity in dummy)
            # Sort by score to pick the best, assuming higher is better
            initial_thoughts.sort(key=lambda t: t.score, reverse=True)
            best_initial_thought = initial_thoughts[0]
            self.logger.info(f"GOT: Elaborating on best initial thought (ID: {best_initial_thought.id})...")

            elaborated_thoughts = self.got_system.generate_thoughts(
                task_description=f"Elaborate on this idea: {best_initial_thought.content}",
                num=1,
                from_ids=[best_initial_thought.id] # Link to parent
            )

            if not elaborated_thoughts or not hasattr(elaborated_thoughts[0], 'id'): # This call should now work
                self.logger.warning("GOT", "Elaboration failed. Using best initial thought as is.")
                elaborated_thought_to_refine = best_initial_thought
            else:
                elaborated_thought_to_refine = elaborated_thoughts[0]
                self.logger.info(f"  Elaborated ID {elaborated_thought_to_refine.id}: '{elaborated_thought_to_refine.content[:60]}...' (Score: {elaborated_thought_to_refine.score:.2f})")

            # Step 3: Refine the elaborated thought
            self.logger.info(f"GOT: Refining thought (ID: {elaborated_thought_to_refine.id})...")
            refined_thought_obj = self.got_system.refine_thought(
                thought_id=elaborated_thought_to_refine.id,
                task_description=task_desc, # Original task context
                instruction="Refine for improved clarity, detail, and actionability."
            )

            final_thought_for_aggregation = None
            if refined_thought_obj and hasattr(refined_thought_obj, 'id'):
                self.logger.info(f"  Refined ID {refined_thought_obj.id}: '{refined_thought_obj.content[:60]}...' (Score: {refined_thought_obj.score:.2f})")
                final_thought_for_aggregation = refined_thought_obj
            else: # This call should now work
                self.logger.warning("GOT", "Refinement failed. Using elaborated thought for potential aggregation.")
                final_thought_for_aggregation = elaborated_thought_to_refine


            # Step 4: Aggregate thoughts - e.g., the refined thought and the second best initial thought if available
            thoughts_to_aggregate_ids = []
            if final_thought_for_aggregation:
                thoughts_to_aggregate_ids.append(final_thought_for_aggregation.id)

            if len(initial_thoughts) > 1: # If there was a second initial thought
                second_best_initial_thought = initial_thoughts[1]
                # Avoid aggregating a thought with its own direct refinement if that's the only other option.
                # Here, we assume initial_thoughts[1] is distinct enough from final_thought_for_aggregation's lineage.
                if second_best_initial_thought.id not in thoughts_to_aggregate_ids and \
                   (not final_thought_for_aggregation.parents or second_best_initial_thought.id not in final_thought_for_aggregation.parents):
                    thoughts_to_aggregate_ids.append(second_best_initial_thought.id)


            if len(thoughts_to_aggregate_ids) >= 2:
                self.logger.info(f"GOT: Aggregating thoughts: {thoughts_to_aggregate_ids}...")
                # The dummy aggregate_thoughts was modified to return a Thought object
                # The original code expected a string. We will get .content from the object.
                # MODIFIED: Changed keyword argument from 'thought_ids_to_aggregate' to 'ids'
                # to match the dummy's definition or a more generic real one.
                aggregated_thought_object = self.got_system.aggregate_thoughts(
                    ids=thoughts_to_aggregate_ids, # Changed from thought_ids_to_aggregate
                    task_description=task_desc
                )
                if aggregated_thought_object and hasattr(aggregated_thought_object, 'content'):
                    aggregated_thought_content = aggregated_thought_object.content
                    self.logger.info(f"  Aggregated result (ID {aggregated_thought_object.id}): '{aggregated_thought_content[:60]}...' (Score: {aggregated_thought_object.score:.2f})")
                else: # This call should now work
                    self.logger.warning("GOT", "Aggregation did not yield a valid result. Using the best single thought.")
                    if final_thought_for_aggregation: # This would be the refined or elaborated one
                         aggregated_thought_content = final_thought_for_aggregation.content
            elif final_thought_for_aggregation: # Only one thought, use its content
                self.logger.info("GOT: Only one primary thought available after refinement. Using it directly.")
                aggregated_thought_content = final_thought_for_aggregation.content
            else: # Should not happen if initial thoughts were generated # This call should now work
                self.logger.warning("GOT", "No suitable thought available to form the GOT phase output.")

            # self.got_system.print_graph() # Optional: print the graph structure

        except Exception as e_got:
            self.logger.error("GOT_PHASE", f"Error during GOT phase execution: {e_got}")
            traceback.print_exc()
            # aggregated_thought_content remains the default

        if proactive_delay > 0:
            self.logger.info(f"Proactively sleeping for {proactive_delay}s after GOT phase.")
            time.sleep(proactive_delay)
        return aggregated_thought_content


    def LOT_phase(self, task_desc, got_best_idea_content = None, proactive_delay = 3): # Simplified params for [1]
        self.logger.info("--- LOT Phase ---")
        lot_detailed_plan_str = f"Default LOT detailed plan for task: {task_desc}..." # Default

        try:
            self.logger.info(f"LOT: Running pipeline based on GOT idea (partial): {str(got_best_idea_content)[:100]}...")
            plan_output = self.lot_system.run_pipeline(
                conceptual_steps=["Analyze task and GOT idea", "Formulate detailed execution steps", "Structure the final plan", "Generate and present the answer based on the plan"],
                main_task_description=task_desc,
                # initial_input=got_best_idea_content # LOT takes the idea from GOT as input
            )
            if plan_output and "dummy" not in str(plan_output).lower() and str(plan_output).strip():
                lot_detailed_plan_str = str(plan_output)
                self.logger.info(f"LOT: Pipeline successful. Plan (partial): {lot_detailed_plan_str[:100]}...")
            else: # This call should now work
                self.logger.warning("LOT", "run_pipeline returned None, empty, or dummy. Using default LOT plan.")
        except Exception as e_lot:
            self.logger.error("LOT_PHASE", f"LOT phase execution error: {e_lot}. Using default LOT plan.")
            traceback.print_exc()
            # lot_detailed_plan_str remains the default

        if proactive_delay > 0:
            self.logger.info(f"Proactively sleeping for {proactive_delay}s after LOT phase.")
            time.sleep(proactive_delay)
        return lot_detailed_plan_str


    def run_collaborative_task(self, initial_task_description, rot_demonstrations=None, problem_instance_for_rot_final_solve=None, num_debate_rounds=4, num_prm_iterations=3, index=None):
        # num_prm_iterations here means number of full (Pipeline -> PRM Feedback -> Optimize Input) cycles

        self.logger.section_start(f"Collaborative Task for CSV Index {index} (Targeting {num_prm_iterations} PRM-Guided Cycles)")
        self.logger.info(f"Initial task description (CSV Index {index}): {initial_task_description[:100]}...")

        proactive_delay_between_stages = 5 # Shortened delay from [1] for faster cycles if desired, adjust as needed

        # Initialize variables that will be updated in each cycle
        current_task_input_for_pipeline = initial_task_description
        best_artifact_overall = f"Placeholder: No successful artifact generated for CSV Index {index}"
        best_prm_score_overall = -1.0
        prm_iteration_details_for_excel = [] # This will store dicts for each PRM cycle's eval

        # For summarizing thoughtflow across cycles
        accumulated_thoughtflow_summary_parts = []
        original_thoughtflow_summary_first_cycle = "" # Capture output of the very first cycle

        for current_prm_cycle_num in range(num_prm_iterations):
            self.logger.section_start(f"PRM Cycle {current_prm_cycle_num + 1}/{num_prm_iterations} for CSV Index {index}")
            self.logger.info(f"Current task input for this cycle (partial): {str(current_task_input_for_pipeline)[:100]}...")

            # --- Run the full pipeline (ROT, GOT, LOT) ---
            # ROT Phase
            rot_solution, refined_task_prompt_from_rot = self.ROT_phase(
                task_desc=current_task_input_for_pipeline,
                rot_demos=rot_demonstrations,
                prob_instance_rot_solve=problem_instance_for_rot_final_solve,
                proactive_delay=proactive_delay_between_stages
            )
            # GOT Phase - uses the task description potentially refined by ROT if that's the design, or original/current_task_input
            # For this version, let's assume GOT works on the task description that this cycle received.
            # If ROT refines the *task description itself* for all subsequent modules, then `refined_task_prompt_from_rot` should be used by GOT.
            # If ROT produces a *solution component* or a *refined query for itself*, then GOT might still use `current_task_input_for_pipeline`.
            # Let's assume current_task_input_for_pipeline is the primary task spec for GOT/LOT. ROT's output is a component.
            # got_aggregated_idea = self.GOT_phase(
            #     task_desc=current_task_input_for_pipeline, # Or refined_task_prompt_from_rot if it's a general task refinement
            #     proactive_delay=proactive_delay_between_stages
            # )
            # LOT Phase
            lot_detailed_plan = self.LOT_phase(
                task_desc=current_task_input_for_pipeline, # Or refined_task_prompt_from_rot
                # got_best_idea_content=got_aggregated_idea,
                proactive_delay=proactive_delay_between_stages
            )

            # --- MAS Debate Phase ---
            self.logger.info(f"--- MAS Debate Phase (Cycle {current_prm_cycle_num + 1}) ---")
            # Use the mission_context as the original task for the debate setting
            mas_debate_transcript = self.conduct_mas_debate(
                mission_context=initial_task_description, # The overarching mission
                rot_idea=rot_solution,
                # got_idea=got_aggregated_idea,
                lot_idea=lot_detailed_plan,
                max_rounds=num_debate_rounds
            )
            debate_summary_str = "Debate Record Summary:\n"
            for entry in mas_debate_transcript:
                debate_summary_str += f"  {entry['speaker']}: {str(entry['utterance']).replace(chr(10), ' ')}...\n"

            # Save debate transcript for this specific cycle and CSV item
            if index is not None: # Ensure debate_transcripts directory exists
                # os.makedirs("debate_transcripts/part1", exist_ok=True)
                debate_df_rows = []
                for idx, entry in enumerate(mas_debate_transcript, start=1):
                    utterance_single_line = str(entry["utterance"]).replace("\n", " ").strip()
                    debate_df_rows.append({"Round": idx, "Speaker": entry["speaker"], "Utterance": utterance_single_line})
                debate_df = pd.DataFrame(debate_df_rows, columns=["Round", "Speaker", "Utterance"])
                # Naming convention from [1]
                csv_filename = f"debate_transcripts/part1/debate_transcript_q{index}_cycle{current_prm_cycle_num + 1}.csv"
                try:
                    debate_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
                    self.logger.info(f"Debate transcript for q:{index} cycle:{current_prm_cycle_num+1} saved to {csv_filename}")
                except Exception as e_csv:
                    self.logger.error("DEBATE_CSV_SAVE", f"Failed to save debate transcript to {csv_filename}: {e_csv}")


            if proactive_delay_between_stages > 0:
                 self.logger.info(f"Proactively sleeping for {proactive_delay_between_stages}s after MAS Debate phase (Cycle {current_prm_cycle_num + 1}).")
                 time.sleep(proactive_delay_between_stages)

            # --- Initial Synthesis for this cycle ---
            self.logger.info(f"--- Synthesis Phase (Cycle {current_prm_cycle_num + 1}) ---")
            synthesized_artifact_this_cycle = f"Default synthesis for cycle {current_prm_cycle_num + 1}"
            synthesis_prompt_this_cycle = f"""
            Task Context (Original): {initial_task_description}

            Outputs from Reasoning Modules This Cycle:
            ROT Solution/Idea: {rot_solution}
            LOT Core Idea: {lot_detailed_plan}
            Debate Summary This Cycle: {debate_summary_str}

            Synthesize these elements into a coherent and comprehensive answer/reasoning process for the task.
            Focus on fulfilling the requirements of 'Current Task Input for this Cycle'.
            **Important:** Do not mention the source of information (e.g., ROT, LOT). Integrate them seamlessly.

            At the very end of your response, include this final section:

            ### Synthesized Accurate Answer:
            (Provide a clear, concise answer that integrates the best insights from ROT and LOT. This section must appear last. If it is a multiple-choice question, your answer must exactly match one of the options provided in the question.)
            """
            if not self.synthesis_llm or isinstance(self.synthesis_llm, BaseDummyLLM) or not hasattr(self.synthesis_llm, 'generate'): # This call should now work
                self.logger.warning("MASOrchestrator", f"Synthesis LLM not effectively initialized for cycle {current_prm_cycle_num + 1}. Using placeholder.")
            else:
                try:
                    synthesis_output = call_llm_with_retry(
                        self.synthesis_llm.generate,
                        prompt=synthesis_prompt_this_cycle,
                        logger=self.logger,
                        llm_name=f"Synthesis_LLM_Cycle_{current_prm_cycle_num + 1}"
                    )
                    if synthesis_output and not str(synthesis_output).lower().startswith("error:") and not str(synthesis_output).lower().startswith("llm dummy response"):
                        synthesized_artifact_this_cycle = str(synthesis_output)
                    else: # This call should now work
                        self.logger.warning("MASOrchestrator", f"Synthesis LLM (cycle {current_prm_cycle_num + 1}) did not produce valid output. Using placeholder. Output: {synthesis_output}")
                except Exception as e_synth:
                    self.logger.error("MASOrchestrator", f"Critical error during synthesis LLM call (cycle {current_prm_cycle_num + 1}, after retries): {e_synth}")
                    traceback.print_exc()

            self.logger.answer(f"SYNTHESIZED_ARTIFACT_CYCLE_{current_prm_cycle_num + 1}", synthesized_artifact_this_cycle, is_final=False, detail_level=1)

            # Capture pre-PRM summary for the very first cycle
            if current_prm_cycle_num == 0:
                original_thoughtflow_summary_first_cycle = (
                    f"Initial Task (CSV Index {index}): {initial_task_description}...\n"
                    f"ROT Output (Cycle 1): {str(rot_solution)}...\n"
                    f"LOT Output (Cycle 1): {str(lot_detailed_plan)}...\n"
                    f"Debate Summary (Cycle 1, partial): {debate_summary_str}...\n"
                    f"Synthesized Artifact (Cycle 1, pre-PRM feedback): {synthesized_artifact_this_cycle}...\n"
                    "--- End of First Cycle Pre-PRM Components ---"
                )
            accumulated_thoughtflow_summary_parts.append(
                f"\n--- Cycle {current_prm_cycle_num + 1} Details ---\n"
                f"Task Input This Cycle: {current_task_input_for_pipeline}...\n"
                f"ROT: {str(rot_solution)[:80]}...\nLOT: {str(lot_detailed_plan)}...\n"
                f"Debate (brief): {debate_summary_str}...\n"
                f"Synthesized Artifact This Cycle: {synthesized_artifact_this_cycle}...\n"
            )

            if proactive_delay_between_stages > 0:
                 self.logger.info(f"Proactively sleeping for {proactive_delay_between_stages}s after Synthesis (Cycle {current_prm_cycle_num + 1}).")
                 time.sleep(proactive_delay_between_stages)

            # --- PRM Feedback for the synthesized artifact of THIS cycle ---
            self.logger.info(f"--- PRM Feedback Stage (Cycle {current_prm_cycle_num + 1}) ---")
            # The task description for PRM should be the *original* task, so it evaluates against the true goal.
            prm_score_this_cycle, prm_justification_this_cycle, _ = self._get_prm_feedback_for_reasoning_process(
                task_description=initial_task_description, # Evaluate against original task
                current_reasoning_process=synthesized_artifact_this_cycle,
                iteration_count_label=f"{current_prm_cycle_num + 1}" # Label for logging
            )
            prm_iteration_details_for_excel.append({
                "iteration": current_prm_cycle_num + 1, # This is the main cycle number
                "score": prm_score_this_cycle,
                "justification": prm_justification_this_cycle,
                "artifact_content_before_opt": str(synthesized_artifact_this_cycle) # Storing the full artifact that was evaluated
            })
            accumulated_thoughtflow_summary_parts.append(
                f"PRM Feedback (Cycle {current_prm_cycle_num + 1}): Score={prm_score_this_cycle:.2f}, Justification: {prm_justification_this_cycle}...\n"
            )

            # Update overall best artifact
            if prm_score_this_cycle > best_prm_score_overall:
                best_prm_score_overall = prm_score_this_cycle
                best_artifact_overall = synthesized_artifact_this_cycle
                self.logger.info(f"PRM Cycle {current_prm_cycle_num + 1}: New best artifact found with score {prm_score_this_cycle:.3f}")

            # Early exit if score is high enough (unless it's the last planned cycle)
            if prm_score_this_cycle >= 0.95 and (current_prm_cycle_num < num_prm_iterations - 1):
                self.logger.info(f"PRM Cycle {current_prm_cycle_num + 1}: Score {prm_score_this_cycle:.3f} reached early termination threshold. Stopping PRM cycles.")
                break # Exit the main PRM cycle loop

            # --- Prepare for the NEXT cycle (Optimization) ---
            if current_prm_cycle_num < num_prm_iterations - 1: # If not the last cycle
                self.logger.info(f"--- Optimization Stage for Next Cycle Input (based on Cycle {current_prm_cycle_num + 1} feedback) ---")
                # The optimization prompt should guide the LLM to refine the *original task description* or generate a *new task input* for the next cycle
                # based on the PRM feedback for the *current artifact*.
                # This is a crucial step: what exactly are we optimizing? The artifact itself, or the prompt for the next full pipeline run?
                # The original code [1] created an `optimization_llm_prompt` which was then used as input for ROT/GOT/LOT in the next cycle.
                optimization_prompt_for_next_input = f"""
                Original Task: {initial_task_description}

                The previous reasoning cycle (Cycle {current_prm_cycle_num + 1}) produced the following artifact:
                \"\"\"
                {synthesized_artifact_this_cycle}
                \"\"\"
                This artifact received the following PRM (Process Reward Model) evaluation:
                PRM Score: {prm_score_this_cycle:.2f}
                PRM Improvement Suggestions: {prm_justification_this_cycle}

                Your goal is to refine the *task description or generate a new focused query* for the *next reasoning cycle*.
                This new task input should guide the subsequent ROT and LOT modules to produce an improved final answer that addresses the PRM's suggestions.
                Consider the original task and the PRM feedback.
                What should the *input* to the next full reasoning pipeline be to achieve a better result?
                Output only the refined task description or new focused query for the next cycle.
                Next Cycle's Task Input:
                """
                if not self.iterative_optimizer_llm or isinstance(self.iterative_optimizer_llm, BaseDummyLLM) or not hasattr(self.iterative_optimizer_llm, 'generate'): # This call should now work
                    self.logger.warning("MASOrchestrator", f"Optimizer LLM (for next cycle's input) not effectively initialized. Using PRM justification as directive if possible, else original task for next cycle.")
                    if "improve" in prm_justification_this_cycle.lower() or "focus on" in prm_justification_this_cycle.lower():
                         current_task_input_for_pipeline = f"Refined Task based on PRM: {initial_task_description}. Address these points: {prm_justification_this_cycle}"
                    else:
                         current_task_input_for_pipeline = initial_task_description # Fallback
                else:
                    try:
                        optimized_next_input_candidate = call_llm_with_retry(
                            self.iterative_optimizer_llm.generate,
                            prompt=optimization_prompt_for_next_input,
                            logger=self.logger,
                            llm_name=f"NextCycleInputOptimizer_LLM_Cycle_{current_prm_cycle_num + 1}"
                        )
                        if optimized_next_input_candidate and not str(optimized_next_input_candidate).lower().startswith("error:") and not str(optimized_next_input_candidate).lower().startswith("llm dummy response") and str(optimized_next_input_candidate).strip():
                            current_task_input_for_pipeline = str(optimized_next_input_candidate)
                            self.logger.info(f"Optimized input for next cycle (Cycle {current_prm_cycle_num + 2}) generated (partial): {current_task_input_for_pipeline[:100]}...")
                        else: # This call should now work
                            self.logger.warning("MASOrchestrator", f"Optimizer LLM (for next cycle's input) did not produce valid output. Using PRM justification to augment original task for next cycle. Output: {optimized_next_input_candidate}")
                            current_task_input_for_pipeline = f"Refined Task based on PRM: {initial_task_description}. Previous attempt score {prm_score_this_cycle:.2f}. Address: {prm_justification_this_cycle}"
                    except Exception as e_opt_input:
                        self.logger.error("MASOrchestrator", f"Critical error during next cycle input optimization (after retries): {e_opt_input}. Using PRM justification to augment original task.")
                        traceback.print_exc()
                        current_task_input_for_pipeline = f"Refined Task based on PRM (error fallback): {initial_task_description}. Address: {prm_justification_this_cycle}"
                self.logger.answer(f"OPTIMIZED_INPUT_FOR_NEXT_CYCLE_{current_prm_cycle_num + 2}", current_task_input_for_pipeline, is_final=False, detail_level=2)

                if proactive_delay_between_stages > 0:
                    self.logger.info(f"Proactively sleeping for {proactive_delay_between_stages}s after Optimization Stage (Cycle {current_prm_cycle_num + 1}).")
                    time.sleep(proactive_delay_between_stages)
            else: # This was the last cycle
                self.logger.info(f"PRM Cycle {current_prm_cycle_num + 1}: Final PRM evaluation for this task. No further optimization input generation.")
            self.logger.section_end(f"PRM Cycle {current_prm_cycle_num + 1}/{num_prm_iterations} for CSV Index {index}")


        self.logger.info(f"MASOrchestrator (CSV Index {index}): After {len(prm_iteration_details_for_excel)} PRM cycle(s), selected artifact with PRM score: {best_prm_score_overall:.3f}")

        full_thoughtflow_summary_all_cycles = (
            f"Overall Task (CSV Index {index}): {initial_task_description}...\n"
            f"{original_thoughtflow_summary_first_cycle if original_thoughtflow_summary_first_cycle else 'First cycle summary not available.'}\n"
            f"\n--- Subsequent PRM Cycle Summaries (if any) ---\n"
            + "".join(accumulated_thoughtflow_summary_parts) # This contains details from all cycles
            + "\n--- End of All Cycle Summaries ---"
        )

        self.logger.answer(f"FINAL_SELECTED_OUTPUT_CSV_INDEX_{index}", best_artifact_overall, is_final=True)
        self.logger.section_end(f"Collaborative Task for CSV Index {index}")

        return {
            "synthesized_final_plan": best_artifact_overall, # The best artifact found across all cycles
            "original_thoughtflow_summary_pre_prm": original_thoughtflow_summary_first_cycle, # From cycle 1 before its PRM eval
            "thoughtflow_summary_incl_prm": full_thoughtflow_summary_all_cycles, # Aggregated summary of all cycles
            "prm_iteration_history_details": prm_iteration_details_for_excel # List of dicts, one per cycle's PRM eval
        }


# --- Evaluation Functions ---
def evaluate_with_llm(task_description, thoughtflow_summary, generated_answer, ground_truth_answer, llm_interface, logger, beta_prm):
    R_score_val = None # Default to None
    receval_assessment_text = "RECEVAL assessment placeholder: LLM not called, error, or dummy."
    label_l = 0 # Default label
    similarity_score = 0.0 # Default similarity

    # Handle cases where LLM is not properly initialized or is a dummy
    if not llm_interface or not hasattr(llm_interface, 'generate'): # This call should now work
        logger.warning("evaluate_with_llm", "Evaluation LLM not effectively initialized. Returning default/placeholder evaluation values.")
        R_score_val = 0.1 # Consistent dummy score from [1]
        similarity_score = 0.5 # Consistent dummy score from [1]
        if similarity_score >= SIMILARITY_THRESHOLD_FOR_LABEL_L: label_l = 1
        return R_score_val, receval_assessment_text, label_l, similarity_score

    if isinstance(llm_interface, BaseDummyLLM): # This call should now work
        logger.warning("evaluate_with_llm", "Evaluation LLM is BaseDummyLLM. Returning dummy placeholder values.")
        # Simulate some varied dummy output for testing
        R_score_val = round(random.uniform(0.1, 0.8), 2)
        similarity_score = round(random.uniform(0.3, 0.9), 2)
        if similarity_score >= SIMILARITY_THRESHOLD_FOR_LABEL_L: label_l = 1
        receval_assessment_text = f"Dummy RECEVAL assessment (BaseDummyLLM): Answer was okay-ish (sim score {similarity_score}). Thoughtflow was... present."
        return R_score_val, receval_assessment_text, label_l, similarity_score

    # Proceed with actual LLM calls if LLM is not dummy
    r_score_output_str = None
    try:
        logger.info("Requesting R-score from LLM...")
        r_score_prompt = f"""Task: Evaluate the quality of the generated answer below.
Original Question: {task_description}
Generated Answer:
\"\"\"
{generated_answer}
\"\"\"
Compare this generated answer to an ideal or "gold standard" reference answer.
How good is the generated answer in terms of correctness, completeness, relevance, and clarity for the original question?
Provide a numerical R-score. Positive scores indicate better than average, negative scores are poorer.
This score is conceptually aligned with an implicit Process Reward Model's evaluation, $R = \\beta \\log(\\frac{{\\pi_{{policy}}(answer)}}{{\\pi_{{reference}}(answer)}})$, where beta = {beta_prm}.
Focus on the quality of the final answer itself.
Output only the numerical R-score:""" # Ensure this prompt doesn't get too long
        r_score_output_str = call_llm_with_retry(
            llm_interface.generate,
            prompt=r_score_prompt, # Truncate prompt if too long
            logger=logger,
            llm_name="R_Score_Eval_LLM"
        )
        r_score_output_str_cleaned = str(r_score_output_str).strip()
        # More robust regex for float, also handling potential text around it.
        score_match_r = re.search(r"([-+]?\d*\.?\d+)", r_score_output_str_cleaned)
        if score_match_r:
            R_score_val = float(score_match_r.group(1))
            logger.info(f"R-score from LLM: {R_score_val}")
        else: # This call should now work
            logger.warning(f"evaluate_with_llm: Could not parse R-score from LLM output: '{r_score_output_str_cleaned}'. Setting R-score to 0.0.")
            R_score_val = 0.0 # Default if parsing fails
    except Exception as e:
        logger.error("evaluate_with_llm", f"Error parsing R-score from LLM (after retries): {e}. Output: '{str(r_score_output_str)[:200] if r_score_output_str else 'N/A'}'")
        traceback.print_exc()
        R_score_val = 0.0 # Default on error

    receval_assessment_text_output = None
    try:
        logger.info("Requesting RECEVAL assessment from LLM...")
        # Truncate thoughtflow_summary if it's excessively long
        thoughtflow_summary_for_prompt = str(thoughtflow_summary) if thoughtflow_summary else "Not provided."
        generated_answer_for_prompt = str(generated_answer) if generated_answer else "Not provided."

        receval_prompt = f"""Task: Evaluate the following thoughtflow based on RECEVAL criteria.
Original Question: {task_description}
Thoughtflow Summary (first 5000 chars):
\"\"\"
{thoughtflow_summary_for_prompt}
\"\"\"
Final Answer generated from this thoughtflow (first 3000 chars):
\"\"\"
{generated_answer_for_prompt}
\"\"\"
RECEVAL Criteria:
1.  Clarity & Coherence: Is the reasoning process easy to understand? Are steps logically connected?
2.  Soundness & Validity: Are arguments sound? Are inferences valid?
3.  Sufficiency & Completeness: Does the reasoning cover all necessary aspects of the question? Any omissions?
4.  Relevance: Are all parts of the reasoning relevant to answering the question?
5.  Efficiency: Is the reasoning concise, or does it include unnecessary detours?
Provide a qualitative assessment of the thoughtflow and the final answer based on these criteria:"""
        receval_assessment_text_output = call_llm_with_retry(
            llm_interface.generate,
            prompt=receval_prompt, # Truncate prompt
            logger=logger,
            llm_name="RECEVAL_Eval_LLM"
        )
        if receval_assessment_text_output and \
           not str(receval_assessment_text_output).lower().strip().startswith("error:") and \
           not str(receval_assessment_text_output).lower().strip().startswith("llm dummy response"):
             receval_assessment_text = str(receval_assessment_text_output)
        else:
            receval_assessment_text = f"RECEVAL assessment: LLM returned invalid or dummy response: {str(receval_assessment_text_output)}"
        logger.info(f"RECEVAL assessment from LLM (length): {len(str(receval_assessment_text))}")
    except Exception as e:
        logger.error("evaluate_with_llm", f"Error getting RECEVAL assessment from LLM (after retries): {e}")
        traceback.print_exc()
        receval_assessment_text = f"RECEVAL assessment error (after retries): {e}"

    similarity_output_str = None
    try:
        logger.info("Requesting semantic similarity score from LLM...")
        similarity_prompt = f"""Task: Compare the semantic similarity of the two answers below.
Answer 1 (Generated):
\"\"\"
{generated_answer}
\"\"\"
Answer 2 (Ground Truth):
\"\"\"
{ground_truth_answer}
\"\"\"
Provide a similarity score between 0.0 (completely different) and 1.0 (semantically identical).
Focus on whether they convey the same core meaning and information.
Output only the numerical similarity score:"""
        similarity_output_str = call_llm_with_retry(
            llm_interface.generate,
            prompt=similarity_prompt, # Truncate prompt
            logger=logger,
            llm_name="Similarity_Eval_LLM"
        )
        similarity_output_str_cleaned = str(similarity_output_str).strip()
        sim_score_match = re.search(r"([0-9.]+)", similarity_output_str_cleaned) # Simpler regex for 0.0-1.0
        if sim_score_match:
            similarity_score_candidate = float(sim_score_match.group(1))
            # Clamp score between 0 and 1, as LLMs can sometimes go out of bounds
            similarity_score = max(0.0, min(1.0, similarity_score_candidate))
            logger.info(f"Semantic similarity score from LLM: {similarity_score} (raw: {similarity_score_candidate})")
        else: # This call should now work
            logger.warning(f"evaluate_with_llm: Could not parse similarity score from LLM output: '{similarity_output_str_cleaned}'. Setting to default 0.0.")
            similarity_score = 0.0

        if similarity_score >= SIMILARITY_THRESHOLD_FOR_LABEL_L:
            label_l = 1
        else:
            label_l = 0
        logger.info(f"Label l set to {label_l} based on similarity {similarity_score} (threshold {SIMILARITY_THRESHOLD_FOR_LABEL_L})")
    except Exception as e:
        logger.error("evaluate_with_llm", f"Error parsing similarity score from LLM (after retries): {e}. Output: '{str(similarity_output_str)[:200] if similarity_output_str else 'N/A'}'")
        traceback.print_exc()
        similarity_score = 0.0 # Default on error
        label_l = 0 # Default on error

    return R_score_val, receval_assessment_text, label_l, similarity_score


def calculate_cross_entropy(R_score, label_l, logger):
    if R_score is None: # Check if R_score is None # This call should now work
        logger.warning("calculate_cross_entropy", "R_score is None. Cannot calculate cross-entropy. Returning None.")
        return None
    try:
        R_score_float = float(R_score) # Convert R_score to float
        # Clamp R_score to avoid math domain errors with exp if it's extremely large or small
        # R_score_float_clamped = max(-700, min(700, R_score_float)) # exp(709) is around max float

        # Using log-sum-exp trick for stability with sigmoid
        # log(sigma(R)) = -log(1 + exp(-R)) = -softplus(-R)
        # log(1 - sigma(R)) = log(exp(-R) / (1 + exp(-R))) = -R - log(1 + exp(-R)) = -R - softplus(-R)

        if R_score_float > 36: # exp(-36) is very small, so 1+exp(-R) approx 1, log(sigma(R)) approx 0
            log_sigma_R = 0.0
            log_one_minus_sigma_R = -R_score_float
        elif R_score_float < -36: # exp(R) is very small, so exp(-R) is large. log(sigma(R)) approx R
            log_sigma_R = R_score_float
            log_one_minus_sigma_R = 0.0
        else:
            # More stable calculation around 0
            # sigma_R = 1 / (1 + math.exp(-R_score_float))
            # epsilon = 1e-9 # Clipping to avoid log(0)
            # sigma_R_clipped = max(epsilon, min(sigma_R, 1 - epsilon))
            # log_sigma_R = math.log(sigma_R_clipped)
            # log_one_minus_sigma_R = math.log(1 - sigma_R_clipped)

            # Using softplus directly can be more stable for log(1+exp(x))
            # softplus(x) = log(1+exp(x))
            log_sigma_R = -math.log(1 + math.exp(-R_score_float)) # -softplus(-R)
            log_one_minus_sigma_R = -R_score_float - math.log(1 + math.exp(-R_score_float)) # -R -softplus(-R)


        label_l_int = int(label_l) # Convert label to int
        loss = - (label_l_int * log_sigma_R + (1 - label_l_int) * log_one_minus_sigma_R)
        return loss
    except (ValueError, TypeError) as e:
        logger.error("calculate_cross_entropy", f"Error converting R_score or label_l to numeric: {e}. R_score: {R_score}, label_l: {label_l}")
        return None # Return None if conversion fails
    except OverflowError as e:
        logger.error("calculate_cross_entropy", f"OverflowError during cross-entropy calculation (R_score likely too large/small): {e}. R_score: {R_score_float}")
        return None
    except Exception as e:
        logger.error("calculate_cross_entropy", f"Unexpected error calculating cross-entropy: {e}")
        traceback.print_exc()
        return None # Return None for other errors

def calculate_nlp_metrics(generated_answer, ground_truth_answer, logger):
    metrics = { # Initialize all metrics to 0.0
        "bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "meteor": 0.0,
        "bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0
    }
    str_generated_answer = str(generated_answer) if generated_answer is not None else ""
    str_ground_truth_answer = str(ground_truth_answer) if ground_truth_answer is not None else ""

    if not str_generated_answer.strip() or not str_ground_truth_answer.strip(): # This call should now work
        logger.warning("calculate_nlp_metrics", "Generated answer or ground truth is empty or whitespace. All NLP metrics will be 0.")
        return metrics

    try:
        # Ensure Punkt is downloaded (already handled globally, but good practice if function is isolated)
        ref_tokens = [word_tokenize(str_ground_truth_answer.lower())] if str_ground_truth_answer.strip() else [[]]
        gen_tokens = word_tokenize(str_generated_answer.lower()) if str_generated_answer.strip() else []
    except LookupError as le:
        logger.error("calculate_nlp_metrics", f"Tokenization failed due to NLTK resource 'punkt' not found: {le}. NLP metrics will be 0.")
        return metrics # Return zeros if tokenization fails
    except Exception as e:
        logger.error("calculate_nlp_metrics", f"Tokenization failed with unexpected error: {e}. NLP metrics will be 0.")
        traceback.print_exc()
        return metrics

    # If tokenization results in empty lists for non-empty strings (highly unlikely with standard text)
    if not gen_tokens and str_generated_answer.strip(): # This call should now work
        logger.warning("calculate_nlp_metrics", "Generated answer non-empty but tokenized to empty list. NLP metrics might be affected.")
    if not ref_tokens[0] and str_ground_truth_answer.strip(): # This call should now work
        logger.warning("calculate_nlp_metrics", "Ground truth non-empty but tokenized to empty list. NLP metrics might be affected.")


    # BLEU Score
    if gen_tokens and ref_tokens[0]: # Check if both are non-empty after tokenization
        try:
            chencherry = SmoothingFunction()
            metrics["bleu"] = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=chencherry.method1)
        except ZeroDivisionError: # Handles cases like very short generated text # This call should now work
            logger.warning("calculate_nlp_metrics", "ZeroDivisionError during BLEU calculation (often due to short generated answer). BLEU set to 0.0.")
            metrics["bleu"] = 0.0
        except Exception as e_bleu:
            logger.error("calculate_nlp_metrics", f"Error calculating BLEU: {e_bleu}")
            # metrics["bleu"] remains 0.0
    else: # This call should now work
        logger.warning("calculate_nlp_metrics", "Skipping BLEU calculation due to empty generated or reference tokens.")


    # ROUGE Scores
    # ROUGE scorer can handle empty strings, but it's better to be explicit.
    # If gen_tokens is empty, ROUGE scores will be 0.
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # Use original strings for ROUGE as it handles its own tokenization/preprocessing
        rouge_scores_dict = scorer.score(str_ground_truth_answer, str_generated_answer)
        metrics["rouge1"] = rouge_scores_dict['rouge1'].fmeasure
        metrics["rouge2"] = rouge_scores_dict['rouge2'].fmeasure
        metrics["rougeL"] = rouge_scores_dict['rougeL'].fmeasure
    except Exception as e_rouge:
        logger.error("calculate_nlp_metrics", f"Error calculating ROUGE: {e_rouge}")
        # metrics for ROUGE remain 0.0

    # METEOR Score
    if gen_tokens and ref_tokens[0]: # Check if both are non-empty
        try:
            # Ensure wordnet and omw-1.4 are available (handled globally)
            metrics["meteor"] = meteor_score(ref_tokens, gen_tokens)
        except LookupError as le_meteor: # Specifically for wordnet/omw-1.4 not found
            logger.error("calculate_nlp_metrics", f"Error calculating METEOR due to NLTK resource (e.g., 'wordnet', 'omw-1.4') not found: {le_meteor}. METEOR set to 0.0.")
            # metrics["meteor"] remains 0.0
        except Exception as e_meteor:
            logger.error("calculate_nlp_metrics", f"Error calculating METEOR: {e_meteor}")
            # metrics["meteor"] remains 0.0
    else: # This call should now work
        logger.warning("calculate_nlp_metrics", "Skipping METEOR calculation due to empty generated or reference tokens.")


    # BERTScore
    try:
        global torch # Check if torch was successfully imported
        if torch:
            # BERTScore expects lists of strings.
            # Handle cases where one might be valid and other is " " to avoid errors with some models.
            cands = [str_generated_answer.strip() if str_generated_answer.strip() else " "] # Use a space if empty
            refs = [str_ground_truth_answer.strip() if str_ground_truth_answer.strip() else " "]

            P, R, F1 = bert_score_calc(
                cands, refs,
                lang="en", verbose=False, model_type='bert-base-uncased', # Default model
                # device='cuda' if torch.cuda.is_available() else 'cpu' # Optional: specify device
            )
            # Ensure results are scalar and not NaN before item()
            metrics["bert_precision"] = P.mean().item() if P is not None and not torch.isnan(P.mean()).any() else 0.0
            metrics["bert_recall"] = R.mean().item() if R is not None and not torch.isnan(R.mean()).any() else 0.0
            metrics["bert_f1"] = F1.mean().item() if F1 is not None and not torch.isnan(F1.mean()).any() else 0.0
        else: # PyTorch not available # This call should now work
            logger.warning("calculate_nlp_metrics", "PyTorch not available. Skipping BERTScore calculation (all BERT scores will be 0.0).")
            # metrics for BERT remain 0.0
    except Exception as e_bert:
        logger.error("calculate_nlp_metrics", f"Error calculating BERTScore: {e_bert}")
        traceback.print_exc()
        # metrics for BERT remain 0.0
    return metrics

# --- Main Processing Logic ---
def main():
    logger = TerminalLogger(verbose=True)
    logger.section_start("Main Evaluation Flow")

    global IMPORTS_SUCCESSFUL, GEMINI_API_KEY, torch # PyTorch import status
    GOOGLE_API_KEY = 'AIzaSyA-4smIZ6201IBDmiPgYRwQYmtQtYEzd_I' # Example if needed elsewhere
    GEMINI_API_KEY = 'AIzaSyA-4smIZ6201IBDmiPgYRwQYmtQtYEzd_I' # Loaded from .env
    logger.info(f"GEMINI_API_KEY check: {'Loaded' if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE' else 'Not loaded or placeholder'}")

    try:
        import torch as pytorch_module
        torch = pytorch_module
        logger.info(f"PyTorch successfully imported. Version: {torch.__version__}. BERTScore should work.")
        # logger.info(f"CUDA available for PyTorch: {torch.cuda.is_available()}")
    except ImportError: # This call should now work
        logger.warning("main", "Failed to import PyTorch (torch). BERTScore calculation will be skipped or result in 0s. Please install PyTorch if BERTScore is needed.")
        torch = None # Explicitly set to None

    if not IMPORTS_SUCCESSFUL: # This call should now work
        logger.warning("main", "One or more core modules (GOT, LOT, ROT) failed to import. Functionality will rely on dummy classes.")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE": # This call should now work
        logger.warning("main", f"GEMINI_API_KEY is invalid or a placeholder ('{str(GEMINI_API_KEY)[:10]}...'). LLM calls will likely use dummy classes or fail if real modules are loaded but key is missing/invalid.")

    orchestrator = MASOrchestrator(api_key=GEMINI_API_KEY, logger=logger)
    # Use a shared LLM instance for evaluation, typically from the orchestrator
    evaluation_llm_interface = orchestrator.synthesis_llm # As in [1]

    if not evaluation_llm_interface or not hasattr(evaluation_llm_interface, 'generate'): # This call should now work
         logger.warning("main", "Evaluation LLM (from orchestrator.synthesis_llm) not effectively initialized. LLM-based evaluation results will be placeholders or based on dummy logic.")
    elif isinstance(evaluation_llm_interface, BaseDummyLLM): # This call should now work
         logger.warning("main", "Evaluation LLM (from orchestrator.synthesis_llm) is BaseDummyLLM. LLM-based evaluation results will be based on dummy logic.")

    # Ensure dataset and result directories exist
    dataset_base_dir = r"C:\Users\user\Documents\GitHub\MAS-PRM\dataset"
    results_base_dir = "result/part1" # From [1]
    # os.makedirs(dataset_base_dir, exist_ok=True)
    # os.makedirs(results_base_dir, exist_ok=True)
    # Ensure debate transcripts directory from [1] exists
    # os.makedirs("debate_transcripts/part1", exist_ok=True)

    csv_file_path = os.path.join(dataset_base_dir, "All_of_dataset_part1.csv") # Path from [1]
    logger.info(f"Attempting to load CSV data from: {os.path.abspath(csv_file_path)}")

    if not os.path.exists(csv_file_path):
        logger.error("main", f"CSV file not found: {csv_file_path}")
        try:
            logger.info(f"Attempting to create a sample {csv_file_path} file at: {os.path.abspath(csv_file_path)}")
            # More comprehensive sample including ROT demo columns
            sample_df_columns = ['instruction', 'context', 'response', 'rot_demonstration_input', 'rot_demonstration_output']
            sample_df_data = [
                ["What is the capital of France?", "France is a country in Europe.", "The capital of France is Paris.", "Old query: Capital of Germany", "Old answer: Berlin"],
                ["Explain the theory of relativity in simple terms.", "", "The theory of relativity, developed by Albert Einstein...", "Old query: What is gravity?", "Old answer: Gravity is a force..."]
            ]
            sample_df = pd.DataFrame(sample_df_data, columns=sample_df_columns)
            sample_df.to_csv(csv_file_path, index=False, encoding='utf-8')
            logger.info(f"Sample {csv_file_path} created with UTF-8 encoding. Please populate it with your data and rerun.")
        except Exception as e_create:
            logger.error("main", f"Failed to create sample {csv_file_path}: {e_create}")
            traceback.print_exc()
        logger.section_end("Main Evaluation Flow")
        return

    try:
        # Attempt to read with UTF-8 first
        try:
            dataset_df = pd.read_csv(csv_file_path, encoding='utf-8')
            logger.info(f"Successfully loaded {len(dataset_df)} records from '{csv_file_path}' using UTF-8 encoding.")
        except UnicodeDecodeError: # This call should now work
            logger.warning("main", f"Failed to load '{csv_file_path}' with UTF-8 encoding. Attempting with 'latin1' encoding...")
            dataset_df = pd.read_csv(csv_file_path, encoding='latin1') # Fallback encoding
            logger.info(f"Successfully loaded {len(dataset_df)} records from '{csv_file_path}' using latin1 encoding.")

        required_columns = ['instruction', 'response'] # Core columns
        missing_cols = [col for col in required_columns if col not in dataset_df.columns]
        if missing_cols:
            logger.error("main", f"CSV file must contain the following columns: {', '.join(required_columns)}. Missing: {', '.join(missing_cols)}. Please correct the CSV.")
            logger.section_end("Main Evaluation Flow")
            return

        # Handle optional columns gracefully
        if "context" not in dataset_df.columns: # This call should now work
            logger.warning("main", "CSV file does not contain 'context' column. Context will be considered empty for all rows.")
            dataset_df["context"] = "" # Add empty context column if missing
        if "rot_demonstration_input" not in dataset_df.columns:
            logger.info("CSV file does not contain 'rot_demonstration_input' column. ROT demonstrations will use defaults or be skipped if not applicable internally.")
            dataset_df["rot_demonstration_input"] = "" # Add empty if missing
        if "rot_demonstration_output" not in dataset_df.columns:
            logger.info("CSV file does not contain 'rot_demonstration_output' column. ROT demonstrations will use defaults or be skipped.")
            dataset_df["rot_demonstration_output"] = "" # Add empty if missing

    except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
        logger.error("main", f"Critical: CSV file not found at {csv_file_path} after attempting to load.")
        logger.section_end("Main Evaluation Flow")
        return
    except Exception as e_load_csv:
        logger.error("main", f"Error loading or processing CSV file '{csv_file_path}': {e_load_csv}")
        traceback.print_exc()
        logger.section_end("Main Evaluation Flow")
        return

    all_results_for_excel = [] # Changed variable name for clarity
    num_processed = 0
    default_max_items = len(dataset_df)

    max_items_to_process_str = input(f"Dataset has {default_max_items} items. How many to process? (Enter number, or press Enter for all [{default_max_items}]): ")
    try:
        max_items_to_process = int(max_items_to_process_str) if max_items_to_process_str.strip() else default_max_items
        if not (0 < max_items_to_process <= default_max_items): # Check if within valid range # This call should now work
            logger.warning("main", f"Invalid number of items ({max_items_to_process}). Must be between 1 and {default_max_items}. Defaulting to all {default_max_items} items.")
            max_items_to_process = default_max_items
    except ValueError: # This call should now work
        logger.warning("main", f"Invalid input for number of items. Defaulting to all {default_max_items} items.")
        max_items_to_process = default_max_items

    logger.info(f"Will process a maximum of {max_items_to_process} items.")
    main_loop_item_delay_seconds = 1 # Delay between processing each CSV item, can be adjusted
    num_prm_cycles_per_item = 3 # Number of PRM-guided cycles for orchestrator, from [1]
    num_debate_rounds_per_cycle = 3 # From [1]

    for index, row in dataset_df.iterrows():
        if num_processed >= max_items_to_process:
            logger.info(f"Reached processing limit of {max_items_to_process} items.")
            break
        # if index<=49:
        #     print("skip")
        #     continue
        # logger.info(f"現在是第{index} round")
        num_processed += 1
        logger.section_start(f"Processing Item {num_processed}/{max_items_to_process} (CSV Index {index})")

        instruction = str(row["instruction"])
        # Ensure context is string, handle NaN or None from CSV
        context = str(row["context"]) if "context" in row and pd.notna(row["context"]) else ""
        ground_truth_answer = str(row["response"])

        if context and context.strip(): # Only add context if it's non-empty
            question_for_orchestrator = f"Instruction: {instruction}\n\nContext: {context}"
        else:
            question_for_orchestrator = f"Instruction: {instruction}"

        logger.info(f"Input for Orchestrator (Instruction & Context combined, CSV Index {index}): {question_for_orchestrator[:250]}...")
        logger.info(f"Ground Truth Answer (Response, CSV Index {index}): {ground_truth_answer[:150]}...")

        rot_demonstrations_for_this_item = None
        # Check for ROT demo data, ensuring it's valid strings
        rot_demo_input = str(row.get("rot_demonstration_input", "")).strip()
        rot_demo_output = str(row.get("rot_demonstration_output", "")).strip()

        if rot_demo_input and rot_demo_output:
            rot_demonstrations_for_this_item = [(rot_demo_input, rot_demo_output)]
            logger.info(f"Using ROT demonstration from CSV (Index {index}): Input='{rot_demo_input[:50]}...', Output='{rot_demo_output[:50]}...'")
        else:
            logger.info(f"No valid ROT demonstration data in CSV for Index {index}. Orchestrator will use defaults if any.")


        orchestrator_outputs = orchestrator.run_collaborative_task(
            initial_task_description=question_for_orchestrator,
            rot_demonstrations=rot_demonstrations_for_this_item,
            problem_instance_for_rot_final_solve=None, # Or pass if available in CSV
            num_debate_rounds=num_debate_rounds_per_cycle,
            num_prm_iterations=num_prm_cycles_per_item, # This controls main PRM cycles in run_collaborative_task
            index=index # Pass CSV index for logging/filename generation
        )
        # This is the overall best answer selected across all PRM cycles by the orchestrator
        generated_answer_overall_best = orchestrator_outputs.get("synthesized_final_plan", "Error: Failed to generate final plan.")
        # This is the summary from the *first* cycle, before any PRM-guided iteration based on previous cycle's feedback
        original_thoughtflow_summary_first_cycle = orchestrator_outputs.get("original_thoughtflow_summary_pre_prm", "N/A: First cycle summary missing.")
        # This is a string that accumulates all PRM iteration justifications and scores from within run_collaborative_task's perspective
        thoughtflow_summary_all_cycles_details_str = orchestrator_outputs.get("thoughtflow_summary_incl_prm", "N/A: Full thoughtflow summary missing.")
        # This list contains one dict per main PRM cycle in run_collaborative_task
        prm_cycle_history_list_from_orchestrator = orchestrator_outputs.get("prm_iteration_history_details", [])


        logger.info(f"Overall Best Generated Answer (CSV Index {index}, post-PRM cycles, partial): {str(generated_answer_overall_best)[:150]}...")
        # logger.info(f"Original Thoughtflow Summary (CSV Index {index}, first cycle, partial): {str(original_thoughtflow_summary_first_cycle)[:250]}...")
        # logger.info(f"Full Thoughtflow Summary (CSV Index {index}, all cycles, partial): {str(thoughtflow_summary_all_cycles_details_str)[:250]}...")

        # Evaluate the overall best answer against ground truth
        R_score_overall, receval_assessment_overall, label_l_overall, llm_similarity_overall = evaluate_with_llm(
            task_description=question_for_orchestrator, # Original task
            thoughtflow_summary=thoughtflow_summary_all_cycles_details_str, # Use the most comprehensive summary
            generated_answer=generated_answer_overall_best,
            ground_truth_answer=ground_truth_answer,
            llm_interface=evaluation_llm_interface,
            logger=logger,
            beta_prm=BETA_PRM
        )
        logger.info(f"LLM Evaluation for Overall Best Answer (CSV Index {index}) - R-score: {R_score_overall}, Label l: {label_l_overall}, Similarity: {llm_similarity_overall}, RECEVAL (len): {len(str(receval_assessment_overall))}")

        ce_loss_overall = calculate_cross_entropy(R_score_overall, label_l_overall, logger)
        logger.info(f"Cross-Entropy Loss for Overall Best Answer (CSV Index {index}): {ce_loss_overall}")

        nlp_metrics_overall = calculate_nlp_metrics(generated_answer_overall_best, ground_truth_answer, logger)
        logger.info(f"NLP Metrics for Overall Best Answer (CSV Index {index}): {nlp_metrics_overall}")

        # Common data that will be part of each Excel row for this CSV item
        common_data_for_excel_rows = {
            "csv_index": index,
            "input_instruction": instruction,
            "input_context": context,
            "combined_input_question": question_for_orchestrator, # Full input to orchestrator
            "ground_truth_answer": ground_truth_answer,
            "overall_best_generated_answer_across_cycles": generated_answer_overall_best,
            "overall_best_R_score": R_score_overall,
            "overall_best_label_l": label_l_overall,
            "overall_best_llm_similarity": llm_similarity_overall,
            "overall_best_cross_entropy_loss": ce_loss_overall,
            "overall_receval_assessment_of_full_process": receval_assessment_overall,
            "first_cycle_thoughtflow_summary": original_thoughtflow_summary_first_cycle, # Summary of the first pipeline run
            "all_cycles_combined_thoughtflow_summary": thoughtflow_summary_all_cycles_details_str, # String summary of all cycles
            **{f"overall_best_nlp_{k}": v for k, v in nlp_metrics_overall.items()} # NLP for the best answer
        }

        if not prm_cycle_history_list_from_orchestrator: # This call should now work
            logger.warning(f"CSV Index {index}: No PRM cycle history returned from orchestrator. Storing only overall results as a single row.")
            entry_row = {**common_data_for_excel_rows}
            entry_row["prm_cycle_number"] = 0 # Indicates no specific cycle data
            entry_row["prm_artifact_evaluated_this_cycle"] = generated_answer_overall_best # The only artifact known
            entry_row["prm_score_this_cycle"] = R_score_overall if R_score_overall is not None else "N/A" # Use overall R as a proxy if no cycle score
            entry_row["prm_justification_this_cycle"] = "No specific PRM cycle data from orchestrator."
            all_results_for_excel.append(entry_row)
        else:
            # Create a new row in Excel for each PRM cycle's evaluation details
            for prm_cycle_detail_dict in prm_cycle_history_list_from_orchestrator:
                entry_row = {**common_data_for_excel_rows} # Start with all common data for this CSV item

                # Add PRM-specific data for this particular cycle
                entry_row["prm_cycle_number"] = prm_cycle_detail_dict.get("iteration", "ERR_ITER_FIELD_MISSING")
                entry_row["prm_artifact_evaluated_this_cycle"] = prm_cycle_detail_dict.get("artifact_content_before_opt", "ERR_ARTIFACT_FIELD_MISSING")
                entry_row["prm_score_this_cycle"] = prm_cycle_detail_dict.get("score", "ERR_SCORE_FIELD_MISSING")
                entry_row["prm_justification_this_cycle"] = prm_cycle_detail_dict.get("justification", "ERR_JUSTIFICATION_FIELD_MISSING")
                all_results_for_excel.append(entry_row)

        logger.section_end(f"Finished Item {num_processed}/{max_items_to_process} (CSV Index {index})")
        if num_processed < max_items_to_process and main_loop_item_delay_seconds > 0:
            logger.info(f"Delaying for {main_loop_item_delay_seconds}s before next CSV item...")
            time.sleep(main_loop_item_delay_seconds)


    logger.section_start("Overall Results Summary & Saving")
    if not all_results_for_excel:
        logger.info("No items were processed or no results were generated to save.")
    else:
        results_df = pd.DataFrame(all_results_for_excel)
        # Ensure results_base_dir exists (already done, but good check)
        # os.makedirs(results_base_dir, exist_ok=True)
        # Dynamic filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_excel_filename = os.path.join(results_base_dir, f"evaluation_results_remove_lot.xlsx")

        try:
            results_df.to_excel(output_excel_filename, index=False, engine='openpyxl') # Requires openpyxl
            logger.info(f"Detailed evaluation results for each PRM cycle saved to: {os.path.abspath(output_excel_filename)}")
        except Exception as e_excel:
            logger.error("main", f"Error saving results to Excel file '{output_excel_filename}': {e_excel}. Please ensure 'openpyxl' is installed ('pip install openpyxl').")
            traceback.print_exc()

        # Define numeric columns for averaging. Prefixes help distinguish overall vs. cycle-specific.
        numeric_cols_for_avg = [
            "overall_best_R_score", "overall_best_label_l", "overall_best_llm_similarity",
            "overall_best_cross_entropy_loss",
            "overall_best_nlp_bleu", "overall_best_nlp_rouge1", "overall_best_nlp_rouge2",
            "overall_best_nlp_rougeL", "overall_best_nlp_meteor",
            "overall_best_nlp_bert_precision", "overall_best_nlp_bert_recall", "overall_best_nlp_bert_f1",
            "prm_score_this_cycle" # This will average PRM scores from all cycles across all items
        ]
        avg_scores_summary = {}
        logger.info("Average scores (Note: 'overall_best_*' metrics are repeated per PRM cycle row for a given CSV item; 'prm_score_this_cycle' is unique per row):")

        for col_name in numeric_cols_for_avg:
            if col_name in results_df.columns:
                # Convert column to numeric, coercing errors (non-numeric become NaN)
                numeric_series = pd.to_numeric(results_df[col_name], errors='coerce')
                if not numeric_series.empty and numeric_series.notna().any(): # Check if any valid numbers exist
                    avg_value = numeric_series.mean() # mean() skips NaNs by default
                    avg_scores_summary[f"avg_{col_name}"] = avg_value
                    logger.info(f"  {avg_value:.4f} : {col_name} (avg over {numeric_series.notna().sum()} valid entries)")
                else:
                    avg_scores_summary[f"avg_{col_name}"] = "N/A (no valid numeric data or all NaN)"
                    logger.info(f"  N/A (no valid numeric data or all NaN) : {col_name}")
            else:
                avg_scores_summary[f"avg_{col_name}"] = f"N/A (column '{col_name}' not found in results)"
                logger.info(f"  N/A (column '{col_name}' not found) : {col_name}")


    logger.section_end("Overall Results Summary & Saving")
    logger.section_end("Main Evaluation Flow")

# --- Discord Notification (from [1]) ---
def send_discord_notification(content, webhook_url):
    try:
        import requests # Keep import local if only used here
        data = {"content": content}
        response = requests.post(webhook_url, json=data)
        if response.status_code == 204 or response.status_code == 200 : # 200 can also be success for some webhooks
            print(f"✅ Successfully sent Discord notification: {content}")
        else:
            print(f"❌ Failed to send Discord notification. Status: {response.status_code} - {response.text}")
    except ImportError:
        print("[INFO] 'requests' library not found. Skipping Discord notification. Install with 'pip install requests'.")
    except Exception as e_discord:
        print(f"[ERROR] Exception during Discord notification: {e_discord}")


if __name__ == "__main__":
    main()

    # Discord notification at the end of script execution
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL") # Try to get from environment
    if not webhook_url:
        # Hardcoded fallback if not in env (less secure, better to use .env)
        webhook_url = "https://discord.com/api/webhooks/1374078975202295929/n-YZFYepU_L7Ar3JzjxGVAxd1OYiNoTsFumMNcJI-idlldxPcPDJDc3zZ5ckCkQnYyD2" # From [1] # This call should now work
        print("[WARNING] Discord webhook URL not found in environment (DISCORD_WEBHOOK_URL). Using hardcoded URL as fallback.")

    if webhook_url and "YOUR_WEBHOOK_URL_HERE" not in webhook_url : # Basic check if it's a placeholder
        script_end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        send_discord_notification(f"✅ Python script 'main_evaluator.py' completed at {script_end_time}.", webhook_url)
    else:
        print("[INFO] Discord webhook URL is missing or a placeholder. Skipping final notification.")
