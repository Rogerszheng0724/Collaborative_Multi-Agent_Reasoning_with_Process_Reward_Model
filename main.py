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

    def warning(self, stage, message):
        print(f"[WARNING][{stage}] {message}")

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

                logger.warning(f"{llm_name}_RETRY", f"Rate limit error encountered. Retrying in {wait_time:.2f} seconds (Attempt {retries}/{max_retries})... Error: {e}")
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
        return f"LLM dummy response (BaseDummyLLM for {self.model_name}): {prompt[:50]}..."

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
class DummyGraphOfThoughts:
    def __init__(self, llm_interface, logger=None):
        self.llm = llm_interface
        if logger and hasattr(logger, 'info'):
            self.logger = logger
        else:
            self.logger = TerminalLogger(verbose=False)
        self.logger.info("DummyGraphOfThoughts initialized.")
    def generate_and_evaluate_thoughts(self, task_description, num_thoughts):
        self.logger.info(f"Dummy GOT: Generating {num_thoughts} thoughts for '{task_description[:20]}...'")
        class DummyThought:
            def __init__(self, content, score=0.0, prm_justification="Dummy justification"):
                self.content = content
                self.score = score
                self.prm_justification = prm_justification
        return [DummyThought(f"GOT dummy thought for {task_description[:20]}")] * num_thoughts
    def print_graph(self): self.logger.info("Dummy GOT: Printing graph (empty).")
    def rank_thoughts(self): return []

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
        return f"LOT dummy plan for {str(initial_input)[:20]}"

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
        self.logger.info(f"Dummy ROT: PGRR warmup for task '{main_task_description_for_prm[:20]}...'")
        return "ROT dummy PGRR output"
    def cognitive_preference_manager(self, original_task_prompt_text, llm_taste_prompt_text, main_task_description_for_prm):
        self.logger.info(f"Dummy ROT: CPM for task '{main_task_description_for_prm[:20]}...'")
        return "ROT dummy CPM output"
    def solve_task_with_final_prompt(self, prompt, instance):
        self.logger.info(f"Dummy ROT: Solving '{str(instance)[:20]}...' with prompt '{str(prompt)[:20]}...'")
        return f"ROT dummy solution for {str(instance)[:20]}"

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

        # full_prompt = f"You are {self.name}, an expert.\nCurrent discussion context: {context_summary}\nYour specific task: {prompt}\nPlease provide your response. Ensure it is clear, correct, structured, and directly addresses the task."
        full_prompt = f"You are {self.name}, an expert.\nYour specific task: {prompt}\nPlease provide your response. Ensure it is clear, correct, structured, and directly addresses the task."
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
            self.logger.warning("MASOrchestrator", "API key is invalid or a placeholder. MAS may use dummy LLMs if real modules were imported.")

        try:
            if IMPORTS_SUCCESSFUL and GotGeminiLLM_cls == _RealGotGeminiLLM:
                self.logger.info("Instantiating real GOT.GeminiLLM (from GOT.py, without logger).")
                self.got_llm = GotGeminiLLM_cls(api_key=self.api_key)
            else:
                self.logger.info(f"Instantiating GotGeminiLLM_cls as {GotGeminiLLM_cls.__name__} (with logger).")
                self.got_llm = GotGeminiLLM_cls(api_key=self.api_key, logger=self.logger)
            self.got_system = GraphOfThoughts_cls(llm_interface=self.got_llm, logger=self.logger)
            self.logger.info("GraphOfThoughts (GOT) system initialized with its LLM.")

            self.lot_llm = LotGeminiLLMInterface_cls(api_key=self.api_key, logger=self.logger)
            self.lot_system = LayerOfThoughts_cls(llm_interface=self.lot_llm, logger=self.logger, prm_evaluator_llm=self.lot_llm)
            self.logger.info("LayerOfThoughts (LOT) system initialized.")

            self.rot_llm = RotGeminiLLMInterface_cls(api_key=self.api_key, logger=self.logger)
            self.rot_embedder = RotGeminiEmbeddingInterface_cls(api_key=self.api_key, logger=self.logger)
            self.rot_system = ReversalOfThought_cls(
                llm_interface=self.rot_llm,
                embedding_model_interface=self.rot_embedder,
                logger=self.logger
            )
            self.logger.info("ReversalOfThought (ROT) system initialized.")

            if self.got_llm and hasattr(self.got_llm, 'generate') and not isinstance(self.got_llm, BaseDummyLLM):
                self.debate_llm = self.got_llm
                self.synthesis_llm = self.got_llm
                self.iterative_optimizer_llm = self.got_llm
                self.logger.info("Debate, Synthesis, Optimizer LLMs set to GOT LLM.")
            elif self.rot_llm and hasattr(self.rot_llm, 'generate') and not isinstance(self.rot_llm, BaseDummyLLM):
                self.debate_llm = self.rot_llm
                self.synthesis_llm = self.rot_llm
                self.iterative_optimizer_llm = self.rot_llm
                self.logger.info("Debate, Synthesis, Optimizer LLMs set to ROT LLM.")
            elif self.lot_llm and hasattr(self.lot_llm, 'generate') and not isinstance(self.lot_llm, BaseDummyLLM):
                self.debate_llm = self.lot_llm
                self.synthesis_llm = self.lot_llm
                self.iterative_optimizer_llm = self.lot_llm
                self.logger.info("Debate, Synthesis, Optimizer LLMs set to LOT LLM.")
            else:
                self.logger.warning("MASOrchestrator", "No suitable real LLM found from initialized systems for Debate/Synthesis/Optimizer. Using BaseDummyLLM.")
                self.debate_llm = BaseDummyLLM(api_key=self.api_key, model_name="debate_dummy_fallback", logger=self.logger)
                self.synthesis_llm = BaseDummyLLM(api_key=self.api_key, model_name="synthesis_dummy_fallback", logger=self.logger)
                self.iterative_optimizer_llm = BaseDummyLLM(api_key=self.api_key, model_name="optimizer_dummy_fallback", logger=self.logger)

        except Exception as e:
            self.logger.error("MASOrchestrator_Init", f"Critical error during subsystem initialization: {e}")
            self.logger.error("MASOrchestrator_Init", "Detailed error traceback:")
            traceback.print_exc()
            self.logger.warning("MASOrchestrator_Init", "All components will fall back to dummy implementations due to initialization error.")

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


    def conduct_mas_debate(self, mission_context,rot_idea, got_idea, lot_idea, max_rounds):
        self.logger.section_start(f"MAS Style Debate (Targeting {max_rounds} Rounds)")
        proactive_delay_between_turns = 5


        debate_transcript = []
        discussion_context_summary = f"Mission Context:\n{mission_context}\n"
        discussion_context_summary += f"Initial Core Idea from GOT:\n{rot_idea}\n"
        discussion_context_summary += f"Initial Core Idea from GOT:\n{got_idea}\n"
        discussion_context_summary += f"Initial Detailed Plan from LOT:\n{lot_idea}\n"
        discussion_context_summary += "The debate will now commence.\n"

        if not self.debate_llm or isinstance(self.debate_llm, BaseDummyLLM) or not hasattr(self.debate_llm, 'generate'):
            self.logger.warning("MAS_DEBATE", "Debate LLM not effectively initialized. Using simulated debate.")
            debate_transcript.append({"speaker": "Moderator", "utterance": "Simulated debate starting."})
            
            rot_sim_statement = "Simulated ROT statement."
            got_sim_statement = "Simulated GOT statement."
            lot_sim_statement = "Simulated LOT statement."
            critic_sim_statement = "Simulated Critic statement."
            
            if hasattr(self.rot_system, 'llm') and isinstance(self.rot_system.llm, BaseDummyLLM):
                 rot_sim_statement = self.rot_system.llm.generate("ROT opening statement prompt")
            if hasattr(self.got_system, 'llm') and isinstance(self.got_system.llm, BaseDummyLLM):
                 got_sim_statement = self.got_system.llm.generate("GOT opening statement prompt")
            if hasattr(self.lot_system, 'llm') and isinstance(self.lot_system.llm, BaseDummyLLM):
                 lot_sim_statement = self.lot_system.llm.generate("LOT opening statement prompt")
            
            temp_critic_llm = self.debate_llm if (self.debate_llm and hasattr(self.debate_llm, 'generate') and not isinstance(self.debate_llm, BaseDummyLLM)) else BaseDummyLLM(logger=self.logger)
            critic_sim_statement = temp_critic_llm.generate("Critic statement prompt")

            debate_transcript.append({"speaker": "ROT_Representative", "utterance": rot_sim_statement})
            debate_transcript.append({"speaker": "GOT_Representative", "utterance": got_sim_statement})
            debate_transcript.append({"speaker": "LOT_Representative", "utterance": lot_sim_statement})
            debate_transcript.append({"speaker": "Critical_Analyst", "utterance": critic_sim_statement})
            self.logger.section_end(f"MAS Style Debate (Simulated)")
            return debate_transcript

        rot_agent = DebateAgent("ROT_Representative", self.debate_llm, self.logger)
        got_agent = DebateAgent("GOT_Representative", self.debate_llm, self.logger)
        lot_agent = DebateAgent("LOT_Representative", self.debate_llm, self.logger)
        critic_agent = DebateAgent("Critical_Analyst", self.debate_llm, self.logger)

        opening_statement = f"Debate Topic: In-depth discussion based on the following mission context, ROT idea, GOT idea, and LOT idea.\n{discussion_context_summary}"
        debate_transcript.append({"speaker": "Moderator", "utterance": opening_statement})

        current_round = 0
        rot_statement = "ROT idea placeholder (if not generated)."
        got_statement = "GOT idea placeholder (if not generated)."
        lot_statement = "LOT idea placeholder (if not generated)."

        if current_round < max_rounds:
            current_round += 1
            self.logger.info(f"Debate Round {current_round}: ROT Representative")
            # prompt_rot = f"As the ROT Representative, elaborate on your core idea: '{str(rot_idea)[:150]}...', considering the mission: '{str(mission_context)[:100]}...'. Explain its strengths and how it addresses the core problem."
            prompt_rot = f"""
                As the ROT(Reversal Of Thought) Representative, elaborate on your core idea: '{str(rot_idea)}', considering the mission: '{str(mission_context)}'. 
                Explain how your idea addresses the core problem and highlight its key strengths. 
                Then, critically evaluate the GOT(Graph Of Thoughts) idea: '{str(got_idea)}' and the LOT(Layer Of Thoughts)  idea: '{str(lot_idea)}'. 
                Discuss their potential weaknesses, overlooked aspects, or limitations compared to your own idea, and explain why your approach is preferable.
                """
            rot_statement = rot_agent.speak(prompt_rot, discussion_context_summary)
            debate_transcript.append({"speaker": rot_agent.name, "utterance": rot_statement})
            discussion_context_summary += f"\nRound {current_round} - {rot_agent.name}:\n{rot_statement}\n"
            self.logger.info(f"Debate: Proactively sleeping for {proactive_delay_between_turns}s after GOT Rep.")
            time.sleep(proactive_delay_between_turns)

        if current_round < max_rounds:
            current_round += 1
            self.logger.info(f"Debate Round {current_round}: GOT Representative")
            # prompt_got = f"As the GOT Representative, elaborate on your core idea: '{str(got_idea)[:150]}...', considering the mission: '{str(mission_context)[:100]}...'. Explain its strengths and how it addresses the core problem."
            prompt_got = f"""
                As the GOT(Graph Of Thoughts) Representative, elaborate on your core idea: '{str(got_idea)}', considering the mission: '{str(mission_context)}'. 
                Explain how your idea addresses the core problem and highlight its key strengths. 
                Then, critically evaluate the ROT(ReversalOfThought) idea: '{str(rot_idea)}' and the LOT(Layer Of Thoughts)  idea: '{str(lot_idea)}'. 
                Discuss their potential weaknesses, overlooked aspects, or limitations compared to your own idea, and explain why your approach is preferable.
                """
            got_statement = got_agent.speak(prompt_got, discussion_context_summary)
            debate_transcript.append({"speaker": got_agent.name, "utterance": got_statement})
            discussion_context_summary += f"\nRound {current_round} - {got_agent.name}:\n{got_statement}\n"
            self.logger.info(f"Debate: Proactively sleeping for {proactive_delay_between_turns}s after GOT Rep.")
            time.sleep(proactive_delay_between_turns)

        if current_round < max_rounds:
            current_round += 1
            self.logger.info(f"Debate Round {current_round}: LOT Representative")
            # prompt_lot = f"As the LOT Representative, elaborate on your core idea: '{str(lot_idea)[:150]}...', considering the mission: '{str(mission_context)[:100]}...'. Explain its strengths and how it addresses the core problem."
            prompt_lot = f"""
                As the LOT(Layer Of Thoughts) Representative, elaborate on your core idea: '{str(lot_idea)}', considering the mission: '{str(mission_context)}'. 
                Explain how your idea addresses the core problem and highlight its key strengths. 
                Then, critically evaluate the GOT(Graph Of Thoughts) idea: '{str(got_idea)}' and the ROT(ReversalOfThought) idea: '{str(rot_idea)}'. 
                Discuss their potential weaknesses, overlooked aspects, or limitations compared to your own idea, and explain why your approach is preferable.
                """
            # prompt_lot = f"As the LOT Representative, elaborate on your core idea:: '{str(lot_idea)[:150]}...'. Explain how it implements GOT's idea ('{str(got_statement)[:100]}...') and addresses potential challenges in implementing the mission '{str(mission_context)[:100]}...'."
            lot_statement = lot_agent.speak(prompt_lot, discussion_context_summary)
            debate_transcript.append({"speaker": lot_agent.name, "utterance": lot_statement})
            discussion_context_summary += f"\nRound {current_round} - {lot_agent.name}:\n{lot_statement}\n"
            self.logger.info(f"Debate: Proactively sleeping for {proactive_delay_between_turns}s after LOT Rep.")
            time.sleep(proactive_delay_between_turns)

        if current_round < max_rounds:
            current_round += 1
            self.logger.info(f"Debate Round {current_round}: Critical Analyst")
            # prompt_critic = f"As the Critical Analyst, critically evaluate ROT's idea ('{str(rot_statement)[:100]}...') , GOT's idea ('{str(got_statement)[:100]}...'), and LOT's idea ('{str(lot_statement)[:100]}...'). Identify potential weaknesses, overlooked aspects, or inconsistencies regarding the mission '{str(mission_context)[:100]}...'. Evaluate the correctness of the response in relation to the mission context. Suggest improvements."
            # prompt_critic = f"""
            #     As the Critical_Analyst, your task is to evaluate the ideas presented by ROT and GOT representatives for the mission: '{str(mission_context)}'.
            #     ROT's latest statement/idea: '{str(rot_idea)}'
            #     GOT's latest statement/idea: '{str(got_idea)}'
            #     GOT's latest statement/idea: '{str(lot_idea)}'
            #     Critically evaluate all three. Identify potential weaknesses, overlooked aspects, or inconsistencies in each, relative to the mission.
            #     Assess the correctness and completeness of each proposed solution.
            #     Suggest specific improvements or points of caution for each.
            #     Provide a balanced overall critique.

            #     Your response must strictly follow the format below:
            #     ### Synthesized Accurate Answer:
            #     (Provide a clear and concise synthesized answer derived from the combined strengths of ROT,GOT and LOT.)
            #     """
            prompt_critic = f"""
                As the Critical_Analyst, your task is to evaluate the ideas presented by ROT and GOT representatives for the mission: '{str(mission_context)}'.
                ROT's latest statement/idea: '{str(rot_idea)}'
                GOT's latest statement/idea: '{str(got_idea)}'
                GOT's latest statement/idea: '{str(lot_idea)}'
                Critically evaluate all three. Identify potential weaknesses, overlooked aspects, or inconsistencies in each, relative to the mission.
                Assess the correctness and completeness of each proposed solution.
                Suggest specific improvements or points of caution for each.
                Provide a balanced overall critique.

                At the very end of your response, include this final section:

                ### Synthesized Accurate Answer:
                (Provide a clear, concise answer that integrates the best insights from ROT, GOT, and LOT. This section must appear last. If it is a multiple-choice question, your answer must exactly match one of the options provided in the question.)
                """
            critic_statement = critic_agent.speak(prompt_critic, discussion_context_summary)
            debate_transcript.append({"speaker": critic_agent.name, "utterance": critic_statement})
            discussion_context_summary += f"\nRound {current_round} - {critic_agent.name}:\n{critic_statement}\n"

        self.logger.section_end(f"MAS Style Debate (Completed {current_round} Rounds)")
        return debate_transcript

    def _get_prm_feedback_for_reasoning_process(self, task_description, current_reasoning_process, iteration_count):
        if not self.iterative_optimizer_llm or not hasattr(self.iterative_optimizer_llm, 'generate'):
            self.logger.warning("MASOrchestrator._get_prm_feedback", "Iterative optimizer LLM not effectively initialized. Returning dummy feedback.")
            return 0.5, "Dummy PRM feedback: Please check logical coherence and completeness.", "DummyPRM_Evaluator"
        if isinstance(self.iterative_optimizer_llm, BaseDummyLLM):
             self.logger.warning("MASOrchestrator._get_prm_feedback", "Iterative optimizer LLM is BaseDummyLLM. Returning dummy feedback.")
             return 0.5, "Dummy PRM feedback (from BaseDummyLLM): Check logic.", "BaseDummyPRM_Evaluator"

        prm_feedback_prompt = f"""
        As an advanced Process Reward Model (PRM) evaluator, your task is to assess the following reasoning process/answer for the given task and provide actionable optimization feedback.
        Main Task:
        {task_description}
        Current Reasoning Process/Answer (Iteration {iteration_count}):
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
        In your justification, clearly point out at least 1-2 specific points most in need of improvement and briefly state how to improve them.
        Output Format (Strictly Adhere):
        PRM Score: [A float between 0.0 and 1.0]
        PRM Justification: [Detailed assessment covering the aspects above and specific, actionable improvement suggestions]
        """
        self.logger.info(f"MASOrchestrator: Getting PRM feedback for reasoning (Iteration {iteration_count})...")
        try:
            llm_response = call_llm_with_retry(
                self.iterative_optimizer_llm.generate,
                prompt=prm_feedback_prompt,
                logger=self.logger,
                llm_name="PRM_Feedback_LLM"
            )
        except Exception as e:
            self.logger.error("MASOrchestrator._get_prm_feedback", f"Critical error during LLM call for PRM feedback (after retries): {e}")
            return 0.1, f"Error getting PRM feedback (after retries): {e}", "PRM_Feedback_Error"

        score_match = re.search(r"PRM Score:\s*([0-9.]+)", str(llm_response), re.IGNORECASE)
        justification_match = re.search(r"PRM Justification:\s*(.+)", str(llm_response), re.IGNORECASE | re.DOTALL)

        prm_score = float(score_match.group(1)) if score_match else 0.0
        prm_justification = justification_match.group(1).strip() if justification_match else "Could not parse PRM justification."

        if not score_match:
            self.logger.warning("MASOrchestrator._get_prm_feedback", f"Could not parse PRM score from response. Raw response: '{llm_response}'")
            if "Could not parse PRM justification." in prm_justification and "PRM Score:" not in str(llm_response) :
                 prm_justification = f"PRM Evaluator LLM raw output (format may be incorrect): {llm_response}"

        self.logger.info(f"MASOrchestrator: Iteration {iteration_count} - PRM Score: {prm_score:.2f}, Justification (start): {prm_justification[:150]}...")
        return prm_score, prm_justification, "ImplicitPRM_LLMEvaluator"

    def ROT_phase(self, initial_task_description, rot_demonstrations=None, problem_instance_for_rot_final_solve=None, num_debate_rounds=4, num_prm_iterations=3,proactive_delay_between_stages=15):
        # 為返回值提供明確的初始/預設值
        rot_solution = f"Default ROT solution due to error or skipped ROT processing for: {initial_task_description[:50]}..."
        refined_task_prompt_for_core_logic = initial_task_description # 這個已經有初始值了，很好

        if hasattr(self.rot_system, 'cognitive_preference_manager') and not isinstance(self.rot_system, DummyReversalOfThought):
            self.logger.info("--- ROT Phase ---")
            try:
                pgrr_output = self.rot_system.preference_guided_reverse_reasoning_warmup(
                    demonstrations=rot_demonstrations if rot_demonstrations else [("Example input", "Example output")],
                    main_task_description_for_prm=initial_task_description, # 將在這裡移除 PRM
                    warm_iterations=1
                )
                print("pgrr_output", pgrr_output)

                current_prompt_for_solving = initial_task_description # 預設使用初始任務描述

                if pgrr_output and "dummy" not in str(pgrr_output).lower() and "error" not in str(pgrr_output).lower():
                    # 即使 pgrr_output 有效，我們也可能需要決定是否調用 cpm
                    # 為了移除 PRM，這裡我們假設 cpm 的主要作用是基於 PRM 進行優化，所以我們可能簡化或跳過它
                    # 或者，如果 cpm 還有其他非 PRM 的邏輯，則保留調用但確保其健壯性
                    # 為了簡化，我們先假設直接使用 pgrr_output (如果有效) 或 initial_task_description
                    # 以下 cpm_output 相關邏輯需要調整或移除
                    cpm_output_internal = None # 初始化 cpm_output_internal
                    try: # 為 cpm_output 的訪問添加保護
                        cpm_output_internal = self.rot_system.cognitive_preference_manager(
                            original_task_prompt_text=initial_task_description,
                            llm_taste_prompt_text=pgrr_output,
                            main_task_description_for_prm=initial_task_description # 將在這裡移除 PRM
                        )
                    except Exception as cpm_exc:
                        self.logger.warning("MASOrchestrator", f"ROT CPM call failed: {cpm_exc}. Will use pgrr_output or initial task description.")

                    if cpm_output_internal and "dummy" not in str(cpm_output_internal).lower() and "error" not in str(cpm_output_internal).lower():
                        refined_task_prompt_for_core_logic = cpm_output_internal
                        current_prompt_for_solving = cpm_output_internal
                    elif pgrr_output and "dummy" not in str(pgrr_output).lower() and "error" not in str(pgrr_output).lower(): # 如果 CPM 失敗但 PGRR 成功
                        refined_task_prompt_for_core_logic = pgrr_output # 可以考慮使用 pgrr 的輸出作為精煉提示
                        current_prompt_for_solving = pgrr_output
                    else: # 如果兩者都無效
                        self.logger.warning("MASOrchestrator", "ROT PGRR and CPM did not yield a usable prompt, using initial task description for core logic.")
                        # refined_task_prompt_for_core_logic 和 current_prompt_for_solving 保持為 initial_task_description
                else: # pgrr_output 無效
                     self.logger.warning("MASOrchestrator", "ROT PGRR returned dummy or error, using initial task description for core logic.")
                     # refined_task_prompt_for_core_logic 和 current_prompt_for_solving 保持為 initial_task_description

                rot_solution = self.rot_system.solve_task_with_final_prompt(
                    str(current_prompt_for_solving), # 使用 current_prompt_for_solving
                    initial_task_description
                )

                self.logger.info(f"ROT Phase refined task prompt: {refined_task_prompt_for_core_logic[:100]}...")
                self.logger.info(f"ROT Phase output : {rot_solution[:100]}...")
                self.logger.info(f"Proactively sleeping for {proactive_delay_between_stages}s after ROT phase.")
                time.sleep(proactive_delay_between_stages)
            except Exception as e_rot: # 捕獲 ROT 流程中的任何異常
                self.logger.warning("MASOrchestrator", f"ROT phase execution error: {e_rot}. Using default rot_solution and initial task description for core logic.")
                # refined_task_prompt_for_core_logic 已經是 initial_task_description
                # rot_solution 已經有了預設值
        else: # 如果 ROT 系統不符合條件 (例如是 Dummy)
            self.logger.info("--- ROT Phase Skipped (system not suitable or is Dummy) ---")
            # rot_solution 和 refined_task_prompt_for_core_logic 使用的是函數開頭的預設值

        return rot_solution, refined_task_prompt_for_core_logic

    def GOT_phase(self, initial_task_description, rot_demonstrations=None, problem_instance_for_rot_final_solve=None, num_debate_rounds=4, num_prm_iterations=3, proactive_delay_between_stages=15):
        self.logger.info("--- GOT Phase Start ---")
        try:
            # Step 1: initial ideas
            ideas = self.got_system.generate_thoughts(initial_task_description, num=2)
            for idea in ideas:
                self.logger.info(f"Initial idea {idea.id}: {idea.content}")

            # Step 2: elaborate best (first) idea
            if ideas:
                eid = ideas[0].id
                elaborated = self.got_system.generate_thoughts(initial_task_description, num=1, from_ids=[eid])
                for e in elaborated:
                    self.logger.info(f"Elaborated {e.id}: {e.content}")
            else:
                elaborated = []

            # Step 3: refine elaboration
            refined = []
            if elaborated:
                refined_t = self.got_system.refine_thought(elaborated[0].id, initial_task_description, "Refine for clarity and detail.")
                if refined_t:
                    self.logger.info(f"Refined {refined_t.id}: {refined_t.content}")
                    refined = [refined_t]

            # Step 4: aggregate if possible
            to_agg = [t.id for t in refined] + ([ideas[1].id] if len(ideas) > 1 else [])
            if len(to_agg) >= 2:
                agg = self.got_system.aggregate_thoughts(to_agg, initial_task_description)
                if agg:
                    self.logger.info(f"Aggregated {agg.id}: {agg.content}")
                    aggregated_thought = agg     
            elif len(to_agg) == 1:
                # only one candidate: just return that Thought
                single_id = to_agg[0]
                candidate = self.got_system.get_thought(single_id)
                if candidate:
                    self.logger.info(f"Only one thought ({single_id}); using it as aggregation.")
                    aggregated_thought = candidate
            # self.got_system.print_graph()
        except Exception as e:
            self.logger.warning("GOT",f"GOT error: {e}")
        self.logger.info(f"Sleeping {proactive_delay_between_stages}s")
        time.sleep(proactive_delay_between_stages)
        return aggregated_thought.content

    
    def LOT_phase(self, initial_task_description, rot_demonstrations=None, problem_instance_for_rot_final_solve=None, num_debate_rounds=4, num_prm_iterations=3,proactive_delay_between_stages=15):
        self.logger.info("--- LOT Phase ---")
        lot_detailed_plan_str = f"Default LOT detailed plan, task: {initial_task_description[:50]}..."
        try:
            plan_output = self.lot_system.run_pipeline(
                conceptual_steps=["Analyze task description", "Formulate detailed steps", "Construct final plan","Generate and present the answer"],
                main_task_description=initial_task_description,
                # initial_input=got_best_idea_content
            )
            if plan_output:
                lot_detailed_plan_str = str(plan_output)
        except Exception as e_lot:
            self.logger.warning("MASOrchestrator", f"LOT phase execution error: {e_lot}. Using default LOT plan.")
        
        self.logger.info(f"Proactively sleeping for {proactive_delay_between_stages}s after LOT phase.")
        time.sleep(proactive_delay_between_stages)
        return lot_detailed_plan_str

    def run_collaborative_task(self, initial_task_description, rot_demonstrations=None, problem_instance_for_rot_final_solve=None, num_debate_rounds=4, num_prm_iterations=3, index=None):
        proactive_delay_between_stages = 15
        self.logger.section_start(f"Collaborative Task")
        self.logger.info(f"Initial task description: {initial_task_description[:100]}...")

        # ROT, GOT, LOT phases (single pass, PRM removed)
        rot_solution, refined_task_prompt_for_core_logic = self.ROT_phase(
            initial_task_description,
            rot_demonstrations=rot_demonstrations,
            problem_instance_for_rot_final_solve=problem_instance_for_rot_final_solve,
            num_debate_rounds=num_debate_rounds,
            num_prm_iterations=num_prm_iterations,
            proactive_delay_between_stages=proactive_delay_between_stages
        )
        aggregated_thought = self.GOT_phase(
            initial_task_description,
            rot_demonstrations=rot_demonstrations,
            problem_instance_for_rot_final_solve=problem_instance_for_rot_final_solve,
            num_debate_rounds=num_debate_rounds,
            num_prm_iterations=num_prm_iterations,
            proactive_delay_between_stages=proactive_delay_between_stages
        )
        lot_detailed_plan_str = self.LOT_phase(
            initial_task_description,
            rot_demonstrations=rot_demonstrations,
            problem_instance_for_rot_final_solve=problem_instance_for_rot_final_solve,
            num_debate_rounds=num_debate_rounds,
            num_prm_iterations=num_prm_iterations,
            proactive_delay_between_stages=proactive_delay_between_stages
        )

        # Debate phase
        self.logger.info("--- MAS Debate Phase ---")
        mas_debate_transcript = self.conduct_mas_debate(
            mission_context=initial_task_description,
            rot_idea=rot_solution,
            got_idea=aggregated_thought,
            lot_idea=lot_detailed_plan_str,
            max_rounds=num_debate_rounds,
        )
        debate_summary_for_synthesis = "Debate Record Summary:\n"
        rows = []
        for idx, entry in enumerate(mas_debate_transcript, start=1):
            utterance_single_line = entry["utterance"].replace("\n", " ").strip()
            debate_summary_for_synthesis += f"{entry['speaker']}: {utterance_single_line}...\n"
            rows.append({
                "Round": idx,
                "Speaker": entry["speaker"],
                "Utterance": utterance_single_line
            })

        # Save transcript CSV
        df = pd.DataFrame(rows, columns=["Round", "Speaker", "Utterance"])
        # df.to_csv(f"C:\\Users\\user\\Documents\\GitHub\\MAS-PRM\\debate_transcripts\\test\\debate_transcript_q{index}_r0.csv", index=False, encoding="utf-8-sig")
        df.to_csv(f"C:\\Users\\user\\Desktop\\MAS-PRM-ablation_remove_prm\\debate_transcripts\\part4\\debate_transcript_q{index}_r0.csv", index=False, encoding="utf-8-sig")
        print("✅ 已輸出 debate_transcript.csv，包含輪次、發言者與內容，可在 Excel 中直接瀏覽。")

        self.logger.info(f"Proactively sleeping for {proactive_delay_between_stages}s after MAS Debate phase.")
        time.sleep(proactive_delay_between_stages)

        # Capture pre-synthesis summary
        original_thoughtflow_summary_pre_prm = (
            f"Initial Task: {initial_task_description}...\n"
            f"ROT Idea: {rot_solution}...\n"
            f"GOT Idea: {aggregated_thought}...\n"
            f"LOT Idea: {lot_detailed_plan_str}...\n"
            f"{debate_summary_for_synthesis}\n"
            "--- End of Pre-PRM Thoughtflow Components ---"
        )
        self.logger.info(f"Original thoughtflow (pre-PRM) captured. Length: {len(original_thoughtflow_summary_pre_prm)}")

        # Initial synthesis
        self.logger.info("--- Initial Synthesis Phase ---")
        current_reasoning_artifact = f"Initial synthesis result, task: {initial_task_description[:50]}..."
        if not self.synthesis_llm or isinstance(self.synthesis_llm, BaseDummyLLM) or not hasattr(self.synthesis_llm, 'generate'):
            self.logger.warning("MASOrchestrator", "Synthesis LLM not effectively initialized. Using placeholder for initial synthesis.")
        else:
            # synthesis_prompt = f"""
            # Task Context (Original): {initial_task_description}

            # Outputs from Reasoning Modules This Cycle:
            # ROT Solution/Idea: {rot_solution}
            # GOT Core Idea: {aggregated_thought}
            # LOT Core Idea: {lot_detailed_plan_str}
            # Debate Summary This Cycle: {debate_summary_for_synthesis}

            # Synthesize these elements into a coherent and comprehensive answer/reasoning process for the task.
            # Focus on fulfilling the requirements of 'Current Task Input for this Cycle'.
            # **Important:** Do not mention the source of information (e.g., ROT, GOT). Integrate them seamlessly.

            # Your response must strictly follow the format below:
            # ### Synthesized Accurate Answer:
            # (Provide a clear and concise synthesized answer derived from the combined strengths of ROT, GOT and LOT.)
            # """
            synthesis_prompt = f"""
            Task Context (Original): {initial_task_description}

            Outputs from Reasoning Modules This Cycle:
            ROT Solution/Idea: {rot_solution}
            GOT Core Idea: {aggregated_thought}
            LOT Core Idea: {lot_detailed_plan_str}
            Debate Summary This Cycle: {debate_summary_for_synthesis}

            Synthesize these elements into a coherent and comprehensive answer/reasoning process for the task.
            Focus on fulfilling the requirements of 'Current Task Input for this Cycle'.
            **Important:** Do not mention the source of information (e.g., ROT, GOT). Integrate them seamlessly.

            At the very end of your response, include this final section:

            ### Synthesized Accurate Answer:
            (Provide a clear, concise answer that integrates the best insights from ROT, GOT, and LOT. This section must appear last. If it is a multiple-choice question, your answer must exactly match one of the options provided in the question.)
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
                    self.logger.warning("MASOrchestrator", f"Initial synthesis LLM did not produce valid output or produced error/dummy response. Using placeholder. Output: {synthesis_output}")
            except Exception as e_synth:
                self.logger.error("MASOrchestrator", f"Critical error during initial synthesis LLM call (after retries): {e_synth}")

        self.logger.answer("INITIAL_SYNTHESIS", current_reasoning_artifact, is_final=False, detail_level=1)
        self.logger.info(f"Proactively sleeping for {proactive_delay_between_stages}s after Initial Synthesis.")
        time.sleep(proactive_delay_between_stages)

        # No PRM iterations
        thoughtflow_summary_incl_prm = original_thoughtflow_summary_pre_prm
        prm_iteration_history_details = []

        # Final output
        final_synthesized_plan = current_reasoning_artifact
        self.logger.answer("FINAL_OUTPUT (POST-PRM-ITERATION)", final_synthesized_plan, is_final=True)
        self.logger.section_end(f"Collaborative Task")

        return {
            "synthesized_final_plan": final_synthesized_plan,
            "original_thoughtflow_summary_pre_prm": original_thoughtflow_summary_pre_prm,
            "thoughtflow_summary_incl_prm": thoughtflow_summary_incl_prm,
            "prm_iteration_history_details": prm_iteration_history_details
        }



# --- Evaluation Functions ---
def evaluate_with_llm(task_description, thoughtflow_summary, generated_answer, ground_truth_answer, llm_interface, logger, beta_prm):
    R_score_val = None
    receval_assessment_text = "RECEVAL assessment placeholder: LLM not called or error."
    label_l = 0
    similarity_score = 0.0

    if not llm_interface or not hasattr(llm_interface, 'generate'):
        logger.warning("evaluate_with_llm", "Evaluation LLM not effectively initialized. Returning placeholder values.")
        R_score_val = 0.1 
        similarity_score = 0.5 
        if similarity_score >= SIMILARITY_THRESHOLD_FOR_LABEL_L: label_l = 1
        return R_score_val, receval_assessment_text, label_l, similarity_score

    if isinstance(llm_interface, BaseDummyLLM):
        logger.warning("evaluate_with_llm", "Evaluation LLM is BaseDummyLLM. Returning dummy placeholder values.")
        R_score_val = 0.1 
        similarity_score = 0.5 
        if similarity_score >= SIMILARITY_THRESHOLD_FOR_LABEL_L: label_l = 1
        return R_score_val, receval_assessment_text, label_l, similarity_score

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
Output only the numerical R-score:"""
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
            logger.warning(f"evaluate_with_llm: Could not parse R-score from LLM output: '{r_score_output_str}'.Setting R-score to default 0.0.")
            R_score_val = 0.0
    except Exception as e:
        logger.error("evaluate_with_llm", f"Error parsing R-score from LLM (after retries): {e}. Output: '{r_score_output_str if r_score_output_str else 'N/A'}'")
        R_score_val = 0.0

    receval_assessment_text_output = None
    try:
        logger.info("Requesting RECEVAL assessment from LLM...")
        receval_prompt = f"""Task: Evaluate the following thoughtflow based on RECEVAL criteria.
Original Question: {task_description}
Thoughtflow Summary (e.g., debate, PRM iterations):
\"\"\"
{thoughtflow_summary} 
\"\"\"
Final Answer generated from this thoughtflow:
\"\"\"
{generated_answer}
\"\"\"
RECEVAL Criteria:
1.  Clarity & Coherence: Is the reasoning process easy to understand? Are steps logically connected?
2.  Soundness & Validity: Are arguments sound? Are inferences valid?
3.  Sufficiency & Completeness: Does the reasoning cover all necessary aspects of the question? Any omissions?
4.  Relevance: Are all parts of the reasoning relevant to answering the question?
5.  Efficiency: Is the reasoning concise, or does it include unnecessary detours?
Provide a qualitative assessment of the thoughtflow:"""
        receval_assessment_text_output = call_llm_with_retry(
            llm_interface.generate,
            prompt=receval_prompt,
            logger=logger,
            llm_name="RECEVAL_Eval_LLM"
        )
        if receval_assessment_text_output and not str(receval_assessment_text_output).lower().startswith("error:") and not str(receval_assessment_text_output).lower().startswith("llm dummy response"):
             receval_assessment_text = str(receval_assessment_text_output)
        logger.info(f"RECEVAL assessment from LLM (length): {len(str(receval_assessment_text))}")
    except Exception as e:
        logger.error("evaluate_with_llm", f"Error getting RECEVAL assessment from LLM (after retries): {e}")
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
            prompt=similarity_prompt,
            logger=logger,
            llm_name="Similarity_Eval_LLM"
        )
        sim_score_match = re.search(r"([0-9.]+)", str(similarity_output_str))
        if sim_score_match:
            similarity_score = float(sim_score_match.group(1))
            logger.info(f"Semantic similarity score from LLM: {similarity_score}")
        else:
            logger.warning(f"evaluate_with_llm: Could not parse similarity score from LLM output: '{similarity_output_str}'. Setting to default 0.0.")
            similarity_score = 0.0

        if similarity_score >= SIMILARITY_THRESHOLD_FOR_LABEL_L:
            label_l = 1
        else:
            label_l = 0
        logger.info(f"Label l set to {label_l} based on similarity {similarity_score} (threshold {SIMILARITY_THRESHOLD_FOR_LABEL_L})")
    except Exception as e:
        logger.error("evaluate_with_llm", f"Error parsing similarity score from LLM (after retries): {e}. Output: '{similarity_output_str if similarity_output_str else 'N/A'}'")
        similarity_score = 0.0
        label_l = 0

    return R_score_val, receval_assessment_text, label_l, similarity_score


def calculate_cross_entropy(R_score, label_l, logger):
    if R_score is None:
        logger.warning("calculate_cross_entropy", "R_score is None. Cannot calculate cross-entropy. Returning None.")
        return None
    try:
        R_score_float = float(R_score)
        if R_score_float > 700: 
            log_sigma_R = 0.0
            log_one_minus_sigma_R = -R_score_float
        elif R_score_float < -700: 
            log_sigma_R = R_score_float
            log_one_minus_sigma_R = 0.0
        else:
            sigma_R = 1 / (1 + math.exp(-R_score_float))
            epsilon = 1e-9 
            sigma_R_clipped = max(epsilon, min(sigma_R, 1 - epsilon))
            log_sigma_R = math.log(sigma_R_clipped)
            log_one_minus_sigma_R = math.log(1 - sigma_R_clipped)

        label_l_int = int(label_l)
        loss = - (label_l_int * log_sigma_R + (1 - label_l_int) * log_one_minus_sigma_R)
        return loss
    except (ValueError, TypeError) as e:
        logger.error("calculate_cross_entropy", f"Error converting R_score or label_l to numeric: {e}. R_score: {R_score}, label_l: {label_l}")
        return None
    except Exception as e:
        logger.error("calculate_cross_entropy", f"Error calculating cross-entropy: {e}")
        return None

def calculate_nlp_metrics(generated_answer, ground_truth_answer, logger):
    metrics = {
        "bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "meteor": 0.0,
        "bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0
    }
    str_generated_answer = str(generated_answer) if generated_answer is not None else ""
    str_ground_truth_answer = str(ground_truth_answer) if ground_truth_answer is not None else ""

    if not str_generated_answer.strip() or not str_ground_truth_answer.strip():
        logger.warning("calculate_nlp_metrics", "Generated answer or ground truth is empty or whitespace. NLP metrics will be 0.")
        return metrics

    try:
        ref_tokens = [word_tokenize(str_ground_truth_answer.lower())] if str_ground_truth_answer.strip() else [[]]
        gen_tokens = word_tokenize(str_generated_answer.lower()) if str_generated_answer.strip() else []
    except LookupError as le: # Specifically for punkt not found
        logger.error("calculate_nlp_metrics", f"Tokenization failed due to NLTK resource 'punkt' not found: {le}. NLP metrics will be 0.")
        return metrics
    except Exception as e:
        logger.error("calculate_nlp_metrics", f"Tokenization failed: {e}. NLP metrics will be 0.")
        return metrics

    if not gen_tokens and not ref_tokens[0]:
        logger.info("calculate_nlp_metrics", "Both generated and reference answers are empty (post-tokenization). NLP metrics will be 0.")
        return metrics
    if not gen_tokens:
        logger.warning("calculate_nlp_metrics", "Generated answer is empty after tokenization. Most NLP metrics will be 0.")
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores_dict = scorer.score(str_ground_truth_answer, "")
            metrics["rouge1"] = rouge_scores_dict['rouge1'].fmeasure
            metrics["rouge2"] = rouge_scores_dict['rouge2'].fmeasure
            metrics["rougeL"] = rouge_scores_dict['rougeL'].fmeasure
        except Exception as e_rouge_empty:
            logger.error("calculate_nlp_metrics", f"Error calculating ROUGE for empty generated answer: {e_rouge_empty}")
        return metrics

    try:
        chencherry = SmoothingFunction()
        metrics["bleu"] = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=chencherry.method1)
    except ZeroDivisionError:
        logger.warning("calculate_nlp_metrics", "ZeroDivisionError during BLEU calculation (often due to short generated answer). BLEU set to 0.0.")
        metrics["bleu"] = 0.0
    except Exception as e:
        logger.error("calculate_nlp_metrics", f"Error calculating BLEU: {e}")

    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores_dict = scorer.score(str_ground_truth_answer if str_ground_truth_answer.strip() else " ",
                                         str_generated_answer if str_generated_answer.strip() else " ")
        metrics["rouge1"] = rouge_scores_dict['rouge1'].fmeasure
        metrics["rouge2"] = rouge_scores_dict['rouge2'].fmeasure
        metrics["rougeL"] = rouge_scores_dict['rougeL'].fmeasure
    except Exception as e:
        logger.error("calculate_nlp_metrics", f"Error calculating ROUGE: {e}")

    try:
        metrics["meteor"] = meteor_score(ref_tokens, gen_tokens)
    except LookupError as le: # Specifically for wordnet/omw-1.4 not found
        logger.error("calculate_nlp_metrics", f"Error calculating METEOR due to NLTK resource (e.g., 'wordnet') not found: {le}. METEOR set to 0.0.")
        metrics["meteor"] = 0.0
    except Exception as e: 
        logger.error("calculate_nlp_metrics", f"Error calculating METEOR: {e}")
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
            logger.warning("calculate_nlp_metrics", "PyTorch not available. Skipping BERTScore calculation.")
            metrics["bert_precision"] = 0.0
            metrics["bert_recall"] = 0.0
            metrics["bert_f1"] = 0.0
    except Exception as e:
        logger.error("calculate_nlp_metrics", f"Error calculating BERTScore: {e}")
        metrics["bert_precision"] = 0.0
        metrics["bert_recall"] = 0.0
        metrics["bert_f1"] = 0.0
    return metrics

# --- Main Processing Logic ---
def main():
    logger = TerminalLogger(verbose=True)
    logger.section_start("Main Evaluation Flow")

    global IMPORTS_SUCCESSFUL, GEMINI_API_KEY, torch 
    # GOOGLE_API_KEY = 'AIzaSyD5claU-RYhHoZp9-NDiwywZTjPbzf6iUo'
    # GEMINI_API_KEY = 'AIzaSyD5claU-RYhHoZp9-NDiwywZTjPbzf6iUo'
    print('gemini_api_key',GEMINI_API_KEY)
    try:
        import torch as pytorch_module 
        torch = pytorch_module 
        logger.info("PyTorch successfully imported. BERTScore should work.")
    except ImportError:
        logger.warning("main", "Failed to import PyTorch (torch). BERTScore calculation will be skipped or fail. Please install PyTorch.")
        torch = None 

    if not IMPORTS_SUCCESSFUL:
        logger.warning("main", "One or more core modules (GOT, LOT, ROT) failed to import. Functionality will rely on dummy classes.")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.warning("main", f"GEMINI_API_KEY is invalid ('{str(GEMINI_API_KEY)[:10]}...'). LLM calls will use dummy classes or fail if real modules are loaded.")

    orchestrator = MASOrchestrator(api_key=GEMINI_API_KEY, logger=logger)
    evaluation_llm_interface = orchestrator.synthesis_llm

    if not evaluation_llm_interface or not hasattr(evaluation_llm_interface, 'generate'):
         logger.warning("main", "Evaluation LLM (from orchestrator.synthesis_llm) not effectively initialized. Evaluation results will be placeholders.")
    elif isinstance(evaluation_llm_interface, BaseDummyLLM):
         logger.warning("main", "Evaluation LLM (from orchestrator.synthesis_llm) is BaseDummyLLM. Evaluation results will be placeholders.")

    # csv_file_path = r"C:\Users\user\Documents\GitHub\MAS-PRM\dataset\All_of_dataset_test.csv" 
    csv_file_path = r"C:\Users\user\Desktop\MAS-PRM-ablation_remove_prm\dataset\All_of_dataset_part4.csv" 
    # csv_file_path = "data.csv"
    logger.info(f"Attempting to load CSV data from: {csv_file_path}")

    if not os.path.exists(csv_file_path):
        logger.error("main", f"CSV file not found: {csv_file_path}")
        try:
            logger.info(f"Attempting to create a sample {csv_file_path} file at: {os.path.abspath(csv_file_path)}")
            sample_df = pd.DataFrame(columns=['instruction', 'context', 'response'])
            sample_df.loc[0] = ["What is the capital of France?", "France is a country in Europe.", "The capital of France is Paris."]
            sample_df.loc[1] = ["Explain the theory of relativity.", "", "The theory of relativity, developed by Albert Einstein, fundamentally changed our understanding of space, time, gravity, and the universe. It consists of two main parts: Special Relativity and General Relativity..."]
            sample_df.to_csv(csv_file_path, index=False, encoding='utf-8') 
            logger.info(f"Sample {csv_file_path} created with UTF-8 encoding. Please populate it with your data and rerun.")
        except Exception as e_create:
            logger.error("main", f"Failed to create sample {csv_file_path}: {e_create}")
        logger.section_end("Main Evaluation Flow")
        return

    try:
        # Attempt to read with UTF-8 first
        try:
            dataset_df = pd.read_csv(csv_file_path, encoding='utf-8')
            logger.info(f"Successfully loaded {len(dataset_df)} records from '{csv_file_path}' using UTF-8 encoding.")
        except UnicodeDecodeError:
            logger.warning("main", f"Failed to load '{csv_file_path}' with UTF-8 encoding. Attempting with 'latin1' encoding...")
            dataset_df = pd.read_csv(csv_file_path, encoding='latin1')
            logger.info(f"Successfully loaded {len(dataset_df)} records from '{csv_file_path}' using latin1 encoding.")
        
        required_columns = ['instruction', 'response'] 
        missing_cols = [col for col in required_columns if col not in dataset_df.columns]
        if missing_cols:
            logger.error("main", f"CSV file must contain the following columns: {', '.join(required_columns)}. Missing: {', '.join(missing_cols)}")
            logger.section_end("Main Evaluation Flow")
            return
        if "context" not in dataset_df.columns:
            logger.warning("main", "CSV file does not contain 'context' column. Context will be considered empty for all rows.")
            dataset_df["context"] = "" 
            
        if "rot_demonstration_input" not in dataset_df.columns or "rot_demonstration_output" not in dataset_df.columns:
            logger.info("CSV file does not contain 'rot_demonstration_input' or 'rot_demonstration_output' columns. ROT demonstrations will use defaults if not provided otherwise.")

    except Exception as e:
        logger.error("main", f"Error loading CSV file: {e}")
        logger.section_end("Main Evaluation Flow")
        return

    all_results = []
    num_processed = 0
    default_max_items = len(dataset_df)
    
    max_items_to_process_str = input(f"Dataset has {default_max_items} items. How many to process? (Enter number, or press Enter for all [{default_max_items}]): ")
    try:
        max_items_to_process = int(max_items_to_process_str) if max_items_to_process_str.strip() else default_max_items
        if max_items_to_process <= 0 or max_items_to_process > default_max_items:
            logger.warning("main", f"Invalid number of items to process ({max_items_to_process}). Defaulting to all {default_max_items} items.")
            max_items_to_process = default_max_items
    except ValueError:
        logger.warning("main", f"Invalid input for number of items to process. Defaulting to all {default_max_items} items.")
        max_items_to_process = default_max_items
    
    logger.info(f"Will process a maximum of {max_items_to_process} items.")


    for index, row in dataset_df.iterrows():
        if num_processed >= max_items_to_process:
            logger.info(f"Reached processing limit of {max_items_to_process} items.")
            break

        num_processed += 1
        logger.section_start(f"Processing Item {num_processed}/{max_items_to_process} (CSV Index {index})")

        instruction = str(row["instruction"])
        context = str(row["context"]) if "context" in row and pd.notna(row["context"]) else ""
        ground_truth_answer = str(row["response"])

        if context and context.strip():
            question = f"Instruction: {instruction}\n\nContext: {context}"
        else:
            question = f"Instruction: {instruction}"

        logger.info(f"Input (Instruction & Context combined): {question[:250]}...")
        logger.info(f"Ground Truth Answer (Response): {ground_truth_answer[:150]}...")

        rot_demonstrations_for_this_item = None
        if "rot_demonstration_input" in dataset_df.columns and "rot_demonstration_output" in dataset_df.columns and \
           "rot_demonstration_input" in row and "rot_demonstration_output" in row and \
           pd.notna(row["rot_demonstration_input"]) and pd.notna(row["rot_demonstration_output"]):
            rot_demonstrations_for_this_item = [(str(row["rot_demonstration_input"]), str(row["rot_demonstration_output"]))]
            logger.info(f"Using ROT demonstration from CSV: Input='{str(row['rot_demonstration_input'])[:50]}...', Output='{str(row['rot_demonstration_output'])[:50]}...'")
        else:
            logger.info("Not using ROT demonstration from CSV (columns missing or data empty).")


        orchestrator_outputs = orchestrator.run_collaborative_task(
            initial_task_description=question, 
            rot_demonstrations=rot_demonstrations_for_this_item,
            num_debate_rounds=4,
            num_prm_iterations=3,
            index = index
        )
        generated_answer = orchestrator_outputs.get("synthesized_final_plan", "Failed to generate answer.")
        original_thoughtflow_summary = orchestrator_outputs.get("original_thoughtflow_summary_pre_prm", "Failed to generate original thoughtflow summary.")
        thoughtflow_summary_incl_prm = orchestrator_outputs.get("thoughtflow_summary_incl_prm", "Failed to generate thoughtflow summary with PRM.")


        logger.info(f"Generated Answer (post-PRM iteration, partial): {str(generated_answer)[:150]}...")
        logger.info(f"Original Thoughtflow Summary (pre-PRM, partial): {str(original_thoughtflow_summary)[:250]}...")
        logger.info(f"Thoughtflow Summary (incl. PRM iterations, partial): {str(thoughtflow_summary_incl_prm)[:250]}...")

        R_score, receval_assessment, label_l, llm_similarity_score = evaluate_with_llm(
            task_description=question,
            thoughtflow_summary=thoughtflow_summary_incl_prm, 
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

        current_result = {
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
            "thoughtflow_summary_incl_prm": thoughtflow_summary_incl_prm,
            "receval_assessment_thoughtflow_incl_prm": receval_assessment, 
            **{f"nlp_{k}": v for k, v in nlp_metrics.items()}
        }
        all_results.append(current_result)
        logger.section_end(f"Finished Item {num_processed}/{max_items_to_process}")

    logger.section_start("Overall Results Summary")
    if not all_results:
        logger.info("No items were processed.")
    else:
        results_df = pd.DataFrame(all_results)
        # output_excel_filename = "C:\\Users\\user\\Documents\\GitHub\\MAS-PRM\\result\\test\\evaluation_results_dolly_gsm8k.xlsx" 
        output_excel_filename = "C:\\Users\\user\\Desktop\\MAS-PRM-ablation_remove_prm\\result\\part4\\evaluation_results_part4.xlsx" 
        try:
            results_df.to_excel(output_excel_filename, index=False, engine='openpyxl')
            logger.info(f"Detailed evaluation results saved to: {output_excel_filename}")
        except Exception as e:
            logger.error("main", f"Error saving results to Excel: {e}. Please ensure 'openpyxl' is installed.")

        numeric_cols_for_avg = [
            "R_score_final_answer", "label_l_final_answer", "llm_similarity_final_answer",
            "cross_entropy_loss_final_answer", "nlp_bleu", "nlp_rouge1", "nlp_rouge2",
            "nlp_rougeL", "nlp_meteor", "nlp_bert_precision", "nlp_bert_recall", "nlp_bert_f1"
        ]
        avg_scores_summary = {}
        logger.info("Average scores for processed items:")
        for col_name in numeric_cols_for_avg:
            if col_name in results_df.columns:
                valid_numeric_series = pd.to_numeric(results_df[col_name], errors='coerce').dropna()
                if not valid_numeric_series.empty:
                    avg_scores_summary[f"avg_{col_name}"] = valid_numeric_series.mean()
                else:
                    avg_scores_summary[f"avg_{col_name}"] = "N/A (no valid data)"
            else:
                avg_scores_summary[f"avg_{col_name}"] = "N/A (column not found)"
            
            avg_value_to_print = avg_scores_summary.get(f"avg_{col_name}")
            if isinstance(avg_value_to_print, float):
                 logger.info(f"  {avg_value_to_print:.4f} : {col_name}")
            else:
                 logger.info(f"  {str(avg_value_to_print)} : {col_name}")


    logger.section_end("Overall Results Summary")
    logger.section_end("Main Evaluation Flow")
def send_discord_notification(content, webhook_url):
    data = {"content": content}
    response = requests.post(webhook_url, json=data)
    if response.status_code == 204:
        print("✅ 成功送出 Discord 通知")
    else:
        print(f"❌ 發送失敗：{response.status_code} - {response.text}")

if __name__ == "__main__":
    main()

    import requests
    # ✅ 程式結束時呼叫這行
    webhook_url = "https://discord.com/api/webhooks/1375009926535057519/3fjUWTbtehoHwk30iNoqrvcETussPnDeD7cvQ6Fe2z0vWQEtRpkUh76pFeqgBxDG6WQa"
    send_discord_notification("✅ 程式完成，快來看結果！", webhook_url)

