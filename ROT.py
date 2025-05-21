import google.generativeai as genai
import numpy as np
import os
import re # Added re module for parsing
from dotenv import load_dotenv

# --- Settings ---
load_dotenv()
GEMINI_API_KEY_FROM_ENV = os.getenv("GEMINI_API_KEY")

# --- Helper: Default Logger ---
class DefaultLogger:
    def info(self, message): print(f"[ROT INFO] {message}")
    def warning(self, message): print(f"[ROT WARNING] {message}")
    def error(self, message): print(f"[ROT ERROR] {message}")

# --- Gemini LLM Interface (ROT Version) ---
class GeminiLLMInterface:
    # def __init__(self, model_name="gemini-1.5-flash-latest", api_key=None, logger=None): # 添加 logger
    def __init__(self, model_name="gemini-2.0-flash-lite", api_key=None, logger=None): # 添加 logger
        self.model = None
        self.logger = logger if logger else DefaultLogger()
        effective_api_key = api_key or GEMINI_API_KEY_FROM_ENV

        if not effective_api_key:
            self.logger.error("ROT.GeminiLLMInterface: API key not provided. LLM will not function.")
            return

        try:
            # Configure genai only if the key has changed or not set yet for this specific purpose
            current_configured_key = getattr(genai, '_last_configured_key_by_rot', None)
            if current_configured_key != effective_api_key:
                genai.configure(api_key=effective_api_key)
                setattr(genai, '_last_configured_key_by_rot', effective_api_key) # Mark as configured with this key
                self.logger.info(f"ROT.GeminiLLMInterface: genai configured with API key ending ...{effective_api_key[-4:]}.")

            self.model = genai.GenerativeModel(model_name)
            self.logger.info(f"ROT.GeminiLLMInterface initialized with model {model_name}.")
        except Exception as e:
            self.logger.error(f"Failed to initialize ROT Gemini GenerativeModel ({model_name}): {e}")
            self.model = None

    def generate(self, prompt_text, temperature=0.7): # Added temperature parameter
        if not self.model:
            self.logger.error("ROT.GeminiLLMInterface: LLM model not initialized. Cannot generate content.")
            return "LLM not initialized or API key error" # Standardized error message
        try:
            self.logger.info(f"\n--- Sending prompt to Gemini (ROT LLM) ---\n{prompt_text[:300]}...\n--- End of Gemini prompt (ROT LLM) ---")
            response = self.model.generate_content(
                prompt_text,
                generation_config=genai.types.GenerationConfig(temperature=temperature) # Use temperature
            )
            llm_response_text = ""
            if hasattr(response, 'parts') and response.parts:
                 llm_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            elif hasattr(response, 'text'):
                 llm_response_text = response.text
            
            if not llm_response_text and hasattr(response, 'prompt_feedback') and \
                 response.prompt_feedback.block_reason != genai.types.BlockReason.BLOCK_REASON_UNSPECIFIED: # Check block reason
                block_reason_str = genai.types.BlockReason(response.prompt_feedback.block_reason).name
                self.logger.warning(f"ROT.GeminiLLMInterface: Prompt blocked due to {block_reason_str}.")
                return f"LLM Error: Prompt blocked due to {block_reason_str}." # More specific error
            
            self.logger.info(f"--- Received Gemini response (ROT LLM) ---\n{llm_response_text[:300]}...\n--- End of Gemini response (ROT LLM) ---")
            return llm_response_text if llm_response_text else "LLM Error: No valid content generated." # Standardized error

        except Exception as e:
            self.logger.error(f"ROT.GeminiLLMInterface: Gemini API call error (generate): {e}")
            return f"LLM Error: {e}" # More specific error

    def generate_with_simulated_score(self, prompt_text, temperature=0.7): # Added temperature
        # This method is used in ROT for the initial candidate prompt generation in PGRR. Its "simulated_score" is NOT a PRM score.
        # PRM scores will be provided by dedicated evaluation methods in the ReversalOfThought class.
        if not self.model:
            self.logger.error("ROT.GeminiLLMInterface: LLM model not initialized. Cannot generate content and score.")
            return "LLM not initialized or API key error", 0.0

        response_text = self.generate(prompt_text, temperature=temperature)
        simulated_score = 0.0 # The prob_score here is a simple simulation based on response length, not a PRM score
        if "LLM not initialized" in response_text or "LLM Error" in response_text: # Check for specific error messages
             simulated_score = 0.0
        else:
            # Keep the original simulated score logic, as PGRR might use it internally
            simulated_score = float(len(response_text)) / 1000.0 
            simulated_score = min(max(simulated_score, 0.0), 1.0) # Clamp between 0 and 1
        return response_text, simulated_score

# --- Gemini Embedding Interface (ROT Version) ---
class GeminiEmbeddingInterface:
    def __init__(self, model_name="models/embedding-001", api_key=None, logger=None): # Added logger
        self.model_name = model_name
        self.api_key_configured_successfully = False
        self.logger = logger if logger else DefaultLogger()
        effective_api_key = api_key or GEMINI_API_KEY_FROM_ENV

        if not effective_api_key:
            self.logger.error("ROT.GeminiEmbeddingInterface: API key not provided. Embedding functionality will not work.")
            return

        try:
            # Use a different attribute to track configuration for embedding to avoid conflict with LLM config
            current_configured_key = getattr(genai, '_last_configured_key_by_rot_embed', None)
            if current_configured_key != effective_api_key:
                genai.configure(api_key=effective_api_key)
                setattr(genai, '_last_configured_key_by_rot_embed', effective_api_key)
                self.logger.info(f"ROT.GeminiEmbeddingInterface: genai configured with API key ending ...{effective_api_key[-4:]} (for embedding).")
            
            self.api_key_configured_successfully = True # Assume success if no exception
            self.logger.info(f"ROT.GeminiEmbeddingInterface initialized for model {model_name}.")
        except Exception as e:
            self.logger.error(f"Error setting API key for ROT.GeminiEmbeddingInterface: {e}")
            self.api_key_configured_successfully = False

    def _get_embedding(self, text):
        if not self.api_key_configured_successfully:
            self.logger.error("ROT.GeminiEmbeddingInterface: API key not set or configuration failed. Cannot get embedding.")
            return None
        try:
            self.logger.info(f"ROT.GeminiEmbeddingInterface: Getting embedding for text: '{text[:50]}...'")
            # Choose task_type based on intended use, e.g., "SEMANTIC_SIMILARITY", "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"
            result = genai.embed_content(model=self.model_name, content=text, task_type="SEMANTIC_SIMILARITY")
            return result['embedding']
        except Exception as e:
            self.logger.error(f"ROT.GeminiEmbeddingInterface: Gemini API call error (embed_content for '{text[:50]}...'): {e}")
            return None

    def calculate_similarity(self, text1, text2):
        if not self.api_key_configured_successfully: # Check if API key was set up
            self.logger.error("ROT.GeminiEmbeddingInterface: API key not set, cannot calculate similarity.")
            return 0.0

        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        if emb1 is None or emb2 is None:
            self.logger.warning("ROT","ROT.GeminiEmbeddingInterface: 無法計算相似度，因為一個或多個嵌入向量為 None。")
            return 0.0

        try:
            emb1_np = np.array(emb1)
            emb2_np = np.array(emb2)
            norm_emb1 = np.linalg.norm(emb1_np)
            norm_emb2 = np.linalg.norm(emb2_np)
            if norm_emb1 == 0 or norm_emb2 == 0: # Prevent division by zero
                return 0.0
            similarity = np.dot(emb1_np, emb2_np) / (norm_emb1 * norm_emb2)
            return float(similarity)
        except Exception as e:
            self.logger.error(f"ROT.GeminiEmbeddingInterface: Error calculating cosine similarity: {e}")
            return 0.0

# --- ReversalOfThought Class ---
class ReversalOfThought:
    def __init__(self, llm_interface, embedding_model_interface, similarity_threshold=0.7, logger=None): # Added logger
        self.llm = llm_interface
        self.embedder = embedding_model_interface
        self.similarity_threshold = similarity_threshold
        self.logger = logger if logger else DefaultLogger()
        # Assume self.llm is an instance of GeminiLLMInterface, which can be used for PRM evaluation.
        # If the evaluation LLM is different, MASOrchestrator needs to pass the specific evaluation LLM when creating ROT.
        self.prm_evaluator_llm = llm_interface # Default to using the same LLM for PRM evaluation

    def _generate_prm_style_scoring_prompt_for_rot_artifact(self, artifact_content, artifact_type, main_task_description):
        """
        Generates a PRM-style scoring prompt for intermediate artifacts produced by ROT (e.g., task definition prompts).
        artifact_type: "Task Definition Prompt" or "Final Optimized Prompt"
        main_task_description: The "higher-level original task" for which the ROT system is trying to generate a good prompt.
        """
        prompt = (
            f"You are an expert evaluator tasked with assessing the quality and potential of a '{artifact_type}'.\n"
            f"This {artifact_type} will ultimately be used to guide a large language model to solve the following main task:\n"
            f"Main Task Objective: '{main_task_description}'\n\n"
            f"Content of the '{artifact_type}' to be evaluated:\n\"\"\"\n{artifact_content}\n\"\"\"\n\n"
            "Evaluation Instructions:\n"
            f"1.  Utility/Contribution: To what extent can this '{artifact_type}' clearly, accurately, and effectively guide an LLM to solve the 'Main Task Objective' above? Is it likely to produce a high-quality solution?\n"
            "2.  Completeness and Accuracy: Does this artifact completely capture the key aspects of the main task? Are there ambiguities, misleading statements, or omissions?\n"
            "3.  Clarity and Actionability: Is this artifact itself easy to understand? Can an LLM easily follow it to perform the task?\n\n"
            "Please provide an overall score and a brief justification.\n"
            "Output Format (Strictly Adhere):\n"
            "Score: [A floating-point number between 0.0 (very poor/unhelpful) and 1.0 (excellent/highly promising)]\n"
            "Justification: [A brief explanation for your score, stating how and why it helps or hinders solving the 'Main Task Objective']"
        )
        return prompt

    def _parse_llm_response_for_prm_score(self, llm_response_text):
        """ Parses PRM-style score and justification from the LLM's response. """
        if not llm_response_text or llm_response_text.startswith("LLM Error") or llm_response_text.startswith("LLM not initialized"): # Check for specific error messages
            self.logger.warning(f"Invalid or error LLM response, cannot parse PRM score: {llm_response_text}")
            return 0.0, f"PRM scoring failed: Invalid LLM response ({llm_response_text})"

        score_match = re.search(r"Score:\s*([0-9.]+)", llm_response_text, re.IGNORECASE)
        # DOTALL makes '.' match newlines as well, to capture multi-line justifications
        justification_match = re.search(r"Justification:\s*(.+)", llm_response_text, re.IGNORECASE | re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.0
        justification = justification_match.group(1).strip() if justification_match else "No justification provided or parsing error."

        if not score_match: # Log if score specifically wasn't found
            self.logger.warning(f"Could not parse PRM score from response. Raw response: '{llm_response_text}'")
        return score, justification

    def _evaluate_rot_artifact_with_prm(self, artifact_content, artifact_type, main_task_description):
        """ Evaluates an intermediate artifact of ROT using the PRM evaluator. """
        if not self.prm_evaluator_llm: # Check if prm_evaluator_llm is set
            self.logger.error("PRM evaluator LLM not set. Cannot evaluate.")
            return 0.0, "PRM evaluator not set"
        
        prompt = self._generate_prm_style_scoring_prompt_for_rot_artifact(artifact_content, artifact_type, main_task_description)
        llm_response = self.prm_evaluator_llm.generate(prompt) # Assume evaluation LLM has a generate method
        score, justification = self._parse_llm_response_for_prm_score(llm_response)
        self.logger.info(f"ROT artifact ({artifact_type}) PRM evaluation - Score: {score:.2f}, Justification: {justification}")
        return score, justification

    def _prompt_for_reverse_reasoning(self, demonstrations_text):
        prompt = (
            "You are a distinguished expert in mathematics and informational reasoning.\n"
            "Based on the given examples, define the specific task, including:\n"
            "1. Task Definition: A clear description of the objective.\n"
            "2. Pseudocode: A step-by-step algorithm described in natural language.\n"
            "3. Logical Pseudocode: Convert the pseudocode into a formal logical representation using symbols (e.g., ∀, ∃, ∧, ∨, ¬, →, etc.). Provide specific examples if needed.\n"
            "4. Case Examples: Illustrative examples derived from the inputs.\n"
            "5. Input-Output Format: Precise specifications for inputs and outputs.\n\n"
            "Demonstrations:\n"
            f"{demonstrations_text}\n\n"
            "Your comprehensive definition (please ensure all five parts above are included):" # Emphasize completeness
        )
        return prompt

    def _prompt_for_pairwise_preference(self, response_A_text, response_B_text, main_task_description): # Added main_task_description
        prompt = (
            f"Main Task Objective: '{main_task_description}'\n\n"
            "Please compare the following two AI-generated 'Task Definition Prompts' (A and B) and choose which one you believe would more effectively guide another AI to solve the 'Main Task Objective' stated above.\n"
            "Evaluation criteria should include:\n"
            "-   **Clarity**: Is the prompt easy to understand?\n"
            "-   **Completeness**: Does the prompt include all necessary instructions and information to solve the main task?\n"
            "-   **Accuracy**: Do the definitions and steps in the prompt accurately reflect the requirements of the main task?\n"
            "-   **Potential Utility for Solving the Main Task**: Which prompt is more likely to lead an AI to a high-quality solution?\n\n"
            f"Prompt (A):\n\"\"\"\n{response_A_text}\n\"\"\"\n\n"
            f"Prompt (B):\n\"\"\"\n{response_B_text}\n\"\"\"\n\n"
            "Your choice (Please answer only A or B) and a brief justification (explaining why your chosen prompt is superior for guiding the solution of the 'Main Task Objective'):"
        )
        return prompt

    def preference_guided_reverse_reasoning_warmup(self, demonstrations, main_task_description_for_prm, warm_iterations=3):
        # main_task_description_for_prm is the "higher-level original task" for which ROT is trying to generate a good prompt.
        demo_text = ""
        for i, (inp, outp) in enumerate(demonstrations):
            demo_text += f"Example {i+1}:\nInput: {inp}\nOutput: {outp}\n\n"

        candidate_responses_info = [] # Stores {'text': ..., 'prob_score': ..., 'prm_score': ..., 'prm_justification': ..., 'id': ...}
        self.logger.info(f"\n--- ROT: Executing Reverse Reasoning Warmup ({warm_iterations} iterations) ---")
        for i in range(warm_iterations):
            rr_prompt = self._prompt_for_reverse_reasoning(demo_text)
            # Use a lower temperature for more diverse initial candidates
            response_text, response_prob_score = self.llm.generate_with_simulated_score(rr_prompt, temperature=0.8)

            if "LLM not initialized" in response_text or "LLM Error" in response_text: # Check for LLM errors
                self.logger.warning(f"ROT: Warmup iteration {i+1} failed due to LLM error or uninitialization. Response: {response_text}")
                continue
            
            # Perform PRM evaluation on the generated candidate prompt
            prm_score, prm_justification = self._evaluate_rot_artifact_with_prm(
                response_text, 
                "PGRR Candidate Task Definition Prompt", 
                main_task_description_for_prm # Pass the main task description
            )
            
            candidate_info = {
                'text': response_text, 
                'prob_score': response_prob_score, # Original simulated score
                'prm_score': prm_score,             # PRM evaluation score
                'prm_justification': prm_justification,
                'id': f"cand_{i}"
            }
            candidate_responses_info.append(candidate_info)
            self.logger.info(f"ROT: Warmup iteration {i+1} generated candidate prompt (Simulated Score: {response_prob_score:.3f}, PRM Score: {prm_score:.3f})")

        if not candidate_responses_info:
            self.logger.error("ROT: PGRR warmup failed to generate any candidate responses.")
            return "PGRR Warmup Failed: No candidate responses" # Return error message

        self.logger.info("\n--- ROT: Executing Pairwise Preference Evaluation (based on Main Task Objective) ---")
        preference_matrix = {} # To store pairwise preferences, e.g., preference_matrix[(id_A, id_B)] = 1.0 if A > B

        num_candidates = len(candidate_responses_info)
        if num_candidates < 2:
            self.logger.info("ROT: Fewer than 2 candidate responses, skipping pairwise preference evaluation.")
            # If only one candidate, its PRM score could be used for ranking if needed.
            # But here, PGRR aims to select one "LLM-Taste Prompt", so return its text directly.
            if candidate_responses_info: # Check if list is not empty
                best_candidate = candidate_responses_info[0]
                self.logger.info(f"ROT: Only one candidate (ID: {best_candidate['id']}), PRM Score: {best_candidate['prm_score']:.3f}, selecting directly.")
                return best_candidate['text']
            else: # Should have been caught by the earlier check, but as a safeguard
                self.logger.error("ROT: No candidates available even after attempting PGRR warmup.")
                return "PGRR Failed: No candidates after warmup"


        for i in range(num_candidates):
            for j in range(i + 1, num_candidates): # Avoid self-comparison and duplicate pairs
                resp_A_info = candidate_responses_info[i]
                resp_B_info = candidate_responses_info[j]
                
                # Include main_task_description_for_prm in the preference prompt
                pref_prompt_A_vs_B = self._prompt_for_pairwise_preference(
                    resp_A_info['text'], 
                    resp_B_info['text'],
                    main_task_description_for_prm # Key:让 LLM base its judgment on this
                )
                # Use a lower temperature for preference selection
                choice_response_A_vs_B = self.llm.generate(pref_prompt_A_vs_B, temperature=0.3) 

                if "LLM not initialized" in choice_response_A_vs_B or "LLM Error" in choice_response_A_vs_B: # Handle LLM errors
                    self.logger.warning(f"ROT: Preference evaluation ({resp_A_info['id']} vs {resp_B_info['id']}) failed. Response: {choice_response_A_vs_B}")
                    # Assign neutral preference on failure
                    preference_matrix[(resp_A_info['id'], resp_B_info['id'])] = 0.5 
                    preference_matrix[(resp_B_info['id'], resp_A_info['id'])] = 0.5
                    continue

                choice_upper = choice_response_A_vs_B.strip().upper() # Normalize to uppercase
                chosen_option = None
                if choice_upper.startswith("A"): chosen_option = "A"
                elif choice_upper.startswith("B"): chosen_option = "B"

                if chosen_option == "A":
                    preference_matrix[(resp_A_info['id'], resp_B_info['id'])] = 1.0 # A is preferred over B
                    preference_matrix[(resp_B_info['id'], resp_A_info['id'])] = 0.0 # B is not preferred over A
                elif chosen_option == "B":
                    preference_matrix[(resp_A_info['id'], resp_B_info['id'])] = 0.0 # A is not preferred over B
                    preference_matrix[(resp_B_info['id'], resp_A_info['id'])] = 1.0 # B is preferred over A
                else: # Tie or unable to determine
                    preference_matrix[(resp_A_info['id'], resp_B_info['id'])] = 0.5
                    preference_matrix[(resp_B_info['id'], resp_A_info['id'])] = 0.5
                
                winner_char = chosen_option if chosen_option else 'Tie/Unknown'
                self.logger.info(f"ROT: Preference: Candidate {resp_A_info['id']} vs {resp_B_info['id']} -> Winner: {winner_char}")
        
        # Combine PRM scores and pairwise preference scores to select the best prompt
        final_scores_for_candidates = []
        for i in range(num_candidates):
            current_candidate_id = candidate_responses_info[i]['id']
            current_candidate_prm_score = candidate_responses_info[i]['prm_score']
            
            total_pairwise_preference_score = 0
            num_comparisons = 0
            for j in range(num_candidates):
                if i == j: continue # Skip self-comparison
                other_candidate_id = candidate_responses_info[j]['id']
                # Get the preference score of current_candidate_id over other_candidate_id
                total_pairwise_preference_score += preference_matrix.get((current_candidate_id, other_candidate_id), 0.5) # Default to 0.5 if not found
                num_comparisons +=1
            
            # Average pairwise preference score for the current candidate
            avg_pairwise_preference_score = (total_pairwise_preference_score / num_comparisons) if num_comparisons > 0 else 0.5 # Default to 0.5 if no comparisons
            
            # Combine scores: can be a weighted average, here a simple average of PRM score and pairwise preference score
            # Weights can be adjusted, e.g., to give more importance to PRM scores
            combined_score = (current_candidate_prm_score * 0.6) + (avg_pairwise_preference_score * 0.4) # Example weights
            
            final_scores_for_candidates.append({
                'id': current_candidate_id, 
                'score': combined_score, 
                'text': candidate_responses_info[i]['text'],
                'prm_score': current_candidate_prm_score,
                'avg_pairwise_pref': avg_pairwise_preference_score
            })
            
        if not final_scores_for_candidates:
            self.logger.error("ROT: Failed to calculate final scores. PGRR ranking failed.")
            return "PGRR Ranking Failed: No final scores" # Return error message

        # Select the prompt with the highest combined score
        best_prompt_info = max(final_scores_for_candidates, key=lambda x: x['score'])
        self.logger.info(f"\n--- ROT: Best LLM-Taste Prompt (ID: {best_prompt_info['id']}, Combined Score: {best_prompt_info['score']:.3f}, PRM Score: {best_prompt_info['prm_score']:.3f}) ---")
        return best_prompt_info['text'] # Return the text content of the best prompt

    def _extract_task_definition(self, prompt_text):
        # This method remains unchanged, as it's for extracting specific parts from a prompt, not directly related to PRM.
        lines = str(prompt_text).splitlines()
        task_def_lines = []
        in_task_def_section = False
        # Define keywords in English for broader compatibility
        start_keywords = ["task definition:", "task definition :"] # Allow variations
        end_keywords = [
            "pseudocode:", "pseudocode :",
            "logical pseudocode:", "logical pseudocode :",
            "case examples:", "case examples :", "case example:", "case example :",
            "input-output format:", "input-output format :"
        ]

        for line in lines:
            stripped_line = line.strip()
            line_lower = stripped_line.lower() # Compare in lowercase
            
            if not in_task_def_section:
                for keyword in start_keywords:
                    if line_lower.startswith(keyword):
                        in_task_def_section = True
                        # Extract content after the keyword on the same line
                        content_after_keyword = stripped_line[len(keyword):].strip()
                        if content_after_keyword: # Add if there's actual content
                            task_def_lines.append(content_after_keyword)
                        break # Move to next line processing
            elif in_task_def_section:
                is_end_keyword_found = False
                for keyword in end_keywords:
                    if line_lower.startswith(keyword):
                        is_end_keyword_found = True
                        break
                if is_end_keyword_found or not stripped_line: # End if an end keyword is found or line is empty
                    break 
                task_def_lines.append(stripped_line)
        
        extracted_definition = "\n".join(task_def_lines).strip()
        
        if not extracted_definition: # 如果提取失敗，返回原始文本或一個標記
            self.logger.warning('ROT',f"未能從提示中提取明確的任務定義部分。將使用整個提示文本進行比較。提示片段：'{str(prompt_text)[:100]}...'")
            return str(prompt_text) 
        return extracted_definition

    def cognitive_preference_manager(self, original_task_prompt_text, llm_taste_prompt_text, main_task_description_for_prm):
        # main_task_description_for_prm is the "higher-level original task" for which ROT is trying to generate a good prompt.
        self.logger.info("\n--- ROT: Executing Cognitive Preference Manager (CPM) ---")
        
        # Evaluate the quality of the incoming llm_taste_prompt_text
        llm_taste_prm_score, llm_taste_prm_justification = self._evaluate_rot_artifact_with_prm(
            llm_taste_prompt_text, 
            "CPM Input LLM-Taste Prompt", # Artifact type
            main_task_description_for_prm
        )
        self.logger.info(f"ROT (CPM): Input LLM-Taste Prompt PRM Score: {llm_taste_prm_score:.3f}")

        # Based on PRM score, decide whether further optimization is needed or if it can be used directly.
        # For example, if PRM score is already high, complex comparison and fusion with the original prompt might not be necessary.
        # For demonstration here, we still perform similarity comparison and fusion, but PRM score-based early exit or strategy adjustment can be added.

        original_task_def_text = self._extract_task_definition(original_task_prompt_text)
        llm_taste_task_def_text = self._extract_task_definition(llm_taste_prompt_text)
        self.logger.info(f"original_task_def_text: {original_task_def_text} (llm_taste_task_def_text: {llm_taste_task_def_text})")

        similarity = self.embedder.calculate_similarity(original_task_def_text, llm_taste_task_def_text)
        self.logger.info(f"ROT: Similarity between original task definition and LLM-taste task definition: {similarity:.4f} (Threshold: {self.similarity_threshold})")

        final_prompt_text_candidate = ""
        instruction_prompt = "" # Initialize
        if similarity >= self.similarity_threshold:
            self.logger.info(f"ROT (CPM): Detected as known task (Similarity {similarity:.4f} >= {self.similarity_threshold}). Will attempt to aggregate merits of both prompts.")
            instruction_prompt = (
                "Please synthesize the following two descriptions/prompts for the same task, aiming to create a single, superior version of the prompt."
                "This new version should fuse the strongest points of both, especially in terms of clarity of task definition, utility of pseudocode, accuracy of logical representation, relevance of examples, and explicitness of input/output format."
                f"Ensure the final prompt is both comprehensive and easy for an LLM to understand and execute, in order to best accomplish the main task: '{main_task_description_for_prm}'.\n\n" # Added main task context
                f"Prompt 1 (Original or Baseline Prompt):\n{original_task_prompt_text}\n\n"
                f"Prompt 2 (LLM-generated Candidate Prompt):\n{llm_taste_prompt_text}\n\n"
                "Synthesized Best Prompt:"
            )
        else:
            self.logger.info(f"ROT (CPM): Detected as unknown or significantly different task (Similarity {similarity:.4f} < {self.similarity_threshold}). Will attempt to adapt style template to original task logic.")
            instruction_prompt = (
                "Below are two prompts. The 'LLM-generated Prompt Template' may not be entirely accurate in its task understanding, but its overall structure and style (e.g., how it organizes task definition, pseudocode, examples, etc.) are preferred."
                "The 'Original Correct Prompt' contains the core logic and correct intent of the task.\n"
                "Your task is: Use the core task definition and logic from the 'Original Correct Prompt' to adjust the 'LLM-generated Prompt Template'."
                f"The goal is to generate a new prompt that retains the good style and structure of the 'LLM-generated Prompt Template' while accurately expressing the task logic from the 'Original Correct Prompt', in order to best accomplish the main task: '{main_task_description_for_prm}'.\n\n" # Added main task context
                f"LLM-generated Prompt Template (Preferred style, but logic might be partially inaccurate):\n{llm_taste_prompt_text}\n\n"
                f"Original Correct Prompt (Core logic and intent are here):\n{original_task_prompt_text}\n\n"
                "Adjusted Final Prompt combining correct logic with preferred style:"
            )
        
        # Use moderate temperature for CPM generation
        final_prompt_text_candidate = self.llm.generate(instruction_prompt, temperature=0.5) 
        
        if "LLM not initialized" in final_prompt_text_candidate or "LLM Error" in final_prompt_text_candidate: # Check for LLM errors
            self.logger.error("ROT (CPM): LLM call failed. Cannot generate final prompt.")
            return f"CPM Failed: LLM Error ({final_prompt_text_candidate})" # Return error message
        
        # Perform PRM evaluation on the final prompt generated by CPM
        final_prm_score, final_prm_justification = self._evaluate_rot_artifact_with_prm(
            final_prompt_text_candidate,
            "CPM Final Optimized Prompt", # Artifact type
            main_task_description_for_prm
        )
        self.logger.info(f"ROT (CPM): Generated Final Prompt PRM Score: {final_prm_score:.3f}")
        
        # Logic can be added here: if final_prm_score is below a certain threshold,
        # one might try returning the un-CPM-modified llm_taste_prompt_text if its PRM score was higher,
        # or trigger a more complex repair/retry mechanism.
        # For simplicity, we return CPM's output here but log its PRM score.
        # In a real application, MASOrchestrator can check this score.
        
        return final_prompt_text_candidate # Return the CPM-generated prompt text

    def solve_task_with_final_prompt(self, final_prompt_text, problem_input):
        # Before solving the task, one could (or MASOrchestrator could) re-verify the PRM score of final_prompt_text.
        # Here, it's assumed final_prompt_text has been PRM-evaluated (possibly in the CPM stage) and is considered acceptable.
        
        # full_solving_prompt = f"{final_prompt_text}\n\nNow, based on the above definition and instructions, solve the following specific problem:\nInput: {problem_input}\nOutput:"
        full_solving_prompt = f"{final_prompt_text}\n\nNow, based on the above definition and instructions, solve the following specific problem:\nInput: {problem_input}\nOutput:Please provide your complete response (you may include reasoning, context, etc.), but ensure you include a clear, concrete answer to the problem."
        
        self.logger.info(f"ROT: Solving problem with final prompt: '{problem_input}'")
        # Use lower temperature for solving specific problems to aim for precision
        solution = self.llm.generate(full_solving_prompt, temperature=0.3) 

        if "LLM not initialized" in solution or "LLM Error" in solution: # Check for LLM errors
            self.logger.error(f"ROT: Failed to solve task '{problem_input}'. Response: {solution}")
            return f"Solution generation failed: {solution}" # Return error message
        return solution.strip()

# --- Encapsulate example usage into a function ---
def run_rot_standalone_example_with_prm(api_key_for_example):
    logger_example = DefaultLogger()
    logger_example.info("Running RoT (PRM-style) Standalone Example...")
    
    if not api_key_for_example: # Check for API key at the start
        logger_example.error("ROT Standalone Example: API key not provided. LLM and embedder calls in the example will fail.")
        # Optionally, one could still proceed to show structure, but LLM calls would fail.

    llm_api_rot = GeminiLLMInterface(api_key=api_key_for_example, logger=logger_example)
    embedder_api = GeminiEmbeddingInterface(api_key=api_key_for_example, logger=logger_example)
    
    # Critical checks for LLM and Embedder initialization
    if not llm_api_rot.model:
        logger_example.error("ROT Standalone Example: LLM interface model not initialized. Aborting example.")
        return
    if not embedder_api.api_key_configured_successfully:
        logger_example.error("ROT Standalone Example: Embedder interface API key not configured successfully. Aborting example.")
        return

    # The main_task_description here is the "meta-task" or "higher-level task" for the ROT system.
    # ROT's goal is to generate a good execution prompt for this main_task_description.
    main_task_for_rot_to_optimize_prompt_for = "Solve the 24-point game problem: use the four input numbers and arithmetic operations (add, subtract, multiply, divide) to get 24."

    rot_system = ReversalOfThought(llm_api_rot, embedder_api, similarity_threshold=0.6, logger=logger_example) # Use a threshold

    # Examples for the 24-point game
    demonstrations_24 = [
       ("1 3 7 10", "For 1,3,7,10, a possible solution is (10-7)*(1+3) but this equals 12, not 24. This input combination cannot easily yield 24. Let's use an example known to have a solution."), # Self-correction in example
       ("3 3 8 8", "8 / (3 - 8/3) = 24") 
    ]
    # This is the "ideal" or "original" task prompt for the 24-point game, provided by a user (or another system)
    original_user_prompt_for_24_game = (
       "Task Definition: Using the four provided integers (order can be changed, each number must be used once) and the operations of addition, subtraction, multiplication, division, and parentheses, construct a mathematical expression that results in 24.\n"
       "Pseudocode: 1. Generate all permutations of the numbers. 2. For each permutation, try all possible combinations of operators and parentheses. 3. Evaluate the expression; if it equals 24, return the expression.\n"
       "Logical Pseudocode: ∀ P(perm(a,b,c,d)) ∃ Ops(op1,op2,op3) ∃ Grouping(g1,g2) such that Evaluate(Expression(P, Ops, Grouping)) = 24 → Print(Expression)\n"
       "Case Example: Input: 1 2 3 4  Output: (1+3)*(2+4) = 24\n"
       "Input-Output Format: Input: 'w x y z' (four space-separated numbers) Output: 'mathematical expression = 24' or 'No solution'"
    )

    logger_example.info("\n--- ROT Standalone Example: Starting PGRR Warmup (with PRM evaluation) ---")
    # Pass main_task_for_rot_to_optimize_prompt_for to PGRR for internal evaluation of candidate prompts
    llm_taste_prompt_text = rot_system.preference_guided_reverse_reasoning_warmup(
        demonstrations_24, 
        main_task_description_for_prm=main_task_for_rot_to_optimize_prompt_for, # Key parameter
        warm_iterations=2 # Reduce iterations to save time for example
    )

    # Check if llm_taste_prompt_text is valid before proceeding
    if llm_taste_prompt_text and "failed" not in str(llm_taste_prompt_text).lower() and "llm not initialized" not in str(llm_taste_prompt_text).lower() and "llm error" not in str(llm_taste_prompt_text).lower():
        logger_example.info(f"\n--- ROT Standalone Example: LLM-Taste Prompt selected by PGRR (partial) ---\n{str(llm_taste_prompt_text)[:500]}...")
        
        logger_example.info("\n--- ROT Standalone Example: Starting CPM (with PRM evaluation) ---")
        # Pass main_task_for_rot_to_optimize_prompt_for to CPM
        final_optimal_prompt_text = rot_system.cognitive_preference_manager(
            original_user_prompt_for_24_game, 
            str(llm_taste_prompt_text), # Ensure it's a string
            main_task_description_for_prm=main_task_for_rot_to_optimize_prompt_for # Key parameter
        )
        
        # Check if final_optimal_prompt_text is valid
        if final_optimal_prompt_text and "failed" not in str(final_optimal_prompt_text).lower() and "llm not initialized" not in str(final_optimal_prompt_text).lower() and "llm error" not in str(final_optimal_prompt_text).lower():
            logger_example.info(f"\n--- ROT Standalone Example: Final Optimized Prompt after CPM (partial) ---\n{str(final_optimal_prompt_text)[:500]}...")
            
            # Here, MASOrchestrator could re-check the PRM score of final_optimal_prompt_text
            # (if _evaluate_rot_artifact_with_prm was not called as the last step inside CPM, or for independent verification).

            problem_instance_24 = "4 6 8 8" # Change to an example with a solution
            logger_example.info(f"\n--- ROT Standalone Example: Attempting to solve problem instance with RoT-generated prompt: '{problem_instance_24}' ---")
            solution = rot_system.solve_task_with_final_prompt(str(final_optimal_prompt_text), problem_instance_24) # Ensure string
            logger_example.info(f"\n--- ROT Standalone Example: Solution for '{problem_instance_24}' ---\n{solution}")
        else:
            logger_example.error(f"ROT Standalone Example: CPM phase failed, cannot proceed to solve task. CPM Output: {final_optimal_prompt_text}")
    else:
       logger_example.error(f"ROT Standalone Example: Failed to generate LLM-taste prompt. PGRR phase might have failed. PGRR Output: {llm_taste_prompt_text}")
    logger_example.info("\n--- End of ROT (PRM-style) Standalone Example ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    logger_main = DefaultLogger() # Use default logger for main block
    logger_main.info("ROT.py executed as a standalone script...")
    api_key_for_standalone = os.getenv("GEMINI_API_KEY") # Get API key from environment
    
    if not api_key_for_standalone:
        logger_main.warning("Warning: GEMINI_API_KEY not found in environment variables.")
        logger_main.warning("ROT.py standalone example requires a valid API key to interact with the Gemini API.")
        run_rot_standalone_example_with_prm(None) # Pass None, let initialization fail and print errors
    else:
        logger_main.info(f"ROT.py standalone execution: Detected API key ending ...{api_key_for_standalone[-4:]}, will be used for the example.")
        run_rot_standalone_example_with_prm(api_key_for_standalone)
    
    logger_main.info("ROT.py standalone example execution finished.")