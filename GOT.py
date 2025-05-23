import google.generativeai as genai
import os
import re # Used for parsing multiple thoughts

class Thought:
    def __init__(self, content, id, score=0.0, generated_by_llm=True, prm_justification="Not yet evaluated"): # Added prm_justification
        self.id = id
        self.content = content # Content
        self.score = score # Score (this score will primarily be determined by PRM-style evaluation)
        self.prm_justification = prm_justification # Justification for PRM evaluation
        self.children = []  # Subsequent thoughts that depend on this thought
        self.parents = []   # Preceding thoughts on which this thought depends
        self.generated_by_llm = generated_by_llm # Is the content of this thought from LLM or initial input?

    def __repr__(self):
        # Replace newlines with spaces for single-line display
        display_content = self.content.replace('\n', ' ')
        return f"Thought(id={self.id}, score={self.score:.2f}, generated_by_llm={self.generated_by_llm}, content='{display_content}...')"

class GeminiLLM:
    # def __init__(self, api_key, model_name="gemini-1.5-flash-latest"): # 更新模型名稱以符合常見用法
    # def __init__(self, api_key, model_name="gemini-2.0-flash"): # 更新模型名稱以符合常見用法
    def __init__(self, api_key, model_name="gemini-2.0-flash-lite"): # 更新模型名稱以符合常見用法
        """
        Initializes the Gemini LLM interface.
        Args:
            api_key (str): Your Google Generative AI API key.
            model_name (str): The Gemini model to use (e.g., "gemini-pro", "gemini-1.5-flash-latest").
        """
        if not api_key: # Check if the key is empty
            raise ValueError("A valid Gemini API key must be provided.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"GeminiLLM initialized with model: {model_name}")

    def generate(self, prompt, temperature=0.7, safety_settings=None):
        """
        Generates content using the Gemini API.
        Args:
            prompt (str): The prompt to send to the LLM.
            temperature (float): Controls randomness; lower values are more deterministic.
            safety_settings (list of dict, optional): Custom safety settings.
        Returns:
            str: The LLM's response text.
        """
        print(f"\n--- Sending prompt to Gemini ---\n{prompt}...\n--- End of Gemini prompt ---") # Shorten printed prompt length
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
                print(f"Warning: Prompt blocked due to {block_reason_str}. Safety ratings: {response.prompt_feedback.safety_ratings}")
                return f"Error: Prompt blocked due to {block_reason_str}."

            print(f"--- Received Gemini response ---\n{llm_response_text}...\n--- End of Gemini response ---") # Shorten printed response length
            return llm_response_text if llm_response_text else "Error: No content generated or issue with the prompt."
        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            return f"Error: An error occurred while generating content: {str(e)}"


class GraphOfThoughts:
    def __init__(self, llm_interface, logger=None): # Added logger
        self.thoughts = {} # Stores all thoughts by id
        self.next_thought_id = 0
        self.llm = llm_interface
        self.next_id = 0
        self.logger = logger or self._default_logger()

    def _get_default_logger(self):
        # A simple default logger if none is provided externally
        class PrintLogger:
            def info(self, message): print(f"[GOT INFO] {message}")
            def warning(self, message): print(f"[GOT WARNING] {message}")
            def error(self, message): print(f"[GOT ERROR] {message}")
        return PrintLogger()
    def _default_logger(self):
        class L:
            def info(s, m): print(f"[GOT INFO] {m}")
            def warning(s, m): print(f"[GOT WARN] {m}")
            def error(s, m): print(f"[GOT ERR] {m}")
        return L()

    def _new_id(self):
        i = self.next_id
        self.next_id += 1
        return i

    def add_thought(self, content, parent_ids=None):
        tid = self._new_id()
        t = Thought(content, tid)
        self.thoughts[tid] = t
        if parent_ids:
            for pid in parent_ids:
                if pid in self.thoughts:
                    self.thoughts[pid].children.append(t)
                    t.parents.append(self.thoughts[pid])
        self.logger.info(f"Added {t}")
        return t

    def _prompt_new(self, task, existing=None, n=1):
        p = f"Task: {task}\n"
        if existing:
            for i,c in enumerate(existing): p += f"Prev{i+1}: {c}\n"
        p += f"Generate {n} new thought(s) to advance the task."
        return p

    def _get_new_id(self):
        new_id = self.next_thought_id
        self.next_thought_id += 1
        return new_id

    # def add_thought(self, content, parent_ids=None, score=0.0, generated_by_llm=True, prm_justification="Not yet evaluated"):
    #     new_id = self._get_new_id()
    #     thought = Thought(content, new_id, score, generated_by_llm, prm_justification)
    #     self.thoughts[new_id] = thought
    #     if parent_ids:
    #         for pid in parent_ids:
    #             if pid in self.thoughts:
    #                 self.thoughts[pid].children.append(thought)
    #                 thought.parents.append(self.thoughts[pid])
    #     self.logger.info(f"Added thought: {thought}")
    #     return thought
    def generate_thoughts(self, task_description, num=1, from_ids=None):
        base = [self.thoughts[i].content for i in (from_ids or []) if i in self.thoughts]
        prompt = self._prompt_new(task_description, base, num)
        resp = self.llm.generate(prompt)
        parts = re.split(r'\n\s*(?=\d+\.)', resp) if num>1 else [resp]
        thoughts = []
        for txt in parts[:num]:
            clean = re.sub(r'^\s*(\d+\.)','',txt).strip()
            if clean:
                thoughts.append(self.add_thought(clean, parent_ids=from_ids))
        return thoughts

    def refine_thought(self, thought_id, task_description, instruction):
        if thought_id not in self.thoughts:
            self.logger.error(f"Refine: id {thought_id} missing")
            return None
        orig = self.thoughts[thought_id].content
        prompt = f"Task: {task_description}\nOriginal: {orig}\nInstruction: {instruction}\nRefine the thought." 
        resp = self.llm.generate(prompt)
        return self.add_thought(resp, parent_ids=[thought_id])

    def aggregate_thoughts(self, ids, task_description):
        if not ids or any(i not in self.thoughts for i in ids):
            self.logger.error("GOT","Aggregate: invalid ids")
            return None
        contents = [self.thoughts[i].content for i in ids]
        prompt = f"Task: {task_description}\nCombine:\n" + "\n".join(contents)
        resp = self.llm.generate(prompt)
        return self.add_thought(resp, parent_ids=ids)
    def print_graph(self):
        self.logger.info("Current Graph:")
        for tid, t in sorted(self.thoughts.items()):
            cnt = t.content[:60].replace('\n',' ')
            pids = [p.id for p in t.parents]
            cids = [c.id for c in t.children]
            self.logger.info(f"ID {tid}: {cnt} | parents={pids} children={cids}")
    # --- Prompter Module ---
    def _generate_prompt_for_new_thought(self, task_description, existing_thoughts_content=None, num_new_thoughts=1):
        prompt = f"Main task: {task_description}\n"
        if existing_thoughts_content:
            prompt += "Based on the following existing information/thoughts:\n"
            for i, content in enumerate(existing_thoughts_content):
                prompt += f"Existing thought {i+1}: {content}\n"
        prompt += f"\nPlease generate {num_new_thoughts} new, distinct thoughts or solution steps to advance the main task. Clearly provide each new thought."
        if num_new_thoughts > 1:
            prompt += "\nPlease start each new thought on a new line, optionally numbered (e.g., '1. <Thought content>')."
        return prompt

    def _generate_prompt_for_aggregation(self, thoughts_to_aggregate_content, task_description):
        prompt = f"Main task: {task_description}\n"
        prompt += "Please aggregate the following distinct thoughts into a single, more comprehensive and refined thought. Identify the core ideas and synthesize them into a coherent summary or an improved solution to better accomplish the main task:\n"
        for i, content in enumerate(thoughts_to_aggregate_content):
            prompt += f"待聚合思維 {i+1}：{content}\n"
        prompt += "\n合併與聚合後的思維(旨在推進主要任務)，並解決問題和給出答案："
        return prompt

    def _generate_prompt_for_refinement(self, thought_content, task_description, refinement_instruction):
        prompt = f"Main task: {task_description}\n"
        prompt += f"Original thought: {thought_content}\n"
        prompt += f"Refinement instruction: {refinement_instruction}\n"
        prompt += "\nPlease provide the refined thought based solely on the original thought and the instruction, making it more helpful for completing the main task:"
        return prompt

    def _generate_prm_style_scoring_prompt(self, thought_content, main_task_description):
        """
        Generates a PRM-style scoring prompt.
        Evaluate the potential contribution of this thought towards completing the `main_task_description`.
        """
        prompt = (
            f"You are an expert evaluator tasked with assessing the quality and potential of a 'Thought'.\n"
            f"Main Task Objective: '{main_task_description}'\n\n"
            f"Thought Content to Evaluate:\n\"\"\"\n{thought_content}\n\"\"\"\n\n"
            "Evaluation Instructions:\n"
            "1.  Relevance: How relevant is this thought to the main task objective?\n"
            "2.  Potential/Contribution: How likely is this thought to lead us one step closer to a successful solution for the main task objective? Does it open promising avenues, or is it a dead-end/low-quality idea?\n"
            "3.  Clarity and Feasibility (if applicable): Is the thought itself clear? If it's an action or plan, does it have initial feasibility?\n\n"
            "Please provide an overall score and a brief justification.\n"
            "Output Format (Strictly Adhere):\n"
            "Score: [A floating-point number between 0.0 (very poor/unhelpful) and 1.0 (excellent/highly promising)]\n"
            "Justification: [A brief explanation for your score, stating how and why it contributes or doesn't contribute to the main task objective]"
        )
        return prompt

    # --- Parser Module ---
    def _parse_llm_response_for_new_thoughts(self, llm_response_text, num_expected_thoughts=1):
        if not llm_response_text or llm_response_text.startswith("Error:"):
            self.logger.warning(f"Invalid or error LLM response, cannot parse new thoughts: {llm_response_text}")
            return []

        if num_expected_thoughts == 1:
            return [llm_response_text.strip()]
        else:
            # Try to split by numbered or bulleted list format first
            potential_thoughts = re.split(r'\n\s*(?=\d+\.\s|-\s|\*\s)', llm_response_text)
            cleaned_thoughts = []
            for pt in potential_thoughts:
                # Remove the numbering/bullet part
                cleaned = re.sub(r'^\s*(\d+\.\s*|-\s*|\*\s*)', '', pt.strip())
                if cleaned: # Only add if there's content after stripping the prefix
                    cleaned_thoughts.append(cleaned)
            
            # If we found thoughts this way and it's enough, use them
            if cleaned_thoughts and len(cleaned_thoughts) >= num_expected_thoughts:
                return cleaned_thoughts[:num_expected_thoughts]
            
            # Fallback to splitting by newline if regex fails or yields too few
            thoughts = llm_response_text.strip().split('\n\n') # Prefer double newline split
            if len(thoughts) < num_expected_thoughts:
                thoughts = llm_response_text.strip().split('\n') # Fallback to single newline
            return [t.strip() for t in thoughts if t.strip()][:num_expected_thoughts]


    def _parse_llm_response_for_prm_score(self, llm_response_text):
        """
        Parses PRM-style score and justification from the LLM's response.
        """
        if not llm_response_text or llm_response_text.startswith("Error:"):
            self.logger.warning(f"Invalid or error LLM response, cannot parse PRM score: {llm_response_text}")
            return 0.0, f"PRM scoring failed: Invalid LLM response ({llm_response_text})"

        score_match = re.search(r"Score:\s*([0-9.]+)", llm_response_text, re.IGNORECASE)
        # Make justification regex non-greedy and handle multiline
        justification_match = re.search(r"Justification:\s*(.+)", llm_response_text, re.IGNORECASE | re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.0
        justification = justification_match.group(1).strip() if justification_match else "No justification provided or parsing error."

        if not score_match: # Log if score specifically wasn't found
            self.logger.warning(f"Could not parse PRM score from response. Raw response: '{llm_response_text}'")
        return score, justification

    # --- Controller and Graph Operations ---
    def generate_and_evaluate_thoughts(self, task_description, num_thoughts=1, from_thought_ids=None, initial_content_list=None):
        """
        Generates thoughts and immediately scores them using a PRM-style evaluator.
        """
        generated_llm_thoughts_objects = []
        parent_ids_for_new_thoughts = []
        base_content_for_prompt = []

        if from_thought_ids:
            parent_ids_for_new_thoughts.extend(from_thought_ids)
            for tid in from_thought_ids:
                if tid in self.thoughts:
                    base_content_for_prompt.append(self.thoughts[tid].content)
                else:
                    self.logger.warning(f"When generating new thoughts, from_thought_id {tid} not found.")
        elif initial_content_list: # If not from existing thoughts, but from initial content
             base_content_for_prompt.extend(initial_content_list)

        prompt = self._generate_prompt_for_new_thought(task_description, base_content_for_prompt, num_new_thoughts=num_thoughts)
        llm_response = self.llm.generate(prompt)
        parsed_contents = self._parse_llm_response_for_new_thoughts(llm_response, num_thoughts)
        
        for content in parsed_contents:
            if not content or content.startswith("Error:"): # Check for empty or error content
                self.logger.warning(f"Skipping thought generation due to error or empty content: '{content}'")
                continue
            
            # Add thought without immediate scoring, create object first
            thought_obj = self.add_thought(content, parent_ids=parent_ids_for_new_thoughts, generated_by_llm=True)
            
            # Perform PRM-style scoring for the newly generated thought
            prm_score, prm_justification = self.score_thought_with_prm_evaluator(thought_obj.id, task_description) # task_description here is the overall task
            thought_obj.score = prm_score # Update score
            thought_obj.prm_justification = prm_justification # Update justification
            self.logger.info(f"New thought {thought_obj.id} evaluated - PRM Score: {prm_score:.2f}, Justification: {prm_justification}")
            
            generated_llm_thoughts_objects.append(thought_obj)
        return generated_llm_thoughts_objects

    def aggregate_and_evaluate_thoughts(self, thought_ids_to_aggregate, task_description):
        """
        Aggregates thoughts and performs PRM-style scoring on the new aggregated thought.
        """
        if not thought_ids_to_aggregate or not all(tid in self.thoughts for tid in thought_ids_to_aggregate):
            self.logger.error("GOT","One or more thought IDs for aggregation not found or list is empty.")
            return None

        contents = [self.thoughts[tid].content for tid in thought_ids_to_aggregate]
        prompt = self._generate_prompt_for_aggregation(contents, task_description)
        llm_response = self.llm.generate(prompt)

        if not llm_response or llm_response.startswith("Error:"): # Check for empty or error response
            self.logger.error("GOT",f"Aggregation failed due to error or empty response: '{llm_response}'")
            return None
        
        aggregated_content = llm_response.strip()
        # Create the new aggregated thought, linking it to its parents
        aggregated_thought_obj = self.add_thought(aggregated_content, parent_ids=thought_ids_to_aggregate, generated_by_llm=True)
        
        # Perform PRM-style scoring for the aggregated thought
        prm_score, prm_justification = self.score_thought_with_prm_evaluator(aggregated_thought_obj.id, task_description)
        aggregated_thought_obj.score = prm_score
        aggregated_thought_obj.prm_justification = prm_justification
        self.logger.info(f"Aggregated thought {aggregated_thought_obj.id} evaluated - PRM Score: {prm_score:.2f}, Justification: {prm_justification}")
        
        return aggregated_thought_obj
    # def aggregate_thoughts(self, thought_ids_to_aggregate, task_description):
        """
        聚合思維，並對聚合後的新思維進行 PRM 風格評分。
        """
        if not thought_ids_to_aggregate or not all(tid in self.thoughts for tid in thought_ids_to_aggregate):
            self.logger.error("GOT","一個或多個用於聚合的思維 ID 未找到或列表為空。")
            return None

        contents = [self.thoughts[tid].content for tid in thought_ids_to_aggregate]
        prompt = self._generate_prompt_for_aggregation(contents, task_description)
        llm_response = self.llm.generate(prompt)

        if not llm_response or llm_response.startswith("錯誤："):
            self.logger.error("GOT",f"聚合失敗，原因：錯誤或空回應：'{llm_response}'")
            return None
        
        aggregated_content = llm_response.strip()
        # aggregated_thought_obj = self.add_thought(aggregated_content, parent_ids=thought_ids_to_aggregate, generated_by_llm=True)
        aggregated_thought_obj = aggregated_content
        return aggregated_thought_obj

    def refine_and_evaluate_thought(self, thought_id, task_description, refinement_instruction="Improve clarity and detail, making it closer to the main task objective."):
        """
        Refines a thought and performs PRM-style scoring on the new refined thought.
        """
        if thought_id not in self.thoughts:
            self.logger.error("GOT",f"Error: Thought ID {thought_id} for refinement not found.")
            return None

        content_to_refine = self.thoughts[thought_id].content
        prompt = self._generate_prompt_for_refinement(content_to_refine, task_description, refinement_instruction)
        llm_response = self.llm.generate(prompt)

        if not llm_response or llm_response.startswith("Error:"): # Check for empty or error response
            self.logger.error("GOT",f"Refinement failed due to error or empty response: '{llm_response}'")
            return None
        
        refined_content = llm_response.strip()
        # Create the new refined thought, linking it to its parent
        refined_thought_obj = self.add_thought(refined_content, parent_ids=[thought_id], generated_by_llm=True)
        
        # Perform PRM-style scoring for the refined thought
        prm_score, prm_justification = self.score_thought_with_prm_evaluator(refined_thought_obj.id, task_description)
        refined_thought_obj.score = prm_score
        refined_thought_obj.prm_justification = prm_justification
        self.logger.info(f"Refined thought {refined_thought_obj.id} evaluated - PRM Score: {prm_score:.2f}, Justification: {prm_justification}")

        return refined_thought_obj

    def score_thought_with_prm_evaluator(self, thought_id, main_task_description):
        """
        Scores a single thought using the LLM as a PRM evaluator.
        The core of this method is _generate_prm_style_scoring_prompt.
        """
        if thought_id not in self.thoughts:
            self.logger.error("GOT",f"Error: Thought ID {thought_id} for PRM evaluation not found.")
            return 0.0, "Thought not found"

        thought_content = self.thoughts[thought_id].content
        # Use PRM-style scoring prompt
        prompt = self._generate_prm_style_scoring_prompt(thought_content, main_task_description)
        llm_response = self.llm.generate(prompt)

        score, justification = self._parse_llm_response_for_prm_score(llm_response) # Use new parser
        
        # Update score and justification in the thought object (can be called separately after add_thought if needed)
        # self.thoughts[thought_id].score = score 
        # self.thoughts[thought_id].prm_justification = justification
        # self.logger.info(f"Thought {thought_id} scored with PRM evaluator: {score:.2f}. Justification: {justification}")
        return score, justification


    def rank_thoughts(self, top_n=None):
        """Ranks thoughts based on score (now PRM-style score)."""
        if not self.thoughts: return []
        # Sort by score (descending), then by ID (ascending as a tie-breaker, though less critical now)
        sorted_thoughts = sorted(self.thoughts.values(), key=lambda t: (t.score, -t.id), reverse=True)
        return sorted_thoughts[:top_n] if top_n is not None else sorted_thoughts

    def get_thought(self, thought_id):
        return self.thoughts.get(thought_id)

    # def print_graph(self):
    #     self.logger.info("\n--- Current Thought Graph ---")
    #     if not self.thoughts:
    #         self.logger.info("Thought graph is empty.")
    #         return
    #     for thought_id in sorted(self.thoughts.keys()): # Iterate in sorted order for consistent output
    #         thought = self.thoughts[thought_id]
    #         display_content = thought.content[:100].replace('\n', ' ').strip() # Limit length and remove newlines for display
    #         self.logger.info(f"\nThought ID: {thought.id} (PRM Score: {thought.score:.2f}, LLM Generated: {thought.generated_by_llm})")
    #         self.logger.info(f"  Content: '{display_content}...'")
    #         self.logger.info(f"  PRM Justification: {thought.prm_justification}")
    #         parent_ids = [p.id for p in thought.parents]
    #         children_ids = [c.id for c in thought.children]
    #         self.logger.info(f"  Parent Thought IDs: {parent_ids if parent_ids else 'None'}")
    #         self.logger.info(f"  Child Thought IDs: {children_ids if children_ids else 'None'}")
    #     self.logger.info("--- End of Thought Graph ---")

# --- Encapsulate example usage into a function ---
def run_got_example_workflow_with_prm_scoring(api_key):
    """
    Executes an example workflow of GraphOfThoughts, integrating PRM-style scoring.
    """
    # Create a simple logger instance for the GOT system
    class SimpleLogger:
        def info(self, msg): print(f"[GOT WORKFLOW INFO] {msg}")
        def warning(self, msg): print(f"[GOT WORKFLOW WARNING] {msg}")
        def error(self, msg): print(f"[GOT WORKFLOW ERROR] {msg}")
    
    logger = SimpleLogger()
    logger.info(f"Running GOT (PRM-style scoring) example with API key: ...{api_key[-4:]}")

    try:
        llm_api = GeminiLLM(api_key=api_key)
    except ValueError as e: # Catch specific API key error
        logger.error(f"Initialization error: {e}")
        logger.error("Example ended. Please provide a valid Gemini API key.")
        return
    except Exception as e: # Catch any other LLM init errors
        logger.error(f"Unexpected error during LLM initialization: {e}")
        return

    got_system = GraphOfThoughts(llm_api, logger=logger)

    main_task = "Develop a unique loyalty program for a small local coffee shop to increase customer retention and daily sales."
    logger.info(f"\nMain Task: {main_task}\n")

    # 1. Initial Brainstorming: Generate initial loyalty program concepts and immediately evaluate with PRM-style
    logger.info("\nStep 1: Initial Brainstorming & PRM Evaluation (Generate 2 initial concepts)")
    initial_ideas = got_system.generate_and_evaluate_thoughts( # Use new integrated method
        task_description="Generate distinct core concepts for a coffee shop loyalty program. " + main_task, # Ensure task_description includes the main task
        num_thoughts=2
    )
    if not initial_ideas or len(initial_ideas) < 1: # Continue even if only one is generated
        logger.error("Failed to generate enough initial concepts. Ending example.")
        # return # Try to continue even if only one is generated

    # Print initial ideas and their PRM scores
    for idea in initial_ideas:
        logger.info(f"Initial Concept ID {idea.id}: '{idea.content[:50]}...' (PRM Score: {idea.score:.2f}, Justification: {idea.prm_justification})")

    # 2. Select the best initial concept and elaborate on it, then PRM evaluate the elaboration
    best_initial_idea = None
    if initial_ideas: # Check if initial_ideas is not empty
        # Select the best initial concept based on PRM score
        initial_ideas.sort(key=lambda t: t.score, reverse=True)
        best_initial_idea = initial_ideas[0]
        logger.info(f"\nStep 2: Selecting initial concept with highest PRM score (ID: {best_initial_idea.id}) for elaboration")
        
        # Here we directly use generate_and_evaluate_thoughts to elaborate and evaluate
        # Note: from_thought_ids should pass a list of IDs
        elaborated_thoughts = got_system.generate_and_evaluate_thoughts(
            task_description=f"Elaborate on the following loyalty program concept '{best_initial_idea.content[:50]}...' to make it more concrete and actionable. " + main_task,
            num_thoughts=1,
            from_thought_ids=[best_initial_idea.id] 
        )
    else:
        logger.warning("No initial concepts to select from. Skipping elaboration step.")
        elaborated_thoughts = [] # Ensure elaborated_thoughts is defined


    elaborated_idea = None
    if elaborated_thoughts: # Check if elaborated_thoughts is not empty
        elaborated_idea = elaborated_thoughts[0]
        logger.info(f"Elaborated Concept ID {elaborated_idea.id}: '{elaborated_idea.content[:50]}...' (PRM Score: {elaborated_idea.score:.2f}, Justification: {elaborated_idea.prm_justification})")

        # 3. Refine the elaborated concept and PRM evaluate the refined result
        logger.info(f"\nStep 3: Refining elaborated concept (ID: {elaborated_idea.id})")
        refined_thought_obj = got_system.refine_and_evaluate_thought( # Use new integrated method
            elaborated_idea.id,
            task_description="Refine the loyalty program to make it more engaging and cost-effective. " + main_task,
            refinement_instruction="Add a unique, low-cost but high perceived value reward element, and consider how to track member progress."
        )
        if refined_thought_obj:
            logger.info(f"Refined Concept ID {refined_thought_obj.id}: '{refined_thought_obj.content[:50]}...' (PRM Score: {refined_thought_obj.score:.2f}, Justification: {refined_thought_obj.prm_justification})")
    else:
        logger.warning("Skipping refinement step as concept elaboration was not successful.")
        refined_thought_obj = None # Ensure refined_thought_obj is defined

    # 4. Attempt Aggregation (if multiple high-quality ideas exist)
    # The aggregation logic here can be more complex, e.g., aggregating best ideas from different branches
    # For simplicity, let's assume aggregating best_initial_idea (if different from elaborated_idea's parent) and refined_thought_obj
    thoughts_for_aggregation = []
    if best_initial_idea and refined_thought_obj and refined_thought_obj.parents and best_initial_idea.id != refined_thought_obj.parents[0].id : # Ensure refined_thought_obj has parents
         thoughts_for_aggregation.append(best_initial_idea.id)
         thoughts_for_aggregation.append(refined_thought_obj.id)
    elif refined_thought_obj: # If only the refined idea exists
        thoughts_for_aggregation.append(refined_thought_obj.id)
        # Try to add a second initial idea if available and different from refined's parent
        if initial_ideas and len(initial_ideas) > 1 and refined_thought_obj.parents and initial_ideas[1].id != refined_thought_obj.parents[0].id:
             thoughts_for_aggregation.append(initial_ideas[1].id)


    if len(thoughts_for_aggregation) >= 2:
        logger.info(f"\nStep 4: Aggregating thoughts {thoughts_for_aggregation}")
        aggregated_thought = got_system.aggregate_and_evaluate_thoughts( # Use new integrated method
            thoughts_for_aggregation,
            task_description="Combine the following aspects of loyalty programs into a unified and robust final proposal. " + main_task
        )
        if aggregated_thought:
            logger.info(f"Aggregated Concept ID {aggregated_thought.id}: '{aggregated_thought.content[:50]}...' (PRM Score: {aggregated_thought.score:.2f}, Justification: {aggregated_thought.prm_justification})")
    else:
        logger.info("Not enough distinct high-quality thoughts for aggregation. Skipping aggregation step.")


    # 5. Final ranking and display of the thought graph
    got_system.print_graph()

    logger.info("\n--- Final Ranking of All Thoughts (Based on PRM-style Scoring) ---")
    top_thoughts = got_system.rank_thoughts()
    for i, thought_instance in enumerate(top_thoughts):
        display_content = thought_instance.content[:80].replace('\n', ' ').strip()
        logger.info(f"{i+1}. (ID: {thought_instance.id}, PRM Score: {thought_instance.score:.2f}) - {display_content}...")
        logger.info(f"    Justification: {thought_instance.prm_justification}")


    logger.info("\n--- End of GOT (PRM-style scoring) Example Usage ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        print("Please set the GEMINI_API_KEY environment variable or provide the API key in GOT.py to run the example.")
    else:
        run_got_example_workflow_with_prm_scoring(GEMINI_API_KEY)