# -*- coding: utf-8 -*-
import google.generativeai as genai
import os
import re # Added re module

# --- Helper: Default Logger ---
class DefaultLogger:
    def info(self, message): print(f"[LOT INFO] {message}")
    def warning(self, message): print(f"[LOT WARNING] {message}")
    def error(self, message): print(f"[LOT ERROR] {message}")

# --- Necessary Helper Classes ---
class Thought:
    def __init__(self, content, id, score=0.0, prm_justification="Not yet evaluated"): # Standardized score to float, added prm_justification
        self.id = id
        self.content = content
        self.score = score # This score will primarily be determined by PRM-style evaluation
        self.prm_justification = prm_justification
        self.parents = []
        self.children = []

    def __repr__(self):
        return f"Thought(id={self.id}, score={self.score:.2f}, content='{self.content[:50]}...')"

class GraphOfThoughts: # GraphOfThoughts in LOT is a base, primarily inherited and used by LayerOfThoughts
    def __init__(self, llm_interface, logger=None): # Added logger
        self.thoughts = {}
        self.llm = llm_interface # This llm is mainly for LOT internal operations; PRM evaluation might use a different LLM
        self.logger = logger if logger else DefaultLogger()
        # PRM evaluator LLM, defaults to the operational LLM, but can be specified externally
        self.prm_evaluator_llm = llm_interface

    def add_thought_object(self, thought_obj): # Allows direct addition of Thought object instances
        if thought_obj.id in self.thoughts:
            self.logger.warning(f"Thought with ID '{thought_obj.id}' already exists. Will not add again.")
            return self.thoughts[thought_obj.id]
        self.thoughts[thought_obj.id] = thought_obj
        # Parent-child relationships should be handled by the logic that creates Thought objects
        return thought_obj

    def get_thought(self, thought_id):
        return self.thoughts.get(thought_id)

    def _generate_prm_style_scoring_prompt_for_lot_artifact(self, artifact_content, artifact_type, layer_conceptual_step, main_task_description):
        """
        Generates a PRM-style scoring prompt for intermediate artifacts produced by LOT (e.g., option thoughts, layer aggregation outputs).
        artifact_type: "Option Thought" or "Layer Aggregation Output"
        layer_conceptual_step: Description of the current layer's conceptual step
        main_task_description: Overall main task objective
        """
        prompt = (
            f"You are an expert evaluator.\n"
            f"Main Task Objective: '{main_task_description}'\n"
            f"Current Hierarchical Conceptual Step being processed: '{layer_conceptual_step}'\n\n"
            f"The content of the '{artifact_type}' to be evaluated is as follows:\n\"\"\"\n{artifact_content}\n\"\"\"\n\n"
            "Evaluation Instructions:\n"
            f"1.  Contribution to 'Hierarchical Conceptual Step': To what extent does this '{artifact_type}' effectively achieve the goals of the 'Hierarchical Conceptual Step' mentioned above?\n"
            f"2.  Advancement of 'Main Task Objective': Does this '{artifact_type}' (as an outcome of the current layer) help in ultimately completing the 'Main Task Objective'? Does it lay a good foundation for subsequent steps, or might it lead off track?\n"
            "3.  Clarity and Feasibility: Is the artifact itself clear and concrete? If it is a plan or partial solution, does it have initial feasibility?\n\n"
            "Please provide an overall score and a brief justification.\n"
            "Output Format (Strictly Adhere):\n"
            "Score: [A floating-point number between 0.0 (very poor/unhelpful) and 1.0 (excellent/highly promising)]\n"
            "Justification: [A brief explanation for your score, stating how and why it helps or hinders the completion of the layer objective and the main task objective]"
        )
        return prompt

    def _parse_llm_response_for_prm_score(self, llm_response_text):
        if not llm_response_text or llm_response_text.startswith("Error (LOT):") or llm_response_text.startswith("LLM not initialized") or llm_response_text.startswith("LLM Error"): # Added specific error checks for LOT
            self.logger.warning(f"Invalid or error LLM response, cannot parse PRM score: {llm_response_text}")
            return 0.0, f"PRM scoring failed: Invalid LLM response ({llm_response_text})"

        score_match = re.search(r"Score:\s*([0-9.]+)", llm_response_text, re.IGNORECASE)
        justification_match = re.search(r"Justification:\s*(.+)", llm_response_text, re.IGNORECASE | re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.0
        justification = justification_match.group(1).strip() if justification_match else "No justification provided or parsing error."

        if not score_match:
            self.logger.warning(f"Could not parse PRM score from response. Raw response: '{llm_response_text}'")
        return score, justification

    def _evaluate_lot_artifact_with_prm(self, artifact_content, artifact_type, layer_conceptual_step, main_task_description):
        """ Evaluates an intermediate artifact of LOT using the PRM evaluator. """
        if not self.prm_evaluator_llm: # Check prm_evaluator_llm
            self.logger.error("PRM evaluator LLM not set. Cannot evaluate.")
            return 0.0, "PRM evaluator not set"
        
        prompt = self._generate_prm_style_scoring_prompt_for_lot_artifact(artifact_content, artifact_type, layer_conceptual_step, main_task_description)
        llm_response = self.prm_evaluator_llm.generate(prompt) # Use prm_evaluator_llm
        score, justification = self._parse_llm_response_for_prm_score(llm_response)
        self.logger.info(f"LOT artifact ({artifact_type} for layer '{layer_conceptual_step[:30]}...') PRM evaluation - Score: {score:.2f}")
        return score, justification


# --- Gemini API Interface ---
class GeminiLLMInterface:
    # def __init__(self, api_key, model_name="gemini-1.5-flash-latest", logger=None): # 添加 logger
    # def __init__(self, api_key, model_name="gemini-2.0-flash", logger=None): # 添加 logger
    def __init__(self, api_key, model_name="gemini-2.0-flash-lite", logger=None): # 添加 logger
        self.model = None
        self.logger = logger if logger else DefaultLogger()
        if not api_key:
            self.logger.error("Gemini API key is required. LOT LLM will not function.")
            return # Allow instance creation, but model will be None
            
        try:
            # Simplified API key configuration, assuming external (e.g., MASOrchestrator) has handled genai.configure
            # If LOT runs standalone, configuration needs to be in its __main__
            self.model = genai.GenerativeModel(model_name)
            self.logger.info(f"Gemini LLM interface initialized with model '{model_name}'.")
        except Exception as e:
            self.logger.error(f"Failed to initialize LOT Gemini GenerativeModel ({model_name}): {e}")
            self.model = None


    def generate(self, prompt_text, temperature=0.7): # Added temperature
        if not self.model:
            self.logger.error("LOT.GeminiLLMInterface: LLM model not initialized. Cannot generate content.")
            return "Error (LOT): LLM not initialized"
        try:
            self.logger.info(f"\n--- Sending prompt to Gemini (LOT Operation LLM) ---\n{prompt_text[:300]}...\n--- End of Gemini prompt (LOT Operation LLM) ---")
            response = self.model.generate_content(
                prompt_text,
                generation_config=genai.types.GenerationConfig(temperature=temperature) # Added temperature
            )
            llm_response_text = ""
            if hasattr(response, 'parts') and response.parts:
                 llm_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            elif hasattr(response, 'text'):
                 llm_response_text = response.text
            
            if not llm_response_text and hasattr(response, 'prompt_feedback') and \
               response.prompt_feedback.block_reason != genai.types.BlockReason.BLOCK_REASON_UNSPECIFIED:
                block_reason_str = genai.types.BlockReason(response.prompt_feedback.block_reason).name
                self.logger.warning(f"Warning (LOT): Prompt blocked due to {block_reason_str}.")
                return f"Error (LOT): Prompt blocked due to {block_reason_str}."

            self.logger.info(f"--- Received Gemini response (LOT Operation LLM) ---\n{llm_response_text[:300]}...\n--- End of Gemini response (LOT Operation LLM) ---")
            return llm_response_text if llm_response_text else "Error (LOT): No content generated or issue with prompt."

        except Exception as e:
            self.logger.error(f"Error calling Gemini API (LOT): {e}")
            return f"Error (LOT): Gemini API error - {str(e)}"


# --- Layer-of-Thoughts (LoT) Implementation ---
class OptionThought(Thought):
    def __init__(self, content, id, criterion, level=1, score=0.0, prm_justification="Not yet evaluated"): # Inherits score and prm_justification
        super().__init__(content, id, score, prm_justification)
        self.criterion = criterion # The criterion based on which this option was generated
        self.level = level # Priority level of the criterion (if any)

    def __repr__(self):
        return f"OptionThought(id={self.id}, level={self.level}, criterion='{self.criterion}', score={self.score:.2f}, content='{self.content[:20]}...')"

class LayerThought(Thought):
    def __init__(self, content, id, layer_index, score=0.0, prm_justification="Not yet evaluated"): # content is the conceptual step description
        super().__init__(content, id, score, prm_justification) # This score will represent the PRM score of the layer's aggregated output
        self.layer_index = layer_index
        self.option_thoughts = [] # List of OptionThoughts belonging to this layer

    def __repr__(self):
        return f"LayerThought(id={self.id}, layer_index={self.layer_index}, options={len(self.option_thoughts)}, score={self.score:.2f}, content='{self.content[:20]}...')"

class LayerOfThoughts(GraphOfThoughts):
    def __init__(self, llm_interface, logger=None, prm_evaluator_llm=None): # Added prm_evaluator_llm
        super().__init__(llm_interface, logger)
        self.layers = [] # Ordered list of LayerThought objects
        # If no dedicated PRM evaluation LLM is provided, default to the same LLM as for operations
        self.prm_evaluator_llm = prm_evaluator_llm if prm_evaluator_llm else llm_interface
        if not self.prm_evaluator_llm:
             self.logger.warning("LOT is not configured with a PRM evaluator LLM. PRM scoring functionality will be limited.")


    def add_layer_thought(self, conceptual_step_description):
        layer_index = len(self.layers)
        layer_thought_id = f"L{layer_index}_main"
        
        if layer_thought_id in self.thoughts: # Check if already exists
            self.logger.warning(f"LayerThought with ID '{layer_thought_id}' already exists. Will not add again.")
            return self.thoughts[layer_thought_id]

        # LayerThought's initial score and justification are updated after aggregation
        layer_thought = LayerThought(conceptual_step_description, layer_thought_id, layer_index)
        self.add_thought_object(layer_thought) # Add using base class method
        self.layers.append(layer_thought)
        self.logger.info(f"Added layer {layer_index}: {layer_thought}")
        return layer_thought

    def _generate_prompt_for_option_thought_criteria(self, layer_thought_content, previous_layer_output=None, main_task_description=None):
        prompt = f"Main Task Objective: '{main_task_description}'\n" if main_task_description else ""
        prompt += f"For the current conceptual step: '{layer_thought_content}'\n"
        if previous_layer_output:
            prompt += f"Based on the output from the previous layer: '{previous_layer_output}...' (this output aims to advance the main task)\n"
        prompt += "Please suggest a series of specific, actionable 'criteria' or 'exploration options' for this conceptual step to generate diverse partial solutions helpful for the main task. If criteria have priorities, please indicate them (e.g., Criterion A (Level 1); Criterion B (Level 1); Criterion C (Level 2)).\nPlease return only the list of criteria, separated by semicolons."
        return prompt
        
    def _generate_prompt_for_option_thought_solution(self, criterion, layer_conceptual_step, previous_layer_output=None, main_task_description=None):
        prompt = f"Main Task Objective: '{main_task_description}'\n" if main_task_description else ""
        prompt += f"Current conceptual step: '{layer_conceptual_step}'\n"
        if previous_layer_output:
            prompt += f"Previous layer output content: '{previous_layer_output[:200]}...'\n"
        prompt += f"Now, for the following 'criterion/exploration option', generate a concrete partial solution or detailed elaboration:\nCriterion/Exploration Option: '{criterion}'\n\nYour partial solution (ensure it is relevant to the main task objective):"
        return prompt

    def generate_and_evaluate_option_thoughts_for_layer(self, layer_id, main_task_description, previous_layer_aggregated_output=None):
        if layer_id not in self.thoughts or not isinstance(self.thoughts[layer_id], LayerThought):
            self.logger.error(f"Error: LayerThought with ID {layer_id} not found.")
        return []

        current_layer_thought = self.thoughts[layer_id]
        
        # Generate criteria for option thoughts
        criteria_prompt = self._generate_prompt_for_option_thought_criteria(
            current_layer_thought.content, 
            previous_layer_aggregated_output,
            main_task_description
        )
        llm_criteria_response = self.llm.generate(criteria_prompt, temperature=0.7)
        if llm_criteria_response.startswith("Error (LOT):"):
            self.logger.error(f"Could not get criteria for layer {current_layer_thought.layer_index}: {llm_criteria_response}")
            return []
        parsed_criteria = self._parse_criteria_from_llm(llm_criteria_response)

        generated_options_with_scores = []
        for i, crit_info in enumerate(parsed_criteria):
            criterion_text = crit_info['text']
            criterion_level = crit_info.get('level', 1)

            solution_prompt = self._generate_prompt_for_option_thought_solution(
                criterion_text,
                current_layer_thought.content,
                previous_layer_aggregated_output,
                main_task_description
            )
            llm_solution_response = self.llm.generate(solution_prompt, temperature=0.6)
            if llm_solution_response.startswith("Error (LOT):"):
                self.logger.warning(f"Could not generate solution for criterion '{criterion_text}': {llm_solution_response}")
                continue

            solution_content = llm_solution_response.strip()
            option_id = f"{current_layer_thought.id}_Opt{i}"

            option_thought = OptionThought(solution_content, option_id, criterion_text, level=criterion_level)
            self.add_thought_object(option_thought)

            if current_layer_thought not in option_thought.parents:
                option_thought.parents.append(current_layer_thought)
            if option_thought not in current_layer_thought.children:
                current_layer_thought.children.append(option_thought)

            if option_thought not in current_layer_thought.option_thoughts:
                current_layer_thought.option_thoughts.append(option_thought)

            generated_options_with_scores.append(option_thought)
            self.logger.info(f"Generated OptionThought for layer {current_layer_thought.layer_index}: {option_thought}")

        return generated_options_with_scores

    def _parse_criteria_from_llm(self, llm_response_text):
        criteria = []
        if not llm_response_text or not isinstance(llm_response_text, str) or llm_response_text.startswith("Error (LOT):"):
            self.logger.warning(f"Criteria from LLM response is empty or incorrectly formatted: {llm_response_text}")
            return [{'text': "Default criterion (due to parsing error)", 'level': 1}] # Default if parsing fail

        parts = llm_response_text.split(';')
        for part in parts:
            part = part.strip()
            if not part: continue
            level = 1 # Default level
            text = part
            # Try to parse "(Level X)" or "(等級 X)"
            match = re.search(r'\((?:level|等級)\s*(\d+)\)$', part, re.IGNORECASE)
            if match:
                try:
                    level = int(match.group(1))
                    text = re.sub(r'\s*\((?:level|等級)\s*\d+\)$', '', part, flags=re.IGNORECASE).strip() # Remove level part from text
                except ValueError:
                    self.logger.warning(f"Could not parse level from criterion '{part}', using default level 1.")
            criteria.append({'text': text, 'level': level})
        
        if not criteria: # If no criteria after splitting
             self.logger.warning(f"Could not parse any criteria from '{llm_response_text}', using response itself as a single criterion.")
             return [{'text': llm_response_text.strip(), 'level': 1}]
        return criteria
        
def aggregate_and_evaluate_option_thoughts_in_layer(self, layer_id, main_task_description, aggregation_strategy='best_prm_score'):
    if layer_id not in self.thoughts or not isinstance(self.thoughts[layer_id], LayerThought):
        self.logger.error(f"Error: LayerThought with ID {layer_id} not found for aggregation.")
        return None, 0.0, "LayerThought not found"

    current_layer_thought = self.thoughts[layer_id]
    options = current_layer_thought.option_thoughts
    if not options:
        self.logger.warning(
            f"No OptionThoughts to aggregate in layer {current_layer_thought.layer_index}."
        )
        current_layer_thought.score = 0.0
        current_layer_thought.prm_justification = "No option thoughts available for aggregation."
        return (
            f"Layer {current_layer_thought.layer_index} did not generate any options.",
            0.0,
            current_layer_thought.prm_justification
        )

    aggregated_content = (
        f"Aggregated result from layer {current_layer_thought.layer_index} "
        f"(Concept: '{current_layer_thought.content}...') "
        f"using strategy '{aggregation_strategy}':\n"
    )

    # Sort options by level ascending, then by score descending
    options.sort(key=lambda ot: (ot.level, -ot.score))

    if aggregation_strategy == 'best_prm_score':
        if options:
            best_option = options[0]
            aggregated_content += (
                f"Best option (Criterion: '{best_option.criterion}', "
                f"Level: {best_option.level}):\n{best_option.content}"
            )
        else:
            aggregated_content += "No best option to select."
    elif aggregation_strategy in ('weighted_sum_content', 'all_content_ranked'):
        for opt in options:
            aggregated_content += (
                f"- (Criterion: '{opt.criterion}', Level: {opt.level}):\n"
                f"  {opt.content}\n\n"
            )
    else:
        self.logger.warning(
            f"Unknown aggregation strategy '{aggregation_strategy}', defaulting to 'best_prm_score'."
        )
        if options:
            best_option = options[0]
            aggregated_content += (
                f"Best option (Criterion: '{best_option.criterion}', "
                f"Level: {best_option.level}):\n{best_option.content}"
            )
        else:
            aggregated_content += "No best option to select."

    # PRM evaluation removed; maintain variable names for compatibility
    layer_output_prm_score = None
    layer_output_prm_justification = None

    # Update LayerThought with placeholders
    current_layer_thought.score = layer_output_prm_score
    current_layer_thought.prm_justification = layer_output_prm_justification

    self.logger.info(f"Aggregated options in layer {current_layer_thought.layer_index}.")
    return aggregated_content, layer_output_prm_score, layer_output_prm_justification



    def run_pipeline(self, conceptual_steps, main_task_description, initial_input=None, min_layer_prm_score_threshold=0.3):
        # main_task_description is the final task the entire LOT process is trying to solve
        # min_layer_prm_score_threshold: If a layer's aggregated output PRM score is below this, consider early termination or remedial actions
        
        previous_layer_output_content = initial_input
        final_pipeline_output = None # The final, considered successful, pipeline output
        
        self.logger.info(f"LOT pipeline execution started, Main Task Objective: {main_task_description}")

        for i, step_description in enumerate(conceptual_steps):
            self.logger.info(f"\n--- Processing Layer {i}: {step_description} ---")
            layer_thought = self.add_layer_thought(step_description) # LayerThought's score is default at this point
            
            # Generate and evaluate option thoughts for the current layer
            # Need to pass main_task_description for PRM evaluation of option thoughts' relevance
            option_thoughts = self.generate_and_evaluate_option_thoughts_for_layer(
                layer_thought.id, 
                main_task_description, # Pass the main task
                previous_layer_output_content
            )
            
            if not option_thoughts:
                self.logger.warning(f"Layer {i} failed to generate any option thoughts. May need to check criteria generation or solution generation prompts.")
                # Even if no options, try to aggregate (will get "no options" message), then let PRM evaluate this "empty" aggregation
            
            # Aggregate option thoughts of this layer, and perform PRM evaluation on the aggregated result
            # Aggregation strategy can be based on option's PRM scores
            aggregated_output_content, layer_prm_score, layer_prm_justification = self.aggregate_and_evaluate_option_thoughts_in_layer(
                layer_thought.id, 
                main_task_description, # Pass the main task
                aggregation_strategy='all_content_ranked' # Or 'best_prm_score'
            )
            
            # LayerThought's score and prm_justification are updated within aggregate_and_evaluate...
            
            if aggregated_output_content is None: # Theoretically aggregate... should always return something unless layer_id is wrong
                self.logger.error(f"Serious error during aggregation of layer {i}. Stopping pipeline.")
                final_pipeline_output = previous_layer_output_content # Use previous layer's output as final result
                break 
            
            self.logger.info(f"Layer {i} aggregated output PRM score: {layer_prm_score:.2f}. Justification: {layer_prm_justification}")

            # if layer_prm_score < min_layer_prm_score_threshold:
            #     self.logger.warning(f"Layer {i}'s aggregated output PRM score ({layer_prm_score:.2f}) is below threshold ({min_layer_prm_score_threshold}).")
                # More complex logic could be implemented here, e.g.:
                # 1. Try different aggregation strategies
                # 2. Regenerate option thoughts for this layer (perhaps adjusting prompts or temperature)
                # 3. Backtrack to a previous layer
                # 4. Terminate pipeline early
                # For simplicity, here we just log a warning but continue (or could choose to terminate)
                # self.logger.warning(f"Pipeline may not achieve optimal results. Consider termination or remedial actions. Continuing for now...")
                # If deciding to terminate:
                # final_pipeline_output = previous_layer_output_content # Or aggregated_output_content, depending on strategy
                # break

            previous_layer_output_content = aggregated_output_content
            final_pipeline_output = aggregated_output_content # Always points to the last successful aggregated output

        self.logger.info(f"LOT pipeline execution finished. : {self.layers[-1].score if self.layers else 'N/A'}")
        return final_pipeline_output

# --- Encapsulate example usage into a function ---
def run_lot_example_workflow_with_prm(api_key):
    logger_example = DefaultLogger()
    logger_example.info(f"Running LOT (PRM-style) example with API key: ...{api_key[-4:]}")
    
    # Configure Gemini API (if not already configured globally)
    # For standalone execution, do a configuration check here
    try:
        if not getattr(genai, '_is_configured_by_lot_example', False): # Avoid duplicate configuration
            genai.configure(api_key=api_key)
            setattr(genai, '_is_configured_by_lot_example', True) # Mark as configured by this example
            logger_example.info("Gemini API configured for LOT example.")
    except Exception as e:
        logger_example.error(f"Error configuring Gemini API: {e}")
        return

    try:
        # 操作型 LLM (用於生成標準、生成選項解決方案等)
        # llm_operator = GeminiLLMInterface(api_key=api_key, model_name="gemini-1.5-flash-latest", logger=logger_example)
        # llm_operator = GeminiLLMInterface(api_key=api_key, model_name="gemini-2.0-flash", logger=logger_example)
        llm_operator = GeminiLLMInterface(api_key=api_key, model_name="gemini-2.0-flash-lite", logger=logger_example)
        # PRM 評估器 LLM (可以與操作型 LLM 相同，也可以是另一個更強的或專門微調的評估模型)
        # 為了演示，這裡使用相同的 LLM 實例
        llm_prm_evaluator = llm_operator 
        
        if not llm_operator.model: # Check if model initialized successfully
             logger_example.error("LOT operational LLM failed to initialize. Aborting example.")
             return

    except ValueError as e: # Catch potential errors from LLM interface init
        logger_example.error(f"Error initializing LLM interface: {e}")
        return
    
    lot_system = LayerOfThoughts(llm_operator, logger=logger_example, prm_evaluator_llm=llm_prm_evaluator)

    # Main Task Objective: Develop a comprehensive marketing plan for an outdoor music festival to be held in the summer.
    # Budget: Medium. Target Audience: Young adults aged 18-35. Location: City park.
    main_task_music_festival = (
        "Develop a comprehensive marketing plan for an outdoor music festival to be held in summer, "
        "located in a city park, with a medium budget, targeting young adults aged 18-35, "
        "aiming to maximize attendance and brand awareness."
    )

    conceptual_steps_marketing_plan = [
        "Phase 1: Market and Audience Analysis - In-depth understanding of target audience preferences, frequented social media platforms, and expectations for a music festival.",
        "Phase 2: Core Messaging and Brand Positioning - Identify the unique selling proposition (USP) of the festival and develop core promotional messages and brand image appealing to the target audience.",
        "Phase 3: Promotion Channels and Activity Planning - Select the most effective online and offline promotion channels, and plan specific pre-launch activities, on-site interactions, and post-event publicity.",
        "Phase 4: Budget Allocation and Performance Measurement - Rationally allocate budget for various promotional activities and set key performance indicators (KPIs) to measure the plan's effectiveness."
    ]
    
    # Initial input can be empty or contain some preliminary ideas/constraints for the main task
    initial_input_for_marketing = "The preliminary theme for the music festival is 'Urban Oasis Sounds', emphasizing the combination of nature and music."

    logger_example.info(f"\nLoT Pipeline execution started, Main Task: {main_task_music_festival}")
    final_marketing_plan = lot_system.run_pipeline(
        conceptual_steps_marketing_plan, 
        main_task_description=main_task_music_festival, # Pass the main task description
        initial_input=initial_input_for_marketing,
        min_layer_prm_score_threshold=0.4 # Set a layer PRM score threshold
    )

    logger_example.info(f"\n--- LoT Pipeline's Final Marketing Plan (or last successful layer output) ---")
    if final_marketing_plan:
        logger_example.info(final_marketing_plan)
        # Print the PRM score and justification of the last layer
        if lot_system.layers:
            last_layer = lot_system.layers[-1]
            logger_example.info(f"\nPRM Evaluation of the Final Layer (L{last_layer.layer_index}):")
            logger_example.info(f"  Score: {last_layer.score:.2f}")
            logger_example.info(f"  Justification: {last_layer.prm_justification}")
    else:
        logger_example.warning("Pipeline failed to produce a final output.")
    
    logger_example.info("\n--- End of LOT (PRM-style) Example Usage ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY")
    logger_main = DefaultLogger() # Use default logger for main block

    if not API_KEY:
        logger_main.warning("Warning: GEMINI_API_KEY not found in environment variables.")
        logger_main.warning("Please set the GEMINI_API_KEY environment variable or provide the API key in LOT.py to run the example.")
    
    # Check if API_KEY is not None AND not the placeholder
    if API_KEY and API_KEY != "YOUR_GEMINI_API_KEY_HERE":
        run_lot_example_workflow_with_prm(API_KEY)
    elif API_KEY == "YOUR_GEMINI_API_KEY_HERE": # If it IS the placeholder
        logger_main.error("*****************************************************************")
        logger_main.error("Warning: You are using the default placeholder API key.")
        logger_main.error("Please replace 'YOUR_GEMINI_API_KEY_HERE' in the code with your actual Gemini API key, or set the environment variable.")
        logger_main.error("Example execution may fail or produce unexpected results.")
        logger_main.error("*****************************************************************")
        # Still attempt to run, let GeminiLLMInterface handle or raise error internally
        try:
            run_lot_example_workflow_with_prm(API_KEY)
        except Exception as e: # Catch errors from GeminiLLMInterface init due to invalid key
             logger_main.error(f"Example execution failed due to API key issue: {e}")
    else: # API_KEY is None (already warned above, but this handles the execution path)
        logger_main.error("No valid API key; LLM interactions in the example will fail.")
        # Optionally, still try to run to see how far it gets or if it handles the missing key gracefully.
        try:
            run_lot_example_workflow_with_prm(API_KEY) # API_KEY is None here
        except Exception as e:
             logger_main.error(f"Example execution failed due to API key issue: {e}")