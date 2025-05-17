import google.generativeai as genai
import os
import re # 用於解析多個思維

class Thought:
    def __init__(self, content, id, score=0.0, generated_by_llm=True, prm_justification="尚未評估"): # 新增 prm_justification
        self.id = id
        self.content = content # 內容
        self.score = score # 分數 (此分數將主要由 PRM 風格評估決定)
        self.prm_justification = prm_justification # PRM 評估的理由
        self.children = []  # 依賴此思維的後續思維
        self.parents = []   # 此思維所依賴的前續思維
        self.generated_by_llm = generated_by_llm # 此思維內容是來自 LLM 還是初始輸入？

    def __repr__(self):
        # 將換行符替換為空格，以便於單行顯示
        display_content = self.content[:30].replace('\n', ' ')
        return f"Thought(id={self.id}, score={self.score:.2f}, generated_by_llm={self.generated_by_llm}, content='{display_content}...')"

class GeminiLLM:
    def __init__(self, api_key, model_name="gemini-1.5-flash-latest"): # 更新模型名稱以符合常見用法
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


class GraphOfThoughts:
    def __init__(self, llm_interface, logger=None): # 添加 logger
        self.thoughts = {} # 以 id 儲存所有思維
        self.next_thought_id = 0
        self.llm = llm_interface
        self.logger = logger if logger else self._get_default_logger() # 使用傳入的 logger 或預設 logger

    def _get_default_logger(self):
        # 一個簡單的預設 logger，如果外部沒有提供
        class PrintLogger:
            def info(self, message): print(f"[GOT INFO] {message}")
            def warning(self, message): print(f"[GOT WARNING] {message}")
            def error(self, message): print(f"[GOT ERROR] {message}")
        return PrintLogger()

    def _get_new_id(self):
        new_id = self.next_thought_id
        self.next_thought_id += 1
        return new_id

    def add_thought(self, content, parent_ids=None, score=0.0, generated_by_llm=True, prm_justification="尚未評估"):
        new_id = self._get_new_id()
        thought = Thought(content, new_id, score, generated_by_llm, prm_justification)
        self.thoughts[new_id] = thought
        if parent_ids:
            for pid in parent_ids:
                if pid in self.thoughts:
                    self.thoughts[pid].children.append(thought)
                    thought.parents.append(self.thoughts[pid])
        self.logger.info(f"已新增思維：{thought}")
        return thought

    # --- Prompter 模組 ---
    def _generate_prompt_for_new_thought(self, task_description, existing_thoughts_content=None, num_new_thoughts=1):
        prompt = f"主要任務：{task_description}\n"
        if existing_thoughts_content:
            prompt += "基於以下現有資訊/思維：\n"
            for i, content in enumerate(existing_thoughts_content):
                prompt += f"現有思維 {i+1}：{content}\n"
        prompt += f"\n請產生 {num_new_thoughts} 個新的、不同的思維或解決方案步驟，以推進主要任務的解決。請清晰地提供每個新思維。"
        if num_new_thoughts > 1:
            prompt += "\n請將每個新思維另起一行開始，可選擇性編號 (例如 '1. <思維內容>')。"
        return prompt

    def _generate_prompt_for_aggregation(self, thoughts_to_aggregate_content, task_description):
        prompt = f"主要任務：{task_description}\n"
        prompt += "請將以下不同的思維聚合成一個單一、更全面且精煉的思維。請識別核心觀點並將其綜合為一個連貫的摘要或改進的解決方案，以更好地完成主要任務：\n"
        for i, content in enumerate(thoughts_to_aggregate_content):
            prompt += f"待聚合思維 {i+1}：{content}\n"
        prompt += "\n合併與聚合後的思維 (旨在推進主要任務)："
        return prompt

    def _generate_prompt_for_refinement(self, thought_content, task_description, refinement_instruction):
        prompt = f"主要任務：{task_description}\n"
        prompt += f"原始思維：{thought_content}\n"
        prompt += f"精煉指示：{refinement_instruction}\n"
        prompt += "\n請僅基於原始思維和指示提供精煉後的思維，使其更有助於完成主要任務："
        return prompt

    def _generate_prm_style_scoring_prompt(self, thought_content, main_task_description):
        """
        產生 PRM 風格的評分提示。
        評估此思維對於完成 `main_task_description` 的潛在貢獻。
        """
        prompt = (
            f"您是一位專家級的評估員，任務是評估一個「思維」的質量和潛力。\n"
            f"主要任務目標：'{main_task_description}'\n\n"
            f"待評估的思維內容：\n\"\"\"\n{thought_content}\n\"\"\"\n\n"
            "評估指示：\n"
            "1.  相關性：此思維與主要任務目標的相關程度如何？\n"
            "2.  潛力/貢獻度：此思維有多大可能引導我們向主要任務目標的成功解決方案邁進一步？它是否開闢了有前景的途徑，或者它是一個死胡同/低質量想法？\n"
            "3.  清晰度和可行性（如果適用）：思維本身是否清晰？如果它是一個行動或計劃，它是否具有初步的可行性？\n\n"
            "請提供一個總體評分和簡要理由。\n"
            "輸出格式（嚴格遵守）：\n"
            "Score: [一個介於 0.0 (非常差/無助益) 到 1.0 (非常好/極具潛力) 之間的浮點數]\n"
            "Justification: [對您的分數的簡要解釋，說明其如何以及為何有助於或無助於主要任務目標]"
        )
        return prompt

    # --- Parser 模組 ---
    def _parse_llm_response_for_new_thoughts(self, llm_response_text, num_expected_thoughts=1):
        if not llm_response_text or llm_response_text.startswith("錯誤："):
            self.logger.warning(f"LLM 回應無效或為錯誤訊息，無法解析新思維: {llm_response_text}")
            return []

        if num_expected_thoughts == 1:
            return [llm_response_text.strip()]
        else:
            potential_thoughts = re.split(r'\n\s*(?=\d+\.\s|-\s|\*\s)', llm_response_text)
            cleaned_thoughts = []
            for pt in potential_thoughts:
                cleaned = re.sub(r'^\s*(\d+\.\s*|-\s*|\*\s*)', '', pt.strip())
                if cleaned:
                    cleaned_thoughts.append(cleaned)
            
            if cleaned_thoughts and len(cleaned_thoughts) >= num_expected_thoughts:
                return cleaned_thoughts[:num_expected_thoughts]
            
            # Fallback to splitting by newline if regex fails or yields too few
            thoughts = llm_response_text.strip().split('\n\n')
            if len(thoughts) < num_expected_thoughts:
                thoughts = llm_response_text.strip().split('\n')
            return [t.strip() for t in thoughts if t.strip()][:num_expected_thoughts]


    def _parse_llm_response_for_prm_score(self, llm_response_text):
        """
        從 LLM 的回應中解析 PRM 風格的評分和理由。
        """
        if not llm_response_text or llm_response_text.startswith("錯誤："):
            self.logger.warning(f"LLM 回應無效或為錯誤訊息，無法解析 PRM 分數: {llm_response_text}")
            return 0.0, f"PRM 評分失敗：LLM 回應無效 ({llm_response_text})"

        score_match = re.search(r"Score:\s*([0-9.]+)", llm_response_text, re.IGNORECASE)
        justification_match = re.search(r"Justification:\s*(.+)", llm_response_text, re.IGNORECASE | re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.0
        justification = justification_match.group(1).strip() if justification_match else "未提供理由或解析錯誤。"

        if not score_match:
            self.logger.warning(f"無法從回應中解析 PRM 分數。原始回應：'{llm_response_text}'")
        return score, justification

    # --- Controller 與圖形操作 ---
    def generate_and_evaluate_thoughts(self, task_description, num_thoughts=1, from_thought_ids=None, initial_content_list=None):
        """
        產生思維，並立即使用 PRM 風格的評估器對其進行評分。
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
                    self.logger.warning(f"在產生新思維時，找不到 from_thought_id {tid}。")
        elif initial_content_list:
             base_content_for_prompt.extend(initial_content_list)

        prompt = self._generate_prompt_for_new_thought(task_description, base_content_for_prompt, num_new_thoughts=num_thoughts)
        llm_response = self.llm.generate(prompt)
        parsed_contents = self._parse_llm_response_for_new_thoughts(llm_response, num_thoughts)

        for content in parsed_contents:
            if not content or content.startswith("錯誤："):
                self.logger.warning(f"因錯誤或空內容跳過思維產生：'{content}'")
                continue
            
            # 添加思維時不立即評分，而是先創建物件
            thought_obj = self.add_thought(content, parent_ids=parent_ids_for_new_thoughts, generated_by_llm=True)
            
            # 對新生成的思維進行 PRM 風格評分
            prm_score, prm_justification = self.score_thought_with_prm_evaluator(thought_obj.id, task_description)
            thought_obj.score = prm_score # 更新分數
            thought_obj.prm_justification = prm_justification # 更新理由
            self.logger.info(f"新思維 {thought_obj.id} 已評分 - PRM Score: {prm_score:.2f}, Justification: {prm_justification}")
            
            generated_llm_thoughts_objects.append(thought_obj)
        return generated_llm_thoughts_objects

    def aggregate_and_evaluate_thoughts(self, thought_ids_to_aggregate, task_description):
        """
        聚合思維，並對聚合後的新思維進行 PRM 風格評分。
        """
        if not thought_ids_to_aggregate or not all(tid in self.thoughts for tid in thought_ids_to_aggregate):
            self.logger.error("一個或多個用於聚合的思維 ID 未找到或列表為空。")
            return None

        contents = [self.thoughts[tid].content for tid in thought_ids_to_aggregate]
        prompt = self._generate_prompt_for_aggregation(contents, task_description)
        llm_response = self.llm.generate(prompt)

        if not llm_response or llm_response.startswith("錯誤："):
            self.logger.error(f"聚合失敗，原因：錯誤或空回應：'{llm_response}'")
            return None
        
        aggregated_content = llm_response.strip()
        aggregated_thought_obj = self.add_thought(aggregated_content, parent_ids=thought_ids_to_aggregate, generated_by_llm=True)
        
        # 對聚合後的思維進行 PRM 風格評分
        prm_score, prm_justification = self.score_thought_with_prm_evaluator(aggregated_thought_obj.id, task_description)
        aggregated_thought_obj.score = prm_score
        aggregated_thought_obj.prm_justification = prm_justification
        self.logger.info(f"聚合思維 {aggregated_thought_obj.id} 已評分 - PRM Score: {prm_score:.2f}, Justification: {prm_justification}")
        
        return aggregated_thought_obj

    def refine_and_evaluate_thought(self, thought_id, task_description, refinement_instruction="提高清晰度和細節，使其更接近主要任務目標。"):
        """
        精煉思維，並對精煉後的新思維進行 PRM 風格評分。
        """
        if thought_id not in self.thoughts:
            self.logger.error(f"錯誤：找不到用於精煉的思維 ID {thought_id}。")
            return None

        content_to_refine = self.thoughts[thought_id].content
        prompt = self._generate_prompt_for_refinement(content_to_refine, task_description, refinement_instruction)
        llm_response = self.llm.generate(prompt)

        if not llm_response or llm_response.startswith("錯誤："):
            self.logger.error(f"精煉失敗，原因：錯誤或空回應：'{llm_response}'")
            return None
        
        refined_content = llm_response.strip()
        refined_thought_obj = self.add_thought(refined_content, parent_ids=[thought_id], generated_by_llm=True)
        
        # 對精煉後的思維進行 PRM 風格評分
        prm_score, prm_justification = self.score_thought_with_prm_evaluator(refined_thought_obj.id, task_description)
        refined_thought_obj.score = prm_score
        refined_thought_obj.prm_justification = prm_justification
        self.logger.info(f"精煉思維 {refined_thought_obj.id} 已評分 - PRM Score: {prm_score:.2f}, Justification: {prm_justification}")

        return refined_thought_obj

    def score_thought_with_prm_evaluator(self, thought_id, main_task_description):
        """
        使用 LLM 作為 PRM 評估器來評分單個思維。
        這個方法的核心是 _generate_prm_style_scoring_prompt。
        """
        if thought_id not in self.thoughts:
            self.logger.error(f"錯誤：找不到用於 PRM 評估的思維 ID {thought_id}。")
            return 0.0, "思維未找到"

        thought_content = self.thoughts[thought_id].content
        # 使用 PRM 風格的評分提示
        prompt = self._generate_prm_style_scoring_prompt(thought_content, main_task_description)
        llm_response = self.llm.generate(prompt)

        score, justification = self._parse_llm_response_for_prm_score(llm_response) # 使用新的解析器
        
        # 更新思維物件中的分數和理由 (如果需要，可以在 add_thought 後單獨調用此方法更新)
        # self.thoughts[thought_id].score = score 
        # self.thoughts[thought_id].prm_justification = justification
        # self.logger.info(f"已使用 PRM 評估器評分思維 {thought_id}：{score:.2f}。理由：{justification}")
        return score, justification


    def rank_thoughts(self, top_n=None):
        """根據分數 (現在是 PRM 風格的分數) 對思維進行排序。"""
        if not self.thoughts: return []
        sorted_thoughts = sorted(self.thoughts.values(), key=lambda t: (t.score, -t.id), reverse=True)
        return sorted_thoughts[:top_n] if top_n is not None else sorted_thoughts

    def get_thought(self, thought_id):
        return self.thoughts.get(thought_id)

    def print_graph(self):
        self.logger.info("\n--- 目前的思維圖 ---")
        if not self.thoughts:
            self.logger.info("思維圖為空。")
            return
        for thought_id in sorted(self.thoughts.keys()):
            thought = self.thoughts[thought_id]
            display_content = thought.content[:100].replace('\n', ' ').strip()
            self.logger.info(f"\n思維 ID：{thought.id} (PRM 分數：{thought.score:.2f}, LLM 產生：{thought.generated_by_llm})")
            self.logger.info(f"  內容：'{display_content}...'")
            self.logger.info(f"  PRM 理由：{thought.prm_justification}")
            parent_ids = [p.id for p in thought.parents]
            children_ids = [c.id for c in thought.children]
            self.logger.info(f"  父思維 ID：{parent_ids if parent_ids else '無'}")
            self.logger.info(f"  子思維 ID：{children_ids if children_ids else '無'}")
        self.logger.info("--- 思維圖結束 ---")

# --- 將範例用法封裝到函式中 ---
def run_got_example_workflow_with_prm_scoring(api_key):
    """
    執行 GraphOfThoughts 的範例工作流程，並整合 PRM 風格的評分。
    """
    # 創建一個簡單的 logger 實例用於 GOT 系統
    class SimpleLogger:
        def info(self, msg): print(f"[GOT WORKFLOW INFO] {msg}")
        def warning(self, msg): print(f"[GOT WORKFLOW WARNING] {msg}")
        def error(self, msg): print(f"[GOT WORKFLOW ERROR] {msg}")
    
    logger = SimpleLogger()
    logger.info(f"正在使用 API 金鑰執行 GOT (PRM 風格評分) 範例: ...{api_key[-4:]}")

    try:
        llm_api = GeminiLLM(api_key=api_key)
    except ValueError as e:
        logger.error(f"初始化錯誤：{e}")
        logger.error("範例結束。請提供一個有效的 Gemini API 金鑰。")
        return
    except Exception as e:
        logger.error(f"LLM 初始化期間發生未預期錯誤：{e}")
        return

    got_system = GraphOfThoughts(llm_api, logger=logger)

    main_task = "為一家小型本地咖啡店制定一個獨特的忠誠度計劃，以提高顧客保留率和每日銷售額。"
    logger.info(f"\n主要任務：{main_task}\n")

    # 1. 初始腦力激盪：產生初步的忠誠度計劃概念，並立即進行 PRM 風格評估
    logger.info("\n步驟 1：初始腦力激盪與 PRM 評估 (產生 2 個初步概念)")
    initial_ideas = got_system.generate_and_evaluate_thoughts( # 使用新的整合方法
        task_description="為咖啡店忠誠度計劃產生不同的核心概念。" + main_task, # 確保 task_description 包含主任務
        num_thoughts=2
    )
    if not initial_ideas or len(initial_ideas) < 1: # 即使只產生一個也繼續
        logger.error("未能產生足夠的初始概念。結束範例。")
        # return # 即使只產生一個也嘗試繼續

    # 打印初始想法及其 PRM 分數
    for idea in initial_ideas:
        logger.info(f"初始概念 ID {idea.id}: '{idea.content[:50]}...' (PRM Score: {idea.score:.2f}, Justification: {idea.prm_justification})")

    # 2. 選擇最佳初始概念並詳細闡述，然後對闡述結果進行 PRM 評估
    best_initial_idea = None
    if initial_ideas:
        # 根據 PRM 分數選擇最佳初始概念
        initial_ideas.sort(key=lambda t: t.score, reverse=True)
        best_initial_idea = initial_ideas[0]
        logger.info(f"\n步驟 2：選擇 PRM 分數最高的初始概念 (ID: {best_initial_idea.id}) 進行詳細闡述")
        
        # 這裡我們直接使用 generate_and_evaluate_thoughts 來闡述並評估
        # 注意：from_thought_ids 應該傳遞 ID 列表
        elaborated_thoughts = got_system.generate_and_evaluate_thoughts(
            task_description=f"詳細闡述以下忠誠度計劃概念 '{best_initial_idea.content[:50]}...'，使其更具體可行。" + main_task,
            num_thoughts=1,
            from_thought_ids=[best_initial_idea.id] 
        )
    else:
        logger.warning("沒有初始概念可供選擇。跳過闡述步驟。")
        elaborated_thoughts = []


    elaborated_idea = None
    if elaborated_thoughts:
        elaborated_idea = elaborated_thoughts[0]
        logger.info(f"闡述後的概念 ID {elaborated_idea.id}: '{elaborated_idea.content[:50]}...' (PRM Score: {elaborated_idea.score:.2f}, Justification: {elaborated_idea.prm_justification})")

        # 3. 精煉已闡述的概念，並對精煉結果進行 PRM 評估
        logger.info(f"\n步驟 3：精煉已闡述概念 (ID: {elaborated_idea.id})")
        refined_thought_obj = got_system.refine_and_evaluate_thought( # 使用新的整合方法
            elaborated_idea.id,
            task_description="精煉忠誠度計劃，使其更具吸引力和成本效益。" + main_task,
            refinement_instruction="增加一個獨特的、低成本但高感知價值的獎勵元素，並考慮如何追踪會員進度。"
        )
        if refined_thought_obj:
            logger.info(f"精煉後的概念 ID {refined_thought_obj.id}: '{refined_thought_obj.content[:50]}...' (PRM Score: {refined_thought_obj.score:.2f}, Justification: {refined_thought_obj.prm_justification})")
    else:
        logger.warning("由於未能成功闡述概念，跳過精煉步驟。")
        refined_thought_obj = None # 確保 refined_thought_obj 已定義

    # 4. 嘗試聚合 (如果有多個高質量的想法)
    # 這裡的聚合邏輯可以更複雜，例如聚合來自不同分支的最佳想法
    # 為了簡化，我們假設聚合 best_initial_idea (如果與 elaborated_idea 的父節點不同) 和 refined_thought_obj
    thoughts_for_aggregation = []
    if best_initial_idea and refined_thought_obj and best_initial_idea.id != refined_thought_obj.parents[0].id : # 確保 refined_thought_obj 有父節點
         thoughts_for_aggregation.append(best_initial_idea.id)
         thoughts_for_aggregation.append(refined_thought_obj.id)
    elif refined_thought_obj: # 如果只有精煉後的想法
        thoughts_for_aggregation.append(refined_thought_obj.id)
        if initial_ideas and len(initial_ideas) > 1 and initial_ideas[1].id != refined_thought_obj.parents[0].id: # 嘗試加入第二個初始想法
             thoughts_for_aggregation.append(initial_ideas[1].id)


    if len(thoughts_for_aggregation) >= 2:
        logger.info(f"\n步驟 4：聚合思維 {thoughts_for_aggregation}")
        aggregated_thought = got_system.aggregate_and_evaluate_thoughts( # 使用新的整合方法
            thoughts_for_aggregation,
            task_description="將以下忠誠度計劃的方面組合成一個統一且強大的最終方案。" + main_task
        )
        if aggregated_thought:
            logger.info(f"聚合後的概念 ID {aggregated_thought.id}: '{aggregated_thought.content[:50]}...' (PRM Score: {aggregated_thought.score:.2f}, Justification: {aggregated_thought.prm_justification})")
    else:
        logger.info("沒有足夠的不同高品質思維進行聚合，跳過聚合步驟。")


    # 5. 最終排序並顯示思維圖
    got_system.print_graph()

    logger.info("\n--- 最終所有思維的排序 (基於 PRM 風格評分) ---")
    top_thoughts = got_system.rank_thoughts()
    for i, thought_instance in enumerate(top_thoughts):
        display_content = thought_instance.content[:80].replace('\n', ' ').strip()
        logger.info(f"{i+1}. (ID: {thought_instance.id}, PRM Score: {thought_instance.score:.2f}) - {display_content}...")
        logger.info(f"    Justification: {thought_instance.prm_justification}")


    logger.info("\n--- GOT (PRM 風格評分) 範例用法結束 ---")

# --- 主執行區塊 ---
if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("警告：未在環境變數中找到 GEMINI_API_KEY。")
        print("請設定 GEMINI_API_KEY 環境變數或在 GOT.py 中提供 API 金鑰以執行範例。")
    else:
        run_got_example_workflow_with_prm_scoring(GEMINI_API_KEY)
