import google.generativeai as genai
import os
import re # 用於解析多個思維

class Thought:
    def __init__(self, content, id, score=0.0, generated_by_llm=True): # 新增 generated_by_llm
        self.id = id
        self.content = content # 內容
        self.score = score # 分數
        self.children = []  # 依賴此思維的後續思維
        self.parents = []   # 此思維所依賴的前續思維
        self.generated_by_llm = generated_by_llm # 此思維內容是來自 LLM 還是初始輸入？

    def __repr__(self):
        # 將換行符替換為空格，以便於單行顯示
        display_content = self.content[:30].replace('\n', ' ')
        return f"Thought(id={self.id}, score={self.score:.2f}, generated_by_llm={self.generated_by_llm}, content='{display_content}...')"

class GeminiLLM:
    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        """
        初始化 Gemini LLM 介面。
        Args:
            api_key (str): 您的 Google Generative AI API 金鑰。
            model_name (str): 要使用的 Gemini 模型 (例如 "gemini-pro")。
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
        print(f"\n--- 正在發送提示到 Gemini ---\n{prompt}\n--- Gemini 提示結束 ---")
        try:
            # 設定預設的安全設定，如果未提供則使用
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
                safety_settings=effective_safety_settings # 套用安全設定
            )

            llm_response_text = ""
            # 檢查 response.parts 是否存在且非空
            if hasattr(response, 'parts') and response.parts:
                 llm_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            # 如果 parts 為空或不存在，嘗試直接從 response.text 獲取 (適用於舊版或不同結構的回應)
            elif hasattr(response, 'text'):
                 llm_response_text = response.text

            # 檢查是否有提示被封鎖的回饋
            if not llm_response_text and hasattr(response, 'prompt_feedback') and \
               response.prompt_feedback.block_reason != genai.types.BlockReason.BLOCK_REASON_UNSPECIFIED:
                # 將 genai.types.BlockReason 枚舉轉換為可讀的字串
                block_reason_str = genai.types.BlockReason(response.prompt_feedback.block_reason).name
                print(f"警告：提示因 {block_reason_str} 被封鎖。安全評級：{response.prompt_feedback.safety_ratings}")
                return f"錯誤：提示因 {block_reason_str} 被封鎖。"

            print(f"--- 收到 Gemini 回應 ---\n{llm_response_text}\n--- Gemini 回應結束 ---")
            return llm_response_text if llm_response_text else "錯誤：未產生內容或提示有問題。"
        except Exception as e:
            print(f"Gemini API 調用期間發生錯誤：{e}")
            return f"錯誤：產生內容時發生錯誤：{str(e)}"


class GraphOfThoughts:
    def __init__(self, llm_interface):
        self.thoughts = {} # 以 id 儲存所有思維
        self.next_thought_id = 0
        self.llm = llm_interface
        # self.graph_reasoning_state = {} # 未來：用於 GRS
        # self.graph_of_operations = None # 未來：用於 GoO

    def _get_new_id(self):
        new_id = self.next_thought_id
        self.next_thought_id += 1
        return new_id

    def add_thought(self, content, parent_ids=None, score=0.0, generated_by_llm=True):
        new_id = self._get_new_id()
        thought = Thought(content, new_id, score, generated_by_llm)
        self.thoughts[new_id] = thought
        if parent_ids:
            for pid in parent_ids:
                if pid in self.thoughts:
                    self.thoughts[pid].children.append(thought) # 將新思維加到父思維的 children 列表
                    thought.parents.append(self.thoughts[pid]) # 將父思維加到新思維的 parents 列表
        print(f"已新增思維：{thought}")
        return thought

    # --- Prompter 模組 (概念性方法) ---
    def _generate_prompt_for_new_thought(self, task_description, existing_thoughts_content=None, num_new_thoughts=1):
        prompt = f"任務：{task_description}\n"
        if existing_thoughts_content:
            prompt += "基於以下現有資訊/思維：\n"
            for i, content in enumerate(existing_thoughts_content):
                prompt += f"現有思維 {i+1}：{content}\n"
        prompt += f"\n請產生 {num_new_thoughts} 個新的、不同的思維或解決方案步驟，以推進任務解決。請清晰地提供每個新思維。"
        if num_new_thoughts > 1:
            # 指示 LLM 如何格式化多個思維
            prompt += "\n請將每個新思維另起一行開始，可選擇性編號 (例如 '1. <思維內容>')。"
        return prompt

    def _generate_prompt_for_aggregation(self, thoughts_to_aggregate_content, task_description):
        prompt = f"任務：{task_description}\n"
        prompt += "請將以下不同的思維聚合成一個單一、更全面且精煉的思維。請識別核心觀點並將其綜合為一個連貫的摘要或改進的解決方案：\n"
        for i, content in enumerate(thoughts_to_aggregate_content):
            prompt += f"待聚合思維 {i+1}：{content}\n"
        prompt += "\n合併與聚合後的思維："
        return prompt

    def _generate_prompt_for_refinement(self, thought_content, task_description, refinement_instruction):
        prompt = f"任務：{task_description}\n"
        prompt += f"原始思維：{thought_content}\n"
        prompt += f"精煉指示：{refinement_instruction}\n"
        prompt += "\n請僅基於原始思維和指示提供精煉後的思維：" # 強調基於原始思維
        return prompt

    def _generate_prompt_for_scoring(self, thought_content, scoring_criteria):
        # 更清晰地指示 LLM 輸出格式
        prompt = (
            f"您是一位專業評估員。請根據以下標準評估提供的思維：'{scoring_criteria}'。\n"
            f"待評分思維：\n\"\"\"\n{thought_content}\n\"\"\"\n"
            "請僅以下列格式提供您的評估：\n"
            "Score: [一個介於 0.0 (差) 到 1.0 (優) 之間的浮點數]\n"
            "Justification: [對您的分數的簡要解釋]"
        )
        return prompt

    # --- Parser 模組 (概念性方法) ---
    def _parse_llm_response_for_new_thoughts(self, llm_response_text, num_expected_thoughts=1):
        # 檢查是否為從 LLM generate 方法返回的錯誤訊息
        if llm_response_text.startswith("錯誤："):
            return [] # 返回空列表表示沒有成功解析的思維

        # 如果期望單個思維，直接返回
        if num_expected_thoughts == 1:
            return [llm_response_text.strip()]
        else:
            # 嘗試根據常見的編號模式或換行符來分割多個思維
            # 此正則表達式尋找以數字加點、或破折號/星號開頭的行，後面跟著一個空格
            potential_thoughts = re.split(r'\n\s*(?=\d+\.\s|-\s|\*\s)', llm_response_text)

            # 如果分割結果大致符合預期數量 (允許一些額外的說明文字)
            if len(potential_thoughts) >= num_expected_thoughts and len(potential_thoughts) <= num_expected_thoughts + 2 : # 允許一些前導文字
                 # 清理：移除可能存在的編號/項目符號
                cleaned_thoughts = []
                for pt in potential_thoughts:
                    # 移除如 "1. ", "- ", "* " 等前導編號
                    cleaned = re.sub(r'^\s*(\d+\.\s*|-\s*|\*\s*)', '', pt.strip())
                    if cleaned: # 僅在移除編號後仍有內容時加入
                        cleaned_thoughts.append(cleaned)
                if cleaned_thoughts: # 如果清理後仍有思維
                    return cleaned_thoughts[:num_expected_thoughts] # 返回預期數量的思維

            # 如果正則表達式分割不明顯，則退回使用換行符分割
            # 首先嘗試雙換行符，然後是單換行符
            thoughts = llm_response_text.strip().split('\n\n')
            if len(thoughts) < num_expected_thoughts:
                thoughts = llm_response_text.strip().split('\n')
            # 過濾掉空字串並返回預期數量的思維
            return [t.strip() for t in thoughts if t.strip()][:num_expected_thoughts]


    def _parse_llm_response_for_score(self, llm_response_text):
        # 檢查是否為錯誤訊息
        if llm_response_text.startswith("錯誤："):
            return 0.0, llm_response_text # 返回預設分數和錯誤訊息作為理由

        # 使用正則表達式提取分數和理由，忽略大小寫
        score_match = re.search(r"Score:\s*([0-9.]+)", llm_response_text, re.IGNORECASE)
        # DOTALL 使 '.' 可以匹配換行符，以捕獲多行理由
        justification_match = re.search(r"Justification:\s*(.+)", llm_response_text, re.IGNORECASE | re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.0
        justification = justification_match.group(1).strip() if justification_match else "未提供理由或解析錯誤。"

        if not score_match: # 如果未找到分數
            print(f"警告：無法解析分數。原始回應：'{llm_response_text}'")
        return score, justification

    # --- Controller 與圖形操作 ---
    def generate_thoughts(self, task_description, num_thoughts=1, from_thought_ids=None, initial_content_list=None):
        generated_llm_thoughts_objects = [] # 儲存產生的 Thought 物件
        parent_ids_for_new_thoughts = [] # 新思維的父 ID
        base_content_for_prompt = [] # 用於提示的基礎內容

        if from_thought_ids: # 如果基於現有思維產生
            parent_ids_for_new_thoughts.extend(from_thought_ids)
            for tid in from_thought_ids:
                if tid in self.thoughts:
                    base_content_for_prompt.append(self.thoughts[tid].content)
                else:
                    print(f"警告：在產生新思維時，找不到 from_thought_id {tid}。")
        elif initial_content_list: # 如果基於初始內容列表產生 (非 LLM 產生的輸入)
             base_content_for_prompt.extend(initial_content_list)
             # 這些初始內容項目可能尚未成為圖中的「父」思維，
             # 或者它們可以表示為 generated_by_llm=False 的初始思維。

        prompt = self._generate_prompt_for_new_thought(task_description, base_content_for_prompt, num_new_thoughts=num_thoughts)
        llm_response = self.llm.generate(prompt) # 調用 LLM
        parsed_contents = self._parse_llm_response_for_new_thoughts(llm_response, num_thoughts) # 解析回應

        for content in parsed_contents:
            if not content or content.startswith("錯誤：提示因"): # 處理被封鎖的提示或空內容
                print(f"因錯誤或空內容跳過思維產生：'{content}'")
                continue
            # 創建新的 Thought 物件並加入圖中
            thought_obj = self.add_thought(content, parent_ids=parent_ids_for_new_thoughts, generated_by_llm=True)
            generated_llm_thoughts_objects.append(thought_obj)
        return generated_llm_thoughts_objects

    def aggregate_thoughts(self, thought_ids_to_aggregate, task_description):
        # 檢查輸入的 ID 是否有效且存在於圖中
        if not thought_ids_to_aggregate or not all(tid in self.thoughts for tid in thought_ids_to_aggregate):
            print("錯誤：一個或多個用於聚合的思維 ID 未找到或列表為空。")
            return None

        contents = [self.thoughts[tid].content for tid in thought_ids_to_aggregate] # 獲取待聚合思維的內容
        prompt = self._generate_prompt_for_aggregation(contents, task_description) # 產生聚合提示
        llm_response = self.llm.generate(prompt) # 調用 LLM

        if not llm_response or llm_response.startswith("錯誤："): # 檢查 LLM 回應是否有效
            print(f"聚合失敗，原因：錯誤或空回應：'{llm_response}'")
            return None
        aggregated_content = llm_response.strip() # 假設解析器直接返回內容
        # 創建新的聚合思維並加入圖中
        return self.add_thought(aggregated_content, parent_ids=thought_ids_to_aggregate, generated_by_llm=True)

    def refine_thought(self, thought_id, task_description, refinement_instruction="提高清晰度和細節。"):
        if thought_id not in self.thoughts: # 檢查思維 ID 是否存在
            print(f"錯誤：找不到用於精煉的思維 ID {thought_id}。")
            return None

        content_to_refine = self.thoughts[thought_id].content # 獲取待精煉思維的內容
        prompt = self._generate_prompt_for_refinement(content_to_refine, task_description, refinement_instruction) # 產生精煉提示
        llm_response = self.llm.generate(prompt) # 調用 LLM

        if not llm_response or llm_response.startswith("錯誤："): # 檢查 LLM 回應是否有效
            print(f"精煉失敗，原因：錯誤或空回應：'{llm_response}'")
            return None
        refined_content = llm_response.strip() # 解析回應
        # 創建新的精煉思維，原始思維作為其父思維
        return self.add_thought(refined_content, parent_ids=[thought_id], generated_by_llm=True)

    def score_thought_with_llm(self, thought_id, scoring_criteria):
        if thought_id not in self.thoughts: # 檢查思維 ID 是否存在
            print(f"錯誤：找不到用於 LLM 評分的思維 ID {thought_id}。")
            # 即使思維最初未找到，也確保 score 屬性存在
            # self.thoughts[thought_id].score = 0.0 # 這行會導致錯誤，因為 thought_id 不存在於 self.thoughts
            return 0.0 # 如果思維不存在，直接返回預設分數

        thought_content = self.thoughts[thought_id].content # 獲取待評分思維的內容
        prompt = self._generate_prompt_for_scoring(thought_content, scoring_criteria) # 產生評分提示
        llm_response = self.llm.generate(prompt) # 調用 LLM

        score, justification = self._parse_llm_response_for_score(llm_response) # 解析 LLM 回應
        self.thoughts[thought_id].score = score # 更新思維分數
        print(f"已使用 LLM 評分思維 {thought_id}：{score:.2f}。理由：{justification}")
        return score

    def score_thought_programmatically(self, thought_id, score_value):
        """允許基於某些外部程式邏輯設定分數。"""
        if thought_id in self.thoughts:
            self.thoughts[thought_id].score = float(score_value)
            print(f"已透過程式設定思維 {thought_id} 的分數為：{self.thoughts[thought_id].score:.2f}")
        else:
            print(f"錯誤：找不到用於程式化評分的思維 ID {thought_id}。")


    def rank_thoughts(self, top_n=None):
        """根據分數對思維進行排序。"""
        if not self.thoughts: return [] # 如果沒有思維，返回空列表
        # 按分數降序排序，然後按 ID 升序作為次要排序標準 (確保排序穩定性)
        sorted_thoughts = sorted(self.thoughts.values(), key=lambda t: (t.score, -t.id), reverse=True)
        return sorted_thoughts[:top_n] if top_n is not None else sorted_thoughts # 返回前 N 個或所有排序後的思維

    def get_thought(self, thought_id):
        """根據 ID 獲取思維物件。"""
        return self.thoughts.get(thought_id)

    def print_graph(self):
        """印出目前思維圖的結構。"""
        print("\n--- 目前的思維圖 ---")
        if not self.thoughts:
            print("思維圖為空。")
            return
        # 按 ID 排序以便於一致的輸出順序
        for thought_id in sorted(self.thoughts.keys()):
            thought = self.thoughts[thought_id]
            # 清理內容中的換行符以便於單行顯示，並去除首尾空格
            display_content = thought.content[:100].replace('\n', ' ').strip()
            print(f"\n思維 ID：{thought.id} (分數：{thought.score:.2f}, LLM 產生：{thought.generated_by_llm})")
            print(f"  內容：'{display_content}...'")
            parent_ids = [p.id for p in thought.parents]
            children_ids = [c.id for c in thought.children]
            print(f"  父思維 ID：{parent_ids if parent_ids else '無'}")
            print(f"  子思維 ID：{children_ids if children_ids else '無'}")
        print("--- 思維圖結束 ---")


# --- 範例用法 ---
if __name__ == "__main__":
    # 重要提示：請將您的 API 金鑰設定為環境變數或直接在此處提供。
    # 為安全起見，生產環境中建議使用環境變數。

    # 直接使用您提供的 API 金鑰
    GEMINI_API_KEY = "AIzaSyDIwLMh_alSR68tezeO1Jme4swT46GXs3w"
    print(f"正在使用提供的 API 金鑰: ...{GEMINI_API_KEY[-4:]}") # 僅顯示金鑰末幾位以確認

    try:
        llm_api = GeminiLLM(api_key=GEMINI_API_KEY)

    except ValueError as e: # 捕獲 GeminiLLM 初始化時的 API 金鑰錯誤
        print(f"初始化錯誤：{e}")
        print("程式結束。請提供一個有效的 Gemini API 金鑰。")
        exit()
    except Exception as e: # 捕獲 genai 初始化時可能發生的其他潛在錯誤
        print(f"LLM 初始化期間發生未預期錯誤：{e}")
        exit()


    got_system = GraphOfThoughts(llm_api)

    # --- 情境：腦力激盪並發展一個短篇故事概念 ---
    main_task = "構思一個關於圖書管理員發現一本會根據讀者內心深處的渴望自動書寫內容，但會帶來意想不到後果的魔法書的短篇故事大綱。"
    print(f"\n目標：{main_task}\n")

    # 1. 初始腦力激盪：產生一些初步的情節點或角色概念。
    print("\n步驟 1：初始腦力激盪 (產生 2 個初步概念)")
    # 我們可以提供非 LLM 產生的初始種子內容
    # 在此範例中，讓 LLM 產生最初的幾個思維
    initial_ideas = got_system.generate_thoughts(
        task_description="為故事產生不同的核心概念：" + main_task,
        num_thoughts=2
    )
    if not initial_ideas or len(initial_ideas) < 2: # 確保產生了足夠的初始概念
        print("未能產生足夠的初始概念。結束範例。")
        exit()

    idea1 = initial_ideas[0]
    idea2 = initial_ideas[1]
    # 使用 LLM 評分初始概念
    got_system.score_thought_with_llm(idea1.id, "原創性和衝突潛力")
    got_system.score_thought_with_llm(idea2.id, "情感深度和讀者參與度")


    # 2. 詳細闡述：選擇一個概念並加以詳細說明。
    # 為確保範例可執行，即使評分失敗，也選擇一個概念繼續
    chosen_idea_id_for_elaboration = idea1.id # 預設選擇第一個
    if idea1.score < idea2.score: # 如果第二個分數更高，則選擇第二個
        chosen_idea_id_for_elaboration = idea2.id
    print(f"\n步驟 2：詳細闡述概念 {chosen_idea_id_for_elaboration}")

    elaborated_thoughts = got_system.generate_thoughts(
        task_description="詳細闡述以下概念，包含主要角色和魔法書魔力的本質：" + main_task,
        num_thoughts=1,
        from_thought_ids=[chosen_idea_id_for_elaboration] # 基於選定的初始概念
    )
    elaborated_idea = None # 初始化 elaborated_idea
    if not elaborated_thoughts:
        print(f"未能詳細闡述思維 {chosen_idea_id_for_elaboration}。跳過此分支的後續步驟。")
    else:
        elaborated_idea = elaborated_thoughts[0]
        got_system.score_thought_with_llm(elaborated_idea.id, "連貫性和細節豐富度")

        # 3. 精煉：精煉已闡述的思維，加入特定的轉折。
        # 確保 elaborated_idea 存在
        if elaborated_idea:
            print(f"\n步驟 3：為已闡述概念 {elaborated_idea.id} 加入轉折並精煉")
            refined_thought = got_system.refine_thought(
                elaborated_idea.id,
                task_description="精煉故事情節：" + main_task,
                refinement_instruction="引入一個與書實現願望相關的重大且具有諷刺意味的後果。"
            )
            if refined_thought:
                got_system.score_thought_with_llm(refined_thought.id, "轉折的影響力和主題一致性")
        else:
            print("由於未能成功闡述概念，跳過精煉步驟。")


    # 4. 另一條路徑：從第二個 (或未被選中進行闡述的) 初始概念產生不同的詳細說明。
    alternative_idea_id = idea2.id if chosen_idea_id_for_elaboration == idea1.id else idea1.id
    print(f"\n步驟 4：探索來自概念 {alternative_idea_id} 的替代路徑")
    alt_elaboration_thoughts = got_system.generate_thoughts(
        task_description="為此概念提供不同的角度或發展，著重於圖書管理員的內心掙扎：" + main_task,
        num_thoughts=1,
        from_thought_ids=[alternative_idea_id]
    )
    alt_elaborated_idea = None # 初始化
    if not alt_elaboration_thoughts:
         print(f"未能詳細闡述思維 {alternative_idea_id}。跳過此分支的後續步驟。")
    else:
        alt_elaborated_idea = alt_elaboration_thoughts[0]
        got_system.score_thought_with_llm(alt_elaborated_idea.id, "內心衝突的合理性和角色弧線潛力")

        # 5. 聚合：嘗試結合主要路徑精煉後的元素和替代路徑的元素
        # (這是一個概念性步驟，真正的聚合需要仔細的提示設計)
        # 需要 refined_thought 和 alt_elaborated_idea 都存在才能進行此步驟
        if 'refined_thought' in locals() and refined_thought and alt_elaborated_idea:
            print(f"\n步驟 5：聚合精煉後的概念 ({refined_thought.id}) 與替代路徑的闡述 ({alt_elaborated_idea.id})")
            aggregated_thought = got_system.aggregate_thoughts(
                [refined_thought.id, alt_elaborated_idea.id], # 聚合兩個不同的發展方向
                task_description="將以下兩個故事發展中最引人入勝的方面綜合起來，形成一個統一且更強大的情節大綱：" + main_task
            )
            if aggregated_thought:
                got_system.score_thought_with_llm(aggregated_thought.id, "整體大綱的凝聚力、完整性和吸引力")
        else:
            missing_elements = []
            if not ('refined_thought' in locals() and refined_thought) : missing_elements.append("主要路徑的精煉思維")
            if not alt_elaborated_idea : missing_elements.append("替代路徑的闡述思維")
            print(f"由於缺少以下元素，跳過聚合步驟：{', '.join(missing_elements)}。")


    # 6. 最終排序並顯示思維圖
    got_system.print_graph() # 印出所有思維及其關係

    print("\n--- 最終所有思維的排序 ---")
    top_thoughts = got_system.rank_thoughts() # 獲取所有排序後的思維
    for i, thought in enumerate(top_thoughts):
        # 清理內容以便於單行顯示
        display_content = thought.content[:80].replace('\n', ' ').strip()
        print(f"{i+1}. (ID: {thought.id}, 分數: {thought.score:.2f}) - {display_content}...")


    print("\n--- 範例用法結束 ---")
    print("注意：如果 API 金鑰無效或調用失敗，LLM 回應將被模擬。")
    print("請確保已安裝 google-generativeai 套件。")