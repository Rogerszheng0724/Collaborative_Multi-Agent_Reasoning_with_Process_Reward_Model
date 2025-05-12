# -*- coding: utf-8 -*-
import google.generativeai as genai
import os

# --- 必要的輔助類別 (來自 GraphOfThoughts 的簡化版本) ---
class Thought:
    """
    代表思考過程中的一個節點或一個想法。
    這是 OptionThought 和 LayerThought 的基礎類別。
    """
    def __init__(self, content, id, score=0):
        self.id = id  # 想法的唯一識別碼
        self.content = content  # 想法的內容 (例如，文字、資料)
        self.score = score  # 評估此想法的分數
        self.parents = []  # 此想法的父節點列表
        self.children = [] # 此想法的子節點列表

    def __repr__(self):
        return f"Thought(id={self.id}, score={self.score}, content='{self.content[:50]}...')"

class GraphOfThoughts:
    """
    代表一個由 Thought 物件組成的圖形結構。
    這是 LayerOfThoughts 的基礎類別。
    """
    def __init__(self, llm_interface):
        self.thoughts = {}  # 儲存圖中所有 Thought 物件的字典，以 id 為鍵
        self.llm = llm_interface # 用於與大型語言模型互動的介面

    def add_thought(self, content, thought_id=None, parent_ids=None):
        """
        在圖中新增一個 Thought。
        """
        if thought_id is None:
            thought_id = f"thought_{len(self.thoughts)}"
        
        # 檢查 ID 是否已存在
        if thought_id in self.thoughts:
            print(f"警告: ID 為 '{thought_id}' 的 Thought 已存在。將不會重複新增。")
            return self.thoughts[thought_id]

        thought = Thought(content, thought_id)
        self.thoughts[thought_id] = thought

        if parent_ids:
            for pid in parent_ids:
                if pid in self.thoughts:
                    parent_thought = self.thoughts[pid]
                    thought.parents.append(parent_thought)
                    parent_thought.children.append(thought)
                else:
                    print(f"警告: 找不到 ID 為 '{pid}' 的父節點 Thought。")
        return thought

    def get_thought(self, thought_id):
        return self.thoughts.get(thought_id)

    def score_thought(self, thought_id, prompt_template, **kwargs):
        """
        使用 LLM 評估一個 Thought。
        (此為簡化版本，實際評分可能更複雜)
        """
        thought = self.get_thought(thought_id)
        if not thought:
            print(f"錯誤: 找不到 ID 為 '{thought_id}' 的 Thought。")
            return 0

        # 建立評分提示
        # 例如: prompt_template = "評估以下想法的品質，滿分10分: {content}"
        prompt = prompt_template.format(content=thought.content, **kwargs)
        
        try:
            response_text = self.llm.generate(prompt)
            # 解析回應以獲取分數 (假設 LLM 直接回傳分數)
            score = float(response_text.strip()) # 簡化處理，實際可能需要更複雜的解析
            thought.score = score
            print(f"Thought '{thought_id}' 已評分: {score}")
        except Exception as e:
            print(f"評分 Thought '{thought_id}' 時發生錯誤: {e}")
            thought.score = 0 # 錯誤時給予預設分數
        return thought.score

# --- Gemini API 介面 ---
class GeminiLLMInterface:
    """
    一個使用 Gemini API 與大型語言模型互動的介面。
    """
    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        """
        初始化 GeminiLLMInterface。

        Args:
            api_key (str): 您的 Gemini API 金鑰。
            model_name (str): 要使用的 Gemini 模型名稱。
        """
        if not api_key:
            raise ValueError("Gemini API 金鑰是必需的。")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"Gemini LLM 介面已使用模型 '{model_name}' 初始化。")

    def generate(self, prompt_text):
        """
        使用提供的提示文字生成內容。

        Args:
            prompt_text (str): 要傳送給 LLM 的提示。

        Returns:
            str: LLM 生成的文字回應。
        """
        try:
            response = self.model.generate_content(prompt_text)
            # 檢查是否有候選回應並且第一個候選回應有內容
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                # 處理沒有有效回應的情況
                # 您可以檢查 response.prompt_feedback 來了解原因
                print(f"警告: Gemini API 未回傳有效內容。提示回饋: {response.prompt_feedback}")
                return "Gemini API 未回傳內容。"
        except Exception as e:
            print(f"呼叫 Gemini API 時發生錯誤: {e}")
            # 可以考慮拋出例外或回傳錯誤訊息
            return f"Gemini API 錯誤: {str(e)}"


# --- Layer-of-Thoughts (LoT) 實作 (修改自使用者提供的 LOT.py) ---
class OptionThought(Thought): # 繼承自簡化的 Thought
    def __init__(self, content, id, criterion, level=1, score=0):
        super().__init__(content, id, score)
        self.criterion = criterion # 此想法處理的特定標準
        self.level = level # 層內的優先等級

    def __repr__(self):
        return f"OptionThought(id={self.id}, level={self.level}, criterion='{self.criterion}', score={self.score}, content='{self.content[:20]}...')"

class LayerThought(Thought): # 繼承自簡化的 Thought
    def __init__(self, content, id, layer_index, score=0):
        super().__init__(content, id, score) # 內容可以是此層的概念步驟/指令
        self.layer_index = layer_index
        self.option_thoughts = [] # 此層內生成的 OptionThought

    def __repr__(self):
        return f"LayerThought(id={self.id}, layer_index={self.layer_index}, options={len(self.option_thoughts)}, content='{self.content[:20]}...')"

class LayerOfThoughts(GraphOfThoughts): # 繼承自簡化的 GraphOfThoughts
    def __init__(self, llm_interface):
        super().__init__(llm_interface)
        self.layers = [] # LayerThought 物件的列表

    def add_layer_thought(self, conceptual_step_description):
        layer_index = len(self.layers)
        # LayerThought 的內容是其概念步驟或指令
        layer_thought_id = f"L{layer_index}_main" # 層次想法的唯一 ID
        
        # 檢查 ID 是否已存在於 self.thoughts 中
        if layer_thought_id in self.thoughts:
            print(f"警告: ID 為 '{layer_thought_id}' 的 LayerThought 已存在。將不會重複新增。")
            # 如果已存在，可以選擇返回現有的或更新它，這裡選擇返回現有的
            return self.thoughts[layer_thought_id]

        layer_thought = LayerThought(conceptual_step_description, layer_thought_id, layer_index)
        self.thoughts[layer_thought_id] = layer_thought # 加入全域 thoughts 字典
        self.layers.append(layer_thought)
        print(f"已新增層次 {layer_index}: {layer_thought}")
        return layer_thought

    def _generate_prompt_for_option_thought_criteria(self, layer_thought_content, previous_layer_output=None):
        # 提示 LLM 為目前層次建議標準
        prompt = f"針對概念步驟: '{layer_thought_content}'\n"
        if previous_layer_output:
            prompt += f"基於先前的輸出: '{previous_layer_output}'\n"
        prompt += "請建議一系列標準 (或可探索的選項) 來生成部分解決方案。如果標準有優先順序，請註明 (例如：標準 A (等級 1); 標準 B (等級 1); 標準 C (等級 2))。請僅回傳標準列表，以分號分隔。"
        return prompt
        
    def _generate_prompt_for_option_thought_solution(self, criterion, layer_conceptual_step, previous_layer_output=None):
        prompt = f"概念步驟: {layer_conceptual_step}\n"
        if previous_layer_output:
            prompt += f"先前輸出內容: {previous_layer_output}\n"
        prompt += f"請針對標準 '{criterion}' 生成一個部分解決方案。"
        return prompt

    def generate_option_thoughts_for_layer(self, layer_id, previous_layer_aggregated_output=None):
        if layer_id not in self.thoughts or not isinstance(self.thoughts[layer_id], LayerThought):
            print(f"錯誤: 找不到 ID 為 {layer_id} 的 LayerThought。")
            return []

        current_layer_thought = self.thoughts[layer_id]
        
        # 步驟 1: 從 LLM 獲取此層的標準
        criteria_prompt = self._generate_prompt_for_option_thought_criteria(current_layer_thought.content, previous_layer_aggregated_output)
        print(f"\n向 LLM 請求標準，提示:\n{criteria_prompt}")
        llm_criteria_response = self.llm.generate(criteria_prompt)
        print(f"LLM 回應的標準: {llm_criteria_response}")
        parsed_criteria = self._parse_criteria_from_llm(llm_criteria_response) 

        generated_options = []
        for i, crit_info in enumerate(parsed_criteria):
            criterion_text = crit_info['text']
            criterion_level = crit_info.get('level', 1) # 預設為等級 1

            # 步驟 2: 為每個標準生成解決方案
            solution_prompt = self._generate_prompt_for_option_thought_solution(
                criterion_text,
                current_layer_thought.content,
                previous_layer_aggregated_output
            )
            print(f"\n向 LLM 請求 '{criterion_text}' 的解決方案，提示:\n{solution_prompt}")
            llm_solution_response = self.llm.generate(solution_prompt)
            # 在 LoT 中，解析 LLM 回應通常是直接使用其文字輸出
            solution_content = llm_solution_response 
            
            option_id = f"{current_layer_thought.id}_Opt{i}"
            
            # 檢查 OptionThought ID 是否已存在
            if option_id in self.thoughts:
                print(f"警告: ID 為 '{option_id}' 的 OptionThought 已存在。將不會重複新增。")
                option_thought = self.thoughts[option_id] # 使用現有的
            else:
                option_thought = OptionThought(solution_content, option_id, criterion_text, level=criterion_level)
                self.thoughts[option_id] = option_thought # 加入全域 thoughts
            
            # 避免重複添加
            if option_thought not in current_layer_thought.option_thoughts:
                 current_layer_thought.option_thoughts.append(option_thought)
            
            # 連結 OptionThought 到其父 LayerThought
            if current_layer_thought not in option_thought.parents:
                option_thought.parents.append(current_layer_thought)
            if option_thought not in current_layer_thought.children:
                current_layer_thought.children.append(option_thought)
            
            generated_options.append(option_thought)
            print(f"已為層次 {current_layer_thought.layer_index} 生成 OptionThought: {option_thought}")
            
        return generated_options

    def _parse_criteria_from_llm(self, llm_response_text):
        # 解析 "標準 A (等級 1); 標準 B" 成結構化資料
        criteria = []
        if not llm_response_text or not isinstance(llm_response_text, str):
            print(f"警告: LLM 回應的標準為空或格式不正確: {llm_response_text}")
            return [{'text': "預設標準 (因解析錯誤)", 'level': 1}]

        parts = llm_response_text.split(';')
        for part in parts:
            part = part.strip()
            if not part: continue
            level = 1 # 預設
            text = part
            if "(等級 " in part and part.endswith(")"):
                try:
                    text_part, level_part = part.rsplit("(等級 ", 1)
                    text = text_part.strip()
                    level_str = level_part.replace(")", "").strip()
                    level = int(level_str)
                except ValueError:
                    print(f"警告: 無法解析標準 '{part}' 中的等級，使用預設等級 1。")
                    level = 1 # 解析失敗時的備援
                    text = part # 維持原始文字作為標準文字
            criteria.append({'text': text, 'level': level})
        
        if not criteria: # 解析失敗時的備援
             print(f"警告: 無法從 '{llm_response_text}' 解析出任何標準，使用回應本身作為單一標準。")
             return [{'text': llm_response_text.strip(), 'level': 1}]
        return criteria
        
    def aggregate_option_thoughts_in_layer(self, layer_id, metric='all_content', **kwargs):
        if layer_id not in self.thoughts or not isinstance(self.thoughts[layer_id], LayerThought):
            print(f"錯誤: 找不到 ID 為 {layer_id} 的 LayerThought 進行聚合。")
            return None

        current_layer_thought = self.thoughts[layer_id]
        options = current_layer_thought.option_thoughts
        if not options:
            print(f"層次 {current_layer_thought.layer_index} 中沒有 OptionThought 可供聚合。")
            # 根據 LoT 論文，如果沒有選項，可能意味著前一層的輸入沒有通過標準
            # 或者 LLM 沒有生成任何標準/選項。
            # 在這種情況下，回傳 None 或一個表示失敗的特定訊息是合理的。
            return f"層次 {current_layer_thought.layer_index} 沒有生成任何選項。"


        # 聚合邏輯的預留位置，基於指標
        # 這通常會先對每個 OptionThought 進行評分
        for opt_thought in options:
            # 假設一個簡單的評分 (例如，基於與標準的相關性，或內容長度)
            # 您可以在這裡呼叫 self.score_thought()，如果需要 LLM 評分
            # self.score_thought(opt_thought.id, "評估此選項與標準 '{criterion}' 的相關性 (0-1): {content}", criterion=opt_thought.criterion)
            opt_thought.score = len(opt_thought.content) # 模擬評分

        aggregated_content = f"從層次 {current_layer_thought.layer_index} 使用 {metric} 聚合的結果:\n"
        
        if metric == 'max_score_option': 
            options.sort(key=lambda ot: ot.score, reverse=True)
            if options:
                 aggregated_content += f"最佳選項 (標準: {options[0].criterion}, 等級: {options[0].level}, 分數: {options[0].score}):\n{options[0].content}"
            else:
                aggregated_content += "沒有可選擇的最佳選項。"
        elif metric == 'all_content': # 範例: 合併所有內容
            options.sort(key=lambda ot: (ot.level, -ot.score)) # 按等級排序，然後按分數降序
            for opt in options:
                aggregated_content += f"- (標準: {opt.criterion}, 等級: {opt.level}, 分數: {opt.score}): {opt.content}\n"
        # 可以加入其他指標: at-least-k, locally-better, max-weight 等 (參考 LoT 論文)
        else:
            aggregated_content += "預設聚合: "
            if options:
                options.sort(key=lambda ot: ot.score, reverse=True)
                aggregated_content += options[0].content 
            else:
                aggregated_content += "沒有內容"
        
        print(f"已聚合層次 {current_layer_thought.layer_index} 中的選項。輸出: {aggregated_content[:150]}...")
        return aggregated_content # 直接回傳內容

    def run_pipeline(self, conceptual_steps, initial_input=None):
        """
        執行 LoT 管線。

        Args:
            conceptual_steps (list): 一個字串列表，每個字串描述一個層次的主要目標。
            initial_input (str, optional): 管線的初始輸入。預設為 None。
        """
        previous_layer_output = initial_input
        final_output = None

        for i, step_description in enumerate(conceptual_steps):
            print(f"\n--- 正在處理層次 {i}: {step_description} ---")
            # 1. 建立 LayerThought
            layer_thought = self.add_layer_thought(step_description)
            
            # 2. 為此層次生成 OptionThought
            # 生成 OptionThought 的輸入是前一層聚合的輸出
            option_thoughts_generated = self.generate_option_thoughts_for_layer(layer_thought.id, previous_layer_output)
            
            if not option_thoughts_generated:
                print(f"層次 {i} 未能生成任何 OptionThought。可能需要檢查 LLM 回應或提示。")
                # 根據 LoT 論文，如果一個層次沒有生成選項，可能需要回溯或調整策略
                # 這裡我們先簡單地將 previous_layer_output 設為一個提示訊息，然後繼續或中斷
                # previous_layer_output = f"層次 {i} ({step_description}) 未產生任何選項，基於輸入: {previous_layer_output}"
                # final_output = previous_layer_output # 更新最終輸出為目前的狀態
                # break # 或者決定中斷管線

            # 3. 聚合此層次的 OptionThought
            # 每個層次的聚合指標可以不同
            aggregated_output_content = self.aggregate_option_thoughts_in_layer(layer_thought.id, metric='all_content') # 或 'max_score_option'
            
            if aggregated_output_content is None or aggregated_output_content.startswith(f"層次 {i} 沒有生成任何選項。"):
                message = aggregated_output_content if aggregated_output_content else f"層次 {i} 未能產生聚合輸出。"
                print(f"{message} 正在停止管線。")
                # 根據 LoT 論文 [cite: 404]，這裡可以實作回溯或精煉邏輯
                final_output = previous_layer_output # 保留前一個成功的輸出作為最終輸出
                break
            
            previous_layer_output = aggregated_output_content
            final_output = aggregated_output_content # 持續追蹤最後一次成功的聚合結果

        return final_output


# --- 主要執行部分 ---
if __name__ == "__main__":
    # 重要：請將 'YOUR_API_KEY' 替換為您的 Gemini API 金鑰
    # 建議使用環境變數來儲存 API 金鑰，例如：
    # API_KEY = os.environ.get("GEMINI_API_KEY")
    # if not API_KEY:
    #     raise ValueError("請設定 GEMINI_API_KEY 環境變數。")
    
    API_KEY = "AIzaSyDIwLMh_alSR68tezeO1Jme4swT46GXs3w" # <--- 在此處填入您的 Gemini API 金鑰

    if API_KEY == "YOUR_API_KEY":
        print("*****************************************************************")
        print("警告：您尚未設定 Gemini API 金鑰。")
        print("請將程式碼中的 'YOUR_API_KEY' 替換為您真實的 Gemini API 金鑰。")
        print("*****************************************************************")
        # 為了讓程式碼在沒有金鑰的情況下至少能被解析，我們這裡不直接退出
        # 但後續的 API 呼叫會失敗。
        # exit() # 如果希望在沒有金鑰時直接退出，取消此行註解

    try:
        # 初始化 Gemini LLM 介面
        # 您可以選擇不同的模型，例如 "gemini-1.5-flash-latest" 或 "gemini-1.0-pro"
        # "gemini-pro" 是 "gemini-1.0-pro" 的一個穩定版本別名
        llm_api_lot = GeminiLLMInterface(api_key=API_KEY, model_name="gemini-1.5-flash-latest") 
    except ValueError as e:
        print(f"初始化 LLM 介面時發生錯誤: {e}")
        exit()
    except Exception as e: # 捕捉 genai.configure 可能拋出的其他錯誤
        print(f"設定 Gemini API 時發生預期外的錯誤: {e}")
        exit()


    # 初始化 LayerOfThoughts 系統
    lot_system = LayerOfThoughts(llm_api_lot)

    # 為任務定義概念步驟 (範例：規劃一個週末短途旅行)
    # conceptual_steps_trip_planning = [
    # "目的地選擇：根據預算和興趣，列出3個潛在的國內旅遊地點。",
    # "活動規劃：針對評分最高的潛在目的地，規劃至少3個主要活動或景點。",
    # "行程細化：為選定的目的地和活動，制定一個兩天一夜的初步行程大綱，包括交通和住宿考量。"
    # ]
    # initial_query_trip = "我想要一個預算友善且包含自然風光的週末短途旅行。"

    # 範例：為法律文件檢索任務定義概念步驟 (來自 LoT 論文)
    # 這些提示是英文的，因為原始 LoT.py 中的範例是英文的
    # 如果您的 Gemini 模型和設定偏好中文，可以將其翻譯
    conceptual_steps_legal_retrieval = [
        "Keyword Filtering: Identify key terms from the user query to find potentially relevant articles from a hypothetical document set. The user query is: 'A 15-year-old made a contract without parental consent, is it valid?'", # Layer 0
        "Semantic Filtering: Apply broader semantic conditions to the filtered articles (output from Layer 0), ordered by generality. For example, check relevance to 'contract law' or 'juridical act'.", # Layer 1
        "Final Confirmation: Verify if the remaining articles (output from Layer 1) directly answer or are highly relevant to the original query: 'A 15-year-old made a contract without parental consent, is it valid?'" # Layer 2
    ]
    initial_query_legal = "A 15-year-old made a contract without parental consent, is it valid?"


    print(f"\nLoT 管線開始執行，使用概念步驟: {conceptual_steps_legal_retrieval}")
    # 執行 LoT 管線
    # 您可以傳入一個初始查詢或上下文作為第一個 Layer 的 `previous_layer_output`
    final_result = lot_system.run_pipeline(conceptual_steps_legal_retrieval, initial_input=initial_query_legal)

    print(f"\n--- LoT 管線的最終輸出 ---")
    if final_result:
        print(final_result)
    else:
        print("管線未能產生最終輸出。")

    # 顯示圖中所有的 Thought (可選)
    # print("\n--- 圖中所有的 Thought ---")
    # for thought_id, thought in lot_system.thoughts.items():
    #     print(thought)
    #     if isinstance(thought, LayerThought):
    #         print(f"  選項: {[opt.id for opt in thought.option_thoughts]}")

