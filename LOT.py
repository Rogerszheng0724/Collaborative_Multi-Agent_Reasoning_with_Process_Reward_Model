# -*- coding: utf-8 -*-
import google.generativeai as genai
import os
import re # 新增 re 模組

# --- 輔助：預設 Logger ---
class DefaultLogger:
    def info(self, message): print(f"[LOT INFO] {message}")
    def warning(self, message): print(f"[LOT WARNING] {message}")
    def error(self, message): print(f"[LOT ERROR] {message}")

# --- 必要的輔助類別 ---
class Thought:
    def __init__(self, content, id, score=0.0, prm_justification="尚未評估"): # 統一分數為 float，新增 prm_justification
        self.id = id
        self.content = content
        self.score = score # 此分數將主要由 PRM 風格評估決定
        self.prm_justification = prm_justification
        self.parents = []
        self.children = []

    def __repr__(self):
        return f"Thought(id={self.id}, score={self.score:.2f}, content='{self.content[:50]}...')"

class GraphOfThoughts: # LOT 中的 GraphOfThoughts 是一個基礎，主要被 LayerOfThoughts 繼承和使用
    def __init__(self, llm_interface, logger=None): # 添加 logger
        self.thoughts = {}
        self.llm = llm_interface # 這個 llm 主要用於 LOT 內部操作，PRM 評估可能使用不同的 LLM
        self.logger = logger if logger else DefaultLogger()
        # PRM 評估器 LLM，預設與操作 LLM 相同，但可以從外部指定
        self.prm_evaluator_llm = llm_interface

    def add_thought_object(self, thought_obj): # 允許直接添加 Thought 物件實例
        if thought_obj.id in self.thoughts:
            self.logger.warning(f"ID 為 '{thought_obj.id}' 的 Thought 已存在。將不會重複新增。")
            return self.thoughts[thought_obj.id]
        self.thoughts[thought_obj.id] = thought_obj
        # 父子關係應由創建 Thought 物件的邏輯處理
        return thought_obj

    def get_thought(self, thought_id):
        return self.thoughts.get(thought_id)

    def _generate_prm_style_scoring_prompt_for_lot_artifact(self, artifact_content, artifact_type, layer_conceptual_step, main_task_description):
        """
        為 LOT 產生的中間成果（如選項思維、層次聚合輸出）產生 PRM 風格的評分提示。
        artifact_type: "選項思維" 或 "層次聚合輸出"
        layer_conceptual_step: 當前層次的概念步驟描述
        main_task_description: 整體的主要任務目標
        """
        prompt = (
            f"您是一位專家級的評估員。\n"
            f"主要任務目標：'{main_task_description}'\n"
            f"當前正在處理的層次化概念步驟：'{layer_conceptual_step}'\n\n"
            f"待評估的「{artifact_type}」內容如下：\n\"\"\"\n{artifact_content}\n\"\"\"\n\n"
            "評估指示：\n"
            f"1.  對「層次概念步驟」的貢獻：此「{artifact_type}」在多大程度上有效地實現了上述「層次化概念步驟」的目標？\n"
            f"2.  對「主要任務目標」的推進：此「{artifact_type}」（作為當前層次的成果）是否有助於最終完成「主要任務目標」？它是否為後續步驟打下了良好基礎，還是可能導致偏離方向？\n"
            "3.  清晰度與可行性：此成果本身是否清晰、具體？如果它是一個計劃或部分解決方案，它是否具有初步的可行性？\n\n"
            "請提供一個總體評分和簡要理由。\n"
            "輸出格式（嚴格遵守）：\n"
            "Score: [一個介於 0.0 (非常差/無助益) 到 1.0 (非常好/極具潛力) 之間的浮點數]\n"
            "Justification: [對您的分數的簡要解釋，說明其如何以及為何有助於或無助於完成層次目標和主要任務目標]"
        )
        return prompt

    def _parse_llm_response_for_prm_score(self, llm_response_text):
        if not llm_response_text or llm_response_text.startswith("錯誤 (LOT):") or llm_response_text.startswith("LLM 未初始化") or llm_response_text.startswith("LLM 錯誤"):
            self.logger.warning(f"LLM 回應無效或為錯誤訊息，無法解析 PRM 分數: {llm_response_text}")
            return 0.0, f"PRM 評分失敗：LLM 回應無效 ({llm_response_text})"

        score_match = re.search(r"Score:\s*([0-9.]+)", llm_response_text, re.IGNORECASE)
        justification_match = re.search(r"Justification:\s*(.+)", llm_response_text, re.IGNORECASE | re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.0
        justification = justification_match.group(1).strip() if justification_match else "未提供理由或解析錯誤。"

        if not score_match:
            self.logger.warning(f"無法從回應中解析 PRM 分數。原始回應：'{llm_response_text}'")
        return score, justification

    def _evaluate_lot_artifact_with_prm(self, artifact_content, artifact_type, layer_conceptual_step, main_task_description):
        """ 使用 PRM 評估器評估 LOT 的中間成果。 """
        if not self.prm_evaluator_llm: # 檢查 prm_evaluator_llm
            self.logger.error("PRM 評估器 LLM 未設定。無法評估。")
            return 0.0, "PRM 評估器未設定"
        
        prompt = self._generate_prm_style_scoring_prompt_for_lot_artifact(artifact_content, artifact_type, layer_conceptual_step, main_task_description)
        llm_response = self.prm_evaluator_llm.generate(prompt) # 使用 prm_evaluator_llm
        score, justification = self._parse_llm_response_for_prm_score(llm_response)
        self.logger.info(f"LOT 成果 ({artifact_type} for layer '{layer_conceptual_step[:30]}...') PRM 評估 - 分數: {score:.2f}")
        return score, justification


# --- Gemini API 介面 ---
class GeminiLLMInterface:
    def __init__(self, api_key, model_name="gemini-1.5-flash-latest", logger=None): # 添加 logger
        self.model = None
        self.logger = logger if logger else DefaultLogger()
        if not api_key:
            self.logger.error("Gemini API 金鑰是必需的。LOT LLM 將無法運作。")
            return # 允許創建實例，但模型將為 None
            
        try:
            # 簡化 API key 配置，假設外部（如 MASOrchestrator）已處理 genai.configure
            # 如果 LOT 單獨運行，則需要在其 __main__ 中配置
            self.model = genai.GenerativeModel(model_name)
            self.logger.info(f"Gemini LLM 介面已使用模型 '{model_name}' 初始化。")
        except Exception as e:
            self.logger.error(f"初始化 LOT Gemini GenerativeModel ({model_name}) 失敗: {e}")
            self.model = None


    def generate(self, prompt_text, temperature=0.7): # 添加 temperature
        if not self.model:
            self.logger.error("LOT.GeminiLLMInterface: LLM 模型未初始化。無法生成內容。")
            return "錯誤 (LOT): LLM 未初始化"
        try:
            self.logger.info(f"\n--- 正在發送提示到 Gemini (LOT 操作 LLM) ---\n{prompt_text[:300]}...\n--- Gemini 提示結束 (LOT 操作 LLM) ---")
            response = self.model.generate_content(
                prompt_text,
                generation_config=genai.types.GenerationConfig(temperature=temperature)
            )
            llm_response_text = ""
            if hasattr(response, 'parts') and response.parts:
                 llm_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            elif hasattr(response, 'text'):
                 llm_response_text = response.text
            
            if not llm_response_text and hasattr(response, 'prompt_feedback') and \
               response.prompt_feedback.block_reason != genai.types.BlockReason.BLOCK_REASON_UNSPECIFIED:
                block_reason_str = genai.types.BlockReason(response.prompt_feedback.block_reason).name
                self.logger.warning(f"警告 (LOT): 提示因 {block_reason_str} 被封鎖。")
                return f"錯誤 (LOT): 提示因 {block_reason_str} 被封鎖。"

            self.logger.info(f"--- 收到 Gemini 回應 (LOT 操作 LLM) ---\n{llm_response_text[:300]}...\n--- Gemini 回應結束 (LOT 操作 LLM) ---")
            return llm_response_text if llm_response_text else "錯誤 (LOT)：未產生內容或提示有問題。"

        except Exception as e:
            self.logger.error(f"呼叫 Gemini API 時發生錯誤 (LOT): {e}")
            return f"錯誤 (LOT): Gemini API 錯誤 - {str(e)}"


# --- Layer-of-Thoughts (LoT) 實作 ---
class OptionThought(Thought):
    def __init__(self, content, id, criterion, level=1, score=0.0, prm_justification="尚未評估"): # 繼承 score 和 prm_justification
        super().__init__(content, id, score, prm_justification)
        self.criterion = criterion # 產生此選項所依據的標準
        self.level = level # 標準的優先級別 (如果有的話)

    def __repr__(self):
        return f"OptionThought(id={self.id}, level={self.level}, criterion='{self.criterion}', score={self.score:.2f}, content='{self.content[:20]}...')"

class LayerThought(Thought):
    def __init__(self, content, id, layer_index, score=0.0, prm_justification="尚未評估"): # content 是概念步驟描述
        super().__init__(content, id, score, prm_justification) # 此 score 將代表該層次聚合輸出的 PRM 分數
        self.layer_index = layer_index
        self.option_thoughts = [] # 屬於此層次的 OptionThought 列表

    def __repr__(self):
        return f"LayerThought(id={self.id}, layer_index={self.layer_index}, options={len(self.option_thoughts)}, score={self.score:.2f}, content='{self.content[:20]}...')"

class LayerOfThoughts(GraphOfThoughts):
    def __init__(self, llm_interface, logger=None, prm_evaluator_llm=None): # 添加 prm_evaluator_llm
        super().__init__(llm_interface, logger)
        self.layers = [] # LayerThought 物件的有序列表
        # 如果未提供專用的 PRM 評估 LLM，則預設使用與操作相同的 LLM
        self.prm_evaluator_llm = prm_evaluator_llm if prm_evaluator_llm else llm_interface
        if not self.prm_evaluator_llm:
             self.logger.warning("LOT 未配置 PRM 評估器 LLM。PRM 評分功能將受限。")


    def add_layer_thought(self, conceptual_step_description):
        layer_index = len(self.layers)
        layer_thought_id = f"L{layer_index}_main"
        
        if layer_thought_id in self.thoughts:
            self.logger.warning(f"ID 為 '{layer_thought_id}' 的 LayerThought 已存在。將不會重複新增。")
            return self.thoughts[layer_thought_id]

        # LayerThought 的初始分數和理由在聚合後更新
        layer_thought = LayerThought(conceptual_step_description, layer_thought_id, layer_index)
        self.add_thought_object(layer_thought) # 使用基類的方法添加
        self.layers.append(layer_thought)
        self.logger.info(f"已新增層次 {layer_index}: {layer_thought}")
        return layer_thought

    def _generate_prompt_for_option_thought_criteria(self, layer_thought_content, previous_layer_output=None, main_task_description=None):
        prompt = f"主要任務目標：'{main_task_description}'\n" if main_task_description else ""
        prompt += f"針對當前概念步驟：'{layer_thought_content}'\n"
        if previous_layer_output:
            prompt += f"基於先前層次的輸出：'{previous_layer_output[:200]}...' (此輸出旨在推進主要任務)\n"
        prompt += "請為此概念步驟建議一系列具體的、可操作的「標準」或「探索選項」，以生成多樣化且有助於解決主要任務的部分解決方案。如果標準有優先順序，請註明 (例如：標準 A (等級 1); 標準 B (等級 1); 標準 C (等級 2))。\n請僅回傳標準列表，以分號分隔。"
        return prompt
        
    def _generate_prompt_for_option_thought_solution(self, criterion, layer_conceptual_step, previous_layer_output=None, main_task_description=None):
        prompt = f"主要任務目標：'{main_task_description}'\n" if main_task_description else ""
        prompt += f"當前概念步驟：'{layer_conceptual_step}'\n"
        if previous_layer_output:
            prompt += f"先前層次輸出內容：'{previous_layer_output[:200]}...'\n"
        prompt += f"現在，請針對以下「標準/探索選項」生成一個具體的部分解決方案或詳細闡述：\n標準/探索選項：'{criterion}'\n\n你的部分解決方案（確保它與主要任務目標相關）："
        return prompt

    def generate_and_evaluate_option_thoughts_for_layer(self, layer_id, main_task_description, previous_layer_aggregated_output=None):
        if layer_id not in self.thoughts or not isinstance(self.thoughts[layer_id], LayerThought):
            self.logger.error(f"錯誤: 找不到 ID 為 {layer_id} 的 LayerThought。")
            return []

        current_layer_thought = self.thoughts[layer_id]
        
        criteria_prompt = self._generate_prompt_for_option_thought_criteria(
            current_layer_thought.content, 
            previous_layer_aggregated_output,
            main_task_description # 傳遞主要任務描述
        )
        llm_criteria_response = self.llm.generate(criteria_prompt, temperature=0.7) # 獲取標準時溫度可以稍高
        if llm_criteria_response.startswith("錯誤 (LOT):"):
            self.logger.error(f"無法為層次 {current_layer_thought.layer_index} 獲取標準: {llm_criteria_response}")
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
                main_task_description # 傳遞主要任務描述
            )
            llm_solution_response = self.llm.generate(solution_prompt, temperature=0.6) # 生成解決方案時溫度適中
            if llm_solution_response.startswith("錯誤 (LOT):"):
                self.logger.warning(f"無法為標準 '{criterion_text}' 生成解決方案: {llm_solution_response}")
                continue 

            solution_content = llm_solution_response.strip()
            option_id = f"{current_layer_thought.id}_Opt{i}"
            
            # 創建 OptionThought，初始分數為0，稍後由 PRM 評估器更新
            option_thought = OptionThought(solution_content, option_id, criterion_text, level=criterion_level)
            self.add_thought_object(option_thought) # 添加到全局 thoughts 字典
            
            # 建立父子關係
            if current_layer_thought not in option_thought.parents: option_thought.parents.append(current_layer_thought)
            if option_thought not in current_layer_thought.children: current_layer_thought.children.append(option_thought)
            
            # 對新生成的 OptionThought 進行 PRM 評估
            prm_score, prm_justification = self._evaluate_lot_artifact_with_prm(
                option_thought.content,
                "選項思維",
                current_layer_thought.content, # 層次概念步驟
                main_task_description          # 主要任務目標
            )
            option_thought.score = prm_score
            option_thought.prm_justification = prm_justification
            
            if option_thought not in current_layer_thought.option_thoughts: # 確保不重複添加
                 current_layer_thought.option_thoughts.append(option_thought)

            generated_options_with_scores.append(option_thought)
            self.logger.info(f"已為層次 {current_layer_thought.layer_index} 生成並評估 OptionThought: {option_thought}")
            
        return generated_options_with_scores # 返回帶有 PRM 分數的 OptionThought 物件列表

    def _parse_criteria_from_llm(self, llm_response_text):
        criteria = []
        if not llm_response_text or not isinstance(llm_response_text, str) or llm_response_text.startswith("錯誤 (LOT):"):
            self.logger.warning(f"LLM 回應的標準為空或格式不正確: {llm_response_text}")
            return [{'text': "預設標準 (因解析錯誤)", 'level': 1}]

        parts = llm_response_text.split(';')
        for part in parts:
            part = part.strip()
            if not part: continue
            level = 1 # 預設等級
            text = part
            # 嘗試解析 "(等級 X)"
            match = re.search(r'\((?:level|等級)\s*(\d+)\)$', part, re.IGNORECASE)
            if match:
                try:
                    level = int(match.group(1))
                    text = re.sub(r'\s*\((?:level|等級)\s*\d+\)$', '', part, flags=re.IGNORECASE).strip()
                except ValueError:
                    self.logger.warning(f"無法解析標準 '{part}' 中的等級，使用預設等級 1。")
            criteria.append({'text': text, 'level': level})
        
        if not criteria: # 如果分割後沒有任何內容
             self.logger.warning(f"無法從 '{llm_response_text}' 解析出任何標準，使用回應本身作為單一標準。")
             return [{'text': llm_response_text.strip(), 'level': 1}]
        return criteria
        
    def aggregate_and_evaluate_option_thoughts_in_layer(self, layer_id, main_task_description, aggregation_strategy='best_prm_score'):
        if layer_id not in self.thoughts or not isinstance(self.thoughts[layer_id], LayerThought):
            self.logger.error(f"錯誤: 找不到 ID 為 {layer_id} 的 LayerThought 進行聚合。")
            return None, 0.0, "LayerThought 未找到" # 返回內容、分數、理由

        current_layer_thought = self.thoughts[layer_id]
        options = current_layer_thought.option_thoughts # 這些選項應該已經有了 PRM 分數
        if not options:
            self.logger.warning(f"層次 {current_layer_thought.layer_index} 中沒有 OptionThought 可供聚合。")
            # 更新 LayerThought 的分數和理由以反映沒有選項
            current_layer_thought.score = 0.0 
            current_layer_thought.prm_justification = "沒有可聚合的選項思維。"
            return f"層次 {current_layer_thought.layer_index} 沒有生成任何選項。", 0.0, current_layer_thought.prm_justification

        aggregated_content = f"從層次 {current_layer_thought.layer_index} (概念: '{current_layer_thought.content[:30]}...') 使用策略 '{aggregation_strategy}' 聚合的結果:\n"
        
        # 根據 PRM 分數排序選項
        options.sort(key=lambda ot: (ot.level, -ot.score)) # 按等級升序，同等級按 PRM 分數降序

        if aggregation_strategy == 'best_prm_score': 
            if options:
                 best_option = options[0] # 已經排序過，第一個就是最好的
                 aggregated_content += (f"最佳選項 (標準: '{best_option.criterion}', 等級: {best_option.level}, "
                                       f"PRM分數: {best_option.score:.2f}):\n{best_option.content}")
            else:
                aggregated_content += "沒有可選擇的最佳選項。"
        elif aggregation_strategy == 'weighted_sum_content' or aggregation_strategy == 'all_content_ranked': # 示例：將所有選項按 PRM 分數排序後串聯
            for opt in options: # options 已經按 PRM 分數排序
                aggregated_content += (f"- (標準: '{opt.criterion}', 等級: {opt.level}, "
                                       f"PRM分數: {opt.score:.2f}, 理由: {opt.prm_justification[:50]}...):\n {opt.content}\n\n")
        else: # 預設也使用最佳 PRM 分數的選項
            self.logger.warning(f"未知的聚合策略 '{aggregation_strategy}'，將使用 'best_prm_score'。")
            if options:
                 best_option = options[0]
                 aggregated_content += (f"最佳選項 (標準: '{best_option.criterion}', 等級: {best_option.level}, "
                                       f"PRM分數: {best_option.score:.2f}):\n{best_option.content}")
            else:
                aggregated_content += "沒有可選擇的最佳選項。"
        
        # 對聚合後的層次輸出進行 PRM 評估
        layer_output_prm_score, layer_output_prm_justification = self._evaluate_lot_artifact_with_prm(
            aggregated_content,
            "層次聚合輸出",
            current_layer_thought.content, # 層次概念步驟
            main_task_description          # 主要任務目標
        )
        # 更新 LayerThought 本身的分數和理由
        current_layer_thought.score = layer_output_prm_score
        current_layer_thought.prm_justification = layer_output_prm_justification
        
        self.logger.info(f"已聚合層次 {current_layer_thought.layer_index} 中的選項。聚合輸出 PRM 分數: {layer_output_prm_score:.2f}")
        return aggregated_content, layer_output_prm_score, layer_output_prm_justification


    def run_pipeline(self, conceptual_steps, main_task_description, initial_input=None, min_layer_prm_score_threshold=0.3):
        # main_task_description 是整個 LOT 流程試圖解決的最終任務
        # min_layer_prm_score_threshold: 如果某個層次的聚合輸出 PRM 分數低於此閾值，可以考慮提前終止或採取補救措施
        
        previous_layer_output_content = initial_input
        final_pipeline_output = None # 最終的、被認為成功的管線輸出
        
        self.logger.info(f"LOT 管線開始執行，主要任務目標: {main_task_description}")

        for i, step_description in enumerate(conceptual_steps):
            self.logger.info(f"\n--- 正在處理層次 {i}: {step_description} ---")
            layer_thought = self.add_layer_thought(step_description) # LayerThought 的分數此時為預設
            
            # 為當前層次生成並評估選項思維
            # 需要傳遞 main_task_description 以便 PRM 評估選項思維的相關性
            option_thoughts = self.generate_and_evaluate_option_thoughts_for_layer(
                layer_thought.id, 
                main_task_description, # 傳遞主要任務
                previous_layer_output_content
            )
            
            if not option_thoughts:
                self.logger.warning(f"層次 {i} 未能生成任何選項思維。可能需要檢查標準生成或解決方案生成提示。")
                # 即使沒有選項，也嘗試聚合（會得到“沒有選項”的訊息），然後讓 PRM 評估這個“空”聚合
            
            # 聚合本層次的選項思維，並對聚合結果進行 PRM 評估
            # 聚合策略可以基於選項的 PRM 分數
            aggregated_output_content, layer_prm_score, layer_prm_justification = self.aggregate_and_evaluate_option_thoughts_in_layer(
                layer_thought.id, 
                main_task_description, # 傳遞主要任務
                aggregation_strategy='all_content_ranked' # 或者 'best_prm_score'
            )
            
            # LayerThought 的 score 和 prm_justification 已在 aggregate_and_evaluate... 中更新
            
            if aggregated_output_content is None: # 理論上 aggregate... 應該總返回一些東西，除非 layer_id 錯誤
                self.logger.error(f"層次 {i} 聚合時發生嚴重錯誤。正在停止管線。")
                final_pipeline_output = previous_layer_output_content # 使用上一層的輸出作為最終結果
                break 
            
            self.logger.info(f"層次 {i} 聚合輸出 PRM 分數: {layer_prm_score:.2f}. 理由: {layer_prm_justification}")

            if layer_prm_score < min_layer_prm_score_threshold:
                self.logger.warning(f"層次 {i} 的聚合輸出 PRM 分數 ({layer_prm_score:.2f}) 低於閾值 ({min_layer_prm_score_threshold})。")
                # 在這裡可以實現更複雜的邏輯，例如：
                # 1. 嘗試不同的聚合策略
                # 2. 重新生成該層次的選項思維（可能調整提示或溫度）
                # 3. 回溯到前一個層次
                # 4. 提前終止管線
                # 為了簡化，我們這裡只記錄警告，但仍然繼續（或可以選擇終止）
                self.logger.warning(f"管線可能無法達到最佳效果。考慮終止或採取補救措施。目前將繼續...")
                # 如果決定終止：
                # final_pipeline_output = previous_layer_output_content # 或 aggregated_output_content，取決於策略
                # break

            previous_layer_output_content = aggregated_output_content
            final_pipeline_output = aggregated_output_content # 總是指向最後一個成功的聚合輸出

        self.logger.info(f"LOT 管線執行完畢。最終輸出 PRM 分數 (來自最後一層): {self.layers[-1].score if self.layers else 'N/A'}")
        return final_pipeline_output

# --- 將範例用法封裝到函式中 ---
def run_lot_example_workflow_with_prm(api_key):
    logger_example = DefaultLogger()
    logger_example.info(f"正在使用 API 金鑰執行 LOT (PRM風格) 範例: ...{api_key[-4:]}")
    
    # 配置 Gemini API (如果尚未在全局配置)
    # 為了獨立運行，這裡做一次配置檢查
    try:
        if not getattr(genai, '_is_configured_by_lot_example', False): # 避免重複配置
            genai.configure(api_key=api_key)
            setattr(genai, '_is_configured_by_lot_example', True)
            logger_example.info("已為 LOT 範例配置 Gemini API。")
    except Exception as e:
        logger_example.error(f"配置 Gemini API 時發生錯誤: {e}")
        return

    try:
        # 操作型 LLM (用於生成標準、生成選項解決方案等)
        llm_operator = GeminiLLMInterface(api_key=api_key, model_name="gemini-1.5-flash-latest", logger=logger_example)
        # PRM 評估器 LLM (可以與操作型 LLM 相同，也可以是另一個更強的或專門微調的評估模型)
        # 為了演示，這裡使用相同的 LLM 實例
        llm_prm_evaluator = llm_operator 
        
        if not llm_operator.model: # 檢查模型是否成功初始化
             logger_example.error("LOT 操作 LLM 未能初始化。中止範例。")
             return

    except ValueError as e:
        logger_example.error(f"初始化 LLM 介面時發生錯誤: {e}")
        return
    
    lot_system = LayerOfThoughts(llm_operator, logger=logger_example, prm_evaluator_llm=llm_prm_evaluator)

    # 主要任務目標：為一個即將在夏季舉辦的戶外音樂節制定一份全面的市場推廣計劃。
    # 預算：中等。目標受眾：18-35歲的年輕人。地點：城市公園。
    main_task_music_festival = (
        "為一個即將在夏季舉辦、地點在城市公園、預算中等、目標受眾為18-35歲年輕人的戶外音樂節，"
        "制定一份全面的市場推廣計劃，旨在最大化參與人數和品牌知名度。"
    )

    conceptual_steps_marketing_plan = [
        "階段 1: 市場與受眾分析 - 深入了解目標受眾的偏好、常去的社交媒體平台以及對音樂節的期望。",
        "階段 2: 核心信息與品牌定位 - 確定音樂節的獨特賣點 (USP)，並制定吸引目標受眾的核心宣傳信息和品牌形象。",
        "階段 3: 推廣渠道與活動策劃 - 選擇最有效的線上和線下推廣渠道，並策劃具體的預熱活動、現場互動和後續宣傳活動。",
        "階段 4: 預算分配與成效衡量 - 為各項推廣活動合理分配預算，並設定關鍵績效指標 (KPIs) 以衡量計劃的成效。"
    ]
    
    # 初始輸入可以為空，或包含一些對主要任務的初步想法/約束
    initial_input_for_marketing = "音樂節主題初步定為「城市綠洲之聲」，強調自然與音樂的結合。"

    logger_example.info(f"\nLoT 管線開始執行，主要任務: {main_task_music_festival}")
    final_marketing_plan = lot_system.run_pipeline(
        conceptual_steps_marketing_plan, 
        main_task_description=main_task_music_festival, # 傳遞主要任務描述
        initial_input=initial_input_for_marketing,
        min_layer_prm_score_threshold=0.4 # 設定一個層次 PRM 分數閾值
    )

    logger_example.info(f"\n--- LoT 管線的最終市場推廣計劃 (或最後成功的層次輸出) ---")
    if final_marketing_plan:
        logger_example.info(final_marketing_plan)
        # 打印最後一層的 PRM 分數和理由
        if lot_system.layers:
            last_layer = lot_system.layers[-1]
            logger_example.info(f"\n最終層次 (L{last_layer.layer_index}) 的 PRM 評估:")
            logger_example.info(f"  分數: {last_layer.score:.2f}")
            logger_example.info(f"  理由: {last_layer.prm_justification}")
    else:
        logger_example.warning("管線未能產生最終輸出。")
    
    logger_example.info("\n--- LOT (PRM風格) 範例用法結束 ---")

# --- 主執行區塊 ---
if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY")
    logger_main = DefaultLogger()

    if not API_KEY:
        logger_main.warning("警告：未在環境變數中找到 GEMINI_API_KEY。")
        logger_main.warning("請設定 GEMINI_API_KEY 環境變數或在 LOT.py 中提供 API 金鑰以執行範例。")
    
    if API_KEY and API_KEY != "YOUR_GEMINI_API_KEY_HERE":
        run_lot_example_workflow_with_prm(API_KEY)
    elif API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger_main.error("*****************************************************************")
        logger_main.error("警告：您使用的是預設的佔位符 API 金鑰。")
        logger_main.error("請將程式碼中的 'YOUR_GEMINI_API_KEY_HERE' 替換為您真實的 Gemini API 金鑰，或設定環境變數。")
        logger_main.error("範例執行可能會失敗或產生非預期結果。")
        logger_main.error("*****************************************************************")
        # 仍然嘗試執行，讓 GeminiLLMInterface 內部處理或拋出錯誤
        try:
            run_lot_example_workflow_with_prm(API_KEY)
        except Exception as e: # 捕捉 GeminiLLMInterface 初始化時可能因無效 key 拋出的錯誤
             logger_main.error(f"由於 API 金鑰問題導致範例執行失敗: {e}")
    else: # API_KEY is None
        logger_main.error("沒有有效的 API 金鑰，範例中的 LLM 互動將失敗。")
        try:
            run_lot_example_workflow_with_prm(API_KEY) 
        except Exception as e:
             logger_main.error(f"由於 API 金鑰問題導致範例執行失敗: {e}")

