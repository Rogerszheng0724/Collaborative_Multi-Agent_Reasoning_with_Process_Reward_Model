import google.generativeai as genai
import numpy as np
import os
import re # 新增 re 模組用於解析
from dotenv import load_dotenv

# --- 設定 ---
load_dotenv()
GEMINI_API_KEY_FROM_ENV = os.getenv("GEMINI_API_KEY")

# --- 輔助：預設 Logger ---
class DefaultLogger:
    def info(self, message): print(f"[ROT INFO] {message}")
    def warning(self, message): print(f"[ROT WARNING] {message}")
    def error(self, message): print(f"[ROT ERROR] {message}")

# --- Gemini LLM 介面 (ROT 版本) ---
class GeminiLLMInterface:
    def __init__(self, model_name="gemini-1.5-flash-latest", api_key=None, logger=None): # 添加 logger
        self.model = None
        self.logger = logger if logger else DefaultLogger()
        effective_api_key = api_key or GEMINI_API_KEY_FROM_ENV

        if not effective_api_key:
            self.logger.error("ROT.GeminiLLMInterface: API 金鑰未提供。LLM 將無法運作。")
            return

        try:
            current_configured_key = getattr(genai, '_last_configured_key_by_rot', None)
            if current_configured_key != effective_api_key:
                genai.configure(api_key=effective_api_key)
                setattr(genai, '_last_configured_key_by_rot', effective_api_key)
                self.logger.info(f"ROT.GeminiLLMInterface: 已使用 API 金鑰尾碼 ...{effective_api_key[-4:]} 配置 genai。")

            self.model = genai.GenerativeModel(model_name)
            self.logger.info(f"ROT.GeminiLLMInterface 已使用模型 {model_name} 初始化。")
        except Exception as e:
            self.logger.error(f"初始化 ROT Gemini GenerativeModel ({model_name}) 失敗: {e}")
            self.model = None

    def generate(self, prompt_text, temperature=0.7): # 添加 temperature 參數
        if not self.model:
            self.logger.error("ROT.GeminiLLMInterface: LLM 模型未初始化。無法生成內容。")
            return "LLM 未初始化或 API 金鑰錯誤"
        try:
            self.logger.info(f"\n--- 正在發送提示到 Gemini (ROT LLM) ---\n{prompt_text[:300]}...\n--- Gemini 提示結束 (ROT LLM) ---")
            response = self.model.generate_content(
                prompt_text,
                generation_config=genai.types.GenerationConfig(temperature=temperature) # 使用 temperature
            )
            llm_response_text = ""
            if hasattr(response, 'parts') and response.parts:
                 llm_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            elif hasattr(response, 'text'):
                 llm_response_text = response.text
            
            if not llm_response_text and hasattr(response, 'prompt_feedback') and \
                 response.prompt_feedback.block_reason != genai.types.BlockReason.BLOCK_REASON_UNSPECIFIED:
                block_reason_str = genai.types.BlockReason(response.prompt_feedback.block_reason).name
                self.logger.warning(f"ROT.GeminiLLMInterface: 提示因 {block_reason_str} 被封鎖。")
                return f"LLM 錯誤：提示因 {block_reason_str} 被封鎖。"
            
            self.logger.info(f"--- 收到 Gemini 回應 (ROT LLM) ---\n{llm_response_text[:300]}...\n--- Gemini 回應結束 (ROT LLM) ---")
            return llm_response_text if llm_response_text else "LLM 錯誤：未產生有效內容。"

        except Exception as e:
            self.logger.error(f"ROT.GeminiLLMInterface: Gemini API 呼叫錯誤 (generate): {e}")
            return f"LLM 錯誤: {e}"

    def generate_with_simulated_score(self, prompt_text, temperature=0.7): # 添加 temperature
        # 此方法在 ROT 中用於 PGRR 階段的初始候選提示生成，其 "simulated_score" 並非 PRM 分數。
        # PRM 分數將由 ReversalOfThought 類中的專用評估方法提供。
        if not self.model:
            self.logger.error("ROT.GeminiLLMInterface: LLM 模型未初始化。無法生成內容並評分。")
            return "LLM 未初始化或 API 金鑰錯誤", 0.0

        response_text = self.generate(prompt_text, temperature=temperature)
        simulated_score = 0.0 # 這裡的 prob_score 是一個基於回應長度的簡單模擬，非 PRM 分數
        if "LLM 未初始化" in response_text or "LLM 錯誤" in response_text:
             simulated_score = 0.0
        else:
            # 保持原有的模擬分數邏輯，因為 PGRR 內部可能會用到它
            simulated_score = float(len(response_text)) / 1000.0 
            simulated_score = min(max(simulated_score, 0.0), 1.0)
        return response_text, simulated_score

# --- Gemini 嵌入介面 (ROT 版本) ---
class GeminiEmbeddingInterface:
    def __init__(self, model_name="models/embedding-001", api_key=None, logger=None): # 添加 logger
        self.model_name = model_name
        self.api_key_configured_successfully = False
        self.logger = logger if logger else DefaultLogger()
        effective_api_key = api_key or GEMINI_API_KEY_FROM_ENV

        if not effective_api_key:
            self.logger.error("ROT.GeminiEmbeddingInterface: API 金鑰未提供。嵌入功能將無法運作。")
            return

        try:
            current_configured_key = getattr(genai, '_last_configured_key_by_rot_embed', None) # 使用不同標記以區分 LLM 配置
            if current_configured_key != effective_api_key:
                genai.configure(api_key=effective_api_key)
                setattr(genai, '_last_configured_key_by_rot_embed', effective_api_key)
                self.logger.info(f"ROT.GeminiEmbeddingInterface: 已使用 API 金鑰尾碼 ...{effective_api_key[-4:]} 配置 genai (用於嵌入)。")
            
            self.api_key_configured_successfully = True
            self.logger.info(f"ROT.GeminiEmbeddingInterface 已為模型 {model_name} 初始化。")
        except Exception as e:
            self.logger.error(f"為 ROT.GeminiEmbeddingInterface 設定 API 金鑰時發生錯誤: {e}")
            self.api_key_configured_successfully = False

    def _get_embedding(self, text):
        if not self.api_key_configured_successfully:
            self.logger.error("ROT.GeminiEmbeddingInterface: API 金鑰未設定或配置失敗。無法獲取嵌入。")
            return None
        try:
            self.logger.info(f"ROT.GeminiEmbeddingInterface: 正在為文字獲取嵌入: '{text[:50]}...'")
            result = genai.embed_content(model=self.model_name, content=text, task_type="SEMANTIC_SIMILARITY") # 或 RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY
            return result['embedding']
        except Exception as e:
            self.logger.error(f"ROT.GeminiEmbeddingInterface: Gemini API 呼叫錯誤 (embed_content for '{text[:50]}...'): {e}")
            return None

    def calculate_similarity(self, text1, text2):
        if not self.api_key_configured_successfully:
            self.logger.error("ROT.GeminiEmbeddingInterface: API 金鑰未設定，無法計算相似度。")
            return 0.0

        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        if emb1 is None or emb2 is None:
            self.logger.warning("ROT.GeminiEmbeddingInterface: 無法計算相似度，因為一個或多個嵌入向量為 None。")
            return 0.0

        try:
            emb1_np = np.array(emb1)
            emb2_np = np.array(emb2)
            norm_emb1 = np.linalg.norm(emb1_np)
            norm_emb2 = np.linalg.norm(emb2_np)
            if norm_emb1 == 0 or norm_emb2 == 0: # 防止除以零
                return 0.0
            similarity = np.dot(emb1_np, emb2_np) / (norm_emb1 * norm_emb2)
            return float(similarity)
        except Exception as e:
            self.logger.error(f"ROT.GeminiEmbeddingInterface: 計算餘弦相似度時出錯: {e}")
            return 0.0

# --- ReversalOfThought 類別 ---
class ReversalOfThought:
    def __init__(self, llm_interface, embedding_model_interface, similarity_threshold=0.7, logger=None): # 添加 logger
        self.llm = llm_interface
        self.embedder = embedding_model_interface
        self.similarity_threshold = similarity_threshold
        self.logger = logger if logger else DefaultLogger()
        # 假設 self.llm 是 GeminiLLMInterface 的一個實例，可以用於 PRM 評估
        # 如果評估 LLM 不同，MASOrchestrator 在創建 ROT 時需要傳入特定的評估 LLM
        self.prm_evaluator_llm = llm_interface # 預設使用相同的 LLM 進行 PRM 評估

    def _generate_prm_style_scoring_prompt_for_rot_artifact(self, artifact_content, artifact_type, main_task_description):
        """
        為 ROT 產生的中間成果（如任務定義提示）產生 PRM 風格的評分提示。
        artifact_type: "任務定義提示" 或 "最終優化提示"
        main_task_description: ROT 系統試圖為其生成優良提示的那個「更高層次的原始任務」。
        """
        prompt = (
            f"您是一位專家級的評估員，任務是評估一個「{artifact_type}」的質量和潛力。\n"
            f"此 {artifact_type} 最終將用於指導一個大型語言模型來解決以下主要任務：\n"
            f"主要任務目標：'{main_task_description}'\n\n"
            f"待評估的「{artifact_type}」內容：\n\"\"\"\n{artifact_content}\n\"\"\"\n\n"
            "評估指示：\n"
            f"1.  效用性/貢獻度：這個「{artifact_type}」在多大程度上能夠清晰、準確且有效地引導一個LLM去解決上述「主要任務目標」？它是否可能產生高質量的解決方案？\n"
            "2.  完整性和準確性：此 artifact 是否完整地捕捉了主要任務的關鍵方面？是否存在模糊、誤導或遺漏之處？\n"
            "3.  清晰度和可操作性：此 artifact 本身是否易於理解？LLM 是否能輕易地按照它來執行？\n\n"
            "請提供一個總體評分和簡要理由。\n"
            "輸出格式（嚴格遵守）：\n"
            "Score: [一個介於 0.0 (非常差/無助益) 到 1.0 (非常好/極具潛力) 之間的浮點數]\n"
            "Justification: [對您的分數的簡要解釋，說明其如何以及為何有助於或無助於解決「主要任務目標」]"
        )
        return prompt

    def _parse_llm_response_for_prm_score(self, llm_response_text):
        """ 從 LLM 的回應中解析 PRM 風格的評分和理由。 """
        if not llm_response_text or llm_response_text.startswith("LLM 錯誤") or llm_response_text.startswith("LLM 未初始化"):
            self.logger.warning(f"LLM 回應無效或為錯誤訊息，無法解析 PRM 分數: {llm_response_text}")
            return 0.0, f"PRM 評分失敗：LLM 回應無效 ({llm_response_text})"

        score_match = re.search(r"Score:\s*([0-9.]+)", llm_response_text, re.IGNORECASE)
        # DOTALL 使 '.' 可以匹配換行符，以捕獲多行理由
        justification_match = re.search(r"Justification:\s*(.+)", llm_response_text, re.IGNORECASE | re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.0
        justification = justification_match.group(1).strip() if justification_match else "未提供理由或解析錯誤。"

        if not score_match:
            self.logger.warning(f"無法從回應中解析 PRM 分數。原始回應：'{llm_response_text}'")
        return score, justification

    def _evaluate_rot_artifact_with_prm(self, artifact_content, artifact_type, main_task_description):
        """ 使用 PRM 評估器評估 ROT 的中間成果。 """
        if not self.prm_evaluator_llm:
            self.logger.error("PRM 評估器 LLM 未設定。無法評估。")
            return 0.0, "PRM 評估器未設定"
        
        prompt = self._generate_prm_style_scoring_prompt_for_rot_artifact(artifact_content, artifact_type, main_task_description)
        llm_response = self.prm_evaluator_llm.generate(prompt) # 假設評估 LLM 有 generate 方法
        score, justification = self._parse_llm_response_for_prm_score(llm_response)
        self.logger.info(f"ROT 成果 ({artifact_type}) PRM 評估 - 分數: {score:.2f}, 理由: {justification}")
        return score, justification

    def _prompt_for_reverse_reasoning(self, demonstrations_text):
        prompt = (
            "您是一位在數學和資訊推理方面非常傑出的專家。\n"
            "根據給定的範例，定義具體任務，包括：\n"
            "1. 任務定義：對目標的清晰描述。\n"
            "2. 偽代碼：用自然語言描述的逐步演算法。\n"
            "3. 邏輯偽代碼：使用符號 (例如 ∀, ∃, ∧, ∨, ¬, → 等) 將偽代碼轉換為形式化的邏輯表示。如果需要，請提供具體範例。\n"
            "4. 案例範例：從輸入中衍生的說明性範例。\n"
            "5. 輸入-輸出格式：輸入和輸出的精確規範。\n\n"
            "示範：\n"
            f"{demonstrations_text}\n\n"
            "您的綜合定義 (請確保包含上述所有五個部分)：" # 強調完整性
        )
        return prompt

    def _prompt_for_pairwise_preference(self, response_A_text, response_B_text, main_task_description): # 添加 main_task_description
        prompt = (
            f"主要任務目標：'{main_task_description}'\n\n"
            "請比較以下兩個由AI生成的「任務定義提示」(A 和 B)，並選擇您認為哪一個更能有效地引導另一個AI解決上述「主要任務目標」。\n"
            "評估標準應包括：\n"
            "-   **清晰度**：提示是否易於理解？\n"
            "-   **完整性**：提示是否包含了所有必要的指令和信息以解決主要任務？\n"
            "-   **準確性**：提示中的定義和步驟是否準確反映了主要任務的需求？\n"
            "-   **對解決主要任務的潛在效用**：哪個提示更有可能引導AI產生高質量的解決方案？\n\n"
            f"提示 (A):\n\"\"\"\n{response_A_text}\n\"\"\"\n\n"
            f"提示 (B):\n\"\"\"\n{response_B_text}\n\"\"\"\n\n"
            "您的選擇 (請僅回答 A 或 B) 以及簡要理由 (說明為何您選擇的提示在引導解決「主要任務目標」方面更優越)："
        )
        return prompt

    def preference_guided_reverse_reasoning_warmup(self, demonstrations, main_task_description_for_prm, warm_iterations=3):
        # main_task_description_for_prm 是 ROT 試圖為其生成良好提示的那個「更高層次的原始任務」
        demo_text = ""
        for i, (inp, outp) in enumerate(demonstrations):
            demo_text += f"範例 {i+1}:\n輸入: {inp}\n輸出: {outp}\n\n"

        candidate_responses_info = [] # 存儲 {'text': ..., 'prob_score': ..., 'prm_score': ..., 'prm_justification': ..., 'id': ...}
        self.logger.info(f"\n--- ROT: 執行反向推理預熱 ({warm_iterations} 次迭代) ---")
        for i in range(warm_iterations):
            rr_prompt = self._prompt_for_reverse_reasoning(demo_text)
            # 使用較低的溫度以獲得更多樣的初始候選
            response_text, response_prob_score = self.llm.generate_with_simulated_score(rr_prompt, temperature=0.8)

            if "LLM 未初始化" in response_text or "LLM 錯誤" in response_text:
                self.logger.warning(f"ROT: 預熱迭代 {i+1} 失敗，因為 LLM 錯誤或未初始化。回應: {response_text}")
                continue
            
            # 對生成的候選提示進行 PRM 評估
            prm_score, prm_justification = self._evaluate_rot_artifact_with_prm(
                response_text, 
                "PGRR候選任務定義提示", 
                main_task_description_for_prm
            )
            
            candidate_info = {
                'text': response_text, 
                'prob_score': response_prob_score, # 原始的模擬分數
                'prm_score': prm_score,             # PRM 評估分數
                'prm_justification': prm_justification,
                'id': f"cand_{i}"
            }
            candidate_responses_info.append(candidate_info)
            self.logger.info(f"ROT: 預熱迭代 {i+1} 生成候選提示 (模擬分數: {response_prob_score:.3f}, PRM分數: {prm_score:.3f})")

        if not candidate_responses_info:
            self.logger.error("ROT: PGRR 預熱未能生成任何候選回應。")
            return "PGRR 預熱失敗：無候選回應"

        self.logger.info("\n--- ROT: 執行成對偏好評估 (基於主要任務目標) ---")
        preference_matrix = {} 

        num_candidates = len(candidate_responses_info)
        if num_candidates < 2:
            self.logger.info("ROT: 候選回應少於2個，跳過成對偏好評估。")
            # 如果只有一個候選，直接使用其 PRM 分數（如果需要排序的話）
            # 但這裡 PGRR 的目標是選出一個 "LLM-Taste Prompt"，所以直接返回其文本
            best_candidate = candidate_responses_info[0]
            self.logger.info(f"ROT: 僅有一個候選 (ID: {best_candidate['id']}), PRM 分數: {best_candidate['prm_score']:.3f}, 直接選用。")
            return best_candidate['text']


        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                resp_A_info = candidate_responses_info[i]
                resp_B_info = candidate_responses_info[j]
                
                # 在偏好提示中加入 main_task_description_for_prm
                pref_prompt_A_vs_B = self._prompt_for_pairwise_preference(
                    resp_A_info['text'], 
                    resp_B_info['text'],
                    main_task_description_for_prm # 關鍵：讓 LLM 基於此來判斷哪個提示更好
                )
                choice_response_A_vs_B = self.llm.generate(pref_prompt_A_vs_B, temperature=0.3) # 偏好選擇時溫度低一些

                if "LLM 未初始化" in choice_response_A_vs_B or "LLM 錯誤" in choice_response_A_vs_B:
                    self.logger.warning(f"ROT: 偏好評估 ({resp_A_info['id']} vs {resp_B_info['id']}) 失敗。回應: {choice_response_A_vs_B}")
                    # 失敗時給予中性偏好
                    preference_matrix[(resp_A_info['id'], resp_B_info['id'])] = 0.5 
                    preference_matrix[(resp_B_info['id'], resp_A_info['id'])] = 0.5
                    continue

                choice_upper = choice_response_A_vs_B.strip().upper()
                chosen_option = None
                if choice_upper.startswith("A"): chosen_option = "A"
                elif choice_upper.startswith("B"): chosen_option = "B"

                if chosen_option == "A":
                    preference_matrix[(resp_A_info['id'], resp_B_info['id'])] = 1.0
                    preference_matrix[(resp_B_info['id'], resp_A_info['id'])] = 0.0
                elif chosen_option == "B":
                    preference_matrix[(resp_A_info['id'], resp_B_info['id'])] = 0.0
                    preference_matrix[(resp_B_info['id'], resp_A_info['id'])] = 1.0
                else: # 平手或無法判斷
                    preference_matrix[(resp_A_info['id'], resp_B_info['id'])] = 0.5
                    preference_matrix[(resp_B_info['id'], resp_A_info['id'])] = 0.5
                
                winner_char = chosen_option if chosen_option else '平手/未知'
                self.logger.info(f"ROT: 偏好：候選 {resp_A_info['id']} vs {resp_B_info['id']} -> 勝者: {winner_char}")
        
        # 結合 PRM 分數和成對偏好分數來選擇最佳提示
        final_scores_for_candidates = []
        for i in range(num_candidates):
            current_candidate_id = candidate_responses_info[i]['id']
            current_candidate_prm_score = candidate_responses_info[i]['prm_score']
            
            total_pairwise_preference_score = 0
            num_comparisons = 0
            for j in range(num_candidates):
                if i == j: continue
                other_candidate_id = candidate_responses_info[j]['id']
                total_pairwise_preference_score += preference_matrix.get((current_candidate_id, other_candidate_id), 0.5)
                num_comparisons +=1
            
            avg_pairwise_preference_score = (total_pairwise_preference_score / num_comparisons) if num_comparisons > 0 else 0.5
            
            # 結合分數：可以加權平均，這裡簡單平均 PRM 分數和成對偏好分數
            # 權重可以調整，例如更側重 PRM 分數
            combined_score = (current_candidate_prm_score * 0.6) + (avg_pairwise_preference_score * 0.4) 
            
            final_scores_for_candidates.append({
                'id': current_candidate_id, 
                'score': combined_score, 
                'text': candidate_responses_info[i]['text'],
                'prm_score': current_candidate_prm_score,
                'avg_pairwise_pref': avg_pairwise_preference_score
            })
            
        if not final_scores_for_candidates:
            self.logger.error("ROT: 未能計算最終分數。PGRR 排序失敗。")
            return "PGRR 排序失敗：無最終分數"

        best_prompt_info = max(final_scores_for_candidates, key=lambda x: x['score'])
        self.logger.info(f"\n--- ROT: 最佳 LLM-Taste 提示 (ID: {best_prompt_info['id']}, 綜合分數: {best_prompt_info['score']:.3f}, PRM分數: {best_prompt_info['prm_score']:.3f}) ---")
        return best_prompt_info['text'] # 返回最佳提示的文本內容

    def _extract_task_definition(self, prompt_text):
        # 此方法保持不變，因為它用於從提示中提取特定部分，與 PRM 無直接關聯
        lines = str(prompt_text).splitlines()
        task_def_lines = []
        in_task_def_section = False
        start_keywords = ["task definition:", "任務定義：", "task definition：", "任務定義:"]
        end_keywords = [
            "pseudocode:", "偽代碼：", "pseudocode：", "偽代碼:",
            "logical pseudocode:", "邏輯偽代碼：", "logical pseudocode：", "邏輯偽代碼:",
            "case examples:", "案例範例：", "case examples：", "案例範例:",
            "input-output format:", "輸入-輸出格式：", "input-output format：", "輸入-輸出格式:"
        ]

        for line in lines:
            stripped_line = line.strip()
            line_lower = stripped_line.lower()
            
            if not in_task_def_section:
                for keyword in start_keywords:
                    if line_lower.startswith(keyword):
                        in_task_def_section = True
                        content_after_keyword = stripped_line[len(keyword):].strip()
                        if content_after_keyword:
                            task_def_lines.append(content_after_keyword)
                        break 
            elif in_task_def_section:
                is_end_keyword_found = False
                for keyword in end_keywords:
                    if line_lower.startswith(keyword):
                        is_end_keyword_found = True
                        break
                if is_end_keyword_found or not stripped_line: 
                    break 
                task_def_lines.append(stripped_line)
        
        extracted_definition = "\n".join(task_def_lines).strip()
        
        if not extracted_definition: # 如果提取失敗，返回原始文本或一個標記
            self.logger.warning(f"未能從提示中提取明確的任務定義部分。將使用整個提示文本進行比較。提示片段：'{str(prompt_text)[:100]}...'")
            return str(prompt_text) 
        return extracted_definition

    def cognitive_preference_manager(self, original_task_prompt_text, llm_taste_prompt_text, main_task_description_for_prm):
        # main_task_description_for_prm 是 ROT 試圖為其生成良好提示的那個「更高層次的原始任務」
        self.logger.info("\n--- ROT: 執行認知偏好管理器 (CPM) ---")
        
        # 評估傳入的 llm_taste_prompt_text 的質量
        llm_taste_prm_score, llm_taste_prm_justification = self._evaluate_rot_artifact_with_prm(
            llm_taste_prompt_text, 
            "CPM輸入的LLM-Taste提示", 
            main_task_description_for_prm
        )
        self.logger.info(f"ROT (CPM): 輸入的 LLM-Taste 提示 PRM 分數: {llm_taste_prm_score:.3f}")

        # 根據 PRM 分數決定是否需要進一步優化或直接使用
        # 例如，如果 PRM 分數已經很高，可能不需要與原始提示進行複雜的比較和融合
        # 這裡為了演示，我們仍然執行相似度比較和融合，但可以加入基於 PRM 分數的早期退出或策略調整

        original_task_def_text = self._extract_task_definition(original_task_prompt_text)
        llm_taste_task_def_text = self._extract_task_definition(llm_taste_prompt_text)
        
        similarity = self.embedder.calculate_similarity(original_task_def_text, llm_taste_task_def_text)
        self.logger.info(f"ROT: 原始任務定義與 LLM-taste 任務定義之間的相似度: {similarity:.4f} (閾值: {self.similarity_threshold})")

        final_prompt_text_candidate = ""
        instruction_prompt = ""
        if similarity >= self.similarity_threshold:
            self.logger.info(f"ROT (CPM): 檢測為已知任務 (相似度 {similarity:.4f} >= {self.similarity_threshold})。將嘗試聚合兩個提示的優點。")
            instruction_prompt = (
                "請綜合以下兩個關於同一任務的描述/提示，目標是創建一個單一的、更優越的提示版本。"
                "這個新版本應該融合兩者的最強點，特別是在任務定義的清晰度、偽代碼的實用性、邏輯表達的準確性、範例的相關性以及輸入/輸出格式的明確性方面。"
                f"請確保最終提示既全面又易於LLM理解和執行，以便最好地完成主要任務：'{main_task_description_for_prm}'。\n\n" # 加入主要任務上下文
                f"提示 1 (原始或基準提示):\n{original_task_prompt_text}\n\n"
                f"提示 2 (LLM 生成的候選提示):\n{llm_taste_prompt_text}\n\n"
                "綜合後的最佳提示："
            )
        else:
            self.logger.info(f"ROT (CPM): 檢測為未知或顯著不同的任務 (相似度 {similarity:.4f} < {self.similarity_threshold})。將嘗試調整風格範本以符合原始任務邏輯。")
            instruction_prompt = (
                "以下有兩個提示。 「LLM 生成提示範本」可能在任務理解上不完全準確，但其整體結構和風格（例如，如何組織任務定義、偽代碼、範例等部分）是偏好的。"
                "「原始正確提示」包含了任務的核心邏輯和正確意圖。\n"
                "您的任務是：使用「原始正確提示」中的核心任務定義和邏輯，來調整「LLM 生成提示範本」。"
                f"目標是生成一個新的提示，這個提示既保留「LLM 生成提示範本」的優良風格和結構，又能準確無誤地表達「原始正確提示」中的任務邏輯，以便最好地完成主要任務：'{main_task_description_for_prm}'。\n\n" # 加入主要任務上下文
                f"LLM 生成提示範本 (風格偏好，但邏輯可能不完全準確):\n{llm_taste_prompt_text}\n\n"
                f"原始正確提示 (核心邏輯和意圖在此):\n{original_task_prompt_text}\n\n"
                "調整後，結合了正確邏輯與偏好風格的最終提示："
            )
        
        final_prompt_text_candidate = self.llm.generate(instruction_prompt, temperature=0.5) # CPM 生成時溫度可以適中
        
        if "LLM 未初始化" in final_prompt_text_candidate or "LLM 錯誤" in final_prompt_text_candidate:
            self.logger.error("ROT (CPM): LLM 呼叫失敗。無法生成最終提示。")
            return f"CPM 失敗：LLM 錯誤 ({final_prompt_text_candidate})"
        
        # 對 CPM 生成的最終提示進行 PRM 評估
        final_prm_score, final_prm_justification = self._evaluate_rot_artifact_with_prm(
            final_prompt_text_candidate,
            "CPM最終優化提示",
            main_task_description_for_prm
        )
        self.logger.info(f"ROT (CPM): 生成的最終提示 PRM 分數: {final_prm_score:.3f}")
        
        # 可以在這裡加入邏輯：如果 final_prm_score 低於某個閾值，
        # 可以嘗試返回未經 CPM 修改的、PRM 分數較高的 llm_taste_prompt_text，
        # 或者觸發一個更複雜的修復/重試機制。
        # 為了簡化，這裡我們總是返回 CPM 的輸出，但記錄其 PRM 分數。
        # 實際應用中，MASOrchestrator 可以檢查這個分數。
        
        return final_prompt_text_candidate # 返回 CPM 生成的提示文本

    def solve_task_with_final_prompt(self, final_prompt_text, problem_input):
        # 在解決任務前，可以再次（或由 MASOrchestrator）確認 final_prompt_text 的 PRM 分數。
        # 這裡假設 final_prompt_text 已經是經過 PRM 評估（可能在 CPM 階段）並且被認為是可接受的。
        
        full_solving_prompt = f"{final_prompt_text}\n\n現在，請基於上述定義和指令，解決以下具體問題：\n輸入：{problem_input}\n輸出："
        
        self.logger.info(f"ROT: 正在使用最終提示解決問題：'{problem_input}'")
        solution = self.llm.generate(full_solving_prompt, temperature=0.3) # 解決具體問題時溫度可以低一些以求精確

        if "LLM 未初始化" in solution or "LLM 錯誤" in solution:
            self.logger.error(f"ROT: 解決任務 '{problem_input}' 失敗。回應: {solution}")
            return f"解決方案生成失敗: {solution}"
        return solution.strip()

# --- 將範例用法封裝到函式中 ---
def run_rot_standalone_example_with_prm(api_key_for_example):
    logger_example = DefaultLogger()
    logger_example.info("執行 RoT (PRM風格) 獨立範例...")
    
    if not api_key_for_example:
        logger_example.error("ROT 獨立範例：未提供 API 金鑰。範例中的 LLM 和嵌入器呼叫將失敗。")

    llm_api_rot = GeminiLLMInterface(api_key=api_key_for_example, logger=logger_example)
    embedder_api = GeminiEmbeddingInterface(api_key=api_key_for_example, logger=logger_example)
    
    if not llm_api_rot.model:
        logger_example.error("ROT 獨立範例：LLM 介面模型未初始化。中止範例。")
        return
    if not embedder_api.api_key_configured_successfully:
        logger_example.error("ROT 獨立範例：嵌入介面 API 金鑰未配置成功。中止範例。")
        return

    # 這裡的 main_task_description 是 ROT 系統的「元任務」或「更高層次的任務」
    # ROT 的目標是為這個 main_task_description 生成一個好的執行提示
    main_task_for_rot_to_optimize_prompt_for = "解決24點遊戲問題：使用輸入的四個數字和加減乘除運算得到24。"

    rot_system = ReversalOfThought(llm_api_rot, embedder_api, similarity_threshold=0.6, logger=logger_example)

    # 24點遊戲的範例
    demonstrations_24 = [
       ("1 3 7 10", "對於 1,3,7,10，一個可能的解是 (10-7)*(1+3) 但這等於12，不是24。這個輸入組合無法簡單得到24。讓我們使用一個已知有解的例子。"),
       ("3 3 8 8", "8 / (3 - 8/3) = 24") 
    ]
    # 這是使用者（或另一個系統）提供的、針對24點遊戲的「理想」或「原始」任務提示
    original_user_prompt_for_24_game = (
       "任務定義：使用提供的四個整數（順序可以打亂，每個數字必須使用一次）以及加、減、乘、除運算和括號，構造一個結果為24的數學表達式。\n"
       "偽代碼：1. 生成數字的所有排列。2. 對於每個排列，嘗試所有可能的運算符組合和括號組合。3. 計算表達式結果，如果等於24，則返回該表達式。\n"
       "邏輯偽代碼：∀ P(perm(a,b,c,d)) ∃ Ops(op1,op2,op3) ∃ Grouping(g1,g2) such that Evaluate(Expression(P, Ops, Grouping)) = 24 → Print(Expression)\n"
       "案例範例：輸入：1 2 3 4  輸出：(1+3)*(2+4) = 24\n"
       "輸入-輸出格式：輸入：'w x y z' (四個以空格分隔的數字) 輸出：'數學表達式 = 24' 或 '無解'"
    )

    logger_example.info("\n--- ROT 獨立範例：開始 PGRR 預熱 (帶 PRM 評估) ---")
    # 傳遞 main_task_for_rot_to_optimize_prompt_for 給 PGRR，以便在內部評估候選提示
    llm_taste_prompt_text = rot_system.preference_guided_reverse_reasoning_warmup(
        demonstrations_24, 
        main_task_description_for_prm=main_task_for_rot_to_optimize_prompt_for, # 關鍵參數
        warm_iterations=2 # 減少迭代以節省時間
    )

    if llm_taste_prompt_text and "失敗" not in str(llm_taste_prompt_text).lower() and "llm 未初始化" not in str(llm_taste_prompt_text).lower() and "llm 錯誤" not in str(llm_taste_prompt_text).lower():
        logger_example.info(f"\n--- ROT 獨立範例：PGRR 選出的 LLM-Taste Prompt (部分) ---\n{str(llm_taste_prompt_text)[:500]}...")
        
        logger_example.info("\n--- ROT 獨立範例：開始 CPM (帶 PRM 評估) ---")
        # 傳遞 main_task_for_rot_to_optimize_prompt_for 給 CPM
        final_optimal_prompt_text = rot_system.cognitive_preference_manager(
            original_user_prompt_for_24_game, 
            str(llm_taste_prompt_text),
            main_task_description_for_prm=main_task_for_rot_to_optimize_prompt_for # 關鍵參數
        )
        
        if final_optimal_prompt_text and "失敗" not in str(final_optimal_prompt_text).lower() and "llm 未初始化" not in str(final_optimal_prompt_text).lower() and "llm 錯誤" not in str(final_optimal_prompt_text).lower():
            logger_example.info(f"\n--- ROT 獨立範例：CPM 後的最終優化提示 (部分) ---\n{str(final_optimal_prompt_text)[:500]}...")
            
            # 在這裡，MASOrchestrator 可以再次檢查 final_optimal_prompt_text 的 PRM 分數
            # （如果 _evaluate_rot_artifact_with_prm 不是在 CPM 內部最後一步調用，或者需要獨立驗證）

            problem_instance_24 = "4 6 8 8" # 換一個有解的例子
            logger_example.info(f"\n--- ROT 獨立範例：嘗試使用 RoT 生成的提示解決問題實例：'{problem_instance_24}' ---")
            solution = rot_system.solve_task_with_final_prompt(str(final_optimal_prompt_text), problem_instance_24)
            logger_example.info(f"\n--- ROT 獨立範例：對於 '{problem_instance_24}' 的解決方案 ---\n{solution}")
        else:
            logger_example.error(f"ROT 獨立範例：CPM 階段失敗，無法繼續解決任務。CPM輸出: {final_optimal_prompt_text}")
    else:
       logger_example.error(f"ROT 獨立範例：未能生成 LLM-taste 提示。PGRR 階段可能已失敗。PGRR輸出: {llm_taste_prompt_text}")
    logger_example.info("\n--- ROT (PRM風格) 獨立範例結束 ---")

# --- 主執行區塊 ---
if __name__ == "__main__":
    logger_main = DefaultLogger()
    logger_main.info("ROT.py 作為腳本獨立執行...")
    api_key_for_standalone = os.getenv("GEMINI_API_KEY") 
    
    if not api_key_for_standalone:
        logger_main.warning("警告：未在環境變數中找到 GEMINI_API_KEY。")
        logger_main.warning("ROT.py 獨立範例需要有效的 API 金鑰才能與 Gemini API 互動。")
        run_rot_standalone_example_with_prm(None) # 傳遞 None，讓初始化失敗並打印錯誤
    else:
        logger_main.info(f"ROT.py 獨立執行：偵測到 API 金鑰尾碼 ...{api_key_for_standalone[-4:]}，將用於範例。")
        run_rot_standalone_example_with_prm(api_key_for_standalone)
    
    logger_main.info("ROT.py 獨立執行範例流程結束。")
