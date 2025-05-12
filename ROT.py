import google.generativeai as genai
import numpy as np
import os
from dotenv import load_dotenv # 用於載入 .env 檔案

# --- 設定 ---
# 載入 .env 檔案中的環境變數
load_dotenv() # 如果您使用 .env 檔案管理 API 金鑰，這行會載入它

# 此處的 GEMINI_API_KEY 和 ACTIVE_GEMINI_API_KEY 主要用於 ROT.py 獨立執行時的備援
# 當由 mas_main.py 協調時，API 金鑰會由協調器傳入
GEMINI_API_KEY_FROM_ENV = os.getenv("GEMINI_API_KEY")
ACTIVE_GEMINI_API_KEY = None # 將由 __init__ 中的邏輯或全域設定決定

# --- Gemini LLM 介面 (修改版) ---
class GeminiLLMInterface:
    def __init__(self, model_name="gemini-1.5-flash", api_key=None): # 新增 api_key 參數
        self.model = None
        # 優先使用傳入的 api_key，其次是環境變數
        effective_api_key = api_key or GEMINI_API_KEY_FROM_ENV

        if not effective_api_key:
            print("ROT.GeminiLLMInterface: API 金鑰未提供（未傳入也未在環境變數中找到）。LLM 將無法運作。")
            return

        try:
            # 確保 genai 使用正確的 API 金鑰進行配置
            # genai.configure 是全域性的。如果金鑰不同，後來的配置會覆蓋先前的。
            # 檢查目前 genai 配置的 API 金鑰是否與我們要使用的金鑰一致
            # google.auth.default() 返回的 credentials 可能沒有直接暴露 api_key 的方法，
            # 因此我們假設如果提供了 effective_api_key，就用它來配置。
            # 如果 genai.configure 多次使用相同的金鑰呼叫，通常是無害的。
            if not hasattr(genai, '_is_configured_by_rot_mas') or genai._is_configured_by_rot_mas != effective_api_key:
                genai.configure(api_key=effective_api_key)
                genai._is_configured_by_rot_mas = effective_api_key # 標記已配置
                print(f"ROT.GeminiLLMInterface: 已使用 API 金鑰尾碼 ...{effective_api_key[-4:]} 配置 genai。")

            self.model = genai.GenerativeModel(model_name)
            print(f"ROT.GeminiLLMInterface 已使用模型 {model_name} 初始化。")
        except Exception as e:
            print(f"初始化 ROT Gemini GenerativeModel ({model_name}) 失敗: {e}")
            self.model = None # 確保初始化失敗時 self.model 為 None

    def generate(self, prompt_text):
        if not self.model:
            print("ROT.GeminiLLMInterface: LLM 模型未初始化。無法生成內容。")
            return "LLM 未初始化或 API 金鑰錯誤"
        try:
            response = self.model.generate_content(prompt_text)
            # 檢查 response.parts 是否存在且非空 (適用於 gemini-1.5-flash 等較新模型)
            if hasattr(response, 'parts') and response.parts:
                 return "".join(part.text for part in response.parts if hasattr(part, 'text'))
            # 如果 parts 為空或不存在，嘗試直接從 response.text 獲取 (適用於舊版或不同結構的回應)
            elif hasattr(response, 'text'):
                 return response.text
            # 處理可能的封鎖情況
            elif hasattr(response, 'prompt_feedback') and \
                 response.prompt_feedback.block_reason != genai.types.BlockReason.BLOCK_REASON_UNSPECIFIED:
                block_reason_str = genai.types.BlockReason(response.prompt_feedback.block_reason).name
                print(f"ROT.GeminiLLMInterface: 提示因 {block_reason_str} 被封鎖。")
                return f"LLM 錯誤：提示因 {block_reason_str} 被封鎖。"
            else:
                print(f"ROT.GeminiLLMInterface: LLM 未返回預期格式的內容。Response: {response}")
                return "LLM 錯誤：未產生有效內容。"

        except Exception as e:
            print(f"ROT.GeminiLLMInterface: Gemini API 呼叫錯誤 (generate): {e}")
            return f"LLM 錯誤: {e}"

    def generate_with_simulated_score(self, prompt_text):
        if not self.model:
            print("ROT.GeminiLLMInterface: LLM 模型未初始化。無法生成內容並評分。")
            return "LLM 未初始化或 API 金鑰錯誤", 0.0

        response_text = self.generate(prompt_text)
        simulated_score = 0.0
        if "LLM 未初始化" in response_text or "LLM 錯誤" in response_text:
             simulated_score = 0.0
        else:
            # 簡化的分數：可以基於長度，或者如果沒有更好的啟發式方法，則為固定值。
            simulated_score = float(len(response_text)) / 100.0 # 粗略標準化
        return response_text, simulated_score

# --- Gemini 嵌入介面 (修改版) ---
class GeminiEmbeddingInterface:
    def __init__(self, model_name="models/embedding-001", api_key=None): # 新增 api_key 參數
        self.model_name = model_name
        self.api_key_configured = False
        effective_api_key = api_key or GEMINI_API_KEY_FROM_ENV

        if not effective_api_key:
            print("ROT.GeminiEmbeddingInterface: API 金鑰未提供。嵌入功能將無法運作。")
            return

        try:
            # 與 LLM 介面類似的配置邏輯
            if not hasattr(genai, '_is_configured_by_rot_mas') or genai._is_configured_by_rot_mas != effective_api_key:
                genai.configure(api_key=effective_api_key)
                genai._is_configured_by_rot_mas = effective_api_key # 標記已配置
                print(f"ROT.GeminiEmbeddingInterface: 已使用 API 金鑰尾碼 ...{effective_api_key[-4:]} 配置 genai (用於嵌入)。")

            # 此處不需要實例化模型，genai.embed_content 直接使用配置好的 API 金鑰
            self.api_key_configured = True
            print(f"ROT.GeminiEmbeddingInterface 已為模型 {model_name} 初始化。")
        except Exception as e:
            print(f"為 ROT.GeminiEmbeddingInterface 設定 API 金鑰時發生錯誤: {e}")

    def _get_embedding(self, text):
        if not self.api_key_configured:
            print("ROT.GeminiEmbeddingInterface: API 金鑰未設定或配置失敗。無法獲取嵌入。")
            return None
        try:
            result = genai.embed_content(model=self.model_name, content=text, task_type="SEMANTIC_SIMILARITY")
            return result['embedding']
        except Exception as e:
            print(f"ROT.GeminiEmbeddingInterface: Gemini API 呼叫錯誤 (embed_content for '{text[:50]}...'): {e}")
            return None

    def calculate_similarity(self, text1, text2):
        if not self.api_key_configured:
            print("ROT.GeminiEmbeddingInterface: API 金鑰未設定，無法計算相似度。")
            return 0.0

        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        if emb1 is None or emb2 is None:
            print("ROT.GeminiEmbeddingInterface: 無法計算相似度，因為一個或多個嵌入向量為 None。")
            return 0.0

        try:
            emb1_np = np.array(emb1)
            emb2_np = np.array(emb2)
            norm_emb1 = np.linalg.norm(emb1_np)
            norm_emb2 = np.linalg.norm(emb2_np)
            if norm_emb1 == 0 or norm_emb2 == 0: # 避免除以零
                return 0.0
            similarity = np.dot(emb1_np, emb2_np) / (norm_emb1 * norm_emb2)
            return float(similarity)
        except Exception as e:
            print(f"ROT.GeminiEmbeddingInterface: 計算餘弦相似度時出錯: {e}")
            return 0.0

# --- ReversalOfThought 類別 (保持不變，但其依賴的介面已更新) ---
class ReversalOfThought:
    def __init__(self, llm_interface, embedding_model_interface, similarity_threshold=0.7):
        self.llm = llm_interface
        self.embedder = embedding_model_interface
        self.similarity_threshold = similarity_threshold

    def _prompt_for_reverse_reasoning(self, demonstrations_text):
        # 提示詞本身可以根據需要調整為中文或保持英文，取決於LLM的理解能力和您的偏好
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
            "您的綜合定義："
        )
        return prompt

    def _prompt_for_pairwise_preference(self, response_A, response_B):
        prompt = (
            "請選擇您更偏好的指令定義（A 或 B），哪個最能捕捉任務核心：\n"
            "請仔細評估兩者在清晰度、完整性、準確性和易理解性方面的表現。\n\n"
            f"(A)\n{response_A}\n\n(B)\n{response_B}\n\n您的選擇 (請僅回答 A 或 B) 以及簡要理由："
        )
        return prompt

    def preference_guided_reverse_reasoning_warmup(self, demonstrations, warm_iterations=3):
        demo_text = ""
        for i, (inp, outp) in enumerate(demonstrations):
            demo_text += f"範例 {i+1}:\n輸入: {inp}\n輸出: {outp}\n\n"

        candidate_responses = []
        print(f"\n--- ROT: 執行反向推理預熱 ({warm_iterations} 次迭代) ---")
        for i in range(warm_iterations):
            rr_prompt = self._prompt_for_reverse_reasoning(demo_text)
            response_text, response_prob_score = self.llm.generate_with_simulated_score(rr_prompt)

            if "LLM 未初始化" in response_text or "LLM 錯誤" in response_text:
                print(f"ROT: 預熱迭代 {i+1} 失敗，因為 LLM 錯誤或未初始化。回應: {response_text}")
                continue
            
            candidate_responses.append({'text': response_text, 'prob_score': response_prob_score, 'id': f"cand_{i}"})
            print(f"ROT: 預熱迭代 {i+1} 生成候選提示 (模擬分數: {response_prob_score:.4f})")

        if not candidate_responses:
            print("ROT: PGRR 預熱未能生成任何候選回應。")
            return "PGRR 預熱失敗：無候選回應"

        print("\n--- ROT: 執行成對偏好評估 ---")
        preference_matrix = {} # (id_A, id_B) -> preference_score_A_over_B (1 for A, 0 for B, 0.5 for tie)

        num_candidates = len(candidate_responses)
        if num_candidates < 2:
            print("ROT: 候選回應少於2個，跳過成對偏好評估。直接使用唯一的候選。")
            return candidate_responses[0]['text']


        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                resp_A_info = candidate_responses[i]
                resp_B_info = candidate_responses[j]
                
                pref_prompt_A_vs_B = self._prompt_for_pairwise_preference(resp_A_info['text'], resp_B_info['text'])
                choice_response_A_vs_B = self.llm.generate(pref_prompt_A_vs_B)

                if "LLM 未初始化" in choice_response_A_vs_B or "LLM 錯誤" in choice_response_A_vs_B:
                    print(f"ROT: 偏好評估 ({resp_A_info['id']} vs {resp_B_info['id']}) 失敗。回應: {choice_response_A_vs_B}")
                    # 將此比較視為平手
                    preference_matrix[(resp_A_info['id'], resp_B_info['id'])] = 0.5
                    preference_matrix[(resp_B_info['id'], resp_A_info['id'])] = 0.5
                    continue

                choice_upper = choice_response_A_vs_B.strip().upper()
                
                # 解析LLM的選擇，允許理由跟在選擇之後
                chosen_option = None
                if choice_upper.startswith("A"):
                    chosen_option = "A"
                elif choice_upper.startswith("B"):
                    chosen_option = "B"

                if chosen_option == "A":
                    preference_matrix[(resp_A_info['id'], resp_B_info['id'])] = 1.0
                    preference_matrix[(resp_B_info['id'], resp_A_info['id'])] = 0.0
                elif chosen_option == "B":
                    preference_matrix[(resp_A_info['id'], resp_B_info['id'])] = 0.0
                    preference_matrix[(resp_B_info['id'], resp_A_info['id'])] = 1.0
                else: # 中性或不清楚
                    preference_matrix[(resp_A_info['id'], resp_B_info['id'])] = 0.5
                    preference_matrix[(resp_B_info['id'], resp_A_info['id'])] = 0.5
                
                winner_char = chosen_option if chosen_option else '平手/未知'
                print(f"ROT: 偏好：候選 {resp_A_info['id']} vs {resp_B_info['id']} -> 勝者: {winner_char} (LLM 選擇原始回應: '{choice_response_A_vs_B[:50]}...')")
        
        final_scores = []
        for i in range(num_candidates):
            current_candidate_id = candidate_responses[i]['id']
            total_preference_score = 0
            num_comparisons = 0
            for j in range(num_candidates):
                if i == j: continue
                other_candidate_id = candidate_responses[j]['id']
                # 獲取 current_candidate_id 相對於 other_candidate_id 的偏好分數
                total_preference_score += preference_matrix.get((current_candidate_id, other_candidate_id), 0.5)
                num_comparisons +=1
            
            avg_preference_score = (total_preference_score / num_comparisons) if num_comparisons > 0 else 0.5
            
            # 結合生成機率分數和平均偏好分數
            combined_score = (candidate_responses[i]['prob_score'] + avg_preference_score) / 2.0
            final_scores.append({'id': current_candidate_id, 'score': combined_score, 'text': candidate_responses[i]['text']})
            
        if not final_scores:
            print("ROT: 未能計算最終分數。PGRR 排序失敗。")
            return "PGRR 排序失敗：無最終分數"

        best_prompt_info = max(final_scores, key=lambda x: x['score'])
        print(f"\n--- ROT: 最佳 LLM-Taste 提示 (ID: {best_prompt_info['id']}, 綜合分數: {best_prompt_info['score']:.4f}) ---")
        return best_prompt_info['text']

    def _extract_task_definition(self, prompt_text):
        # 此函數嘗試從完整的提示文字中提取「任務定義」部分。
        # 這是一個啟發式方法，可能需要根據提示的實際格式進行調整。
        lines = str(prompt_text).splitlines()
        task_def_lines = []
        in_task_def_section = False
        
        # 關鍵字來識別任務定義部分的開始和結束
        # 注意處理中英文冒號和大小寫
        start_keywords = ["task definition:", "任務定義：", "task definition：", "任務定義:"]
        # 遇到這些關鍵字時，認為任務定義部分結束
        end_keywords = [
            "pseudocode:", "偽代碼：", "pseudocode：", "偽代碼:",
            "logical pseudocode:", "邏輯偽代碼：",
            "case examples:", "案例範例：",
            "input-output format:", "輸入-輸出格式："
        ]

        for line in lines:
            line_lower = line.strip().lower()
            
            if not in_task_def_section:
                for keyword in start_keywords:
                    if line_lower.startswith(keyword):
                        in_task_def_section = True
                        # 取冒號後面的內容
                        content_after_keyword = line.split(keyword[-1], 1)[-1].strip()
                        if content_after_keyword: # 如果關鍵字同行有內容
                            task_def_lines.append(content_after_keyword)
                        break 
            elif in_task_def_section:
                is_end_keyword_found = False
                for keyword in end_keywords:
                    if line_lower.startswith(keyword):
                        is_end_keyword_found = True
                        break
                if is_end_keyword_found or not line.strip(): # 如果遇到下個部分的關鍵字或空行
                    break 
                task_def_lines.append(line.strip())
        
        extracted_definition = "\n".join(task_def_lines).strip()
        
        if not extracted_definition:
            # print("ROT: 警告：未能從提示中提取明確的任務定義部分。將使用完整提示進行相似度計算。")
            return str(prompt_text) # 回傳整個提示作為備用
        # print(f"ROT: 提取的任務定義用於相似度計算:\n'''{extracted_definition}'''")
        return extracted_definition


    def cognitive_preference_manager(self, original_task_prompt_text, llm_taste_prompt_text):
        print("\n--- ROT: 執行認知偏好管理器 (CPM) ---")
        original_task_def_text = self._extract_task_definition(original_task_prompt_text)
        llm_taste_task_def_text = self._extract_task_definition(llm_taste_prompt_text)
        
        similarity = self.embedder.calculate_similarity(original_task_def_text, llm_taste_task_def_text)
        print(f"ROT: 原始任務定義與 LLM-taste 任務定義之間的相似度: {similarity:.4f} (閾值: {self.similarity_threshold})")

        final_prompt_text = ""
        instruction_prompt = ""
        if similarity >= self.similarity_threshold:
            print(f"ROT (CPM): 檢測為已知任務 (相似度 {similarity:.4f} >= {self.similarity_threshold})。將嘗試聚合兩個提示的優點。")
            instruction_prompt = (
                "請綜合以下兩個關於同一任務的描述/提示，目標是創建一個單一的、更優越的提示版本。"
                "這個新版本應該融合兩者的最強點，特別是在任務定義的清晰度、偽代碼的實用性、邏輯表達的準確性、範例的相關性以及輸入/輸出格式的明確性方面。"
                "請確保最終提示既全面又易於LLM理解和執行。\n\n"
                f"提示 1 (原始或基準提示):\n{original_task_prompt_text}\n\n"
                f"提示 2 (LLM 生成的候選提示):\n{llm_taste_prompt_text}\n\n"
                "綜合後的最佳提示："
            )
        else:
            print(f"ROT (CPM): 檢測為未知或顯著不同的任務 (相似度 {similarity:.4f} < {self.similarity_threshold})。將嘗試調整風格範本以符合原始任務邏輯。")
            instruction_prompt = (
                "以下有兩個提示。 「LLM 生成提示範本」可能在任務理解上不完全準確，但其整體結構和風格（例如，如何組織任務定義、偽代碼、範例等部分）是偏好的。"
                "「原始正確提示」包含了任務的核心邏輯和正確意圖。\n"
                "您的任務是：使用「原始正確提示」中的核心任務定義和邏輯，來調整「LLM 生成提示範本」。"
                "目標是生成一個新的提示，這個提示既保留「LLM 生成提示範本」的優良風格和結構，又能準確無誤地表達「原始正確提示」中的任務邏輯。\n\n"
                f"LLM 生成提示範本 (風格偏好，但邏輯可能不完全準確):\n{llm_taste_prompt_text}\n\n"
                f"原始正確提示 (核心邏輯和意圖在此):\n{original_task_prompt_text}\n\n"
                "調整後，結合了正確邏輯與偏好風格的最終提示："
            )
        
        final_prompt_text = self.llm.generate(instruction_prompt)
        
        if "LLM 未初始化" in final_prompt_text or "LLM 錯誤" in final_prompt_text:
            print("ROT (CPM): LLM 呼叫失敗。無法生成最終提示。")
            return f"CPM 失敗：LLM 錯誤 ({final_prompt_text})"
            
        # print(f"\n--- ROT (CPM): 生成的最終優化提示 ---\n{final_prompt_text}")
        return final_prompt_text

    def solve_task_with_final_prompt(self, final_prompt, problem_input):
        # 在最終提示後附加實際問題
        # 確保 final_prompt 已經包含了完整的任務定義和指令結構
        full_solving_prompt = f"{final_prompt}\n\n現在，請基於上述定義和指令，解決以下具體問題：\n輸入：{problem_input}\n輸出："
        
        print(f"ROT: 正在使用最終提示解決問題：'{problem_input}'")
        solution = self.llm.generate(full_solving_prompt)

        if "LLM 未初始化" in solution or "LLM 錯誤" in solution:
            print(f"ROT: 解決任務 '{problem_input}' 失敗。回應: {solution}")
            return f"解決方案生成失敗: {solution}"
        return solution.strip()


# --- 範例使用 (用於 ROT.py 獨立測試) ---
def run_rot_standalone_example():
    print("執行 RoT 獨立範例...")
    # 確保 API 金鑰已設定 (例如，透過 .env 或直接在程式碼頂部設定 GEMINI_API_KEY_FROM_ENV)
    if not GEMINI_API_KEY_FROM_ENV:
        print("警告：未在環境變數中找到 GEMINI_API_KEY。ROT 獨立範例可能無法正確執行 LLM 呼叫。")
        # 可以選擇在此處返回或繼續執行，但 LLM 呼叫會失敗
        # return

    # 使用修改後的介面，即使在獨立執行時，如果環境變數已設定，它們也能運作
    llm_api_rot = GeminiLLMInterface()
    embedder_api = GeminiEmbeddingInterface()
    
    # 檢查 LLM 和嵌入器是否至少看起來已配置（不保證 API 金鑰有效）
    if not llm_api_rot.model: # 檢查模型是否已初始化
        print("ROT 獨立範例：LLM 介面模型未初始化。中止範例。")
        return
    if not embedder_api.api_key_configured:
        print("ROT 獨立範例：嵌入介面 API 金鑰未配置。中止範例。")
        return

    rot_system = ReversalOfThought(llm_api_rot, embedder_api, similarity_threshold=0.6)

    # 24點遊戲的範例 (來自原始 ROT.py，但輸出可能需要調整以符合實際的24點解法)
    demonstrations_24 = [
       ("1 3 7 10", "((10 - 7) * (1 + 3)) = 12  <-- 注意：此處原始碼範例的輸出並非24，可能用於測試或簡化。一個24的解是 (10-1)*(7-3) = 9*4 = 36 (不對) 或 (7-3)*(10-1) ... 實際上 1,3,7,10 無法組成24。我們用一個可以的：1,2,3,4 -> (4+2)*(3+1)=24 (不對), (4-2+1)*3=9 (不對) (3+1)*(2*4-2) (4/(1-3/7)) * 10 (不對)。用 (8,3,8,3) -> (8-3)*(8-3)=25 (不對) (8/ (3 - 8/3)) = 24"),
       ("3 3 8 8", "8 / (3 - 8/3) = 24") # (8 / (9/3 - 8/3)) = 8 / (1/3) = 24. 這個是對的。
    ]
    original_user_prompt_for_24_game = (
       "任務定義：使用提供的四個整數（順序可以打亂，每個數字必須使用一次）以及加、減、乘、除運算和括號，構造一個結果為24的數學表達式。\n"
       "偽代碼：1. 生成數字的所有排列。2. 對於每個排列，嘗試所有可能的運算符組合和括號組合。3. 計算表達式結果，如果等於24，則返回該表達式。\n"
       "邏輯偽代碼：∀ P(perm(a,b,c,d)) ∃ Ops(op1,op2,op3) ∃ Grouping(g1,g2) such that Evaluate(Expression(P, Ops, Grouping)) = 24 → Print(Expression)\n"
       "案例範例：輸入：1 2 3 4  輸出：(4 * (1 + 2)) + 3 = 15 (不對, 應為 (1+2+3)*4 = 24 (不對), (4-1)*(2+3)=15 (不對), (2*3-1)*4 (不對) (4-1)*(3+2+?) (4+2)*(1+3)=24 (不對), 4*(2+1)*3 (不對), (3-1)*(4*2)=16 (不對) ((2+3)-1)*4 (不對) (1+3)*(2+4)=24)\n" # 1 2 3 4 -> (1+3)*(2+4) = 4*6 = 24.
       "輸入-輸出格式：輸入：'w x y z' (四個以空格分隔的數字) 輸出：'數學表達式 = 24' 或 '無解'"
    )

    print("\n--- ROT 獨立範例：開始 PGRR 預熱 ---")
    llm_taste_prompt = rot_system.preference_guided_reverse_reasoning_warmup(demonstrations_24, warm_iterations=2)

    if llm_taste_prompt and "失敗" not in str(llm_taste_prompt) and "LLM 未初始化" not in str(llm_taste_prompt):
        print(f"\n--- ROT 獨立範例：生成的 LLM-Taste Prompt ---\n{llm_taste_prompt}")
        print("\n--- ROT 獨立範例：開始 CPM ---")
        final_optimal_prompt = rot_system.cognitive_preference_manager(original_user_prompt_for_24_game, llm_taste_prompt)
        
        if final_optimal_prompt and "失敗" not in str(final_optimal_prompt) and "LLM 未初始化" not in str(final_optimal_prompt):
            print(f"\n--- ROT 獨立範例：CPM 後的最終優化提示 ---\n{final_optimal_prompt}")
            problem_instance_24 = "1 2 3 4" # 這個組合有解: (1+3)*(2+4)=24
            print(f"\n--- ROT 獨立範例：嘗試使用 RoT 生成的提示解決問題實例：'{problem_instance_24}' ---")
            solution = rot_system.solve_task_with_final_prompt(final_optimal_prompt, problem_instance_24)
            print(f"\n--- ROT 獨立範例：對於 '{problem_instance_24}' 的解決方案 ---\n{solution}")
        else:
            print("ROT 獨立範例：CPM 階段失敗，無法繼續解決任務。")
    else:
       print("ROT 獨立範例：未能生成 LLM-taste 提示。PGRR 階段可能已失敗。")

if __name__ == "__main__":
    # 這裡的 if __name__ == "__main__": 區塊用於 ROT.py 腳本的獨立測試。
    # 當 mas_main.py 導入此文件時，此區塊不會執行。
    print("ROT.py 作為腳本獨立執行...")
    run_rot_standalone_example()
    print("ROT.py 獨立執行範例結束。")
