# mas_main_enhanced.py
import os
# from dotenv import load_dotenv # 如果您希望集中在此處載入 .env

# 從使用者提供的腳本中導入必要的類別
# 假設 GOT.py, LOT.py, ROT.py (修改版) 都在同一個目錄下
from GOT import GraphOfThoughts, GeminiLLM as GotGeminiLLM
from LOT import LayerOfThoughts, GeminiLLMInterface as LotGeminiLLMInterface
from ROT import ReversalOfThought, GeminiLLMInterface as RotGeminiLLMInterface, GeminiEmbeddingInterface as RotGeminiEmbeddingInterface

# --- API 金鑰設定 ---
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_DEFAULT_GEMINI_API_KEY_HERE") # 優先使用環境變數

if API_KEY == "YOUR_DEFAULT_GEMINI_API_KEY_HERE" and not os.getenv("GEMINI_API_KEY"):
    print("警告：正在使用佔位符 API 金鑰。請設定您自己的 GEMINI_API_KEY 環境變數或直接在程式碼中替換 API_KEY 的值。")

class TerminalLogger:
    """
    一個簡單的日誌記錄器，用於在終端機上以結構化方式輸出資訊。
    """
    def __init__(self, verbose=True):
        self.verbose = verbose

    def _print(self, tag, stage, message, indent_level=0):
        if self.verbose:
            indent = "  " * indent_level
            print(f"[{tag}][{stage}]{indent} {message}")

    def thoughtflow(self, stage, message, detail_level=0):
        self._print("THOUGHTFLOW", stage, message, detail_level)

    def discussion(self, stage, message, detail_level=0):
        self._print("DISCUSSION", stage, message, detail_level)

    def answer(self, source_system, content, is_final=False, detail_level=0):
        tag_prefix = "FINAL " if is_final else "INTERMEDIATE "
        indent = "  " * detail_level
        if self.verbose:
            print(f"[{tag_prefix}ANSWER][{source_system}]{indent} \n{content}")
            print(f"{'='*70}")


    def section_start(self, section_name):
        if self.verbose:
            print(f"\n{'#'*20} STARTING: {section_name.upper()} {'#'*20}")

    def section_end(self, section_name):
        if self.verbose:
            print(f"{'#'*20} FINISHED: {section_name.upper()} {'#'*20}\n")

    def error(self, stage, message):
        print(f"[ERROR][{stage}] {message}")

    def info(self, message):
         if self.verbose:
            print(f"[INFO] {message}")

class MASOrchestrator:
    """
    多代理系統協調器。
    負責初始化 GOT, LOT, ROT 系統，並協調它們之間的互動以完成任務。
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.logger = TerminalLogger() # 初始化日誌記錄器

        if not self.api_key or self.api_key == "YOUR_DEFAULT_GEMINI_API_KEY_HERE":
            self.logger.error("MASOrchestrator", "無效的 API 金鑰。請確保已正確設定。")
            raise ValueError("MASOrchestrator 需要有效的 API 金鑰。")

        self.logger.info(f"正在使用 API 金鑰尾碼為 ...{self.api_key[-4:]} 初始化各系統。")

        try:
            # 初始化 GOT 系統
            self.got_llm = GotGeminiLLM(api_key=self.api_key)
            self.got_system = GraphOfThoughts(llm_interface=self.got_llm)
            self.logger.info("GraphOfThoughts (GOT) 系統已初始化。")

            # 初始化 LOT 系統
            self.lot_llm = LotGeminiLLMInterface(api_key=self.api_key)
            self.lot_system = LayerOfThoughts(llm_interface=self.lot_llm)
            self.logger.info("LayerOfThoughts (LOT) 系統已初始化。")

            # 初始化 ROT 系統
            self.rot_llm = RotGeminiLLMInterface(api_key=self.api_key)
            self.rot_embedder = RotGeminiEmbeddingInterface(api_key=self.api_key)
            self.rot_system = ReversalOfThought(
                llm_interface=self.rot_llm,
                embedding_model_interface=self.rot_embedder
            )
            self.logger.info("ReversalOfThought (ROT) 系統已初始化。")
        except Exception as e:
            self.logger.error("Initialization", f"初始化子系統時發生錯誤: {e}")
            raise

    def run_collaborative_task(self, initial_task_description, rot_demonstrations, problem_instance_for_rot_final_solve=None):
        self.logger.section_start("協作任務")
        self.logger.info(f"初始任務描述: {initial_task_description}")

        refined_task_prompt = initial_task_description
        llm_taste_prompt_text_from_rot = None

        # === 階段 1: 任務定義精煉 (ROT) ===
        self.logger.section_start("階段 1: ROT - 精煉任務定義")
        self.logger.thoughtflow("ROT", "準備用於 ROT 的原始使用者提示框架。")
        original_user_prompt_for_rot = (
            f"任務定義：{initial_task_description}\n"
            "偽代碼：[請LLM根據任務和範例填寫詳細的逐步執行流程]\n"
            "邏輯偽代碼：[請LLM將上述偽代碼轉換為更形式化的邏輯表達式]\n"
            "案例範例：[請LLM從下方提供的輸入輸出範例中選擇或生成一個代表性案例]\n"
            "輸入-輸出格式：輸入：[清晰定義輸入的結構和類型] 輸出：[清晰定義輸出的結構和類型]"
        )
        demo_text_for_original_prompt = ""
        for i, (inp, outp) in enumerate(rot_demonstrations):
            demo_text_for_original_prompt += f"  範例 {i+1}:\n    輸入: {inp}\n    輸出: {outp}\n"
        if demo_text_for_original_prompt:
             original_user_prompt_for_rot += f"\n提供的輸入輸出範例參考：\n{demo_text_for_original_prompt}"
        self.logger.discussion("ROT", f"用於PGRR的原始使用者提示框架:\n{original_user_prompt_for_rot[:300]}...")

        self.logger.thoughtflow("ROT", "開始執行 Preference-Guided Reverse Reasoning (PGRR) 預熱...")
        llm_taste_prompt_text_from_rot = self.rot_system.preference_guided_reverse_reasoning_warmup(
            demonstrations=rot_demonstrations,
            warm_iterations=2 # 保持為2以節省時間，實際應用可增加
        )

        if llm_taste_prompt_text_from_rot and "失敗" not in str(llm_taste_prompt_text_from_rot).lower() and "LLM 未初始化" not in str(llm_taste_prompt_text_from_rot):
            self.logger.discussion("ROT", f"PGRR 生成的 LLM-Taste Prompt (部分內容):\n{str(llm_taste_prompt_text_from_rot)[:300]}...")
            self.logger.thoughtflow("ROT", "開始執行 Cognitive Preference Manager (CPM)...")
            final_optimal_prompt_from_rot = self.rot_system.cognitive_preference_manager(
                original_task_prompt_text=original_user_prompt_for_rot,
                llm_taste_prompt_text=str(llm_taste_prompt_text_from_rot)
            )
            if final_optimal_prompt_from_rot and "失敗" not in str(final_optimal_prompt_from_rot).lower() and "LLM 未初始化" not in str(final_optimal_prompt_from_rot):
                refined_task_prompt = str(final_optimal_prompt_from_rot)
                self.logger.discussion("ROT", f"CPM 精煉後的任務提示 (部分內容):\n{refined_task_prompt[:300]}...")
            else:
                self.logger.error("ROT", "CPM 失敗或未返回有效提示。將嘗試使用 PGRR 的 LLM-Taste prompt。")
                if llm_taste_prompt_text_from_rot : # 確保它不是 None
                     refined_task_prompt = str(llm_taste_prompt_text_from_rot)
                else:
                    self.logger.error("ROT", "PGRR 也未能提供有效的 LLM-Taste prompt。將使用初始任務描述。")
                    # refined_task_prompt 保持為 initial_task_description
        else:
            self.logger.error("ROT", "PGRR 未能成功生成 LLM-Taste prompt。後續階段將使用初始任務描述。")
            # refined_task_prompt 保持為 initial_task_description
        self.logger.section_end("階段 1: ROT - 精煉任務定義")

        # === 階段 2: 解決方案腦力激盪與結構化 (GOT) ===
        self.logger.section_start("階段 2: GOT - 腦力激盪與結構化解決方案")
        got_task_description_for_generation = (
            f"基於以下（可能經過精煉的）任務描述，請產生3個核心想法或解決方案的主要方向。"
            f"確保這些想法具有創新性、可行性，並直接回應任務需求。\n\n任務描述：\n{refined_task_prompt}"
        )
        self.logger.thoughtflow("GOT", "為產生初始思維準備任務描述。")
        self.logger.discussion("GOT", f"用於生成思維的任務描述 (部分):\n{got_task_description_for_generation[:300]}...")

        initial_thoughts = self.got_system.generate_thoughts(
            task_description=got_task_description_for_generation,
            num_thoughts=3
        )
        if not initial_thoughts:
            self.logger.error("GOT", "未能產生初始思維。任務中止。")
            self.logger.section_end("階段 2: GOT - 腦力激盪與結構化解決方案")
            self.logger.section_end("協作任務")
            return {"error": "GOT 未能產生初始思維。", "final_rot_prompt_for_reference": refined_task_prompt}

        self.logger.thoughtflow("GOT", f"產生了 {len(initial_thoughts)} 個初始思維。現在進行評分...")
        for i, thought in enumerate(initial_thoughts):
            self.logger.discussion("GOT", f"評分初始思維 {i+1} (ID: {thought.id}): '{thought.content[:100]}...'", detail_level=1)
            self.got_system.score_thought_with_llm(thought.id, "概念的原創性、清晰度以及與任務的相關性")

        ranked_thoughts = self.got_system.rank_thoughts()
        if not ranked_thoughts:
            self.logger.error("GOT", "沒有思維可供排序。任務中止。")
            self.logger.section_end("階段 2: GOT - 腦力激盪與結構化解決方案")
            self.logger.section_end("協作任務")
            return {"error": "GOT: 沒有思維可供排序。", "final_rot_prompt_for_reference": refined_task_prompt}

        best_initial_thought = ranked_thoughts[0]
        self.logger.discussion("GOT", f"選定的最佳初始思維 (ID {best_initial_thought.id}, 分數 {best_initial_thought.score:.2f}): {best_initial_thought.content[:150]}...")
        self.logger.thoughtflow("GOT", "準備闡述選定的最佳初始思維。")

        got_task_description_for_elaboration = (
            f"請詳細闡述以下選定的核心想法，使其成為一個更完整的初步解決方案或計劃框架。"
            f"考慮其關鍵組成部分、潛在挑戰和預期成果。\n\n選定想法：\n{best_initial_thought.content}\n\n原始任務背景：\n{refined_task_prompt}"
        )
        elaborated_thoughts = self.got_system.generate_thoughts(
            task_description=got_task_description_for_elaboration,
            num_thoughts=1,
            from_thought_ids=[best_initial_thought.id]
        )

        got_output_for_lot = best_initial_thought.content # 預設使用最佳初始思維
        if not elaborated_thoughts:
            self.logger.discussion("GOT", "未能闡述最佳思維。將使用原始最佳初始思維傳遞給 LOT。", detail_level=1)
        else:
            elaborated_thought = elaborated_thoughts[0]
            self.logger.discussion("GOT", f"評分已闡述的思維 (ID: {elaborated_thought.id}): '{elaborated_thought.content[:100]}...'", detail_level=1)
            self.got_system.score_thought_with_llm(elaborated_thought.id, "闡述內容的完整性、邏輯性和可行性")
            if elaborated_thought.score >= best_initial_thought.score :
                 got_output_for_lot = elaborated_thought.content
                 self.logger.discussion("GOT", f"已闡述的思維被選中 (ID {elaborated_thought.id}, 分數 {elaborated_thought.score:.2f}): {elaborated_thought.content[:150]}...", detail_level=1)
            else:
                 self.logger.discussion("GOT", f"闡述思維分數 ({elaborated_thought.score:.2f}) 未高於初始思維 ({best_initial_thought.score:.2f})。仍使用初始最佳思維。", detail_level=1)
        
        self.logger.thoughtflow("GOT", "顯示 GOT 的思維圖...")
        self.got_system.print_graph() # GOT 內部的 print 語句會在此處輸出
        self.logger.thoughtflow("GOT", "GOT 思維圖顯示完畢。")
        self.logger.section_end("階段 2: GOT - 腦力激盪與結構化解決方案")

        # === 階段 3: 詳細步驟規劃 (LOT) ===
        self.logger.section_start("階段 3: LOT - 詳細步驟規劃")
        lot_initial_input_context = got_output_for_lot
        lot_conceptual_steps = [
            f"步驟 1：分析與解析輸入的高級計劃/想法。識別其核心目標和關鍵組成部分。輸入參考：\n'{lot_initial_input_context[:250]}...'",
            "步驟 2：針對每個識別出的關鍵組成部分，生成具體的、可操作的子任務或行動項目。",
            "步驟 3：將所有生成的子任務/行動項目整合成一個結構清晰、邏輯連貫的詳細執行計劃。"
        ]
        self.logger.thoughtflow("LOT", f"準備用於 LOT 管線的概念步驟: {lot_conceptual_steps}")
        self.logger.discussion("LOT", f"用於 LOT 管線的初始輸入上下文 (來自 GOT，部分內容):\n{lot_initial_input_context[:250]}...")

        lot_final_result = self.lot_system.run_pipeline(
            conceptual_steps=lot_conceptual_steps,
            initial_input=lot_initial_input_context
        ) # LOT 內部的 print 語句會在此處輸出

        if lot_final_result and "未產生任何選項" not in lot_final_result and "未能產生聚合輸出" not in lot_final_result:
            self.logger.answer("LOT", lot_final_result, is_final=False) # 將 LOT 輸出視為一個中間答案/詳細計劃
        else:
            self.logger.error("LOT", f"管線未能產生有效的最終結果。LOT 輸出: {lot_final_result}")
            self.logger.section_end("階段 3: LOT - 詳細步驟規劃")
            self.logger.section_end("協作任務")
            return {"error": "LOT 管線失敗。", "lot_output": lot_final_result, "final_rot_prompt_for_reference": refined_task_prompt}
        self.logger.section_end("階段 3: LOT - 詳細步驟規劃")

        # === 階段 4 (可選): 使用 ROT 的最終提示解決特定實例 ===
        rot_solution_output = None
        if problem_instance_for_rot_final_solve:
            self.logger.section_start("階段 4: ROT - 解決特定實例")
            # 檢查 refined_task_prompt 是否是有效的、經過ROT處理的提示
            # 而非僅僅是初始任務描述或PGRR失敗時的備用llm_taste_prompt
            rot_prompt_to_use_for_solving = None
            if refined_task_prompt != initial_task_description and \
               (llm_taste_prompt_text_from_rot is None or refined_task_prompt != str(llm_taste_prompt_text_from_rot)) and \
               "失敗" not in str(refined_task_prompt).lower() and \
               "LLM 未初始化" not in str(refined_task_prompt):
                rot_prompt_to_use_for_solving = refined_task_prompt # 這是經過 CPM 的最佳提示
                self.logger.thoughtflow("ROT", "將使用 CPM 精煉後的提示來解決問題。")
            elif llm_taste_prompt_text_from_rot and \
                 "失敗" not in str(llm_taste_prompt_text_from_rot).lower() and \
                 "LLM 未初始化" not in str(llm_taste_prompt_text_from_rot):
                rot_prompt_to_use_for_solving = str(llm_taste_prompt_text_from_rot) # 退而求其次使用PGRR的輸出
                self.logger.thoughtflow("ROT", "CPM 提示不可用或與PGRR輸出相同，將使用 PGRR 的 LLM-Taste prompt 來解決問題。")
            else:
                self.logger.discussion("ROT", "ROT 未能生成有效的精煉提示 (無論是CPM還是PGRR)。跳過使用 ROT 解決特定實例的步驟。")


            if rot_prompt_to_use_for_solving:
                self.logger.thoughtflow("ROT", f"嘗試使用精煉提示解決問題實例：'{problem_instance_for_rot_final_solve}'")
                self.logger.discussion("ROT", f"用於解決問題的提示 (部分):\n{rot_prompt_to_use_for_solving[:300]}...")
                solution = self.rot_system.solve_task_with_final_prompt(rot_prompt_to_use_for_solving, problem_instance_for_rot_final_solve)
                if solution and "失敗" not in str(solution).lower() and "LLM 未初始化" not in str(solution):
                    rot_solution_output = solution
                    self.logger.answer("ROT", rot_solution_output, is_final=True) # 將 ROT 解決方案視為一個最終答案
                else:
                    self.logger.error("ROT", f"未能為 '{problem_instance_for_rot_final_solve}' 生成解決方案。ROT輸出: {solution}")
            self.logger.section_end("階段 4: ROT - 解決特定實例")
        else:
            self.logger.info("跳過 ROT 解決特定實例階段，因為未提供問題實例。")


        self.logger.section_end("協作任務")
        return {
            "lot_detailed_plan": lot_final_result,
            "rot_specific_solution": rot_solution_output,
            "final_rot_prompt_for_reference": refined_task_prompt if refined_task_prompt != initial_task_description else "ROT 未產生優於初始描述的提示"
        }


if __name__ == "__main__":
    logger_main = TerminalLogger() # 主腳本的日誌記錄器

    mas_task_description = "為一個目標是探索火星蓋爾撞擊坑中夏普山古代沉積層是否存在有機分子的為期14個火星日的機器人漫遊車任務，制定一個包含樣本採集、初步分析及數據回傳策略的高級別科學操作大綱。"
    mas_rot_demonstrations = [
        (
            "問題：為一個小型無人機設計一個3小時的城市公園環境監測飛行計劃，重點是植被健康和人流密度。",
            "輸出結構：1. 航線規劃（包括起飛點、關鍵航點、備降點）。2. 感測器配置（植被用多光譜相機，人流用熱成像）。3.數據採集協議（每10分鐘記錄一次）。4. 緊急預案（訊號丟失、低電量）。"
        ),
        (
            "問題：草擬一份關於利用AI進行早期森林火災偵測系統的初步研究提案大綱。",
            "輸出結構：1. 引言（問題陳述、重要性）。2. 文獻綜述（現有技術）。3. 研究目標與方法（AI模型選擇、數據來源、預期演算法）。4. 預期成果與影響。5. 時間表與預算初步估計。"
        )
    ]
    problem_instance_for_rot = "為毅力號火星車下一個火星日的任務制定一個簡要指令：分析一塊名為 '堅韌岩' 的目標岩石，使用 WATSON 相機進行近距離成像，並用 SuperCam 進行雷射誘導擊穿光譜分析。"

    if not API_KEY or API_KEY == "YOUR_DEFAULT_GEMINI_API_KEY_HERE": # 再次檢查以防萬一
        logger_main.error("__main__", "無法執行 MAS。請設定有效的 GEMINI_API_KEY 環境變數，或直接在程式碼中更新 API_KEY 的值。")
    else:
        try:
            logger_main.info("正在初始化 MAS 協調器...")
            orchestrator = MASOrchestrator(api_key=API_KEY) # Orchestrator 內部有自己的 logger
            logger_main.info("MAS 協調器初始化完成。開始執行協作任務...")
            
            final_outputs = orchestrator.run_collaborative_task(
                initial_task_description=mas_task_description,
                rot_demonstrations=mas_rot_demonstrations,
                problem_instance_for_rot_final_solve=problem_instance_for_rot
            )
            
            logger_main.section_start("MAS 協同運作最終產出總結")
            if final_outputs:
                if "error" in final_outputs:
                    logger_main.error("MAS_SUMMARY", f"任務執行中發生錯誤: {final_outputs['error']}")

                lot_plan = final_outputs.get("lot_detailed_plan")
                if lot_plan:
                    logger_main.answer("FINAL SUMMARY (LOT - Detailed Plan)", lot_plan, is_final=True)
                else:
                    logger_main.info("LOT 未產生詳細計劃。")

                rot_solution = final_outputs.get("rot_specific_solution")
                if rot_solution:
                     logger_main.answer("FINAL SUMMARY (ROT - Specific Solution)", rot_solution, is_final=True)
                elif problem_instance_for_rot:
                     logger_main.info(f"ROT 未能為 '{problem_instance_for_rot}' 提供特定解決方案。")
                
                rot_prompt_ref = final_outputs.get("final_rot_prompt_for_reference")
                if rot_prompt_ref:
                    logger_main.discussion("MAS_SUMMARY", f"供參考的最終ROT提示 (部分):\n{str(rot_prompt_ref)[:500]}...")

            logger_main.section_end("MAS 協同運作最終產出總結")

        except ValueError as ve:
            logger_main.error("__main__", f"值錯誤 (通常與 API 金鑰或配置相關): {ve}")
        except ImportError as ie:
            logger_main.error("__main__", f"導入錯誤: {ie}。請確保 GOT.py, LOT.py, 和 ROT.py (修改版) 與此腳本在同一目錄下，或者在 Python 路徑中。")
        except Exception as e:
            logger_main.error("__main__", f"MAS 執行過程中發生未預期錯誤: {e}")
            import traceback
            traceback.print_exc()