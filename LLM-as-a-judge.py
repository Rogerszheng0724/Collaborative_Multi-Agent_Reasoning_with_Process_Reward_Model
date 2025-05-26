from __future__ import annotations # 處理前向參照型別提示
import os
import re
import json
import asyncio
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import typing # 用於型別提示

# --- 設定 ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY 未在環境變數中設定")
genai.configure(api_key=API_KEY)
print(f"使用的 API Key: {API_KEY[:5]}...{API_KEY[-5:]}") # 僅顯示部分 API Key

MODEL_NAME = "gemini-2.5-pro-preview-05-06"
INPUT_EXCEL_PATH = r"D:\data_science\final_project\MAS-PRM\evaluation_v4\RoT-LLM-judge\processed_Rot_eval_output_Part4.xlsx"
OUTPUT_EXCEL_PATH = r"D:\data_science\final_project\MAS-PRM\evaluation_v4\RoT-LLM-judge\LLM-judge-baseline-Rot_evaluation_results_Part5.xlsx" # 修改輸出檔案名以區分

FIXED_CRITERIA = [
    {
        "name": "Clarity", # 清晰度
        "description": (
            "• Uses precise terminology and unambiguous phrasing.\n"
            "• Maintains a clear logical flow with no undefined pronouns or jargon.\n"
            "• Avoids redundancy, filler words, and contradictory assertions."
        )
    },
    {
        "name": "Completeness", # 完整性
        "description": (
            "• Explicitly addresses every sub-question or requirement.\n"
            "• Provides necessary definitions, examples, or data to support each point.\n"
            "• Considers edge cases or counterarguments where relevant, without omitting any critical aspect."
        )
    },
    {
        "name": "Relevance", # 相關性
        "description": (
            "• Stays focused on the core prompt with no off-topic digressions.\n"
            "• Ties every claim directly to the question and backs it with reasoning or citations.\n"
            "• Omits generic filler and unsupported opinions."
        )
    }
]

# --- 用於並行控制的信號量 ---
api_semaphore = asyncio.Semaphore(1)   # Gemini API 最大併發請求數 (可依據您的 API 限制調整)
row_semaphore = asyncio.Semaphore(1)  # 同時處理列的最大數量 (可依據您的機器性能調整)

# --- 共用的模型實例 ---
model = genai.GenerativeModel(MODEL_NAME)

async def call_gemini_api(prompt: str) -> dict:
    """使用 api_semaphore 限制 API 並行性來呼叫 Gemini。"""
    async with api_semaphore:
        try:
            # print(f"\n--- Sending Prompt to Gemini ---\n{prompt}\n--- End of Prompt ---\n") # 用於調試，可取消註解
            resp = await model.generate_content_async(
                contents=[prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5, # 稍微降低 temperature，期望更一致的 JSON 格式
                    # response_mime_type="application/json" # 如果模型支援，強制JSON輸出
                )
            )
            # 嘗試從回應中提取 JSON
            text = getattr(resp, "text", "").strip()
            # print(f"\n--- Raw Gemini Response ---\n{text}\n--- End of Raw Response ---\n") # 用於調試

            # 更穩健地提取 JSON，處理前後可能存在的非 JSON 文本
            match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
            if not match:
                match = re.search(r"(\{.*\})", text, re.DOTALL) # 備用：尋找第一個大括號包圍的內容

            if match:
                json_str = match.group(1)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as je:
                    print(f"[警告] API 回應 JSON 解析失敗: {je}")
                    print(f"[警告] 問題 JSON 字串: {json_str}")
                    return {} # 返回空字典，讓後續處理有預設值
            else:
                print(f"[警告] API 回應中未找到有效的 JSON 結構。回應文本: {text[:200]}...") # 顯示部分回應
                return {}
        except Exception as e:
            print(f"[警告] API 呼叫失敗: {e}")
            # 可以考慮加入重試機制或更詳細的錯誤記錄
    return {}

def python_check_correctness(q: str, gt: str, gen: str) -> int:
    """備用的正確性檢查。"""
    if gt.isdigit():
        opts = dict(re.findall(r"(\d+)\.\s*(.+?)(?=(?:\d+\.)|$)", q, re.S))
        low = gen.lower()
        if gt in low or opts.get(gt, "").lower() in low:
            return 1
        return 0
    return 1 if gt.strip().lower() in gen.lower() else 0

async def evaluate_all(question: str, ground_truth: str, generated: str) -> dict:
    """
    Evaluates the generated answer for correctness and multiple criteria.
    Requires the model to first state shortcomings and then deduct points from 10.
    """
    crit_text = "\n".join(
        f"- {c['name']}: {c['description']}" # This will be in English as names and descriptions in FIXED_CRITERIA are English
        for c in FIXED_CRITERIA
    )
    # *** Prompt has been translated to English ***
    prompt = (
        "You are an expert judge. Your task is to evaluate the \"Generated Answer\" based on the \"Question\" and the \"Reference Answer\".\n"
        "Please perform the following two evaluations, and your output MUST be strictly a single JSON object, without any additional text before or after the JSON object.\n\n"
        "1) correctness_score: An integer (0 or 1). This score indicates whether the \"Generated Answer\" contains the essential information from the \"Reference Answer\".\n"
        "   - 1: The \"Generated Answer\" correctly and accurately includes the core information of the \"Reference Answer\".\n"
        "   - 0: The \"Generated Answer\" fails to capture the core information of the \"Reference Answer\" or its content is incorrect.\n\n"
        "2) evaluation_results: An array of objects, each corresponding to an evaluation criterion. For each criterion:\n"
        "   a) First, provide a \"shortcomings\" string. This string should detail any specific defects, weaknesses, or areas for improvement in the \"Generated Answer\" regarding that particular criterion. Be specific and, if possible, provide examples from the \"Generated Answer\". If there are no shortcomings for a criterion, explicitly state \"No shortcomings found\" or a similar neutral statement.\n"
        "   b) Second, provide a \"score\" as an integer between 0 and 10. This score is calculated by starting with 10 points and then deducting points for each shortcoming identified in step (a). The more severe or numerous the shortcomings, the lower the score. A score of 10 indicates the answer is excellent in that criterion, while a score of 0 indicates severe issues.\n\n"
        f"Question:\n{question}\n\n"
        f"Reference Answer:\n{ground_truth}\n\n"
        f"Generated Answer:\n{generated}\n\n"
        "Evaluation Criteria:\n" + crit_text + "\n\n"
        "Important Note: Ensure your entire output is strictly a single JSON object. Do not include any text before or after the JSON object.\n"
        "Expected JSON format:\n"
        "{\n"
        '  "correctness_score": 0 or 1,\n'
        '  "evaluation_results": [\n'
        '    {"criterion_name": "Clarity", "shortcomings": "Detailed explanation of clarity issues, or \'No shortcomings found\'.", "score": integer_between_0_and_10},\n'
        '    {"criterion_name": "Completeness", "shortcomings": "Detailed explanation of completeness issues, or \'No shortcomings found\'.", "score": integer_between_0_and_10},\n'
        '    {"criterion_name": "Relevance", "shortcomings": "Detailed explanation of relevance issues, or \'No shortcomings found\'.", "score": integer_between_0_and_10}\n'
        # If FIXED_CRITERIA changes, more criteria should be added here accordingly
        "  ]\n"
        "}"
    )

    res = await call_gemini_api(prompt)
    corr = res.get("correctness_score")

    # *** 已修改正確性得分的處理和警告 ***
    if corr is None: # 檢查是否為 None，因為 0 是有效值
        print(f"[警告] 模型未提供 correctness_score。將使用 python_check_correctness 進行備用檢查。問題: {question[:50]}...")
        corr = python_check_correctness(question, ground_truth, generated)
    elif corr not in (0, 1):
        print(f"[警告] 模型提供的 correctness_score ({corr}) 無效。將使用 python_check_correctness。問題: {question[:50]}...")
        corr = python_check_correctness(question, ground_truth, generated)


    evals = res.get("evaluation_results", [])
    out = {"correctness": corr}

    # *** 已修改評估結果的解析邏輯，以包含缺點並提供預設值 ***
    # 為所有預期標準初始化預設值
    for crit_data in FIXED_CRITERIA:
        crit_name = crit_data["name"]
        out[crit_name] = 0  # 預設分數
        out[f"{crit_name}_shortcomings"] = "模型未返回此標準的評估數據。" # 預設缺點描述

    if isinstance(evals, list): # 確保 evals 是列表
        for e_dict in evals:
            if isinstance(e_dict, dict): # 確保列表中的元素是字典
                crit_name = e_dict.get("criterion_name")
                if crit_name and any(c["name"] == crit_name for c in FIXED_CRITERIA): # 檢查是否為預期標準
                    score = e_dict.get("score")
                    shortcomings = e_dict.get("shortcomings", "模型未提供此標準的缺點描述。")

                    if isinstance(score, (int, float)) and 0 <= score <= 10:
                        out[crit_name] = int(round(score)) # 確保分數是整數
                    else:
                        print(f"[警告] 模型為 {crit_name} 提供了無效或缺失的分數。收到: {score}。將預設為 0。")
                        out[crit_name] = 0 # 如果分數無效，則預設為 0

                    out[f"{crit_name}_shortcomings"] = str(shortcomings) # 確保缺點是字串
                else:
                    print(f"[警告] 模型返回了一個非預期的評估標準: {crit_name}")
            else:
                print(f"[警告] evaluation_results 中的項目格式不正確: {e_dict}")
    else:
        print(f"[警告] 模型返回的 evaluation_results 不是一個列表: {evals}")
    return out

async def process_row(idx: int, row: typing.Any, total: int) -> dict: # 使用 typing.Any 避免 NameError
    """用 row_semaphore 限制最大同時處理列的數量。"""
    async with row_semaphore:
        # 從 row 物件中安全地獲取屬性，如果屬性不存在則提供預設值
        q   = str(getattr(row, "combined_input_question", "N/A"))
        gt  = str(getattr(row, "ground_truth_answer", "N/A"))
        gen = str(getattr(row, "overall_best_generated_answer_across_cycles", "N/A"))

        print(f"[{idx}/{total}] 開始處理第 {idx} 列")
        scores_and_shortcomings = await evaluate_all(q, gt, gen)
        
        # *** 已修改輸出訊息以包含缺點摘要 ***
        print(f"[{idx}/{total}] 完成處理第 {idx} 列:")
        print(f"  正確性 (Correctness): {scores_and_shortcomings.get('correctness', '-')}")
        for crit_data in FIXED_CRITERIA:
            crit_name = crit_data["name"]
            score = scores_and_shortcomings.get(crit_name, '-')
            shortcomings = scores_and_shortcomings.get(f"{crit_name}_shortcomings", 'N/A')
            # 為了終端機輸出簡潔，只顯示缺點的前100個字元
            shortcomings_summary = shortcomings[:100] + ('...' if len(shortcomings) > 100 else '')
            print(f"  {crit_name}: 分數={score}, 缺點='{shortcomings_summary}'")

        # 準備要寫入 DataFrame 的數據
        # scores_and_shortcomings 字典已包含所有需要的鍵
        # 例如：'Clarity', 'Clarity_shortcomings', 'Completeness', 'Completeness_shortcomings'
        return_data = {
            "combined_input_question": q,
            "ground_truth_answer":      gt,
            "overall_best_generated_answer_across_cycles": gen,
        }
        return_data.update(scores_and_shortcomings) # 將所有分數和缺點合併進來
        return return_data

async def main():
    # 1. 讀取整份 Excel
    try:
        df = pd.read_excel(INPUT_EXCEL_PATH, engine="openpyxl")
    except FileNotFoundError:
        print(f"[錯誤] 輸入的 Excel 檔案未找到: {INPUT_EXCEL_PATH}")
        return
    except Exception as e:
        print(f"[錯誤] 讀取 Excel 檔案時發生錯誤: {e}")
        return
        
    required_cols = [
        "combined_input_question",
        "ground_truth_answer",
        "overall_best_generated_answer_across_cycles"
    ]
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(f"[錯誤] 輸入的 Excel 檔案缺少必要的欄位: {', '.join(missing_cols)}")
        print(f"必需欄位為: {required_cols}")
        return

    # 2. 詢問使用者要從第幾筆到第幾筆（以 1 為起始）
    total_rows = len(df)
    if total_rows == 0:
        print("[錯誤] Excel 檔案中沒有數據。")
        return

    start_row_num = None
    while start_row_num is None:
        try:
            val_str = input(f"資料集共有 {total_rows} 筆，請輸入起始評分筆數 (1–{total_rows}): ")
            if not val_str: continue # 允許用戶直接按 Enter 使用預設值（如果有的話）或重新提示
            v = int(val_str)
            if 1 <= v <= total_rows:
                start_row_num = v
            else:
                print(f"請輸入 1 到 {total_rows} 之間的數字。")
        except ValueError:
            print("請輸入有效的整數。")

    end_row_num = None
    while end_row_num is None:
        try:
            val_str = input(f"請輸入結束評分筆數 ({start_row_num}–{total_rows}): ")
            if not val_str: continue
            v = int(val_str)
            if start_row_num <= v <= total_rows:
                end_row_num = v
            else:
                print(f"請輸入 {start_row_num} 到 {total_rows} 之間的數字。")
        except ValueError:
            print("請輸入有效的整數。")

    # 3. 轉換為 DataFrame 的 iloc 範圍（因為 user 以 1 為起始）
    # iloc 的結束索引是不包含的，所以 end_row_num 不需要減 1
    df_subset = df.iloc[start_row_num-1 : end_row_num]

    # 4. 重新計算 subset 的總長度
    current_processing_total = len(df_subset)
    if current_processing_total == 0:
        print("[資訊] 根據您選擇的範圍，沒有數據需要處理。")
        return

    print(f"\n將處理從第 {start_row_num} 筆到第 {end_row_num} 筆的數據，共 {current_processing_total} 筆。\n")

    # 5. 產生要執行的任務
    # 使用 itertuples(index=True) 以便能用 df_subset.index[i] 取得原始索引 (如果需要)
    # 但此處的 i 是 enumerate 的索引，從 start_row_num 開始
    tasks = [
        process_row(original_idx, row_data, current_processing_total)
        for original_idx, row_data in enumerate(df_subset.itertuples(index=False, name="Row"), start=start_row_num)
        # 調整 process_row 的第一個參數為原始的行號 (start_row_num + i)
        # 或者，如果 process_row 的 idx 只是用於顯示進度 (e.g., 1/N, 2/N)，則可以用 enumerate(..., start=1)
        # 此處的 original_idx 將會是 start_row_num, start_row_num+1, ...
        # process_row 的 idx 參數現在接收的是原始Excel中的行號（基於用戶輸入的start_row_num）
    ]
    
    # 為了讓 process_row 的 idx 參數是 1 到 current_processing_total 的計數器
    # 並且仍然傳遞正確的 row_data
    tasks = []
    for i, (df_idx, row_data_tuple) in enumerate(df_subset.iterrows(), start=1):
        # iterrows() 返回 (index, Series)
        # 我們需要將 Series 轉換為類似 itertuples() 的 Row 物件，或者直接傳遞 Series
        # 為了與現有的 process_row(idx, row, total) 簽名兼容，我們需要一個 'row' 物件
        # 最簡單的方式是繼續使用 itertuples，但調整傳給 process_row 的 idx
        pass # 重新思考 tasks 的建立

    # 修正 tasks 的建立，讓 process_row 的 idx 是 1 到 N 的計數器
    # 並且 row 參數是正確的 Row 物件
    tasks = [
        process_row(i, row_obj, current_processing_total)
        for i, row_obj in enumerate(df_subset.itertuples(index=False, name="Row"), start=1)
    ]


    # 6. 執行並輸出
    results = await asyncio.gather(*tasks)

    # *** 已修改輸出欄位的定義以包含缺點 ***
    # 定義輸出 Excel 檔案的欄位順序
    base_cols = [
        "combined_input_question",
        "ground_truth_answer",
        "overall_best_generated_answer_across_cycles",
        "correctness"
    ]
    criteria_cols_ordered = []
    for c_config in FIXED_CRITERIA: # 迭代 FIXED_CRITERIA 以保持順序
        crit_name = c_config["name"]
        criteria_cols_ordered.append(crit_name) # 例如 "Clarity"
        criteria_cols_ordered.append(f"{crit_name}_shortcomings") # 例如 "Clarity_shortcomings"

    # 確保所有預期欄位都存在於 results 中的字典裡，如果沒有則用 pd.NA 填充
    # DataFrame 構造函數會自動處理缺失的鍵（如果 results 是字典列表）
    
    # 最終欄位順序
    final_cols_ordered = base_cols + criteria_cols_ordered

    out_df = pd.DataFrame(results)
    
    # 重新索引欄位以確保順序，並處理可能因 API 問題導致的缺失欄位
    # (如果 results 中的某些字典缺少了某些鍵，reindex 會用 NaN 填充)
    out_df = out_df.reindex(columns=final_cols_ordered)

    try:
        out_df.to_excel(OUTPUT_EXCEL_PATH, index=False, engine="openpyxl")
        print(f"\n已將第 {start_row_num} 筆到第 {end_row_num} 筆的評分結果（包含缺點）存到：{OUTPUT_EXCEL_PATH}")
    except Exception as e:
        print(f"[錯誤] 儲存 Excel 檔案時發生錯誤: {e}")
        print(f"數據可能仍在記憶體中，DataFrame 的前幾行為:\n{out_df.head()}")


if __name__ == "__main__":
    # 在 Windows 上設定 asyncio 事件循環策略 (如果遇到 ProactorBasePipeTransport 相關錯誤)
    # if os.name == 'nt':
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
