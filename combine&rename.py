import pandas as pd

# 讀取原始 Excel 檔案
file_path = "./evaluation_v3/baseline-Rot_eval_output.xlsx"
df = pd.read_excel(file_path)


# 建立合併後的 combined_input_question 欄位，套用指定格式
df["combined_input_question"] = (
    "Instruction: " + df["input_instruction"].fillna('') + "\n" +
    "Context: " + df["input_context"].fillna('')
)

# 將 combined_input_question 插入在 input_context 後面
context_idx = df.columns.get_loc("input_context")
cols = df.columns.tolist()
# 先移除，再插入
cols.remove("combined_input_question")
cols.insert(context_idx + 1, "combined_input_question")
df = df[cols]

# 將 generated_answer_final 欄位重新命名
df = df.rename(columns={
    "generated_answer_final": "overall_best_generated_answer_across_cycles"
})

# 儲存處理後的結果為新檔案
output_path = "./evaluation_v3/processed_Rot_eval_output.xlsx"
df.to_excel(output_path, index=False)

print(f"處理完成，已儲存至 {output_path}")
