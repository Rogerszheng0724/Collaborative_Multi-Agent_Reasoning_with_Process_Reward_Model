from dotenv import load_dotenv
import os
import google.generativeai as genai

# 讀取 .env 檔案裡的環境變數到 os.environ
load_dotenv()

# 取用剛剛從 .env 載入的 API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("找不到 GOOGLE_API_KEY，請確認 .env 檔案已正確放置並設定")

genai.configure(api_key=api_key)

# 現在就可以安全地列出模型或呼叫其他 genai API
for model in genai.list_models():
    print(model.name)