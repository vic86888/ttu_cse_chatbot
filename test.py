print('hello world')

# chat_stream.py
import sys
import json
import requests
from datetime import datetime

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "ttu-news-bot"  # 你要用的模型
# MODEL = "llama3.1:8b"  # 你要用的模型

def check_server():
    try:
        r = requests.get("http://127.0.0.1:11434/api/version", timeout=2)
        r.raise_for_status()
        return True
    except Exception as e:
        print("❌ 無法連到 Ollama 伺服器，請先執行：`ollama serve`")
        return False

def stream_chat(messages, temperature=0.7):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            # 視需要設定 KV cache、批次等參數：
            # "num_ctx": 32768,  # 若想提高上下文長度（取決於模型支援）
            # "num_batch": 512,
        }
    }
    with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
        r.raise_for_status()
        assistant_text = []
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                # 偶爾可能有殘缺行，直接略過
                continue

            if "message" in chunk and "content" in chunk["message"]:
                token = chunk["message"]["content"]
                assistant_text.append(token)
                # 即時輸出，不換行
                print(token, end="", flush=True)

            # 串流結束訊號
            if chunk.get("done", False):
                print()  # 換行
                break

        return "".join(assistant_text)

def main():
    if not check_server():
        sys.exit(1)

    print(f"💬 與 {MODEL} 對話（輸入 exit 離開）")
    # 可選 system prompt
    system_prompt = "你是一位懂台灣在地語境、使用繁體中文回覆的助理。"
    history = [{"role": "system", "content": system_prompt}]

    try:
        while True:
            user = input("你：")
            if user.strip().lower() == "exit":
                break

            history.append({"role": "user", "content": user})
            print("模型：", end="", flush=True)
            reply = stream_chat(history)
            history.append({"role": "assistant", "content": reply})
    except KeyboardInterrupt:
        print("\n👋 已中止")

if __name__ == "__main__":
    main()