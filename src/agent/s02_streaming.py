import requests
import json

# ─── 設定 ───────────────────────────────────────────
BASE_URL = "http://192.168.0.182:8080/v1/chat/completions"
MODEL    = "local-model"          # llama.cpp 通常填 "local-model" 即可
SYSTEM   = "你是一個友善的 AI 助手。"
# ────────────────────────────────────────────────────
messages = [{"role": "system", "content": SYSTEM}]

def chat(user_input: str) -> str:
    messages.append({"role": "user", "content": user_input})
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 32768,
        "temperature": 0.7,
        "stream": True,           # ← 開啟串流
    }

    resp = requests.post(
        BASE_URL,
        json=payload,
        timeout=120,
        stream=True,              # ← requests 層也開串流
    )
    resp.raise_for_status()
    resp.encoding = "utf-8" 

    collected = []                # 累積完整回覆，供 messages 記錄用

    print("AI: ", end="", flush=True)

    for line in resp.iter_lines(decode_unicode=True):
        # SSE 格式：每行以 "data: " 開頭，空行與註解行跳過
        if not line or not line.startswith("data: "):
            continue

        data = line[len("data: "):]  # 去掉 "data: " 前綴

        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        # llama.cpp 每個 chunk 的 delta 內容
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content", "")

        if content:
            print(content, end="", flush=True)  # 即時輸出
            collected.append(content)

    print()  # 換行收尾

    full_reply = "".join(collected)
    messages.append({"role": "assistant", "content": full_reply})
    return full_reply


def get_multiline_input(prompt="你: "):
    print(prompt, end="")
    lines = []
    while True:
        line = input()
        if line == "":          # 空行 → 結束
            break
        lines.append(line)
    return "\n".join(lines).strip()


# ─── 主迴圈 ─────────────────────────────────────────
if __name__ == "__main__":
    print("=== llama.cpp 多輪對話 (輸入 'quit' 結束) ===\n")
    while True:
        user = get_multiline_input()
        print(f"sent:{user}")
        if user.lower() in ("quit", "exit", "q"):
            break
        if not user:
            continue
        try:
            answer = chat(user)
            print()  # 回覆與下次輸入之間留一行空白
        except requests.exceptions.ConnectionError:
            print("⚠  連不上 server，請確認 llama-server 已啟動")
        except Exception as e:
            print(f"⚠  錯誤: {e}")
