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
        "stream": False,
    }

    resp = requests.post(BASE_URL, json=payload, timeout=120)
    resp.raise_for_status()

    reply = resp.json()["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    return reply

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
        #user = input("你: ").strip()
        user = get_multiline_input()
        #print(f"你輸入:\n{user}")
        if user.lower() in ("quit", "exit", "q"):
            break
        if not user:
            continue

        try:
            answer = chat(user)
            print(f"\nAI: {answer}\n")
        except requests.exceptions.ConnectionError:
            print("⚠  連不上 server，請確認 llama-server 已啟動")
        except Exception as e:
            print(f"⚠  錯誤: {e}")
