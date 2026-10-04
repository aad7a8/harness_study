import argparse
import requests
import json
import subprocess
import shlex
from prompt_toolkit import PromptSession
from prompt_toolkit.keys import Keys
from prompt_toolkit.key_binding import KeyBindings

BASE_URL = "http://192.168.0.182:8080/v1/chat/completions"
MODEL    = "local-model"
SYSTEM   = "你是一個友善的 AI 助手。"

messages = [{"role": "system", "content": SYSTEM}]

# -bypass 模式：跳過所有 bash 執行確認（由 main() 依命令列參數設定）
BYPASS = False

# ─── Tool 定義 ─────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "執行 bash 指令，回傳 stdout、stderr 與 exit_code。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要執行的 bash 指令",
                    }
                },
                "required": ["command"],
            },
        },
    }
]

# ─── 執行 bash（附安全確認）────────────────────────────
def run_bash(command: str) -> str:
    print(f"\n🖥  準備執行: {command}")
    if BYPASS:
        print("\U0001f6eb  [bypass] 已跳過確認，直接執行")
    else:
        confirm = input("\u2753  是否執行？(y/n): ").strip().lower()
        if confirm != "y":
            return json.dumps({"stdout": "", "stderr": "使用者拒絕執行", "exit_code": -1})

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = {
            "stdout": result.stdout[:4096],   # 截斷避免過長
            "stderr": result.stderr[:4096],
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        output = {"stdout": "", "stderr": "指令執行逾時 (30s)", "exit_code": -1}
    except Exception as e:
        output = {"stdout": "", "stderr": str(e), "exit_code": -1}

    return json.dumps(output, ensure_ascii=False)


# ─── 串流 chat（含 tool-call 處理）────────────────────
def chat(user_input: str) -> str:
    messages.append({"role": "user", "content": user_input})

    while True:  # tool-call 迴圈：AI 可能連續呼叫多次工具
        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 32768,
            "temperature": 0.7,
            "stream": True,
            "tools": TOOLS,
        }

        resp = requests.post(
            BASE_URL,
            json=payload,
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"

        # 累積本輪的 text 與 tool_calls
        content_parts: list[str] = []
        tool_calls: dict[int, dict] = {}  # index → {id, name, arguments}

        print("AI: ", end="", flush=True)

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            delta = chunk.get("choices", [{}])[0].get("delta", {})

            # 一般文字
            text = delta.get("content", "")
            if text:
                print(text, end="", flush=True)
                content_parts.append(text)

            # tool_calls（流式：arguments 會分多段送來）
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc.get("id", ""),
                        "name": "",
                        "arguments": "",
                    }
                if tc.get("id"):
                    tool_calls[idx]["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    tool_calls[idx]["name"] = fn["name"]
                if fn.get("arguments"):
                    tool_calls[idx]["arguments"] += fn["arguments"]

        print()  # 換行
        content_text = "".join(content_parts)

        # ── 沒有 tool_call → 回合結束 ──
        if not tool_calls:
            messages.append({"role": "assistant", "content": content_text})
            return content_text

        # ── 有 tool_call → 執行並回傳 ──
        messages.append({
            "role": "assistant",
            "content": content_text or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls.values()
            ],
        })

        for tc in tool_calls.values():
            try:
                args = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                args = {"command": tc["arguments"]}

            if tc["name"] == "bash":
                result = run_bash(args.get("command", ""))
                print(f"📤  回傳結果長度: {len(result)} chars\n")
            else:
                result = json.dumps({"error": f"未知工具: {tc['name']}"})

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        # 繼續迴圈，讓 AI 根據工具結果繼續回答


# ─── 多行輸入 ──────────────────────────────────────────
def get_multiline_input(prompt="user: "):
    kb = KeyBindings()

    @kb.add(Keys.Enter)
    def _(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    session = PromptSession(key_bindings=kb)
    text = session.prompt(prompt, multiline=True, mouse_support=False)
    return text.strip()


# ─── 主迴圈 ────────────────────────────────────────────
def main() -> None:
    global BYPASS
    parser = argparse.ArgumentParser(description="s03: 附 bash tool 的 chat agent")
    parser.add_argument(
        "-bypass",
        action="store_true",
        help="跳過所有 bash 執行確認（危險：指令將直接執行）",
    )
    args = parser.parse_args()
    BYPASS = args.bypass

    if BYPASS:
        print("\u26a0  [bypass] 模式已開啟：所有 bash 指令將跳過確認直接執行！\n")
    print("  Enter 送出 | Alt+Enter 換行 | /exit 結束\n")
    while True:
        user = get_multiline_input()
        if user.lower() in ("/quit", "/exit", "/q"):
            break
        if user.lower() == "/clear":
            messages.clear()
            messages.append({"role": "system", "content": SYSTEM})
            print("✓ 對話已清空，開始新對話。\n")
            continue
        if not user:
            continue
        try:
            answer = chat(user)
            print()
        except requests.exceptions.ConnectionError:
            print("⚠  連不上 server，請確認 llama-server 已啟動")
        except Exception as e:
            print(f"⚠  錯誤: {e}")


if __name__ == "__main__":
    main()
