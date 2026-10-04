import argparse
import requests
import json
import subprocess
from prompt_toolkit import PromptSession
from prompt_toolkit.keys import Keys
from prompt_toolkit.key_binding import KeyBindings

SERVER     = "http://192.168.0.182:8080"
MESSAGES_URL = f"{SERVER}/v1/messages"
COUNT_URL  = f"{SERVER}/v1/messages/count_tokens"
HEALTH_URL = f"{SERVER}/health"
MODEL      = "local-model"
SYSTEM     = "你是一個友善的 AI 助手。"

system_prompt = SYSTEM
messages: list[dict] = []

BYPASS = False

MAX_CONTEXT = None
LAST_USAGE  = None

def fetch_max_context() -> None:
    global MAX_CONTEXT
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        r.raise_for_status()
        MAX_CONTEXT = r.json().get("max_context")
    except Exception as e:
        print(f"⚠  無法取得 max_context: {e}")

def current_context() -> int | None:
    """本輪結束時的 context 佔用 = input + cache_read + output。
    ⚠ 有 KV cache 時 input_tokens 只剩 uncached 部分，
    漏加 cache_read_input_tokens 會少算一個數量級。"""
    if not LAST_USAGE:
        return None
    return (LAST_USAGE.get("input_tokens", 0)
            + LAST_USAGE.get("cache_read_input_tokens", 0)
            + LAST_USAGE.get("output_tokens", 0))

def fmt_tokens(n: int) -> str:
    return f"{n/1000:.1f}K" if n >= 1000 else str(n)

def context_status_line() -> str:
    """📊 Context ▓▓░░… 3.2K / 256K (1.2%)"""
    used = current_context()
    if used is None or not MAX_CONTEXT:
        return "📊 Context: —"
    pct = used / MAX_CONTEXT * 100
    width = 20
    filled = min(width, round(pct / 100 * width))
    bar = "▓" * filled + "░" * (width - filled)
    return (f"\033[90m📊 Context: {bar} {fmt_tokens(used)} / "
            f"{fmt_tokens(MAX_CONTEXT)} ({pct:.1f}%)\033[0m")

def count_context_exact() -> None:
    """payload 必須與 /v1/messages 完全一致（system+messages+tools 全帶）；
    漏帶 tools 會少算 tool schema 的 token。"""
    try:
        r = requests.post(COUNT_URL, json=build_payload(max_tokens=1), timeout=10)
        r.raise_for_status()
        n = r.json().get("input_tokens")
        if n is None:
            print("⚠  count_tokens 未回傳 input_tokens")
            return
        if MAX_CONTEXT:
            pct = n / MAX_CONTEXT * 100
            print(f"🎯 count_tokens 精算: {n:,} tokens / {MAX_CONTEXT:,} "
                  f"({pct:.2f}%)　← 下一輪實際會讀取的 prompt 大小")
        else:
            print(f"🎯 count_tokens 精算: {n:,} tokens")
    except Exception as e:
        print(f"⚠  count_tokens 失敗: {e}")

TOOLS = [
    {
        "name": "bash",
        "description": "執行 bash 指令，回傳 stdout、stderr 與 exit_code。",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要執行的 bash 指令",
                }
            },
            "required": ["command"],
        },
    }
]

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
        def truncate(s: str, limit: int = 5000) -> str:
            if len(s) > limit:
                return s[:limit] + f"\n…(輸出已截斷，總長度 {len(s)} chars)"
            return s

        output = {
            "stdout": truncate(result.stdout),
            "stderr": truncate(result.stderr),
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        output = {"stdout": "", "stderr": "指令執行逾時 (30s)", "exit_code": -1}
    except Exception as e:
        output = {"stdout": "", "stderr": str(e), "exit_code": -1}

    return json.dumps(output, ensure_ascii=False)

def build_payload(max_tokens: int = 32768) -> dict:
    return {
        "model": MODEL,
        "system": system_prompt,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "tools": TOOLS,
    }

def chat(user_input: str) -> str:
    global LAST_USAGE
    messages.append({"role": "user", "content": user_input})

    while True:
        resp = requests.post(
            MESSAGES_URL,
            json=build_payload(),
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"

        blocks: dict[int, dict] = {}
        in_thinking = False

        print("AI: ", end="", flush=True)

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue

            etype = ev.get("type")

            if etype == "message_delta":
                usage = ev.get("usage")
                if usage:
                    LAST_USAGE = usage

            elif etype == "content_block_start":
                block = ev.get("content_block", {})
                blocks[ev.get("index", 0)] = {
                    "type": block.get("type", ""),
                    "text": "",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                }

            elif etype == "content_block_delta":
                delta = ev.get("delta", {})
                dtype = delta.get("type")
                blk = blocks.setdefault(ev.get("index", 0), {"type": "", "text": ""})

                if dtype == "thinking_delta":
                    if not in_thinking:
                        in_thinking = True
                        print("\033[90m💭 [思考]\033[0m\n\033[90m", end="", flush=True)
                    print(delta.get("thinking", ""), end="", flush=True)

                elif dtype == "text_delta":
                    if in_thinking:
                        in_thinking = False
                        print("\033[0m\n", end="", flush=True)
                    text = delta.get("text", "")
                    print(text, end="", flush=True)
                    blk["text"] += text

                elif dtype == "input_json_delta":
                    blk["text"] += delta.get("partial_json", "")

        if in_thinking:
            print("\033[0m", end="", flush=True)
        print()

        content_blocks = []
        tool_uses = []
        for idx in sorted(blocks):
            blk = blocks[idx]
            if blk["type"] == "text" and blk["text"]:
                content_blocks.append({"type": "text", "text": blk["text"]})
            elif blk["type"] == "tool_use":
                content_blocks.append({
                    "type": "tool_use",
                    "id": blk["id"],
                    "name": blk["name"],
                    "input": safe_json(blk["text"]),
                })
                tool_uses.append((blk["id"], blk["name"], blk["text"]))

        if not tool_uses:
            messages.append({"role": "assistant", "content": content_blocks})
            print(context_status_line())
            return "".join(b["text"] for b in content_blocks if b["type"] == "text")

        messages.append({"role": "assistant", "content": content_blocks})

        tool_results = []
        for tc_id, name, args_str in tool_uses:
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"command": args_str}

            if name == "bash":
                result = run_bash(args.get("command", ""))
                print(f"📤  回傳結果長度: {len(result)} chars\n")
            else:
                result = json.dumps({"error": f"未知工具: {name}"})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc_id,
                "content": result,
            })

        # Anthropic 規定：同輪所有 tool_result 必須放同一則 user 訊息，分開 append 會 400
        messages.append({"role": "user", "content": tool_results})

        print(context_status_line() + "\n")

def safe_json(s: str) -> dict:
    try:
        return json.loads(s) if s else {}
    except json.JSONDecodeError:
        return {"command": s}

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

def main() -> None:
    global BYPASS, system_prompt
    parser = argparse.ArgumentParser(description="s03.1: 附 bash tool 的 chat agent（Anthropic 原生格式 + context status line）")
    parser.add_argument(
        "-bypass",
        action="store_true",
        help="跳過所有 bash 執行確認（危險：指令將直接執行）",
    )
    args = parser.parse_args()
    BYPASS = args.bypass

    fetch_max_context()
    if MAX_CONTEXT:
        print(f"ℹ  max_context = {MAX_CONTEXT:,} tokens")

    if BYPASS:
        print("\u26a0  [bypass] 模式已開啟：所有 bash 指令將跳過確認直接執行！\n")
    print("  Enter 送出 | Alt+Enter 換行 | /context 精算 context | /exit 結束\n")
    while True:
        user = get_multiline_input()
        if user.lower() in ("/quit", "/exit", "/q"):
            break
        if user.lower() == "/clear":
            messages.clear()
            print("✓ 對話已清空，開始新對話。\n")
            continue
        if user.lower() == "/context":
            count_context_exact()
            continue
        if not user:
            continue
        try:
            answer = chat(user)
            print()
        except requests.exceptions.ConnectionError:
            print("⚠  連不上 server，請確認 Strata server 已啟動")
        except Exception as e:
            print(f"⚠  錯誤: {e}")

if __name__ == "__main__":
    main()
