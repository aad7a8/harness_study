import argparse
import requests
import json
import subprocess
import os
from itertools import islice
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
LAST_STOP_REASON: str | None = None

MAX_TOOL_ROUNDS = 20

READ_MAX_LINES = 2000
READ_MAX_BYTES = 256 * 1024
READ_PAGE_CHARS = 30_000
READ_LINE_CHARS = 2_000

READ_STATE: dict[str, int] = {}

LAST_READS: dict[tuple[str, int, int], int] = {}


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
    """
    if not LAST_USAGE:
        return None
    return (LAST_USAGE.get("input_tokens", 0)
            + LAST_USAGE.get("cache_read_input_tokens", 0)
            + LAST_USAGE.get("output_tokens", 0))

def fmt_tokens(n: int) -> str:
    return f"{n/1000:.1f}K" if n >= 1000 else str(n)

def context_status_line() -> str:
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
    """
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
    },
    {
        "name": "read_file",
        "description": ("讀取文字檔，回傳帶行號的內容（cat -n 格式）。"
                        "大檔用 offset/limit 分頁；同區間讀過且檔案沒改過不會重送內容。"
                        "要 edit_file / write_file 覆蓋既有的檔，必須先讀過。"),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":   {"type": "string", "description": "檔案路徑"},
                "offset": {"type": "integer", "description": "起始行號（1-based），預設 1"},
                "limit":  {"type": "integer",
                           "description": f"最多讀取行數，預設 {READ_MAX_LINES}"},
                "force":  {"type": "boolean",
                           "description": "即使檔案未變也更強制重送內容"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": ("建立或覆蓋文字檔（必要時建立上層目錄）。"
                        "覆蓋既有的檔之前，必須先 read_file 讀過該檔。"),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "檔案路徑"},
                "content": {"type": "string", "description": "要寫入的完整檔案內容"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": ("exact-match 字串替換：old_string 必須與檔案內容完全一致，"
                        "且在檔案中唯一（多处符合時加長上下文，或明確傳 replace_all=true）。"
                        "必須先 read_file；讀過之後檔案又被改過會要求重讀。"),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":        {"type": "string", "description": "檔案路徑"},
                "old_string":  {"type": "string", "description": "要被替換的原始字串"},
                "new_string":  {"type": "string", "description": "替換成的新字串"},
                "replace_all": {"type": "boolean",
                                "description": "取代所有符合處（預設 false，要求唯一）"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
]

def truncate(s: str, limit: int = 5000) -> str:
    if len(s) > limit:
        return s[:limit] + f"\n…(輸出已截斷，總長度 {len(s)} chars)"
    return s

def run_bash(command: str) -> dict:
    print(f"\n🖥  準備執行: {command}")
    if BYPASS:
        print("\U0001f6eb  [bypass] 已跳過確認，直接執行")
    else:
        confirm = input("\u2753  是否執行？(y/n): ").strip().lower()
        if confirm != "y":
            return {"stdout": "", "stderr": "使用者拒絕執行", "exit_code": -1}

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = {
            "stdout": truncate(result.stdout),
            "stderr": truncate(result.stderr),
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        output = {"stdout": "", "stderr": "指令執行逾時 (30s)", "exit_code": -1}
    except Exception as e:
        output = {"stdout": "", "stderr": str(e), "exit_code": -1}

    return output


# ─── 檔案工具：read / write / edit（Claude Code 語意）────────
#
# 三條從主流 harness 抄來的鐵律：
#   1. read 回傳帶行號（cat -n），offset/limit 分頁；
#   2. edit 用 exact-match 字串替換，old_string 要求唯一，
#      比讓模型給行號區間的錯誤率低得多；
#   3. read-before-edit：沒讀過的檔不給改；讀過之後 mtime 變過
#      （user 或 linter 動的）一律強制重讀 —— 防盲覆蓋最重要的柵欄。

def _require_read(path: str) -> dict | None:
    """read-before-edit 閘門：放行回 None，擋下回錯誤 dict。"""
    if path not in READ_STATE:
        return {"error": f"未先讀取過檔案，拒絕操作；請先 read_file：{path}"}
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError as e:
        return {"error": f"無法存取檔案 {path}: {e}"}
    if mtime != READ_STATE[path]:
        return {"error": f"檔案自上次讀取後已被修改（user 或外部工具動的），"
                         f"請重新 read_file 後再操作：{path}"}
    return None


def _confirm(desc: str) -> bool:
    """破壞性檔案操作的確認，與 bash 確認共用 BYPASS 同一個開關。"""
    if BYPASS:
        print(f"\U0001f6eb  [bypass] 已跳過確認，直接執行")
        return True
    return input(f"\u2753  {desc}，是否執行？(y/n): ").strip().lower() == "y"


def _clip_line(line: str) -> str:
    if len(line) <= READ_LINE_CHARS:
        return line
    return line[:READ_LINE_CHARS] + f"…（此行超過 {READ_LINE_CHARS} 字元已截斷）"


def _preview(s: str, limit: int = 120) -> str:
    """字串單行預覽（换行顯示為 \n），用於告知將讀/寫/改什麼。"""
    s = s.replace("\n", "\\n")
    return s if len(s) <= limit else s[:limit] + "…"


def run_read_file(path: str, offset: int = 1, limit: int | None = None,
                  force: bool = False) -> dict:
    full_read = limit is None
    limit = limit or READ_MAX_LINES

    rng = "全檔" if full_read else f"第 {offset} 行起、最多 {limit} 行"
    print(f"\n📖  準備讀取: {path}　（{rng}"
          + ("，強制重送" if force else "") + "）")

    if not os.path.isfile(path):
        return {"error": f"檔案不存在或不是普通檔案：{path}"}
    st = os.stat(path)

    if not force and LAST_READS.get((path, offset, limit)) == st.st_mtime_ns:
        return {"ok": True,
                "content": f"檔案自上次讀取後未變（同 offset/limit），不重送內容；"
                           f"真需要再看一次請傳 force=true：{path}"}

    if full_read and st.st_size > READ_MAX_BYTES:
        return {"error": f"檔案太大（{st.st_size:,} bytes > {READ_MAX_BYTES:,} 上限），"
                         f"請用 offset/limit 分頁讀取：{path}"}

    with open(path, "rb") as f:
        if b"\x00" in f.read(4096):
            return {"error": f"二元檔（偵測到 NUL byte），不支援讀取：{path}"}

    try:
        with open(path, "r", encoding="utf-8") as f:
            # 多取一行當觸偵測「還有沒有更多」，不靠全檔行數
            window = list(islice(f, offset - 1, offset - 1 + limit + 1))
    except UnicodeDecodeError:
        return {"error": f"不是 UTF-8 文字檔，無法讀取：{path}"}

    if not window:
        return {"error": f"offset={offset} 已超過檔案總行數：{path}"}

    has_more = len(window) > limit
    shown = window[:limit]

    # 預算是行邊界切的：截在半行會讓模型把半截內容抄進 old_string。
    kept, total = [], 0
    for ln in shown:
        cost = min(len(ln), READ_LINE_CHARS) + 8
        if kept and total + cost > READ_PAGE_CHARS:
            break
        kept.append(ln)
        total += cost
    cut = len(kept) < len(shown)
    shown = kept

    header = f"{path}（第 {offset}-{offset + len(shown) - 1} 行"
    if has_more or cut:
        header += f"，後續還有，可用 offset={offset + len(shown)} 續讀"
    if cut:
        header += f"；本頁已達 {READ_PAGE_CHARS:,} 字元預算"
    header += "）"
    numbered = "\n".join(f"{offset + i:6d}\t{_clip_line(ln.rstrip(chr(10)))}"
                          for i, ln in enumerate(shown))

    READ_STATE[path] = st.st_mtime_ns
    LAST_READS[(path, offset, limit)] = st.st_mtime_ns
    return {"ok": True, "content": header + "\n" + numbered}


def run_write_file(path: str, content: str) -> dict:
    exists = os.path.isfile(path)
    print(f"\n✍️  準備寫入: {path}　（{len(content)} chars，"
          + ("覆蓋既有檔案" if exists else "新檔") + "）")
    if exists:
        err = _require_read(path)
        if err:
            return err
    elif os.path.exists(path):
        return {"error": f"不是普通檔案，拒絕寫入：{path}"}

    if not _confirm(f"寫入檔案 {path}（{len(content)} chars"
                    + ("，將覆蓋原本內容" if exists else "，新檔") + "）"):
        return {"error": "使用者拒絕執行"}
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(content)
    except OSError as e:
        return {"error": f"寫入失敗: {e}"}

    READ_STATE[path] = os.stat(path).st_mtime_ns   # 剛寫的檔當「已讀」，不必重讀
    return {"ok": True,
            "content": f"已{'覆蓋' if exists else '建立'} {path}（{len(content)} chars）"}


def run_edit_file(path: str, old_string: str, new_string: str,
                  replace_all: bool = False) -> dict:
    print(f"\n✏️  準備編輯: {path}　（{'全部符合處' if replace_all else '唯一符合'}）")
    print(f"      - {_preview(old_string)}")
    print(f"      + {_preview(new_string)}")
    err = _require_read(path)
    if err:
        return err
    if old_string == new_string:
        return {"error": "old_string 與 new_string 相同，不需替換"}

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError) as e:
        return {"error": f"讀取失敗: {e}"}

    n = text.count(old_string)
    if n == 0:
        return {"error": "找不到 old_string，請先 read_file 確認確切內容"
                         "（注意縮排、空白與空行）：" + path}
    if n > 1 and not replace_all:
        return {"error": f"old_string 在檔案中出現 {n} 次，不唯一；"
                         f"請加長上下文使其唯一，或明確傳 replace_all=true"}

    if not _confirm(f"編輯檔案 {path}（{n} 處替換）"):
        return {"error": "使用者拒絕執行"}
    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text.replace(old_string, new_string))
    except OSError as e:
        return {"error": f"寫入失敗: {e}"}

    READ_STATE[path] = os.stat(path).st_mtime_ns
    return {"ok": True, "content": f"已替換 {path} 的 {n} 處"}


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

def safe_json(s: str) -> dict | None:
    """唯一的 tool input 解析路徑：只接受 JSON object，否則回 None。

    回 None 代表「這不是一個 dict」，呼叫端必須回錯誤 tool_result 讓模型重送。
    """
    if not s:
        return {}
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ─── 渲染層：只負責終端輸出 ───────────────────────────────

class StreamRenderer:
    """串流輸出渲染器。執行到 close() 務必呼叫，確保思考區 ANSI 色碼收合。"""

    def __init__(self):
        self._in_thinking = False

    def text(self, s: str) -> None:
        self._close_thinking()
        print(s, end="", flush=True)

    def thinking(self, s: str) -> None:
        if not self._in_thinking:
            self._in_thinking = True
            print("\033[90m💭 [思考]\033[0m\n\033[90m", end="", flush=True)
        print(s, end="", flush=True)

    def close(self) -> None:
        self._close_thinking()
        print(flush=True)

    def _close_thinking(self) -> None:
        if self._in_thinking:
            self._in_thinking = False
            print("\033[0m\n", end="", flush=True)


# ─── 串流層：只負責收 SSE event、組裝 content blocks ──────

def stream_turn(renderer: StreamRenderer) -> tuple[list[dict], list[dict]]:
    """送出一輪請求，收完 SSE 串流。

    回傳 (content_blocks, tool_uses)：
      content_blocks: 要 append 進 messages 的 assistant content
      tool_uses:      [{"id":..., "name":..., "input": dict, "error": str|None}, ...]
                      （input 已解析、只解析一次；error 非 None 代表這筆不可執行）

    同時記下 LAST_STOP_REASON：若是 "max_tokens"，最後一個 content block 是被截斷的，
    它的 tool_use 一律視為不完整（不執行，改回錯誤 tool_result 讓模型重送）。
    """
    global LAST_USAGE, LAST_STOP_REASON
    LAST_STOP_REASON = None      # 每輪重置，避免沿用上一輪的 stop_reason

    resp = requests.post(MESSAGES_URL, json=build_payload(),
                         timeout=120, stream=True)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    blocks: dict[int, dict] = {}

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[len("data: "):])
        except json.JSONDecodeError:
            continue

        etype = ev.get("type")

        if etype == "message_start":
            LAST_STOP_REASON = ev.get("message", {}).get("stop_reason")

        elif etype == "message_delta":
            usage = ev.get("usage")
            if usage:
                LAST_USAGE = usage
            # stop_reason 通常在最後一個 message_delta 才送達
            stop_reason = ev.get("delta", {}).get("stop_reason")
            if stop_reason:
                LAST_STOP_REASON = stop_reason

        elif etype == "content_block_start":
            block = ev.get("content_block", {})
            blocks[ev.get("index", 0)] = {
                "type": block.get("type", ""),
                "text": "",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                # thinking block 開頭自帶 signature（可能為空字串，
                # 之後由 signature_delta 逐步補上）
                "signature": block.get("signature", ""),
            }

        elif etype == "content_block_delta":
            delta = ev.get("delta", {})
            dtype = delta.get("type")
            blk = blocks.setdefault(ev.get("index", 0), {"type": "", "text": ""})

            if dtype == "thinking_delta":
                renderer.thinking(delta.get("thinking", ""))
                blk["text"] += delta.get("thinking", "")
            elif dtype == "signature_delta":
                blk["signature"] = blk.get("signature", "") + delta.get("signature", "")
            elif dtype == "text_delta":
                renderer.text(delta.get("text", ""))
                blk["text"] += delta.get("text", "")
            elif dtype == "input_json_delta":
                blk["text"] += delta.get("partial_json", "")

    # max_tokens 截斷只會落在最後一個 content block 上
    return assemble_blocks(blocks, truncated=(LAST_STOP_REASON == "max_tokens"))

def assemble_blocks(blocks: dict[int, dict], *,
                    truncated: bool = False) -> tuple[list[dict], list[dict]]:
    """把串流累積的 blocks 組裝成 assistant content_blocks 與已解析的 tool_uses。

    兩條鐵律：
      1. 寫進 messages 的 tool_use.input 一定是 dict（無法解析時為 {}），
         否則下一輪的 payload 本身就不合法，會一路污染後續每一輪。
      2. input 解析不出 dict、或該 block 正好落在 max_tokens 截斷點上，
         該 tool_use 只記入歷史、標記 error，由 execute_tools 回錯誤
         tool_result 讓模型重送 —— 絕不執行半截指令。
    """
    content_blocks, tool_uses = [], []
    last_idx = max(blocks) if blocks else None
    for idx in sorted(blocks):
        blk = blocks[idx]
        if blk["type"] == "thinking" and (blk["text"] or blk.get("signature")):
            # extended thinking 規定：下一輪請求必須原樣帶回 thinking+signature，
            # 丟掉 signature 會讓 tool chain 的後續請求被 API 拒絕。
            content_blocks.append({
                "type": "thinking",
                "thinking": blk["text"],
                "signature": blk.get("signature", ""),
            })
        elif blk["type"] == "text" and blk["text"]:
            content_blocks.append({"type": "text", "text": blk["text"]})
        elif blk["type"] == "tool_use":
            tool_input = safe_json(blk["text"])
            error = None
            if tool_input is None:
                error = ("tool_use 的 input 不是合法的 JSON object"
                         f"（收到的片段：{blk['text'][:200]!r}）。"
                         "未執行，請重新產生完整且合法的 tool input。")
            elif truncated and idx == last_idx:
                error = ("本輪輸出已達 max_tokens 上限遭截斷，"
                         "最後一個 tool_use 視為不完整，未執行；"
                         "請把工作拆小或縮短指令後重送。")
            safe_input = tool_input if tool_input is not None else {}
            content_blocks.append({
                "type": "tool_use",
                "id": blk["id"],
                "name": blk["name"],
                "input": safe_input,
            })
            tool_uses.append({"id": blk["id"], "name": blk["name"],
                              "input": safe_input, "error": error})
    return content_blocks, tool_uses


def result_preview(result: dict, limit: int = 200) -> str:
    """tool result 的終端預覽。完整結果已進 messages 給模型看；
    人只需要知道大致發生了什麼。

    形狀由工具自己決定：bash 回 exit_code/stdout/stderr，
    檔案工具回 ok/content/error；這裡分派，不強迫檔案工具裝成 bash 形狀。
    """
    parts = []
    if "error" in result:
        parts.append(f"❌ {result['error']}")
    elif "exit_code" in result:
        parts.append(f"exit={result.get('exit_code')}")
        stdout = (result.get("stdout") or "").strip()
        stderr = (result.get("stderr") or "").strip()
        if stdout:
            parts.append(f"stdout: {stdout[:limit]}"
                         + ("…" if len(stdout) > limit else ""))
        if stderr:
            parts.append(f"stderr: {stderr[:limit]}"
                         + ("…" if len(stderr) > limit else ""))
        if not stdout and not stderr:
            parts.append("(無輸出)")
    else:
        content = str(result.get("content") or "").strip() or "(無輸出)"
        parts.append(("✓ " if result.get("ok") else "→ ") + content[:limit]
                     + ("…" if len(content) > limit else ""))
    return "\033[90m📤  " + " | ".join(parts) + "\033[0m"


# ─── 工具層：只負責分派與執行 ──────────────────────────────

def dispatch_tool(name: str, tool_input: dict) -> dict:
    """執行單一工具，一律回傳 dict（序列化由呼叫端負責）。

    呼叫端已保證 tool_input 是 dict，但欄位值仍可能是 None / int / list，
    所以這裡驗各工具欄位的型別，不讓 run_* 收到非預期的值。
    """
    try:
        if name == "bash":
            command = tool_input.get("command")
            if not isinstance(command, str) or not command.strip():
                return {"stdout": "",
                        "stderr": f'bash 需要非空字串的 "command" 欄位，收到: {command!r}。'
                                  "未執行，請重送正確的 input。",
                        "exit_code": -1}
            return run_bash(command)
        if name == "read_file":
            path = tool_input.get("path")
            if not isinstance(path, str) or not path.strip():
                return {"error": f'read_file 需要非空字串的 "path" 欄位，收到: {path!r}。'
                                 "未執行，請重送正確的 input。"}
            offset = tool_input.get("offset", 1)
            if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
                return {"error": f'read_file 的 "offset" 需為 >= 1 的整數，收到: {offset!r}'}
            limit = tool_input.get("limit")
            if limit is not None and (not isinstance(limit, int)
                                      or isinstance(limit, bool) or limit < 1):
                return {"error": f'read_file 的 "limit" 需為 >= 1 的整數或省略，收到: {limit!r}'}
            force = tool_input.get("force", False)
            if not isinstance(force, bool):
                return {"error": f'read_file 的 "force" 需為布林值，收到: {force!r}'}
            return run_read_file(path, offset, limit, force)

        if name == "write_file":
            path = tool_input.get("path")
            content = tool_input.get("content")
            if not isinstance(path, str) or not path.strip():
                return {"error": f'write_file 需要非空字串的 "path" 欄位，收到: {path!r}。'
                                 "未執行，請重送正確的 input。"}
            if not isinstance(content, str):
                return {"error": f'write_file 的 "content" 需為字串（空檔傳 ""），'
                                 f"收到: {type(content).__name__}"}
            return run_write_file(path, content)

        if name == "edit_file":
            path = tool_input.get("path")
            old_string = tool_input.get("old_string")
            new_string = tool_input.get("new_string")
            replace_all = tool_input.get("replace_all", False)
            if not isinstance(path, str) or not path.strip():
                return {"error": f'edit_file 需要非空字串的 "path" 欄位，收到: {path!r}。'
                                 "未執行，請重送正確的 input。"}
            if not isinstance(old_string, str) or not isinstance(new_string, str):
                return {"error": "edit_file 的 old_string / new_string 需為字串，"
                                 f"收到: old={type(old_string).__name__}, "
                                 f"new={type(new_string).__name__}"}
            if not old_string:
                return {"error": "old_string 不可為空字串（會匹配所有位置）"}
            if not isinstance(replace_all, bool):
                return {"error": f'edit_file 的 "replace_all" 需為布林值，收到: {replace_all!r}'}
            return run_edit_file(path, old_string, new_string, replace_all)

        return {"error": f"未知工具: {name}"}
    except Exception as e:
        return {"stdout": "", "stderr": f"工具執行異常: {e}", "exit_code": -1}

def execute_tools(tool_uses: list[dict]) -> None:
    """執行本輪所有工具並 append tool_result 訊息（role=user）。

    带 error 的筆次（input 不是 dict / 被 max_tokens 截斷）一律不執行，
    直接回 is_error 的 tool_result，把「重送完整 input」的決策交還給模型。
    注意：tool_use 仍會留在 assistant 訊息裡，所以每個 tool_use_id
    都一定有對應的 tool_result，歷史不會對不上。
    """
    tool_results = []
    for tc in tool_uses:
        if tc.get("error"):
            result = {"error": tc["error"]}
            print(f"\033[90m🚫  未執行（{tc['name']}）: {tc['error']}\033[0m\n")
        else:
            result = dispatch_tool(tc["name"], tc["input"])
            print(result_preview(result) + "\n")
        entry = {
            "type": "tool_result",
            "tool_use_id": tc["id"],
            "content": json.dumps(result, ensure_ascii=False),
        }
        if "error" in result:
            entry["is_error"] = True
        tool_results.append(entry)
    messages.append({"role": "user", "content": tool_results})


# ─── 編排層：tool loop 只留在這裡 ─────────────────────────

def chat(user_input: str) -> None:
    messages.append({"role": "user", "content": user_input})

    for _round in range(MAX_TOOL_ROUNDS):
        renderer = StreamRenderer()
        print("AI: ", end="", flush=True)
        content_blocks, tool_uses = stream_turn(renderer)
        renderer.close()

        if not content_blocks:
            # 空 content 陣列進 messages 後，下一輪請求會被
            # Anthropic 相容 API 以 400 拒絕；這輪什麼都沒收到就直接結束。
            print("⚠  本輪未收到任何 content block，略過不記入對話。")
            return

        messages.append({"role": "assistant", "content": content_blocks})

        if not tool_uses:
            print(context_status_line())
            return

        execute_tools(tool_uses)
        print(context_status_line() + "\n")

    print(f"⚠  已達最大工具迴圈次數 ({MAX_TOOL_ROUNDS})，中止本輪。")


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
    parser = argparse.ArgumentParser(description="s04: chat agent 重構版（拆分 god function：stream / render / dispatch / orchestrate）")
    parser.add_argument(
        "-bypass",
        action="store_true",
        help="跳過所有 bash 與檔案寫入/編輯確認（危險：操作將直接執行）",
    )
    args = parser.parse_args()
    BYPASS = args.bypass

    fetch_max_context()
    if MAX_CONTEXT:
        print(f"ℹ  max_context = {MAX_CONTEXT:,} tokens")

    if BYPASS:
        print("\u26a0  [bypass] 模式已開啟：所有 bash 指令與檔案寫入/編輯將跳過確認直接執行！\n")
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
            chat(user)
            print()
        except requests.exceptions.ConnectionError:
            print("⚠  連不上 server，請確認 Strata server 已啟動")
        except Exception as e:
            print(f"⚠  錯誤: {e}")

if __name__ == "__main__":
    main()
