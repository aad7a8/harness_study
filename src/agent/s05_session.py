import argparse
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from itertools import islice

import requests
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

SERVER       = "http://192.168.0.182:8080"
MESSAGES_URL = f"{SERVER}/v1/messages"
COUNT_URL    = f"{SERVER}/v1/messages/count_tokens"
HEALTH_URL   = f"{SERVER}/health"
MODEL        = "local-model"
SYSTEM       = "你是一個友善的 AI 助手。"

MAX_CONTEXT = None   # server 層屬性，非 session 狀態

MAX_TOOL_ROUNDS = 20

READ_MAX_LINES = 2000
READ_MAX_BYTES = 256 * 1024
READ_PAGE_CHARS = 30_000
READ_LINE_CHARS = 2_000

DEFAULT_TRACE_DIR = "traces"

ABORTED_MSG = ("工具未執行（上一輪被中斷或發生意外），此 tool_use 作廢；"
               "需要的話請重送。")


def fetch_max_context() -> int | None:
    """max_context 是 server 的屬性，取一次交給 Session 存著。"""
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        r.raise_for_status()
        return r.json().get("max_context")
    except Exception as e:
        print(f"⚠  無法取得 max_context: {e}")
        return None


def fmt_tokens(n: int) -> str:
    return f"{n/1000:.1f}K" if n >= 1000 else str(n)


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
                        "大檔用 offset/limit 分頁。"
                        "要 edit_file / write_file 覆蓋既有的檔，必須先讀過。"),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":   {"type": "string", "description": "檔案路徑"},
                "offset": {"type": "integer", "description": "起始行號（1-based），預設 1"},
                "limit":  {"type": "integer",
                           "description": f"最多讀取行數，預設 {READ_MAX_LINES}"},
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


# ─── 路徑與換行：read / edit / write 必須看同一個檔案視窗 ──────
#
# 踩過的坑：read 用 universal newlines（\r\n 被悄悄轉成 \n），
# edit / write 用 newline=""（保留 \r\n）。模型照 read 看到的內容下
# old_string，在 CRLF 檔上永遠「找不到 old_string」。
# 三条規則：
#   1. read_state 的 key 一律用正規化路徑（./x.py 與 x.py 是同一個檔）；
#   2. read 用 newline="" 看原始內容，顯示時才去行尾，並在 header
#      告知這是 CRLF 檔；
#   3. edit / write 會自動把 old_string / content 對應到檔案原本的
#      換行風格，不悄悄把 CRLF 檔改寫成 LF。

def canon_path(path: str) -> str:
    """read_state 的 key：把 ./x.py、符號連結等都收斂成同一個字串。"""
    try:
        return os.path.realpath(path)
    except OSError:
        return os.path.abspath(path)


def detect_newline(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_eol(line: str) -> str:
    """去行尾換行（\r\n / \n / \r 都算），顯示用。"""
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith(("\n", "\r")):
        return line[:-1]
    return line


# ─── trace 層：session 級 JSONL 事件記錄 ────────────────────

class TraceLog:

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "a", encoding="utf-8", newline="\n")

    def log(self, etype: str, **data) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
               "type": etype}
        rec.update(data)
        try:
            self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._f.flush()
        except Exception as e:
            print(f"⚠  trace 寫入失敗（不影響主流程）: {e}")

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


# ─── 渲染層：只負責終端輸出（無狀態）───────────────────────

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


def assemble_blocks(blocks: dict[int, dict], *,
                    truncated: bool = False) -> tuple[list[dict], list[dict]]:
    """把串流累積的 blocks 組裝成 assistant content_blocks 與已解析的 tool_uses。
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


# ─── Session：所有對話狀態 + trace 都住在這裡 ───────────────

class Session:
    """一個 session = 一段對話（messages）+ 檔案讀取狀態 + trace 檔。
    """

    def __init__(self, bypass: bool = False,
                 trace_dir: str = DEFAULT_TRACE_DIR,
                 max_context: int | None = None):
        self.bypass = bypass
        self.max_context = max_context
        self.messages: list[dict] = []
        self.read_state: dict[str, int] = {}
        self.usage: dict | None = None          # 最近一次 message_delta 的 usage
        self.stop_reason: str | None = None     # 最近一輪的 stop_reason
        self.turn = 0                           # 使用者送了幾輪
        self._closed = False
        self.session_id = (f"{datetime.now():%Y%m%d-%H%M%S}"
                           f"-{uuid.uuid4().hex[:6]}")
        os.makedirs(trace_dir, exist_ok=True)
        self.trace = TraceLog(os.path.join(trace_dir, f"{self.session_id}.jsonl"))
        self.trace.log("session_start",
                       session_id=self.session_id,
                       model=MODEL, system=SYSTEM, cwd=os.getcwd(),
                       bypass=bypass, max_context=max_context,
                       tools=[t["name"] for t in TOOLS])

    # ── context 顯示與精算 ──

    def current_context(self) -> int | None:
        """本輪結束時的 context 佔用 = input + cache_read + output。"""
        if not self.usage:
            return None
        return (self.usage.get("input_tokens", 0)
                + self.usage.get("cache_read_input_tokens", 0)
                + self.usage.get("output_tokens", 0))

    def context_status_line(self) -> str:
        used = self.current_context()
        if used is None or not self.max_context:
            return "📊 Context: —"
        pct = used / self.max_context * 100
        width = 20
        filled = min(width, round(pct / 100 * width))
        bar = "▓" * filled + "░" * (width - filled)
        return (f"\033[90m📊 Context: {bar} {fmt_tokens(used)} / "
                f"{fmt_tokens(self.max_context)} ({pct:.1f}%)\033[0m")

    def count_context_exact(self) -> None:
        """payload 必須與 /v1/messages 完全一致（system+messages+tools 全帶）；"""
        try:
            r = requests.post(COUNT_URL, json=self.build_payload(max_tokens=1),
                              timeout=10)
            r.raise_for_status()
            n = r.json().get("input_tokens")
            if n is None:
                print("⚠  count_tokens 未回傳 input_tokens")
                return
            self.trace.log("context_count", turn=self.turn, input_tokens=n)
            if self.max_context:
                pct = n / self.max_context * 100
                print(f"🎯 count_tokens 精算: {n:,} tokens / {self.max_context:,} "
                      f"({pct:.2f}%)　← 下一輪實際會讀取的 prompt 大小")
            else:
                print(f"🎯 count_tokens 精算: {n:,} tokens")
        except Exception as e:
            print(f"⚠  count_tokens 失敗: {e}")

    def clear(self) -> None:
        """/clear：清空對話與讀檔狀態，但 trace 不切檔（同一 session 接續記錄）。"""
        self.messages.clear()
        self.read_state.clear()
        self.usage = None
        self.stop_reason = None
        self.turn += 1
        self.trace.log("session_clear", turn=self.turn)

    def close(self, reason: str) -> None:
        """冪等：無論正常結束、Ctrl+C、EOF 或 crash（finally），只寫一次 session_end。"""
        if self._closed:
            return
        self._closed = True
        self.trace.log("session_end", turn=self.turn, reason=reason,
                       messages=len(self.messages))
        self.trace.close()

    # ── 歷史不变量：tool_use / tool_result 一定成對 ──

    @staticmethod
    def _aborted_result(tool_use_id: str) -> dict:
        """合成 tool_result：用來補上中斷/崩潰留下的缺口。"""
        return {"type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps({"error": ABORTED_MSG}, ensure_ascii=False),
                "is_error": True}

    def repair_history(self) -> list[str]:
        """補齊缺 tool_result 的 tool_use，回傳被補的 id 清單。

        assistant 的 tool_use 已進 messages、tool_result 卻沒進去
        （工具執行中被 Ctrl+C、崩潰、execute_tools 出錯）時，下一輪請求
        會被 Anthropic 相容 API 以 400 拒；messages 是 session 狀態，
        不修就每一輪都 400，整個 session 只能 /clear 全部丟掉。
        """
        dangling: list[tuple[int, str]] = []
        for i, msg in enumerate(self.messages):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            if msg.get("role") == "assistant":
                dangling += [(i, blk.get("id")) for blk in content
                             if blk.get("type") == "tool_use"]
            elif msg.get("role") == "user":
                answered = {blk.get("tool_use_id") for blk in content
                            if blk.get("type") == "tool_result"}
                dangling = [(j, tid) for j, tid in dangling
                            if tid not in answered]
        if not dangling:
            return []

        by_index: dict[int, list[str]] = {}
        for i, tid in dangling:
            by_index.setdefault(i, []).append(tid)
        for i in sorted(by_index, reverse=True):
            self.messages.insert(i + 1, {
                "role": "user",
                "content": [self._aborted_result(tid) for tid in by_index[i]],
            })
        ids = [tid for _, tid in dangling]
        self.trace.log("history_repaired", turn=self.turn, tool_use_ids=ids)
        print(f"⚠  已補上 {len(ids)} 筆遺漏的 tool_result"
              f"（上一輪工具被中斷或發生意外）")
        return ids

    def _append_user_input(self, user_input: str) -> None:
        """避免出現連續兩筆 user：上一輪中途失敗時歷史會停在 user。
        API 本來就會把同 role 的相鄰訊息併成一個 turn，這裡直接併掉，
        行為一致，也不留半截狀態。"""
        if self.messages and self.messages[-1].get("role") == "user":
            last = self.messages[-1]
            if isinstance(last["content"], list):
                last["content"].append({"type": "text", "text": user_input})
            else:
                last["content"] = f"{last['content']}\n{user_input}"
            self.trace.log("user_message_merged", turn=self.turn)
        else:
            self.messages.append({"role": "user", "content": user_input})

    # ── 檔案工具閘門（read-before-edit / 確認）──

    def _require_read(self, path: str) -> dict | None:
        """read-before-edit 閘門：放行回 None，擋下回錯誤 dict。
        key 用正規化路徑，否則 read 'x.py' 之後 edit './x.py' 會被誤擋。"""
        key = canon_path(path)
        if key not in self.read_state:
            return {"error": f"未先讀取過檔案，拒絕操作；請先 read_file：{path}"}
        try:
            mtime = os.stat(path).st_mtime_ns
        except OSError as e:
            return {"error": f"無法存取檔案 {path}: {e}"}
        if mtime != self.read_state[key]:
            return {"error": f"檔案自上次讀取後已被修改（user 或外部工具動的），"
                             f"請重新 read_file 後再操作：{path}"}
        return None

    def _confirm(self, desc: str) -> bool:
        """破壞性檔案操作的確認，與 bash 確認共用 bypass 同一個開關。"""
        if self.bypass:
            print("\U0001f6eb  [bypass] 已跳過確認，直接執行")
            return True
        return input(f"\u2753  {desc}，是否執行？(y/n): ").strip().lower() == "y"

    # ── 工具實作 ──

    def run_bash(self, command: str) -> dict:
        print(f"\n🖥  準備執行: {command}")
        if self.bypass:
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

    def _clip_line(self, line: str) -> str:
        if len(line) <= READ_LINE_CHARS:
            return line
        return line[:READ_LINE_CHARS] + f"…（此行超過 {READ_LINE_CHARS} 字元已截斷）"

    def _preview(self, s: str, limit: int = 120) -> str:
        """字串單行預覽（换行顯示為 \\n），用於告知將讀/寫/改什麼。"""
        s = s.replace("\n", "\\n")
        return s if len(s) <= limit else s[:limit] + "…"

    def run_read_file(self, path: str, offset: int = 1,
                      limit: int | None = None) -> dict:
        full_read = limit is None
        limit = limit or READ_MAX_LINES

        rng = "全檔" if full_read else f"第 {offset} 行起、最多 {limit} 行"
        print(f"\n📖  準備讀取: {path}　（{rng}）")

        if not os.path.isfile(path):
            return {"error": f"檔案不存在或不是普通檔案：{path}"}
        st = os.stat(path)

        if full_read and st.st_size > READ_MAX_BYTES:
            return {"error": f"檔案太大（{st.st_size:,} bytes > {READ_MAX_BYTES:,} 上限），"
                             f"請用 offset/limit 分頁讀取：{path}"}

        with open(path, "rb") as f:
            if b"\x00" in f.read(4096):
                return {"error": f"二元檔（偵測到 NUL byte），不支援讀取：{path}"}

        try:
            # newline=""：看檔案原本的換行風格。用 universal newlines 會
            # 把 \r\n 悄悄變成 \n，模型照內容下的 old_string 就匹配不上。
            with open(path, "r", encoding="utf-8", newline="") as f:
                window = list(islice(f, offset - 1, offset - 1 + limit + 1))
        except UnicodeDecodeError:
            return {"error": f"不是 UTF-8 文字檔，無法讀取：{path}"}

        if not window:
            return {"error": f"offset={offset} 已超過檔案總行數：{path}"}

        has_more = len(window) > limit
        shown = window[:limit]
        crlf = any(ln.endswith("\r\n") for ln in window)

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
        if crlf:
            header += "；CRLF 換行，內容以 LF 顯示，edit_file 會自動對應"
        header += "）"
        numbered = "\n".join(f"{offset + i:6d}\t{self._clip_line(strip_eol(ln))}"
                             for i, ln in enumerate(shown))

        self.read_state[canon_path(path)] = st.st_mtime_ns
        return {"ok": True, "content": header + "\n" + numbered}

    def run_write_file(self, path: str, content: str) -> dict:
        exists = os.path.isfile(path)
        print(f"\n✍️  準備寫入: {path}　（{len(content)} chars，"
              + ("覆蓋既有檔案" if exists else "新檔") + "）")
        if exists:
            err = self._require_read(path)
            if err:
                return err
        elif os.path.exists(path):
            return {"error": f"不是普通檔案，拒絕寫入：{path}"}

        if not self._confirm(f"寫入檔案 {path}（{len(content)} chars"
                             + ("，將覆蓋原本內容" if exists else "，新檔") + "）"):
            return {"error": "使用者拒絕執行"}

        # 覆蓋既有檔案時沿用原本的換行風格：模型一律給 LF，
        # 但 CRLF 檔不該被 write_file 悄悄改寫成 LF（整個檔 diff 爆掉）。
        newline = "\n"
        if exists:
            try:
                with open(path, "r", encoding="utf-8", newline="") as f:
                    newline = detect_newline(f.read(4096))
            except (OSError, UnicodeDecodeError):
                newline = "\n"
        if newline != "\n":
            content = to_lf(content).replace("\n", newline)

        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(content)
        except OSError as e:
            return {"error": f"寫入失敗: {e}"}

        # 剛寫的檔當「已讀」，不必重讀
        self.read_state[canon_path(path)] = os.stat(path).st_mtime_ns
        return {"ok": True,
                "content": (f"已{'覆蓋' if exists else '建立'} {path}"
                            f"（{len(content)} chars"
                            + ("，CRLF 換行" if newline != "\n" else "") + "）")}

    def run_edit_file(self, path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> dict:
        print(f"\n✏️  準備編輯: {path}　（{'全部符合處' if replace_all else '唯一符合'}）")
        print(f"      - {self._preview(old_string)}")
        print(f"      + {self._preview(new_string)}")
        err = self._require_read(path)
        if err:
            return err
        if old_string == new_string:
            return {"error": "old_string 與 new_string 相同，不需替換"}

        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError) as e:
            return {"error": f"讀取失敗: {e}"}

        # 模型看到的是 LF 視窗（read_file 顯示時去行尾），檔案可能是 CRLF：
        # 找不到時把 old/new 對應到檔案原本的換行風格再試一次。
        n = text.count(old_string)
        newline = detect_newline(text)
        if n == 0 and newline != "\n" and "\n" in old_string:
            alt_old = old_string.replace("\n", newline)
            alt_new = new_string.replace("\n", newline)
            alt_n = text.count(alt_old)
            if alt_n:
                old_string, new_string, n = alt_old, alt_new, alt_n

        if n == 0:
            return {"error": "找不到 old_string，請先 read_file 確認確切內容"
                             "（注意縮排、空白與空行）：" + path}
        if n > 1 and not replace_all:
            return {"error": f"old_string 在檔案中出現 {n} 次，不唯一；"
                             f"請加長上下文使其唯一，或明確傳 replace_all=true"}

        if not self._confirm(f"編輯檔案 {path}（{n} 處替換）"):
            return {"error": "使用者拒絕執行"}
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text.replace(old_string, new_string))
        except OSError as e:
            return {"error": f"寫入失敗: {e}"}

        self.read_state[canon_path(path)] = os.stat(path).st_mtime_ns
        return {"ok": True, "content": f"已替換 {path} 的 {n} 處"
                                       + ("（CRLF 對應）" if newline != "\n" else "")}

    # ── 工具層：分派與執行 ──

    def dispatch_tool(self, name: str, tool_input: dict) -> dict:
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
                return self.run_bash(command)
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
                return self.run_read_file(path, offset, limit)

            if name == "write_file":
                path = tool_input.get("path")
                content = tool_input.get("content")
                if not isinstance(path, str) or not path.strip():
                    return {"error": f'write_file 需要非空字串的 "path" 欄位，收到: {path!r}。'
                                     "未執行，請重送正確的 input。"}
                if not isinstance(content, str):
                    return {"error": f'write_file 的 "content" 需為字串（空檔傳 ""），'
                                     f"收到: {type(content).__name__}"}
                return self.run_write_file(path, content)

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
                return self.run_edit_file(path, old_string, new_string, replace_all)

            return {"error": f"未知工具: {name}"}
        except Exception as e:
            return {"stdout": "", "stderr": f"工具執行異常: {e}", "exit_code": -1}

    def execute_tools(self, tool_uses: list[dict], turn: int, rnd: int) -> None:
        """執行本輪所有工具並 append tool_result 訊息（role=user）。

        带 error 的筆次（input 不是 dict / 被 max_tokens 截斷）一律不執行，
        直接回 is_error 的 tool_result，把「重送完整 input」的決策交還給模型。
        注意：tool_use 仍會留在 assistant 訊息裡，所以每個 tool_use_id
        都一定有對應的 tool_result，歷史不會對不上。

        每個 tool_use（含未執行的）都記一筆 tool_result event，
        帶 input、result 全文、duration_ms，事後可完整重放。
        """
        tool_results = []
        try:
            for tc in tool_uses:
                t0 = time.monotonic()
                if tc.get("error"):
                    result = {"error": tc["error"]}
                    print(f"\033[90m🚫  未執行（{tc['name']}）: {tc['error']}\033[0m\n")
                else:
                    result = self.dispatch_tool(tc["name"], tc["input"])
                    print(result_preview(result) + "\n")
                self.trace.log("tool_result",
                               turn=turn, round=rnd,
                               tool_use_id=tc["id"], name=tc["name"],
                               input=tc["input"], result=result,
                               skipped=bool(tc.get("error")),
                               duration_ms=round((time.monotonic() - t0) * 1000))
                entry = {
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
                if "error" in result:
                    entry["is_error"] = True
                tool_results.append(entry)
        finally:
            answered = {e["tool_use_id"] for e in tool_results}
            for tc in tool_uses:
                if tc["id"] not in answered:
                    tool_results.append(self._aborted_result(tc["id"]))
            self.messages.append({"role": "user", "content": tool_results})

    # ── 串流層：收 SSE event、組裝 content blocks ──

    def build_payload(self, max_tokens: int = 32768) -> dict:
        return {
            "model": MODEL,
            "system": SYSTEM,
            "messages": self.messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True,
            "tools": TOOLS,
        }

    def stream_turn(self, renderer: StreamRenderer,
                    turn: int, rnd: int) -> tuple[list[dict], list[dict]]:
        """送出一輪請求，收完 SSE 串流。

        回傳 (content_blocks, tool_uses)：
          content_blocks: 要 append 進 messages 的 assistant content
          tool_uses:      [{"id":..., "name":..., "input": dict, "error": str|None}, ...]
                          （input 已解析、只解析一次；error 非 None 代表這筆不可執行）

        同時記下 self.stop_reason：若是 "max_tokens"，最後一個 content block
        是被截斷的，它的 tool_use 一律視為不完整（不執行，改回錯誤 tool_result
        讓模型重送）。

        trace：api_request（request 摘要）→ api_response（完整 blocks/usage）
        或 api_error（連不上、HTTP 錯誤等，記完再 raise）。
        """
        self.stop_reason = None      # 每輪重置，避免沿用上一輪的 stop_reason
        self.usage = None            # 同上：這輪沒收到 usage 就顯示 —，不拿舊數糊弄
        self.trace.log("api_request", turn=turn, round=rnd,
                       messages=len(self.messages))
        t0 = time.monotonic()

        blocks: dict[int, dict] = {}
        stream_error: str | None = None
        try:
            # with：SSE 連線一定要關，否則長 session 會累積 socket/fd
            with requests.post(MESSAGES_URL, json=self.build_payload(),
                               timeout=120, stream=True) as resp:
                resp.raise_for_status()
                resp.encoding = "utf-8"

                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        ev = json.loads(line[len("data: "):])
                    except json.JSONDecodeError:
                        continue

                    etype = ev.get("type")

                    if etype == "message_start":
                        message = ev.get("message", {})
                        self.stop_reason = message.get("stop_reason")
                        # message_start 通常已帶 input_tokens；
                        # 最後一個 message_delta 的 usage 會蓋掉它。
                        if message.get("usage"):
                            self.usage = message["usage"]

                    elif etype == "message_delta":
                        usage = ev.get("usage")
                        if usage:
                            self.usage = usage
                        # stop_reason 通常在最後一個 message_delta 才送達
                        stop_reason = ev.get("delta", {}).get("stop_reason")
                        if stop_reason:
                            self.stop_reason = stop_reason

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
                        blk = blocks.setdefault(ev.get("index", 0),
                                                {"type": "", "text": ""})

                        if dtype == "thinking_delta":
                            renderer.thinking(delta.get("thinking", ""))
                            blk["text"] += delta.get("thinking", "")
                        elif dtype == "signature_delta":
                            blk["signature"] = (blk.get("signature", "")
                                                + delta.get("signature", ""))
                        elif dtype == "text_delta":
                            renderer.text(delta.get("text", ""))
                            blk["text"] += delta.get("text", "")
                        elif dtype == "input_json_delta":
                            blk["text"] += delta.get("partial_json", "")

                    elif etype == "error":
                        # 吞掉 server 的 error event，最後只會看到
                        # 「未收到任何 content block」，誰都不知道出了什麼事。
                        err = ev.get("error") or {}
                        stream_error = (f"{err.get('type', 'unknown')}: "
                                        f"{err.get('message', '(無訊息)')}")
        except Exception as e:
            self.trace.log("api_error", turn=turn, round=rnd,
                           error=f"{type(e).__name__}: {e}",
                           duration_ms=round((time.monotonic() - t0) * 1000))
            raise

        if stream_error:
            self.trace.log("api_error", turn=turn, round=rnd,
                           error=stream_error,
                           duration_ms=round((time.monotonic() - t0) * 1000))
            raise RuntimeError(f"server 在串流中回報錯誤：{stream_error}")

        # 串流收完卻一個 block 都沒有：多半是 server 提早斷線
        # （缺 message_stop），記下來免得事後查不到。
        if not blocks:
            self.trace.log("api_empty", turn=turn, round=rnd,
                           stop_reason=self.stop_reason,
                           duration_ms=round((time.monotonic() - t0) * 1000))

        # max_tokens 截斷只會落在最後一個 content block 上
        content_blocks, tool_uses = assemble_blocks(
            blocks, truncated=(self.stop_reason == "max_tokens"))
        self.trace.log("api_response", turn=turn, round=rnd,
                       stop_reason=self.stop_reason,
                       usage=self.usage,
                       content_blocks=content_blocks,
                       duration_ms=round((time.monotonic() - t0) * 1000))
        return content_blocks, tool_uses

    # ── 編排層：tool loop 只留在這裡 ──

    def chat(self, user_input: str) -> None:
        self.turn += 1
        turn = self.turn
        # 上一輪可能中途失敗：先補齊 dangling tool_use，
        # 否則這輪請求會被 API 以 400 一路拒到底。
        self.repair_history()
        self.trace.log("user_message", turn=turn, text=user_input)
        self._append_user_input(user_input)

        for rnd in range(1, MAX_TOOL_ROUNDS + 1):
            renderer = StreamRenderer()
            print("AI: ", end="", flush=True)
            try:
                content_blocks, tool_uses = self.stream_turn(renderer, turn, rnd)
            finally:
                # 串流中途出錯也要收合思考區的 ANSI 色碼，
                # 否則之後的終端輸出會一路被染色。
                renderer.close()

            if not content_blocks:
                # 空 content 陣列進 messages 後，下一輪請求會被
                # Anthropic 相容 API 以 400 拒絕；這輪什麼都沒收到就直接結束。
                print("⚠  本輪未收到任何 content block，略過不記入對話。")
                self.trace.log("turn_end", turn=turn, rounds=rnd,
                                status="empty_content",
                                context=self.current_context())
                return

            self.messages.append({"role": "assistant", "content": content_blocks})

            if not tool_uses:
                print(self.context_status_line())
                self.trace.log("turn_end", turn=turn, rounds=rnd,
                                status="done", context=self.current_context())
                return

            self.execute_tools(tool_uses, turn, rnd)
            print(self.context_status_line() + "\n")

        print(f"⚠  已達最大工具迴圈次數 ({MAX_TOOL_ROUNDS})，中止本輪。")
        self.trace.log("turn_end", turn=turn, rounds=MAX_TOOL_ROUNDS,
                        status="max_rounds", context=self.current_context())


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
    parser = argparse.ArgumentParser(
        description="s05: Session OOP 版（狀態收攏成 Session + session 級 JSONL trace）")
    parser.add_argument(
        "-bypass",
        action="store_true",
        help="跳過所有 bash 與檔案寫入/編輯確認（危險：操作將直接執行）",
    )
    parser.add_argument(
        "--trace-dir",
        default=DEFAULT_TRACE_DIR,
        help=f"JSONL trace 存放目錄（預設 {DEFAULT_TRACE_DIR}/）",
    )
    args = parser.parse_args()

    max_context = fetch_max_context()
    if max_context:
        print(f"ℹ  max_context = {max_context:,} tokens")

    session = Session(bypass=args.bypass, trace_dir=args.trace_dir,
                      max_context=max_context)
    print(f"📝  session {session.session_id}，trace → {session.trace.path}")

    if args.bypass:
        print("\u26a0  [bypass] 模式已開啟：所有 bash 指令與檔案寫入/編輯將跳過確認直接執行！\n")
    print("  Enter 送出 | Alt+Enter 換行 | /context 精算 context | "
          "/trace trace 路徑 | /exit 結束\n")
    reason = "unknown"
    try:
        while True:
            user = get_multiline_input()
            if user.lower() in ("/quit", "/exit", "/q"):
                reason = "user_exit"
                break
            if user.lower() == "/clear":
                session.clear()
                print("✓ 對話已清空，開始新對話（同一份 trace 接續記錄）。\n")
                continue
            if user.lower() == "/context":
                session.count_context_exact()
                continue
            if user.lower() == "/trace":
                print(f"📝  trace: {session.trace.path}")
                continue
            if not user:
                continue
            try:
                session.chat(user)
                print()
            except requests.exceptions.ConnectionError:
                print("⚠  連不上 server，請確認 Strata server 已啟動")
            except Exception as e:
                print(f"⚠  錯誤: {e}")
    except KeyboardInterrupt:
        print("\nbye")
        reason = "keyboard_interrupt"
    except EOFError:
        reason = "eof"          # Ctrl+D
    except Exception as e:
        print(f"⚠  未預期錯誤，session 中止: {e}")
        reason = "crash"
    finally:
        session.close(reason)


if __name__ == "__main__":
    main()
