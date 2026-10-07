# harness_study

從一個只會呼叫一次 API 的程式開始，逐步疊代成可用的 agent，最後讓它自己寫後續程式碼直到成為harness。

## 模型 / inference engine
qwen3.8-flash-next-iq3_s context:262K
https://github.com/Niko1221/Strata

## 階段

| 階段 | 檔案 | 說明 | 產生方式 |
|---|---|---|---|
| s00 | `src/agent/s00_api_call.py` | 單次 API call | 手寫 |
| s01 | `src/agent/s01_chat.py` | 多輪chat | s00_api_call.py |
| s02 | `src/agent/s02_streaming.py` | streaming | s01_chat.py |
| s03 | `src/agent/s03_bash.py` | bash tool（OpenAI 格式）+ bypass | s02_streaming.py |
| s03.1 | `src/agent/s03_1_bash.py` | 改用 Anthropic 原生格式 + context 狀態列 | s03_bash.py |
| s04 | `src/agent/s04_file_tool.py` | 加入 read/write/edit 檔案工具（分頁、讀寫保護） | s03_1_bash.py |
| s05 | `src/agent/s05_session.py` | Session OOP 化（狀態收攏）+ session 級 JSONL trace | s04_file_tool.py |
| s06 | `src/agent/s06_permissions.py` | 拿掉 bypass：解析 + 白名單 + 沙箱 + 快照 + judge 顧問 | s05_session.py |

每個階段是獨立可執行的單一檔案，方便對照演進。權限與記錄已進 harness 層，
下一步是重構（單一檔案已包太多東西）。

## 執行

```bash
uv run agent                          # 最新階段（目前 = s06）
uv run agent-s00                      # 指定階段
...
uv run agent-s05
uv run agent-s06 --help               # --policy/--sandbox/--judge/--on-ask

uv run agent --judge veto             # judge 可否決（把放行降級成問人）
uv run agent --policy strict          # 改動物業一律問人
uv run agent --non-interactive --on-ask allow   # 腳本模式
uv run agent -bypass                  # 舊行為（等同 --policy off，仍走沙箱）
```

session 內指令：`/permissions` 看規則、`/snap` 列快照、`/undo [id]` 回滾、
`/context` 精算、`/trace` 路徑。

