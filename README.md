# harness_study

從一個只會呼叫一次 API 的程式開始，逐步疊代成可用的 agent，最後讓它自己寫後續程式碼直到成為harness。

## 階段

| 階段 | 檔案 | 說明 | 產生方式 |
|---|---|---|---|
| s00 | `src/agent/s00_api_call.py` | 單次 API call | 手寫 |
| s01 | `src/agent/s01_chat.py` | 多輪chat | s00_api_call.py |
| s02 | `src/agent/s02_streaming.py` | streaming | s01_chat.py |
| s03 | `src/agent/s03_bash.py` | bash tool | s02_streaming.py |

每個階段是獨立可執行的單一檔案，方便對照演進。目前仍在 agent 層，尚未到 harness 層
（上下文管理、權限、記錄等）。

## 執行

```bash
uv run agent                          # 最新階段（目前 = s04）
uv run agent-s00                      # 指定階段
...
uv run agent-s03
```
