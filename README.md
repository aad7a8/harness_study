# harness_study

從一個只會呼叫一次 API 的程式開始，逐步疊代成可用的 agent，最後讓它自己寫後續程式碼直到成為harness。

## 階段

| 階段 | 檔案 | 說明 | 產生方式 |
|---|---|---|---|
| s00 | `src/agent/s00_api_call.py` | 單次 API call | 手寫 |

每個階段是獨立可執行的單一檔案，方便對照演進。目前仍在 agent 層，尚未到 harness 層
（上下文管理、權限、記錄等）。

## 執行

```bash
uv run agent                          # 最新階段
uv run python -m agent.s00_api_call   # 指定階段
```
