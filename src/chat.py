import requests

BASE_URL = "http://192.168.0.182:8080"
MODEL = "local"

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "你好，請用中文自我介紹。"
        }
    ],
    "temperature": 0.2,
    "max_tokens": 512,
    "stream": False
}

r = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json=payload,
    timeout=120
)

r.raise_for_status()
data = r.json()

print(data["choices"][0]["message"]["content"])
