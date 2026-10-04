import requests

BASE_URL = "http://192.168.0.182:8080"
MODEL = "local"


def main() -> None:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "給我一版python 地端server llama.cpp 多輪對話的簡易code 只用request套件"
            }
        ],
        "temperature": 0.2,
        "max_tokens": 32768,
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


if __name__ == "__main__":
    main()
