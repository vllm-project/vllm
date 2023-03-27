import requests
import json

def http_bot():
    prompt = "The future of cloud computing is"

    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
    }
    response = requests.post("http://localhost:10002/generate", headers=headers, json=pload, stream=True)

    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output

for h in http_bot():
    print(h, flush=True)