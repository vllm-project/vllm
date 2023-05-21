import requests
import json

def http_request():
    prompt = "Ion Stoica is a"

    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": 4,
        "use_beam_search": True,
        "temperature": 0.0,
    }
    response = requests.post("http://localhost:10002/generate", headers=headers, json=pload, stream=True)

    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output

for h in http_request():
    print(h, flush=True)
