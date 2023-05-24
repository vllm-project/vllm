import argparse
import requests
import json

def clear_line(n=1):
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def http_request(prompt: str, api_url: str, n: int = 1):
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": n,
        "use_beam_search": True,
        "temperature": 0.0,
        "max_tokens": 16,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)

    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    args = parser.parse_args()
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate"
    n = args.n

    print(f"Prompt: {prompt}\n", flush=True)
    num_printed_lines = 0
    for h in http_request(prompt, api_url, n):
        clear_line(num_printed_lines)
        num_printed_lines = 0
        for i, line in enumerate(h):
            num_printed_lines += 1
            print(f"Beam candidate {i}: {line}", flush=True)
