"""Example code for running queries from vLLM API server.
Sample Usage:
1. Launch a vLLM server with speculative decoding enabled:
python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 8 --draft-model TinyLlama/TinyLlama-1.1B-Chat-v0.6 --speculate-length 5
2. Run query using this script:
python api_client_spec_dec.py --prompt "San Francisco is a" --stream
"""

import argparse
import json
from typing import Iterable, List

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      max_tokens: int = 256,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_content(
            chunk_size=8192,
            decode_unicode=True,
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8")[:-1])
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate"
    max_tokens = args.max_tokens
    stream = args.stream

    print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(prompt, api_url, max_tokens, stream)

    if stream:
        num_printed_lines = 0
        char_printed = 0
        for h in get_streaming_response(response):
            line = h[0]
            new_chars = line[char_printed:]
            char_printed = len(line)
            print(f"{new_chars}", flush=True, end='')
            num_printed_lines += 1
        print()
    else:
        output = get_response(response)
        line = output[0]
        print(f"{line!r}", flush=True)
