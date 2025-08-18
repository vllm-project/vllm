# SPDX-License-Identifier: Apache-2.0
"""Example Python client for `vllm.entrypoints.api_server`
NOTE: The API server is used only for demonstration and simple performance
benchmarks. It is not intended for production use.
For production use, we recommend `vllm serve` and the OpenAI client API.
"""

import argparse
import json
from collections.abc import Iterable

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": n,
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": stream,
    }
    response = requests.post(api_url,
                             headers=headers,
                             json=pload,
                             stream=stream)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[list[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\n"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> list[str]:
    data = json.loads(response.content)
    output = data["generated_text"]
    token_latency = data["per_token_latency"]
    metrics = {}
    if 'ttft' in data:
        metrics['ttft'] = data['ttft']
    if 'waiting_latency' in data:
        metrics['waiting_latency'] = data['waiting_latency']
    if 'inference_latency' in data:
        metrics['inference_latency'] = data['inference_latency']
    return output, token_latency, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--return_scheduler_trace", type=bool, default=True)
    args = parser.parse_args()
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate_benchmark"
    n = args.n
    stream = args.stream
    return_scheduler_trace = args.return_scheduler_trace

    print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(prompt, api_url, n, stream)
    if return_scheduler_trace:
        api_url = f"http://{args.host}:{args.port}/schedule_trace"
        scheduler_response = get_http_request(api_url)
        trace_data = json.loads(scheduler_response.content)
        print(f"Scheduler trace: {trace_data}", flush=True)

    if stream:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                print(f"Beam candidate {i}: {line!r}", flush=True)
    else:
        output, token_latency, metrics = get_response(response)
        print(output)
        for i in range(len(token_latency)):
            print(f"latency of {i} token finished at {token_latency[i][0]} and taken {token_latency[i][1]} miliseconds")
        print(metrics)

