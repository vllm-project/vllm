# SPDX-License-Identifier: Apache-2.0

import argparse
import pprint

import requests


def post_http_request(payload: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=payload)
    return response


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--host", type=str, default="localhost")
    parse.add_argument("--port", type=int, default=8000)
    parse.add_argument("--model", type=str, default="jason9693/Qwen2.5-1.5B-apeach")
    return parse.parse_args()


def main(args):
    host = args.host
    port = args.port
    model_name = args.model

    api_url = f"http://{host}:{port}/classify"
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    payload = {
        "model": model_name,
        "input": prompts,
    }

    classify_response = post_http_request(payload=payload, api_url=api_url)
    pprint.pprint(classify_response.json())


if __name__ == "__main__":
    args = parse_args()
    main(args)
