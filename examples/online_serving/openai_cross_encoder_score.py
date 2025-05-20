# SPDX-License-Identifier: Apache-2.0
"""
Example online usage of Score API.

Run `vllm serve <model> --task score` to start up the server in vLLM.
"""

import argparse
import pprint

import requests


def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-v2-m3")
    return parser.parse_args()


def main(args):
    api_url = f"http://{args.host}:{args.port}/score"
    model_name = args.model

    text_1 = "What is the capital of Brazil?"
    text_2 = "The capital of Brazil is Brasilia."
    prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}
    score_response = post_http_request(prompt=prompt, api_url=api_url)
    print("\nPrompt when text_1 and text_2 are both strings:")
    pprint.pprint(prompt)
    print("\nScore Response:")
    pprint.pprint(score_response.json())

    text_1 = "What is the capital of France?"
    text_2 = ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]
    prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}
    score_response = post_http_request(prompt=prompt, api_url=api_url)
    print("\nPrompt when text_1 is string and text_2 is a list:")
    pprint.pprint(prompt)
    print("\nScore Response:")
    pprint.pprint(score_response.json())

    text_1 = ["What is the capital of Brazil?", "What is the capital of France?"]
    text_2 = ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]
    prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}
    score_response = post_http_request(prompt=prompt, api_url=api_url)
    print("\nPrompt when text_1 and text_2 are both lists:")
    pprint.pprint(prompt)
    print("\nScore Response:")
    pprint.pprint(score_response.json())


if __name__ == "__main__":
    args = parse_args()
    main(args)
