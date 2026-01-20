# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example online usage of Score API.

Run `vllm serve <model> --runner pooling` to start up the server in vLLM.
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

    queries = "What is the capital of Brazil?"
    documents = "The capital of Brazil is Brasilia."
    prompt = {"model": model_name, "queries": queries, "documents": documents}
    score_response = post_http_request(prompt=prompt, api_url=api_url)
    print("\nPrompt when queries and documents are both strings:")
    pprint.pprint(prompt)
    print("\nScore Response:")
    pprint.pprint(score_response.json())

    queries = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]
    prompt = {"model": model_name, "queries": queries, "documents": documents}
    score_response = post_http_request(prompt=prompt, api_url=api_url)
    print("\nPrompt when queries is string and documents is a list:")
    pprint.pprint(prompt)
    print("\nScore Response:")
    pprint.pprint(score_response.json())

    queries = ["What is the capital of Brazil?", "What is the capital of France?"]
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]
    prompt = {"model": model_name, "queries": queries, "documents": documents}
    score_response = post_http_request(prompt=prompt, api_url=api_url)
    print("\nPrompt when queries and documents are both lists:")
    pprint.pprint(prompt)
    print("\nScore Response:")
    pprint.pprint(score_response.json())


if __name__ == "__main__":
    args = parse_args()
    main(args)
