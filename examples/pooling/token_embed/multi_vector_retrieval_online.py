# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example online usage of Pooling API for multi vector retrieval.

Run `vllm serve <model> --runner pooling`
to start up the server in vLLM. e.g.

vllm serve BAAI/bge-m3 --pooler-config.task token_embed
"""

import argparse
import pprint

import requests
import torch


def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")

    return parser.parse_args()


def main(args):
    pooling_url = f"http://{args.host}:{args.port}/pooling"
    score_url = f"http://{args.host}:{args.port}/score"
    model_name = args.model

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompt = {"model": model_name, "input": prompts}

    pooling_response = post_http_request(prompt=prompt, api_url=pooling_url)
    for output in pooling_response.json()["data"]:
        multi_vector = torch.tensor(output["data"])
        print(multi_vector.shape)

    queries = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]
    prompt = {"model": model_name, "queries": queries, "documents": documents}
    score_response = post_http_request(prompt=prompt, api_url=score_url)
    print("\nPrompt when queries is string and documents is a list:")
    pprint.pprint(prompt)
    print("\nScore Response:")
    pprint.pprint(score_response.json())


if __name__ == "__main__":
    args = parse_args()
    main(args)
