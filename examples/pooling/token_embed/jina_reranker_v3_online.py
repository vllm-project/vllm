# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

"""
Example online usage of the Jina Reranker v3 score and rerank APIs with a task
instruction.

Run `vllm serve jinaai/jina-reranker-v3 --runner pooling` to start up the
server in vLLM.
"""

import argparse
import json

import requests


def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response


def print_response(name: str, prompt: dict, response: requests.Response) -> None:
    print(f"\n{name} request:")
    print(json.dumps(prompt, indent=2))
    print(f"\n{name} response:")
    print(json.dumps(response.json(), indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="jinaai/jina-reranker-v3")
    return parser.parse_args()


def main(args):
    score_url = f"http://{args.host}:{args.port}/score"
    rerank_url = f"http://{args.host}:{args.port}/rerank"
    model_name = args.model

    query = "Which passage is about sports?"
    documents = [
        "Basketball is played by two teams on a court.",
        "Green tea contains antioxidants and may support metabolism.",
    ]
    instruction = "Rank passages about sports higher than passages about nutrition."

    score_prompt = {
        "model": model_name,
        "queries": query,
        "documents": documents,
        "instruction": instruction,
    }
    score_response = post_http_request(prompt=score_prompt, api_url=score_url)
    print_response("Score", score_prompt, score_response)

    rerank_prompt = {
        "model": model_name,
        "query": query,
        "documents": documents,
        "instruction": instruction,
    }
    rerank_response = post_http_request(prompt=rerank_prompt, api_url=rerank_url)
    print_response("Rerank", rerank_prompt, rerank_response)


if __name__ == "__main__":
    args = parse_args()
    main(args)
