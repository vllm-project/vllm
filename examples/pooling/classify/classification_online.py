# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for classification API using vLLM API server
NOTE:
    start a supported classification model server with `vllm serve`, e.g.
    vllm serve jason9693/Qwen2.5-1.5B-apeach
"""

import argparse
import pprint

import requests

headers = {"accept": "application/json", "Content-Type": "application/json"}


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--host", type=str, default="localhost")
    parse.add_argument("--port", type=int, default=8000)
    return parse.parse_args()


def main(args):
    base_url = f"http://{args.host}:{args.port}"
    models_url = base_url + "/v1/models"
    classify_url = base_url + "/classify"
    tokenize_url = base_url + "/tokenize"

    response = requests.get(models_url, headers=headers)
    model = response.json()["data"][0]["id"]

    # /classify can accept str as input
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    payload = {
        "model": model,
        "input": prompts,
    }
    response = requests.post(classify_url, headers=headers, json=payload)
    pprint.pprint(response.json())

    # /classify can accept token ids as input
    token_ids = []
    for prompt in prompts:
        response = requests.post(
            tokenize_url,
            json={"model": model, "prompt": prompt},
        )
        token_ids.append(response.json()["tokens"])

    payload = {
        "model": model,
        "input": token_ids,
    }
    response = requests.post(classify_url, headers=headers, json=payload)
    pprint.pprint(response.json())


if __name__ == "__main__":
    args = parse_args()
    main(args)
