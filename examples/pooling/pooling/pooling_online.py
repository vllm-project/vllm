# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example online usage of Pooling API.

Run `vllm serve <model> --runner pooling`
to start up the server in vLLM. e.g.

vllm serve internlm/internlm2-1_8b-reward --trust-remote-code
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

    return parser.parse_args()


def main(args):
    base_url = f"http://{args.host}:{args.port}"
    models_url = base_url + "/v1/models"
    pooing_url = base_url + "/pooling"

    response = requests.get(models_url)
    model = response.json()["data"][0]["id"]

    # Input like Completions API
    prompt = {"model": model, "input": "vLLM is great!"}
    pooling_response = post_http_request(prompt=prompt, api_url=pooing_url)
    print("-" * 50)
    print("Pooling Response:")
    pprint.pprint(pooling_response.json())
    print("-" * 50)

    # Input like Chat API
    prompt = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "vLLM is great!"}],
            }
        ],
    }
    pooling_response = post_http_request(prompt=prompt, api_url=pooing_url)
    print("Pooling Response:")
    pprint.pprint(pooling_response.json())
    print("-" * 50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
