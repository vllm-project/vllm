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
    parser.add_argument("--model", type=str, default="jinaai/jina-reranker-m0")
    return parser.parse_args()


def main(args):
    api_url = f"http://{args.host}:{args.port}/score"
    model_name = args.model

    text_1 = "slm markdown"
    text_2 = {
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
                },
            },
        ]
    }
    prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}
    score_response = post_http_request(prompt=prompt, api_url=api_url)
    print("\nPrompt when text_1 is string and text_2 is a image list:")
    pprint.pprint(prompt)
    print("\nScore Response:")
    pprint.pprint(score_response.json())


if __name__ == "__main__":
    args = parse_args()
    main(args)
