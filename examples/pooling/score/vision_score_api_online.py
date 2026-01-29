# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

"""
Example online usage of Score API.

Run `vllm serve <model> --runner pooling` to start up the server in vLLM.
e.g.
    vllm serve jinaai/jina-reranker-m0 --runner pooling

    vllm serve Qwen/Qwen3-VL-Reranker-2B \
        --runner pooling \
        --max-model-len 4096 \
        --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
        --chat-template examples/pooling/score/template/qwen3_vl_reranker.jinja
"""

import argparse
import json
import pprint

import requests

headers = {"accept": "application/json", "Content-Type": "application/json"}

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main(args):
    base_url = f"http://{args.host}:{args.port}"
    models_url = base_url + "/v1/models"
    score_url = base_url + "/score"

    response = requests.get(models_url, headers=headers)
    model = response.json()["data"][0]["id"]

    prompt = {"model": model, "text_1": text_1, "text_2": text_2}
    response = requests.post(score_url, headers=headers, json=prompt)
    print("\nPrompt when text_1 is string and text_2 is a image list:")
    pprint.pprint(prompt)
    print("\nScore Response:")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)
