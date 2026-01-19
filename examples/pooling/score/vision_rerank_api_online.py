# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

"""
Example Python client for multimodal rerank API which is compatible with
Jina and Cohere https://jina.ai/reranker

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
import base64
import json

import requests


def encode_base64_content_from_url(content_url: str) -> dict[str, str]:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url, headers=headers) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return {"url": f"data:image/jpeg;base64,{result}"}


headers = {"accept": "application/json", "Content-Type": "application/json"}

query = "A woman playing with her dog on a beach at sunset."
documents = {
    "content": [
        {
            "type": "text",
            "text": (
                "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, "
                "as the dog offers its paw in a heartwarming display of companionship and trust."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
        },
        {
            "type": "image_url",
            "image_url": encode_base64_content_from_url(
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            ),
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
    rerank_url = base_url + "/rerank"

    response = requests.get(models_url, headers=headers)
    model = response.json()["data"][0]["id"]

    data = {
        "model": model,
        "query": query,
        "documents": documents,
    }
    response = requests.post(rerank_url, headers=headers, json=data)

    # Check the response
    if response.status_code == 200:
        print("Request successful!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    args = parse_args()
    main(args)
