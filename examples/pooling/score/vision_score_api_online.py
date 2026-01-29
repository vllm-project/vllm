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
import pprint
import requests
from vllm.utils.mm_utils import encode_base64_content_from_url, DEFAULT_HEADERS

query = "A woman playing with her dog on a beach at sunset."
document = (
    "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, "
    "as the dog offers its paw in a heartwarming display of companionship and trust."
)
image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
documents = [
    {
        "type": "text",
        "text": document,
    },
    {
        "type": "image_url",
        "image_url": {"url": image_url},
    },
    {
        "type": "image_url",
        "image_url": encode_base64_content_from_url(image_url),
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main(args):
    base_url = f"http://{args.host}:{args.port}"
    models_url = base_url + "/v1/models"
    score_url = base_url + "/score"

    response = requests.get(models_url, headers=DEFAULT_HEADERS)
    model = response.json()["data"][0]["id"]

    print("Query: string & Document: string")
    prompt = {"model": model, "queries": query, "documents": document}
    response = requests.post(score_url, headers=DEFAULT_HEADERS, json=prompt)
    pprint.pprint(response.json())

    print("Query: string & Document: text")
    prompt = {
        "model": model,
        "queries": query,
        "documents": {"content": [documents[0]]},
    }
    response = requests.post(score_url, headers=DEFAULT_HEADERS, json=prompt)
    pprint.pprint(response.json())

    print("Query: string & Document: image url")
    prompt = {
        "model": model,
        "queries": query,
        "documents": {"content": [documents[1]]},
    }
    response = requests.post(score_url, headers=DEFAULT_HEADERS, json=prompt)
    pprint.pprint(response.json())

    print("Query: string & Document: image base64")
    prompt = {
        "model": model,
        "queries": query,
        "documents": {"content": [documents[2]]},
    }
    response = requests.post(score_url, headers=DEFAULT_HEADERS, json=prompt)
    pprint.pprint(response.json())


if __name__ == "__main__":
    args = parse_args()
    main(args)
