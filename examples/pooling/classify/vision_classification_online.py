# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Example Python client for multimodal classification API using vLLM API server
NOTE:
    start a supported multimodal classification model server with `vllm serve`, e.g.
    vllm serve muziyongshixin/Qwen2.5-VL-7B-for-VideoCls \
         --runner pooling \
         --max-model-len 5000 \
         --limit-mm-per-prompt.video 1 \
         --hf-overrides '{"text_config": {"architectures": ["Qwen2_5_VLForSequenceClassification"]}}'
"""

import argparse
import pprint

import requests

from vllm.multimodal.utils import encode_image_url, fetch_image

input_text = "This product was excellent and exceeded my expectations"
image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
image_base64 = {"url": encode_image_url(fetch_image(image_url))}
video_url = "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4"


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--host", type=str, default="localhost")
    parse.add_argument("--port", type=int, default=8000)
    return parse.parse_args()


def main(args):
    base_url = f"http://{args.host}:{args.port}"
    models_url = base_url + "/v1/models"
    classify_url = base_url + "/classify"

    response = requests.get(models_url)
    model_name = response.json()["data"][0]["id"]

    print("Text classification output:")
    messages = [
        {
            "role": "assistant",
            "content": "Please classify this text request.",
        },
        {
            "role": "user",
            "content": input_text,
        },
    ]
    response = requests.post(
        classify_url,
        json={"model": model_name, "messages": messages},
    )
    pprint.pprint(response.json())

    print("Image url classification output:")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please classify this image."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]
    response = requests.post(
        classify_url,
        json={"model": model_name, "messages": messages},
    )
    pprint.pprint(response.json())

    print("Image base64 classification output:")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please classify this image."},
                {"type": "image_url", "image_url": image_base64},
            ],
        }
    ]
    response = requests.post(
        classify_url,
        json={"model": model_name, "messages": messages},
    )
    pprint.pprint(response.json())

    print("Video url classification output:")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please classify this video."},
                {"type": "video_url", "video_url": {"url": video_url}},
            ],
        }
    ]
    response = requests.post(
        classify_url,
        json={"model": model_name, "messages": messages},
    )
    pprint.pprint(response.json())


if __name__ == "__main__":
    args = parse_args()
    main(args)
