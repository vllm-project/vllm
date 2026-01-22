# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for embedding API using vLLM API server
NOTE:
    start a supported embeddings model server with `vllm serve`, e.g.
    vllm serve intfloat/e5-small
"""

import argparse
import base64

import requests
import torch

from vllm.utils.serial_utils import (
    EMBED_DTYPE_TO_TORCH_DTYPE,
    ENDIANNESS,
    binary2tensor,
)


def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--host", type=str, default="localhost")
    parse.add_argument("--port", type=int, default=8000)
    return parse.parse_args()


def main(args):
    base_url = f"http://{args.host}:{args.port}"
    models_url = base_url + "/v1/models"
    embeddings_url = base_url + "/v1/embeddings"

    response = requests.get(models_url)
    model = response.json()["data"][0]["id"]

    input_texts = [
        "The best thing about vLLM is that it supports many different models",
    ] * 2

    # The OpenAI client does not support the embed_dtype and endianness parameters.
    for embed_dtype in EMBED_DTYPE_TO_TORCH_DTYPE:
        for endianness in ENDIANNESS:
            prompt = {
                "model": model,
                "input": input_texts,
                "encoding_format": "base64",
                "embed_dtype": embed_dtype,
                "endianness": endianness,
            }
            response = post_http_request(prompt=prompt, api_url=embeddings_url)

            embedding = []
            for data in response.json()["data"]:
                binary = base64.b64decode(data["embedding"])
                tensor = binary2tensor(binary, (-1,), embed_dtype, endianness)
                embedding.append(tensor.to(torch.float32))
            embedding = torch.stack(embedding)
            print(embed_dtype, endianness, embedding.shape)


if __name__ == "__main__":
    args = parse_args()
    main(args)
