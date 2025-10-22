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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="intfloat/e5-small")

    return parser.parse_args()


def main(args):
    api_url = f"http://{args.host}:{args.port}/v1/embeddings"
    model_name = args.model

    # The OpenAI client does not support the embed_dtype and endianness parameters.
    for embed_dtype in EMBED_DTYPE_TO_TORCH_DTYPE:
        for endianness in ENDIANNESS:
            prompt = {
                "model": model_name,
                "input": "vLLM is great!",
                "encoding_format": "base64",
                "embed_dtype": embed_dtype,
                "endianness": endianness,
            }
            response = post_http_request(prompt=prompt, api_url=api_url)

            embedding = []
            for data in response.json()["data"]:
                binary = base64.b64decode(data["embedding"])
                tensor = binary2tensor(binary, (-1,), embed_dtype, endianness)
                embedding.append(tensor.to(torch.float32))
            embedding = torch.cat(embedding)
            print(embed_dtype, endianness, embedding.shape)


if __name__ == "__main__":
    args = parse_args()
    main(args)
