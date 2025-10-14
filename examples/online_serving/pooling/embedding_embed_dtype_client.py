# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for embedding API using vLLM API server
NOTE:
    start a supported embeddings model server with `vllm serve`, e.g.
    vllm serve intfloat/e5-small
"""

import argparse
import base64
import io

import requests
import torch

from vllm.entrypoints.openai.protocol import EMBED_DTYPE_TO_TORCH_DTYPE


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

    for embed_dtype, torch_dtype in EMBED_DTYPE_TO_TORCH_DTYPE.items():
        prompt = {
            "model": model_name,
            "input": "vLLM is great!",
            "encoding_format": "base64",
            "embed_dtype": embed_dtype,
        }
        response = post_http_request(prompt=prompt, api_url=api_url)

        embedding = []
        for data in response.json()["data"]:
            # Decode base64 into a writable BytesIO buffer for zero-copy tensor creation
            bio_in = io.BytesIO(data["embedding"].encode("ascii"))
            bio_out = io.BytesIO()
            base64.decode(bio_in, bio_out)

            # Get writable memoryview and create tensor with zero-copy semantics
            mv = bio_out.getbuffer()
            tensor = torch.frombuffer(mv, dtype=torch_dtype)

            # Keep buffer alive to ensure memory validity
            tensor._buffer_owner = bio_out

            embedding.append(tensor.to(torch.float32))
        embedding = torch.cat(embedding)
        print(embed_dtype, embedding.shape)


if __name__ == "__main__":
    args = parse_args()
    main(args)
