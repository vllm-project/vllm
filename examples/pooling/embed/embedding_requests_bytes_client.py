# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for embedding API using vLLM API server
NOTE:
    start a supported embeddings model server with `vllm serve`, e.g.
    vllm serve intfloat/e5-small
"""

import argparse
import json

import requests
import torch

from vllm.utils.serial_utils import (
    EMBED_DTYPE_TO_TORCH_DTYPE,
    ENDIANNESS,
    MetadataItem,
    build_metadata_items,
    decode_pooling_output,
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
    embedding_size = 0

    input_texts = [
        "The best thing about vLLM is that it supports many different models",
    ] * 2

    # The OpenAI client does not support the bytes encoding_format.
    # The OpenAI client does not support the embed_dtype and endianness parameters.
    for embed_dtype in EMBED_DTYPE_TO_TORCH_DTYPE:
        for endianness in ENDIANNESS:
            prompt = {
                "model": model_name,
                "input": input_texts,
                "encoding_format": "bytes",
                "embed_dtype": embed_dtype,
                "endianness": endianness,
            }
            response = post_http_request(prompt=prompt, api_url=api_url)
            metadata = json.loads(response.headers["metadata"])
            body = response.content
            items = [MetadataItem(**x) for x in metadata["data"]]

            embedding = decode_pooling_output(items=items, body=body)
            embedding = [x.to(torch.float32) for x in embedding]
            embedding = torch.stack(embedding)
            embedding_size = embedding.shape[-1]
            print(embed_dtype, endianness, embedding.shape)

    # The vllm server always sorts the returned embeddings in the order of input. So
    # returning metadata is not necessary. You can set encoding_format to bytes_only
    # to let the server not return metadata.
    for embed_dtype in EMBED_DTYPE_TO_TORCH_DTYPE:
        for endianness in ENDIANNESS:
            prompt = {
                "model": model_name,
                "input": input_texts,
                "encoding_format": "bytes_only",
                "embed_dtype": embed_dtype,
                "endianness": endianness,
            }
            response = post_http_request(prompt=prompt, api_url=api_url)
            body = response.content

            items = build_metadata_items(
                embed_dtype=embed_dtype,
                endianness=endianness,
                shape=(embedding_size,),
                n_request=len(input_texts),
            )
            embedding = decode_pooling_output(items=items, body=body)
            embedding = [x.to(torch.float32) for x in embedding]
            embedding = torch.stack(embedding)
            print(embed_dtype, endianness, embedding.shape)


if __name__ == "__main__":
    args = parse_args()
    main(args)
