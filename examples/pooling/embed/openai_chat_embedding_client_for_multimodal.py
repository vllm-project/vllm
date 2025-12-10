# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Example Python client for multimodal embedding API using vLLM API server.

Refer to each `run_*` function for the command to run the server for that model.
"""

import argparse
import base64
import io
from typing import Literal

from openai import OpenAI
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessageParam
from openai.types.create_embedding_response import CreateEmbeddingResponse
from PIL import Image

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


def create_chat_embeddings(
    client: OpenAI,
    *,
    messages: list[ChatCompletionMessageParam],
    model: str,
    encoding_format: Literal["base64", "float"] | NotGiven = NOT_GIVEN,
) -> CreateEmbeddingResponse:
    """
    Convenience function for accessing vLLM's Chat Embeddings API,
    which is an extension of OpenAI's existing Embeddings API.
    """
    return client.post(
        "/embeddings",
        cast_to=CreateEmbeddingResponse,
        body={"messages": messages, "model": model, "encoding_format": encoding_format},
    )


def run_clip(client: OpenAI, model: str):
    """
    Start the server using:

    vllm serve openai/clip-vit-base-patch32 \
        --runner pooling
    """

    response = create_chat_embeddings(
        client,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        model=model,
        encoding_format="float",
    )

    print("Image embedding output:", response.data[0].embedding)

    response = create_chat_embeddings(
        client,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a photo of a cat"},
                ],
            }
        ],
        model=model,
        encoding_format="float",
    )

    print("Text embedding output:", response.data[0].embedding)


def run_dse_qwen2_vl(client: OpenAI, model: str):
    """
    Start the server using:

    vllm serve MrLight/dse-qwen2-2b-mrl-v1 \
        --runner pooling \
        --trust-remote-code \
        --max-model-len 8192 \
        --chat-template examples/template_dse_qwen2_vl.jinja
    """
    response = create_chat_embeddings(
        client,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            }
        ],
        model=model,
        encoding_format="float",
    )

    print("Image embedding output:", response.data[0].embedding)

    # MrLight/dse-qwen2-2b-mrl-v1 requires a placeholder image
    # of the minimum input size
    buffer = io.BytesIO()
    image_placeholder = Image.new("RGB", (56, 56))
    image_placeholder.save(buffer, "png")
    buffer.seek(0)
    image_placeholder = base64.b64encode(buffer.read()).decode("utf-8")
    response = create_chat_embeddings(
        client,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_placeholder}",
                        },
                    },
                    {"type": "text", "text": "Query: What is the weather like today?"},
                ],
            }
        ],
        model=model,
        encoding_format="float",
    )

    print("Text embedding output:", response.data[0].embedding)


def run_siglip(client: OpenAI, model: str):
    """
    Start the server using:

    vllm serve google/siglip-base-patch16-224 \
        --runner pooling \
        --chat-template template_basic.jinja
    """

    response = create_chat_embeddings(
        client,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        model=model,
        encoding_format="float",
    )

    print("Image embedding output:", response.data[0].embedding)

    response = create_chat_embeddings(
        client,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a photo of a cat"},
                ],
            }
        ],
        model=model,
        encoding_format="float",
    )

    print("Text embedding output:", response.data[0].embedding)


def run_vlm2vec(client: OpenAI, model: str):
    """
    Start the server using:

    vllm serve TIGER-Lab/VLM2Vec-Full \
        --runner pooling \
        --trust-remote-code \
        --max-model-len 4096 \
        --chat-template examples/template_vlm2vec_phi3v.jinja
    """

    response = create_chat_embeddings(
        client,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Represent the given image."},
                ],
            }
        ],
        model=model,
        encoding_format="float",
    )

    print("Image embedding output:", response.data[0].embedding)

    response = create_chat_embeddings(
        client,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": "Represent the given image with the following question: What is in the image.",
                    },
                ],
            }
        ],
        model=model,
        encoding_format="float",
    )

    print("Image+Text embedding output:", response.data[0].embedding)

    response = create_chat_embeddings(
        client,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A cat and a dog"},
                ],
            }
        ],
        model=model,
        encoding_format="float",
    )

    print("Text embedding output:", response.data[0].embedding)


model_example_map = {
    "clip": run_clip,
    "dse_qwen2_vl": run_dse_qwen2_vl,
    "siglip": run_siglip,
    "vlm2vec": run_vlm2vec,
}


def parse_args():
    parser = argparse.ArgumentParser(
        "Script to call a specified VLM through the API. Make sure to serve "
        "the model with `--runner pooling` before running this."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=model_example_map.keys(),
        required=True,
        help="The name of the embedding model.",
    )
    return parser.parse_args()


def main(args):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model_id = models.data[0].id

    model_example_map[args.model](client, model_id)


if __name__ == "__main__":
    args = parse_args()
    main(args)
