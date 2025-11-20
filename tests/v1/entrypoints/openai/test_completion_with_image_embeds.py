# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import io
import json

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import torch
from transformers import AutoConfig

from tests.conftest import ImageTestAssets
from tests.utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
CONFIG = AutoConfig.from_pretrained(MODEL_NAME)
MAXIMUM_IMAGES = 2


@pytest.fixture(scope="module")
def default_image_embeds_server_args() -> list[str]:
    return [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"image": MAXIMUM_IMAGES}),
        "--enable-mm-embeds",
    ]


@pytest.fixture(scope="module")
def server_with_image_embeds(default_image_embeds_server_args):
    with RemoteOpenAIServer(
        MODEL_NAME, default_image_embeds_server_args, max_wait_seconds=600
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client_with_image_embeds(server_with_image_embeds):
    async with server_with_image_embeds.get_async_client() as async_client:
        yield async_client


def encode_image_embedding_to_base64(image_embedding) -> str:
    """
    Encode image embedding to base64 string
    """
    buffer = io.BytesIO()
    torch.save(image_embedding, buffer)
    buffer.seek(0)
    binary_data = buffer.read()
    base64_image_embedding = base64.b64encode(binary_data).decode("utf-8")
    return base64_image_embedding


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("dtype", [torch.half, torch.float16, torch.float32])
async def test_completions_with_image_embeds(
    client_with_image_embeds: openai.AsyncOpenAI,
    model_name: str,
    image_assets: ImageTestAssets,
    dtype: torch.dtype,
):
    # Test case: Single image embeds input
    image_embeds = image_assets[0].image_embeds.to(dtype=dtype)
    base64_image_embedding = encode_image_embedding_to_base64(image_embeds)
    chat_completion = await client_with_image_embeds.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe these images separately. For each image,"
                        "reply with a short sentence (no more than 10 words).",
                    },
                    {
                        "type": "image_embeds",
                        "image_embeds": base64_image_embedding,
                    },
                ],
            },
        ],
        model=model_name,
    )
    assert chat_completion.choices[0].message.content is not None
    assert isinstance(chat_completion.choices[0].message.content, str)
    assert len(chat_completion.choices[0].message.content) > 0
