# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E test for mixing `prompt_embeds` with `image_embeds` in a single
Chat Completions request."""

import json

import openai
import pytest
import pytest_asyncio
import torch
from transformers import AutoConfig

from tests.utils import RemoteOpenAIServer
from vllm.assets.image import ImageAsset
from vllm.utils.serial_utils import tensor2base64

LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"

# Use the model's native dtype to avoid mixed-precision casting warnings.
# `safe_load_prompt_embeds` validates that the input embedding tensors dtype matches
# `ModelConfig.dtype`, so fixture tensors must use the same value.
LLAVA_DTYPE = torch.float16


@pytest.fixture(scope="module")
def llava_server_args() -> list[str]:
    return [
        "--dtype",
        "float16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.4",
        "--limit-mm-per-prompt",
        json.dumps({"image": 1}),
        "--enable-prompt-embeds",
        "--enable-mm-embeds",
    ]


@pytest.fixture(scope="module")
def llava_server(llava_server_args):
    with RemoteOpenAIServer(
        LLAVA_MODEL,
        llava_server_args,
        max_wait_seconds=600,
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def llava_client(llava_server):
    async with llava_server.get_async_client() as async_client:
        yield async_client


@pytest.fixture(scope="module")
def llava_prompt_embeds_b64() -> str:
    """Synthetic `prompt_embeds` tensor shaped to LLaVA's text-backbone
    hidden size."""
    hidden_size = AutoConfig.from_pretrained(LLAVA_MODEL).text_config.hidden_size
    tensor = torch.randn(4, hidden_size, dtype=LLAVA_DTYPE)
    return tensor2base64(tensor)


@pytest.fixture(scope="module")
def llava_image_embeds_b64() -> str:
    """Real precomputed image_embeds for LLaVA-1.5."""
    tensor = ImageAsset("stop_sign").image_embeds.to(dtype=LLAVA_DTYPE)
    return tensor2base64(tensor)


@pytest.mark.asyncio
async def test_prompt_embeds_plus_image_embeds(
    llava_client: openai.AsyncOpenAI,
    llava_prompt_embeds_b64: str,
    llava_image_embeds_b64: str,
):
    """Single user message carrying both `prompt_embeds` and `image_embeds` parts."""
    chat = await llava_client.chat.completions.create(
        model=LLAVA_MODEL,
        max_tokens=5,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": llava_prompt_embeds_b64},
                    {
                        "type": "image_embeds",
                        "image_embeds": llava_image_embeds_b64,
                    },
                    {"type": "text", "text": "Continue."},
                ],
            }
        ],
    )
    assert chat.choices[0].message.content is not None
    assert len(chat.choices[0].message.content) > 0
