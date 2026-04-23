# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for mixing `prompt_embeds` with other embed modalities
(`image_embeds`, `audio_embeds`) in a single Chat Completions request."""

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
QWEN2AUDIO_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"

# Use each model's native dtype to avoid mixed-precision casting warnings.
# `safe_load_prompt_embeds` validates that the input embedding tensors dtype matches
# `ModelConfig.dtype`, so fixture tensors must use the same value.
LLAVA_DTYPE = torch.float16
QWEN2AUDIO_DTYPE = torch.bfloat16


#################################################################
# LLaVA: Test prompt_embeds + image_embeds in a single request. #
#################################################################


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


##################################################################
# Qwen2-Audio: prompt_embeds + audio_embeds in a single request. #
##################################################################


@pytest.fixture(scope="module")
def qwen2audio_server_args() -> list[str]:
    return [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.4",
        "--limit-mm-per-prompt",
        json.dumps({"audio": 1}),
        "--enable-prompt-embeds",
        "--enable-mm-embeds",
    ]


@pytest.fixture(scope="module")
def qwen2audio_server(qwen2audio_server_args):
    with RemoteOpenAIServer(
        QWEN2AUDIO_MODEL,
        qwen2audio_server_args,
        max_wait_seconds=600,
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def qwen2audio_client(qwen2audio_server):
    async with qwen2audio_server.get_async_client() as async_client:
        yield async_client


@pytest.fixture(scope="module")
def qwen2audio_hidden_size() -> int:
    config = AutoConfig.from_pretrained(QWEN2AUDIO_MODEL, trust_remote_code=True)
    return config.text_config.hidden_size


@pytest.fixture(scope="module")
def qwen2audio_prompt_embeds_b64(qwen2audio_hidden_size: int) -> str:
    tensor = torch.randn(4, qwen2audio_hidden_size, dtype=QWEN2AUDIO_DTYPE)
    return tensor2base64(tensor)


@pytest.fixture(scope="module")
def qwen2audio_audio_embeds_b64(qwen2audio_hidden_size: int) -> str:
    # Shape matches the `audio_embeds` unit-test fixture.
    tensor = torch.randn(1, 128, qwen2audio_hidden_size, dtype=QWEN2AUDIO_DTYPE)
    return tensor2base64(tensor)


@pytest.mark.asyncio
async def test_prompt_embeds_plus_audio_embeds(
    qwen2audio_client: openai.AsyncOpenAI,
    qwen2audio_prompt_embeds_b64: str,
    qwen2audio_audio_embeds_b64: str,
):
    """Single user message carrying both prompt_embeds and audio_embeds parts."""
    chat = await qwen2audio_client.chat.completions.create(
        model=QWEN2AUDIO_MODEL,
        max_tokens=5,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "prompt_embeds",
                        "data": qwen2audio_prompt_embeds_b64,
                    },
                    {
                        "type": "audio_embeds",
                        "audio_embeds": qwen2audio_audio_embeds_b64,
                    },
                    {"type": "text", "text": "Continue."},
                ],
            }
        ],
    )
    assert chat.choices[0].message.content is not None
    assert len(chat.choices[0].message.content) > 0
