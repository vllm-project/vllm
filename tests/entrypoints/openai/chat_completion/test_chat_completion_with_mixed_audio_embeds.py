# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E test for mixing `prompt_embeds` with `audio_embeds` in a single
Chat Completions request."""

import json

import openai
import pytest
import pytest_asyncio
import torch
from transformers import AutoConfig

from tests.utils import RemoteOpenAIServer
from vllm.utils.serial_utils import tensor2base64

QWEN2AUDIO_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"

# Use the model's native dtype to avoid an implicit cast inside
# `safe_load_prompt_embeds` (mismatched floating-point dtypes are cast to the
# model's dtype automatically, matching here just skips the conversion).
QWEN2AUDIO_DTYPE = torch.bfloat16


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
