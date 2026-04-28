# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E test for mixing `prompt_embeds` with `audio_embeds` in a single
Chat Completions request."""

import asyncio
import json

import openai
import pytest
import pytest_asyncio
import torch
from transformers import AutoConfig, Qwen2AudioForConditionalGeneration

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


@pytest.fixture(scope="module")
def qwen2audio_aligned_content_and_embeds_b64(hf_runner) -> tuple[str, str]:
    """Return `(content, base64_embeds)` where the embeddings are the model's
    embedding of `content` tokenized WITHOUT special tokens.
    """
    content = "Describe this audio."
    with hf_runner(
        QWEN2AUDIO_MODEL,
        auto_cls=Qwen2AudioForConditionalGeneration,
    ) as hf_model:
        ids = hf_model.tokenizer(
            content, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        ids = hf_model.wrap_device({"input_ids": ids})["input_ids"]
        embed_layer = hf_model.model.get_input_embeddings()
        embeds = embed_layer(ids).squeeze(0).to(QWEN2AUDIO_DTYPE).cpu()
    return content, tensor2base64(embeds)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "audio_first",
    [True, False],
    ids=["audio_embeds-then-text", "text-then-audio_embeds"],
)
async def test_text_content_and_prompt_embeds_match_with_audio_embeds(
    qwen2audio_client: openai.AsyncOpenAI,
    qwen2audio_audio_embeds_b64: str,
    qwen2audio_aligned_content_and_embeds_b64: tuple[str, str],
    audio_first: bool,
):
    """Same content as text vs `prompt_embeds` should yield identical Chat
    Completions output when mixed with `audio_embeds` in the same message.
    """
    content, encoded_text_embeds = qwen2audio_aligned_content_and_embeds_b64

    audio_part = {
        "type": "audio_embeds",
        "audio_embeds": qwen2audio_audio_embeds_b64,
    }
    text_part = {"type": "text", "text": content}
    embeds_part = {"type": "prompt_embeds", "data": encoded_text_embeds}

    if audio_first:
        text_content = [audio_part, text_part]
        embeds_content = [audio_part, embeds_part]
    else:
        text_content = [text_part, audio_part]
        embeds_content = [embeds_part, audio_part]

    # Submit both requests concurrently.
    text_resp, embeds_resp = await asyncio.gather(
        qwen2audio_client.chat.completions.create(
            model=QWEN2AUDIO_MODEL,
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": text_content}],
        ),
        qwen2audio_client.chat.completions.create(
            model=QWEN2AUDIO_MODEL,
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": embeds_content}],
        ),
    )

    text_out = text_resp.choices[0].message.content
    embeds_out = embeds_resp.choices[0].message.content
    assert text_out is not None and len(text_out) > 0
    assert embeds_out is not None and len(embeds_out) > 0
    assert text_out == embeds_out
