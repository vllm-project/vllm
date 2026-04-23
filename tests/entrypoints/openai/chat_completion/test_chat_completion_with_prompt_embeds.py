# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for `prompt_embeds` content parts in the Chat Completions API."""

import io

import openai
import pybase64 as base64
import pytest
import pytest_asyncio
import torch
from openai import BadRequestError

from tests.utils import VLLM_PATH, RemoteOpenAIServer

MODEL_NAME = "facebook/opt-125m"
CHAT_TEMPLATE = VLLM_PATH / "examples/template_chatml.jinja"
# Must match `--dtype` in `server_args`, `safe_load_prompt_embeds` rejects
# embedding tensors whose dtype doesn't match the loaded model.
SERVER_DTYPE: torch.dtype = torch.bfloat16


@pytest.fixture(scope="module")
def server_args() -> list[str]:
    return [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        "--chat-template",
        str(CHAT_TEMPLATE),
        # Prompt Embeds server args
        "--enable-prompt-embeds",
    ]


@pytest.fixture(scope="module")
def server(server_args):
    with RemoteOpenAIServer(MODEL_NAME, server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


def _encode_embeds(embeds: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(embeds, buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture(scope="module")
def prompt_embeds_b64(hf_runner) -> list[str]:
    """Pre-compute embeddings for two short prompts and return as base64."""
    prompts = ["Hello, my name is", "What is an LLM?"]
    with hf_runner(MODEL_NAME) as hf_model:
        embeddings = hf_model.get_prompt_embeddings(prompts)
    # Cast to the server's dtype so `safe_load_prompt_embeds` accepts them.
    return [_encode_embeds(e.to(SERVER_DTYPE)) for e in embeddings]


@pytest.mark.asyncio
async def test_single_prompt_embeds_part(
    client: openai.AsyncOpenAI,
    prompt_embeds_b64: list[str],
):
    """A user message with one prompt_embeds part + text."""
    b64 = prompt_embeds_b64[0]
    chat = await client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=5,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": b64},
                    {"type": "text", "text": "Continue:"},
                ],
            }
        ],
    )
    assert chat.choices[0].message.content is not None
    assert len(chat.choices[0].message.content) > 0


@pytest.mark.asyncio
async def test_multiple_prompt_embeds_parts(
    client: openai.AsyncOpenAI,
    prompt_embeds_b64: list[str],
):
    """Multiple prompt_embeds parts in a single message."""
    b64_a, b64_b = prompt_embeds_b64
    chat = await client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=5,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": b64_a},
                    {"type": "text", "text": " and "},
                    {"type": "prompt_embeds", "data": b64_b},
                ],
            }
        ],
    )
    assert chat.choices[0].message.content is not None
    assert len(chat.choices[0].message.content) > 0


@pytest.mark.asyncio
async def test_multi_message_conversation(
    client: openai.AsyncOpenAI,
    prompt_embeds_b64: list[str],
):
    """prompt_embeds in both system and user messages."""
    b64_sys, b64_usr = prompt_embeds_b64
    chat = await client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=5,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are helpful."},
                    {"type": "prompt_embeds", "data": b64_sys},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": b64_usr},
                    {"type": "text", "text": "Summarize."},
                ],
            },
        ],
    )
    assert chat.choices[0].message.content is not None
    assert len(chat.choices[0].message.content) > 0


@pytest.mark.asyncio
async def test_streaming(
    client: openai.AsyncOpenAI,
    prompt_embeds_b64: list[str],
):
    """Streaming chat completion with prompt_embeds."""
    b64 = prompt_embeds_b64[0]

    # Non-streaming baseline.
    baseline = await client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=5,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": b64},
                    {"type": "text", "text": "Continue:"},
                ],
            }
        ],
    )
    expected = baseline.choices[0].message.content

    # Streaming.
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": b64},
                    {"type": "text", "text": "Continue:"},
                ],
            }
        ],
    )
    chunks: list[str] = []
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            chunks.append(delta)
    assert "".join(chunks) == expected


@pytest.mark.asyncio
async def test_text_only_still_works(
    client: openai.AsyncOpenAI,
):
    """Sanity check: plain text chat (no embeds) still works when
    --enable-prompt-embeds is set."""
    chat = await client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=5,
        temperature=0.0,
        messages=[{"role": "user", "content": "Hello World!"}],
    )
    assert chat.choices[0].message.content is not None
    assert len(chat.choices[0].message.content) > 0


@pytest.mark.asyncio
async def test_missing_data_field(
    client: openai.AsyncOpenAI,
):
    """A prompt_embeds part without `data` should return a clear error."""
    with pytest.raises(BadRequestError):
        await client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "prompt_embeds"}],
                }
            ],
        )


@pytest.mark.asyncio
async def test_invalid_base64(
    client: openai.AsyncOpenAI,
):
    """Invalid base64 in the `data` field should return a clear error."""
    with pytest.raises(BadRequestError):
        await client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "prompt_embeds", "data": "not_valid_base64!!"},
                    ],
                }
            ],
        )
