# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test: calling ``/tokenize`` with multimodal data followed by
``/v1/chat/completions`` with the same data must not cause an error.

The bug: after multimodal tokenization support was added, ``/tokenize`` fully
processes multimodal inputs (images) through the renderer but discards the
mm features.  When a subsequent ``/v1/chat/completions`` request with the same
image arrives, the engine may fail because it expects multimodal features that
were consumed/cached incorrectly by the tokenization call.
"""

import json

import openai
import pytest
import pytest_asyncio
import requests

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "5",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"image": 1}),
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


def test_tokenize_then_chat_completion_with_image(
    server: RemoteOpenAIServer,
    local_asset_server,
):
    """Tokenize a multimodal message, then send the same message to chat
    completions.  The chat completion must succeed (not 500)."""

    image_url = local_asset_server.url_for("stop_sign.jpg")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]

    # Step 1: tokenize (this triggers multimodal processing in the renderer)
    tok_resp = requests.post(
        server.url_for("tokenize"),
        json={"model": MODEL_NAME, "messages": messages},
    )
    tok_resp.raise_for_status()
    tok_data = tok_resp.json()
    assert tok_data["count"] > 0, "Tokenization must return tokens"

    # Step 2: chat completion with the SAME multimodal message
    chat_resp = requests.post(
        server.url_for("v1/chat/completions"),
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 10,
            "temperature": 0.0,
        },
    )

    assert chat_resp.status_code == 200, (
        f"Chat completion failed after tokenize: "
        f"status={chat_resp.status_code}, body={chat_resp.text}"
    )
    chat_data = chat_resp.json()
    assert chat_data["choices"][0]["message"]["content"], (
        "Chat completion must produce non-empty content"
    )


@pytest.mark.asyncio
async def test_tokenize_then_chat_completion_with_image_async(
    client: openai.AsyncOpenAI,
    server: RemoteOpenAIServer,
    local_asset_server,
):
    """Async variant: tokenize then chat complete with the same image."""

    image_url = local_asset_server.url_for("stop_sign.jpg")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]

    # Step 1: tokenize
    tok_resp = requests.post(
        server.url_for("tokenize"),
        json={"model": MODEL_NAME, "messages": messages},
    )
    tok_resp.raise_for_status()

    # Step 2: chat completion via the OpenAI async client
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
    )

    assert chat_completion.choices[0].message.content, (
        "Chat completion must produce non-empty content after tokenize"
    )
