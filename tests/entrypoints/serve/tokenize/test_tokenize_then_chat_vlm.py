# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test: calling ``/tokenize`` with multimodal data followed by
``/v1/chat/completions`` with the same data must not cause an error.

Ensures that the ``/tokenize`` endpoint does not pollute internal caches
(e.g. multimodal feature caches) and that a subsequent
``/v1/chat/completions`` request with the same multimodal payload
completes successfully.
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


@pytest.mark.asyncio
async def test_tokenize_then_chat_completion_with_image(
    client: openai.AsyncOpenAI,
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

    tok_resp = requests.post(
        server.url_for("tokenize"),
        json={"model": MODEL_NAME, "messages": messages},
    )
    tok_resp.raise_for_status()
    tok_data = tok_resp.json()
    assert tok_data["count"] > 0, "Tokenization must return tokens"

    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
    )

    assert chat_completion.choices[0].message.content, (
        "Chat completion must produce non-empty content after tokenize"
    )
