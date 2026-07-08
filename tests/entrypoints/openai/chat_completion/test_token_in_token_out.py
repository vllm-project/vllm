# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-tokenized input (token-in, text-out) on the chat completions API."""

import os
import tempfile

import openai
import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf

MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "qwen3_06b_chat_token_in")

MESSAGES = [{"role": "user", "content": "Hello, how are you today?"}]


@pytest.fixture(scope="module")
def server():
    global MODEL_PATH
    # Chat token-in still detokenizes the output, so unlike the completion
    # token-in test we keep the tokenizer and only skip the weights.
    MODEL_PATH = download_weights_from_hf(
        MODEL_NAME,
        allow_patterns=["*"],
        cache_dir=MODEL_PATH,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt"],
    )
    args = [
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "32",
        "--enforce-eager",
        "--load-format",
        "dummy",
    ]
    with RemoteOpenAIServer(MODEL_PATH, args) as remote_server:
        yield remote_server


def _render_prompt_token_ids(server: RemoteOpenAIServer) -> list[int]:
    """Ask the server for the token ids it would produce for MESSAGES."""
    resp = requests.post(
        server.url_for("tokenize"),
        json={"model": MODEL_PATH, "messages": MESSAGES},
    )
    resp.raise_for_status()
    return resp.json()["tokens"]


@pytest.mark.asyncio
async def test_chat_token_in_matches_messages(server):
    """prompt_token_ids drives generation identically to the templated path."""
    prompt_token_ids = _render_prompt_token_ids(server)

    async with server.get_async_client() as client:
        baseline = await client.chat.completions.create(
            model=MODEL_PATH,
            messages=MESSAGES,
            max_completion_tokens=16,
            temperature=0,
        )
        token_in = await client.chat.completions.create(
            model=MODEL_PATH,
            messages=[],
            max_completion_tokens=16,
            temperature=0,
            extra_body={
                "prompt_token_ids": prompt_token_ids,
                "return_token_ids": True,
            },
        )

    # text-out: pre-tokenized input still yields a detokenized message.
    assert token_in.choices[0].message.content is not None
    # the engine saw exactly the ids we supplied.
    assert token_in.prompt_token_ids == prompt_token_ids
    # equivalence: identical prompt tokens -> identical greedy completion.
    assert token_in.choices[0].message.content == baseline.choices[0].message.content


IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}
        ],
    }
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages, extra_options",
    [
        # A chat-template option cannot apply to pre-tokenized input.
        ([], {"add_generation_prompt": True}),
        # Token ids carry no multimodal features.
        (IMAGE_MESSAGES, {}),
    ],
    ids=["template-option", "multimodal-content"],
)
async def test_chat_token_in_rejects_incompatible_input(
    server, messages, extra_options
):
    """Options and message content that cannot apply are rejected, not ignored."""
    prompt_token_ids = _render_prompt_token_ids(server)
    async with server.get_async_client() as client:
        with pytest.raises(openai.BadRequestError):
            await client.chat.completions.create(
                model=MODEL_PATH,
                messages=messages,
                max_completion_tokens=4,
                extra_body={"prompt_token_ids": prompt_token_ids, **extra_options},
            )
