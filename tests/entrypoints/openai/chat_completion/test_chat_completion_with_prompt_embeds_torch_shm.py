# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for `prompt_embeds` sent over `torch_shm` tensor IPC.

Kept apart from `test_chat_completion_with_prompt_embeds.py` so that its
server does not share the GPU with that module's server.
"""

import asyncio

import openai
import pytest
import pytest_asyncio
from transformers import AutoTokenizer

from tests.utils import RemoteOpenAIServer

from .test_chat_completion_with_prompt_embeds import (  # noqa: F401
    MODEL_NAME,
    _assert_text_and_prompt_embeds_match,
    aligned_content_and_embeds_b64,
    server_args,
)


@pytest.fixture(scope="module")
def torch_shm_server(server_args: list[str]):  # noqa: F811
    with RemoteOpenAIServer(
        MODEL_NAME,
        [*server_args, "--mm-tensor-ipc", "torch_shm"],
        env_dict={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
    ) as server:
        yield server


@pytest_asyncio.fixture
async def client(torch_shm_server: RemoteOpenAIServer):
    async with torch_shm_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_prompt_embeds_over_torch_shm_match_text(
    client: openai.AsyncOpenAI,
    aligned_content_and_embeds_b64: tuple[str, str],  # noqa: F811
):
    """With `--mm-tensor-ipc torch_shm`, `prompt_embeds` reach EngineCore
    through shared memory instead of the request message; the output must not
    change. The embeddings here require grad, which the queue cannot pickle
    unless the sender detaches them."""
    await _assert_text_and_prompt_embeds_match(client, aligned_content_and_embeds_b64)


@pytest.mark.asyncio
async def test_completions_prompt_embeds_over_torch_shm_match_tokens(
    client: openai.AsyncOpenAI,
    aligned_content_and_embeds_b64: tuple[str, str],  # noqa: F811
):
    """Completions load `prompt_embeds` through a different renderer path than
    chat; over `torch_shm` they must match the same prompt as token IDs."""
    content, encoded_embeds = aligned_content_and_embeds_b64
    token_ids = AutoTokenizer.from_pretrained(MODEL_NAME)(
        content, add_special_tokens=False
    ).input_ids

    tokens_resp, embeds_resp = await asyncio.gather(
        client.completions.create(
            model=MODEL_NAME, prompt=token_ids, max_tokens=10, temperature=0.0
        ),
        client.completions.create(
            model=MODEL_NAME,
            prompt=None,
            max_tokens=10,
            temperature=0.0,
            extra_body={"prompt_embeds": encoded_embeds},
        ),
    )

    assert len(tokens_resp.choices[0].text) > 0
    assert embeds_resp.choices[0].text == tokens_resp.choices[0].text
