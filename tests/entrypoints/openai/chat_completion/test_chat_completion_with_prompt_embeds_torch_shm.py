# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E test for `prompt_embeds` content parts sent over `torch_shm` tensor IPC.

Kept apart from `test_chat_completion_with_prompt_embeds.py` so that its
server does not share the GPU with that module's server.
"""

import pytest

from tests.utils import RemoteOpenAIServer

from .test_chat_completion_with_prompt_embeds import (  # noqa: F401
    MODEL_NAME,
    _assert_text_and_prompt_embeds_match,
    aligned_content_and_embeds_b64,
    server_args,
)


@pytest.mark.asyncio
async def test_prompt_embeds_over_torch_shm_match_text(
    server_args: list[str],  # noqa: F811
    aligned_content_and_embeds_b64: tuple[str, str],  # noqa: F811
):
    """With `--mm-tensor-ipc torch_shm`, `prompt_embeds` reach EngineCore
    through shared memory instead of the request message; the output must not
    change. The embeddings here require grad, which the queue cannot pickle
    unless the sender detaches them."""
    with RemoteOpenAIServer(
        MODEL_NAME,
        [*server_args, "--mm-tensor-ipc", "torch_shm"],
        env_dict={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
    ) as server:
        async with server.get_async_client() as client:
            await _assert_text_and_prompt_embeds_match(
                client, aligned_content_and_embeds_b64
            )
