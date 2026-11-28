# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer


@pytest.fixture(scope="module")
def server():
    # Qwen3.5's chat template rejects any system message after the first.
    args = ["--max-model-len", "2048", "--enforce-eager", "--language-model-only"]
    with RemoteOpenAIServer("Qwen/Qwen3.5-0.8B", args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_instructions_with_system_input_item(client: openai.AsyncOpenAI):
    """`instructions` plus a leading system input item must render as one
    system message for templates that accept only one."""
    response = await client.responses.create(
        instructions="Finish the answer with QED.",
        input=[
            {"role": "system", "content": "Answer with only the number."},
            {"role": "user", "content": "What is 13 * 24?"},
        ],
        max_output_tokens=16,
        temperature=0.0,
    )
    assert response.status in ("completed", "incomplete")
    assert response.output
