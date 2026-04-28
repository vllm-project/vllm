# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import httpx
import pytest

from tests.utils import RemoteOpenAIServer

# any model with a chat template defined in tokenizer_config should work here
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_batched_chat_completions(
    server: RemoteOpenAIServer, model_name: str
) -> None:
    conversations = [
        [{"role": "user", "content": "Reply with exactly the word: alpha"}],
        [{"role": "user", "content": "Reply with exactly the word: beta"}],
    ]

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{server.url_for('v1/chat/completions/batch')}",
            json={
                "model": model_name,
                "messages": conversations,
            },
            timeout=60,
        )

    assert response.status_code == 200, response.text
    data = response.json()

    choices = data["choices"]
    assert len(choices) == 2

    indices = {choice["index"] for choice in choices}
    assert indices == {0, 1}

    # Each conversation should produce a non-empty text response.
    for choice in choices:
        assert choice["message"]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_batched_chat_completions_with_json_schema(
    server: RemoteOpenAIServer, model_name: str
) -> None:
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["yes", "no"]},
        },
        "required": ["answer"],
    }
    conversations = [
        [{"role": "user", "content": "Is the sky blue? Answer in JSON."}],
        [{"role": "user", "content": "Is fire cold? Answer in JSON."}],
    ]

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{server.url_for('v1/chat/completions/batch')}",
            json={
                "model": model_name,
                "messages": conversations,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": schema, "strict": True},
                },
            },
            timeout=60,
        )

    assert response.status_code == 200, response.text
    data = response.json()

    choices = data["choices"]
    assert len(choices) == 2

    for choice in choices:
        parsed = json.loads(choice["message"]["content"])
        assert "answer" in parsed
        assert parsed["answer"] in ("yes", "no")
