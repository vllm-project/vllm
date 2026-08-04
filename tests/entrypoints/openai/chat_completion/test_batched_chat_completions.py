# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.openai.chat_completion import batch_serving
from vllm.entrypoints.openai.chat_completion.batch_serving import (
    OpenAIServingChatBatch,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
)

# any model with a chat template defined in tokenizer_config should work here
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.mark.parametrize(
    ("thinking_mode", "expected_reasoning_ended"),
    [
        ("adaptive", None),
        ("enabled", False),
        ("disabled", True),
    ],
)
def test_batch_forwards_prompt_reasoning_state(
    monkeypatch,
    thinking_mode: str,
    expected_reasoning_ended: bool | None,
) -> None:
    """Batch requests initialize reasoning like standard chat requests."""

    class BatchParser:
        def __init__(self, tokenizer, tools, chat_template_kwargs, **kwargs):
            self.reasoning_parser = object()
            self.thinking_mode = chat_template_kwargs["thinking_mode"]

        def is_reasoning_end_from_prompt(self, prompt_token_ids):
            return {
                "adaptive": None,
                "enabled": False,
                "disabled": True,
            }[self.thinking_mode]

    serving = OpenAIServingChatBatch.__new__(OpenAIServingChatBatch)
    serving.renderer = SimpleNamespace(tokenizer=MagicMock())
    serving.parser_cls = BatchParser
    serving.model_config = SimpleNamespace(max_model_len=128)
    serving.default_sampling_params = {}
    serving.override_max_tokens = None
    serving.engine_client = MagicMock()
    serving.engine_client.generate.return_value = MagicMock()
    serving.models = MagicMock()
    serving.models.model_name.return_value = "test-model"
    serving._effective_chat_template_kwargs = MagicMock(
        return_value={"thinking_mode": thinking_mode}
    )
    engine_prompt = {"type": "tokens", "prompt_token_ids": [1, 2, 3]}
    serving.render_batch_chat_request = AsyncMock(
        return_value=([[], []], [engine_prompt, engine_prompt])
    )
    serving._base_request_id = MagicMock(return_value="batch-test")
    serving._maybe_get_adapters = MagicMock(return_value=None)
    serving._get_data_parallel_rank = MagicMock(return_value=None)
    serving._extract_prompt_components = MagicMock(
        return_value=SimpleNamespace(token_ids=[1, 2, 3])
    )
    serving._extract_prompt_len = MagicMock(return_value=3)
    serving._log_inputs = MagicMock()
    serving.chat_completion_full_generator_batch = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(batch_serving, "get_max_tokens", lambda *args, **kwargs: 8)

    request = BatchChatCompletionRequest(
        model="test-model",
        messages=[
            [{"role": "user", "content": "Return JSON."}],
            [{"role": "user", "content": "Return JSON."}],
        ],
        response_format={"type": "json_object"},
        chat_template_kwargs={"thinking_mode": thinking_mode},
        max_tokens=8,
    )

    asyncio.run(serving.create_batch_chat_completion(request))

    assert serving.engine_client.generate.call_count == 2
    for generate_call in serving.engine_client.generate.call_args_list:
        generate_kwargs = generate_call.kwargs
        assert generate_kwargs["reasoning_ended"] is expected_reasoning_ended
        assert generate_kwargs["reasoning_parser_kwargs"] == {
            "chat_template_kwargs": {
                "thinking_mode": thinking_mode,
                "_vllm_continue_final_message": False,
            }
        }

    parsers = serving.chat_completion_full_generator_batch.call_args.args[-1]
    assert len(parsers) == 2
    assert parsers[0] is not parsers[1]


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_batched_chat_completions_logprobs_not_token_id_placeholders(
    server: RemoteOpenAIServer, model_name: str
) -> None:
    # Regression test: requesting `return_token_ids` alongside logprobs must not
    # corrupt the logprob `token` fields into "token_id:{id}" placeholders. That
    # placeholder rendering is controlled by `return_tokens_as_token_ids`, which
    # this request leaves unset.
    conversations = [
        [{"role": "user", "content": "Reply with exactly the word: alpha"}],
    ]

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{server.url_for('v1/chat/completions/batch')}",
            json={
                "model": model_name,
                "messages": conversations,
                "logprobs": True,
                "top_logprobs": 1,
                "return_token_ids": True,
            },
            timeout=60,
        )

    assert response.status_code == 200, response.text
    data = response.json()

    content = data["choices"][0]["logprobs"]["content"]
    assert content
    for entry in content:
        assert not entry["token"].startswith("token_id:")
        for top in entry["top_logprobs"]:
            assert not top["token"].startswith("token_id:")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_batched_chat_completions_return_tokens_as_token_ids(
    server: RemoteOpenAIServer, model_name: str
) -> None:
    # Complementary check: when `return_tokens_as_token_ids` is explicitly set,
    # the logprob tokens *should* be rendered as "token_id:{id}" placeholders,
    # proving the new field is actually wired through.
    conversations = [
        [{"role": "user", "content": "Reply with exactly the word: alpha"}],
    ]

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{server.url_for('v1/chat/completions/batch')}",
            json={
                "model": model_name,
                "messages": conversations,
                "logprobs": True,
                "top_logprobs": 1,
                "return_tokens_as_token_ids": True,
            },
            timeout=60,
        )

    assert response.status_code == 200, response.text
    data = response.json()

    content = data["choices"][0]["logprobs"]["content"]
    assert content
    assert all(entry["token"].startswith("token_id:") for entry in content)


@pytest.mark.asyncio
async def test_batched_chat_completions_logprob_token_ids(
    server: RemoteOpenAIServer,
) -> None:
    conversations = [[{"role": "user", "content": "Hello"}]]

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{server.url_for('v1/chat/completions/batch')}",
            json={
                "model": MODEL_NAME,
                "messages": conversations,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": 5,
                "logprob_token_ids": [100, 1000, 5000],
                "return_tokens_as_token_ids": True,
            },
            timeout=60,
        )

    assert response.status_code == 200, response.text
    content = response.json()["choices"][0]["logprobs"]["content"]
    assert content
    sampled_token = content[0]["token"]
    assert {entry["token"] for entry in content[0]["top_logprobs"]} == {
        "token_id:100",
        "token_id:1000",
        "token_id:5000",
        sampled_token,
    }
