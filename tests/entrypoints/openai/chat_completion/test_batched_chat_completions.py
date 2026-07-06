# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.openai.chat_completion.batch_serving import (
    OpenAIServingChatBatch,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
)

# any model with a chat template defined in tokenizer_config should work here
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class _FakeRender:
    use_harmony = False
    chat_template = None
    chat_template_content_format = "auto"
    default_chat_template_kwargs: dict = {}
    tool_parser = None
    trust_request_chat_template = False

    def __init__(self):
        self.preprocessed_requests = []
        self.reasoning_parser_args = []

    def validate_chat_template(self, **kwargs):
        return None

    async def preprocess_chat(self, single_request, messages, **kwargs):
        self.preprocessed_requests.append(single_request)
        self.reasoning_parser_args.append(kwargs["reasoning_parser"])
        single_request.skip_special_tokens = False
        return messages, [{"prompt_token_ids": [1, 2]}]


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
async def test_batch_render_uses_adjusted_reasoning_requests() -> None:
    request = BatchChatCompletionRequest(
        model="test-model",
        messages=[
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ],
    )
    reasoning_parser_cls = object()

    handler = object.__new__(OpenAIServingChatBatch)
    handler._check_model = AsyncMock(return_value=None)
    handler.engine_client = SimpleNamespace(errored=False)
    handler.openai_serving_render = _FakeRender()
    handler.reasoning_parser_cls = reasoning_parser_cls

    result = await handler.render_batch_chat_request(request)

    conversations, engine_prompts, adjusted_requests = result
    assert conversations == request.messages
    assert engine_prompts == [{"prompt_token_ids": [1, 2]}] * 2
    assert handler.openai_serving_render.preprocessed_requests == adjusted_requests
    assert handler.openai_serving_render.reasoning_parser_args == [
        reasoning_parser_cls,
        reasoning_parser_cls,
    ]
    assert [r.messages for r in adjusted_requests] == request.messages
    assert [r.skip_special_tokens for r in adjusted_requests] == [False, False]


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
