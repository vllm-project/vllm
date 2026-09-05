# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import AsyncGenerator

import httpx
import pytest

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.chat_completion.batch_serving import (
    OpenAIServingChatBatch,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
)
from vllm.outputs import CompletionOutput, RequestOutput

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


def _make_request_output(prompt_idx: int, text: str) -> RequestOutput:
    return RequestOutput(
        request_id=f"req-{prompt_idx}",
        prompt=None,
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[4, 5],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        finished=True,
    )


async def _generator(prompt_idx: int, text: str) -> AsyncGenerator[RequestOutput, None]:
    yield _make_request_output(prompt_idx, text)


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_batched_echo_does_not_prepend_user_prompt() -> None:
    """echo must only echo the assistant turn, never the user's prompt.

    With `add_generation_prompt` (the default) the response role is `assistant`
    while the last conversation message is the user's, so there is nothing to
    echo; without the role check the user prompt leaks into the answer.
    """
    serving = OpenAIServingChatBatch.__new__(OpenAIServingChatBatch)
    serving.response_role = "assistant"
    serving.system_fingerprint = None

    conversations = [[{"role": "user", "content": "USER PROMPT"}]]
    request = BatchChatCompletionRequest(
        model="test-model",
        messages=conversations,
        echo=True,
    )

    response = await serving.chat_completion_full_generator_batch(
        request=request,
        generators=[_generator(0, "ASSISTANT ANSWER")],
        request_id="req-echo",
        model_name="test-model",
        all_conversations=conversations,
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="req-echo"),
    )

    assert response.choices[0].message.content == "ASSISTANT ANSWER"


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_batched_echo_prepends_matching_assistant_prefix() -> None:
    """A trailing assistant turn is a real prefix and must still be echoed."""
    serving = OpenAIServingChatBatch.__new__(OpenAIServingChatBatch)
    serving.response_role = "assistant"
    serving.system_fingerprint = None

    conversations = [
        [
            {"role": "user", "content": "USER PROMPT"},
            {"role": "assistant", "content": "PREFIX "},
        ]
    ]
    request = BatchChatCompletionRequest(
        model="test-model",
        messages=conversations,
        echo=True,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    response = await serving.chat_completion_full_generator_batch(
        request=request,
        generators=[_generator(0, "ASSISTANT ANSWER")],
        request_id="req-echo-2",
        model_name="test-model",
        all_conversations=conversations,
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="req-echo-2"),
    )

    assert response.choices[0].message.content == "PREFIX ASSISTANT ANSWER"
