# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import httpx
import pytest

from tests.entrypoints.openai.chat_completion.test_serving_chat import (
    BASE_MODEL_PATHS,
    CHAT_TEMPLATE,
    MockHFConfig,
    MockModelConfig,
    _build_renderer,
)
from tests.entrypoints.openai.chat_completion.test_serving_chat import (
    MODEL_NAME as MOCK_MODEL_NAME,
)
from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.chat_completion.batch_serving import (
    OpenAIServingChatBatch,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.exceptions import VLLMValidationError
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.reasoning import ReasoningParserManager
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.v1.engine.async_llm import AsyncLLM

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


def _make_request_output(
    prompt_idx: int,
    text: str,
    prompt_token_ids: list[int] | None = None,
) -> RequestOutput:
    return RequestOutput(
        request_id=f"req-{prompt_idx}",
        prompt=None,
        prompt_token_ids=prompt_token_ids or [1, 2, 3],
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


async def _generator(
    prompt_idx: int,
    text: str,
    prompt_token_ids: list[int] | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    yield _make_request_output(prompt_idx, text, prompt_token_ids)


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("include_reasoning", [False, True])
async def test_batched_reasoning_metadata_follows_include_reasoning(
    include_reasoning: bool,
) -> None:
    """Hidden reasoning must not be recoverable from response metadata."""
    private_reasoning = "PRIVATE_REASONING"
    public_content = "PUBLIC_ANSWER"
    raw_output = f"<think>{private_reasoning}</think>{public_content}"

    result = _make_request_output(0, raw_output)
    result.outputs[0].logprobs = [
        {
            4: Logprob(
                logprob=-0.1,
                rank=1,
                decoded_token=f"<think>{private_reasoning}</think>",
            )
        },
        {
            5: Logprob(
                logprob=-0.1,
                rank=1,
                decoded_token=public_content,
            )
        },
    ]

    async def generator() -> AsyncGenerator[RequestOutput, None]:
        yield result

    serving = OpenAIServingChatBatch.__new__(OpenAIServingChatBatch)
    serving.response_role = "assistant"
    serving.system_fingerprint = None
    serving.return_tokens_as_token_ids = False

    parser = MagicMock()
    parser.parse.return_value = (private_reasoning, public_content, [])
    conversations = [[{"role": "user", "content": "Answer the probe."}]]
    request = BatchChatCompletionRequest(
        model="test-model",
        messages=conversations,
        include_reasoning=include_reasoning,
        logprobs=True,
        top_logprobs=0,
        return_token_ids=True,
    )

    response = await serving.chat_completion_full_generator_batch(
        request=request,
        generators=[generator()],
        request_id="req-reasoning-metadata",
        model_name="test-model",
        all_conversations=conversations,
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="req-reasoning-metadata"),
        parser=parser,
    )

    choice = response.choices[0]
    assert choice.message.content == public_content
    if include_reasoning:
        assert choice.message.reasoning == private_reasoning
        assert choice.logprobs is not None
        assert choice.logprobs.content is not None
        assert [item.token for item in choice.logprobs.content] == [
            f"<think>{private_reasoning}</think>",
            public_content,
        ]
        assert choice.token_ids == [4, 5]
    else:
        assert choice.message.reasoning is None
        assert choice.logprobs is None
        assert choice.token_ids is None


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_batched_echo_does_not_prepend_user_prompt() -> None:
    """Echo must only echo the assistant turn, never the user's prompt.

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


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_batched_harmony_response_format_uses_structural_tag() -> None:
    """Harmony must constrain only the final channel of every conversation."""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.model_config = MockModelConfig()
    mock_engine.errored = False
    mock_engine.input_processor = MagicMock()
    mock_engine.model_config.hf_config = MockHFConfig(model_type="gpt_oss")
    mock_engine.renderer = _build_renderer(mock_engine.model_config)
    mock_engine.generate.side_effect = lambda *args, **kwargs: _generator(0, "ok")

    models = OpenAIServingModels(mock_engine, BASE_MODEL_PATHS)
    online_renderer = OnlineRenderer(
        model_config=mock_engine.model_config,
        renderer=mock_engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        reasoning_parser="openai_gptoss",
    )
    serving = OpenAIServingChatBatch(
        mock_engine,
        models,
        response_role="assistant",
        online_renderer=online_renderer,
        chat_template=None,
        chat_template_content_format="auto",
        request_logger=None,
    )

    request = BatchChatCompletionRequest(
        model=MOCK_MODEL_NAME,
        messages=[
            [{"role": "user", "content": "France?"}],
            [{"role": "user", "content": "Japan?"}],
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "city", "schema": {"type": "object"}},
        },
    )
    await serving.create_batch_chat_completion(request)

    calls = mock_engine.generate.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.args[1].structured_outputs.structural_tag is not None


@pytest.mark.skip_global_cleanup
def test_batch_rejects_kv_transfer_prompt_token_ids():
    """One pre-tokenized prompt cannot stand in for every conversation."""
    with pytest.raises(
        VLLMValidationError, match=r"parameter=kv_transfer_params\.prompt_token_ids"
    ):
        BatchChatCompletionRequest(
            model="test-model",
            messages=[
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}],
            ],
            kv_transfer_params={"prompt_token_ids": [10, 20, 30]},
        )


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_batched_parser_receives_each_prompt_prefix() -> None:
    prefixes = []

    class RecordingParser:
        def __init__(self, *args, **kwargs):
            self.prompt_token_ids = None

        def set_prompt_token_ids(self, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

        def parse(self, model_output, **kwargs):
            prefixes.append(self.prompt_token_ids)
            return None, model_output, None

    serving = OpenAIServingChatBatch.__new__(OpenAIServingChatBatch)
    serving.response_role = "assistant"
    serving.system_fingerprint = None
    serving.chat_template = None
    serving.chat_template_content_format = "auto"
    serving.default_chat_template_kwargs = {}

    request = BatchChatCompletionRequest(
        model="test-model",
        messages=[
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
    )
    await serving.chat_completion_full_generator_batch(
        request=request,
        generators=[_generator(0, "one", [1]), _generator(1, "two", [2])],
        request_id="req-prefix",
        model_name="test-model",
        all_conversations=request.messages,
        tokenizer=object(),
        request_metadata=RequestResponseMetadata(request_id="req-prefix"),
        parser_cls=RecordingParser,
    )

    assert prefixes == [[1], [2]]


REASONING_PARSER = "kimi_k3"


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_batched_reasoning_wiring_matches_single_chat() -> None:
    """`/v1/chat/completions/batch` must hand the engine the same reasoning
    wiring as `/v1/chat/completions`.

    Both `reasoning_ended` and `reasoning_parser_kwargs` are carried on the
    request into `EngineCoreRequest`, where
    `StructuredOutputManager._get_reasoner()` rebuilds the request-local
    reasoning parser from the forwarded `chat_template_kwargs` -- "so the
    structured-output gate observes the same template kwargs used by the
    frontend". A divergence makes a template-kwarg-driven parser fall back to
    its default (`thinking` on) for a request that disabled thinking, so
    `is_reasoning_end()` keeps reporting False, the structured-output gate
    never opens, and the grammar is silently never applied.

    Both frontends run over one and the same model and renderer, so the only
    thing that can differ between the two calls is the serving code itself.
    """
    mock_model_config = MockModelConfig()
    shared_renderer = _build_renderer(mock_model_config)

    def _build_serving(cls):
        mock_engine = MagicMock(spec=AsyncLLM)
        mock_engine.model_config = mock_model_config
        mock_engine.errored = False
        mock_engine.input_processor = MagicMock()
        mock_engine.renderer = shared_renderer
        mock_engine.generate.side_effect = lambda *args, **kwargs: _generator(0, "ok")

        models = OpenAIServingModels(mock_engine, BASE_MODEL_PATHS)
        online_renderer = OnlineRenderer(
            model_config=mock_model_config,
            renderer=mock_engine.renderer,
            request_logger=None,
            chat_template=CHAT_TEMPLATE,
            chat_template_content_format="auto",
            reasoning_parser=REASONING_PARSER,
        )
        return mock_engine, cls(
            mock_engine,
            models,
            response_role="assistant",
            online_renderer=online_renderer,
            chat_template=CHAT_TEMPLATE,
            chat_template_content_format="auto",
            request_logger=None,
            reasoning_parser=REASONING_PARSER,
        )

    conversation = [{"role": "user", "content": "What is the capital of France?"}]
    chat_template_kwargs = {"enable_thinking": False}

    single_engine, single_serving = _build_serving(OpenAIServingChat)
    await single_serving.create_chat_completion(
        ChatCompletionRequest(
            model=MOCK_MODEL_NAME,
            messages=conversation,
            chat_template_kwargs=chat_template_kwargs,
        )
    )
    single_call = single_engine.generate.call_args_list[-1]

    batch_engine, batch_serving = _build_serving(OpenAIServingChatBatch)
    await batch_serving.create_batch_chat_completion(
        BatchChatCompletionRequest(
            model=MOCK_MODEL_NAME,
            messages=[conversation],
            chat_template_kwargs=chat_template_kwargs,
        )
    )
    batch_call = batch_engine.generate.call_args_list[-1]

    # Guard against a vacuous comparison: the reference path must be wired up.
    assert single_call.kwargs["reasoning_ended"] is not None
    assert single_call.kwargs["reasoning_parser_kwargs"] is not None

    assert batch_call.kwargs["reasoning_ended"] == single_call.kwargs["reasoning_ended"]
    assert (
        batch_call.kwargs["reasoning_parser_kwargs"]
        == single_call.kwargs["reasoning_parser_kwargs"]
    )

    # ...and the forwarded kwargs must actually turn thinking off for the
    # engine-side parser, i.e. the structured-output gate opens right away.
    engine_chat_template_kwargs = batch_call.kwargs["reasoning_parser_kwargs"][
        "chat_template_kwargs"
    ]
    reasoner = ReasoningParserManager.get_reasoning_parser(REASONING_PARSER)(
        tokenizer=batch_engine.renderer.tokenizer,
        chat_template_kwargs=engine_chat_template_kwargs,
    )
    assert reasoner.is_reasoning_end([]) is True
