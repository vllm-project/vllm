# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Non-object JSON bodies must fail validation cleanly (4xx), not AttributeError (500).

mode=before validators that call data.get(...) without an isinstance(data, dict)
guard raise AttributeError for string/list/scalar bodies and surface as HTTP 500.

This extends the chat completion coverage added in #51654 to the remaining
request models whose before-validators were missing the same guard.
"""

from typing import Any

import pytest
from openai_harmony import Message, Role
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import TypedDict

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.generate.generative_scoring.serving import (
    GenerativeScoringRequest,
)
from vllm.entrypoints.generate.structured_decisions.protocol import (
    StructuredDecisionRequest,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.pooling.classify.protocol import ClassificationChatRequest
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingChatRequest,
    EmbeddingRequest,
)
from vllm.entrypoints.pooling.pooling.protocol import PoolingChatRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest
from vllm.entrypoints.serve.utils.request_validation import FailFastRequestMixin
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionRequest
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationRequest
from vllm.exceptions import VLLMValidationError

pytestmark = pytest.mark.skip_global_cleanup

REQUEST_MODELS = [
    GenerativeScoringRequest,
    CompletionRequest,
    ResponsesRequest,
    EmbeddingChatRequest,
    ClassificationChatRequest,
    PoolingChatRequest,
    TokenizeChatRequest,
    TranscriptionRequest,
    TranslationRequest,
]


@pytest.mark.parametrize("request_model", REQUEST_MODELS, ids=lambda m: m.__name__)
@pytest.mark.parametrize(
    "payload",
    [
        "this is not valid json{{{",
        ["not", "an", "object"],
        42,
        None,
        True,
    ],
)
def test_request_models_reject_non_object_body(request_model, payload):
    with pytest.raises(ValidationError):
        request_model.model_validate(payload)


def test_completion_request_still_validates_dict_bodies():
    """The guard must not swallow real field-level errors on object bodies."""
    with pytest.raises(VLLMValidationError, match="prompt"):
        CompletionRequest.model_validate({"model": "qwen", "prompt": ""})


def test_tokenize_chat_request_still_validates_dict_bodies():
    with pytest.raises(VLLMValidationError, match="add_generation_prompt"):
        TokenizeChatRequest.model_validate(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "continue_final_message": True,
                "add_generation_prompt": True,
            }
        )


CACHE_SALT_MODELS = [
    (
        GenerativeScoringRequest,
        {
            "query": "Is this city the capital of France?",
            "items": ["Paris"],
            "label_token_ids": [9454, 2753],
        },
    ),
    (
        StructuredDecisionRequest,
        {
            "state": "hello",
            "questions": {
                "lang": {
                    "type": "choice",
                    "criteria": {"English": "Text written in English."},
                }
            },
        },
    ),
    (CompletionRequest, {"prompt": "hello"}),
    (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hello"}]}),
    (ResponsesRequest, {"input": "hello"}),
    (GenerateRequest, {"token_ids": [1], "sampling_params": {}}),
]


@pytest.mark.parametrize(
    ("request_model", "payload"),
    CACHE_SALT_MODELS,
    ids=[m.__name__ for m, _ in CACHE_SALT_MODELS],
)
def test_request_models_reject_forbidden_cache_salt(request_model, payload):
    with pytest.raises(VLLMValidationError, match="cache_salt"):
        request_model.model_validate({**payload, "cache_salt": "/"})


def test_request_models_reject_overlong_cache_salt():
    with pytest.raises(VLLMValidationError, match="cache_salt"):
        CompletionRequest.model_validate({"prompt": "hello", "cache_salt": "a" * 129})


def test_request_models_accept_safe_cache_salt():
    request = CompletionRequest.model_validate(
        {"prompt": "hello", "cache_salt": "safe-cache_salt-123"}
    )
    assert request.cache_salt == "safe-cache_salt-123"


@pytest.mark.parametrize(
    "request_type",
    [
        ChatCompletionRequest,
        TokenizeChatRequest,
        EmbeddingChatRequest,
        ClassificationChatRequest,
        PoolingChatRequest,
    ],
    ids=lambda model: model.__name__,
)
def test_chat_message_validation_stops_at_first_invalid_item(request_type):
    valid = {"role": "user", "content": "Hello"}
    invalid: dict[str, Any] = {}
    assert request_type.model_validate({"messages": [valid, valid]}).messages == [
        valid,
        valid,
    ]

    with pytest.raises(ValidationError) as exc_info:
        request_type.model_validate({"messages": [valid, invalid, invalid]})

    assert {error["loc"][:2] for error in exc_info.value.errors()} == {("messages", 1)}


def test_anthropic_message_content_validation_stops_at_first_invalid_item():
    request = {
        "model": "test-model",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": [{}, {}]}],
    }
    with pytest.raises(ValidationError) as exc_info:
        AnthropicMessagesRequest.model_validate(request)

    block_locations = {
        error["loc"][-2:]
        for error in exc_info.value.errors()
        if error["loc"][-1] == "type"
    }
    assert block_locations == {(0, "type")}


def test_typed_request_mapping_validation_stops_at_first_invalid_value():
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "logit_bias": {"1": "invalid", "2": "invalid"},
            }
        )

    assert [error["loc"] for error in exc_info.value.errors()] == [("logit_bias", "1")]


def test_embedding_batch_validation_preserves_sequence_inputs():
    message = {"role": "user", "content": "Hello"}
    adapter: TypeAdapter[Any] = TypeAdapter(EmbeddingRequest)

    request = adapter.validate_python({"messages": ([message], [message])})
    assert request.messages == ([message], [message])

    with pytest.raises(ValidationError) as exc_info:
        adapter.validate_python({"messages": ([message], [], [])})

    batch_errors = {
        error["loc"][-2:]
        for error in exc_info.value.errors()
        if error["type"] == "too_short"
    }
    assert batch_errors == {("messages", 1)}


@pytest.mark.parametrize("guarded_first", [False, True])
def test_fail_fast_validation_keeps_shared_schema_references_independent(
    guarded_first,
):
    class Items(TypedDict):
        numbers: list[int]

    class Plain(BaseModel):
        items: Items

    class Guarded(FailFastRequestMixin, BaseModel):
        items: Items

    adapter = TypeAdapter(
        tuple[Guarded, Plain] if guarded_first else tuple[Plain, Guarded]
    )
    item = {"items": {"numbers": ["a", "b"]}}

    with pytest.raises(ValidationError) as exc_info:
        adapter.validate_python((item, item))

    error_locations = {error["loc"] for error in exc_info.value.errors()}
    plain_index = int(guarded_first)
    assert error_locations == {
        (0, "items", "numbers", 0),
        (1, "items", "numbers", 0),
        (plain_index, "items", "numbers", 1),
    }


def test_harmony_request_sdk_validation_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = Message.from_role_and_content(Role.USER, "Hello")
    raw = message.model_dump()
    (parsed,) = ChatCompletionRequest.model_validate({"messages": [raw]}).messages
    assert isinstance(parsed, Message)
    assert parsed.author == message.author

    monkeypatch.setattr(
        "vllm.entrypoints.serve.utils.request_validation._MAX_SDK_VALIDATION_ENTRIES",
        1,
    )
    assert (
        ChatCompletionRequest.model_validate({"messages": [message]}).messages[0]
        is message
    )
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest.model_validate({"messages": [raw]})

    assert any(
        error["type"] == "sdk_validation_budget" for error in exc_info.value.errors()
    )
