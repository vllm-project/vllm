# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for response_format handling shared by the chat APIs.

``strict=false`` asks for schema adherence without a guarantee, so no grammar
constraint is applied while the schema is still forwarded to the renderer.
``strict=true`` and an absent ``strict`` keep guided decoding. Responses
``text.format`` reaches renderers in the same shape as the Chat Completions
``response_format``.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

_WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}

_ABSENT = object()


def _build_chat_request(**kwargs) -> ChatCompletionRequest:
    defaults = dict(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _build_responses_request(**kwargs) -> ResponsesRequest:
    defaults = dict(
        model="test-model",
        input=[{"role": "user", "content": "Hello"}],
    )
    defaults.update(kwargs)
    return ResponsesRequest(**defaults)


def _chat_json_schema_request(strict=_ABSENT) -> ChatCompletionRequest:
    json_schema: dict = {"name": "weather", "schema": _WEATHER_SCHEMA}
    if strict is not _ABSENT:
        json_schema["strict"] = strict
    return _build_chat_request(
        response_format={"type": "json_schema", "json_schema": json_schema}
    )


def _responses_json_schema_request(strict=_ABSENT) -> ResponsesRequest:
    fmt: dict = {
        "type": "json_schema",
        "name": "weather",
        "schema": _WEATHER_SCHEMA,
    }
    if strict is not _ABSENT:
        fmt["strict"] = strict
    return _build_responses_request(text={"format": fmt})


def _guided_json(structured_outputs) -> dict | None:
    if structured_outputs is None:
        return None
    return structured_outputs.json


class TestChatCompletionStrict:
    def _guided_json(self, request: ChatCompletionRequest) -> dict | None:
        params = request.to_sampling_params(max_tokens=100, default_sampling_params={})
        return _guided_json(params.structured_outputs)

    def test_strict_false_disables_guided_decoding(self):
        assert self._guided_json(_chat_json_schema_request(strict=False)) is None

    def test_strict_true_keeps_guided_decoding(self):
        assert self._guided_json(_chat_json_schema_request(strict=True)) == (
            _WEATHER_SCHEMA
        )

    def test_strict_absent_keeps_guided_decoding(self):
        assert self._guided_json(_chat_json_schema_request()) == _WEATHER_SCHEMA

    def test_strict_false_keeps_existing_structured_outputs(self):
        request = _build_chat_request(
            structured_outputs={"choice": ["a", "b"]},
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "schema": _WEATHER_SCHEMA,
                    "strict": False,
                },
            },
        )
        params = request.to_sampling_params(max_tokens=100, default_sampling_params={})
        assert params.structured_outputs is not None
        assert params.structured_outputs.json is None
        assert params.structured_outputs.choice == ["a", "b"]

    def test_strict_false_still_forwards_schema_to_renderer(self):
        params = _chat_json_schema_request(strict=False).build_chat_params(None, "auto")
        assert params.response_format is not None
        assert params.response_format.json_schema is not None
        assert params.response_format.json_schema.json_schema == _WEATHER_SCHEMA
        assert params.response_format.json_schema.strict is False


class TestResponsesStrict:
    def _guided_json(self, request: ResponsesRequest) -> dict | None:
        params = request.to_sampling_params(default_max_tokens=100)
        return _guided_json(params.structured_outputs)

    def test_strict_false_disables_guided_decoding(self):
        assert self._guided_json(_responses_json_schema_request(strict=False)) is None

    def test_strict_true_keeps_guided_decoding(self):
        assert self._guided_json(_responses_json_schema_request(strict=True)) == (
            _WEATHER_SCHEMA
        )

    def test_strict_absent_keeps_guided_decoding(self):
        assert self._guided_json(_responses_json_schema_request()) == _WEATHER_SCHEMA


class TestResponsesResponseFormatForwarding:
    def _responses_chat_params(self, **kwargs):
        return _build_responses_request(**kwargs).build_chat_params(None, "auto")

    @pytest.mark.parametrize("strict", [True, False, _ABSENT])
    def test_json_schema_matches_chat_completions(self, strict):
        chat = _chat_json_schema_request(strict=strict).build_chat_params(None, "auto")
        responses = _responses_json_schema_request(strict=strict).build_chat_params(
            None, "auto"
        )
        assert responses.response_format is not None
        assert responses.response_format == chat.response_format

    @pytest.mark.parametrize("format_type", ["json_object", "text"])
    def test_schemaless_format_matches_chat_completions(self, format_type):
        chat = _build_chat_request(
            response_format={"type": format_type}
        ).build_chat_params(None, "auto")
        responses = self._responses_chat_params(text={"format": {"type": format_type}})
        assert responses.response_format is not None
        assert responses.response_format == chat.response_format

    def test_no_text_format_not_forwarded(self):
        assert self._responses_chat_params().response_format is None
        assert self._responses_chat_params(text={}).response_format is None
