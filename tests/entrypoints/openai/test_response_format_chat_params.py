# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for top-level response_format forwarding into ChatParams.

Most models treat response_format purely as a decoding constraint (guided
decoding via structured_outputs), so it is not part of chat template
rendering. Models whose chat encoding renders response_format into the
prompt need the field at the renderer layer; the forwarding is inert for
renderers that don't read it.
"""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)

_WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


def _build_chat_request(**kwargs) -> ChatCompletionRequest:
    defaults = dict(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


class TestResponseFormatForwarding:
    def test_json_schema_forwarded(self):
        request = _build_chat_request(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "weather", "schema": _WEATHER_SCHEMA},
            },
        )
        params = request.build_chat_params(None, "auto")
        rf = params.response_format
        assert rf is not None
        assert rf.type == "json_schema"
        assert rf.json_schema is not None
        assert rf.json_schema.name == "weather"
        assert rf.json_schema.json_schema == _WEATHER_SCHEMA

    def test_json_object_forwarded(self):
        request = _build_chat_request(
            response_format={"type": "json_object"},
        )
        params = request.build_chat_params(None, "auto")
        assert params.response_format is not None
        assert params.response_format.type == "json_object"

    def test_no_response_format_not_injected(self):
        request = _build_chat_request()
        params = request.build_chat_params(None, "auto")
        assert params.response_format is None

    def test_explicit_user_kwarg_kept_when_no_top_level_field(self):
        # A client-provided chat_template_kwargs entry survives when the
        # top-level field is absent.
        request = _build_chat_request(
            chat_template_kwargs={"response_format": {"type": "json_object"}},
        )
        params = request.build_chat_params(None, "auto")
        assert params.response_format is None
        assert params.chat_template_kwargs["response_format"] == {"type": "json_object"}

    def test_guided_decoding_still_set(self):
        # Forwarding must not eat the structured_outputs path.
        request = _build_chat_request(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "weather", "schema": _WEATHER_SCHEMA},
            },
        )
        params = request.to_sampling_params(max_tokens=100, default_sampling_params={})
        assert params.structured_outputs is not None
        assert params.structured_outputs.json == _WEATHER_SCHEMA


class TestStrictFalseDisablesGuidedDecoding:
    """strict=false means prompt-instruction-only: the schema is forwarded
    to the renderer, but no grammar constraint is applied. strict=true/absent
    keeps full guided decoding."""

    def _guided_json(self, request) -> dict | None:
        params = request.to_sampling_params(max_tokens=100, default_sampling_params={})
        if params.structured_outputs is None:
            return None
        return params.structured_outputs.json

    def _chat_request(self, strict):
        json_schema = {"name": "weather", "schema": _WEATHER_SCHEMA}
        if strict is not ...:
            json_schema["strict"] = strict
        return _build_chat_request(
            response_format={"type": "json_schema", "json_schema": json_schema}
        )

    def test_strict_false_disables_guided_decoding(self):
        assert self._guided_json(self._chat_request(strict=False)) is None

    def test_strict_true_keeps_guided_decoding(self):
        assert self._guided_json(self._chat_request(strict=True)) == (_WEATHER_SCHEMA)

    def test_strict_absent_keeps_guided_decoding(self):
        # Compatibility: clients that never pass strict must not silently
        # lose grammar constraints.
        assert self._guided_json(self._chat_request(strict=...)) == (_WEATHER_SCHEMA)

    def test_strict_false_still_forwards_to_renderer(self):
        # The prompt-instruction channel must stay active when the grammar
        # channel is off -- together they form "prompt-only" mode.
        request = self._chat_request(strict=False)
        params = request.build_chat_params(None, "auto")
        rf = params.response_format
        assert rf is not None and rf.json_schema is not None
        assert rf.json_schema.json_schema == _WEATHER_SCHEMA
        assert rf.json_schema.strict is False


class TestStrictMustBeBoolean:
    """Non-boolean `strict` is rejected at request validation (the server
    maps the ValidationError to 400 BadRequest) instead of being
    lax-coerced to a bool (e.g. "yes" -> True)."""

    @pytest.mark.parametrize("bad_strict", ["yes", "true", "false", 1, 0])
    def test_non_boolean_strict_rejected(self, bad_strict):
        with pytest.raises(ValidationError):
            _build_chat_request(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "weather",
                        "schema": _WEATHER_SCHEMA,
                        "strict": bad_strict,
                    },
                },
            )

    @pytest.mark.parametrize("good_strict", [True, False, None])
    def test_boolean_strict_accepted(self, good_strict):
        request = _build_chat_request(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "schema": _WEATHER_SCHEMA,
                    "strict": good_strict,
                },
            },
        )
        assert request.response_format is not None
        assert request.response_format.json_schema is not None
        assert request.response_format.json_schema.strict is good_strict


class TestResponsesStrictFalse:
    def _build_responses_request(self, strict):
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

        fmt = {
            "type": "json_schema",
            "name": "weather",
            "schema": _WEATHER_SCHEMA,
        }
        if strict is not ...:
            fmt["strict"] = strict
        return ResponsesRequest(
            model="test-model",
            input=[{"role": "user", "content": "Hello"}],
            text={"format": fmt},
        )

    def _guided_json(self, request) -> dict | None:
        params = request.to_sampling_params(default_max_tokens=100)
        if params.structured_outputs is None:
            return None
        return params.structured_outputs.json

    def test_strict_false_disables_guided_decoding(self):
        assert self._guided_json(self._build_responses_request(False)) is None

    def test_strict_true_keeps_guided_decoding(self):
        assert self._guided_json(self._build_responses_request(True)) == (
            _WEATHER_SCHEMA
        )

    def test_strict_absent_keeps_guided_decoding(self):
        assert self._guided_json(self._build_responses_request(...)) == (
            _WEATHER_SCHEMA
        )


class TestResponsesResponseFormatForwarding:
    """Responses API: text.format is flattened (no `json_schema` nesting)
    and must be re-nested to the Chat Completions shape for renderers."""

    def _build_responses_request(self, **kwargs):
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

        defaults = dict(
            model="test-model",
            input=[{"role": "user", "content": "Hello"}],
        )
        defaults.update(kwargs)
        return ResponsesRequest(**defaults)

    def test_json_schema_renested(self):
        request = self._build_responses_request(
            text={
                "format": {
                    "type": "json_schema",
                    "name": "weather",
                    "schema": _WEATHER_SCHEMA,
                    "strict": True,
                }
            },
        )
        params = request.build_chat_params(None, "auto")
        assert params.response_format == {
            "type": "json_schema",
            "json_schema": {
                "name": "weather",
                "description": None,
                "schema": _WEATHER_SCHEMA,
                "strict": True,
            },
        }

    def test_json_object_forwarded(self):
        request = self._build_responses_request(
            text={"format": {"type": "json_object"}},
        )
        params = request.build_chat_params(None, "auto")
        assert params.response_format == {
            "type": "json_object",
            "json_schema": None,
        }

    def test_no_text_format_not_injected(self):
        request = self._build_responses_request()
        params = request.build_chat_params(None, "auto")
        assert params.response_format is None
