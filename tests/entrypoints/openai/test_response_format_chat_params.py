# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for top-level response_format forwarding into ChatParams.

Most models treat response_format purely as a decoding constraint (guided
decoding via structured_outputs), so it is not part of chat template
rendering. Models whose chat encoding renders response_format into the
prompt need the field at the renderer layer; the forwarding is inert for
renderers that don't read it.
"""

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

    def test_strict_false_keeps_explicit_structured_outputs(self):
        # strict=false contributes no decoding constraint, so an explicit
        # structured_outputs coexists with it without conflict.
        from vllm.sampling_params import StructuredOutputsParams

        request = self._build_responses_request(False)
        explicit = StructuredOutputsParams(regex=r"\d+")
        request = request.model_copy(update={"structured_outputs": explicit})
        params = request.to_sampling_params(default_max_tokens=100)
        assert params.structured_outputs is explicit

    def test_strict_true_keeps_guided_decoding(self):
        assert self._guided_json(self._build_responses_request(True)) == (
            _WEATHER_SCHEMA
        )

    def test_strict_absent_keeps_guided_decoding(self):
        assert self._guided_json(self._build_responses_request(...)) == (
            _WEATHER_SCHEMA
        )
