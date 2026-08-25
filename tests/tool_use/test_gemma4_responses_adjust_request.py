# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Responses API tool-calling request adjustment.

Covers two bugs on the ``/v1/responses`` path that broke streaming tool
calling for parsers relying on special-token delimiters (Gemma4):

1. :class:`Gemma4ToolParser.adjust_request` used an
   ``isinstance(request, ChatCompletionRequest)`` guard, so a
   :class:`ResponsesRequest` with tools never had
   ``skip_special_tokens`` flipped to ``False``. The default (``True``)
   stripped ``<|tool_call>`` / ``<tool_call|>`` delimiters, causing
   :meth:`Gemma4ToolParser.extract_tool_calls_streaming` to fall through
   to the content branch and leak the raw ``call:fn{...}`` body via
   ``response.output_text.delta``.

2. :meth:`ToolParser.adjust_request` built
   :class:`ResponseTextConfig` in two steps (bare constructor then
   ``.format = ...``). Under Pydantic v2 the later assignment is not
   tracked in ``__fields_set__``, which can drop the nested config from
   ``model_dump``. It also passed a ``description`` kwarg carrying the
   wrong-purpose string ``"Response format for tool calling"``.

3. :class:`Gemma4EngineToolParser` (the engine-based parser, #45588) sets
   ``supports_required_and_named=False`` but did not skip the forced
   ``structured_outputs`` JSON for ``required``/named tool choice. The model
   was constrained to JSON the native parser cannot read, so the call leaked
   as content with empty ``tool_calls``. ``adjust_request`` now skips that
   constraint so Gemma4 emits its native ``<|tool_call>`` syntax.
"""

from __future__ import annotations

from typing import Any

from openai.types.responses.tool_param import FunctionToolParam

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.gemma4_engine_tool_parser import (
    Gemma4EngineToolParser as Gemma4ToolParser,
)


def _get_weather_tool() -> FunctionToolParam:
    return FunctionToolParam(
        type="function",
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        strict=True,
    )


def _build_responses_request(*, tool_choice: str | dict[str, Any]) -> ResponsesRequest:
    return ResponsesRequest(
        model="gemma4-test",
        input=[{"role": "user", "content": "What is the weather in Hanoi?"}],
        tools=[_get_weather_tool()],
        tool_choice=tool_choice,
        stream=True,
        max_output_tokens=200,
    )


def _build_chat_request(
    *,
    tool_choice: str | dict[str, Any],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> ChatCompletionRequest:
    data: dict[str, Any] = {
        "model": "gemma4-test",
        "messages": [{"role": "user", "content": "What is the weather in Hanoi?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": tool_choice,
    }
    if chat_template_kwargs is not None:
        data["chat_template_kwargs"] = chat_template_kwargs
    return ChatCompletionRequest.model_validate(data)


class _StubTokenizer:
    """Minimal tokenizer stub to satisfy ``Gemma4EngineToolParser.__init__``."""

    _VOCAB: dict[str, int] = {
        "<|tool_call>": 256_000,
        "<tool_call|>": 256_001,
        '<|"|>': 52,
        "<|channel>": 256_002,
        "<channel|>": 256_003,
    }

    def get_vocab(self) -> dict[str, int]:
        return dict(self._VOCAB)

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self._VOCAB.keys())

    @property
    def all_special_ids(self) -> list[int]:
        return list(self._VOCAB.values())


def test_gemma4_adjust_request_sets_skip_special_tokens_on_responses() -> None:
    """``Gemma4ToolParser.adjust_request`` must flip
    ``skip_special_tokens=False`` for both ``ChatCompletionRequest`` and
    ``ResponsesRequest`` so that ``<|tool_call>`` delimiters reach the
    streaming extractor. The previous
    ``isinstance(ChatCompletionRequest)`` guard omitted the Responses
    path, causing raw ``call:fn{...}`` text to leak via
    ``response.output_text.delta``.
    """
    parser = Gemma4ToolParser(_StubTokenizer())

    request = _build_responses_request(tool_choice="auto")
    assert request.skip_special_tokens is True, (
        "Precondition: ResponsesRequest.skip_special_tokens default is True"
    )

    parser.adjust_request(request)

    assert request.skip_special_tokens is False


def test_tool_parser_adjust_request_builds_valid_response_text_config() -> None:
    """``ToolParser.adjust_request`` must produce a ``ResponseTextConfig``
    whose dumped form contains the JSON schema under the ``schema`` alias
    and does not leak the unrelated ``"Response format for tool calling"``
    description string that the previous two-step construction injected.
    """
    parser = ToolParser.__new__(ToolParser)
    parser.model_tokenizer = None

    request = _build_responses_request(tool_choice="required")
    ToolParser.adjust_request(parser, request)

    assert request.text is not None
    assert request.text.format is not None
    assert request.text.format.type == "json_schema"

    dump: dict[str, Any] = request.text.model_dump(mode="json", by_alias=True)
    fmt = dump.get("format") or {}
    assert fmt.get("type") == "json_schema"
    assert fmt.get("name") == "tool_calling_response"
    assert fmt.get("strict") is True
    # Nested config must be present under the alias. Two-step Pydantic v2
    # construction could drop it from __fields_set__.
    assert "schema" in fmt and isinstance(fmt["schema"], dict)
    # The old code passed a wrong-purpose string; valid field should now
    # either be absent or None (the openai-python default).
    assert fmt.get("description") in (None, "")


def test_gemma4_required_skips_structured_outputs_chatcompletion() -> None:
    """required + ChatCompletion: ``Gemma4EngineToolParser`` must skip the
    forced JSON ``structured_outputs`` so the model emits its native
    ``<|tool_call>`` syntax. The base parser constrained output to JSON the
    native parser cannot read, leaking it as content with empty
    ``tool_calls`` (regression after #45588).
    """
    parser = Gemma4ToolParser(_StubTokenizer())
    request = _build_chat_request(tool_choice="required")

    parser.adjust_request(request)

    assert request.structured_outputs is None
    assert request.skip_special_tokens is False


def test_gemma4_named_skips_structured_outputs_chatcompletion() -> None:
    """named + ChatCompletion: the forced single-function JSON schema must be
    skipped, same as ``required``.
    """
    parser = Gemma4ToolParser(_StubTokenizer())
    request = _build_chat_request(
        tool_choice={"type": "function", "function": {"name": "get_weather"}}
    )

    parser.adjust_request(request)

    assert request.structured_outputs is None
    assert request.skip_special_tokens is False


def test_gemma4_required_skips_structured_outputs_responses() -> None:
    """required + Responses: the forced JSON schema (``request.text``) must be
    skipped so the native delimiters reach the extractor.
    """
    parser = Gemma4ToolParser(_StubTokenizer())
    request = _build_responses_request(tool_choice="required")

    parser.adjust_request(request)

    assert request.text is None
    assert request.skip_special_tokens is False


def test_gemma4_named_skips_structured_outputs_responses() -> None:
    """named (``ToolChoiceFunction``) + Responses: the forced single-function
    JSON schema must be skipped.
    """
    parser = Gemma4ToolParser(_StubTokenizer())
    request = _build_responses_request(
        tool_choice={"type": "function", "name": "get_weather"}
    )

    parser.adjust_request(request)

    assert request.text is None
    assert request.skip_special_tokens is False


def test_gemma4_keeps_special_tokens_with_tools_thinking_disabled() -> None:
    """tools active + thinking disabled: ``skip_special_tokens`` must stay
    False so ``<|tool_call>`` delimiters reach the extractor. The merged
    enable_thinking early-return stripped them, breaking tool calling when
    thinking is off.
    """
    parser = Gemma4ToolParser(_StubTokenizer())
    request = _build_chat_request(
        tool_choice="auto", chat_template_kwargs={"enable_thinking": False}
    )

    parser.adjust_request(request)

    assert request.skip_special_tokens is False


def test_gemma4_keeps_skip_special_tokens_false_when_nothing_to_preserve() -> None:
    """No active tools + thinking disabled: ``skip_special_tokens`` stays
    ``False`` because the parser engine's ``__DROP__`` terminal mechanism
    strips unconfigured special tokens automatically.
    """
    parser = Gemma4ToolParser(_StubTokenizer())
    request = _build_chat_request(
        tool_choice="none", chat_template_kwargs={"enable_thinking": False}
    )

    parser.adjust_request(request)

    assert request.skip_special_tokens is False
