# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Cohere v2 -> OpenAI request / response conversion
implemented in ``vllm/entrypoints/cohere/serving.py``.

These cover the pure-Python classmethods so we don't need an engine.
For the instance methods that read ``self._is_reasoning_model`` we
build a lightweight :class:`_FakeServing` subclass that skips the
heavy ``OpenAIServingChat.__init__`` chain (which would otherwise need
a real engine client, model registry, etc.) — the same pattern used in
``test_serving_streaming.py``.
"""

from typing import Any

import pytest

from vllm.entrypoints.cohere.protocol import (
    CohereChatV2Request,
    CohereChatV2Response,
)
from vllm.entrypoints.cohere.serving import (
    _FINISH_REASON_MAP,
    CohereServingChatV2,
    ContentBlockType,
    _map_finish_reason,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    Citation as VLLMCitation,
)
from vllm.entrypoints.openai.engine.protocol import (
    CitationSource,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_request(**kwargs) -> CohereChatV2Request:
    kwargs.setdefault("model", "m")
    kwargs.setdefault("messages", [{"role": "user", "content": "hi"}])
    return CohereChatV2Request(**kwargs)


def _convert(request: CohereChatV2Request) -> ChatCompletionRequest:
    return CohereServingChatV2._convert_v2_to_chat_completion(request)


class _FakeServing(CohereServingChatV2):
    """Lightweight stand-in for :class:`CohereServingChatV2` that skips
    the heavy ``OpenAIServingChat.__init__`` chain.

    Only ``_is_reasoning_model`` is read by the methods under test
    (``_chat_completion_to_v2`` and friends); the rest of the parent
    state is dead weight for unit testing.
    """

    def __init__(self, is_reasoning_model: bool = True) -> None:
        # Intentionally skipping super().__init__ — see class docstring.
        self._is_reasoning_model = is_reasoning_model


def _serving(is_reasoning_model: bool = True) -> CohereServingChatV2:
    return _FakeServing(is_reasoning_model=is_reasoning_model)


def _build_chat_completion_response(
    *,
    response_id: str = "resp_1",
    content: str | None = "hello",
    reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = "stop",
    citations: list[Any] | None = None,
    usage: dict[str, Any] | None = None,
    kv_transfer_params: dict[str, Any] | None = None,
) -> ChatCompletionResponse:
    message: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if reasoning is not None:
        message["reasoning"] = reasoning
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if citations is not None:
        message["citations"] = citations
    kwargs: dict[str, Any] = dict(
        id=response_id,
        object="chat.completion",
        created=0,
        model="m",
        choices=[{"index": 0, "message": message, "finish_reason": finish_reason}],
        # ``usage`` is a required field on ChatCompletionResponse, but the
        # production code defensively handles ``None`` -> no usage block;
        # we round-trip that behavior by post-setting the attribute below.
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    if kv_transfer_params is not None:
        kwargs["kv_transfer_params"] = kv_transfer_params
    resp = ChatCompletionResponse(**kwargs)
    if usage is None:
        resp.usage = None
    else:
        # Replace the placeholder with the caller-provided usage.
        resp = ChatCompletionResponse.model_validate(
            {**resp.model_dump(), "usage": usage}
        )
        if kv_transfer_params is not None:
            resp.kv_transfer_params = kv_transfer_params
    return resp


# ======================================================================
# _map_finish_reason
# ======================================================================


class TestMapFinishReason:
    @pytest.mark.parametrize(
        "openai, cohere",
        [
            ("stop", "COMPLETE"),
            ("length", "MAX_TOKENS"),
            ("tool_calls", "TOOL_CALL"),
            ("stop_sequence", "STOP_SEQUENCE"),
            ("error", "ERROR"),
            (None, "COMPLETE"),
        ],
    )
    def test_known_reasons(self, openai, cohere):
        assert _map_finish_reason(openai) == cohere

    def test_unknown_reason_defaults_to_complete(self):
        assert _map_finish_reason("not_a_real_reason") == "COMPLETE"

    def test_finish_reason_map_is_complete(self):
        # Sanity check that the lookup table covers all documented states.
        assert set(_FINISH_REASON_MAP) == {
            "stop",
            "length",
            "tool_calls",
            "stop_sequence",
            "error",
            None,
        }


# ======================================================================
# _coerce_text_content (system / tool string fallback)
# ======================================================================


class TestCoerceTextContent:
    def test_string_passthrough(self):
        assert CohereServingChatV2._coerce_text_content("hi") == "hi"

    def test_concatenates_text_blocks(self):
        from cohere.types import SystemChatMessageV2

        sys_msg = SystemChatMessageV2(
            content=[
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ]
        )
        assert CohereServingChatV2._coerce_text_content(sys_msg.content) == "ab"


# ======================================================================
# User message conversion
# ======================================================================


class TestConvertUserMessage:
    def test_string_content(self):
        req = _make_request(messages=[{"role": "user", "content": "hi"}])
        result = _convert(req)
        assert result.messages == [{"role": "user", "content": "hi"}]

    def test_text_only_list_flattened_to_string(self):
        # Single-text-block list is flattened back to a string for maximum
        # downstream-template compatibility.
        req = _make_request(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                }
            ]
        )
        result = _convert(req)
        assert result.messages[0] == {"role": "user", "content": "hi"}

    def test_image_url_content_with_detail(self):
        req = _make_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,xxx",
                                "detail": "high",
                            },
                        }
                    ],
                }
            ]
        )
        result = _convert(req)
        msg = result.messages[0]
        assert msg["role"] == "user"
        assert msg["content"] == [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,xxx",
                    "detail": "high",
                },
            }
        ]

    def test_image_url_without_detail_omits_field(self):
        req = _make_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x/i.png"},
                        }
                    ],
                }
            ]
        )
        result = _convert(req)
        assert result.messages[0]["content"][0]["image_url"] == {
            "url": "https://x/i.png"
        }

    def test_text_plus_image_keeps_list(self):
        req = _make_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x/i.png"},
                        },
                    ],
                }
            ]
        )
        result = _convert(req)
        content = result.messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "describe"}
        assert content[1]["type"] == "image_url"


# ======================================================================
# Assistant message conversion
# ======================================================================


class TestConvertAssistantMessage:
    def test_string_content(self):
        req = _make_request(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        result = _convert(req)
        asst = result.messages[1]
        assert asst == {"role": "assistant", "content": "hello"}

    def test_text_and_thinking_blocks(self):
        # ``thinking`` blocks collapse back into the ``reasoning`` field on
        # the OpenAI message; ``text`` blocks become ``content``.
        req = _make_request(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "let me think"},
                        {"type": "text", "text": "Hi!"},
                    ],
                },
            ]
        )
        result = _convert(req)
        asst = result.messages[1]
        assert asst["role"] == "assistant"
        assert asst["content"] == "Hi!"
        assert asst["reasoning"] == "let me think"

    def test_thinking_only(self):
        # Thinking-only assistant messages have no ``content`` set.
        req = _make_request(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "ponder"},
                    ],
                },
            ]
        )
        result = _convert(req)
        asst = result.messages[1]
        assert asst.get("reasoning") == "ponder"
        assert "content" not in asst

    def test_multiple_thinking_blocks_concatenated(self):
        req = _make_request(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "first."},
                        {"type": "thinking", "thinking": "second."},
                        {"type": "text", "text": "done."},
                    ],
                },
            ]
        )
        result = _convert(req)
        asst = result.messages[1]
        assert asst["reasoning"] == "first.second."
        assert asst["content"] == "done."

    def test_tool_plan_collapses_into_reasoning(self):
        # Cohere's ``tool_plan`` is the older chain-of-thought field; it
        # should be appended to ``reasoning`` so the rendered template
        # preserves the planning context.
        req = _make_request(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll call a tool."},
                    ],
                    "tool_plan": "plan: use calculator",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "calc", "arguments": "{}"},
                        }
                    ],
                },
            ]
        )
        result = _convert(req)
        asst = result.messages[1]
        assert asst["content"] == "I'll call a tool."
        assert asst["reasoning"] == "plan: use calculator"
        assert asst["tool_calls"][0]["function"] == {
            "name": "calc",
            "arguments": "{}",
        }

    def test_tool_calls_with_missing_function_pieces_get_defaults(self):
        # The conversion defends against missing function name/arguments
        # by emitting empty string / "{}" defaults so downstream
        # validation never sees a None.
        req = _make_request(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    ],
                },
            ]
        )
        result = _convert(req)
        tc = result.messages[1]["tool_calls"][0]
        assert tc["id"] == "c1"
        assert tc["type"] == "function"
        assert tc["function"] == {"name": "", "arguments": "{}"}


# ======================================================================
# Tool message conversion
# ======================================================================


class TestConvertToolMessage:
    def _request_with_tool_message(self, content: Any) -> CohereChatV2Request:
        return _make_request(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": content},
            ]
        )

    def test_string_content(self):
        req = self._request_with_tool_message("result text")
        result = _convert(req)
        tool_msg = result.messages[-1]
        assert tool_msg == {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "result text",
        }

    def test_text_only_list_flattened_to_newline_string(self):
        # Text-only tool results are flattened to a single newline-joined
        # string for compatibility with vanilla chat templates.
        req = self._request_with_tool_message(
            [
                {"type": "text", "text": "line 1"},
                {"type": "text", "text": "line 2"},
            ]
        )
        result = _convert(req)
        tool_msg = result.messages[-1]
        assert tool_msg["content"] == "line 1\nline 2"

    def test_with_document_preserves_structured_content(self):
        # When documents appear in the tool result, we keep the list shape
        # so the Cohere renderer can lift them into grounding sources.
        req = self._request_with_tool_message(
            [
                {"type": "text", "text": "see attachment"},
                {
                    "type": "document",
                    "document": {"data": {"text": "doc text"}, "id": "d1"},
                },
            ]
        )
        result = _convert(req)
        tool_msg = result.messages[-1]
        assert isinstance(tool_msg["content"], list)
        assert tool_msg["content"][0] == {"type": "text", "text": "see attachment"}
        assert tool_msg["content"][1] == {
            "type": "document",
            "document": {"data": {"text": "doc text"}, "id": "d1"},
        }


# ======================================================================
# System message
# ======================================================================


class TestSystemMessage:
    def test_system_string(self):
        req = _make_request(
            messages=[
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hi"},
            ]
        )
        result = _convert(req)
        assert result.messages[0] == {
            "role": "system",
            "content": "be helpful",
        }

    def test_system_text_blocks_concatenated(self):
        req = _make_request(
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "part1 "},
                        {"type": "text", "text": "part2"},
                    ],
                },
                {"role": "user", "content": "hi"},
            ]
        )
        result = _convert(req)
        assert result.messages[0]["content"] == "part1 part2"


# ======================================================================
# Base ChatCompletionRequest field mapping
# ======================================================================


class TestBuildBaseChatCompletion:
    def test_sampling_and_limits_mapped(self):
        req = _make_request(
            max_tokens=128,
            stop_sequences=["</s>", "STOP"],
            temperature=0.5,
            seed=42,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            k=50,
            p=0.95,
            logprobs=True,
            priority=2,
            kv_transfer_params={"x": 1},
            chat_template_kwargs={"y": 2},
        )
        result = _convert(req)
        assert result.model == "m"
        # ``max_tokens`` is deprecated in favor of ``max_completion_tokens``
        # but the serving code intentionally sets both for compatibility.
        assert result.max_completion_tokens == 128
        assert result.stop == ["</s>", "STOP"]
        assert result.temperature == 0.5
        assert result.seed == 42
        assert result.frequency_penalty == 0.1
        assert result.presence_penalty == 0.2
        assert result.top_k == 50
        assert result.top_p == 0.95
        assert result.logprobs is True
        assert result.priority == 2
        assert result.kv_transfer_params == {"x": 1}
        # ``chat_template_kwargs`` may be expanded by _apply_cohere_*; the
        # base build at least preserves what the caller passed.
        assert (result.chat_template_kwargs or {}).get("y") == 2

    def test_priority_defaults_to_zero(self):
        # ChatCompletionRequest.priority defaults to 0; ``None`` Cohere
        # priority must be coerced rather than passed through.
        req = _make_request()
        result = _convert(req)
        assert result.priority == 0


# ======================================================================
# Streaming options
# ======================================================================


class TestStreamingOptions:
    def test_no_stream_leaves_defaults(self):
        result = _convert(_make_request(stream=False))
        assert not result.stream
        assert result.stream_options is None

    def test_stream_enables_usage_options(self):
        # The v2 translator forces ``include_usage=True`` so the
        # ``message-end`` event can surface ``billed_units`` / ``tokens``;
        # ``continuous_usage_stats`` is intentionally left at its
        # ``StreamOptions`` default (False) — Cohere v2 only reports
        # usage on the terminal event.
        result = _convert(_make_request(stream=True))
        assert result.stream is True
        assert result.stream_options is not None
        assert result.stream_options.include_usage is True


# ======================================================================
# Response format
# ======================================================================


class TestResponseFormat:
    def test_text_is_passthrough(self):
        result = _convert(_make_request(response_format={"type": "text"}))
        assert result.response_format is None

    def test_json_object(self):
        result = _convert(_make_request(response_format={"type": "json_object"}))
        assert result.response_format is not None
        assert result.response_format.type == "json_object"
        assert result.response_format.json_schema is None

    def test_json_schema(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        result = _convert(
            _make_request(
                response_format={"type": "json_object", "json_schema": schema}
            )
        )
        assert result.response_format is not None
        assert result.response_format.type == "json_schema"
        assert result.response_format.json_schema is not None
        assert result.response_format.json_schema.name == "cohere_v2_json_schema"
        # ``JsonSchemaResponseFormat.json_schema`` has alias=``schema`` on
        # the Pydantic field, so we observe the value via the serialized
        # payload (which is what downstream consumers actually read).
        dumped = result.response_format.json_schema.model_dump(exclude_none=True)
        assert dumped["json_schema"] == schema


# ======================================================================
# Tools / tool_choice
# ======================================================================


class TestApplyTools:
    def test_no_tools(self):
        result = _convert(_make_request())
        assert result.tools is None

    def test_basic_tool(self):
        result = _convert(
            _make_request(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "calc",
                            "description": "calculator",
                            "parameters": {"type": "object"},
                        },
                    }
                ]
            )
        )
        assert result.tools is not None
        assert len(result.tools) == 1
        tool = result.tools[0]
        assert tool.type == "function"
        assert tool.function.name == "calc"
        assert tool.function.description == "calculator"
        # ``strict`` is an extra attribute on FunctionDefinition (the
        # field is only stamped onto the OpenAI tool when strict_tools is
        # set on the request). The default path must not set it.
        assert getattr(tool.function, "strict", None) is None

    def test_strict_tools_propagates_to_function(self):
        result = _convert(
            _make_request(
                strict_tools=True,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "calc",
                            "description": "",
                            "parameters": {},
                        },
                    }
                ],
            )
        )
        assert result.tools[0].function.strict is True


class TestApplyToolChoice:
    def test_required(self):
        result = _convert(
            _make_request(
                tool_choice="REQUIRED",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "description": "",
                            "parameters": {},
                        },
                    }
                ],
            )
        )
        assert result.tool_choice == "required"

    def test_none(self):
        result = _convert(
            _make_request(
                tool_choice="NONE",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "description": "",
                            "parameters": {},
                        },
                    }
                ],
            )
        )
        assert result.tool_choice == "none"

    def test_default_to_auto_when_tools_present(self):
        # No explicit ``tool_choice`` + tools present → auto, mirroring
        # Cohere's documented "free choice" default.
        result = _convert(
            _make_request(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "description": "",
                            "parameters": {},
                        },
                    }
                ]
            )
        )
        assert result.tool_choice == "auto"

    def test_no_tools_no_choice_left_unset(self):
        # When there are no tools the underlying ChatCompletionRequest
        # default applies; we must not stamp ``auto``.
        result = _convert(_make_request())
        assert result.tool_choice != "auto"


# ======================================================================
# Cohere-specific template kwargs forwarding
# ======================================================================


class TestApplyCohereTemplateKwargs:
    def test_string_documents_wrapped(self):
        result = _convert(_make_request(documents=["doc 1", "doc 2"]))
        docs = (result.chat_template_kwargs or {}).get("documents")
        assert docs == [
            {"id": "doc_0", "data": {"text": "doc 1"}},
            {"id": "doc_1", "data": {"text": "doc 2"}},
        ]

    def test_document_with_explicit_id_preserved(self):
        result = _convert(
            _make_request(
                documents=[
                    {"id": "custom", "data": {"text": "t"}},
                    {"data": {"text": "t2"}},  # no id -> synthesized
                ]
            )
        )
        docs = result.chat_template_kwargs["documents"]
        assert docs[0] == {"id": "custom", "data": {"text": "t"}}
        assert docs[1]["id"] == "doc_1"

    def test_safety_mode_normalized_to_lowercase(self):
        result = _convert(_make_request(safety_mode="CONTEXTUAL"))
        assert result.chat_template_kwargs["safety_mode"] == "contextual"

    def test_citation_options_forwarded_as_dict(self):
        result = _convert(_make_request(citation_options={"mode": "accurate"}))
        assert result.chat_template_kwargs["citation_options"] == {"mode": "accurate"}

    def test_thinking_forwarded_as_dict(self):
        result = _convert(
            _make_request(thinking={"type": "enabled", "token_budget": 16})
        )
        assert result.chat_template_kwargs["thinking"] == {
            "type": "enabled",
            "token_budget": 16,
        }

    def test_existing_chat_template_kwargs_preserved(self):
        # User-supplied kwargs should not be clobbered by the v2 fields
        # (setdefault semantics).
        result = _convert(
            _make_request(
                chat_template_kwargs={
                    "safety_mode": "user-explicit",
                    "extra": "x",
                },
                safety_mode="CONTEXTUAL",
            )
        )
        assert result.chat_template_kwargs["safety_mode"] == "user-explicit"
        assert result.chat_template_kwargs["extra"] == "x"

    def test_no_template_kwargs_when_no_cohere_fields(self):
        # Without any of the Cohere-specific fields and no caller-supplied
        # kwargs, we must leave ``chat_template_kwargs`` as None so other
        # renderers see a clean request.
        result = _convert(_make_request())
        assert result.chat_template_kwargs is None


# ======================================================================
# _chat_completion_to_v2 (non-streaming response builder)
# ======================================================================


class TestChatCompletionToV2:
    def test_text_only(self):
        serving = _serving(is_reasoning_model=True)
        resp = _build_chat_completion_response(content="hello")
        v2 = serving._chat_completion_to_v2(resp, _make_request())
        assert isinstance(v2, CohereChatV2Response)
        assert v2.id == "resp_1"
        assert v2.finish_reason == "COMPLETE"
        assert v2.message.role == "assistant"
        assert len(v2.message.content) == 1
        assert v2.message.content[0].type == "text"
        assert v2.message.content[0].text == "hello"
        assert v2.message.tool_calls is None
        assert v2.message.tool_plan is None
        assert v2.usage is None

    def test_reasoning_model_keeps_thinking_with_tool_calls(self):
        # Reasoning Command models: ``thinking`` block stays in
        # ``message.content`` and ``tool_plan`` is left unset, even when
        # tool calls are present.
        serving = _serving(is_reasoning_model=True)
        resp = _build_chat_completion_response(
            content="resp text",
            reasoning="thoughts",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
            finish_reason="tool_calls",
        )
        v2 = serving._chat_completion_to_v2(resp, _make_request())
        assert v2.finish_reason == "TOOL_CALL"
        assert v2.message.tool_plan is None
        types = [c.type for c in v2.message.content]
        assert types == ["thinking", "text"]
        assert v2.message.content[0].thinking == "thoughts"
        assert v2.message.content[1].text == "resp text"
        assert v2.message.tool_calls[0].id == "c1"
        assert v2.message.tool_calls[0].function.name == "f"

    def test_non_reasoning_model_moves_reasoning_to_tool_plan(self):
        # Older Command models surface reasoning as ``tool_plan`` on tool-
        # call turns; the thinking block should be dropped from content.
        serving = _serving(is_reasoning_model=False)
        resp = _build_chat_completion_response(
            content=None,
            reasoning="plan",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
            finish_reason="tool_calls",
        )
        v2 = serving._chat_completion_to_v2(resp, _make_request())
        assert v2.message.tool_plan == "plan"
        assert v2.message.content is None
        assert v2.message.tool_calls[0].id == "c1"

    def test_non_reasoning_model_keeps_thinking_when_no_tool_calls(self):
        # No tool calls => non-reasoning behavior is identical to
        # reasoning behavior; the thinking block stays.
        serving = _serving(is_reasoning_model=False)
        resp = _build_chat_completion_response(
            content="answer", reasoning="plan", tool_calls=None
        )
        v2 = serving._chat_completion_to_v2(resp, _make_request())
        assert v2.message.tool_plan is None
        types = [c.type for c in v2.message.content]
        assert types == ["thinking", "text"]

    def test_id_synthesized_when_response_id_missing(self):
        serving = _serving()
        resp = _build_chat_completion_response(content="hi", response_id="")
        v2 = serving._chat_completion_to_v2(resp, _make_request())
        assert v2.id.startswith("chat_")

    def test_kv_transfer_params_propagated(self):
        serving = _serving()
        resp = _build_chat_completion_response(
            content="hi", kv_transfer_params={"k": 1}
        )
        v2 = serving._chat_completion_to_v2(resp, _make_request())
        assert v2.kv_transfer_params == {"k": 1}


# ======================================================================
# _build_usage
# ======================================================================


class TestBuildUsage:
    def test_none_passthrough(self):
        resp = _build_chat_completion_response(content="hi")
        # default usage is None
        assert CohereServingChatV2._build_usage(resp) is None

    def test_basic_usage(self):
        resp = _build_chat_completion_response(
            content="hi",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )
        usage = CohereServingChatV2._build_usage(resp)
        assert usage is not None
        assert usage.billed_units.input_tokens == 10
        assert usage.billed_units.output_tokens == 5
        assert usage.tokens.input_tokens == 10
        assert usage.tokens.output_tokens == 5
        assert usage.cached_tokens is None

    def test_completion_tokens_default_to_zero_when_missing(self):
        resp = _build_chat_completion_response(
            content="hi",
            usage={"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        )
        usage = CohereServingChatV2._build_usage(resp)
        assert usage.billed_units.output_tokens == 0

    def test_cached_tokens_propagated(self):
        resp = _build_chat_completion_response(
            content="hi",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        )
        usage = CohereServingChatV2._build_usage(resp)
        assert usage.cached_tokens == 3


# ======================================================================
# _extract_citations_if_any
# ======================================================================


class TestExtractCitations:
    def test_none_or_empty_returns_none(self):
        assert (
            CohereServingChatV2._extract_citations_if_any(
                type("M", (), {"citations": None})()
            )
            is None
        )
        assert (
            CohereServingChatV2._extract_citations_if_any(
                type("M", (), {"citations": []})()
            )
            is None
        )
        # Missing field entirely.
        assert (
            CohereServingChatV2._extract_citations_if_any(type("M", (), {})()) is None
        )

    def test_vllm_citation_objects_normalized(self):
        msg = type("M", (), {})()
        msg.citations = [
            VLLMCitation(
                start=0,
                end=5,
                text="hello",
                sources=[CitationSource(type="document", id="d1")],
            )
        ]
        out = CohereServingChatV2._extract_citations_if_any(msg)
        assert out is not None
        assert len(out) == 1
        assert out[0].start == 0
        assert out[0].end == 5
        assert out[0].text == "hello"

    def test_dict_citation_payloads_accepted(self):
        msg = type("M", (), {})()
        msg.citations = [
            {
                "start": 0,
                "end": 3,
                "text": "hi!",
                "sources": [{"type": "document", "id": "d1"}],
            }
        ]
        out = CohereServingChatV2._extract_citations_if_any(msg)
        assert out is not None
        assert out[0].start == 0
        assert out[0].text == "hi!"

    def test_malformed_citation_skipped_not_raised(self):
        msg = type("M", (), {})()
        msg.citations = [object()]  # neither dict nor Pydantic
        out = CohereServingChatV2._extract_citations_if_any(msg)
        # All citations dropped → None
        assert out is None


# ======================================================================
# create_error_response sanity
# ======================================================================


class TestCreateErrorResponse:
    def test_envelope_uses_400(self):
        serving = _serving()
        err = serving.create_error_response("oops")
        assert err.error.message == "oops"
        assert err.error.code == 400
        assert err.error.type == "bad_request"


# ======================================================================
# ContentBlockType enum
# ======================================================================


class TestContentBlockType:
    def test_values(self):
        assert ContentBlockType.THINKING == "thinking"
        assert ContentBlockType.TEXT == "text"
        assert ContentBlockType.TOOL_CALL == "tool_call"
