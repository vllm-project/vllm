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

from vllm.entrypoints.cohere.cohere_chat_message import (
    Citation as VLLMCitation,
)
from vllm.entrypoints.cohere.cohere_chat_message import (
    CitationSource,
)
from vllm.entrypoints.cohere.protocol import (
    CohereChatV2Request,
    CohereChatV2Response,
)
from vllm.entrypoints.cohere.serving import (
    _FINISH_REASON_MAP,
    CohereServingChatV2,
    _map_finish_reason,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
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
        # string for compatibility with standard chat templates.
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
# Message-citation forwarding (chat_template_kwargs["_messages_citations"])
# ======================================================================


class TestMessageCitations:
    """Covers the ``_messages_citations`` chat_template_kwargs entry
    that carries ``AssistantChatMessageV2.citations`` from the request
    through to the Cohere renderer. Response-shape ``Citation`` ->
    melody ``FilterCitation`` conversion is best-effort: some
    melody-side fields have no equivalent on the Cohere wire and get
    defaulted.
    """

    def test_absent_when_no_assistant_citations(self):
        result = _convert(
            _make_request(
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            )
        )
        assert result.chat_template_kwargs is None or (
            "_messages_citations" not in (result.chat_template_kwargs or {})
        )

    def test_document_citation_forwarded_by_index(self):
        # Assistant at request-index 1 cites the *second* top-level
        # document (``doc_x``, position 1). Melody addresses documents
        # as ``tool_call_index=0`` (the reserved documents bucket) with
        # ``tool_result_indices`` selecting positions inside the
        # top-level ``documents`` list.
        result = _convert(
            _make_request(
                documents=[
                    {"id": "doc_a", "data": {"text": "irrelevant"}},
                    {"id": "doc_x", "data": {"text": "cited doc"}},
                ],
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": "a",
                        "citations": [
                            {
                                "start": 0,
                                "end": 1,
                                "text": "a",
                                "sources": [{"type": "document", "id": "doc_x"}],
                                "type": "TEXT_CONTENT",
                            }
                        ],
                    },
                ],
            )
        )
        citations_by_index = result.chat_template_kwargs["_messages_citations"]
        assert list(citations_by_index.keys()) == [1]
        cite = citations_by_index[1][0]
        assert cite["start_index"] == 0
        assert cite["end_index"] == 1
        assert cite["text"] == "a"
        assert cite["is_thinking"] is False
        # ``document_ids`` is a parser-*output* field in melody (see
        # ``PromptRenderIds`` in src/templating/util.rs); on the input
        # side melody expects the id resolved into
        # ``tool_result_indices``. Setting a ``document_ids`` list here
        # would have no effect at all -- the renderer would still
        # render ``</co: 0:[]>`` (no anchor).
        assert cite["sources"] == [{"tool_call_index": 0, "tool_result_indices": [1]}]

    def test_document_citation_with_unresolvable_id_drops_citation(self):
        # A citation source pointing at an id that doesn't appear in
        # the request's ``documents`` array leaves the citation with
        # nothing to anchor to. Emitting it with ``sources=[]`` would
        # render a malformed ``<co>text</co: :>`` marker, and silently
        # misattributing to ``documents[0]`` would be worse -- so we
        # drop the whole citation, matching the cohere api's behavior.
        # Since this message's only citation was dropped, the message
        # index shouldn't appear in the forwarded map at all.
        result = _convert(
            _make_request(
                documents=[{"id": "doc_present", "data": {"text": "x"}}],
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": "a",
                        "citations": [
                            {
                                "start": 0,
                                "end": 1,
                                "text": "a",
                                "sources": [{"type": "document", "id": "doc_ghost"}],
                                "type": "TEXT_CONTENT",
                            }
                        ],
                    },
                ],
            )
        )
        forwarded = (result.chat_template_kwargs or {}).get("_messages_citations")
        assert not forwarded

    def test_document_citation_resolves_auto_assigned_fallback_id(self):
        # Requests may include documents without an explicit ``id``;
        # ``_apply_cohere_template_kwargs`` synthesizes ``doc_{idx}``
        # ids for them. A citation that targets that fallback id must
        # still resolve to the right position.
        result = _convert(
            _make_request(
                documents=[
                    {"data": {"text": "unnamed 0"}},
                    {"data": {"text": "unnamed 1"}},
                ],
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": "a",
                        "citations": [
                            {
                                "start": 0,
                                "end": 1,
                                "text": "a",
                                "sources": [{"type": "document", "id": "doc_1"}],
                                "type": "TEXT_CONTENT",
                            }
                        ],
                    },
                ],
            )
        )
        (cite,) = result.chat_template_kwargs["_messages_citations"][1]
        assert cite["sources"] == [{"tool_call_index": 0, "tool_result_indices": [1]}]

    def test_tool_citation_resolves_to_bucket_and_result_index(self):
        # ``ToolSource.id`` on the wire is the id of a specific document
        # inside a tool result (NOT the tool_call_id -- ``type`` is a
        # payload-shape hint, not an id-space discriminator). Here the
        # citation points at the second doc in the second tool call's
        # results, which melody addresses as bucket ``1`` (first tool
        # call, no top-level documents present so it takes bucket 0
        # ... wait: buckets are per unique tool_call_id, so with no
        # top-level docs, ``call_a`` = 0 and ``call_b`` = 1) and
        # ``tool_result_indices=[1]`` (2nd doc inside that bucket).
        result = _convert(
            _make_request(
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "tool_plan": "plan",
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_a",
                        "content": [
                            {
                                "type": "document",
                                "document": {"id": "res_a0", "data": {"text": "r_a0"}},
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "tool_plan": "plan2",
                        "tool_calls": [
                            {
                                "id": "call_b",
                                "type": "function",
                                "function": {"name": "g", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_b",
                        "content": [
                            {
                                "type": "document",
                                "document": {"id": "res_b0", "data": {"text": "r_b0"}},
                            },
                            {
                                "type": "document",
                                "document": {"id": "res_b1", "data": {"text": "r_b1"}},
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "final",
                        "citations": [
                            {
                                "start": 0,
                                "end": 5,
                                "text": "final",
                                "sources": [{"type": "tool", "id": "res_b1"}],
                                "type": "TEXT_CONTENT",
                            }
                        ],
                    },
                ]
            )
        )
        (cite,) = result.chat_template_kwargs["_messages_citations"][5]
        assert cite["sources"] == [
            {"tool_call_index": 1, "tool_result_indices": [1]},
        ]

    def test_tool_citation_shifts_bucket_when_documents_present(self):
        # When the request has a top-level ``documents`` array, melody
        # reserves ``tool_call_index=0`` for it (see
        # ``PromptRenderIds::new`` in src/templating/util.rs), so the
        # first tool-call bucket becomes 1. A regression that forgets
        # this shift would silently attribute tool citations to the
        # documents bucket instead.
        result = _convert(
            _make_request(
                documents=[{"id": "d0", "data": {"text": "x"}}],
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "tool_plan": "plan",
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_a",
                        "content": [
                            {
                                "type": "document",
                                "document": {"id": "res_a0", "data": {"text": "r"}},
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "a",
                        "citations": [
                            {
                                "start": 0,
                                "end": 1,
                                "text": "a",
                                "sources": [{"type": "tool", "id": "res_a0"}],
                                "type": "TEXT_CONTENT",
                            }
                        ],
                    },
                ],
            )
        )
        (cite,) = result.chat_template_kwargs["_messages_citations"][3]
        assert cite["sources"] == [
            {"tool_call_index": 1, "tool_result_indices": [0]},
        ]

    def test_multiple_sources_in_same_bucket_aggregate(self):
        # Two citation sources both point at docs inside the same tool
        # result. Melody's ``<co>text</co: N:[i,j]>`` marker packs them
        # into a single ``Source`` with a list of ``tool_result_indices``
        # rather than two separate ``Source`` entries; the converter
        # groups per-bucket to match that -- matching the cohere api's
        # inbound citation shape.
        result = _convert(
            _make_request(
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "tool_plan": "plan",
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_a",
                        "content": [
                            {
                                "type": "document",
                                "document": {"id": "res0", "data": {"text": "0"}},
                            },
                            {
                                "type": "document",
                                "document": {"id": "res1", "data": {"text": "1"}},
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "ans",
                        "citations": [
                            {
                                "start": 0,
                                "end": 3,
                                "text": "ans",
                                "sources": [
                                    {"type": "tool", "id": "res0"},
                                    {"type": "tool", "id": "res1"},
                                ],
                                "type": "TEXT_CONTENT",
                            }
                        ],
                    },
                ]
            )
        )
        (cite,) = result.chat_template_kwargs["_messages_citations"][3]
        assert cite["sources"] == [
            {"tool_call_index": 0, "tool_result_indices": [0, 1]},
        ]

    def test_tool_citation_with_unresolvable_id_drops_citation(self):
        # Same policy as ``test_document_citation_with_unresolvable_id
        # _drops_citation``: any unresolvable source id drops the whole
        # citation rather than emitting a malformed
        # ``<co>text</co: :>`` marker.
        result = _convert(
            _make_request(
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": "a",
                        "citations": [
                            {
                                "start": 0,
                                "end": 1,
                                "text": "a",
                                "sources": [{"type": "tool", "id": "ghost_doc"}],
                                "type": "TEXT_CONTENT",
                            }
                        ],
                    },
                ]
            )
        )
        forwarded = (result.chat_template_kwargs or {}).get("_messages_citations")
        assert not forwarded

    def test_partial_unresolvable_source_drops_whole_citation(self):
        # If any source in a citation fails to resolve, drop the entire
        # citation -- mixing resolved and unresolved sources produces a
        # partially-correct ``<co>`` marker, which is worse than none.
        # This matches the cohere api's behavior.
        result = _convert(
            _make_request(
                documents=[{"id": "doc_real", "data": {"text": "x"}}],
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": "a",
                        "citations": [
                            {
                                "start": 0,
                                "end": 1,
                                "text": "a",
                                "sources": [
                                    {"type": "document", "id": "doc_real"},
                                    {"type": "document", "id": "doc_ghost"},
                                ],
                                "type": "TEXT_CONTENT",
                            }
                        ],
                    },
                ],
            )
        )
        forwarded = (result.chat_template_kwargs or {}).get("_messages_citations")
        assert not forwarded

    def test_thinking_content_type_maps_to_is_thinking(self):
        result = _convert(
            _make_request(
                documents=[{"id": "d", "data": {"text": "x"}}],
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "chain of thought"},
                            {"type": "text", "text": "answer"},
                        ],
                        "citations": [
                            {
                                "start": 0,
                                "end": 5,
                                "text": "chain",
                                "sources": [{"type": "document", "id": "d"}],
                                "type": "THINKING_CONTENT",
                            }
                        ],
                    },
                ],
            )
        )
        (cite,) = result.chat_template_kwargs["_messages_citations"][1]
        assert cite["is_thinking"] is True
        # And the source resolved (sanity: is_thinking is meaningful
        # only if the citation actually renders).
        assert cite["sources"] == [{"tool_call_index": 0, "tool_result_indices": [0]}]

    def test_plan_type_maps_to_is_thinking(self):
        # PLAN citations sit on tool-plan-style non-reasoning assistant
        # turns; melody treats them like THINKING for template purposes.
        result = _convert(
            _make_request(
                documents=[{"id": "d", "data": {"text": "x"}}],
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "tool_plan": "plan",
                        "citations": [
                            {
                                "start": 0,
                                "end": 4,
                                "text": "plan",
                                "sources": [{"type": "document", "id": "d"}],
                                "type": "PLAN",
                            }
                        ],
                    },
                ],
            )
        )
        (cite,) = result.chat_template_kwargs["_messages_citations"][1]
        assert cite["is_thinking"] is True

    def test_tool_content_string_registers_bucket_but_has_no_doc_ids(self):
        # A string tool result has no citable substructure. The bucket
        # still gets registered (so a later tool call's bucket doesn't
        # slide into its slot), but any citation targeting the string
        # message fails to resolve and the citation is dropped.
        result = _convert(
            _make_request(
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "tool_plan": "plan",
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_a", "content": "plain text"},
                    {
                        "role": "assistant",
                        "content": "a",
                        "citations": [
                            {
                                "start": 0,
                                "end": 1,
                                "text": "a",
                                "sources": [{"type": "tool", "id": "call_a"}],
                                "type": "TEXT_CONTENT",
                            }
                        ],
                    },
                ]
            )
        )
        forwarded = (result.chat_template_kwargs or {}).get("_messages_citations")
        assert not forwarded

    def test_tool_content_all_text_blocks_registers_bucket_but_no_docs(self):
        # Same as the string case, but with a structured content list
        # of pure text blocks. No doc ids are exposed, so any citation
        # targeting the message drops.
        result = _convert(
            _make_request(
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "tool_plan": "plan",
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_a",
                        "content": [
                            {"type": "text", "text": "part 1"},
                            {"type": "text", "text": "part 2"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "a",
                        "citations": [
                            {
                                "start": 0,
                                "end": 1,
                                "text": "a",
                                "sources": [{"type": "tool", "id": "part 1"}],
                                "type": "TEXT_CONTENT",
                            }
                        ],
                    },
                ]
            )
        )
        forwarded = (result.chat_template_kwargs or {}).get("_messages_citations")
        assert not forwarded

    def test_tool_content_mixed_text_and_docs_uses_content_array_position(self):
        # Melody advances ``tool_result_index`` for *every* content
        # item (text items push an empty id into the row; see
        # ``push_tool_message_contents`` in melody/src/templating/
        # util.rs). Docs at positions 1 and 3 in a 4-item mixed
        # content must resolve to result_indices 1 and 3, not the
        # "position among documents only" numbering (which would give
        # 0 and 1).
        result = _convert(
            _make_request(
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "tool_plan": "plan",
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_a",
                        "content": [
                            {"type": "text", "text": "before"},
                            {
                                "type": "document",
                                "document": {"id": "doc1", "data": {"text": "d1"}},
                            },
                            {"type": "text", "text": "between"},
                            {
                                "type": "document",
                                "document": {"id": "doc2", "data": {"text": "d2"}},
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "ans",
                        "citations": [
                            {
                                "start": 0,
                                "end": 3,
                                "text": "ans",
                                "sources": [
                                    {"type": "tool", "id": "doc1"},
                                    {"type": "tool", "id": "doc2"},
                                ],
                                "type": "TEXT_CONTENT",
                            }
                        ],
                    },
                ]
            )
        )
        (cite,) = result.chat_template_kwargs["_messages_citations"][3]
        assert cite["sources"] == [
            {"tool_call_index": 0, "tool_result_indices": [1, 3]},
        ]

    def test_same_tool_call_id_across_messages_offsets_result_indices(self):
        # If a client splits one tool call's outputs across multiple
        # tool messages with the same ``tool_call_id``, melody
        # accumulates them into the same bucket (each message's
        # content array is *appended* to the bucket's slot list). The
        # second message's docs must therefore be offset by the first
        # message's content length. Dory's melody adapter relies on
        # exactly this behavior when it flattens each tool-result
        # document into its own melody ``Message``.
        result = _convert(
            _make_request(
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "tool_plan": "plan",
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_a",
                        "content": [
                            {
                                "type": "document",
                                "document": {"id": "d0", "data": {"text": "0"}},
                            },
                            {
                                "type": "document",
                                "document": {"id": "d1", "data": {"text": "1"}},
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_a",
                        "content": [
                            {
                                "type": "document",
                                "document": {"id": "d2", "data": {"text": "2"}},
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "ans",
                        "citations": [
                            {
                                "start": 0,
                                "end": 3,
                                "text": "ans",
                                "sources": [{"type": "tool", "id": "d2"}],
                                "type": "TEXT_CONTENT",
                            }
                        ],
                    },
                ]
            )
        )
        (cite,) = result.chat_template_kwargs["_messages_citations"][4]
        # ``d2`` is at position 0 within its own message's content,
        # but the previous tool message contributed 2 slots, so its
        # result index in bucket 0 is 2.
        assert cite["sources"] == [
            {"tool_call_index": 0, "tool_result_indices": [2]},
        ]


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
    """Test the parser-output -> wire-shape citation coercion.

    Sources are already fully resolved by the reasoning parser (see
    ``_melody_sources_to_vllm`` in
    ``vllm/reasoning/cohere_command_reasoning_parser.py``, which
    consumes the position map forwarded via ``chat_template_kwargs``).
    The serving layer's remaining responsibilities on the outbound
    path are: coerce to :class:`cohere.types.Citation`, apply the
    ``THINKING_CONTENT`` -> ``PLAN`` rewrite for non-reasoning models,
    and drop citations whose ``sources`` list is entirely empty
    (unattributable anchors). Id-less sources with a payload are
    preserved to match the cohere api's optional-``id`` source shape.
    """

    @staticmethod
    def _make_citation(**overrides: Any) -> VLLMCitation:
        base = dict(
            start=0,
            end=11,
            text="Shakespeare",
            sources=[
                CitationSource(
                    type="document",
                    id="doc_hamlet",
                    document={"id": "doc_hamlet", "text": "Hamlet."},
                ),
            ],
            type="TEXT_CONTENT",
        )
        base.update(overrides)
        return VLLMCitation(**base)

    def test_none_or_empty_returns_none(self):
        serving = _serving()
        assert (
            serving._extract_citations_if_any(type("M", (), {"citations": None})())
            is None
        )
        assert (
            serving._extract_citations_if_any(type("M", (), {"citations": []})())
            is None
        )
        assert serving._extract_citations_if_any(type("M", (), {})()) is None

    def test_resolved_document_source_survives_to_wire(self):
        # A parser-resolved document source flows through unchanged
        # (aside from removing ``None`` fields).
        serving = _serving()
        msg = type("M", (), {})()
        msg.citations = [self._make_citation()]
        out = serving._extract_citations_if_any(msg)
        assert out is not None and len(out) == 1
        wire = out[0].model_dump(exclude_none=True)
        assert wire["type"] == "TEXT_CONTENT"
        assert wire["sources"] == [
            {
                "type": "document",
                "id": "doc_hamlet",
                "document": {"id": "doc_hamlet", "text": "Hamlet."},
            }
        ]

    def test_resolved_tool_source_survives_to_wire(self):
        # ``type="tool"`` sources carry ``tool_output``; validated by
        # the SDK's discriminated union.
        serving = _serving()
        msg = type("M", (), {})()
        msg.citations = [
            self._make_citation(
                sources=[
                    CitationSource(
                        type="tool",
                        id="res_a0",
                        tool_output={"id": "res_a0", "text": "r"},
                    ),
                ],
            )
        ]
        out = serving._extract_citations_if_any(msg)
        assert out is not None
        wire = out[0].model_dump(exclude_none=True)
        assert wire["sources"] == [
            {
                "type": "tool",
                "id": "res_a0",
                "tool_output": {"id": "res_a0", "text": "r"},
            }
        ]

    def test_source_without_id_preserved_with_payload(self):
        # Matches the cohere api: text-only tool results (and top-level
        # docs without a client-provided id) surface on the wire
        # without an ``id`` field, but the payload still rides through
        # so the client can see what was cited.
        serving = _serving()
        msg = type("M", (), {})()
        msg.citations = [
            self._make_citation(
                sources=[
                    CitationSource(type="tool", tool_output={"content": "hi"}),
                ]
            )
        ]
        out = serving._extract_citations_if_any(msg)
        assert out is not None
        wire = out[0].model_dump(exclude_none=True)
        assert wire["sources"] == [{"type": "tool", "tool_output": {"content": "hi"}}]
        assert "id" not in wire["sources"][0]

    def test_citation_with_no_surviving_sources_dropped(self):
        # If every source in a citation is empty (e.g. parser produced
        # a stray citation with no resolvable positions), the whole
        # citation is dropped so we don't emit unattributed anchors.
        serving = _serving()
        msg = type("M", (), {})()
        msg.citations = [self._make_citation(sources=[])]
        assert serving._extract_citations_if_any(msg) is None

    def test_multiple_sources_survive(self):
        serving = _serving()
        msg = type("M", (), {})()
        msg.citations = [
            self._make_citation(
                sources=[
                    CitationSource(type="document", id="d0", document={"id": "d0"}),
                    CitationSource(type="document", id="d1", document={"id": "d1"}),
                ],
            )
        ]
        out = serving._extract_citations_if_any(msg)
        assert out is not None
        wire = out[0].model_dump(exclude_none=True)
        assert [s["id"] for s in wire["sources"]] == ["d0", "d1"]

    def test_plan_type_on_non_reasoning_model(self):
        # ``is_thinking=True`` melody citations arrive tagged as
        # ``THINKING_CONTENT``; on non-reasoning models the reasoning
        # block is surfaced as ``tool_plan`` (see
        # ``_chat_completion_to_v2``), so citations on it are
        # ``PLAN`` on the wire.
        serving = _serving(is_reasoning_model=False)
        msg = type("M", (), {})()
        msg.citations = [self._make_citation(type="THINKING_CONTENT")]
        out = serving._extract_citations_if_any(msg)
        assert out is not None
        assert out[0].type == "PLAN"

    def test_thinking_type_preserved_on_reasoning_model(self):
        serving = _serving(is_reasoning_model=True)
        msg = type("M", (), {})()
        msg.citations = [self._make_citation(type="THINKING_CONTENT")]
        out = serving._extract_citations_if_any(msg)
        assert out is not None
        assert out[0].type == "THINKING_CONTENT"

    def test_dict_shape_from_streaming_pipeline(self):
        # The streaming path receives citations as dicts (because
        # ``DeltaMessage.citations`` is an untyped extras field on the
        # OpenAI wire protocol). ``_to_wire_citation`` must accept
        # both dict and object shapes -- unary uses objects.
        serving = _serving()
        msg = type("M", (), {})()
        msg.citations = [
            {
                "start": 0,
                "end": 3,
                "text": "abc",
                "sources": [
                    {"type": "document", "id": "d0", "document": {"id": "d0"}},
                ],
                "type": "TEXT_CONTENT",
            }
        ]
        out = serving._extract_citations_if_any(msg)
        assert out is not None
        assert out[0].sources[0].id == "d0"


# ======================================================================
# _build_position_to_source
# ======================================================================


class TestBuildPositionToSource:
    """Pin the outbound numbering invariant.

    ``_build_position_to_source`` inverts the same numbering rule that
    ``_build_doc_id_to_prompt_position`` follows on the inbound path
    (melody's ``PromptRenderIds::from_messages`` -- see
    ``melody/src/templating/util.rs``). These tests exist to catch
    numbering drift between the two helpers, which would cause
    outbound citations to attribute to the wrong document.
    """

    def test_top_level_documents_populate_bucket_zero(self):
        request = _make_request(
            documents=[
                {"id": "d0", "data": {"text": "zero"}},
                {"id": "d1", "data": {"text": "one"}},
            ],
        )
        result = CohereServingChatV2._build_position_to_source(request)
        assert set(result.keys()) == {(0, 0), (0, 1)}
        assert result[(0, 0)].type == "document"
        assert result[(0, 0)].id == "d0"
        assert result[(0, 0)].document == {"id": "d0", "text": "zero"}
        assert result[(0, 1)].id == "d1"

    def test_tool_result_documents_start_at_bucket_one_when_docs_present(self):
        request = _make_request(
            documents=[{"id": "d0", "data": {"text": "top"}}],
            messages=[
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "tool_plan": "plan",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_a",
                    "content": [
                        {
                            "type": "document",
                            "document": {"id": "res_a0", "data": {"text": "r"}},
                        }
                    ],
                },
            ],
        )
        result = CohereServingChatV2._build_position_to_source(request)
        assert set(result.keys()) == {(0, 0), (1, 0)}
        assert result[(1, 0)].type == "tool"
        assert result[(1, 0)].id == "res_a0"
        assert result[(1, 0)].tool_output == {"id": "res_a0", "text": "r"}

    def test_tool_call_id_registered_from_assistant_message(self):
        # Bucket assignment must match the cohere api's rule: the
        # first-seen ``tool_call_id`` (whether on the assistant
        # tool_calls or on the tool message) claims the next integer.
        request = _make_request(
            messages=[
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "tool_plan": "plan",
                    "tool_calls": [
                        {
                            "id": "first_call",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        },
                        {
                            "id": "second_call",
                            "type": "function",
                            "function": {"name": "g", "arguments": "{}"},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "second_call",
                    "content": [
                        {
                            "type": "document",
                            "document": {"id": "res_second", "data": {"text": "s"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "first_call",
                    "content": [
                        {
                            "type": "document",
                            "document": {"id": "res_first", "data": {"text": "f"}},
                        }
                    ],
                },
            ],
        )
        result = CohereServingChatV2._build_position_to_source(request)
        # ``first_call`` was seen first -> bucket 0; ``second_call``
        # -> bucket 1. The subsequent tool messages fill their
        # respective buckets, not the message-order buckets.
        assert result[(0, 0)].id == "res_first"
        assert result[(1, 0)].id == "res_second"

    def test_tool_content_text_and_document_blocks_both_populate_map(self):
        # Mixed text + document content: text blocks now surface in
        # the map as id-less ``tool`` sources (matching the cohere
        # api's handling of text-only tool content) so a model
        # citation pointing at the text slot still has a payload to
        # attach.
        request = _make_request(
            messages=[
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "tool_plan": "plan",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_a",
                    "content": [
                        {"type": "text", "text": "prelude"},
                        {
                            "type": "document",
                            "document": {"id": "res_x", "data": {"text": "x"}},
                        },
                    ],
                },
            ],
        )
        result = CohereServingChatV2._build_position_to_source(request)
        assert set(result.keys()) == {(0, 0), (0, 1)}
        assert result[(0, 0)].type == "tool"
        assert result[(0, 0)].id is None
        assert result[(0, 0)].tool_output == {"content": "prelude"}
        assert result[(0, 1)].id == "res_x"

    def test_tool_content_json_text_parsed_as_tool_output(self):
        # The cohere api tries JSON first for text tool content; parity
        # here means a JSON-object text block is spread into
        # ``tool_output`` directly instead of nested under ``content``.
        request = _make_request(
            messages=[
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "tool_plan": "plan",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_a",
                    "content": [{"type": "text", "text": '{"title": "hi"}'}],
                },
            ],
        )
        result = CohereServingChatV2._build_position_to_source(request)
        assert result[(0, 0)].tool_output == {"title": "hi"}

    def test_top_level_document_without_id_omits_id_but_keeps_payload(self):
        # Client didn't provide a doc id: match cohere's optional-id
        # source shape by leaving ``id=None`` on the wire, but keep
        # the payload so the citation is still meaningful.
        request = _make_request(
            documents=[{"data": {"text": "anon"}}],
        )
        result = CohereServingChatV2._build_position_to_source(request)
        assert result[(0, 0)].type == "document"
        assert result[(0, 0)].id is None
        assert result[(0, 0)].document == {"text": "anon"}

    def test_top_level_string_document_has_no_wire_id(self):
        request = _make_request(documents=["raw text"])
        result = CohereServingChatV2._build_position_to_source(request)
        assert result[(0, 0)].id is None
        assert result[(0, 0)].document == {"text": "raw text"}

    def test_tool_content_string_synthesizes_idless_source(self):
        # ``ToolChatMessageV2.content`` may be a bare string; the
        # cohere renderer wraps it as a single text block, so we
        # synthesize a matching id-less source for the slot.
        request = _make_request(
            messages=[
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "tool_plan": "plan",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "just text"},
            ],
        )
        result = CohereServingChatV2._build_position_to_source(request)
        assert result[(0, 0)].type == "tool"
        assert result[(0, 0)].id is None
        assert result[(0, 0)].tool_output == {"content": "just text"}

    def test_multiple_tool_messages_same_call_id_offset_slots(self):
        # Tool results streamed across multiple messages accumulate
        # into the same bucket, so later docs are offset by the sum
        # of previous messages' content lengths.
        request = _make_request(
            messages=[
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "tool_plan": "plan",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_a",
                    "content": [
                        {
                            "type": "document",
                            "document": {"id": "res_first", "data": {"text": "1"}},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_a",
                    "content": [
                        {
                            "type": "document",
                            "document": {"id": "res_second", "data": {"text": "2"}},
                        },
                    ],
                },
            ],
        )
        result = CohereServingChatV2._build_position_to_source(request)
        assert result[(0, 0)].id == "res_first"
        assert result[(0, 1)].id == "res_second"


# ======================================================================
# create_error_response
# ======================================================================


class TestCreateErrorResponse:
    def test_envelope_uses_400(self):
        serving = _serving()
        err = serving.create_error_response("oops")
        assert err.error.message == "oops"
        assert err.error.code == 400
        assert err.error.type == "bad_request"
