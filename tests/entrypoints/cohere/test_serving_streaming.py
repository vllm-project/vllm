# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Cohere v2 SSE stream conversion in
``vllm/entrypoints/cohere/serving.py``.

The stream-translation entry point is
:meth:`CohereServingChatV2._chat_completion_stream_to_v2`, which turns an
async iterable of OpenAI SSE chunks into Cohere's
``message-start → (content|tool-call|citation)* → message-end → [DONE]``
event stream.

We test the helpers (``_StreamState``, ``_handle_*_delta``, etc.) in
isolation, plus a handful of end-to-end scenarios that exercise the
state machine.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from vllm.entrypoints.cohere.protocol import (
    CohereChatV2Request,
    MessageStartEvent,
)
from vllm.entrypoints.cohere.serving import (
    _DONE_FRAME,
    CohereServingChatV2,
    ContentBlockType,
    _emit,
    _sse,
    _StreamState,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionStreamResponse,
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


class _FakeServing(CohereServingChatV2):
    """Lightweight stand-in for :class:`CohereServingChatV2` that skips
    the heavy ``OpenAIServingChat.__init__`` chain (which would need a
    real engine client, model registry, render service, etc.).

    Only ``_is_reasoning_model`` is read by the methods under test
    (``_chat_completion_stream_to_v2`` and the per-delta handlers); the
    rest is dead weight for unit testing.
    """

    def __init__(self, is_reasoning_model: bool = True) -> None:
        # Intentionally skipping super().__init__ — see class docstring.
        self._is_reasoning_model = is_reasoning_model


def _serving(is_reasoning_model: bool = True) -> CohereServingChatV2:
    return _FakeServing(is_reasoning_model=is_reasoning_model)


def _parse_event(frame: str) -> dict[str, Any]:
    """Strip the ``data: ... \\n\\n`` wrapper and parse the JSON payload."""
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    return json.loads(frame[len("data: ") : -2])


def _make_chunk(
    *,
    chunk_id: str = "chunk_0",
    role: str | None = None,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    omit_choices: bool = False,
    citations: list[Any] | None = None,
) -> str:
    """Build the ``data: {...}\\n\\n`` SSE frame the production code
    consumes."""
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning"] = reasoning
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    if citations is not None:
        delta["citations"] = citations

    payload: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
    }
    if not omit_choices:
        payload["choices"] = [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ]
    else:
        payload["choices"] = []
    if usage is not None:
        payload["usage"] = usage
    # Use ChatCompletionStreamResponse to normalize the payload shape.
    chunk = ChatCompletionStreamResponse.model_validate(payload)
    return f"data: {chunk.model_dump_json(exclude_none=False)}\n\n"


def _make_done() -> str:
    return "data: [DONE]\n\n"


async def _stream_from(items: list[str]) -> AsyncGenerator[str, None]:
    for item in items:
        yield item


async def _drain(serving: CohereServingChatV2, items: list[str]) -> list[str]:
    """Drive ``_chat_completion_stream_to_v2`` over ``items`` and collect
    the emitted SSE frames."""
    request = CohereChatV2Request(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    gen = serving._chat_completion_stream_to_v2(_stream_from(items), request)
    return [frame async for frame in gen]


# ======================================================================
# Low-level helpers: _sse, _emit, _DONE_FRAME
# ======================================================================


class TestSSEHelpers:
    def test_sse_wraps_payload(self):
        assert _sse("hello") == "data: hello\n\n"

    def test_done_frame_constant(self):
        # cohere-python and Fern-generated clients key off this exact
        # sentinel; keep it byte-for-byte stable.
        assert _DONE_FRAME == "data: [DONE]\n\n"

    def test_emit_serializes_event_with_type_discriminator(self):
        frame = _emit(
            MessageStartEvent(id="abc", delta={"message": {"role": "assistant"}})
        )
        payload = _parse_event(frame)
        assert payload["type"] == "message-start"
        assert payload["id"] == "abc"
        assert payload["delta"] == {"message": {"role": "assistant"}}


# ======================================================================
# _StreamState
# ======================================================================


class TestStreamState:
    def test_defaults(self):
        st = _StreamState()
        assert st.started is False
        assert st.ended is False
        assert st.finish_reason is None
        assert st.last_chunk_id == ""
        assert st.active_block is None
        assert st.active_block_index is None
        assert st.active_tool_index is None
        assert st.tool_calls_seen == set()

    def test_content_index_monotonic(self):
        st = _StreamState()
        assert st.next_content_index() == 0
        assert st.next_content_index() == 1
        assert st.next_content_index() == 2

    def test_citation_index_separate_from_content_index(self):
        st = _StreamState()
        st.next_content_index()  # 0
        st.next_content_index()  # 1
        # Citation indexing is independent of content indexing.
        assert st.next_citation_index() == 0
        assert st.next_citation_index() == 1


# ======================================================================
# _close_open_blocks
# ======================================================================


class TestCloseOpenBlocks:
    def test_no_block_open_emits_nothing(self):
        serving = _serving()
        state = _StreamState()
        assert serving._close_open_blocks(state) == []

    def test_text_block_emits_content_end(self):
        serving = _serving()
        state = _StreamState()
        state.active_block = ContentBlockType.TEXT
        state.active_block_index = 1
        out = serving._close_open_blocks(state)
        assert len(out) == 1
        payload = _parse_event(out[0])
        assert payload == {"type": "content-end", "index": 1}
        assert state.active_block is None
        assert state.active_block_index is None

    def test_thinking_block_emits_content_end(self):
        serving = _serving()
        state = _StreamState()
        state.active_block = ContentBlockType.THINKING
        state.active_block_index = 3
        out = serving._close_open_blocks(state)
        payload = _parse_event(out[0])
        assert payload == {"type": "content-end", "index": 3}

    def test_tool_call_block_emits_tool_call_end(self):
        serving = _serving()
        state = _StreamState()
        state.active_block = ContentBlockType.TOOL_CALL
        state.active_tool_index = 7
        out = serving._close_open_blocks(state)
        payload = _parse_event(out[0])
        assert payload == {"type": "tool-call-end", "index": 7}
        assert state.active_tool_index is None


# ======================================================================
# _handle_text_delta
# ======================================================================


class TestHandleTextDelta:
    def test_opens_block_first_time(self):
        serving = _serving()
        state = _StreamState()
        events = serving._handle_text_delta(state, "Hi")
        assert len(events) == 2
        start = _parse_event(events[0])
        delta = _parse_event(events[1])
        assert start["type"] == "content-start"
        assert start["index"] == 0
        assert start["delta"]["message"]["content"]["type"] == "text"
        assert delta["type"] == "content-delta"
        assert delta["index"] == 0
        assert delta["delta"]["message"]["content"]["text"] == "Hi"
        assert state.active_block == ContentBlockType.TEXT
        assert state.active_block_index == 0

    def test_continues_block_with_just_delta(self):
        serving = _serving()
        state = _StreamState()
        serving._handle_text_delta(state, "Hi")
        events = serving._handle_text_delta(state, " there")
        # Only a delta event, no new content-start.
        assert len(events) == 1
        delta = _parse_event(events[0])
        assert delta["type"] == "content-delta"
        assert delta["delta"]["message"]["content"]["text"] == " there"

    def test_switches_from_thinking_block(self):
        serving = _serving(is_reasoning_model=True)
        state = _StreamState()
        # Open a thinking block first, then switch to text.
        serving._handle_thinking_delta(state, "ponder")
        events = serving._handle_text_delta(state, "answer")
        types = [_parse_event(ev)["type"] for ev in events]
        assert types == ["content-end", "content-start", "content-delta"]
        # The text block gets a new index (1), distinct from thinking's 0.
        assert _parse_event(events[1])["index"] == 1
        assert state.active_block == ContentBlockType.TEXT


# ======================================================================
# _handle_thinking_delta
# ======================================================================


class TestHandleThinkingDelta:
    def test_reasoning_model_opens_thinking_block(self):
        serving = _serving(is_reasoning_model=True)
        state = _StreamState()
        events = serving._handle_thinking_delta(state, "thought")
        assert len(events) == 2
        start = _parse_event(events[0])
        delta = _parse_event(events[1])
        assert start["type"] == "content-start"
        assert start["delta"]["message"]["content"]["type"] == "thinking"
        assert delta["type"] == "content-delta"
        assert delta["delta"]["message"]["content"]["thinking"] == "thought"
        assert state.active_block == ContentBlockType.THINKING

    def test_reasoning_model_continues_thinking_block(self):
        serving = _serving(is_reasoning_model=True)
        state = _StreamState()
        serving._handle_thinking_delta(state, "first")
        events = serving._handle_thinking_delta(state, " more")
        assert len(events) == 1
        assert _parse_event(events[0])["type"] == "content-delta"

    def test_non_reasoning_model_emits_tool_plan_delta(self):
        # Older Command models stream reasoning as ``tool_plan`` deltas;
        # no content-start/end pair is emitted.
        serving = _serving(is_reasoning_model=False)
        state = _StreamState()
        events = serving._handle_thinking_delta(state, "planning")
        assert len(events) == 1
        payload = _parse_event(events[0])
        assert payload["type"] == "tool-plan-delta"
        assert payload["delta"]["message"]["tool_plan"] == "planning"
        # ``tool_plan`` deltas don't claim an active content block.
        assert state.active_block is None

    def test_non_reasoning_model_closes_open_text_block(self):
        serving = _serving(is_reasoning_model=False)
        state = _StreamState()
        # Open a text block first.
        serving._handle_text_delta(state, "answer")
        events = serving._handle_thinking_delta(state, "rethink")
        types = [_parse_event(ev)["type"] for ev in events]
        assert types == ["content-end", "tool-plan-delta"]


# ======================================================================
# _handle_tool_call_deltas
# ======================================================================


class TestHandleToolCallDeltas:
    def test_new_tool_call_opens_tool_call_start(self):
        serving = _serving()
        state = _StreamState()
        deltas = [
            type(
                "Delta",
                (),
                {
                    "index": 0,
                    "id": "c1",
                    "function": type(
                        "Fn", (), {"name": "calc", "arguments": '{"x":'}
                    )(),
                },
            )()
        ]
        events = serving._handle_tool_call_deltas(state, deltas)
        assert len(events) == 1
        payload = _parse_event(events[0])
        assert payload["type"] == "tool-call-start"
        assert payload["index"] == 0
        tc = payload["delta"]["message"]["tool_calls"]
        assert tc["id"] == "c1"
        assert tc["function"]["name"] == "calc"
        assert tc["function"]["arguments"] == '{"x":'
        assert state.active_block == ContentBlockType.TOOL_CALL
        assert state.active_tool_index == 0
        assert 0 in state.tool_calls_seen

    def test_subsequent_arguments_emit_delta(self):
        serving = _serving()
        state = _StreamState()
        # First call: start.
        first = [
            type(
                "Delta",
                (),
                {
                    "index": 0,
                    "id": "c1",
                    "function": type("Fn", (), {"name": "calc", "arguments": ""})(),
                },
            )()
        ]
        serving._handle_tool_call_deltas(state, first)
        # Second call: same index, additional arguments fragment.
        more = [
            type(
                "Delta",
                (),
                {
                    "index": 0,
                    "id": None,
                    "function": type("Fn", (), {"name": None, "arguments": "1}"})(),
                },
            )()
        ]
        events = serving._handle_tool_call_deltas(state, more)
        assert len(events) == 1
        payload = _parse_event(events[0])
        assert payload["type"] == "tool-call-delta"
        assert payload["index"] == 0
        assert (
            payload["delta"]["message"]["tool_calls"]["function"]["arguments"] == "1}"
        )

    def test_new_tool_call_closes_existing_content_block(self):
        serving = _serving()
        state = _StreamState()
        # Open a text block, then start a tool call.
        serving._handle_text_delta(state, "I'll call:")
        deltas = [
            type(
                "Delta",
                (),
                {
                    "index": 0,
                    "id": "c1",
                    "function": type("Fn", (), {"name": "calc", "arguments": "{}"})(),
                },
            )()
        ]
        events = serving._handle_tool_call_deltas(state, deltas)
        types = [_parse_event(ev)["type"] for ev in events]
        assert types == ["content-end", "tool-call-start"]


# ======================================================================
# _handle_citation_deltas
# ======================================================================


class TestHandleCitationDeltas:
    def test_citation_objects_emit_start_and_end(self):
        serving = _serving()
        state = _StreamState()
        citations = [
            VLLMCitation(
                start=0,
                end=5,
                text="hello",
                sources=[CitationSource(type="document", id="d1")],
            )
        ]
        events = serving._handle_citation_deltas(state, citations)
        assert len(events) == 2
        start = _parse_event(events[0])
        end = _parse_event(events[1])
        assert start["type"] == "citation-start"
        assert start["index"] == 0
        cit_payload = start["delta"]["message"]["citations"]
        assert cit_payload["start"] == 0
        assert cit_payload["end"] == 5
        assert cit_payload["text"] == "hello"
        assert end == {"type": "citation-end", "index": 0}

    def test_dict_citations_accepted(self):
        serving = _serving()
        state = _StreamState()
        citations = [{"start": 0, "end": 3, "text": "Hi!", "sources": []}]
        events = serving._handle_citation_deltas(state, citations)
        assert len(events) == 2
        assert _parse_event(events[0])["type"] == "citation-start"
        assert _parse_event(events[1])["type"] == "citation-end"

    def test_malformed_citation_skipped(self):
        serving = _serving()
        state = _StreamState()
        # Neither a dict nor a Pydantic-model-like object.
        events = serving._handle_citation_deltas(state, [object()])
        assert events == []


# ======================================================================
# _build_message_end_event
# ======================================================================


class TestBuildMessageEndEvent:
    def test_without_usage(self):
        serving = _serving()
        frame = serving._build_message_end_event(chunk_id="abc", finish_reason="stop")
        payload = _parse_event(frame)
        assert payload["type"] == "message-end"
        assert payload["id"] == "abc"
        assert payload["delta"]["finish_reason"] == "COMPLETE"
        assert "usage" not in payload["delta"]

    def test_with_usage(self):
        serving = _serving()
        chunk = ChatCompletionStreamResponse.model_validate(
            {
                "id": "abc",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )
        frame = serving._build_message_end_event(
            chunk_id="abc", finish_reason="length", usage_chunk=chunk
        )
        payload = _parse_event(frame)
        assert payload["delta"]["finish_reason"] == "MAX_TOKENS"
        usage = payload["delta"]["usage"]
        assert usage["billed_units"] == {"input_tokens": 10, "output_tokens": 5}
        assert usage["tokens"] == {"input_tokens": 10, "output_tokens": 5}
        assert "cached_tokens" not in usage

    def test_with_cached_tokens(self):
        serving = _serving()
        chunk = ChatCompletionStreamResponse.model_validate(
            {
                "id": "abc",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 3},
                },
            }
        )
        frame = serving._build_message_end_event(
            chunk_id="abc", finish_reason="stop", usage_chunk=chunk
        )
        payload = _parse_event(frame)
        assert payload["delta"]["usage"]["cached_tokens"] == 3


# ======================================================================
# End-to-end: _chat_completion_stream_to_v2
# ======================================================================


class TestChatCompletionStreamToV2:
    """End-to-end stream lifecycle tests."""

    @pytest.mark.asyncio
    async def test_text_only_happy_path_emits_full_lifecycle(self):
        serving = _serving(is_reasoning_model=True)
        items = [
            _make_chunk(role="assistant"),
            _make_chunk(content="Hi"),
            _make_chunk(content=" there"),
            _make_chunk(
                omit_choices=True,
                usage={
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                },
            ),
        ]
        frames = await _drain(serving, items)
        types = [_parse_event(f)["type"] for f in frames[:-1]]
        # message-start, content-start, content-delta, content-delta,
        # content-end, message-end, then [DONE] as the last frame.
        assert types == [
            "message-start",
            "content-start",
            "content-delta",
            "content-delta",
            "content-end",
            "message-end",
        ]
        assert frames[-1] == _DONE_FRAME
        # message-end should carry usage stats from the trailing chunk.
        end_payload = _parse_event(frames[-2])
        assert end_payload["delta"]["finish_reason"] == "COMPLETE"
        assert end_payload["delta"]["usage"]["billed_units"]["input_tokens"] == 4

    @pytest.mark.asyncio
    async def test_text_then_tool_call_closes_text_first(self):
        serving = _serving(is_reasoning_model=True)
        items = [
            _make_chunk(role="assistant"),
            _make_chunk(content="planning..."),
            _make_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    }
                ]
            ),
            _make_chunk(finish_reason="tool_calls"),
            _make_chunk(
                omit_choices=True,
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            ),
        ]
        frames = await _drain(serving, items)
        types = [_parse_event(f)["type"] for f in frames[:-1]]
        # Text block is opened+delta, then closed before the tool-call
        # opens, and the final close happens on the usage chunk path.
        assert types == [
            "message-start",
            "content-start",
            "content-delta",
            "content-end",
            "tool-call-start",
            "tool-call-end",
            "message-end",
        ]
        # finish_reason captured from the prior chunk.
        end = _parse_event(frames[-2])
        assert end["delta"]["finish_reason"] == "TOOL_CALL"

    @pytest.mark.asyncio
    async def test_thinking_then_text_reasoning_model(self):
        serving = _serving(is_reasoning_model=True)
        items = [
            _make_chunk(role="assistant"),
            _make_chunk(reasoning="thinking..."),
            _make_chunk(content="answer"),
            _make_chunk(finish_reason="stop"),
            _make_chunk(
                omit_choices=True,
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            ),
        ]
        frames = await _drain(serving, items)
        types = [_parse_event(f)["type"] for f in frames[:-1]]
        # Thinking block opens with index 0; text block reopens with index 1.
        assert types == [
            "message-start",
            "content-start",
            "content-delta",
            "content-end",
            "content-start",
            "content-delta",
            "content-end",
            "message-end",
        ]
        thinking_start = _parse_event(frames[1])
        assert thinking_start["delta"]["message"]["content"]["type"] == "thinking"
        text_start = _parse_event(frames[4])
        assert text_start["delta"]["message"]["content"]["type"] == "text"

    @pytest.mark.asyncio
    async def test_reasoning_on_non_reasoning_model_emits_tool_plan_delta(self):
        serving = _serving(is_reasoning_model=False)
        items = [
            _make_chunk(role="assistant"),
            _make_chunk(reasoning="planning"),
            _make_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ]
            ),
            _make_chunk(finish_reason="tool_calls"),
            _make_chunk(
                omit_choices=True,
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            ),
        ]
        frames = await _drain(serving, items)
        types = [_parse_event(f)["type"] for f in frames[:-1]]
        assert types == [
            "message-start",
            "tool-plan-delta",
            "tool-call-start",
            "tool-call-end",
            "message-end",
        ]
        # No thinking content blocks should be present.
        assert "content-start" not in types
        assert "content-end" not in types

    @pytest.mark.asyncio
    async def test_done_marker_in_middle_closes_open_block(self):
        # Some upstreams send [DONE] without a trailing usage-only chunk.
        # The translator must still emit message-end before [DONE] so
        # Cohere clients don't hang.
        serving = _serving(is_reasoning_model=True)
        items = [
            _make_chunk(role="assistant"),
            _make_chunk(content="Hi"),
            _make_done(),
        ]
        frames = await _drain(serving, items)
        types = [_parse_event(f)["type"] for f in frames[:-1]]
        assert types == [
            "message-start",
            "content-start",
            "content-delta",
            "content-end",
            "message-end",
        ]
        assert frames[-1] == _DONE_FRAME

    @pytest.mark.asyncio
    async def test_skips_empty_and_non_data_lines(self):
        serving = _serving()
        items = [
            "\n",
            "event: ping\n\n",
            _make_chunk(role="assistant"),
            "data: \n\n",  # empty data
            _make_chunk(content="Hi"),
            _make_chunk(
                omit_choices=True,
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            ),
        ]
        frames = await _drain(serving, items)
        assert frames[-1] == _DONE_FRAME
        # Should still produce a complete lifecycle.
        types = [_parse_event(f)["type"] for f in frames[:-1]]
        assert types[0] == "message-start"
        assert types[-1] == "message-end"

    @pytest.mark.asyncio
    async def test_exception_in_chunk_parsing_emits_error_message_end(self):
        # An invalid JSON payload after the first chunk triggers the
        # error path: a synthetic message-end with finish_reason=ERROR
        # followed by [DONE].
        serving = _serving()
        items = [
            _make_chunk(role="assistant"),
            "data: {not valid json}\n\n",
        ]
        frames = await _drain(serving, items)
        assert frames[-1] == _DONE_FRAME
        # Find the error-shaped message-end.
        error_end = _parse_event(frames[-2])
        assert error_end["type"] == "message-end"
        assert error_end["delta"]["finish_reason"] == "ERROR"
        assert "error" in error_end["delta"]

    @pytest.mark.asyncio
    async def test_citations_in_delta_emit_citation_events(self):
        serving = _serving()
        # Pass citation dicts the way the cohere2 reasoning parser does.
        items = [
            _make_chunk(role="assistant"),
            _make_chunk(content="hello"),
            _make_chunk(
                citations=[
                    {
                        "start": 0,
                        "end": 5,
                        "text": "hello",
                        "sources": [{"type": "document", "id": "d1"}],
                    }
                ]
            ),
            _make_chunk(finish_reason="stop"),
            _make_chunk(
                omit_choices=True,
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            ),
        ]
        frames = await _drain(serving, items)
        types = [_parse_event(f)["type"] for f in frames[:-1]]
        assert "citation-start" in types
        assert "citation-end" in types

    @pytest.mark.asyncio
    async def test_first_chunk_emits_message_start_with_chunk_id(self):
        serving = _serving()
        items = [
            _make_chunk(chunk_id="my-id", role="assistant"),
            _make_chunk(chunk_id="my-id", content="hi"),
            _make_chunk(
                chunk_id="my-id",
                omit_choices=True,
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            ),
        ]
        frames = await _drain(serving, items)
        ms = _parse_event(frames[0])
        assert ms["type"] == "message-start"
        assert ms["id"] == "my-id"
        assert ms["delta"]["message"]["role"] == "assistant"
