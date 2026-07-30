# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the Anthropic Messages streaming converter.

The converter (``AnthropicServingMessages.message_stream_converter``) translates
the OpenAI-compatible chat-completion stream into Anthropic ``message_stream``
events. Unlike the OpenAI stream -- which passes each delta through verbatim and
lets the client accumulate ``content`` and ``tool_calls`` independently -- the
Anthropic protocol is block-structured, so the converter must *demultiplex* a
single OpenAI delta into separate content blocks (text / tool_use), each with
its own ``content_block_start`` / ``_delta`` / ``_stop`` lifecycle.

A single OpenAI delta can carry BOTH ``content`` and ``tool_calls`` in the same
chunk: the ``iquest_coder_v2`` tool parser emits a completed tool call in one
delta together with any leading text (e.g. a stray ``"\\n"`` between ``</think>``
and the call), and MTP speculative decoding makes this frequent by bundling
those tokens into one decode step. The converter must emit the ``tool_use``
block in that case. The earlier ``if content ... elif tool_calls`` + ``continue``
structure only handled the text and dropped the tool call, leaving the client
with a ``stop_reason: tool_use`` and no tool_use block.
"""

import asyncio

import pytest

from vllm.entrypoints.anthropic.protocol import AnthropicStreamEvent
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)


def _make_converter() -> AnthropicServingMessages:
    """Build a converter without running the heavy __init__ (no engine needed).

    ``message_stream_converter`` only reads ``self.stop_reason_map``.
    """
    conv = object.__new__(AnthropicServingMessages)
    conv.stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    return conv


def _chunk(delta: DeltaMessage, finish_reason=None) -> str:
    return (
        "data: "
        + ChatCompletionStreamResponse(
            id="req-1",
            model="m",
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0, delta=delta, finish_reason=finish_reason
                )
            ],
        ).model_dump_json(exclude_unset=True)
        + "\n\n"
    )


def _usage_chunk() -> str:
    """The trailing ``choices=[]`` chunk that carries final usage."""
    return (
        "data: "
        + ChatCompletionStreamResponse(
            id="req-1", model="m", choices=[]
        ).model_dump_json(exclude_unset=True)
        + "\n\n"
    )


def _tool_call(index=0, tool_id=None, name=None, arguments=None) -> DeltaToolCall:
    return DeltaToolCall(
        index=index,
        id=tool_id,
        type="function" if tool_id is not None else None,
        function=DeltaFunctionCall(name=name, arguments=arguments),
    )


def _collect_events(stream) -> list[AnthropicStreamEvent]:
    """Drive the converter and parse each emitted SSE ``data:`` line.

    Returns the parsed :class:`AnthropicStreamEvent`s (skipping the terminal
    ``data: [DONE]`` sentinel), so tests can assert on structure rather than
    raw substrings.
    """

    async def _run():
        raw = ""
        async for item in stream:
            raw += item
        return raw

    raw = asyncio.run(_run())

    events: list[AnthropicStreamEvent] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        # Each SSE frame is "event: <name>\ndata: <json>".
        data_line = next(
            (ln for ln in block.splitlines() if ln.startswith("data:")), None
        )
        if data_line is None:
            continue
        payload = data_line[len("data:") :].strip()
        if payload == "[DONE]":
            continue
        events.append(AnthropicStreamEvent.model_validate_json(payload))
    return events


def _assert_well_formed_blocks(events: list[AnthropicStreamEvent]) -> None:
    """Every content block must be opened, then delta'd, then closed once."""
    open_index = None
    for ev in events:
        if ev.type == "content_block_start":
            assert open_index is None, "nested content_block_start"
            open_index = ev.index
        elif ev.type == "content_block_delta":
            assert open_index is not None, "delta outside a block"
            assert ev.index == open_index
        elif ev.type == "content_block_stop":
            assert open_index is not None, "stop without a start"
            assert ev.index == open_index
            open_index = None
    assert open_index is None, "a content block was never closed"


def _blocks(events: list[AnthropicStreamEvent]) -> list[dict]:
    """Reduce the event stream to one summary dict per content block."""
    blocks: list[dict] = []
    for ev in events:
        if ev.type == "content_block_start":
            cb = ev.content_block
            blocks.append(
                {
                    "index": ev.index,
                    "type": cb.type if cb else None,
                    "id": cb.id if cb else None,
                    "name": cb.name if cb else None,
                    "text": "",
                    "partial_json": "",
                    "thinking": "",
                }
            )
        elif ev.type == "content_block_delta":
            d = ev.delta
            if d is None:
                continue
            if d.text:
                blocks[-1]["text"] += d.text
            if d.partial_json:
                blocks[-1]["partial_json"] += d.partial_json
            if d.thinking:
                blocks[-1]["thinking"] += d.thinking
    return blocks


def _stop_reason(events: list[AnthropicStreamEvent]) -> str | None:
    for ev in events:
        if ev.type == "message_delta" and ev.delta is not None:
            return ev.delta.stop_reason
    return None


def _combined_tool_stream(content_value):
    """A stream whose tool-call delta ALSO carries leading ``content``.

    Mirrors the iquest_coder_v2 + MTP shape: reasoning, then one delta with both
    the trailing content and the fully-formed tool call, then finish.
    """

    async def gen():
        yield _chunk(DeltaMessage(role="assistant"))
        yield _chunk(DeltaMessage(reasoning="Let me start."))
        yield _chunk(
            DeltaMessage(
                content=content_value,
                tool_calls=[_tool_call(0, "call_1", "Agent", '{"description":"x"}')],
            )
        )
        yield _chunk(DeltaMessage(), finish_reason="tool_calls")
        yield _usage_chunk()
        yield "data: [DONE]\n\n"

    return gen()


# --------------------------------------------------------------------------- #
# The regression: content + tool_call bundled in one delta
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("leading", ["\n", "", "   ", "\n\n"])
def test_tool_call_survives_whitespace_leading_content(leading):
    """A stray whitespace/empty ``content`` must not swallow the tool call."""
    events = _collect_events(
        _make_converter().message_stream_converter(_combined_tool_stream(leading))
    )
    _assert_well_formed_blocks(events)

    tool_blocks = [b for b in _blocks(events) if b["type"] == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["name"] == "Agent"
    assert tool_blocks[0]["id"] == "call_1"
    assert tool_blocks[0]["partial_json"] == '{"description":"x"}'
    assert _stop_reason(events) == "tool_use"


def test_tool_call_with_real_leading_text_emits_both_blocks():
    """Real leading text becomes a text block AND the tool call is preserved."""
    events = _collect_events(
        _make_converter().message_stream_converter(
            _combined_tool_stream("Sure, let me check. ")
        )
    )
    _assert_well_formed_blocks(events)

    blocks = _blocks(events)
    # thinking block, then text block, then tool_use block -- in that order.
    types = [b["type"] for b in blocks]
    assert types == ["thinking", "text", "tool_use"]
    assert blocks[1]["text"] == "Sure, let me check. "
    assert blocks[2]["name"] == "Agent"
    assert blocks[2]["partial_json"] == '{"description":"x"}'
    assert _stop_reason(events) == "tool_use"


def test_reasoning_then_tool_transitions_cleanly():
    """thinking -> tool_use with no interleaving text still closes each block."""

    async def gen():
        yield _chunk(DeltaMessage(role="assistant"))
        yield _chunk(DeltaMessage(reasoning="thinking..."))
        # content=None: goes straight to the tool_use branch.
        yield _chunk(
            DeltaMessage(tool_calls=[_tool_call(0, "call_9", "Agent", '{"a":1}')])
        )
        yield _chunk(DeltaMessage(), finish_reason="tool_calls")
        yield _usage_chunk()
        yield "data: [DONE]\n\n"

    events = _collect_events(_make_converter().message_stream_converter(gen()))
    _assert_well_formed_blocks(events)
    blocks = _blocks(events)
    assert [b["type"] for b in blocks] == ["thinking", "tool_use"]
    assert blocks[0]["thinking"] == "thinking..."
    assert blocks[1]["partial_json"] == '{"a":1}'
    assert _stop_reason(events) == "tool_use"


# --------------------------------------------------------------------------- #
# Regressions guarding the pre-existing behavior
# --------------------------------------------------------------------------- #
def test_content_only_stream_unchanged():
    async def gen():
        yield _chunk(DeltaMessage(role="assistant"))
        yield _chunk(DeltaMessage(content="Hello "))
        yield _chunk(DeltaMessage(content="world"))
        yield _chunk(DeltaMessage(), finish_reason="stop")
        yield _usage_chunk()
        yield "data: [DONE]\n\n"

    events = _collect_events(_make_converter().message_stream_converter(gen()))
    _assert_well_formed_blocks(events)
    blocks = _blocks(events)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "Hello world"
    assert _stop_reason(events) == "end_turn"


def test_tool_call_streamed_across_multiple_deltas():
    """The common shape: id-bearing start delta, then argument fragments."""

    async def gen():
        yield _chunk(DeltaMessage(role="assistant"))
        yield _chunk(
            DeltaMessage(tool_calls=[_tool_call(0, "call_1", "Agent", '{"de')])
        )
        yield _chunk(DeltaMessage(tool_calls=[_tool_call(0, arguments='sc":1}')]))
        yield _chunk(DeltaMessage(), finish_reason="tool_calls")
        yield _usage_chunk()
        yield "data: [DONE]\n\n"

    events = _collect_events(_make_converter().message_stream_converter(gen()))
    _assert_well_formed_blocks(events)
    blocks = _blocks(events)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "tool_use"
    assert blocks[0]["partial_json"] == '{"desc":1}'
    assert _stop_reason(events) == "tool_use"


def test_empty_content_only_stream_emits_no_text_delta():
    """A lone empty-string content delta must not crash or emit a text delta."""

    async def gen():
        yield _chunk(DeltaMessage(role="assistant"))
        yield _chunk(DeltaMessage(content=""))
        yield _chunk(DeltaMessage(content="real"))
        yield _chunk(DeltaMessage(), finish_reason="stop")
        yield _usage_chunk()
        yield "data: [DONE]\n\n"

    events = _collect_events(_make_converter().message_stream_converter(gen()))
    _assert_well_formed_blocks(events)
    blocks = _blocks(events)
    assert [b["type"] for b in blocks] == ["text"]
    assert blocks[0]["text"] == "real"
