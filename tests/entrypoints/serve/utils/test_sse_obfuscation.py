# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.serve.utils.sse_keep_alive import SSE_KEEP_ALIVE_COMMENT
from vllm.entrypoints.serve.utils.sse_obfuscation import (
    OBFUSCATION_BLOCK_SIZE,
    obfuscate_sse_chunk,
    should_obfuscate,
    with_sse_obfuscation,
)


def _chunk(content: str) -> str:
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": content}}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _payload(frame: str) -> str:
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    return frame[len("data: ") : -2]


@pytest.mark.parametrize(
    ("requested", "server_default", "expected"),
    [
        (None, False, False),
        (None, True, True),
        (True, False, True),
        (False, True, False),
    ],
)
def test_request_overrides_server_default(requested, server_default, expected):
    assert should_obfuscate(requested, server_default) is expected


def test_disabled_returns_same_generator():
    """Disabled obfuscation must add no wrapper or per-chunk work."""

    async def gen():
        yield "x"

    g = gen()
    assert with_sse_obfuscation(g, enabled=False) is g


@pytest.mark.parametrize("token", ["a", "hello", " world!", "é", "日本語"])
def test_padding_rounds_up_to_block(token):
    """Payloads, including multi-byte UTF-8, are padded to the next block
    boundary and no further."""
    frame = _chunk(token)
    unpadded = len(_payload(frame).encode()) + len(',"obfuscation":""')
    size = len(_payload(obfuscate_sse_chunk(frame)).encode())
    assert size % OBFUSCATION_BLOCK_SIZE == 0
    assert 0 <= size - unpadded < OBFUSCATION_BLOCK_SIZE


def test_token_lengths_are_hidden():
    """Across typical token lengths, chunk sizes reveal at most whether a
    token crosses one block boundary, not its length."""
    sizes = {
        len(_payload(obfuscate_sse_chunk(_chunk("x" * n))).encode())
        for n in range(1, 17)
    }
    assert len(sizes) <= 2


def test_payload_is_preserved():
    """Padding adds only the `obfuscation` field; existing fields and the SSE
    framing are unchanged."""
    frame = _chunk('say "hi"\n\\ ok')
    padded = obfuscate_sse_chunk(frame)
    original = json.loads(_payload(frame))
    obfuscated = json.loads(_payload(padded))
    obfuscation = obfuscated.pop("obfuscation")
    assert obfuscated == original
    assert isinstance(obfuscation, str)


def test_padding_is_random():
    """Fixed padding could be removed by HTTP compression."""
    paddings = {
        json.loads(_payload(obfuscate_sse_chunk(_chunk("a"))))["obfuscation"]
        for _ in range(8)
    }
    assert len(paddings) > 1


@pytest.mark.parametrize(
    "frame", ["data: [DONE]\n\n", SSE_KEEP_ALIVE_COMMENT, "data: not-json\n\n"]
)
def test_non_json_frames_pass_through(frame):
    assert obfuscate_sse_chunk(frame) == frame


def test_event_suffix_limits_padding_to_matching_events():
    """Responses streams pad only `*.delta` events, as upstream does."""
    delta = 'event: response.output_text.delta\ndata: {"delta": "hi"}\n\n'
    done = 'event: response.completed\ndata: {"type": "done"}\n\n'
    assert "obfuscation" in obfuscate_sse_chunk(delta, event_suffix=".delta")
    assert obfuscate_sse_chunk(done, event_suffix=".delta") == done


@pytest.mark.asyncio
async def test_stream_pads_chunks_and_closes_upstream():
    closed = False

    async def gen():
        nonlocal closed
        try:
            yield _chunk("a")
            yield "data: [DONE]\n\n"
        finally:
            closed = True

    out = [chunk async for chunk in with_sse_obfuscation(gen(), enabled=True)]
    assert "obfuscation" in json.loads(_payload(out[0]))
    assert out[1] == "data: [DONE]\n\n"
    assert closed
