# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for GLMReasoningParser streaming tool-call routing.

At long context, vLLM may strip ``<|observation|>`` (id 154829) before the
reasoning parser sees streamed text, so the ``<tool_call>`` token (from vocab)
emitted after
``</think>`` can be routed to ``delta.reasoning`` instead of
``delta.content``. That leaves ``tool_calls`` empty for OpenAI-compatible
clients (SWE-agent / SWE-bench on GLM-5.x).
"""

from unittest.mock import MagicMock

import pytest

from vllm.reasoning.glm_reasoning_parser import GLMReasoningParser

_THINK_START = 154841
_THINK_END = 154842
# Arbitrary id — parser must read ``<tool_call>`` from vocab, not hardcode GLM-5 ids.
_TOOL_CALL = 999843
_OBSERVATION = 154829


def _make_parser() -> GLMReasoningParser:
    tok = MagicMock()
    tok.get_vocab = MagicMock(
        return_value={
            "<think>": _THINK_START,
            "</think>": _THINK_END,
            "<tool_call>": _TOOL_CALL,
            "<|observation|>": _OBSERVATION,
        }
    )
    return GLMReasoningParser(tok)


def test_streaming_tool_call_forced_to_content_post_think():
    parser = _make_parser()
    delta = "<tool_call>\nbash\n"
    result = parser.extract_reasoning_streaming(
        previous_text="<think>some reasoning</think>",
        current_text="<think>some reasoning</think>" + delta,
        delta_text=delta,
        previous_token_ids=[_THINK_START, 100, 101, _THINK_END],
        current_token_ids=[_THINK_START, 100, 101, _THINK_END, _TOOL_CALL, 200],
        delta_token_ids=[_TOOL_CALL, 200],
    )
    assert result is not None
    assert result.content == delta
    assert result.reasoning is None


def test_streaming_continuation_after_tool_call_routes_to_content():
    parser = _make_parser()
    delta = "<arg_key>command</arg_key><arg_value>pwd</arg_value>\n</tool_call>"
    result = parser.extract_reasoning_streaming(
        previous_text="<think>reasoning</think><tool_call>\nbash\n",
        current_text="<think>reasoning</think><tool_call>\nbash\n" + delta,
        delta_text=delta,
        previous_token_ids=[_THINK_START, 100, _THINK_END, _TOOL_CALL, 200],
        current_token_ids=[_THINK_START, 100, _THINK_END, _TOOL_CALL, 200, 300],
        delta_token_ids=[300],
    )
    assert result is not None
    assert result.content == delta
    assert result.reasoning is None


def test_streaming_inside_think_not_intercepted():
    parser = _make_parser()
    delta = "still reasoning here"
    result = parser.extract_reasoning_streaming(
        previous_text="<think>some",
        current_text="<think>some" + delta,
        delta_text=delta,
        previous_token_ids=[_THINK_START, 100],
        current_token_ids=[_THINK_START, 100, 200],
        delta_token_ids=[200],
    )
    assert result is not None
    assert result.reasoning == delta
    assert result.content is None


def test_streaming_post_think_without_tool_call_token_falls_through():
    parser = _make_parser()
    delta = "plain text after think"
    result = parser.extract_reasoning_streaming(
        previous_text="<think>reasoning</think>",
        current_text="<think>reasoning</think>" + delta,
        delta_text=delta,
        previous_token_ids=[_THINK_START, 100, _THINK_END],
        current_token_ids=[_THINK_START, 100, _THINK_END, 200],
        delta_token_ids=[200],
    )
    assert result is not None
    assert result.content == delta
    assert result.reasoning is None


def test_streaming_end_token_id_from_delegated_parser():
    """end_token_id may live on the delegated parser, not GLMReasoningParser."""
    parser = _make_parser()
    parser.__dict__.pop("end_token_id", None)
    delegated = MagicMock()
    delegated.end_token_id = _THINK_END
    parser._parser = delegated

    delta = "<tool_call>\n"
    result = parser.extract_reasoning_streaming(
        previous_text="</think>",
        current_text="</think>" + delta,
        delta_text=delta,
        previous_token_ids=[_THINK_END],
        current_token_ids=[_THINK_END, _TOOL_CALL],
        delta_token_ids=[_TOOL_CALL],
    )
    assert result is not None
    assert result.content == delta


def test_tool_call_token_id_read_from_vocab_not_hardcoded():
    """Routing uses vocab['<tool_call>'], not a fixed model id."""
    alt_id = 42
    tok = MagicMock()
    tok.get_vocab = MagicMock(
        return_value={
            "<think>": _THINK_START,
            "</think>": _THINK_END,
            "<tool_call>": alt_id,
            "<|observation|>": _OBSERVATION,
        }
    )
    parser = GLMReasoningParser(tok)
    assert parser._tool_call_token_id == alt_id
    delta = "x"
    result = parser.extract_reasoning_streaming(
        previous_text="",
        current_text=delta,
        delta_text=delta,
        previous_token_ids=[_THINK_END],
        current_token_ids=[_THINK_END, alt_id],
        delta_token_ids=[alt_id],
    )
    assert result is not None
    assert result.content == delta


def test_init_raises_when_tool_call_missing_from_vocab():
    tok = MagicMock()
    tok.get_vocab = MagicMock(
        return_value={
            "<think>": _THINK_START,
            "</think>": _THINK_END,
            "<|observation|>": _OBSERVATION,
        }
    )
    with pytest.raises(RuntimeError, match="could not locate"):
        GLMReasoningParser(tok)
