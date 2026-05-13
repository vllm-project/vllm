# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the DeepSeek V4 reasoning parser.

The V4 parser is a defensive extension of :class:`DeepSeekR1ReasoningParser`
that treats the DSML tool-call start marker as an implicit end-of-reasoning
when ``</think>`` is missing. That failure mode is observed at long context
on DSv4-Flash and was previously trapping tool calls inside reasoning,
leaving the agent loop with nothing to dispatch.
"""

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.deepseek_v4_reasoning_parser import (
    DeepSeekV4ReasoningParser,
    DeepSeekV4ThinkingReasoningParser,
)
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser

START_TOKEN = "<think>"
END_TOKEN = "</think>"
START_TOKEN_ID = 9001
END_TOKEN_ID = 9002
DSML_MARKER = "<｜DSML｜tool_calls>"
DSML_MARKER_TOKEN_ID = 9100


def _make_tokenizer() -> MagicMock:
    """Mock tokenizer mapping the four special strings we care about."""
    tok = MagicMock()
    vocab = {
        START_TOKEN: START_TOKEN_ID,
        END_TOKEN: END_TOKEN_ID,
        DSML_MARKER: DSML_MARKER_TOKEN_ID,
    }
    tok.get_vocab.return_value = vocab

    def _decode(ids, *args, **kwargs):
        out = []
        for tid in ids:
            for s, sid in vocab.items():
                if sid == tid:
                    out.append(s)
                    break
            else:
                out.append(f"<unk{tid}>")
        return "".join(out)

    tok.decode = _decode
    return tok


@pytest.fixture
def tokenizer() -> MagicMock:
    return _make_tokenizer()


@pytest.fixture
def parser(tokenizer) -> DeepSeekV4ThinkingReasoningParser:
    return DeepSeekV4ThinkingReasoningParser(tokenizer)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registration_resolves_to_v4_class():
    parser_cls = ReasoningParserManager.get_reasoning_parser("deepseek_v4")
    assert parser_cls is DeepSeekV4ReasoningParser


@pytest.mark.parametrize(
    "thinking_kwargs,expected_inner",
    [
        ({"thinking": True}, DeepSeekV4ThinkingReasoningParser),
        ({"enable_thinking": True}, DeepSeekV4ThinkingReasoningParser),
        ({"thinking": False}, IdentityReasoningParser),
        ({}, IdentityReasoningParser),
    ],
)
def test_dispatch_based_on_thinking_kwarg(
    tokenizer, thinking_kwargs, expected_inner
):
    parser = DeepSeekV4ReasoningParser(
        tokenizer, chat_template_kwargs=thinking_kwargs
    )
    assert isinstance(parser._parser, expected_inner)


# ---------------------------------------------------------------------------
# Healthy path (explicit </think>) — must match parent behavior
# ---------------------------------------------------------------------------


def test_healthy_implicit_start_explicit_end_in_delta(parser):
    """Model emits reasoning text then </think> in the same delta; the
    parent's R1 splitter handles this — V4 must not interfere."""
    delta = parser.extract_reasoning_streaming(
        previous_text="some reasoning ",
        current_text="some reasoning </think>after",
        delta_text="</think>after",
        previous_token_ids=[100, 101],
        current_token_ids=[100, 101, END_TOKEN_ID, 102],
        delta_token_ids=[END_TOKEN_ID, 102],
    )
    assert delta is not None
    assert delta.reasoning == ""
    assert delta.content == "after"


def test_healthy_explicit_end_in_previous_emits_content(parser):
    """Once </think> has been seen, the parser emits delta_text as content."""
    delta = parser.extract_reasoning_streaming(
        previous_text="reasoning</think>",
        current_text="reasoning</think>some content",
        delta_text="some content",
        previous_token_ids=[100, END_TOKEN_ID],
        current_token_ids=[100, END_TOKEN_ID, 101, 102],
        delta_token_ids=[101, 102],
    )
    assert delta is not None
    assert delta.reasoning is None
    assert delta.content == "some content"


def test_healthy_explicit_start_in_delta(parser):
    """Model emits both <think> and </think> in the same delta."""
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="<think>quick thought</think>answer",
        delta_text="<think>quick thought</think>answer",
        previous_token_ids=[],
        current_token_ids=[START_TOKEN_ID, 200, 201, END_TOKEN_ID, 202],
        delta_token_ids=[START_TOKEN_ID, 200, 201, END_TOKEN_ID, 202],
    )
    assert delta is not None
    assert delta.reasoning == "quick thought"
    assert delta.content == "answer"


# ---------------------------------------------------------------------------
# Defensive path (no </think>, DSML marker appears) — the fix
# ---------------------------------------------------------------------------


def test_implicit_end_marker_in_isolated_delta(parser):
    """Marker arrives as its own delta after pure reasoning. The marker
    token alone should be classified as content, not reasoning."""
    # First, two reasoning-only deltas to populate state.
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="step 1 ",
        delta_text="step 1 ",
        previous_token_ids=[],
        current_token_ids=[300, 301],
        delta_token_ids=[300, 301],
    )
    assert delta is not None
    assert delta.reasoning == "step 1 "
    assert delta.content is None
    assert parser._implicit_end_seen is False

    delta = parser.extract_reasoning_streaming(
        previous_text="step 1 ",
        current_text="step 1 step 2",
        delta_text="step 2",
        previous_token_ids=[300, 301],
        current_token_ids=[300, 301, 302, 303],
        delta_token_ids=[302, 303],
    )
    assert delta is not None
    assert delta.reasoning == "step 2"
    assert delta.content is None
    assert parser._implicit_end_seen is False

    # Now the marker arrives.
    delta = parser.extract_reasoning_streaming(
        previous_text="step 1 step 2",
        current_text=f"step 1 step 2{DSML_MARKER}",
        delta_text=DSML_MARKER,
        previous_token_ids=[300, 301, 302, 303],
        current_token_ids=[300, 301, 302, 303, DSML_MARKER_TOKEN_ID],
        delta_token_ids=[DSML_MARKER_TOKEN_ID],
    )
    assert delta is not None
    assert delta.reasoning is None
    assert delta.content == DSML_MARKER
    assert parser._implicit_end_seen is True


def test_implicit_end_marker_within_delta_split(parser):
    """Marker appears partway through a delta — split it at the boundary."""
    delta_text = f"tail of reasoning{DSML_MARKER}\n<｜DSML｜invoke name=\"w\""
    delta = parser.extract_reasoning_streaming(
        previous_text="head ",
        current_text=f"head {delta_text}",
        delta_text=delta_text,
        previous_token_ids=[400],
        current_token_ids=[400, 401, DSML_MARKER_TOKEN_ID, 402, 403],
        delta_token_ids=[401, DSML_MARKER_TOKEN_ID, 402, 403],
    )
    assert delta is not None
    assert delta.reasoning == "tail of reasoning"
    assert delta.content == f"{DSML_MARKER}\n<｜DSML｜invoke name=\"w\""
    assert parser._implicit_end_seen is True


def test_subsequent_delta_after_implicit_end_is_content(parser):
    """Once the implicit end fires, every later delta is content."""
    # Seed the parser by flipping the sticky flag via a marker delta.
    parser.extract_reasoning_streaming(
        previous_text="reasoning",
        current_text=f"reasoning{DSML_MARKER}",
        delta_text=DSML_MARKER,
        previous_token_ids=[500],
        current_token_ids=[500, DSML_MARKER_TOKEN_ID],
        delta_token_ids=[DSML_MARKER_TOKEN_ID],
    )
    assert parser._implicit_end_seen is True

    # Next delta: pure tool-call body. Should be content.
    delta = parser.extract_reasoning_streaming(
        previous_text=f"reasoning{DSML_MARKER}",
        current_text=f"reasoning{DSML_MARKER}\n<｜DSML｜invoke",
        delta_text="\n<｜DSML｜invoke",
        previous_token_ids=[500, DSML_MARKER_TOKEN_ID],
        current_token_ids=[500, DSML_MARKER_TOKEN_ID, 600, 601],
        delta_token_ids=[600, 601],
    )
    assert delta is not None
    assert delta.reasoning is None
    assert delta.content == "\n<｜DSML｜invoke"


def test_marker_does_not_fire_when_explicit_start_present(parser):
    """If the explicit ``<think>`` token is in the stream, defer to parent.
    This guards against false-positive splits when something that looks
    like a marker shows up in the user's prompt history.
    """
    delta = parser.extract_reasoning_streaming(
        previous_text="<think>",
        current_text=f"<think>discussing {DSML_MARKER}",
        delta_text=f"discussing {DSML_MARKER}",
        previous_token_ids=[START_TOKEN_ID],
        current_token_ids=[START_TOKEN_ID, 700, 701, DSML_MARKER_TOKEN_ID],
        delta_token_ids=[700, 701, DSML_MARKER_TOKEN_ID],
    )
    assert delta is not None
    # Parent puts everything after <think> into reasoning until </think>.
    assert delta.reasoning == f"discussing {DSML_MARKER}"
    assert delta.content is None
    assert parser._implicit_end_seen is False


# ---------------------------------------------------------------------------
# is_reasoning_end / is_reasoning_end_streaming
# ---------------------------------------------------------------------------


def test_is_reasoning_end_with_explicit_end_token(parser):
    assert parser.is_reasoning_end([100, END_TOKEN_ID, 101]) is True


def test_is_reasoning_end_with_implicit_marker(parser):
    """When start/end tokens are absent, decoding to text and finding the
    marker counts as end-of-reasoning."""
    assert parser.is_reasoning_end([300, 301, DSML_MARKER_TOKEN_ID]) is True


def test_is_reasoning_end_pure_reasoning(parser):
    assert parser.is_reasoning_end([300, 301, 302]) is False


def test_is_reasoning_end_streaming_sticky_after_split(parser):
    """After ``extract_reasoning_streaming`` flips the sticky flag,
    ``is_reasoning_end_streaming`` must report end-of-reasoning for any
    subsequent delta — even one that contains neither </think> nor the
    marker."""
    # Seed via marker delta.
    parser.extract_reasoning_streaming(
        previous_text="r",
        current_text=f"r{DSML_MARKER}",
        delta_text=DSML_MARKER,
        previous_token_ids=[800],
        current_token_ids=[800, DSML_MARKER_TOKEN_ID],
        delta_token_ids=[DSML_MARKER_TOKEN_ID],
    )
    assert parser._implicit_end_seen is True
    # Now a plain content-only delta.
    assert parser.is_reasoning_end_streaming([800, DSML_MARKER_TOKEN_ID, 900], [900]) is True


# ---------------------------------------------------------------------------
# Sanity: parent's empty-delta and single-token guards still apply
# ---------------------------------------------------------------------------


def test_single_end_token_delta_returns_none(parser):
    """Parent contract: a delta containing only the end token returns
    ``None`` — the orchestrator handles the transition via
    ``is_reasoning_end``."""
    out = parser.extract_reasoning_streaming(
        previous_text="r",
        current_text="r</think>",
        delta_text="</think>",
        previous_token_ids=[900],
        current_token_ids=[900, END_TOKEN_ID],
        delta_token_ids=[END_TOKEN_ID],
    )
    assert out is None
