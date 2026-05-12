# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HyperCLOVAX-SEED-Think reasoning parser."""

from unittest.mock import MagicMock

import pytest

from tests.reasoning.utils import run_reasoning_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.hyperclovax_seed_think_reasoning_parser import (
    HyperCLOVAXSeedThinkReasoningParser,
)

PARSER_NAME = "hyperclovax_seed_think"

# Token IDs used by the mock tokenizer. The parser only needs:
#   - tokenizer.encode("</think>", add_special_tokens=False) -> list[int]
#   - tokenizer.get_vocab() -> dict[str, int]  (accessed lazily via .vocab)
THINK_END_ID = 9001
IM_END_ID = 9002


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "</think>": THINK_END_ID,
        "<|im_end|>": IM_END_ID,
    }
    tokenizer.encode.side_effect = (
        lambda text, add_special_tokens=True: [THINK_END_ID]
        if text == "</think>"
        else []
    )
    # `run_reasoning_extraction_streaming` in tests/reasoning/utils.py calls
    # `tokenizer.tokenize(delta)` to derive `token_delta`. The parser's
    # streaming path is text-driven, so any iterable works.
    tokenizer.tokenize.return_value = []
    return tokenizer


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_parser_is_registered():
    """The lazy-loading table must expose `hyperclovax_seed_think`."""
    parser_cls = ReasoningParserManager.get_reasoning_parser(PARSER_NAME)
    assert parser_cls is HyperCLOVAXSeedThinkReasoningParser


def test_reasoning_end_str_property(mock_tokenizer):
    parser = HyperCLOVAXSeedThinkReasoningParser(mock_tokenizer)
    assert parser.reasoning_end_str == "</think>"


# ---------------------------------------------------------------------------
# Constructor / mode capture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "thinking,expected_no_reasoning_content",
    [(True, False), (False, True), (None, True)],
)
def test_thinking_flag_capture(
    mock_tokenizer, thinking, expected_no_reasoning_content
):
    """`thinking` from chat_template_kwargs flips no_reasoning_content."""
    kwargs = {}
    if thinking is not None:
        kwargs["chat_template_kwargs"] = {"thinking": thinking}
    parser = HyperCLOVAXSeedThinkReasoningParser(mock_tokenizer, **kwargs)
    assert parser.thinking is bool(thinking)
    assert parser.no_reasoning_content is expected_no_reasoning_content


# ---------------------------------------------------------------------------
# Non-streaming extract_reasoning
# ---------------------------------------------------------------------------


class TestExtractReasoning:
    def _request(self):
        return ChatCompletionRequest(model="test-model", messages=[])

    def test_with_think_end_splits(self, mock_tokenizer):
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        reasoning, content = parser.extract_reasoning(
            "step by step</think>\n\nfinal answer", self._request()
        )
        assert reasoning == "step by step"
        assert content == "final answer"

    def test_with_think_end_strips_leading_newlines(self, mock_tokenizer):
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        # `\n\n` between </think> and content is stripped via lstrip("\n").
        reasoning, content = parser.extract_reasoning(
            "reasoning</think>\n\n\nfinal", self._request()
        )
        assert reasoning == "reasoning"
        assert content == "final"

    def test_no_think_end_thinking_true_treats_all_as_reasoning(self, mock_tokenizer):
        """thinking=true + no </think> → truncated mid-reasoning."""
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        reasoning, content = parser.extract_reasoning(
            "thinking aloud, never closed", self._request()
        )
        assert reasoning == "thinking aloud, never closed"
        assert content is None

    def test_no_think_end_thinking_false_treats_all_as_content(self, mock_tokenizer):
        """thinking=false: </think> is already in the prompt; model output is content."""
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": False}
        )
        reasoning, content = parser.extract_reasoning(
            "direct answer", self._request()
        )
        assert reasoning is None
        assert content == "direct answer"

    def test_empty_reasoning_segment_becomes_none(self, mock_tokenizer):
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        reasoning, content = parser.extract_reasoning(
            "</think>only content", self._request()
        )
        assert reasoning is None
        assert content == "only content"

    def test_empty_content_segment_becomes_none(self, mock_tokenizer):
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        reasoning, content = parser.extract_reasoning(
            "reasoning only</think>", self._request()
        )
        assert reasoning == "reasoning only"
        assert content is None

    def test_stops_at_first_think_end(self, mock_tokenizer):
        """str.partition is used → stops at the first </think>."""
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        reasoning, content = parser.extract_reasoning(
            "first</think>middle</think>last", self._request()
        )
        assert reasoning == "first"
        assert content == "middle</think>last"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreamingExtraction:
    @pytest.mark.parametrize(
        "deltas,expected_reasoning,expected_content",
        [
            (
                ["step ", "by ", "step", "</think>", "\n\nanswer"],
                "step by step",
                "answer",
            ),
            (
                ["thinking", "</think>", "\n\nfinal", " answer"],
                "thinking",
                "final answer",
            ),
            (
                # </think> arrives split across two deltas; the partial-prefix
                # guard must hold the buffer until the close-tag completes.
                ["reason", "</thi", "nk>", "\n\nanswer"],
                "reason",
                "answer",
            ),
            (
                # No </think> ever; everything stays as reasoning.
                ["never ", "closes ", "reasoning"],
                "never closes reasoning",
                None,
            ),
        ],
    )
    def test_thinking_true_streaming_paths(
        self, mock_tokenizer, deltas, expected_reasoning, expected_content
    ):
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        reasoning, content = run_reasoning_extraction(
            parser, deltas, streaming=True
        )
        assert reasoning == expected_reasoning
        assert content == expected_content

    def test_thinking_false_streams_as_content_only(self, mock_tokenizer):
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": False}
        )
        reasoning, content = run_reasoning_extraction(
            parser, ["plain ", "content ", "only"], streaming=True
        )
        assert reasoning is None
        assert content == "plain content only"

    def test_streaming_empty_initial_delta_returns_none(self, mock_tokenizer):
        """No emission when there's nothing yet (current_text empty)."""
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        result = parser.extract_reasoning_streaming(
            previous_text="",
            current_text="",
            delta_text="",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        assert result is None

    def test_streaming_close_in_single_delta(self, mock_tokenizer):
        """A delta containing both `</think>` and the start of content should
        emit a single DeltaMessage carrying the reasoning prefix as
        ``reasoning`` and the post-marker content as ``content``."""
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        delta = parser.extract_reasoning_streaming(
            previous_text="",
            current_text="reason</think>\n\ntail",
            delta_text="reason</think>\n\ntail",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        assert isinstance(delta, DeltaMessage)
        assert delta.reasoning == "reason"
        assert delta.content == "tail"


# ---------------------------------------------------------------------------
# is_reasoning_end / extract_content_ids
# ---------------------------------------------------------------------------


class TestTokenIdHelpers:
    def test_is_reasoning_end_when_think_end_at_tail(self, mock_tokenizer):
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        # Suffix matches the encoded </think> sequence.
        assert parser.is_reasoning_end([1, 2, THINK_END_ID]) is True

    def test_is_reasoning_end_when_think_end_in_middle(self, mock_tokenizer):
        """Structured decoding requires detecting the marker anywhere in the
        sequence, not only at the tail (regression test for #42366 review)."""
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        # </think> followed by additional content tokens — still ended.
        assert parser.is_reasoning_end([1, 2, THINK_END_ID, 3, 4, 5]) is True

    def test_is_reasoning_end_no_think_end_thinking_true(self, mock_tokenizer):
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        assert parser.is_reasoning_end([1, 2, 3]) is False

    def test_is_reasoning_end_thinking_false_short_input(self, mock_tokenizer):
        """thinking=false starts with no_reasoning_content=True, so a single
        token (no </think> needed) should already report end."""
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": False}
        )
        assert parser.is_reasoning_end([42]) is True

    def test_is_reasoning_end_im_end_short_input(self, mock_tokenizer):
        """Single token list containing <|im_end|> should signal end."""
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        assert parser.is_reasoning_end([IM_END_ID]) is True

    def test_extract_content_ids_returns_post_think_end(self, mock_tokenizer):
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        # [reasoning_tok, reasoning_tok, </think>, content_tok, content_tok]
        ids = [1, 2, THINK_END_ID, 4, 5]
        assert parser.extract_content_ids(ids) == [4, 5]

    def test_extract_content_ids_think_end_at_tail_returns_empty(self, mock_tokenizer):
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        # Per the parser's docstring: trailing </think> with no content yet → [].
        assert parser.extract_content_ids([1, 2, THINK_END_ID]) == []

    def test_extract_content_ids_no_think_end_returns_empty(self, mock_tokenizer):
        parser = HyperCLOVAXSeedThinkReasoningParser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        assert parser.extract_content_ids([1, 2, 3, 4]) == []
