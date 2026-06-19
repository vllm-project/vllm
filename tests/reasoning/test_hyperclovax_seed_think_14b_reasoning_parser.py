# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HyperCLOVAX-SEED-Think-14B reasoning parser.

These tests use a mock tokenizer so they do not require the model weights.
They cover the force_reasoning x skip_reasoning matrix (the chat_template_kwargs
axes the parser must disambiguate), the reasoning->content boundary split, the
reasoning-end boundary detection used by the structured-output
(``tool_choice="required"``) path, and the streaming variants.
"""

from unittest.mock import MagicMock

import pytest

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.hyperclovax_seed_think_14b_reasoning_parser import (
    HyperCLOVAXSeedThink14BReasoningParser,
)

PARSER_NAME = "hyperclovax_seed_think_14b"
THINK_START = "/think\n"
ASSISTANT = "<|im_end|>\n<|im_start|>assistant"
ROLE = " -> tool/function_call\n"


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.get_vocab.return_value = {}
    # Deterministic, dependency-free encoding: one fake id per character.
    tok.encode.side_effect = lambda text, add_special_tokens=False: (
        [ord(ch) % 1000 for ch in text] or [0]
    )
    tok.tokenize.return_value = []
    return tok


def make_request(tool_choice=None, tools=None, **chat_template_kwargs):
    request = MagicMock()
    request.chat_template_kwargs = dict(chat_template_kwargs)
    request.tool_choice = tool_choice
    request.tools = tools
    return request


def build_parser(tokenizer, **chat_template_kwargs):
    return HyperCLOVAXSeedThink14BReasoningParser(
        tokenizer, chat_template_kwargs=chat_template_kwargs
    )


def _ctk(force, skip):
    ctk = {}
    if force is not None:
        ctk["force_reasoning"] = force
    if skip is not None:
        ctk["skip_reasoning"] = skip
    return ctk


# The 9 (force_reasoning, skip_reasoning) combinations the 54-case matrix
# exercises. With no boundary in the output, the mode is decided purely by the
# flags: force_reasoning wins -> reasoning; skip_reasoning -> content;
# otherwise (default/auto) -> content.
_TRISTATE = [False, True, None]
NO_BOUNDARY_CASES = []
for _force in _TRISTATE:
    for _skip in _TRISTATE:
        if _force is True:
            NO_BOUNDARY_CASES.append((_force, _skip, "thinking", "thinking", None))
        elif _skip is True:
            NO_BOUNDARY_CASES.append((_force, _skip, "\nanswer", None, "answer"))
        else:
            NO_BOUNDARY_CASES.append((_force, _skip, "answer", None, "answer"))


class TestRegistration:
    def test_lazy_registered(self, tokenizer):
        cls = ReasoningParserManager.get_reasoning_parser(PARSER_NAME)
        assert cls is HyperCLOVAXSeedThink14BReasoningParser
        assert isinstance(cls(tokenizer), HyperCLOVAXSeedThink14BReasoningParser)


class TestExtractReasoningMatrix:
    @pytest.mark.parametrize("force,skip,raw,exp_r,exp_c", NO_BOUNDARY_CASES)
    def test_no_boundary_force_skip_matrix(
        self, tokenizer, force, skip, raw, exp_r, exp_c
    ):
        ctk = _ctk(force, skip)
        parser = build_parser(tokenizer, **ctk)
        tool_choice = "auto" if (force is not True and skip is not True) else None
        reasoning, content = parser.extract_reasoning(
            raw, make_request(tool_choice=tool_choice, **ctk)
        )
        assert reasoning == exp_r
        assert content == exp_c

    @pytest.mark.parametrize(
        "raw,exp_r,exp_c",
        [
            ("reason" + ASSISTANT + "\nanswer", "reason", "answer"),
            (THINK_START + "thinking" + ASSISTANT + "\nanswer", "thinking", "answer"),
        ],
    )
    def test_boundary_splits_reasoning_and_content(self, tokenizer, raw, exp_r, exp_c):
        parser = build_parser(tokenizer)
        reasoning, content = parser.extract_reasoning(raw, make_request())
        assert reasoning == exp_r
        assert content == exp_c

    def test_required_tool_call_strips_header_and_normalizes(self, tokenizer):
        parser = build_parser(tokenizer, force_reasoning=True)
        payload = '[{"name": "get_weather", "arguments": {"city": "Seoul"}}]'
        out = "reason" + ASSISTANT + ROLE + payload + "<|im_end|>"
        reasoning, content = parser.extract_reasoning(
            out,
            make_request(
                tool_choice="required", tools=[{"x": 1}], force_reasoning=True
            ),
        )
        assert reasoning == "reason"
        assert content is not None
        # The " -> tool/function_call" header and end token are stripped, and
        # "arguments" is normalized to "parameters" for the bare JSON array.
        assert content.startswith("[{")
        assert '"parameters"' in content
        assert "<|im_end|>" not in content
        assert ROLE not in content

    def test_literal_im_end_in_argument_not_truncated(self, tokenizer):
        # A literal <|im_end|> inside a string argument must not be taken as the
        # tool-call boundary; the reasoner hands the full array downstream and
        # only the trailing terminator is stripped.
        parser = build_parser(tokenizer, force_reasoning=True)
        payload = '[{"name": "get_weather", "arguments": {"city": "a <|im_end|> b"}}]'
        out = "reason" + ASSISTANT + ROLE + payload + "<|im_end|>"
        reasoning, content = parser.extract_reasoning(
            out,
            make_request(tool_choice="auto", tools=[{"x": 1}], force_reasoning=True),
        )
        assert reasoning == "reason"
        assert content == payload


def _stream_extract(parser, raw):
    """Drive the streaming API one character per delta and rebuild the split."""
    previous = ""
    reasoning = ""
    content = ""
    for i in range(1, len(raw) + 1):
        msg = parser.extract_reasoning_streaming(
            previous, raw[:i], raw[i - 1], [], [], []
        )
        if msg is not None:
            if getattr(msg, "reasoning", None):
                reasoning += msg.reasoning
            if msg.content:
                content += msg.content
        previous = raw[:i]
    return reasoning or None, content or None


class TestExtractReasoningStreaming:
    def test_force_reasoning_streaming_splits_at_boundary(self, tokenizer):
        parser = build_parser(tokenizer, force_reasoning=True)
        reasoning, content = _stream_extract(parser, "reason" + ASSISTANT + "\nanswer")
        assert reasoning == "reason"
        assert content is not None and content.strip() == "answer"

    def test_skip_reasoning_streaming_is_content(self, tokenizer):
        parser = build_parser(tokenizer, skip_reasoning=True)
        reasoning, content = _stream_extract(parser, "answer")
        assert reasoning is None
        assert content is not None and content.strip() == "answer"


class TestIsReasoningEnd:
    def test_bare_assistant_boundary_is_end(self, tokenizer):
        parser = build_parser(tokenizer, force_reasoning=True)
        # The bare "<|im_end|>\n<|im_start|>assistant" token sequence is the
        # decisive boundary that lets the grammar engage at the right token.
        bare_boundary_ids = parser.think_end_tokens[-1]
        assert parser.is_reasoning_end(list(bare_boundary_ids))

    def test_unrelated_ids_are_not_end(self, tokenizer):
        parser = build_parser(tokenizer, force_reasoning=True)
        assert not parser.is_reasoning_end([1, 2, 3])
