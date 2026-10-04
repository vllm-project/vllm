# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning closers served without a reasoning parser.

When a tool parser is configured on its own, reasoning markup is plain
content: ``DelegatingParser.__init__`` turns on ``skip_reasoning_parsing``
so the engine passes those markers through instead of consuming them.
These tests pin the observable half of that — a stray reasoning closer
with no opener survives into ``content`` verbatim, identically whether
the response is parsed at once or streamed with the closer landing in the
first delta or a later one.

Parsers are built through ``ParserManager.get_parser`` because the opt-in
lives in ``DelegatingParser.__init__``; a hand-assembled engine would not
exercise it.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from tests.parser.engine.replay_harness import MockTokenizer, _test_request
from vllm.parser.parser_manager import ParserManager

_PREFIX = "work it out: 17*23"
_SUFFIX = "the answer is 391"

_XML_VOCAB = {
    "<think>": 1,
    "</think>": 2,
    "<tool_call>": 3,
    "</tool_call>": 4,
    "<arg_key>": 5,
    "</arg_key>": 6,
    "<arg_value>": 7,
    "</arg_value>": 8,
}


class _Case(NamedTuple):
    """One tool parser, addressed by the name the registry knows it by."""

    tool_parser_name: str
    closer: str
    vocab: dict[str, int]


_CASES = [
    _Case(
        "seed_oss",
        "</seed:think>",
        {
            "<seed:think>": 50,
            "</seed:think>": 51,
            "<seed:tool_call>": 60,
            "</seed:tool_call>": 61,
        },
    ),
    # glm47 and glm45 are two registry names for one parser class, whose
    # engine config is named glm47_moe.
    _Case("glm47", "</think>", _XML_VOCAB),
    _Case("ling3", "</think>", _XML_VOCAB),
    _Case(
        "kimi_k2",
        "</think>",
        {
            "<think>": 1,
            "</think>": 2,
            "<|tool_calls_section_begin|>": 10,
            "<|tool_calls_section_end|>": 11,
            "<|tool_call_begin|>": 12,
            "<|tool_call_argument_begin|>": 13,
            "<|tool_call_end|>": 14,
        },
    ),
    _Case(
        "minimax_m2",
        "</think>",
        {
            "<think>": 1,
            "</think>": 2,
            "<minimax:tool_call>": 3,
            "</minimax:tool_call>": 4,
        },
    ),
    _Case(
        "mistral",
        "[/THINK]",
        {"[THINK]": 1, "[/THINK]": 2, "[TOOL_CALLS]": 3, "[ARGS]": 4},
    ),
]


def _tokens(case: _Case) -> list[tuple[int, str]]:
    """Token stream for ``_PREFIX + closer + _SUFFIX``, closer at index 4."""
    return [
        (101, "work"),
        (102, " it"),
        (103, " out:"),
        (104, " 17*23"),
        (case.vocab[case.closer], case.closer),
        (105, "the"),
        (106, " answer"),
        (107, " is"),
        (108, " 391"),
    ]


def _tool_only_parser(case: _Case):
    """A tool parser with no reasoning parser, as the serving layer builds it."""
    parser_cls = ParserManager.get_parser(
        tool_parser_name=case.tool_parser_name,
        enable_auto_tools=True,
    )
    tokenizer = MockTokenizer(vocab=dict(case.vocab), tokens=_tokens(case))
    return parser_cls(tokenizer, None)


def _stream(parser, tokens, chunk_size):
    """Accumulate ``DeltaMessage`` content and reasoning across ``parse_delta``.

    The probes carry no tool syntax, so any tool-call delta is a failure.
    """
    request = _test_request()
    content, reasoning = "", ""
    step = chunk_size or len(tokens)
    for start in range(0, len(tokens), step):
        batch = tokens[start : start + step]
        delta = parser.parse_delta(
            "".join(text for _, text in batch),
            [tid for tid, _ in batch],
            request,
            prompt_token_ids=[] if start == 0 else None,
            finished=(start + step >= len(tokens)),
        )
        if delta is None:
            continue
        if delta.content:
            content += delta.content
        if delta.reasoning:
            reasoning += delta.reasoning
        assert not delta.tool_calls, f"unexpected tool-call delta: {delta.tool_calls}"
    return content, reasoning


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.tool_parser_name)
def test_unpaired_closer_stays_in_content(case: _Case):
    reasoning, content, tool_calls = _tool_only_parser(case).parse(
        _PREFIX + case.closer + _SUFFIX,
        _test_request(),
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content == _PREFIX + case.closer + _SUFFIX
    assert not tool_calls


@pytest.mark.parametrize(
    "chunk_size",
    [None, 1, 3, 4],
    ids=["closer-in-first-delta", "chunk=1", "chunk=3", "chunk=4"],
)
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.tool_parser_name)
def test_unpaired_closer_stays_in_streamed_content(case: _Case, chunk_size):
    content, reasoning = _stream(_tool_only_parser(case), _tokens(case), chunk_size)

    assert reasoning == ""
    assert content == _PREFIX + case.closer + _SUFFIX
