# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay tests for engine parsers (holdback, skip-tool-parsing, adapters).

Replays dynamically built token sequences at different chunk sizes and
holdback depths to verify chunk-size invariance and terminal-token hygiene.
"""

from __future__ import annotations

import pytest

from tests.parser.engine.replay_harness import (
    _test_request,
    assert_no_terminal_leakage,
    assert_parse_output,
    collect_output,
    make_mock_tokenizer,
    replay_streaming,
)
from tests.parser.engine.trace_builder import build_samples
from vllm.parser.abstract_parser import Parser
from vllm.parser.engine.registered_adapters import (
    Qwen3Parser,
)

_ENGINE_PARSERS: dict[str, type[Parser]] = {
    "qwen3_engine": Qwen3Parser,
}

_qwen3_samples = build_samples("qwen3")

_QWEN3_TERMINALS = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
]

HOLDBACK_CONFIGS = [6, 12, 24]


@pytest.mark.parametrize("holdback", HOLDBACK_CONFIGS, ids=lambda h: f"holdback{h}")
@pytest.mark.parametrize("chunk_size", [5, 10], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("sample", _qwen3_samples, ids=lambda s: s.id)
class TestQwen3ReplayWithHoldback:
    """Replay Qwen3 with simulated detokenizer holdback."""

    def test_replay(self, sample, chunk_size, holdback):
        tokenizer = make_mock_tokenizer(sample)
        parser = Qwen3Parser(tokenizer, sample.tools)
        deltas = replay_streaming(
            parser,
            sample.tokens,
            chunk_size=chunk_size,
            holdback_chars=holdback,
        )
        output = collect_output(deltas)

        assert_parse_output(output, sample)
        assert_no_terminal_leakage(
            output,
            _QWEN3_TERMINALS,
            context=f"chunk_size={chunk_size}, holdback={holdback}",
        )


_TOOL_CALL_SAMPLES = [
    (Qwen3Parser, s)
    for s in _qwen3_samples
    if s.expected_tool_calls and s.expected_reasoning
]


def _suppressed_expectations(sample) -> tuple[str, str]:
    """Compute expected (reasoning, content) when tools are suppressed.

    When an explicit reasoning-end delimiter (``</think>``, ``<channel|>``)
    is present, reasoning ends there and the tool call block becomes content.
    When reasoning ends implicitly (the tool-start token triggers both
    REASONING_END and TOOL_CALL_START), reasoning still ends at the tool
    start and the raw tool call block becomes content text — only the
    structured tool parsing is suppressed, not the reasoning boundary.
    """
    full_text = "".join(text for _, text in sample.tokens)
    reasoning = sample.expected_reasoning
    idx = full_text.find(reasoning)
    if idx < 0:
        return (full_text, "")
    after_reasoning = full_text[idx + len(reasoning) :]
    for delim in ("</think>", "<channel|>"):
        pos = after_reasoning.find(delim)
        if pos >= 0:
            return (reasoning, after_reasoning[pos + len(delim) :])
    for delim in ("<tool_call>",):
        pos = after_reasoning.find(delim)
        if pos >= 0:
            return (reasoning, after_reasoning[pos:])
    return (full_text, "")


_DUMMY_TOOLS = [
    {
        "type": "function",
        "function": {"name": "stub", "parameters": {"type": "object"}},
    }
]


@pytest.mark.parametrize("chunk_size", [1, 5, None], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize(
    "parser_cls,sample",
    _TOOL_CALL_SAMPLES,
    ids=lambda v: v.id if hasattr(v, "id") else v.__name__,
)
class TestSkipToolParsingReplay:
    """Replay with skip_tool_parsing=True (tool_choice='none').

    Verifies that reasoning is extracted normally and the raw tool call
    block appears as content text with no tool calls parsed.
    """

    def test_replay(self, parser_cls, sample, chunk_size):
        tokenizer = make_mock_tokenizer(sample)
        kwargs = {}
        if sample.chat_template_kwargs:
            kwargs["chat_template_kwargs"] = sample.chat_template_kwargs
        parser = parser_cls(tokenizer, **kwargs)

        request = _test_request()
        request.tool_choice = "none"
        request.tools = _DUMMY_TOOLS

        all_ids = [tid for tid, _ in sample.tokens]
        all_texts = [text for _, text in sample.tokens]
        if chunk_size is None:
            chunk_size = len(all_ids)

        results = []
        chunks = list(range(0, len(all_ids), chunk_size))
        for i, start in enumerate(chunks):
            end = min(start + chunk_size, len(all_ids))
            is_last = i == len(chunks) - 1
            result = parser.parse_delta(
                "".join(all_texts[start:end]),
                all_ids[start:end],
                request,
                prompt_token_ids=[] if start == 0 else None,
                finished=is_last,
            )
            results.append(result)

        output = collect_output(results)

        expected_reasoning, expected_content = _suppressed_expectations(sample)

        assert output.reasoning == expected_reasoning, (
            f"Reasoning mismatch:\n"
            f"  expected: {expected_reasoning!r}\n"
            f"  actual:   {output.reasoning!r}"
        )
        assert output.tool_calls == [], (
            f"Expected no tool calls but got {output.tool_calls}"
        )
        assert output.content == expected_content, (
            f"Content mismatch:\n"
            f"  expected: {expected_content!r}\n"
            f"  actual:   {output.content!r}"
        )


class TestAdapterReferences:
    """Verify make_adapters sets reasoning/tool parser class refs on parser engine
    parser classes so the serving layer finds them and calls adjust_request."""

    @pytest.mark.parametrize(
        "parser_name",
        list(_ENGINE_PARSERS.keys()),
    )
    def test_adapter_cls_refs_set(self, parser_name):
        parser_cls = _ENGINE_PARSERS[parser_name]
        assert parser_cls.reasoning_parser_cls is not None, (
            f"{parser_name}: reasoning_parser_cls is None"
        )
        assert parser_cls.tool_parser_cls is not None, (
            f"{parser_name}: tool_parser_cls is None"
        )
