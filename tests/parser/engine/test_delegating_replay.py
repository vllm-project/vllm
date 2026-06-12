# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay tests for DelegatingParser with engine adapters.

Exercises DelegatingParser in engine-adapter mode to verify that delegated
routing produces correct output across chunk sizes.
See test_replay.py for tests that target engine parsers directly.
"""

from __future__ import annotations

from functools import lru_cache

import pytest
from pydantic import TypeAdapter

from tests.parser.engine.replay_harness import (
    assert_parse_output,
    collect_output,
    make_mock_tokenizer,
    replay_streaming,
)
from tests.parser.engine.trace_builder import build_samples
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.parser.abstract_parser import Parser
from vllm.parser.parser_manager import ParserManager

_TOOLS_VALIDATOR = TypeAdapter(list[ChatCompletionToolsParam])

_PAIRINGS: dict[str, tuple[str, str]] = {
    "engine": ("qwen3_coder", "qwen3"),
}

CHUNK_SIZES = [1, 2, 3, 5, 11, 23, None]


@lru_cache
def _get_delegating_parser_cls(pairings: str) -> type[Parser]:
    tool_name, reasoning_name = _PAIRINGS[pairings]
    parser_cls = ParserManager.get_parser(
        tool_parser_name=tool_name,
        reasoning_parser_name=reasoning_name,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    return parser_cls


_all_samples = build_samples("qwen3")


@pytest.mark.parametrize(
    "pairings",
    list(_PAIRINGS),
    ids=lambda p: f"mode={p}",
)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
@pytest.mark.parametrize("sample", _all_samples, ids=lambda s: s.id)
def test_delegating_replay(sample, chunk_size, pairings):
    parser_cls = _get_delegating_parser_cls(pairings=pairings)

    tokenizer = make_mock_tokenizer(sample)
    validated_tools = (
        _TOOLS_VALIDATOR.validate_python(sample.tools) if sample.tools else None
    )
    parser = parser_cls(
        tokenizer,
        validated_tools,
        chat_template_kwargs=sample.chat_template_kwargs,
    )

    deltas = replay_streaming(
        parser,
        sample.tokens,
        chunk_size=chunk_size,
        finished_on_last=True,
        tools=sample.tools,
    )
    output = collect_output(deltas)
    assert_parse_output(output, sample)
