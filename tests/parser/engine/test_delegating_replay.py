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

_PAIRINGS: dict[str, tuple[str, str, str]] = {
    "engine": ("qwen3_coder", "qwen3", "qwen3"),
    "gemma4_engine": ("gemma4", "gemma4", "gemma4"),
}

CHUNK_SIZES = [1, 2, 3, 5, 11, 23, None]


@lru_cache
def _get_delegating_parser_cls(pairings: str) -> type[Parser]:
    tool_name, reasoning_name, _ = _PAIRINGS[pairings]
    parser_cls = ParserManager.get_parser(
        tool_parser_name=tool_name,
        reasoning_parser_name=reasoning_name,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    return parser_cls


def _pairing_samples() -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    for pairing_name, (_, _, model) in _PAIRINGS.items():
        for sample in build_samples(model):
            items.append((pairing_name, sample))
    return items


_all_pairing_samples = _pairing_samples()


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
@pytest.mark.parametrize(
    "pairings,sample",
    _all_pairing_samples,
    ids=lambda v: v.id if hasattr(v, "id") else v,
)
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
        prompt_token_ids=sample.prompt_token_ids,
    )
    output = collect_output(deltas)
    assert_parse_output(output, sample)
