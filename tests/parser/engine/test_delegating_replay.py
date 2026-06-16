# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay tests for DelegatingParser with engine adapters.

Exercises DelegatingParser in engine-adapter mode to verify that delegated
routing produces correct output across chunk sizes.
See test_replay.py for tests that target engine parsers directly.

Parser discovery is automatic: any engine parser in ``registered_adapters``
that has both tool and reasoning adapters and a builder in
``trace_builder._BUILDERS`` is picked up with zero manual wiring.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
from pydantic import TypeAdapter

from tests.parser.engine.replay_harness import (
    MockTokenizer,
    assert_parse_output,
    collect_output,
    make_mock_tokenizer,
    replay_streaming,
)
from tests.parser.engine.trace_builder import _BUILDERS, build_samples
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.parser.abstract_parser import DelegatingParser, Parser
from vllm.parser.engine import registered_adapters as _adapters_mod
from vllm.parser.engine.adapters import (
    ParserEngineReasoningAdapter,
    ParserEngineToolAdapter,
)

_TOOLS_VALIDATOR = TypeAdapter(list[ChatCompletionToolsParam])

# ── Pairing discovery ────────────────────────────────────────────────


class _PairingInfo(NamedTuple):
    parser_cls: type[Parser]
    name: str
    samples: tuple


def _discover_pairings() -> list[_PairingInfo]:
    """Discover valid delegating pairings from registered engine adapters.

    Groups tool and reasoning adapters by their engine class, then builds
    a DelegatingParser subclass for each engine that has both adapters
    and a test builder.
    """
    bare_tok = MockTokenizer(vocab={}, tokens=[])
    engines: dict[type, dict[str, type]] = {}
    for obj in vars(_adapters_mod).values():
        if not isinstance(obj, type):
            continue
        if (
            issubclass(obj, ParserEngineToolAdapter)
            and obj is not ParserEngineToolAdapter
        ):
            tool_adapter: type[ParserEngineToolAdapter] = obj
            engines.setdefault(tool_adapter._parser_engine_cls, {})["tool"] = obj
        elif (
            issubclass(obj, ParserEngineReasoningAdapter)
            and obj is not ParserEngineReasoningAdapter
        ):
            reasoning_adapter: type[ParserEngineReasoningAdapter] = obj
            engines.setdefault(reasoning_adapter._parser_engine_cls, {})[
                "reasoning"
            ] = obj

    found: list[_PairingInfo] = []
    missing_builders: list[str] = []
    for engine_cls, adapters in engines.items():
        if "tool" not in adapters or "reasoning" not in adapters:
            continue
        cfg = engine_cls(bare_tok, None).parser_engine_config
        if cfg.name not in _BUILDERS:
            missing_builders.append(f"{engine_cls.__name__} (config.name={cfg.name!r})")
            continue

        parser_cls = type(
            f"_Delegating{engine_cls.__name__}",
            (DelegatingParser,),
            {
                "reasoning_parser_cls": adapters["reasoning"],
                "tool_parser_cls": adapters["tool"],
            },
        )
        found.append(
            _PairingInfo(
                parser_cls=parser_cls,
                name=cfg.name,
                samples=build_samples(cfg.name),
            )
        )
    if missing_builders:
        raise RuntimeError(
            f"Engine adapters in registered_adapters have no test builder "
            f"in trace_builder._BUILDERS: {', '.join(missing_builders)}. "
            f"Add a builder to _BUILDERS for each new parser."
        )
    found.sort(key=lambda p: p.name)
    return found


_PAIRINGS = _discover_pairings()

_ALL_SAMPLES = [(p.parser_cls, s) for p in _PAIRINGS for s in p.samples]

CHUNK_SIZES = [1, 2, 3, 5, 11, 23, None]


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
@pytest.mark.parametrize(
    "parser_cls,sample",
    _ALL_SAMPLES,
    ids=lambda v: v.id if hasattr(v, "id") else "",
)
def test_delegating_replay(parser_cls, sample, chunk_size):
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
