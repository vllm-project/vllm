# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay tests for engine parsers (holdback, skip-tool-parsing, adapters).

Replays dynamically built token sequences at different chunk sizes and
holdback depths to verify chunk-size invariance and terminal-token hygiene.

Parser discovery is automatic: any ``ParserEngine`` subclass registered in
``registered_adapters`` that also has a builder in ``trace_builder._BUILDERS``
is picked up with zero manual wiring.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import pytest

from tests.parser.engine.replay_harness import (
    DUMMY_TOOLS,
    MockTokenizer,
    Sample,
    _test_request,
    assert_no_terminal_leakage,
    assert_parse_output,
    collect_output,
    make_mock_tokenizer,
    parse_non_streaming,
    replay_streaming,
    replay_with_text_holdback,
)
from tests.parser.engine.trace_builder import _BUILDERS, build_samples
from vllm.parser.engine import registered_adapters as _adapters_mod
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import ParserState

# ── Parser discovery ─────────────────────────────────────────────────


class _ParserInfo(NamedTuple):
    parser_cls: type[ParserEngine]
    name: str
    samples: tuple
    terminals: list[str]
    tool_end: str
    think_end: str
    tool_start: str


def _discover_parsers() -> list[_ParserInfo]:
    """Discover engine parsers from registered_adapters that have test builders.

    Returns one ``_ParserInfo`` per parser, sorted by config name.
    Raises ``RuntimeError`` if any registered parser lacks a builder.
    """
    bare_tok = MockTokenizer(vocab={}, tokens=[])
    found: list[_ParserInfo] = []
    missing_builders: list[str] = []
    for obj in vars(_adapters_mod).values():
        if not (
            isinstance(obj, type)
            and issubclass(obj, ParserEngine)
            and obj is not ParserEngine
        ):
            continue
        cfg = obj(bare_tok, None).parser_engine_config
        if cfg.name not in _BUILDERS:
            missing_builders.append(f"{obj.__name__} (config.name={cfg.name!r})")
            continue
        tool_end = cfg.token_id_terminals.get("TOOL_END")
        if not tool_end:
            raise RuntimeError(
                f"{obj.__name__} config missing 'TOOL_END' in token_id_terminals"
            )
        all_vals = set(cfg.terminals.values()) | set(cfg.token_id_terminals.values())
        found.append(
            _ParserInfo(
                parser_cls=obj,
                name=cfg.name,
                samples=build_samples(cfg.name),
                terminals=sorted(v for v in all_vals if len(v) > 1),
                tool_end=tool_end,
                think_end=cfg.terminals.get("THINK_END", ""),
                tool_start=(
                    cfg.terminals["TOOL_SECTION_START"]
                    if (ParserState.CONTENT, "TOOL_SECTION_START") in cfg.transitions
                    else cfg.terminals.get("TOOL_START", "")
                ),
            )
        )
    if missing_builders:
        raise RuntimeError(
            f"Engine parsers in registered_adapters have no test builder "
            f"in trace_builder._BUILDERS: {', '.join(missing_builders)}. "
            f"Add a builder to _BUILDERS for each new parser."
        )
    found.sort(key=lambda p: p.name)
    return found


_PARSERS = _discover_parsers()


def _make_parser(parser_cls: type[ParserEngine], tokenizer, sample: Sample, **extra):
    kwargs = dict(extra)
    if sample.chat_template_kwargs:
        kwargs["chat_template_kwargs"] = sample.chat_template_kwargs
    return parser_cls(tokenizer, sample.tools, **kwargs)


_ENGINE_PARSERS: dict[str, type[ParserEngine]] = {
    f"{p.name}_engine": p.parser_cls for p in _PARSERS
}

# ── Parametrize sample lists ─────────────────────────────────────────

HOLDBACK_CONFIGS = [6, 12, 24]

_REPLAY_SAMPLES = [(p.parser_cls, s, p.terminals) for p in _PARSERS for s in p.samples]


@pytest.mark.parametrize("holdback", HOLDBACK_CONFIGS, ids=lambda h: f"holdback{h}")
@pytest.mark.parametrize("chunk_size", [3, 5, 10], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize(
    "parser_cls,sample,terminals",
    _REPLAY_SAMPLES,
    ids=lambda v: v.id if hasattr(v, "id") else "",
)
class TestReplayWithHoldback:
    """Replay all parsers with simulated detokenizer holdback."""

    def test_replay(self, parser_cls, sample, terminals, chunk_size, holdback):
        tokenizer = make_mock_tokenizer(sample)
        parser = _make_parser(parser_cls, tokenizer, sample)
        deltas = replay_streaming(
            parser,
            sample.tokens,
            chunk_size=chunk_size,
            holdback_chars=holdback,
            prompt_token_ids=sample.prompt_token_ids,
        )
        output = collect_output(deltas)

        assert_parse_output(output, sample)
        assert_no_terminal_leakage(
            output,
            terminals,
            context=f"chunk_size={chunk_size}, holdback={holdback}",
        )


TEXT_HOLDBACK_DELAYS = [1, 2, 3]


@pytest.mark.parametrize("delay", TEXT_HOLDBACK_DELAYS, ids=lambda d: f"delay{d}")
@pytest.mark.parametrize(
    "parser_cls,sample,terminals",
    _REPLAY_SAMPLES,
    ids=lambda v: v.id if hasattr(v, "id") else "",
)
class TestTextHoldback:
    """Replay with production-like text/token-ID misalignment.

    In production the detokenizer sends token IDs immediately but holds
    back text by N tokens.  This exercises the TokenIDScanner deferred
    terminal path that aligned-holdback tests do not cover.
    """

    def test_replay(self, parser_cls, sample, terminals, delay):
        tokenizer = make_mock_tokenizer(sample)
        parser = _make_parser(parser_cls, tokenizer, sample)
        deltas = replay_with_text_holdback(
            parser,
            sample.tokens,
            text_delay=delay,
            prompt_token_ids=sample.prompt_token_ids,
        )
        output = collect_output(deltas)

        assert_parse_output(output, sample)
        assert_no_terminal_leakage(
            output,
            terminals,
            context=f"text_delay={delay}",
        )


@pytest.mark.parametrize(
    "chunk_size", [1, 2, 3, 5, 10, 19, 20, None], ids=lambda c: f"chunk{c}"
)
@pytest.mark.parametrize(
    "parser_cls,sample,terminals",
    _REPLAY_SAMPLES,
    ids=lambda v: v.id if hasattr(v, "id") else "",
)
class TestReplay:
    """Replay all parsers at varied chunk sizes without holdback."""

    def test_replay(self, parser_cls, sample, terminals, chunk_size):
        tokenizer = make_mock_tokenizer(sample)
        parser = _make_parser(parser_cls, tokenizer, sample)
        deltas = replay_streaming(
            parser,
            sample.tokens,
            chunk_size=chunk_size,
            prompt_token_ids=sample.prompt_token_ids,
        )
        output = collect_output(deltas)

        assert_parse_output(output, sample)
        assert_no_terminal_leakage(output, terminals)


_DEFERRAL_SAMPLES = [
    (p.parser_cls, s, p.tool_end)
    for p in _PARSERS
    for s in p.samples
    if s.expected_tool_calls
]


@pytest.mark.parametrize(
    "parser_cls,sample,tool_end_text",
    _DEFERRAL_SAMPLES,
    ids=lambda v: v.id if hasattr(v, "id") else getattr(v, "__name__", ""),
)
class TestDeferralFinish:
    """Test that parse_delta(finished=True) resolves deferred scanner state.

    Simulates a production failure where delta_text is missing the
    tool-call-end text but delta_token_ids has the token, causing the
    scanner to defer it.  Without finish(), the deferred state is lost
    and tool call arguments are empty.
    """

    def test_misaligned_last_delta_with_finish(self, parser_cls, sample, tool_end_text):
        tokenizer = make_mock_tokenizer(sample)
        parser = _make_parser(parser_cls, tokenizer, sample)

        request = _test_request()

        all_ids = [tid for tid, _ in sample.tokens]
        all_texts = [text for _, text in sample.tokens]

        tool_end_id = sample.vocab.get(tool_end_text)
        split_idx = None
        for i in range(len(all_ids) - 1, -1, -1):
            if all_ids[i] == tool_end_id:
                split_idx = i
                break

        if split_idx is None:
            pytest.skip(f"no {tool_end_text} token found")

        first_ids = all_ids[:split_idx]
        first_text = "".join(all_texts[:split_idx])

        last_ids = all_ids[split_idx:]
        last_text_missing = "".join(all_texts[split_idx:]).replace(tool_end_text, "")

        result1 = parser.parse_delta(
            first_text,
            first_ids,
            request,
            prompt_token_ids=[],
            finished=False,
        )
        result2 = parser.parse_delta(
            last_text_missing, last_ids, request, finished=True
        )

        output = collect_output([result1, result2])

        tool_calls_only = dataclasses.replace(
            sample, expected_reasoning=None, expected_content=None
        )
        assert_parse_output(output, tool_calls_only)


@pytest.mark.parametrize(
    "parser_cls,sample",
    [(p.parser_cls, p.samples[0]) for p in _PARSERS],
    ids=[p.name for p in _PARSERS],
)
class TestParserEngineAdjustRequest:
    """Verify ParserEngine and its adapters set skip_special_tokens=False."""

    def test_adjust_request_disables_skip_special_tokens(self, parser_cls, sample):
        tokenizer = make_mock_tokenizer(sample)
        parser = parser_cls(tokenizer, sample.tools)
        request = _test_request()
        assert request.skip_special_tokens is True
        adjusted = parser.adjust_request(request)
        assert adjusted.skip_special_tokens is False


_TOOL_CALL_SAMPLES = [
    (p.parser_cls, s, p.think_end, p.tool_start)
    for p in _PARSERS
    for s in p.samples
    if s.expected_tool_calls and s.expected_reasoning
]


def _tool_suppression_expectations(
    sample, think_end: str, tool_start: str, *, include_tool_block: bool
) -> tuple[str, str]:
    """Expected (reasoning, content) when tool calls are not extracted.

    With ``include_tool_block=True`` (skip_tool_parsing / reasoning
    adapter first pass), tool terminal text is preserved as content so
    a second-pass parser can see it.

    With ``include_tool_block=False`` (_suppress_tool_calls /
    tool_choice='none'), the state machine consumes tool blocks and
    only non-tool content survives.
    """
    full_text = "".join(text for _, text in sample.tokens)
    reasoning = sample.expected_reasoning
    idx = full_text.find(reasoning)
    if idx < 0:
        return (full_text, "")
    after_reasoning = full_text[idx + len(reasoning) :]
    if think_end:
        pos = after_reasoning.find(think_end)
        if pos >= 0:
            if include_tool_block:
                return (reasoning, after_reasoning[pos + len(think_end) :])
            after_reasoning = after_reasoning[pos + len(think_end) :]
    if tool_start:
        pos = after_reasoning.find(tool_start)
        if pos >= 0:
            if include_tool_block:
                return (reasoning, after_reasoning[pos:])
            return (reasoning, after_reasoning[:pos])
    if include_tool_block:
        return (full_text, "")
    return (reasoning, after_reasoning)


@pytest.mark.parametrize("chunk_size", [1, 5, None], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize(
    "mode",
    ["skip_tool_parsing", "suppress_tool_calls"],
    ids=["skip_tool_parsing", "suppress_tool_calls"],
)
@pytest.mark.parametrize(
    "parser_cls,sample,think_end,tool_start",
    _TOOL_CALL_SAMPLES,
    ids=lambda v: v.id if hasattr(v, "id") else getattr(v, "__name__", ""),
)
class TestToolCallFilteringReplay:
    """Replay with tool calls not extracted, in both filtering modes.

    ``skip_tool_parsing`` (reasoning adapter first pass): tool terminal
    text is preserved as content for a second-pass tool parser.

    ``suppress_tool_calls`` (tool_choice='none'): tool call blocks are
    consumed by the state machine and do not leak into content.
    """

    def test_replay(self, parser_cls, sample, think_end, tool_start, mode, chunk_size):
        tokenizer = make_mock_tokenizer(sample)
        kwargs = {}
        if sample.chat_template_kwargs:
            kwargs["chat_template_kwargs"] = sample.chat_template_kwargs
        parser = parser_cls(tokenizer, **kwargs)

        request = _test_request()
        request.tools = DUMMY_TOOLS
        if mode == "skip_tool_parsing":
            parser.skip_tool_parsing = True
        else:
            request.tool_choice = "none"

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
                prompt_token_ids=(sample.prompt_token_ids or [])
                if start == 0
                else None,
                finished=is_last,
            )
            results.append(result)

        output = collect_output(results)

        include_block = mode == "skip_tool_parsing"
        expected_reasoning, expected_content = _tool_suppression_expectations(
            sample, think_end, tool_start, include_tool_block=include_block
        )

        assert output.reasoning == expected_reasoning, (
            f"Reasoning mismatch (mode={mode}):\n"
            f"  expected: {expected_reasoning!r}\n"
            f"  actual:   {output.reasoning!r}"
        )
        assert output.tool_calls == [], (
            f"Expected no tool calls (mode={mode}) but got {output.tool_calls}"
        )
        assert output.content == expected_content, (
            f"Content mismatch (mode={mode}):\n"
            f"  expected: {expected_content!r}\n"
            f"  actual:   {output.content!r}"
        )


@pytest.mark.parametrize(
    "parser_cls,sample,think_end,tool_start",
    _TOOL_CALL_SAMPLES,
    ids=lambda v: v.id if hasattr(v, "id") else getattr(v, "__name__", ""),
)
class TestToolCallFilteringNonStreaming:
    """Non-streaming parse() with tool_choice='none' must suppress tool
    calls and not leak special tokens into content."""

    def test_parse(self, parser_cls, sample, think_end, tool_start):
        tokenizer = make_mock_tokenizer(sample)
        kwargs = {}
        if sample.chat_template_kwargs:
            kwargs["chat_template_kwargs"] = sample.chat_template_kwargs
        parser = parser_cls(tokenizer, **kwargs)

        request = _test_request()
        request.tools = DUMMY_TOOLS
        request.tool_choice = "none"

        output = parse_non_streaming(parser, sample, request)

        expected_reasoning, expected_content = _tool_suppression_expectations(
            sample, think_end, tool_start, include_tool_block=False
        )
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


_WS_TOOL_SAMPLES = [(t[0], t[1]) for t in _TOOL_CALL_SAMPLES if "whitespace" in t[1].id]


@pytest.mark.parametrize(
    "parser_cls,sample",
    _WS_TOOL_SAMPLES,
    ids=lambda v: v.id if hasattr(v, "id") else getattr(v, "__name__", ""),
)
class TestToolChoiceNoneStreamingParity:
    """Streaming and non-streaming must return the same content
    when tool_choice='none' suppresses tool calls."""

    def test_content_matches(self, parser_cls, sample):
        tokenizer = make_mock_tokenizer(sample)
        kwargs = {}
        if sample.chat_template_kwargs:
            kwargs["chat_template_kwargs"] = sample.chat_template_kwargs
        request = _test_request()
        request.tools = DUMMY_TOOLS
        request.tool_choice = "none"

        ns_output = parse_non_streaming(
            parser_cls(tokenizer, **kwargs),
            sample,
            request,
        )

        s_parser = parser_cls(tokenizer, **kwargs)
        results = []
        for i, (tid, text) in enumerate(sample.tokens):
            is_last = i == len(sample.tokens) - 1
            results.append(
                s_parser.parse_delta(
                    text,
                    [tid],
                    request,
                    prompt_token_ids=(sample.prompt_token_ids or [])
                    if i == 0
                    else None,
                    finished=is_last,
                )
            )
        s_output = collect_output(results)

        assert ns_output.content == s_output.content, (
            f"Streaming/non-streaming content mismatch:\n"
            f"  streaming:     {s_output.content!r}\n"
            f"  non-streaming: {ns_output.content!r}"
        )


_DROP_TOKENS = {"<bos>": 99990, "<eos>": 99991}


def _inject_drop_tokens(sample):
    """Insert <bos> at stream start and <eos> between the first two tokens."""
    new_vocab = {**sample.vocab, **_DROP_TOKENS}
    tokens = list(sample.tokens)
    tokens.insert(0, (99990, "<bos>"))
    if len(tokens) >= 3:
        tokens.insert(2, (99991, "<eos>"))
    else:
        tokens.append((99991, "<eos>"))
    return dataclasses.replace(sample, vocab=new_vocab, tokens=tokens)


class TestDropTokenReplay:
    """Verify unconfigured special tokens are silently dropped across
    all parsers and chunk sizes."""

    @pytest.mark.parametrize(
        "parser_info",
        _PARSERS,
        ids=[p.name for p in _PARSERS],
    )
    @pytest.mark.parametrize("chunk_size", [1, 3, None])
    def test_drop_tokens_removed_from_output(self, parser_info, chunk_size):
        for sample in parser_info.samples:
            injected = _inject_drop_tokens(sample)
            tokenizer = make_mock_tokenizer(injected)
            parser = _make_parser(parser_info.parser_cls, tokenizer, sample)

            results = replay_streaming(
                parser,
                injected.tokens,
                chunk_size=chunk_size,
                tools=sample.tools,
                prompt_token_ids=sample.prompt_token_ids,
            )
            output = collect_output(results)

            assert_no_terminal_leakage(
                output,
                list(_DROP_TOKENS.keys()),
                context=f"parser={parser_info.name}, chunk={chunk_size}",
            )
            assert_parse_output(output, sample)


class TestDropTokenNonStreaming:
    """Non-streaming parse() must also strip unconfigured special tokens."""

    @pytest.mark.parametrize(
        "parser_info",
        _PARSERS,
        ids=[p.name for p in _PARSERS],
    )
    def test_drop_tokens_removed_from_output(self, parser_info):
        for sample in parser_info.samples:
            injected = _inject_drop_tokens(sample)
            tokenizer = make_mock_tokenizer(injected)
            parser = _make_parser(parser_info.parser_cls, tokenizer, sample)

            request = _test_request(tools=sample.tools)
            output = parse_non_streaming(parser, injected, request)

            assert_no_terminal_leakage(
                output,
                list(_DROP_TOKENS.keys()),
                context=f"parser={parser_info.name}",
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
