# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data-driven replay harness for parser engine testing.

Replays token sequences through parsers at different chunk sizes to
verify chunk-size invariance: the same token sequence must produce
identical output regardless of how tokens are batched.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage


@dataclass
class Sample:
    """One test sample loaded from a JSONL file."""

    id: str
    description: str
    source: str
    vocab: dict[str, int]
    tokens: list[tuple[int, str]]
    expected_reasoning: str | None
    expected_content: str | None
    expected_tool_calls: list[dict] | None
    tools: list[dict] | None = None
    chat_template_kwargs: dict | None = None
    prompt_token_ids: list[int] | None = None


@dataclass
class ParseOutput:
    """Accumulated parse output from replaying a token stream."""

    reasoning: str = ""
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)


class MockTokenizer:
    """Lightweight tokenizer mock that avoids unittest.mock overhead.

    Used by ``benchmarks/benchmark_parsers.py`` in tight timing loops,
    so hot-path methods (``decode``, ``get_vocab``) must be cheap.
    MagicMock's call-recording machinery added ~40% overhead to small-
    sample benchmarks, inflating the per-token cost of the parser engine.
    """

    __slots__ = (
        "_vocab",
        "_token_ids",
        "_token_decode_map",
        "_special_ids",
        "eos_token_id",
        "bos_token_id",
        "pad_token_id",
    )

    def __init__(
        self,
        vocab: dict[str, int],
        tokens: list[tuple[int, str]],
    ) -> None:
        self._vocab = vocab
        self._token_ids = [tid for tid, _ in tokens]
        self._token_decode_map = {tid: text for tid, text in tokens}
        self._special_ids = set(vocab.values())
        self.eos_token_id = None
        self.bos_token_id = None
        self.pad_token_id = None

    def set_vocab(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def encode(self, text: str, **kwargs) -> list[int]:
        return self._token_ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        parts: list[str] = []
        for tid in ids:
            if skip_special_tokens and tid in self._special_ids:
                continue
            text = self._token_decode_map.get(tid, f"?{tid}?")
            parts.append(text)
        return "".join(parts)


CHUNK_SIZES = [1, 2, 3, 5, 11, 23, None]


def make_mock_tokenizer(sample: Sample) -> MockTokenizer:
    """Build a mock tokenizer from a sample's vocab and token data."""
    return MockTokenizer(
        vocab=dict(sample.vocab),
        tokens=sample.tokens,
    )


def _test_request(
    tools: list[dict] | None = None,
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        tools=tools,
    )


def replay_streaming(
    parser,
    tokens: list[tuple[int, str]],
    chunk_size: int | None = None,
    holdback_chars: int = 0,
    finished_on_last: bool = False,
    tools: list[dict] | None = None,
    prompt_token_ids: list[int] | None = None,
) -> list[DeltaMessage | None]:
    """Feed tokens through ``parser.parse_delta()`` at a given chunk size.

    Args:
        parser: A :class:`Parser` instance with ``parse_delta()`` method.
        tokens: List of ``(token_id, decoded_text)`` pairs.
        chunk_size: Number of tokens per batch. ``None`` means all at once.
        holdback_chars: Simulate detokenizer holdback by holding back
            this many characters of decoded text between batches.
        finished_on_last: When True, pass ``finished=True`` on the last
            ``parse_delta()`` call, matching real server behavior.
        tools: Optional tool definitions to include on the request,
            matching the serving layer where tools set
            ``tool_choice`` to ``"auto"``.

    Returns:
        List of ``DeltaMessage`` results from each ``parse_delta()`` call.
    """
    if chunk_size is None:
        chunk_size = len(tokens)

    results: list[DeltaMessage | None] = []
    all_ids = [tid for tid, _ in tokens]
    all_texts = [text for _, text in tokens]

    request = _test_request(tools=tools)
    first_prompt_ids = prompt_token_ids if prompt_token_ids is not None else []

    if holdback_chars <= 0:
        chunks = list(range(0, len(tokens), chunk_size))
        for i, start in enumerate(chunks):
            batch_end = min(start + chunk_size, len(tokens))
            batch_ids = all_ids[start:batch_end]
            delta_text = "".join(all_texts[start:batch_end])
            is_last = i == len(chunks) - 1

            result = parser.parse_delta(
                delta_text,
                batch_ids,
                request,
                prompt_token_ids=first_prompt_ids if start == 0 else None,
                finished=finished_on_last and is_last,
            )
            results.append(result)
        return results

    emitted_up_to = 0
    is_first = True

    for start in range(0, len(tokens), chunk_size):
        batch_end = min(start + chunk_size, len(tokens))

        if batch_end < len(tokens):
            held_chars = 0
            safe_end = batch_end
            while safe_end > emitted_up_to and held_chars < holdback_chars:
                safe_end -= 1
                held_chars += len(all_texts[safe_end])
        else:
            safe_end = batch_end

        if safe_end <= emitted_up_to:
            continue

        batch_ids = all_ids[emitted_up_to:safe_end]
        delta_text = "".join(all_texts[emitted_up_to:safe_end])
        emitted_up_to = safe_end

        is_last_chunk = batch_end >= len(tokens)
        result = parser.parse_delta(
            delta_text,
            batch_ids,
            request,
            prompt_token_ids=first_prompt_ids if is_first else None,
            finished=finished_on_last and is_last_chunk,
        )
        results.append(result)
        is_first = False

    if emitted_up_to < len(tokens):
        batch_ids = all_ids[emitted_up_to:]
        delta_text = "".join(all_texts[emitted_up_to:])
        result = parser.parse_delta(
            delta_text,
            batch_ids,
            request,
            prompt_token_ids=first_prompt_ids if is_first else None,
            finished=finished_on_last,
        )
        results.append(result)

    return results


def replay_with_text_holdback(
    parser,
    tokens: list[tuple[int, str]],
    text_delay: int = 1,
    tools: list[dict] | None = None,
    prompt_token_ids: list[int] | None = None,
) -> list[DeltaMessage | None]:
    """Replay token-by-token with text arriving *text_delay* steps late.

    Simulates the production detokenizer holdback where token IDs arrive
    immediately but decoded text is delayed.  On the last token all
    remaining held-back text is flushed, matching real server behavior::

        step 0:   ids=[tok0], text=""               (held back)
        step 1:   ids=[tok1], text=tok0_text         (tok0 released)
        ...
        step N-1: ids=[tokN-1], text=remaining_texts (flush all)

    This exercises the TokenIDScanner deferred-terminal path that
    ``replay_streaming`` (which keeps text and IDs aligned) does not.
    """
    results: list[DeltaMessage | None] = []
    request = _test_request(tools=tools)
    first_prompt_ids = prompt_token_ids if prompt_token_ids is not None else []

    n = len(tokens)
    held_texts: list[str] = []

    for i in range(n):
        token_id = tokens[i][0]
        held_texts.append(tokens[i][1])

        is_last = i == n - 1
        if is_last:
            delta_text = "".join(held_texts)
            held_texts.clear()
        elif len(held_texts) > text_delay:
            delta_text = held_texts.pop(0)
        else:
            delta_text = ""

        result = parser.parse_delta(
            delta_text,
            [token_id],
            request,
            prompt_token_ids=first_prompt_ids if i == 0 else None,
            finished=is_last,
        )
        results.append(result)

    return results


def accumulate_deltas(
    deltas: Sequence[DeltaMessage | None],
) -> dict:
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls_by_idx: dict[int, dict] = {}

    for delta in deltas:
        if delta is None:
            continue
        if delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta.content:
            content_parts.append(delta.content)
        if delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.name:
                    existing = tool_calls_by_idx.get(tc.index)
                    if existing is None:
                        tool_calls_by_idx[tc.index] = {
                            "name": tc.function.name,
                            "_args_parts": [tc.function.arguments or ""],
                        }
                    else:
                        existing["_args_parts"].append(tc.function.arguments or "")
                elif tc.function and tc.function.arguments:
                    existing = tool_calls_by_idx.get(tc.index)
                    if existing is not None:
                        existing["_args_parts"].append(tc.function.arguments)

    return {
        "reasoning": "".join(reasoning_parts),
        "content": "".join(content_parts),
        "tool_calls": [
            {"name": tc["name"], "arguments": "".join(tc["_args_parts"])}
            for tc in tool_calls_by_idx.values()
        ],
    }


def collect_output(results: list[DeltaMessage | None]) -> ParseOutput:
    """Accumulate ``DeltaMessage`` results into a :class:`ParseOutput`."""
    result = accumulate_deltas(results)
    return ParseOutput(
        reasoning=result["reasoning"],
        content=result["content"],
        tool_calls=result["tool_calls"],
    )


def assert_parse_output(actual: ParseOutput, sample: Sample) -> None:
    """Compare actual parse output against expected values from a sample."""
    if sample.expected_reasoning is not None:
        assert actual.reasoning == sample.expected_reasoning, (
            f"Reasoning mismatch:\n"
            f"  expected: {sample.expected_reasoning!r}\n"
            f"  actual:   {actual.reasoning!r}"
        )

    if sample.expected_content is not None:
        assert actual.content == sample.expected_content, (
            f"Content mismatch:\n"
            f"  expected: {sample.expected_content!r}\n"
            f"  actual:   {actual.content!r}"
        )
    if sample.expected_tool_calls is not None:
        assert len(actual.tool_calls) == len(sample.expected_tool_calls), (
            f"Tool call count mismatch: "
            f"expected {len(sample.expected_tool_calls)}, "
            f"got {len(actual.tool_calls)}"
        )
        for i, (expected_tc, actual_tc) in enumerate(
            zip(sample.expected_tool_calls, actual.tool_calls)
        ):
            assert actual_tc["name"] == expected_tc["name"], (
                f"Tool call {i} name mismatch: "
                f"expected {expected_tc['name']!r}, "
                f"got {actual_tc['name']!r}"
            )
            if "arguments" in expected_tc:
                expected_args = expected_tc["arguments"]
                actual_args_str = actual_tc.get("arguments", "{}")
                if isinstance(expected_args, dict):
                    try:
                        actual_args = json.loads(actual_args_str)
                    except json.JSONDecodeError as e:
                        raise AssertionError(
                            f"Tool call {i} arguments not valid JSON: "
                            f"{actual_args_str!r}"
                        ) from e
                    assert actual_args == expected_args, (
                        f"Tool call {i} arguments mismatch:\n"
                        f"  expected: {expected_args}\n"
                        f"  actual:   {actual_args}"
                    )


def assert_no_terminal_leakage(
    actual: ParseOutput,
    terminals: list[str],
    context: str = "",
) -> None:
    """Assert that none of *terminals* appear in reasoning or content."""
    suffix = f" ({context})" if context else ""
    for terminal in terminals:
        assert terminal not in actual.reasoning, (
            f"{terminal!r} leaked into reasoning{suffix}"
        )
        assert terminal not in actual.content, (
            f"{terminal!r} leaked into content{suffix}"
        )
