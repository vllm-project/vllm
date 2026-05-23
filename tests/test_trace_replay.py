#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Trace-replay test for Qwen3 reasoning and XML tool-call parsers.

Parses a VLLM_RESPONSE_TRACE_LOG file, reconstructs streaming sessions,
feeds each session chunk-by-chunk through both parsers, and asserts correct
behaviour.  Also contains targeted synthetic tests for known edge cases.

Usage:
    .venv/bin/python tests/test_trace_replay.py /tmp/failure.txt

Exit code 0 = all tests passed.  Exit code 1 = at least one failure.

Trace file format (one entry per chunk):
    [ISO-8601-timestamp] byte_count\n
    raw_text\n
    \n

Sessions are identified by gaps larger than SESSION_GAP_SECONDS between
consecutive chunk timestamps.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Minimal imports from vllm — no GPU required
# ---------------------------------------------------------------------------
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.tool_parsers.qwen3xml_tool_parser import StreamingXMLToolCallParser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_GAP_SECONDS: float = 2.0

# Special token IDs matching the Qwen3 tokenizer vocabulary.
_THINK_ID = 151648
_END_THINK_ID = 151649
_TOOL_CALL_ID = 151657
_TOOL_CALL_END_ID = 151658
# Qwen3 uses <function as a special prefix token (not just regular text).
# We need it in the vocab so the 8-token lookahead in _tool_call_is_genuine_end
# can decode "\n<function=bash>" from a compact window of IDs.
_FUNCTION_PREFIX_ID = 151652

# ---------------------------------------------------------------------------
# Minimal mock tokenizer for Qwen3 special tokens
# ---------------------------------------------------------------------------

_SPECIAL_TOKENS: dict[str, int] = {
    "<think>": _THINK_ID,
    "</think>": _END_THINK_ID,
    "<tool_call>": _TOOL_CALL_ID,
    "</tool_call>": _TOOL_CALL_END_ID,
    "<function": _FUNCTION_PREFIX_ID,
}
_SPECIAL_IDS: dict[int, str] = {v: k for k, v in _SPECIAL_TOKENS.items()}
# Split on any of the special token strings, longest first to avoid prefix
# ambiguity (e.g. "<tool_call>" before "<tool_").
_SPLIT_RE = re.compile(
    r"(<think>|</think>|<tool_call>|</tool_call>|<function)"
)

# Offset added to ord(char) to produce a unique, non-colliding token ID.
_CHAR_ID_OFFSET = 200_000


class MockQwen3Tokenizer:
    """
    Minimal stand-in for the Qwen3 tokenizer.

    Only purpose: return correct token IDs for the special tokens
    (<think>, </think>, <tool_call>, </tool_call>) and decode those IDs
    back to strings.  All other characters are tokenised one-per-character
    with IDs derived from their code-point to keep decode() invertible.
    """

    def get_vocab(self) -> dict[str, int]:
        vocab: dict[str, int] = dict(_SPECIAL_TOKENS)
        # Add every printable ASCII character so decode() can reconstruct
        # the look-ahead window in _tool_call_is_genuine_end.
        for c in (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            ' \t\n\r=_-<>/.,;:!?\"\\'
            "`'@#$%^&*()[]{}|~+"
        ):
            vocab[c] = ord(c) + _CHAR_ID_OFFSET
        return vocab

    def tokenize(self, text: str) -> list[str]:
        """Split *text* into token strings, keeping special tokens atomic."""
        tokens: list[str] = []
        for part in _SPLIT_RE.split(text):
            if part in _SPECIAL_TOKENS:
                tokens.append(part)
            else:
                tokens.extend(list(part))  # one character per token
        return [t for t in tokens if t]

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> str:
        chars: list[str] = []
        for tid in ids:
            if tid in _SPECIAL_IDS:
                if not skip_special_tokens:
                    chars.append(_SPECIAL_IDS[tid])
            elif tid > _CHAR_ID_OFFSET:
                chars.append(chr(tid - _CHAR_ID_OFFSET))
        return "".join(chars)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [
            _SPECIAL_IDS.get(i, f"<unk_{i}>") for i in ids
        ]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)


# ---------------------------------------------------------------------------
# Trace-file parser
# ---------------------------------------------------------------------------

@dataclass
class TraceChunk:
    timestamp: datetime
    text: str


def parse_trace(path: Path) -> list[TraceChunk]:
    """
    Parse a VLLM_RESPONSE_TRACE_LOG file into a flat list of TraceChunks.

    Each entry in the file looks like::

        [2026-04-27T11:33:55.755Z] 3
        The

    (timestamp+byte-count on one line, then the raw text, then a blank line)
    """
    raw = path.read_text(encoding="utf-8")
    chunks: list[TraceChunk] = []

    # Pattern: '[' ISO-timestamp ']' SPACE byte-count NEWLINE text NEWLINE NEWLINE
    # We split on the header lines to group entries.
    header_re = re.compile(
        r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\]\s+\d+\s*$",
        re.MULTILINE,
    )
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = header_re.match(line)
        if not m:
            i += 1
            continue
        ts = datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
        i += 1
        # Collect lines until the next header (or EOF), stripping the blank
        # separator lines but preserving internal newlines.
        text_lines: list[str] = []
        while i < len(lines) and not header_re.match(lines[i]):
            text_lines.append(lines[i])
            i += 1
        # The file writer adds a trailing '\n' after the text and a blank
        # separator line; when split by line, this produces one or more
        # trailing empty strings — trim them.
        while text_lines and text_lines[-1] == "":
            text_lines.pop()
        text = "\n".join(text_lines)
        if text or True:  # keep even empty chunks
            chunks.append(TraceChunk(timestamp=ts, text=text))
    return chunks


def split_sessions(
    chunks: list[TraceChunk],
    gap: float = SESSION_GAP_SECONDS,
) -> list[list[TraceChunk]]:
    """Group chunks into sessions separated by timestamp gaps > *gap* seconds."""
    if not chunks:
        return []
    sessions: list[list[TraceChunk]] = [[chunks[0]]]
    for chunk in chunks[1:]:
        prev_ts = sessions[-1][-1].timestamp
        if (chunk.timestamp - prev_ts).total_seconds() > gap:
            sessions.append([])
        sessions[-1].append(chunk)
    return sessions


# ---------------------------------------------------------------------------
# Streaming replay helpers
# ---------------------------------------------------------------------------

def _token_ids_for_delta(tokenizer: MockQwen3Tokenizer, delta: str) -> list[int]:
    """
    Return the token IDs that would be in ``delta_token_ids`` for *delta*.
    Only IDs that appear in the parser's vocab are emitted (mirrors the
    behaviour in tests/reasoning/utils.py).
    """
    vocab = tokenizer.get_vocab()
    return [
        vocab[tok]
        for tok in tokenizer.tokenize(delta)
        if tok in vocab
    ]


def replay_reasoning_streaming(
    chunks: list[TraceChunk],
) -> tuple[Optional[str], Optional[str]]:
    """
    Replay *chunks* through the Qwen3 reasoning parser in streaming mode.

    Returns (reasoning, content) after consuming all chunks.
    """
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser

    tokenizer = MockQwen3Tokenizer()
    parser = Qwen3ReasoningParser(tokenizer)

    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    previous_text = ""
    previous_token_ids: list[int] = []

    for chunk in chunks:
        delta_text = chunk.text
        delta_ids = _token_ids_for_delta(tokenizer, delta_text)
        current_text = previous_text + delta_text
        current_ids = previous_token_ids + delta_ids

        delta_msg: Optional[DeltaMessage] = parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_ids,
            delta_token_ids=delta_ids,
        )
        if delta_msg is not None:
            if delta_msg.reasoning:
                reasoning_parts.append(delta_msg.reasoning)
            if delta_msg.content:
                content_parts.append(delta_msg.content)

        previous_text = current_text
        previous_token_ids = current_ids

    reasoning = "".join(reasoning_parts) or None
    content = "".join(content_parts) or None
    return reasoning, content


def replay_tool_parser_streaming(
    chunks: list[TraceChunk],
) -> DeltaMessage:
    """
    Feed *chunks* character-by-character through the XML tool-call parser.

    Returns the merged result after all chunks + finalize().
    """
    parser = StreamingXMLToolCallParser()
    for chunk in chunks:
        parser.parse_single_streaming_chunks(chunk.text)
    parser.finalize()
    return parser.collect_all()


# ---------------------------------------------------------------------------
# Test framework
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""


_results: list[TestResult] = []


def _pass(name: str) -> TestResult:
    r = TestResult(name=name, passed=True)
    _results.append(r)
    print(f"  PASS  {name}")
    return r


def _fail(name: str, msg: str) -> TestResult:
    r = TestResult(name=name, passed=False, message=msg)
    _results.append(r)
    print(f"  FAIL  {name}")
    print(f"        {msg}")
    return r


def check(name: str, condition: bool, msg: str = "") -> TestResult:
    return _pass(name) if condition else _fail(name, msg or "condition was False")


# ---------------------------------------------------------------------------
# XML tool-parser tests (no tokenizer needed)
# ---------------------------------------------------------------------------

def test_xml_tool_parser_basic_tool_call() -> None:
    """A well-formed tool call is parsed correctly."""
    xml = (
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>\necho hello\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    result = replay_tool_parser_streaming(
        [TraceChunk(timestamp=_now(), text=xml)]
    )
    check(
        "xml/basic_tool_call/produces_tool_call",
        bool(result.tool_calls),
        f"Expected tool call, got none. result={result}",
    )
    if result.tool_calls:
        check(
            "xml/basic_tool_call/function_name",
            result.tool_calls[0].function.name == "bash",
            f"Expected 'bash', got {result.tool_calls[0].function.name!r}",
        )
        import json
        args = json.loads(result.tool_calls[0].function.arguments)
        check(
            "xml/basic_tool_call/arguments",
            args == {"command": "echo hello"},
            f"Unexpected args: {args}",
        )


def test_xml_tool_parser_fake_tool_call_no_function() -> None:
    """
    Plain text mentioning <tool_call> without any <function=…> must not
    produce a tool call.
    """
    text = (
        "I'll explain the <tool_call> format and how it works.\n"
        "No actual function is being called here."
    )
    result = replay_tool_parser_streaming(
        [TraceChunk(timestamp=_now(), text=text)]
    )
    check(
        "xml/no_function_after_tool_call/no_tool_calls",
        not result.tool_calls,
        f"Expected no tool calls, got {result.tool_calls}",
    )
    check(
        "xml/no_function_after_tool_call/content_preserved",
        result.content is not None and "explain" in result.content,
        f"Content not preserved: {result.content!r}",
    )


def test_xml_tool_parser_function_placeholder_false_positive() -> None:
    """
    <tool_call> followed by literal '<function=...>' (a documentation
    placeholder) must NOT produce a tool call.

    This is the pattern observed in the failure trace around 11:39:14:
      via `"<tool_call>"` tag followed by `<function=...>`. Uses a lookahead…
    """
    texts = [
        # Exact trace pattern 1: "followed by `<function=...>`"
        'via `"<tool_call>"` tag followed by `<function=...>`. Uses a lookahead.',
        # Exact trace pattern 2: tool_call + function=... on separate lines
        '. Implicit end via `"<tool_call>"` + `<function=...>`: split at that point',
        # Exact trace pattern 3: TOOL_CALL_PENDING description
        '`TOOL_CALL_PENDING` — saw `"<tool_call>"`, waiting for `<function=...>` lookahead',
    ]
    for i, text in enumerate(texts):
        parser = StreamingXMLToolCallParser()
        parser.parse_single_streaming_chunks(text)
        parser.finalize()
        result = parser.collect_all()
        check(
            f"xml/function_placeholder_false_positive/{i}",
            not result.tool_calls,
            f"False positive: got tool_calls={result.tool_calls!r} "
            f"from markdown text {text[:60]!r}…",
        )


def test_xml_tool_parser_streaming_chunked() -> None:
    """
    A tool call split across many small chunks (simulating real streaming)
    must be assembled correctly.
    """
    full = (
        "<tool_call>\n"
        "<function=read>\n"
        "<parameter=path>\n/etc/hosts\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    # Feed one character at a time
    parser = StreamingXMLToolCallParser()
    for ch in full:
        parser.parse_single_streaming_chunks(ch)
    parser.finalize()
    result = parser.collect_all()
    check(
        "xml/streaming_chunked/produces_tool_call",
        bool(result.tool_calls),
        f"Expected tool call from char-by-char feed, got none",
    )
    if result.tool_calls:
        import json
        args = json.loads(result.tool_calls[0].function.arguments)
        check(
            "xml/streaming_chunked/arguments",
            args == {"path": "/etc/hosts"},
            f"Unexpected args: {args}",
        )


def test_xml_tool_parser_trace_session(session_chunks: list[TraceChunk]) -> None:
    """
    Feed the content portion of the trace session to the XML tool parser.

    The session contains markdown prose that mentions <tool_call> and
    <function=...> as documentation examples.  No real tool call should
    be produced.
    """
    full_text = "".join(c.text for c in session_chunks)

    # Split at </think> — content portion is what goes to the tool parser.
    if "</think>" in full_text:
        content_text = full_text.split("</think>", 1)[1]
    else:
        content_text = full_text

    parser = StreamingXMLToolCallParser()
    # Feed in the same chunk boundaries as the real trace.
    offset = 0
    end_of_reasoning = False
    for chunk in session_chunks:
        if not end_of_reasoning:
            if "</think>" in chunk.text:
                end_of_reasoning = True
                # Feed only the part after </think>
                post = chunk.text.split("</think>", 1)[1]
                if post:
                    parser.parse_single_streaming_chunks(post)
        else:
            parser.parse_single_streaming_chunks(chunk.text)

    parser.finalize()
    result = parser.collect_all()

    # Any tool call with a function name of "..." (three dots) is a false positive.
    false_positives = [
        tc for tc in (result.tool_calls or [])
        if tc.function and tc.function.name in ("...", "…", ".")
    ]
    check(
        "xml/trace_session/no_placeholder_tool_calls",
        not false_positives,
        f"False-positive tool calls from markdown text: {false_positives}",
    )

    # The session should not contain any real tool calls (it's pure content).
    real_tool_calls = [
        tc for tc in (result.tool_calls or [])
        if tc.function and tc.function.name not in ("...", "…", ".")
    ]
    check(
        "xml/trace_session/no_spurious_real_tool_calls",
        not real_tool_calls,
        f"Unexpected tool calls: {real_tool_calls}",
    )


# ---------------------------------------------------------------------------
# Reasoning-parser tests (requires mock tokenizer)
# ---------------------------------------------------------------------------

def test_reasoning_think_end() -> None:
    """</think> correctly ends reasoning."""
    chunks = _text_to_chunks(
        "I need to think about this.\n</think>\nHere is my answer."
    )
    reasoning, content = replay_reasoning_streaming(chunks)
    check(
        "reasoning/think_end/reasoning_correct",
        reasoning is not None and "think about" in reasoning,
        f"reasoning={reasoning!r}",
    )
    check(
        "reasoning/think_end/content_correct",
        content is not None and "answer" in content,
        f"content={content!r}",
    )


def test_reasoning_tool_call_genuine_end() -> None:
    """
    A genuine tool call (<tool_call> immediately followed by <function=…>)
    ends reasoning; content starts at <tool_call>.

    Feed <tool_call> and <function=bash> together in one chunk — in real
    vLLM the model typically emits both in close succession (often the same
    decode step), so this is the realistic path for the lookahead to fire.
    """
    # One chunk containing both <tool_call> and the start of <function=bash>
    # ensures that when the <tool_call> token arrives in delta_ids, the
    # 8-token lookahead window already includes <function.
    chunks = [
        TraceChunk(timestamp=_now(), text="Let me run a command.\n\n"),
        TraceChunk(
            timestamp=_now(),
            text="<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>",
        ),
    ]
    reasoning, content = replay_reasoning_streaming(chunks)
    check(
        "reasoning/genuine_tool_call/reasoning_not_none",
        reasoning is not None,
        f"reasoning={reasoning!r}",
    )
    check(
        "reasoning/genuine_tool_call/content_starts_with_tool_call",
        content is not None and "<tool_call>" in content,
        f"content={content!r}",
    )


def test_reasoning_fake_tool_call_in_reasoning() -> None:
    """
    <tool_call> token appearing inside reasoning text — NOT followed by
    <function=…> — must NOT terminate reasoning prematurely.

    This tests the lookahead in extract_reasoning_streaming.
    The model is writing explanation text that happens to contain the
    <tool_call> special token but it's not a real tool call.
    """
    # Deliver in two chunks so <tool_call> token arrives alone in delta_ids,
    # followed by ordinary text (no <function=…>).
    chunks = [
        TraceChunk(timestamp=_now(), text="I'll describe the "),
        TraceChunk(timestamp=_now(), text="<tool_call>"),  # special token, no function follows
        TraceChunk(timestamp=_now(), text=" syntax used by the model.\n"),
        TraceChunk(timestamp=_now(), text="More reasoning here.\n"),
    ]
    reasoning, content = replay_reasoning_streaming(chunks)
    check(
        "reasoning/fake_tool_call/stays_as_reasoning",
        reasoning is not None and "<tool_call>" in reasoning,
        f"Expected '<tool_call>' in reasoning, got reasoning={reasoning!r}, content={content!r}",
    )
    check(
        "reasoning/fake_tool_call/no_premature_content",
        content is None,
        f"Reasoning terminated prematurely: content={content!r}",
    )


def test_reasoning_fake_tool_call_followed_by_non_function() -> None:
    """
    <tool_call> followed by text (not <function=…>) in reasoning must stay as reasoning.
    """
    # Simulate: `"<tool_call>"` tag followed by `<function=...>` — the full
    # string from the trace, delivered in realistic chunk sizes.
    chunks = [
        TraceChunk(timestamp=_now(), text="Handling: "),
        TraceChunk(timestamp=_now(), text='via `"'),
        TraceChunk(timestamp=_now(), text="<tool_call>"),  # special token
        TraceChunk(timestamp=_now(), text='"`'),
        TraceChunk(timestamp=_now(), text=" tag followed by `<function=...>`.\n"),
    ]
    reasoning, content = replay_reasoning_streaming(chunks)
    check(
        "reasoning/fake_tool_call_with_ellipsis/stays_as_reasoning",
        content is None,
        f"Reasoning terminated at fake '<tool_call>'; content={content!r}",
    )


def test_reasoning_trace_session(session_chunks: list[TraceChunk]) -> None:
    """
    Streaming replay of the trace session through the reasoning parser.

    Verifies:
    - Reasoning ends at </think>
    - Everything after </think> is classified as content (not reasoning)
    """
    reasoning, content = replay_reasoning_streaming(session_chunks)
    full_text = "".join(c.text for c in session_chunks)

    if "</think>" in full_text:
        expected_reasoning = full_text.split("</think>", 1)[0]
        check(
            "reasoning/trace_session/reasoning_ends_at_think",
            reasoning is not None,
            "Reasoning should not be None for a session that has </think>",
        )
        check(
            "reasoning/trace_session/content_starts_after_think",
            content is not None,
            "Content should not be None after </think> was seen",
        )
        # Reasoning must not contain text that appears after </think>
        if reasoning and content:
            check(
                "reasoning/trace_session/no_post_think_text_in_reasoning",
                "comprehensive picture" not in reasoning,
                "Post-</think> text leaked into reasoning",
            )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _text_to_chunks(text: str, chunk_size: int = 1) -> list[TraceChunk]:
    """
    Split *text* into TraceChunks of at most *chunk_size* characters each,
    keeping every special token (</think>, <tool_call>, <function, …) as an
    indivisible chunk so their IDs always appear atomically in delta_token_ids.
    """
    ts = _now()
    result: list[TraceChunk] = []
    for part in _SPLIT_RE.split(text):
        if part in _SPECIAL_TOKENS:
            result.append(TraceChunk(timestamp=ts, text=part))
        else:
            for i in range(0, len(part), chunk_size):
                sub = part[i : i + chunk_size]
                if sub:
                    result.append(TraceChunk(timestamp=ts, text=sub))
    return result


def _find_session_containing(
    sessions: list[list[TraceChunk]],
    substring: str,
) -> Optional[list[TraceChunk]]:
    """Return the first session whose concatenated text contains *substring*."""
    for session in sessions:
        text = "".join(c.text for c in session)
        if substring in text:
            return session
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"Usage: {argv[0]} <trace_file>", file=sys.stderr)
        return 2

    trace_path = Path(argv[1])
    if not trace_path.exists():
        print(f"Trace file not found: {trace_path}", file=sys.stderr)
        return 2

    print(f"Parsing trace file: {trace_path}")
    chunks = parse_trace(trace_path)
    print(f"  {len(chunks)} chunks parsed")

    sessions = split_sessions(chunks, gap=SESSION_GAP_SECONDS)
    print(f"  {len(sessions)} sessions identified (gap={SESSION_GAP_SECONDS}s)")

    # --- Find the failing session (around 11:39:14) ---
    failing_session = _find_session_containing(
        sessions,
        # Unique string from the failing session text
        "TOOL_CALL_PENDING",
    )
    if failing_session:
        session_text = "".join(c.text for c in failing_session)
        print(
            f"  Failing session found: {len(failing_session)} chunks, "
            f"{len(session_text)} chars"
        )
        ts_start = failing_session[0].timestamp.strftime("%H:%M:%S")
        ts_end = failing_session[-1].timestamp.strftime("%H:%M:%S")
        print(f"  Session time range: {ts_start} – {ts_end}")
    else:
        print("  WARNING: Could not find expected failing session in trace")

    print()
    print("=" * 60)
    print("XML TOOL-PARSER TESTS")
    print("=" * 60)
    test_xml_tool_parser_basic_tool_call()
    test_xml_tool_parser_fake_tool_call_no_function()
    test_xml_tool_parser_function_placeholder_false_positive()
    test_xml_tool_parser_streaming_chunked()
    if failing_session:
        test_xml_tool_parser_trace_session(failing_session)
    else:
        print("  SKIP  xml/trace_session (failing session not found)")

    print()
    print("=" * 60)
    print("REASONING-PARSER TESTS (streaming)")
    print("=" * 60)
    test_reasoning_think_end()
    test_reasoning_tool_call_genuine_end()
    test_reasoning_fake_tool_call_in_reasoning()
    test_reasoning_fake_tool_call_followed_by_non_function()
    if failing_session:
        test_reasoning_trace_session(failing_session)
    else:
        print("  SKIP  reasoning/trace_session (failing session not found)")

    print()
    print("=" * 60)
    passed = sum(1 for r in _results if r.passed)
    failed = sum(1 for r in _results if not r.passed)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(_results)} tests")
    print("=" * 60)

    if failed:
        print()
        print("FAILURES:")
        for r in _results:
            if not r.passed:
                print(f"  {r.name}: {r.message}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
