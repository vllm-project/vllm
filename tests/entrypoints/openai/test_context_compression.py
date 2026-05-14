# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for ACE (Attention-Weighted Context Eviction) context compression."""

import pytest

from vllm.entrypoints.context_compression import (
    _score_line,
    ace_compress,
    apply_ace_eviction,
)


# ---------------------------------------------------------------------------
# _score_line tests
# ---------------------------------------------------------------------------


def test_score_blank_line():
    assert _score_line("") == 0.0
    assert _score_line("   ") == 0.0


def test_score_error_line():
    assert _score_line("Error: connection refused") >= 0.95
    assert _score_line("Traceback (most recent call last):") >= 0.95
    assert _score_line("exit code 1") >= 0.95


def test_score_tool_call_json():
    line = '{"name": "bash", "arguments": {"cmd": "ls"}}'
    assert _score_line(line) == 1.0


def test_score_numeric_data():
    assert _score_line("Processed 12345 rows in 3.2s") >= 0.7
    assert _score_line("Available: 2048 bytes remaining") >= 0.7


def test_score_boilerplate():
    assert _score_line("done") < 0.5
    assert _score_line("ok") < 0.5


def test_score_meta_commentary():
    assert _score_line("Here are the results") <= 0.15
    assert _score_line("I executed the command") <= 0.15


# ---------------------------------------------------------------------------
# ace_compress tests
# ---------------------------------------------------------------------------


def _make_long_content(n_lines: int = 20) -> str:
    """Generate a synthetic multi-line tool output for testing."""
    lines = ["Tool output start"]
    for i in range(n_lines - 2):
        lines.append(f"Regular output line {i}: some verbose data here")
    lines.append("Tool output end")
    return "\n".join(lines)


def test_ace_compress_basic():
    content = _make_long_content(20)
    compressed = ace_compress(content, target_ratio=0.4)
    original_lines = content.split("\n")
    compressed_lines = compressed.split("\n")
    # Should be shorter than original (some omitted lines replaced by markers)
    assert len(compressed_lines) < len(original_lines)
    # Omission marker must be present
    assert any("omitted by ACE" in line for line in compressed_lines)


def test_ace_compress_first_last_always_kept():
    content = _make_long_content(20)
    lines = content.split("\n")
    compressed = ace_compress(content, target_ratio=0.3)
    compressed_lines = compressed.split("\n")
    # First line must appear as first element (no preceding omission marker)
    assert compressed_lines[0] == lines[0]
    # Last line must be the final line
    assert compressed_lines[-1] == lines[-1]


def test_ace_compress_ratio_1_no_compression():
    content = _make_long_content(10)
    compressed = ace_compress(content, target_ratio=1.0)
    # With ratio=1.0 all lines are kept; no omission markers
    assert "omitted by ACE" not in compressed


def test_ace_compress_short_content_unchanged():
    content = "line one\nline two\nline three"
    assert ace_compress(content, target_ratio=0.4) == content


def test_ace_compress_error_line_preserved():
    """Error lines should score high and survive aggressive compression."""
    lines = ["start"] + ["verbose filler line"] * 20 + ["Error: fatal crash"] + ["end"]
    content = "\n".join(lines)
    compressed = ace_compress(content, target_ratio=0.2)
    assert "Error: fatal crash" in compressed


# ---------------------------------------------------------------------------
# apply_ace_eviction tests
# ---------------------------------------------------------------------------


def _make_messages(tool_content_size: int = 500, n_tools: int = 4) -> list[dict]:
    messages = [{"role": "user", "content": "Do some work"}]
    for i in range(n_tools):
        messages.append(
            {"role": "assistant", "content": f"Calling tool {i}"}
        )
        messages.append(
            {
                "role": "tool",
                "content": f"Tool {i} result:\n" + ("data " * (tool_content_size // 5)),
            }
        )
    messages.append({"role": "user", "content": "What is the answer?"})
    return messages


def test_apply_ace_no_compression_when_under_budget():
    messages = _make_messages(tool_content_size=100, n_tools=2)
    original = [dict(m) for m in messages]
    saved = apply_ace_eviction(messages, budget_chars=1_000_000)
    assert saved == 0
    assert messages == original


def test_apply_ace_compresses_when_over_budget():
    messages = _make_messages(tool_content_size=2000, n_tools=4)
    total_before = sum(
        len(m["content"]) for m in messages if isinstance(m.get("content"), str)
    )
    budget = total_before // 2  # Force compression
    saved = apply_ace_eviction(messages, budget_chars=budget)
    assert saved > 0
    total_after = sum(
        len(m["content"]) for m in messages if isinstance(m.get("content"), str)
    )
    assert total_after < total_before


def test_apply_ace_keep_recent_messages_unmodified():
    """The most-recent tool messages should not be compressed."""
    messages = _make_messages(tool_content_size=2000, n_tools=4)
    # Get the content of the last two tool messages before compression
    tool_msgs = [
        (i, m)
        for i, m in enumerate(messages)
        if m.get("role") == "tool"
    ]
    last_two_original = [(i, m["content"]) for i, m in tool_msgs[-2:]]

    budget = 1  # Force maximum compression
    apply_ace_eviction(messages, budget_chars=budget, keep_recent=2)

    for idx, original_content in last_two_original:
        assert messages[idx]["content"] == original_content, (
            f"Message at index {idx} should not have been compressed"
        )


def test_apply_ace_returns_chars_removed():
    messages = _make_messages(tool_content_size=2000, n_tools=4)
    total_before = sum(
        len(m["content"]) for m in messages if isinstance(m.get("content"), str)
    )
    saved = apply_ace_eviction(messages, budget_chars=total_before // 3)
    total_after = sum(
        len(m["content"]) for m in messages if isinstance(m.get("content"), str)
    )
    assert saved == total_before - total_after


def test_apply_ace_skips_short_messages():
    """Messages under min_chars should be left alone even if over budget."""
    short_tool_content = "short"
    messages = [
        {"role": "user", "content": "x"},
        {"role": "tool", "content": short_tool_content},
    ]
    saved = apply_ace_eviction(
        messages, budget_chars=1, min_chars=len(short_tool_content) + 1
    )
    assert saved == 0
    assert messages[1]["content"] == short_tool_content


def test_apply_ace_idempotent_on_already_compressed():
    """Running ACE twice should not double-compress (omission markers skipped)."""
    messages = _make_messages(tool_content_size=2000, n_tools=3)
    budget = sum(
        len(m["content"]) for m in messages if isinstance(m.get("content"), str)
    ) // 2
    apply_ace_eviction(messages, budget_chars=budget)
    snapshot = [dict(m) for m in messages]
    apply_ace_eviction(messages, budget_chars=budget)
    assert messages == snapshot
