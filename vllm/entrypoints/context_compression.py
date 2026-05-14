# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ACE (Attention-Weighted Context Eviction) — content-aware context compression.

Compresses tool result messages in a conversation by scoring lines for importance
and keeping only the highest-scoring fraction. Applied before chat template rendering
so that the model always sees the most relevant parts of long tool outputs.
"""
from __future__ import annotations

import re
from typing import Any


def _score_line(line: str) -> float:
    """Score a single line for importance. Returns float in [0, 1]."""
    if not line.strip():
        return 0.0
    sl = line.lower()
    # Tool call JSON
    if '"name"' in line and ('"arguments"' in line or '"parameters"' in line):
        return 1.0
    # Errors
    if any(
        p in sl
        for p in [
            "error",
            "failed",
            "not found",
            "traceback",
            "exception",
            "exit code",
        ]
    ):
        return 0.95
    # File paths
    if any(
        p in line
        for p in ["/results/", "/workspace/", "/tmp/", "/home/", "/var/"]
    ):
        return 0.9
    # Task keywords
    if any(k in sl for k in ["task:", "step ", "verify", "inspect", "begin"]):
        return 0.85
    # Numeric data
    if re.search(r"\d+\.?\d*%|\d{4,}|bytes|rows|columns|tokens", sl):
        return 0.7
    # Shell prompts / commands
    if line.startswith(("$ ", ">>> ", "root@", "agent@")) or any(
        k in sl
        for k in ["wget", "curl", "pip", "python3", "apt-get", "docker", "git "]
    ):
        return 0.6
    # Success boilerplate
    if sl.strip() in {"done", "ok", "installed", "saved", "complete", "success"}:
        return 0.3
    # Long verbose dump
    if len(line) > 200:
        return 0.2
    # Meta-commentary
    if any(
        k in sl
        for k in [
            "here are the results",
            "i executed",
            "tool results:",
            "output:",
        ]
    ):
        return 0.1
    return 0.5


def ace_compress(content: str, target_ratio: float = 0.4) -> str:
    """
    Compress a multi-line string to approximately target_ratio of its line count.

    Lines are scored by importance; the top-scoring fraction is kept.
    Consecutive dropped lines are replaced with a single omission marker.
    The first and last lines are always kept.
    """
    lines = content.split("\n")
    if len(lines) <= 3:
        return content

    scored = []
    for i, line in enumerate(lines):
        score = _score_line(line)
        # Anchor first and last lines
        if i == 0 or i == len(lines) - 1:
            score = max(score, 0.7)
        scored.append((i, score))

    n_keep = max(2, int(len(lines) * target_ratio))
    keep = set(i for i, _ in sorted(scored, key=lambda x: -x[1])[:n_keep])
    keep.add(0)
    keep.add(len(lines) - 1)

    out: list[str] = []
    skipped = 0
    for i, line in enumerate(lines):
        if i in keep:
            if skipped:
                out.append(f"[...{skipped} lines omitted by ACE...]")
            skipped = 0
            out.append(line)
        else:
            skipped += 1
    if skipped:
        out.append(f"[...{skipped} lines omitted by ACE...]")
    return "\n".join(out)


def apply_ace_eviction(
    messages: list[dict[str, Any]],
    budget_chars: int,
    keep_recent: int = 2,
    target_ratio: float = 0.4,
    min_chars: int = 200,
) -> int:
    """
    Compress old tool-result messages until total message size <= budget_chars.

    Args:
        messages:     Conversation messages list (OpenAI format).
        budget_chars: Target max total chars.
        keep_recent:  Number of most-recent tool messages to leave uncompressed.
        target_ratio: Fraction of lines to keep (default 0.4 = 40%).
        min_chars:    Don't compress messages shorter than this.

    Returns:
        Total chars removed.
    """

    def _total(msgs: list[dict]) -> int:
        return sum(
            len(m["content"])
            for m in msgs
            if isinstance(m.get("content"), str)
        )

    if _total(messages) <= budget_chars:
        return 0

    # Collect indices of tool/user messages with compressible content
    tool_indices = [
        i
        for i, m in enumerate(messages)
        if m.get("role") in ("tool", "user")
        and isinstance(m.get("content"), str)
        and len(m["content"]) >= min_chars
    ]

    compressable = tool_indices[: max(0, len(tool_indices) - keep_recent)]
    saved = 0

    for idx in compressable:
        if _total(messages) <= budget_chars:
            break
        original = messages[idx]["content"]
        if "omitted by ACE" in original:
            continue
        compressed = ace_compress(original, target_ratio)
        delta = len(original) - len(compressed)
        if delta <= 0:
            continue
        saved += delta
        messages[idx] = {**messages[idx], "content": compressed}

    return saved
