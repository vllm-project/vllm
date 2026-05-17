# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ACE (Attention-Weighted Context Eviction) — content-aware context compression.

## Scoring modes

### Mode 1 — Heuristic (default, no query required)
Static rules based on line content: errors score high, blank lines score zero.
Fast, model-agnostic, works without knowledge of the current query.

### Mode 2 — Query-relevance (recommended for multi-turn agents)
Lines are scored by BM25 relevance to the current query context — an
approximation of the attention pattern the model would apply when reading
old tool results from its current position. Lines the model is likely to
attend to are kept; lines it would ignore are evicted.

This is the intended direction for ACE: replicating the importance signal
that transformer attention heads produce, without running a full forward pass.
A future Phase 3 will use accumulated attention weights from vLLM's inference
engine directly, replacing this BM25 proxy with exact attention scores.

## Usage

    from vllm.entrypoints.context_compression import apply_ace_eviction

    # Heuristic mode (backward-compatible)
    saved = apply_ace_eviction(messages, budget_chars=40_000)

    # Query-relevance mode — pass the current query text
    query = " ".join(m["content"] for m in messages[-3:] if m.get("role") != "tool")
    saved = apply_ace_eviction(messages, budget_chars=40_000, query=query)
"""
from __future__ import annotations

import math
import re
from typing import Any


# ---------------------------------------------------------------------------
# Mode 1: Heuristic line scorer
# ---------------------------------------------------------------------------

def _heuristic_score(line: str) -> float:
    """Score a single line for importance using content heuristics.

    Returns float in [0, 1]. Higher = more important to keep.

    Priority tiers:
      1.0  tool-call JSON (name + arguments)
      0.95 error / exception lines
      0.9  file / workspace paths
      0.85 task-framing keywords
      0.7  numeric data (%, large numbers, units)
      0.6  shell commands / prompts
      0.5  default
      0.3  success boilerplate
      0.2  very long lines (verbose dumps)
      0.1  meta-commentary
      0.0  blank lines
    """
    if not line.strip():
        return 0.0
    sl = line.lower()
    if '"name"' in line and ('"arguments"' in line or '"parameters"' in line):
        return 1.0
    if any(p in sl for p in
           ["error", "failed", "not found", "traceback", "exception", "exit code"]):
        return 0.95
    if any(p in line for p in ["/results/", "/workspace/", "/tmp/", "/home/", "/var/"]):
        return 0.9
    if any(k in sl for k in ["task:", "step ", "verify", "inspect", "begin"]):
        return 0.85
    if re.search(r"\d+\.?\d*%|\d{4,}|bytes|rows|columns|tokens", sl):
        return 0.7
    if line.startswith(("$ ", ">>> ", "root@", "agent@")) or any(
        k in sl for k in ["wget", "curl", "pip", "python3", "apt-get", "docker", "git "]
    ):
        return 0.6
    if sl.strip() in {"done", "ok", "installed", "saved", "complete", "success"}:
        return 0.3
    if len(line) > 200:
        return 0.2
    if any(k in sl for k in ["here are the results", "i executed", "tool results:", "output:"]):
        return 0.1
    return 0.5


# ---------------------------------------------------------------------------
# Mode 2: Query-relevance scorer (BM25)
# ---------------------------------------------------------------------------

class _BM25Scorer:
    """
    Score lines by BM25 relevance to a query string.

    This approximates the attention pattern the model would apply when reading
    old tool results from the position of the current query: lines containing
    terms the model is currently "thinking about" score high; unrelated lines
    score low.

    BM25 parameters follow Robertson et al. (k1=1.5, b=0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def score_lines(self, query: str, lines: list[str]) -> list[float]:
        """Return a relevance score in [0, 1] for each line."""
        query_terms = set(self._tokenize(query))
        if not query_terms or not lines:
            return [0.5] * len(lines)

        tokenized = [self._tokenize(line) for line in lines]
        n = len(lines)

        # Document frequency per term
        df: dict[str, int] = {}
        for terms in tokenized:
            for t in set(terms):
                df[t] = df.get(t, 0) + 1

        # IDF (Robertson-Sparck Jones smoothed)
        idf: dict[str, float] = {
            t: math.log((n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0)
            for t in query_terms
        }

        avg_dl = sum(len(t) for t in tokenized) / max(n, 1)

        raw: list[float] = []
        for terms in tokenized:
            dl = len(terms)
            tf_map: dict[str, int] = {}
            for t in terms:
                tf_map[t] = tf_map.get(t, 0) + 1

            score = 0.0
            for t in query_terms:
                if t not in tf_map:
                    continue
                tf = tf_map[t]
                norm_tf = tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(avg_dl, 1))
                )
                score += idf[t] * norm_tf
            raw.append(score)

        max_score = max(raw) if raw else 0.0
        if max_score <= 0:
            return [0.5] * len(lines)
        return [s / max_score for s in raw]


_bm25 = _BM25Scorer()


def _query_relevance_score(query: str, lines: list[str]) -> list[float]:
    """Return BM25 relevance scores for lines given a query."""
    return _bm25.score_lines(query, lines)


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def ace_compress(
    content: str,
    target_ratio: float = 0.4,
    query: str | None = None,
) -> str:
    """
    Compress a multi-line string to approximately target_ratio of its line count.

    Lines are scored for importance and the top-scoring fraction is kept.
    Consecutive dropped lines are replaced with a single omission marker.
    The first and last lines are always kept.

    Args:
        content:      Text to compress.
        target_ratio: Fraction of lines to keep (default 0.4 = 40%).
        query:        If provided, use BM25 query-relevance scoring instead of
                      heuristics. Pass the recent conversation context so that
                      lines relevant to the current task are preserved.
    """
    lines = content.split("\n")
    if len(lines) <= 3:
        return content

    if query is not None:
        # Mode 2: query-relevance scoring
        raw_scores = _query_relevance_score(query, lines)
        scores = []
        for i, s in enumerate(raw_scores):
            # Anchor first/last; blend with heuristic floor so structure is preserved
            h = _heuristic_score(lines[i])
            combined = max(s, h * 0.3)  # heuristic provides a floor, query drives ranking
            if i == 0 or i == len(lines) - 1:
                combined = max(combined, 0.7)
            scores.append((i, combined))
    else:
        # Mode 1: heuristic scoring
        scores = []
        for i, line in enumerate(lines):
            s = _heuristic_score(line)
            if i == 0 or i == len(lines) - 1:
                s = max(s, 0.7)
            scores.append((i, s))

    n_keep = max(2, int(len(lines) * target_ratio))
    keep = set(i for i, _ in sorted(scores, key=lambda x: -x[1])[:n_keep])
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


# ---------------------------------------------------------------------------
# Eviction loop
# ---------------------------------------------------------------------------

def _extract_query(messages: list[dict[str, Any]], n_recent: int = 3) -> str:
    """
    Extract a query string from the most recent non-tool messages.

    Used to build the BM25 query for relevance-based scoring.
    """
    recent = [
        m["content"]
        for m in messages[-n_recent * 2:]  # look back far enough
        if m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
        and len(m["content"]) > 10
    ]
    return " ".join(recent[-n_recent:])


def apply_ace_eviction(
    messages: list[dict[str, Any]],
    budget_chars: int,
    keep_recent: int = 2,
    target_ratio: float = 0.4,
    min_chars: int = 200,
    use_query_relevance: bool = True,
) -> int:
    """
    Compress old tool-result messages until total message size <= budget_chars.

    When use_query_relevance=True (default), lines are scored by BM25 relevance
    to the current conversation context — approximating which parts of old tool
    outputs the model would attend to from its current position.

    Args:
        messages:             Conversation messages list (OpenAI format).
        budget_chars:         Target max total chars.
        keep_recent:          Number of most-recent tool messages to leave uncompressed.
        target_ratio:         Fraction of lines to keep (default 0.4 = 40%).
        min_chars:            Don't compress messages shorter than this.
        use_query_relevance:  Use BM25 query-relevance scoring (True) or heuristics (False).

    Returns:
        Total chars removed.
    """

    def _total(msgs: list[dict]) -> int:
        return sum(
            len(m["content"]) for m in msgs if isinstance(m.get("content"), str)
        )

    if _total(messages) <= budget_chars:
        return 0

    query = _extract_query(messages) if use_query_relevance else None

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
        compressed = ace_compress(original, target_ratio, query=query)
        delta = len(original) - len(compressed)
        if delta <= 0:
            continue
        saved += delta
        messages[idx] = {**messages[idx], "content": compressed}

    return saved
