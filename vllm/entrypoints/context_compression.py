# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ACE (Attention-Weighted Context Eviction) — content-aware context compression.

## Scoring modes

### Mode 1 — Heuristic
Static content rules (error=high, blank=zero). Fast, no model required.

### Mode 2 — BM25 Query-Relevance
Lines scored by BM25 against the current conversation context.
Approximates attention without running a forward pass.

### Mode 3 — Accumulated Attention (the real thing)
Uses attention weights accumulated from the model's previous forward passes.
Each turn, the attention layers report how much the new tokens attended to
each past token. We sum these contributions per position, map to line-level
scores, and use them for eviction.

This is what "Attention-Weighted" means: the model's own attention signal
decides what to keep — not heuristics, not text similarity.

## Architecture

    ┌─────────────────────────────────────────────────────────┐
    │  Turn N generation                                        │
    │   Attention layers ──attn_weights──► AttentionTracker    │
    │                                           │               │
    │                              accumulates per-token scores │
    └─────────────────────────────────────────────────────────┘
                                               │
                                               ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Turn N+1 request arrives, context > budget              │
    │   AttentionTracker ──scores──► ace_compress()            │
    │                                    │                      │
    │                  keeps high-attention lines, evicts rest  │
    └─────────────────────────────────────────────────────────┘

## Related work

This is the message-level analogue of H2O (Zhang et al., 2023) and
Scissorhands (Liu et al., 2023), which apply accumulated attention scores
to KV cache token eviction. ACE applies the same principle at the coarser
granularity of text lines within tool-result messages.

Key differences from H2O/Scissorhands:
- Operates on text (before tokenization) not on KV cache entries
- Removes content from the context window, not from GPU memory
- Designed for multi-turn agentic loops, not single-sequence KV compression
"""
from __future__ import annotations

import math
import re
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Mode 3: Accumulated attention importance tracker
# ---------------------------------------------------------------------------

class AttentionImportanceTracker:
    """
    Accumulates per-token attention importance across generation turns.

    The attention pattern for turn N tells us which tokens in the existing
    context the model found useful when generating its response. By summing
    these patterns across all layers, heads, and new tokens, we get a score
    for each past token position reflecting how much the model has relied on it.

    This is the ground-truth signal for context eviction: positions with
    consistently low cumulative attention are safe to remove.

    Thread-safe for concurrent request handling.
    """

    def __init__(self, max_seq_len: int = 32768) -> None:
        self._max_seq_len = max_seq_len
        self._scores: list[float] = [0.0] * max_seq_len
        self._n_updates = 0
        self._lock = threading.Lock()

    def accumulate(
        self,
        attn_weights: "np.ndarray",
        new_token_start: int,
    ) -> None:
        """
        Add attention from new tokens to accumulated per-position scores.

        Args:
            attn_weights:    Attention weight tensor, shape
                             [n_layers, n_heads, n_new_tokens, seq_len].
                             Values should be in [0, 1] and sum to 1 over seq_len.
            new_token_start: Index of the first new token in the sequence.
                             Only positions before this index are candidates
                             for eviction.
        """
        import numpy as np  # optional dep — only needed when Mode 3 is active

        weights = np.asarray(attn_weights, dtype=np.float32)
        # Mean over layers, heads, new_tokens → shape [seq_len]
        contribution = weights.mean(axis=(0, 1, 2))
        n = min(len(contribution), new_token_start, self._max_seq_len)
        with self._lock:
            for i in range(n):
                self._scores[i] += float(contribution[i])
            self._n_updates += 1

    def score_lines(
        self, token_spans: list[tuple[int, int]]
    ) -> list[float]:
        """
        Map accumulated token-level scores to line-level importance scores.

        Args:
            token_spans: (start_token, end_token) for each line, as absolute
                         positions in the full sequence.

        Returns:
            Normalized importance score in [0, 1] for each line.
            Returns 0.5 for all lines if no attention data has been accumulated.
        """
        with self._lock:
            scores = self._scores[:]
            n_updates = self._n_updates

        if n_updates == 0:
            return [0.5] * len(token_spans)

        raw: list[float] = []
        for start, end in token_spans:
            if start >= end or start >= len(scores):
                raw.append(0.0)
            else:
                end = min(end, len(scores))
                raw.append(sum(scores[start:end]) / max(end - start, 1))

        max_score = max(raw) if raw else 0.0
        if max_score <= 0:
            return [0.5] * len(token_spans)
        return [s / max_score for s in raw]

    def reset(self) -> None:
        with self._lock:
            self._scores = [0.0] * self._max_seq_len
            self._n_updates = 0

    @property
    def has_data(self) -> bool:
        return self._n_updates > 0


# Global registry: request_id → tracker
# Populated by serve/render/serving.py when context_compression="ace" and
# updated by the attention capture hook after each forward pass.
_tracker_registry: dict[str, AttentionImportanceTracker] = {}
_registry_lock = threading.Lock()


def register_tracker(
    request_id: str, max_seq_len: int = 32768
) -> AttentionImportanceTracker:
    """Create and register an attention tracker for a request."""
    tracker = AttentionImportanceTracker(max_seq_len=max_seq_len)
    with _registry_lock:
        _tracker_registry[request_id] = tracker
    return tracker


def get_tracker(request_id: str) -> AttentionImportanceTracker | None:
    """Retrieve the tracker for a request (returns None if not registered)."""
    with _registry_lock:
        return _tracker_registry.get(request_id)


def release_tracker(request_id: str) -> None:
    """Remove a tracker after the request is complete."""
    with _registry_lock:
        _tracker_registry.pop(request_id, None)


# ---------------------------------------------------------------------------
# Token-to-line span mapping
# ---------------------------------------------------------------------------

def compute_line_token_spans(
    text: str,
    tokenizer: Any,
    message_start_token: int,
) -> list[tuple[int, int]]:
    """
    Map each line in text to its token span in the full sequence.

    Args:
        text:                The tool-result text to map.
        tokenizer:           vLLM tokenizer (must support `encode` with offsets).
        message_start_token: Absolute token position of the first token of text.

    Returns:
        List of (start_token, end_token) for each line in text.split('\\n').
    """
    lines = text.split("\n")

    # Encode with char offsets if tokenizer supports it
    try:
        encoding = tokenizer(text, return_offsets_mapping=True)
        char_to_token = encoding["offset_mapping"]  # list of (char_start, char_end)
    except Exception:
        # Fallback: estimate uniform token distribution
        total_tokens = max(len(tokenizer.encode(text)), 1)
        chars_per_tok = max(len(text) / total_tokens, 1)
        spans = []
        pos = 0
        running_tok = message_start_token
        for line in lines:
            n_toks = max(1, round(len(line) / chars_per_tok))
            spans.append((running_tok, running_tok + n_toks))
            running_tok += n_toks + 1  # +1 for newline token
            pos += len(line) + 1
        return spans

    # Build char-position → token-index map
    char_to_tok_idx: dict[int, int] = {}
    for tok_idx, (cs, ce) in enumerate(char_to_token):
        for c in range(cs, ce):
            char_to_tok_idx[c] = tok_idx

    spans = []
    pos = 0
    for line in lines:
        line_start_char = pos
        line_end_char = pos + len(line)

        # Find first and last token for this line's char range
        tok_start = None
        tok_end = None
        for c in range(line_start_char, min(line_end_char, len(text))):
            t = char_to_tok_idx.get(c)
            if t is not None:
                if tok_start is None:
                    tok_start = t
                tok_end = t + 1

        if tok_start is None:
            tok_start = tok_end = char_to_tok_idx.get(line_start_char, 0)

        spans.append((
            message_start_token + (tok_start or 0),
            message_start_token + (tok_end or 1),
        ))
        pos += len(line) + 1  # +1 for \n

    return spans


# ---------------------------------------------------------------------------
# Mode 1: Heuristic line scorer
# ---------------------------------------------------------------------------

def _heuristic_score(line: str) -> float:
    """Score a line by content heuristics. Returns float in [0, 1]."""
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
    if any(k in sl for k in
           ["here are the results", "i executed", "tool results:", "output:"]):
        return 0.1
    return 0.5


# ---------------------------------------------------------------------------
# Mode 2: BM25 query-relevance scorer
# ---------------------------------------------------------------------------

class _BM25Scorer:
    """Score lines by BM25 relevance to a query. k1=1.5, b=0.75."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    @staticmethod
    def _tok(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def score_lines(self, query: str, lines: list[str]) -> list[float]:
        query_terms = set(self._tok(query))
        if not query_terms or not lines:
            return [0.5] * len(lines)

        tokenized = [self._tok(l) for l in lines]
        n = len(lines)
        df: dict[str, int] = {}
        for terms in tokenized:
            for t in set(terms):
                df[t] = df.get(t, 0) + 1

        idf = {
            t: math.log((n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0)
            for t in query_terms
        }
        avg_dl = sum(len(t) for t in tokenized) / max(n, 1)

        raw: list[float] = []
        for terms in tokenized:
            dl = len(terms)
            tf: dict[str, int] = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
            score = sum(
                idf[t] * tf[t] * (self.k1 + 1) /
                (tf[t] + self.k1 * (1 - self.b + self.b * dl / max(avg_dl, 1)))
                for t in query_terms if t in tf
            )
            raw.append(score)

        max_s = max(raw) if raw else 0.0
        return [s / max_s for s in raw] if max_s > 0 else [0.5] * len(lines)


_bm25 = _BM25Scorer()


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def ace_compress(
    content: str,
    target_ratio: float = 0.4,
    query: str | None = None,
    attention_scores: list[float] | None = None,
) -> str:
    """
    Compress a multi-line string to approximately target_ratio of its line count.

    Scoring priority:
      1. attention_scores (Mode 3) — model-derived, most accurate
      2. query + BM25   (Mode 2) — query-relative relevance
      3. heuristics     (Mode 1) — fallback, always available

    Heuristic scores provide a 30% floor in Modes 2 and 3 so structural
    anchors (errors, file paths) are never fully ignored.

    Args:
        content:          Text to compress.
        target_ratio:     Fraction of lines to keep (default 0.4).
        query:            Current conversation context for BM25 scoring.
        attention_scores: Pre-computed per-line attention scores from
                          AttentionImportanceTracker.score_lines().
    """
    lines = content.split("\n")
    if len(lines) <= 3:
        return content

    if attention_scores is not None and len(attention_scores) == len(lines):
        # Mode 3: attention-weighted
        scored = []
        for i, (line, attn) in enumerate(zip(lines, attention_scores)):
            h = _heuristic_score(line)
            s = max(attn, h * 0.3)
            if i == 0 or i == len(lines) - 1:
                s = max(s, 0.7)
            scored.append((i, s))

    elif query:
        # Mode 2: BM25 query-relevance
        bm25_scores = _bm25.score_lines(query, lines)
        scored = []
        for i, (line, bm) in enumerate(zip(lines, bm25_scores)):
            h = _heuristic_score(line)
            s = max(bm, h * 0.3)
            if i == 0 or i == len(lines) - 1:
                s = max(s, 0.7)
            scored.append((i, s))

    else:
        # Mode 1: heuristics only
        scored = []
        for i, line in enumerate(lines):
            s = _heuristic_score(line)
            if i == 0 or i == len(lines) - 1:
                s = max(s, 0.7)
            scored.append((i, s))

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


# ---------------------------------------------------------------------------
# Eviction loop
# ---------------------------------------------------------------------------

def _extract_query(messages: list[dict[str, Any]], n: int = 3) -> str:
    recent = [
        m["content"] for m in messages[-(n * 2):]
        if m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
        and len(m["content"]) > 10
    ]
    return " ".join(recent[-n:])


def apply_ace_eviction(
    messages: list[dict[str, Any]],
    budget_chars: int,
    keep_recent: int = 2,
    target_ratio: float = 0.4,
    min_chars: int = 200,
    use_query_relevance: bool = True,
    tracker: AttentionImportanceTracker | None = None,
    tokenizer: Any | None = None,
    token_offsets: list[int] | None = None,
    recency_blend: float = 0.3,
) -> int:
    """
    recency_blend: fraction of the per-line score contributed by message recency.
      0.0 = pure content scoring (BM25 / heuristics).
      0.3 = 30% recency + 70% content (default). Oldest compressible messages
            are evicted more aggressively; newest compressible messages keep
            more of their lines. This prevents ACE from destroying context the
            agent is actively using in recent turns.
    """
    """
    Compress old tool-result messages until total size <= budget_chars.

    Scoring mode is selected automatically based on available inputs:
      - tracker + tokenizer → Mode 3 (accumulated attention weights)
      - use_query_relevance  → Mode 2 (BM25 query-relevance)
      - otherwise            → Mode 1 (heuristics)

    Args:
        messages:            Conversation messages list (OpenAI format).
        budget_chars:        Target max total chars.
        keep_recent:         Most-recent tool messages to leave uncompressed.
        target_ratio:        Fraction of lines to keep (default 0.4).
        min_chars:           Don't compress messages shorter than this.
        use_query_relevance: Enable BM25 scoring when no tracker is available.
        tracker:             AttentionImportanceTracker for this request.
                             When provided, uses accumulated attention scores.
        tokenizer:           vLLM tokenizer. Required when tracker is provided.
        token_offsets:       Cumulative token count per message. Required when
                             tracker is provided. token_offsets[i] = absolute
                             token position of the first token of messages[i].

    Returns:
        Total chars removed.
    """

    def _total(msgs: list[dict]) -> int:
        return sum(
            len(m["content"]) for m in msgs if isinstance(m.get("content"), str)
        )

    if _total(messages) <= budget_chars:
        return 0

    use_attention = (
        tracker is not None
        and tracker.has_data
        and tokenizer is not None
        and token_offsets is not None
    )
    query = _extract_query(messages) if (use_query_relevance and not use_attention) else None

    tool_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") in ("tool", "user")
        and isinstance(m.get("content"), str)
        and len(m["content"]) >= min_chars
    ]
    compressable = tool_indices[: max(0, len(tool_indices) - keep_recent)]
    n_comp = max(len(compressable) - 1, 1)
    saved = 0

    for rank, idx in enumerate(compressable):
        if _total(messages) <= budget_chars:
            break
        original = messages[idx]["content"]
        if "omitted by ACE" in original:
            continue

        # Compute per-line content scores (attention / BM25 / heuristic)
        attn_scores: list[float] | None = None
        if use_attention and idx < len(token_offsets):
            try:
                spans = compute_line_token_spans(
                    original, tokenizer, token_offsets[idx]
                )
                attn_scores = tracker.score_lines(spans)
            except Exception:
                pass

        # Recency blend: rank 0 = oldest (recency=0.0), rank n-1 = newest (recency=1.0)
        # Older messages get a lower keep-ratio → compressed more aggressively.
        if recency_blend > 0:
            msg_recency = rank / n_comp          # 0.0 (oldest) → 1.0 (newest)
            # Blend content scores with recency floor
            content_scores = attn_scores  # may be None → ace_compress handles it
            n_lines = len(original.split("\n"))
            if content_scores is None:
                # Generate content scores first so we can blend them
                lines = original.split("\n")
                if query:
                    content_scores = _bm25.score_lines(query, lines)
                else:
                    content_scores = [_heuristic_score(l) for l in lines]
                    mx = max(content_scores) or 1.0
                    content_scores = [s / mx for s in content_scores]
            # Blend: older messages → recency is low → content dominates → more eviction
            blended = [
                (1 - recency_blend) * cs + recency_blend * msg_recency
                for cs in content_scores
            ]
            # More aggressive ratio for older messages
            effective_ratio = target_ratio * (0.5 + 0.5 * msg_recency)
            compressed = ace_compress(
                original, effective_ratio, attention_scores=blended
            )
        else:
            effective_ratio = target_ratio
            compressed = ace_compress(
                original, effective_ratio, query=query, attention_scores=attn_scores
            )

        delta = len(original) - len(compressed)
        if delta <= 0:
            continue
        saved += delta
        messages[idx] = {**messages[idx], "content": compressed}

    return saved
