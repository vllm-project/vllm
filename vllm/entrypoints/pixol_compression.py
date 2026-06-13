# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Pixol-ACE: context compression with baked-in multi-dimensional line scores.

Inspired by ZBrush's pixol concept: instead of scoring lines at eviction time
(like BM25 does), each line is scored on multiple dimensions the moment it
enters the context. Eviction reads pre-computed scores — no re-scoring needed.

## Dimensions

  structural    Static content rules (error=high, blank=zero). Computed once.
  type          Line classification: error | code_structure | diff | output
  references    How many times subsequent messages reference terms from this line.
  age           Turns since the line was written. High age → lower weight.

## Why this is faster than BM25

BM25 must tokenize and score every line against every new query at eviction time.
Pixol scores are computed once at ingestion and updated O(1) per turn (reference
counting + age increment). Eviction is a simple sort on cached float scores.

## Usage

    store = PixolStore()

    # When a tool message arrives:
    store.ingest(msg_idx, tool_content)

    # Each new turn — age existing lines and scan for references:
    store.tick(new_assistant_content)

    # At eviction time — get pre-computed scores for a message's lines:
    scores = store.get_scores(msg_idx)
    # Pass to ace_compress(..., attention_scores=scores)
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Pixol line
# ---------------------------------------------------------------------------

_ERROR_PAT  = re.compile(r'error|traceback|exception|failed|not found|exit code', re.I)
_PATH_PAT   = re.compile(r'[/\\][\w/\\.\-]+\.\w+')
_NUM_PAT    = re.compile(r'\d{3,}|bytes|rows|%')
_STRUCT_PAT = re.compile(r'^\s*(def |class |import |from |@|\s*#)')
_DIFF_PAT   = re.compile(r'^[+\-@]{1,3}')
_BLANK_PAT  = re.compile(r'^\s*$')


def _classify(line: str) -> str:
    if _BLANK_PAT.match(line):    return 'blank'
    if _ERROR_PAT.search(line):   return 'error'
    if _DIFF_PAT.match(line):     return 'diff'
    if _STRUCT_PAT.match(line):   return 'code_structure'
    if _PATH_PAT.search(line):    return 'path'
    if _NUM_PAT.search(line):     return 'numeric'
    return 'output'


_TYPE_BASE = {
    'error':          1.0,
    'diff':           0.9,
    'path':           0.85,
    'code_structure': 0.75,
    'numeric':        0.65,
    'output':         0.45,
    'blank':          0.0,
}


def _structural_score(line: str, line_type: str) -> float:
    base = _TYPE_BASE[line_type]
    if '"name"' in line and ('"arguments"' in line or '"parameters"' in line):
        base = 1.0
    if line_type == 'output' and len(line) > 200:
        base = 0.2
    if any(k in line.lower() for k in ('done', 'ok', 'installed', 'success', 'complete')):
        base = min(base, 0.3)
    return base


@dataclass
class PixolLine:
    """A line of tool output with baked-in multi-dimensional importance scores."""
    text:         str
    line_type:    str
    structural:   float
    references:   int   = 0    # incremented when subsequent turns mention this line's terms
    age:          int   = 0    # turns since this line was ingested
    _terms:       frozenset = field(default_factory=frozenset, repr=False)

    @classmethod
    def from_text(cls, text: str) -> "PixolLine":
        ltype = _classify(text)
        struct = _structural_score(text, ltype)
        terms  = frozenset(re.findall(r'\b\w{4,}\b', text.lower()))
        return cls(text=text, line_type=ltype, structural=struct, _terms=terms)

    @property
    def score(self) -> float:
        """Composite importance score in [0, 1]."""
        if self.line_type == 'blank':
            return 0.0
        ref_boost = 1.0 + 0.25 * min(self.references, 4)
        # Exponential decay: 0.92^age — smooth and never fully zeroes out.
        # At age 8: 0.513; age 16: 0.263; age 32: 0.069.
        # Much better than linear for 50+ turn organic sessions.
        age_decay = max(0.1, 0.92 ** self.age)
        return min(1.0, self.structural * ref_boost * age_decay)


# ---------------------------------------------------------------------------
# PixolStore — the sidecar registry
# ---------------------------------------------------------------------------

class PixolStore:
    """
    Maintains per-message lists of PixolLines.

    Thread-safe. Attach one store per conversation; pass it into
    apply_pixol_eviction() instead of apply_ace_eviction().
    """

    def __init__(self) -> None:
        self._msgs:  dict[int, list[PixolLine]] = {}
        self._lock   = threading.Lock()

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest(self, msg_idx: int, content: str) -> None:
        """Parse content into PixolLines and store under msg_idx."""
        lines = [PixolLine.from_text(l) for l in content.split('\n')]
        with self._lock:
            self._msgs[msg_idx] = lines

    # ── Per-turn update ──────────────────────────────────────────────────────

    def tick(self, new_content: str) -> None:
        """
        Called once per turn with the latest assistant/user message.
        Ages all lines by 1 and increments reference count for lines
        whose key terms appear in new_content.
        """
        new_terms = frozenset(re.findall(r'\b\w{4,}\b', new_content.lower()))
        with self._lock:
            for lines in self._msgs.values():
                for pline in lines:
                    pline.age += 1
                    if pline._terms & new_terms:
                        pline.references += 1

    # ── Score retrieval ──────────────────────────────────────────────────────

    def get_scores(self, msg_idx: int) -> Optional[list[float]]:
        """Return normalized per-line scores for msg_idx, or None if not found."""
        with self._lock:
            lines = self._msgs.get(msg_idx)
        if not lines:
            return None
        raw = [pl.score for pl in lines]
        mx  = max(raw) if raw else 0.0
        return [s / mx for s in raw] if mx > 0 else [0.5] * len(raw)

    def release(self, msg_idx: int) -> None:
        with self._lock:
            self._msgs.pop(msg_idx, None)


# ---------------------------------------------------------------------------
# Drop-in eviction function
# ---------------------------------------------------------------------------

def apply_pixol_eviction(
    messages:    list[dict],
    store:       PixolStore,
    budget_chars: int,
    keep_recent:  int = 2,
    target_ratio: float = 0.4,
    min_chars:    int  = 200,
) -> int:
    """
    Like apply_ace_eviction() but uses pre-computed PixolLine scores.

    Args:
        messages:     Conversation messages (OpenAI format). Modified in place.
        store:        PixolStore attached to this conversation.
        budget_chars: Target max total chars.
        keep_recent:  Most-recent tool messages to leave uncompressed.
        target_ratio: Fraction of lines to keep (used when store has no data).
        min_chars:    Skip messages shorter than this.

    Returns:
        Total chars removed.
    """
    # Lazy import to avoid circular deps
    from vllm.entrypoints.context_compression import ace_compress

    def _total(msgs):
        return sum(len(m["content"]) for m in msgs if isinstance(m.get("content"), str))

    if _total(messages) <= budget_chars:
        return 0

    # Only compress tool-result messages. User/system turns may carry the task
    # description, security constraints, or behavioral guardrails that must never
    # be silently evicted (see context_compression.apply_ace_eviction).
    tool_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool"
        and isinstance(m.get("content"), str)
        and len(m["content"]) >= min_chars
    ]
    compressible = tool_indices[:max(0, len(tool_indices) - keep_recent)]
    saved = 0

    for idx in compressible:
        if _total(messages) <= budget_chars:
            break
        original = messages[idx]["content"]
        if "omitted by ACE" in original:
            continue

        pixol_scores = store.get_scores(idx)
        compressed = ace_compress(
            original,
            target_ratio=target_ratio,
            attention_scores=pixol_scores,  # reuse Mode 3 path in ace_compress
        )
        delta = len(original) - len(compressed)
        if delta <= 0:
            continue
        saved += delta
        messages[idx] = {**messages[idx], "content": compressed}
        store.release(idx)

    return saved
