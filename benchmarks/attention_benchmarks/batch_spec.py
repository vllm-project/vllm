# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Simplified batch specification grammar for attention benchmarks.

Grammar (underscore-separated segments):
  Format: (<count>?) q<q_len>(k?) (s<seq_len>(k?))?

  - count: Number of identical requests (optional, default=1)
  - q_len: Query length (number of new tokens)
  - seq_len: Total sequence length (optional, defaults to q_len for prefill)
  - 'k' suffix: Multiplies value by 1024

Common patterns:
  - Prefill:  q_len == seq_len  (e.g., "q2k" → 2048 new tokens, 2048 seq)
  - Decode:   q_len == 1        (e.g., "q1s1k" → 1 token, 1024 seq length)
  - Extend:   q_len < seq_len   (e.g., "q4s1k" → 4 tokens, 1024 seq length)

Examples:
  q2k              -> [(2048, 2048)]           # Prefill: 2048 tokens
  q1s1k            -> [(1, 1024)]              # Decode: 1 token, 1K sequence
  8q1s1k           -> [(1, 1024)] * 8          # 8 decode requests
  q4s1k            -> [(4, 1024)]              # 4-token extend (spec decode)
  2q1k_32q1s1k     -> [(1024, 1024)] * 2 + [(1, 1024)] * 32  # Mixed batch
  16q4s1k          -> [(4, 1024)] * 16         # 16 spec decode requests
"""

from collections import Counter
from dataclasses import dataclass

import regex as re


@dataclass
class BatchRequest:
    """Represents a single request in a batch."""

    q_len: int  # Query length (number of new tokens)
    kv_len: int  # Total KV cache length

    @property
    def is_decode(self) -> bool:
        """True if this is a decode request (q_len == 1)."""
        return self.q_len == 1

    @property
    def is_prefill(self) -> bool:
        """True if this is a pure prefill (q_len == kv_len)."""
        return self.q_len == self.kv_len

    @property
    def is_extend(self) -> bool:
        """True if this is context extension (q_len > 1, kv_len > q_len)."""
        return self.q_len > 1 and self.kv_len > self.q_len

    @property
    def context_len(self) -> int:
        """Context length (KV cache - query)."""
        return self.kv_len - self.q_len

    def as_tuple(self) -> tuple[int, int]:
        """Return as (q_len, kv_len) tuple for compatibility."""
        return (self.q_len, self.kv_len)


def _parse_size(size_str: str, k_suffix: str) -> int:
    """Parse size string with optional 'k' suffix."""
    size = int(size_str)
    return size * 1024 if k_suffix == "k" else size


def parse_batch_spec(spec: str) -> list[BatchRequest]:
    """
    Parse batch specification string into list of BatchRequest objects.

    Grammar: (<count>?) q<q_len>(k?) (s<seq_len>(k?))?

    Args:
        spec: Batch specification string (see module docstring for grammar)

    Returns:
        List of BatchRequest objects

    Raises:
        ValueError: If spec format is invalid
    """
    requests = []

    for seg in spec.split("_"):
        # Unified pattern: (<count>?) q<q_len>(k?) (s<seq_len>(k?))?
        m = re.match(r"^(?:(\d+))?q(\d+)(k?)(?:s(\d+)(k?))?$", seg)
        if m:
            cnt = int(m.group(1)) if m.group(1) else 1
            q_len = _parse_size(m.group(2), m.group(3))
            kv_len = _parse_size(m.group(4), m.group(5)) if m.group(4) else q_len
            requests.extend([BatchRequest(q_len=q_len, kv_len=kv_len)] * cnt)
            continue

        raise ValueError(f"Invalid batch spec segment: '{seg}'")

    return requests


def format_batch_spec(requests: list[BatchRequest]) -> str:
    """
    Format list of BatchRequest into human-readable string.

    Groups requests by type and provides counts and sizes.

    Args:
        requests: List of BatchRequest objects

    Returns:
        Formatted string describing the batch
    """
    kinds = {
        "prefill": [],
        "extend": [],
        "decode": [],
    }

    for req in requests:
        tup = (req.q_len, req.kv_len)
        if req.is_prefill:
            kinds["prefill"].append(tup)
        elif req.is_extend:
            kinds["extend"].append(tup)
        elif req.is_decode:
            kinds["decode"].append(tup)

    parts = []
    for kind in ["prefill", "extend", "decode"]:
        lst = kinds[kind]
        if not lst:
            continue

        cnt_total = len(lst)
        ctr = Counter(lst)
        inner = []

        for (q, kv), cnt in ctr.items():
            if kind == "prefill":
                size = f"{q // 1024}k" if q % 1024 == 0 else str(q)
                inner.append(f"{cnt}x{size}")
            elif kind == "decode":
                size = f"{kv // 1024}k" if kv % 1024 == 0 else str(kv)
                inner.append(f"{cnt}x{size}")
            else:  # extend
                qstr = f"{q // 1024}k" if q % 1024 == 0 else str(q)
                kstr = f"{kv // 1024}k" if kv % 1024 == 0 else str(kv)
                inner.append(f"{cnt}xq{qstr}kv{kstr}")

        parts.append(f"{cnt_total} {kind} ({', '.join(inner)})")

    return ", ".join(parts)


def reorder_for_flashinfer(requests: list[BatchRequest]) -> list[BatchRequest]:
    """
    Reorder requests for FlashInfer: decode first, then prefill.

    FlashInfer expects decode requests before prefill requests for
    optimal performance.

    Args:
        requests: Original list of BatchRequest

    Returns:
        Reordered list with decode requests first
    """
    decodes = [r for r in requests if r.is_decode]
    non_decodes = [r for r in requests if not r.is_decode]
    return decodes + non_decodes


def split_by_type(
    requests: list[BatchRequest],
) -> dict[str, list[BatchRequest]]:
    """
    Split requests by type for analysis.

    Args:
        requests: List of BatchRequest

    Returns:
        Dict with keys: 'decode', 'prefill', 'extend'
    """
    result = {
        "decode": [],
        "prefill": [],
        "extend": [],
    }

    for req in requests:
        if req.is_decode:
            result["decode"].append(req)
        elif req.is_prefill:
            result["prefill"].append(req)
        elif req.is_extend:
            result["extend"].append(req)

    return result


def get_batch_stats(requests: list[BatchRequest]) -> dict:
    """
    Compute statistics about a batch.

    Args:
        requests: List of BatchRequest

    Returns:
        Dict with batch statistics
    """
    by_type = split_by_type(requests)

    return {
        "total_requests": len(requests),
        "num_decode": len(by_type["decode"]),
        "num_prefill": len(by_type["prefill"]),
        "num_extend": len(by_type["extend"]),
        "total_tokens": sum(r.q_len for r in requests),
        "total_kv_cache": sum(r.kv_len for r in requests),
        "max_q_len": max((r.q_len for r in requests), default=0),
        "max_kv_len": max((r.kv_len for r in requests), default=0),
        "avg_q_len": sum(r.q_len for r in requests) / len(requests) if requests else 0,
        "avg_kv_len": (
            sum(r.kv_len for r in requests) / len(requests) if requests else 0
        ),
    }
