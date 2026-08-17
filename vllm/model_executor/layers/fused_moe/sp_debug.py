# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TEMPORARY debug instrumentation for validating sequence parallelism.

Enable with VLLM_SP_DEBUG=1. Prints are rate-limited per call-site key so a
long run does not drown in output; VLLM_SP_DEBUG_STEPS controls how many
forward passes are printed (default 3), VLLM_SP_DEBUG_RANKS restricts which
global ranks print (comma-separated, default all).

DELETE THIS FILE before landing anything.
"""

import os
from collections import defaultdict

import torch

_ENABLED = os.environ.get("VLLM_SP_DEBUG", "0") == "1"
_MAX_STEPS = int(os.environ.get("VLLM_SP_DEBUG_STEPS", "3"))
_RANK_FILTER = os.environ.get("VLLM_SP_DEBUG_RANKS", "")

_counts: dict[str, int] = defaultdict(int)


def sp_debug_enabled() -> bool:
    return _ENABLED


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def sp_print(key: str, msg: str, *, max_steps: int | None = None) -> None:
    """Print `msg` tagged with `key`, at most `max_steps` times per key."""
    if not _ENABLED:
        return
    rank = _rank()
    if _RANK_FILTER and str(rank) not in _RANK_FILTER.split(","):
        return
    n = _counts[key]
    if n >= (max_steps if max_steps is not None else _MAX_STEPS):
        return
    _counts[key] = n + 1
    print(f"[SP-DBG r{rank}] {key} #{n} | {msg}", flush=True)


def can_sync() -> bool:
    """False while a CUDA graph is capturing, where .item() would raise."""
    return not (
        torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    )


def tensor_desc(name: str, t: torch.Tensor | None) -> str:
    if t is None:
        return f"{name}=None"
    nbytes = t.numel() * t.element_size()
    return (
        f"{name}=shape{tuple(t.shape)} {str(t.dtype).removeprefix('torch.')} "
        f"{nbytes / 1024:.1f}KiB"
    )
