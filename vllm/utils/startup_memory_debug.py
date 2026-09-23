# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-stage GPU memory logging during startup, gated by
VLLM_DEBUG_STARTUP_MEMORY=1."""

import socket

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemoryProfilingResult, MemorySnapshot, format_gib

logger = init_logger(__name__)

_PREFIX = "[MEM DEBUG]"


def _enabled() -> bool:
    return envs.VLLM_DEBUG_STARTUP_MEMORY


class StartupMemoryTracker:
    """Logs a memory checkpoint at each startup stage.

    Each checkpoint reports gpu_used / torch_reserved / torch_allocated /
    non_torch, plus deltas since the previous checkpoint.
    """

    def __init__(self) -> None:
        self._prev: MemorySnapshot | None = None
        self._prev_stage: str | None = None

    def checkpoint(self, stage: str, rank: int) -> None:
        if not _enabled():
            return
        snap = MemorySnapshot()
        delta = ""
        if self._prev is not None:
            delta = (
                f" (+gpu_used={format_gib(snap.cuda_memory - self._prev.cuda_memory)}g"
                f", +non_torch="
                f"{format_gib(snap.non_torch_memory - self._prev.non_torch_memory)}g"
                f" since {self._prev_stage})"
            )
        logger.info(
            "%s rank=%d host=%s   %-15s: gpu_used=%s torch_reserved=%s "
            "torch_allocated=%s non_torch=%s%s",
            _PREFIX,
            rank,
            socket.gethostname(),
            stage,
            format_gib(snap.cuda_memory),
            format_gib(snap.torch_memory),
            format_gib(snap.torch_allocated),
            format_gib(snap.non_torch_memory),
            delta,
        )
        self._prev = snap
        self._prev_stage = stage

    def log_summary(self, result: MemoryProfilingResult, rank: int) -> None:
        if not _enabled():
            return
        logger.info(
            "%s rank=%d host=%s total_consumed=%s GiB = torch_peak_increase=%s "
            "+ non_torch_increase=%s (weights=%s, transient_peak_headroom=%s)",
            _PREFIX,
            rank,
            socket.gethostname(),
            format_gib(result.total_consumed),
            format_gib(result.torch_peak_increase),
            format_gib(result.non_torch_increase),
            format_gib(result.weights_memory),
            format_gib(result.transient_peak_headroom),
        )


def log_allocator_stats(rank: int) -> None:
    if not _enabled():
        return
    try:
        stats = torch.accelerator.memory_stats()
    except Exception:
        return
    logger.info(
        "%s rank=%d host=%s allocator: allocated=%s reserved=%s "
        "inactive_split=%s segments=%s alloc_retries=%s",
        _PREFIX,
        rank,
        socket.gethostname(),
        format_gib(stats.get("allocated_bytes.all.current", 0)),
        format_gib(stats.get("reserved_bytes.all.current", 0)),
        format_gib(stats.get("inactive_split_bytes.all.current", 0)),
        stats.get("segment.all.current", 0),
        stats.get("num_alloc_retries", 0),
    )


def log_largest_segments(rank: int, top_k: int = 5) -> None:
    if not _enabled() or not current_platform.is_cuda():
        return
    try:
        segments = torch.cuda.memory_snapshot()
    except Exception:
        return
    if not segments:
        return
    by_total = sorted(segments, key=lambda s: s["total_size"], reverse=True)
    top = [
        f"{format_gib(s['total_size'])}/{format_gib(s['active_size'])}"
        for s in by_total[:top_k]
    ]
    partial = [s for s in segments if 0 < s["active_size"] < s["total_size"]]
    free_in_partial = sum(s["total_size"] - s["active_size"] for s in partial)
    logger.info(
        "%s rank=%d host=%s largest segments GiB total/allocated: [%s] "
        "(partially_used_segments=%d, free_bytes_in_partial_segments=%s)",
        _PREFIX,
        rank,
        socket.gethostname(),
        ", ".join(top),
        len(partial),
        format_gib(free_in_partial),
    )


def log_workspace_sizes(rank: int) -> None:
    if not _enabled():
        return
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        is_workspace_manager_initialized,
    )

    if not is_workspace_manager_initialized():
        return
    sizes = current_workspace_manager().workspace_sizes()
    logger.info(
        "%s rank=%d host=%s workspace GiB: [%s]",
        _PREFIX,
        rank,
        socket.gethostname(),
        ", ".join(format_gib(b) for b in sizes),
    )
