# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from unittest.mock import patch

import torch

from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemoryProfilingResult, MemorySnapshot
from vllm.utils.startup_memory_debug import (
    StartupMemoryTracker,
    log_allocator_stats,
    log_largest_segments,
)

_LOGGER_NAME = "vllm.utils.startup_memory_debug"
_GiB = 1 << 30


def _snapshot(
    torch_allocated: int = 0,
    torch_memory: int = 0,
    cuda_memory: int = 0,
    non_torch_memory: int = 0,
):
    snap = MemorySnapshot(auto_measure=False)
    snap.torch_allocated = torch_allocated
    snap.torch_memory = torch_memory
    snap.cuda_memory = cuda_memory
    snap.non_torch_memory = non_torch_memory
    return snap


def _mem_debug_lines(caplog):
    return [r.message for r in caplog.records if "[MEM DEBUG]" in r.message]


def _make_tracker(monkeypatch, snapshots: list[MemorySnapshot]) -> StartupMemoryTracker:
    """A tracker whose checkpoints replay the given canned snapshots."""
    monkeypatch.setenv("VLLM_DEBUG_STARTUP_MEMORY", "1")
    it = iter(snapshots)

    def fake_measure(self):
        canned = next(it)
        for name in (
            "torch_peak",
            "torch_allocated",
            "free_memory",
            "total_memory",
            "cuda_memory",
            "torch_memory",
            "non_torch_memory",
        ):
            setattr(self, name, getattr(canned, name))

    monkeypatch.setattr(MemorySnapshot, "measure", fake_measure)
    return StartupMemoryTracker()


def test_checkpoint_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_DEBUG_STARTUP_MEMORY", raising=False)
    tracker = StartupMemoryTracker()
    with patch.object(MemorySnapshot, "measure", side_effect=AssertionError) as m:
        tracker.checkpoint("before_create", rank=0)
        assert not m.called


def test_checkpoint_logs_stage_and_deltas(monkeypatch, caplog):
    tracker = _make_tracker(
        monkeypatch,
        [
            _snapshot(cuda_memory=2 * _GiB, non_torch_memory=1 * _GiB),
            _snapshot(cuda_memory=5 * _GiB, non_torch_memory=2 * _GiB),
        ],
    )
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        tracker.checkpoint("before_create", rank=3)
        tracker.checkpoint("after_load_model", rank=3)

    lines = _mem_debug_lines(caplog)
    assert len(lines) == 2
    first, second = lines
    assert "rank=3" in first and "host=" in first
    assert "before_create" in first
    assert "gpu_used=2.0" in first
    assert "after_load_model" in second
    assert "gpu_used=5.0" in second
    assert "+gpu_used=3.0" in second
    assert "+non_torch=1.0" in second
    assert "since before_create" in second


def test_log_summary(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_DEBUG_STARTUP_MEMORY", "1")
    tracker = StartupMemoryTracker()
    result = MemoryProfilingResult(
        total_consumed=10 * _GiB,
        torch_peak_increase=4 * _GiB,
        non_torch_increase=2 * _GiB,
        weights_memory=3 * _GiB,
        transient_peak_headroom=1 * _GiB,
        before_create=MemorySnapshot(auto_measure=False),
    )
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        tracker.log_summary(result, rank=28)
    (line,) = _mem_debug_lines(caplog)
    assert "rank=28" in line
    assert "total_consumed=10.0" in line
    assert "torch_peak_increase=4.0" in line
    assert "non_torch_increase=2.0" in line
    assert "weights=3.0" in line
    assert "transient_peak_headroom=1.0" in line


def test_log_allocator_stats(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_DEBUG_STARTUP_MEMORY", "1")

    def fake_memory_stats(device=None):
        return {
            "allocated_bytes.all.current": 2 * _GiB,
            "reserved_bytes.all.current": 3 * _GiB,
            "inactive_split_bytes.all.current": 512 * (1 << 20),
            "segment.all.current": 42,
            "num_alloc_retries": 7,
        }

    with (
        patch.object(torch.accelerator, "memory_stats", fake_memory_stats),
        caplog.at_level(logging.INFO, logger=_LOGGER_NAME),
    ):
        log_allocator_stats(rank=0)
    (line,) = _mem_debug_lines(caplog)
    assert "allocated=2.0" in line
    assert "reserved=3.0" in line
    assert "inactive_split=0.5" in line
    assert "segments=42" in line
    assert "alloc_retries=7" in line


def test_log_largest_segments(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_DEBUG_STARTUP_MEMORY", "1")

    segments = [
        {"total_size": 4 * _GiB, "active_size": 4 * _GiB},
        {"total_size": 2 * _GiB, "active_size": 1 * _GiB},
        {"total_size": 1 * _GiB, "active_size": 0},
    ]

    with (
        patch.object(current_platform.__class__, "is_cuda", return_value=True),
        patch("torch.cuda.memory_snapshot", return_value=segments),
        caplog.at_level(logging.INFO, logger=_LOGGER_NAME),
    ):
        log_largest_segments(rank=1, top_k=2)
    (line,) = _mem_debug_lines(caplog)
    assert "4.0/4.0" in line
    assert "2.0/1.0" in line
    assert "1.0/0.0" not in line  # top_k=2 excludes the third segment
    assert "partially_used_segments=1" in line
    assert "free_bytes_in_partial_segments=1.0" in line


def test_log_largest_segments_skips_non_cuda(monkeypatch):
    monkeypatch.setenv("VLLM_DEBUG_STARTUP_MEMORY", "1")

    with (
        patch.object(current_platform.__class__, "is_cuda", return_value=False),
        patch(
            "torch.cuda.memory_snapshot",
            side_effect=AssertionError("should not be called"),
        ) as m,
    ):
        log_largest_segments(rank=0)
    assert not m.called
