# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional stage timer for ZoomKV sparse-decode localization.

Enable with ``VLLM_ZOOMKV_STAGE_TIMER=1``.  CUDA-event timings are accumulated
per stage; call ``dump_and_reset()`` after a decode-heavy generate to print a
report.  Disabled by default so production paths stay untouched.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import torch

_ENABLED = os.environ.get("VLLM_ZOOMKV_STAGE_TIMER", "0") == "1"

# stage -> accumulated GPU ms / CPU wall ms / call count
_gpu_ms: dict[str, float] = defaultdict(float)
_cpu_ms: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)
_pending: list[tuple[str, Any, Any, float, float]] = []


def enabled() -> bool:
    return _ENABLED


class Stage:
    """CUDA-event + host wall timer for one named stage."""

    __slots__ = ("name", "start_evt", "end_evt", "t0")

    def __init__(self, name: str) -> None:
        self.name = name
        self.start_evt: Any | None = None
        self.end_evt: Any | None = None
        self.t0 = 0.0

    def __enter__(self) -> Stage:
        if not _ENABLED:
            return self
        self.t0 = time.perf_counter()
        if torch.accelerator.is_available():
            self.start_evt = torch.Event(enable_timing=True)
            self.end_evt = torch.Event(enable_timing=True)
            self.start_evt.record()
        return self

    def __exit__(self, *exc) -> None:
        if not _ENABLED:
            return
        t1 = time.perf_counter()
        if self.start_evt is not None and self.end_evt is not None:
            self.end_evt.record()
            _pending.append((self.name, self.start_evt, self.end_evt, self.t0, t1))
        else:
            _cpu_ms[self.name] += (t1 - self.t0) * 1e3
            _counts[self.name] += 1


def _flush_pending() -> None:
    if not _pending:
        return
    torch.accelerator.synchronize()
    for name, start_evt, end_evt, t0, t1 in _pending:
        _gpu_ms[name] += float(start_evt.elapsed_time(end_evt))
        _cpu_ms[name] += (t1 - t0) * 1e3
        _counts[name] += 1
    _pending.clear()


def dump_and_reset(label: str = "", decode_tokens: int | None = None) -> str:
    """Flush pending events and return a human-readable report."""
    if not _ENABLED:
        return "stage timer disabled"
    _flush_pending()
    names = sorted(_counts.keys(), key=lambda n: -_cpu_ms[n])
    lines = [f"==== ZoomKV stage timer {label} ===="]
    total_cpu = sum(_cpu_ms.values())
    total_gpu = sum(_gpu_ms.values())
    lines.append(
        f"{'stage':28s} {'calls':>8s} {'cpu_ms':>10s} {'gpu_ms':>10s} "
        f"{'cpu%':>6s} {'gap_ms':>10s}"
    )
    for name in names:
        c = _counts[name]
        cpu = _cpu_ms[name]
        gpu = _gpu_ms[name]
        gap = cpu - gpu
        pct = 100.0 * cpu / total_cpu if total_cpu > 0 else 0.0
        lines.append(
            f"{name:28s} {c:8d} {cpu:10.1f} {gpu:10.1f} {pct:5.1f}% {gap:10.1f}"
        )
    lines.append(
        f"{'TOTAL':28s} {'':>8s} {total_cpu:10.1f} {total_gpu:10.1f} "
        f"{'100.0%':>6s} {total_cpu - total_gpu:10.1f}"
    )
    if decode_tokens and decode_tokens > 0:
        # Approximate: each decode step hits all layers once; calls/layers ~ tokens.
        lines.append(
            f"per-decode-token cpu_ms≈{total_cpu / decode_tokens:.2f}  "
            f"gpu_ms≈{total_gpu / decode_tokens:.2f}  "
            f"(includes all layers)"
        )
    report = "\n".join(lines)
    _gpu_ms.clear()
    _cpu_ms.clear()
    _counts.clear()
    return report
