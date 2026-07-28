# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for hierarchical expert staging."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TierStats:
    """In-process counters for hierarchical staging."""

    device_hits: int = 0
    ram_hits: int = 0
    disk_hits: int = 0
    dma_bytes: int = 0
    stall_ns: int = 0
    unique_experts: int = 0
    ensures: int = 0

    def snapshot(self) -> dict[str, int | float]:
        total = self.device_hits + self.ram_hits + self.disk_hits
        return {
            "device_hits": self.device_hits,
            "ram_hits": self.ram_hits,
            "disk_hits": self.disk_hits,
            "dma_bytes": self.dma_bytes,
            "stall_ms": self.stall_ns / 1e6,
            "unique_experts": self.unique_experts,
            "ensures": self.ensures,
            "device_hit_rate": self.device_hits / max(1, total),
        }


_PROM_REGISTERED = False
_prom_counters: dict[str, object] = {}


def _ensure_prometheus() -> None:
    global _PROM_REGISTERED
    if _PROM_REGISTERED:
        return
    try:
        from prometheus_client import Counter, Gauge

        _prom_counters["hits"] = Counter(
            "vllm_tier_expert_hits_total",
            "Hierarchical expert staging hits by tier",
            ["tier"],
        )
        _prom_counters["dma_bytes"] = Counter(
            "vllm_tier_expert_dma_bytes_total",
            "Bytes DMA'd into device expert slots",
        )
        _prom_counters["stall_seconds"] = Counter(
            "vllm_tier_expert_stall_seconds_total",
            "Seconds stalled waiting for expert DMA",
        )
        _prom_counters["hit_rate"] = Gauge(
            "vllm_tier_expert_device_hit_rate",
            "Device-tier hit rate for hierarchical expert staging",
        )
    except Exception:
        pass
    _PROM_REGISTERED = True


def record_stats(stats: TierStats) -> None:
    """Push current stats into Prometheus if available."""
    _ensure_prometheus()
    snap = stats.snapshot()
    gauge = _prom_counters.get("hit_rate")
    if gauge is not None:
        try:
            gauge.set(snap["device_hit_rate"])  # type: ignore[attr-defined]
        except Exception:
            pass


def increment_prom(tier: str, *, dma_bytes: int = 0, stall_ns: int = 0) -> None:
    """Increment prometheus counters for a single ensure event."""
    _ensure_prometheus()
    hits = _prom_counters.get("hits")
    if hits is not None:
        try:
            hits.labels(tier=tier).inc()  # type: ignore[attr-defined]
        except Exception:
            pass
    if dma_bytes and (c := _prom_counters.get("dma_bytes")) is not None:
        try:
            c.inc(dma_bytes)  # type: ignore[attr-defined]
        except Exception:
            pass
    if stall_ns and (c := _prom_counters.get("stall_seconds")) is not None:
        try:
            c.inc(stall_ns / 1e9)  # type: ignore[attr-defined]
        except Exception:
            pass
