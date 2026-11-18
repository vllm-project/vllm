# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded interval statistics and per-engine peer L1 Prometheus metrics."""

from bisect import bisect_left
from math import inf, nextafter
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)

SECONDS = (0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 100.0)
POSITIONS = (1.0, 4.0, 16.0, 64.0, 256.0, 512.0, 1024.0, 4096.0, 16384.0, 65536.0)
BYTES = (
    1024.0,
    16384.0,
    262144.0,
    1048576.0,
    4194304.0,
    16777216.0,
    67108864.0,
    268435456.0,
    1073741824.0,
    4294967296.0,
    17179869184.0,
    68719476736.0,
)
RDMA_FAILURE_REASONS = (
    "deadline_before_post",
    "deadline_after_post",
    "status_failed",
    "exception",
    "queue_full",
    "owner_changed",
    "cancelled",
)

COUNTERS = {
    "reader_rdma_cooldowns": "Owner failure feedback applying RDMA backoff.",
    "reader_rdma_quarantine_stops": "RDMA quarantine disable episodes.",
    **{
        f"owner_deferred_reject_{reason}": "Deferred plans rejected by the owner."
        for reason in ("frozen", "lease_expiry", "missing_key", "cap", "budget")
    },
    "reader_lookups": "Calls to peer prefix matching, including retries.",
    "reader_quarantined_rows": "Target rows held against uncertain RDMA writes.",
    "worker_rdma_reads": "RDMA read grants submitted.",
    "worker_rdma_successes": "Requests completing an RDMA receive within deadline.",
    **{
        f"worker_rdma_failure_{reason}": "Failed asynchronous receive requests."
        for reason in RDMA_FAILURE_REASONS
    },
    "worker_rdma_bytes": "Bytes submitted for RDMA reads.",
    "reader_peer_tokens": "Eligible lookup tokens by peer or no-peer source.",
    "reader_none_tokens": "Eligible lookup tokens by peer or no-peer source.",
    "reader_peer_decisions": "Terminal peer lookup decisions by source.",
    "reader_none_decisions": "Terminal peer lookup decisions by source.",
    "reader_busy_replies": "Busy authorization replies received.",
    "reader_deferred_requests": "Requests first deferred for a busy owner.",
    "reader_deadline_fallbacks": "Requests falling back at the busy deadline.",
    "reader_window_truncations": "Candidate plans shortened by the binding row limit.",
    "reader_pin_cap_truncations": "Candidate plans shortened by the binding row limit.",
    "reader_granted_rows": "Row positions in selected grants, including aliases.",
    "owner_grants": "New grants issued by this owner, excluding replayed RPCs.",
    "owner_rows_pinned": "Row pin references issued, unique within each grant.",
    "owner_busy_window": (
        "Authorizations rejected by the new-pin window, before checking cap."
    ),
    "owner_busy_cap": "Pin-cap rejections with window space available.",
    "owner_absorbs": "Atomic owner absorbs.",
    "owner_deferred_grants": "Grants committed on the owner scheduler at absorb.",
    "owner_deferred_rows": "Unique row pin references issued by deferred grants.",
    "worker_load_steps": "Steps submitting peer loads.",
    "worker_copy_bytes": "Bytes submitted to peer copies.",
    "worker_copy_descriptors": "Copy descriptors before driver batch splitting.",
    "worker_timing_dropped": "Load steps without timing due to a full event queue.",
}
GAUGES = {
    "reader_rdma_owners_cooling": "Owners cooling at the last scheduler snapshot.",
    "reader_rdma_quarantine_disabled": "RDMA selection disabled by quarantine.",
    "reader_rdma_quarantine_limit_rows": "Quarantine threshold for RDMA selection.",
    "owner_external_refs": "Outstanding external row references after absorb.",
    "reader_quarantine_refs": "Current target row references held for uncertain DMA.",
    "owner_withheld_high_water": "Lifetime peak of physically withheld blocks.",
    "owner_withheld_blocks": "Physically withheld blocks after the last absorb.",
    "owner_reserve_before_absorb": "Reserve before the last window reset.",
    "owner_reserve": "Admission reserve immediately after the last absorb.",
    "owner_effective_window": "Effective new-pin budget until the next absorb.",
    "reader_pool_free_blocks": "Physical free blocks after owner maintenance.",
    "reader_admissible_free_blocks": "Free blocks minus reserve, clamped at zero.",
}
HISTOGRAMS = {
    "scheduler_seconds_per_step": (
        SECONDS,
        "CPU wall time in connector hooks between metadata builds.",
    ),
    **{
        f"reader_busy_{outcome}_seconds": (
            SECONDS,
            "Wall time from first busy reply to resolution.",
        )
        for outcome in ("grant", "fallback", "local", "cancel")
    },
    **{
        f"reader_deferral_{outcome}_seconds": (
            SECONDS,
            "Wall time from queued acquire to resolution.",
        )
        for outcome in ("grant", "fallback", "local", "cancel")
    },
    **{
        f"reader_{reason}_{stage}_positions": (
            POSITIONS,
            "Row positions before or after a candidate truncation.",
        )
        for reason in ("window", "pin_cap")
        for stage in ("requested", "granted")
    },
    "worker_copy_bytes_per_step": (BYTES, "Copy bytes per step with loads."),
    "worker_copy_descriptors_per_step": (
        POSITIONS,
        "Coalesced descriptors per step with loads.",
    ),
    "worker_copy_gpu_seconds": (
        SECONDS,
        "Copy-stream GPU span after the prior-write fence.",
    ),
    "worker_main_wait_seconds": (
        SECONDS,
        "GPU span bracketing the main-stream wait for peer loads.",
    ),
}
# Each label has a fixed vocabulary; request IDs never enter the payload.
LABELS = {
    **{
        f"worker_rdma_failure_{reason}": ("worker_rdma_failures", "reason", reason)
        for reason in RDMA_FAILURE_REASONS
    },
    **{
        f"owner_deferred_reject_{reason}": (
            "owner_deferred_rejections",
            "reason",
            reason,
        )
        for reason in ("frozen", "lease_expiry", "missing_key", "cap", "budget")
    },
    **{
        f"reader_{source}_{metric}": (f"reader_{metric}", "source", source)
        for source in ("peer", "none")
        for metric in ("tokens", "decisions")
    },
    **{
        f"reader_{reason}_truncations": ("reader_truncations", "reason", reason)
        for reason in ("window", "pin_cap")
    },
    **{
        f"reader_busy_{outcome}_seconds": ("reader_busy_seconds", "outcome", outcome)
        for outcome in ("grant", "fallback", "local", "cancel")
    },
    **{
        f"reader_deferral_{outcome}_seconds": (
            "reader_deferral_seconds",
            "outcome",
            outcome,
        )
        for outcome in ("grant", "fallback", "local", "cancel")
    },
    **{
        f"reader_{reason}_{stage}_positions": (
            f"reader_truncation_{stage}_positions",
            "reason",
            reason,
        )
        for reason in ("window", "pin_cap")
        for stage in ("requested", "granted")
    },
}


class PeerL1Stats(KVConnectorStats):
    """Counters, latest gauges and non-cumulative fixed-bin histogram counts."""

    def inc(self, name: str, value: int = 1) -> None:
        counters = self.data.setdefault("counters", {})
        counters[name] = counters.get(name, 0) + value

    def gauge(self, name: str, value: int) -> None:
        gauges = self.data.setdefault("gauges", {})
        if name == "owner_withheld_high_water":
            value = max(value, gauges.get(name, 0))
        gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        buckets = HISTOGRAMS[name][0]
        histograms = self.data.setdefault("histograms", {})
        hist = histograms.get(name)
        if hist is None:
            hist = {"counts": [0] * (len(buckets) + 1), "sum": 0.0}
            histograms[name] = hist
        index = bisect_left(buckets, value)
        hist["counts"][index] += 1
        hist["sum"] += value
        sums = hist.setdefault("sums", [0.0] * (len(buckets) + 1))
        sums[index] += value

    def reset(self) -> None:
        self.data = {}

    def clone_and_reset(self) -> "PeerL1Stats":
        snapshot = PeerL1Stats(self.data)
        self.reset()
        return snapshot

    def aggregate(self, other: KVConnectorStats) -> "PeerL1Stats":
        if not isinstance(other, PeerL1Stats):
            raise TypeError("Cannot aggregate incompatible peer L1 stats")
        for name, value in other.data.get("counters", {}).items():
            self.inc(name, value)
        for name, value in other.data.get("gauges", {}).items():
            self.gauge(name, value)
        for name, hist in other.data.get("histograms", {}).items():
            target = self.data.setdefault("histograms", {}).setdefault(
                name, {"counts": [0] * len(hist["counts"]), "sum": 0.0}
            )
            for index, count in enumerate(hist["counts"]):
                target["counts"][index] += count
            target["sum"] += hist["sum"]
            if "sums" in hist:
                sums = target.setdefault("sums", [0.0] * len(hist["sums"]))
                for index, value in enumerate(hist["sums"]):
                    sums[index] += value
        return self

    def reduce(self) -> dict[str, int | float]:
        result = {**self.data.get("counters", {}), **self.data.get("gauges", {})}
        for name, hist in self.data.get("histograms", {}).items():
            result[f"{name}_count"] = sum(hist["counts"])
            result[f"{name}_sum"] = hist["sum"]
        return result

    def is_empty(self) -> bool:
        return not self.data


class PeerL1PromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self.metrics: dict[tuple[int, str], Any] = {}
        families: dict[str, Any] = {}
        for kind, definitions in (
            ("counters", COUNTERS),
            ("gauges", GAUGES),
            ("histograms", HISTOGRAMS),
        ):
            for key, definition in definitions.items():
                alias = LABELS.get(key)
                suffix = alias[0] if alias else key
                name = f"vllm:peer_l1_{suffix}"
                kwargs: dict[str, Any] = {
                    "name": name,
                    "labelnames": labelnames + ([alias[1]] if alias else []),
                }
                if kind == "histograms":
                    kwargs.update(buckets=definition[0], documentation=definition[1])
                    cls = self._histogram_cls
                else:
                    kwargs["documentation"] = definition
                    if kind == "gauges":
                        cls = self._gauge_cls
                        kwargs["multiprocess_mode"] = "mostrecent"
                    else:
                        cls = self._counter_cls
                        kwargs["name"] += "_total"
                if name not in families:
                    families[name] = cls(**kwargs)
                for engine, values in per_engine_labelvalues.items():
                    self.metrics[engine, key] = families[name].labels(
                        *(values + ([alias[2]] if alias else []))
                    )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for name, value in transfer_stats_data.get("counters", {}).items():
            if value:
                self.metrics[engine_idx, name].inc(value)
        for name, value in transfer_stats_data.get("gauges", {}).items():
            self.metrics[engine_idx, name].set(value)
        for name, hist in transfer_stats_data.get("histograms", {}).items():
            metric = self.metrics[engine_idx, name]
            bounds = (0.0, *HISTOGRAMS[name][0], inf)
            for index, (count, total) in enumerate(
                zip(hist["counts"], hist["sums"], strict=True)
            ):
                if not count:
                    continue
                value = min(
                    bounds[index + 1],
                    max(
                        0.0 if index == 0 else nextafter(bounds[index], inf),
                        total / count,
                    ),
                )
                for _ in range(count):
                    metric.observe(value)
