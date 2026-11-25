# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer L1 operator counters and latest per-engine gauges."""

from itertools import product
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)

Definitions = dict[str, tuple[str, dict[str, tuple[str, ...]]]]

RDMA_FAILURE_REASONS = (
    "deadline_before_post",
    "deadline_after_post",
    "status_failed",
    "exception",
    "queue_full",
    "owner_changed",
    "cancelled",
)
TRANSPORT_LABELS: dict[str, tuple[str, ...]] = {
    "transport": ("ipc", "rdma"),
}
# Label vocabularies are closed; peer identities never enter the payload.
COUNTERS: Definitions = {
    "reader_lookups": ("Peer prefix lookups, including retries.", {}),
    "reader_tokens": ("Selected peer prefix tokens.", TRANSPORT_LABELS),
    "reader_granted_rows": ("Row positions in selected grants, including aliases.", {}),
    "reader_truncations": (
        "Candidate plans shortened by a row limit.",
        {
            "reason": ("window", "pin_cap"),
        },
    ),
    "reader_deadline_fallbacks": ("Requests falling back at the busy deadline.", {}),
    "worker_copy_bytes": ("Bytes submitted to peer copies.", TRANSPORT_LABELS),
    "worker_rdma_reads": ("RDMA receive grants submitted.", {}),
    "worker_rdma_failures": (
        "Failed asynchronous receive requests.",
        {
            "reason": RDMA_FAILURE_REASONS,
        },
    ),
    "owner_rejections": (
        "Owner authorization rejections.",
        {
            "reason": (
                "window",
                "cap",
                "frozen",
                "lease_expiry",
                "missing_key",
                "budget",
            ),
            "stage": ("immediate", "deferred"),
        },
    ),
}
GAUGES: Definitions = {
    "reader_quarantine_refs": ("Target row references held for uncertain DMA.", {}),
    "reader_rdma_quarantine_disabled": ("RDMA selection disabled by quarantine.", {}),
}


def stat_key(name: str, labels: dict[str, str], definitions: Definitions) -> str:
    _, vocabulary = definitions[name]
    if labels.keys() != vocabulary.keys() or any(
        value not in vocabulary[label] for label, value in labels.items()
    ):
        raise ValueError(f"Invalid labels for peer L1 metric {name!r}: {labels}")
    return ",".join((name, *(labels[label] for label in vocabulary)))


class PeerL1Stats(KVConnectorStats):
    def inc(self, name: str, value: int = 1, **labels: str) -> None:
        key = stat_key(name, labels, COUNTERS)
        counters = self.data.setdefault("counters", {})
        counters[key] = counters.get(key, 0) + value

    def gauge(self, name: str, value: int) -> None:
        key = stat_key(name, {}, GAUGES)
        self.data.setdefault("gauges", {})[key] = value

    def reset(self) -> None:
        self.data = {}

    def clone_and_reset(self) -> "PeerL1Stats":
        snapshot = PeerL1Stats(self.data)
        self.reset()
        return snapshot

    def aggregate(self, other: KVConnectorStats) -> "PeerL1Stats":
        if not isinstance(other, PeerL1Stats):
            raise TypeError("Cannot aggregate incompatible peer L1 stats")
        for key, value in other.data.get("counters", {}).items():
            name, *values = key.split(",")
            labels = dict(zip(COUNTERS[name][1], values, strict=True))
            self.inc(name, value, **labels)
        for name, value in other.data.get("gauges", {}).items():
            self.gauge(name, value)
        return self

    def reduce(self) -> dict[str, int | float]:
        result = dict(self.data.get("gauges", {}))
        for key, value in self.data.get("counters", {}).items():
            name = key.split(",", 1)[0]
            result[name] = result.get(name, 0) + value
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
        for definitions, cls, suffix in (
            (COUNTERS, self._counter_cls, "_total"),
            (GAUGES, self._gauge_cls, ""),
        ):
            for name, (documentation, vocabulary) in definitions.items():
                kwargs: dict[str, Any] = {}
                if not suffix:
                    kwargs["multiprocess_mode"] = "mostrecent"
                family = cls(
                    name=f"vllm:peer_l1_{name}{suffix}",
                    documentation=documentation,
                    labelnames=[*labelnames, *vocabulary],
                    **kwargs,
                )
                for values in product(*vocabulary.values()):
                    key = stat_key(name, dict(zip(vocabulary, values)), definitions)
                    for engine, engine_values in per_engine_labelvalues.items():
                        self.metrics[engine, key] = family.labels(
                            *engine_values, *values
                        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for key, value in transfer_stats_data.get("counters", {}).items():
            self.metrics[engine_idx, key].inc(value)
        for key, value in transfer_stats_data.get("gauges", {}).items():
            self.metrics[engine_idx, key].set(value)
