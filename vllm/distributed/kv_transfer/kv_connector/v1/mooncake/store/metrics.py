# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-operation telemetry for MooncakeStoreConnector.

Records one row per Mooncake RPC (``save_exists``, ``save_put``, ``load_get``,
``lookup_exists``) with duration, key/byte counts, status, and failed-key
count. Exposed to the logger via ``KVConnectorLogging`` and to Prometheus
via ``MooncakeStorePromMetrics``.

The scheduler role can also attach a snapshot of the Mooncake master's memory
gauges (see ``MooncakeMasterUsagePoller``), from which the store usage
fractions are derived.
"""

import threading
import urllib.request
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


# Reserved ``data`` key for master memory snapshots. It is not an operation.
MASTER_USAGE_KEY = "master_usage"

# Same buckets as vllm:kv_offload_cpu_allocation_size.
_ALLOCATION_SIZE_BUCKETS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144]


# Master gauge name -> snapshot field. processing/leased bytes exist only on
# masters that publish them; without them only ``occupancy`` is derived.
_MASTER_GAUGES = {
    "master_total_capacity_bytes": "capacity_bytes",
    "master_allocated_bytes": "allocated_bytes",
    "master_mem_processing_bytes": "processing_bytes",
    "master_mem_leased_bytes": "leased_bytes",
}


def master_usage_fractions(snapshot: dict[str, Any]) -> dict[str, float]:
    """Store usage fractions from one master memory snapshot.

    ``usage`` counts memory eviction cannot reclaim now: puts in flight plus
    data under a read lease. ``occupancy`` also counts evictable data.
    """
    capacity = float(snapshot.get("capacity_bytes", 0))
    if capacity <= 0:
        return {}
    fractions = {}
    if "allocated_bytes" in snapshot:
        fractions["occupancy"] = snapshot["allocated_bytes"] / capacity
    processing = snapshot.get("processing_bytes")
    leased = snapshot.get("leased_bytes")
    if processing is not None:
        fractions["write_usage"] = processing / capacity
    if leased is not None:
        fractions["read_usage"] = leased / capacity
    if processing is not None and leased is not None:
        fractions["usage"] = (processing + leased) / capacity
    return fractions


def parse_master_metrics(text: str) -> dict[str, int]:
    """Pick the memory gauges out of the master's Prometheus text."""
    snapshot: dict[str, int] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)[0], line.rsplit("}", 1)[1]
        else:
            name, _, rest = line.partition(" ")
        field = _MASTER_GAUGES.get(name)
        if field is None:
            continue
        snapshot[field] = snapshot.get(field, 0) + int(float(rest.split()[0]))
    return snapshot


class MooncakeMasterUsagePoller:
    """Polls the Mooncake master's /metrics endpoint on a daemon thread.

    The scheduler step must not block on HTTP, so the step only takes the
    latest snapshot, at most once per poll.
    """

    def __init__(self, url: str, interval_s: float = 1.0, timeout_s: float = 2.0):
        self._url = url
        self._interval_s = interval_s
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._snapshot: dict[str, int] | None = None
        self._stop = threading.Event()
        self._warned = False
        self._thread = threading.Thread(
            target=self._run, name="mooncake-master-usage", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(self._url, timeout=self._timeout_s) as r:
                    snapshot = parse_master_metrics(r.read().decode())
                if "processing_bytes" not in snapshot and not self._warned:
                    logger.warning(
                        "Mooncake master at %s publishes no "
                        "master_mem_processing_bytes/master_mem_leased_bytes; "
                        "only store occupancy is reported.",
                        self._url,
                    )
                    self._warned = True
                with self._lock:
                    self._snapshot = snapshot
            except Exception as e:
                logger.debug("Mooncake master metrics poll failed: %s", e)
            self._stop.wait(self._interval_s)

    def take(self) -> dict[str, int] | None:
        with self._lock:
            snapshot, self._snapshot = self._snapshot, None
        return snapshot

    def close(self) -> None:
        self._stop.set()


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = max(
        0, min(len(sorted_values) - 1, int(percentile * len(sorted_values) - 1e-12))
    )
    return sorted_values[rank]


@dataclass
class MooncakeStoreConnectorStats(KVConnectorStats):
    """Serializable Mooncake store communication telemetry."""

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        self.data: dict[str, list[dict[str, int | float | str]]] = {}

    def is_empty(self) -> bool:
        return not self.data

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if other.is_empty():
            return self
        for operation, records in other.data.items():
            self.data.setdefault(operation, []).extend(records)
        return self

    def reduce(self) -> dict[str, int | float]:
        reduced: dict[str, int | float] = {}
        for operation, records in sorted(self.data.items()):
            if not records:
                continue
            if operation == MASTER_USAGE_KEY:
                for name, value in master_usage_fractions(records[-1]).items():
                    reduced[f"store_{name}_perc"] = round(value * 100, 2)
                continue
            durations = [float(record["duration_seconds"]) for record in records]
            reduced[f"{operation}_count"] = len(records)
            reduced[f"{operation}_avg_ms"] = round(fmean(durations) * 1e3, 3)
            reduced[f"{operation}_p90_ms"] = round(
                _nearest_rank_percentile(durations, 0.9) * 1e3, 3
            )
            reduced[f"{operation}_total_keys"] = sum(
                int(record["num_keys"]) for record in records
            )
            reduced[f"{operation}_total_bytes"] = sum(
                int(record["num_bytes"]) for record in records
            )
            reduced[f"{operation}_failed_keys"] = sum(
                int(record["num_failed_keys"]) for record in records
            )
            reduced[f"{operation}_error_count"] = sum(
                1 for record in records if record["status"] == "error"
            )
            if operation == "save_exists":
                reduced[f"{operation}_existing_keys"] = sum(
                    int(record.get("num_existing_keys", 0)) for record in records
                )
        return reduced

    def record_operation(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
        num_existing_keys: int = 0,
    ) -> None:
        self.data.setdefault(operation, []).append(
            {
                "duration_seconds": duration_seconds,
                "num_keys": num_keys,
                "num_bytes": num_bytes,
                "status": status,
                "num_failed_keys": num_failed_keys,
                "num_existing_keys": num_existing_keys,
            }
        )

    def record_master_usage(self, snapshot: dict[str, int]) -> None:
        self.data.setdefault(MASTER_USAGE_KEY, []).append(dict(snapshot))


class MooncakeStorePromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for Mooncake store communication."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        metric_labelnames = labelnames + ["operation", "status"]
        self._metric_cache: dict[tuple[int, str, str], dict[str, PromMetric]] = {}

        self._histogram_operation_time = self._histogram_cls(
            name="vllm:mooncake_store_operation_time_seconds",
            documentation="Histogram of Mooncake store communication time.",
            buckets=[
                1e-3,
                5e-3,
                1e-2,
                5e-2,
                1e-1,
                2e-1,
                3e-1,
                4e-1,
                5e-1,
                7.5e-1,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
            ],
            labelnames=metric_labelnames,
        )
        self._counter_operation_calls = self._counter_cls(
            name="vllm:mooncake_store_operation_total",
            documentation="Number of Mooncake store communication operations.",
            labelnames=metric_labelnames,
        )
        self._counter_operation_keys = self._counter_cls(
            name="vllm:mooncake_store_operation_keys_total",
            documentation="Number of Mooncake store keys touched by operations.",
            labelnames=metric_labelnames,
        )
        self._counter_operation_bytes = self._counter_cls(
            name="vllm:mooncake_store_operation_bytes_total",
            documentation="Number of bytes transferred by Mooncake store operations.",
            labelnames=metric_labelnames,
        )
        self._counter_failed_keys = self._counter_cls(
            name="vllm:mooncake_store_operation_failed_keys_total",
            documentation="Number of Mooncake store keys that failed in operations.",
            labelnames=metric_labelnames,
        )
        self._histogram_allocation_size = self._histogram_cls(
            name="vllm:mooncake_store_allocation_size",
            documentation="Histogram of keys per save_put call, after "
            "dropping keys the store already holds.",
            buckets=_ALLOCATION_SIZE_BUCKETS,
            labelnames=labelnames,
        )
        self._counter_existing_keys = self._counter_cls(
            name="vllm:mooncake_store_save_existing_keys",
            documentation="Save candidate keys skipped because the store "
            "already holds them.",
            labelnames=labelnames,
        )
        gauge_docs = {
            "usage": "Fraction of Mooncake store memory eviction cannot "
            "reclaim now: puts in flight plus data under a read lease.",
            "write_usage": "Fraction of Mooncake store memory held by puts in flight.",
            "read_usage": "Fraction of Mooncake store memory held by "
            "complete data under an unexpired read lease.",
            "occupancy": "Fraction of Mooncake store memory allocated, "
            "including evictable data.",
        }
        self._usage_gauges = {
            name: self._gauge_cls(
                name=f"vllm:mooncake_store_cache_{name}_perc",
                documentation=doc,
                labelnames=labelnames,
                multiprocess_mode="mostrecent",
            )
            for name, doc in gauge_docs.items()
        }

    def _get_metrics(
        self,
        engine_idx: int,
        operation: str,
        status: str,
    ) -> dict[str, PromMetric]:
        cache_key = (engine_idx, operation, status)
        if cache_key not in self._metric_cache:
            label_values = self.per_engine_labelvalues[engine_idx] + [operation, status]
            self._metric_cache[cache_key] = {
                "time": self._histogram_operation_time.labels(*label_values),
                "calls": self._counter_operation_calls.labels(*label_values),
                "keys": self._counter_operation_keys.labels(*label_values),
                "bytes": self._counter_operation_bytes.labels(*label_values),
                "failed_keys": self._counter_failed_keys.labels(*label_values),
            }
        return self._metric_cache[cache_key]

    def observe(self, transfer_stats_data: dict[str, Any] | None, engine_idx: int = 0):
        if not transfer_stats_data:
            return
        engine_labelvalues = self.per_engine_labelvalues[engine_idx]
        for operation, records in transfer_stats_data.items():
            assert isinstance(records, list)
            if operation == MASTER_USAGE_KEY:
                if records:
                    fractions = master_usage_fractions(records[-1])
                    for name, value in fractions.items():
                        self._usage_gauges[name].labels(*engine_labelvalues).set(value)
                continue
            for record in records:
                assert isinstance(record, dict)
                status = str(record["status"])
                metrics = self._get_metrics(engine_idx, operation, status)
                metrics["time"].observe(float(record["duration_seconds"]))
                metrics["calls"].inc()
                metrics["keys"].inc(int(record["num_keys"]))
                metrics["bytes"].inc(int(record["num_bytes"]))
                metrics["failed_keys"].inc(int(record["num_failed_keys"]))
                if operation == "save_put":
                    self._histogram_allocation_size.labels(*engine_labelvalues).observe(
                        int(record["num_keys"])
                    )
                elif operation == "save_exists":
                    self._counter_existing_keys.labels(*engine_labelvalues).inc(
                        int(record.get("num_existing_keys", 0))
                    )
