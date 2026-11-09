# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-operation telemetry for MooncakeStoreConnector.

Records one row per Mooncake RPC (``save_exists``, ``save_put``, ``load_get``,
``lookup_exists``) with duration, key/byte counts, status, and failed-key
count. Exposed to the logger via ``KVConnectorLogging`` and to Prometheus
via ``MooncakeStorePromMetrics``.

``MooncakeStorePromMetrics`` can also poll the Mooncake master's memory
gauges (see ``poll_master_metrics``) and export the store usage
fractions. It polls from the frontend process because connector stats only
arrive on engine steps: an idle engine would freeze the gauges while leases
expire and puts land in the store.

The master does not export memory pinned by puts in flight or by read leases,
so workers report their own: bytes entering and leaving a put, and the leases
the master grants them (reads and existence hits) as bytes per expiry second.
The frontend keeps the running write total and sums the lease seconds that
have not passed.
"""

import math
import threading
import time
import urllib.request
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from prometheus_client.parser import text_string_to_metric_families

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Reserved ``data`` keys for lease and in-flight write deltas. Not operations.
LEASES_KEY = "leases"
WRITES_KEY = "writes"

# Same buckets as vllm:kv_offload_cpu_allocation_size.
_ALLOCATION_SIZE_BUCKETS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144]


# Master gauge name -> snapshot field. Every Mooncake master publishes these.
_MASTER_GAUGES = {
    "master_total_capacity_bytes": "capacity_bytes",
    "master_allocated_bytes": "allocated_bytes",
    "master_put_start_discarded_staging_size": "discarded_bytes",
}


def master_usage_fractions(
    snapshot: dict[str, Any], leased_bytes: int, writing_bytes: int
) -> dict[str, float]:
    """Store usage fractions from one master snapshot and this instance's
    leased and in-flight write bytes.

    ``usage`` counts memory eviction cannot reclaim now: puts in flight, data
    under a read lease, and displaced replicas awaiting deferred release.
    ``occupancy`` also counts evictable data.
    """
    capacity = float(snapshot.get("capacity_bytes", 0))
    if capacity <= 0:
        return {}
    discarded = snapshot.get("discarded_bytes", 0)
    fractions = {
        "read_usage": leased_bytes / capacity,
        "write_usage": writing_bytes / capacity,
        "usage": (writing_bytes + leased_bytes + discarded) / capacity,
    }
    if "allocated_bytes" in snapshot:
        fractions["occupancy"] = snapshot["allocated_bytes"] / capacity
    return fractions


def parse_master_metrics(text: str) -> dict[str, int]:
    """Pick the memory gauges out of the master's Prometheus text."""
    snapshot: dict[str, int] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            field = _MASTER_GAUGES.get(sample.name)
            if field is not None:
                snapshot[field] = snapshot.get(field, 0) + int(sample.value)
    return snapshot


def poll_master_metrics(url: str) -> dict[str, int] | None:
    """One snapshot of the master's memory gauges; None if the poll fails."""
    try:
        with urllib.request.urlopen(url, timeout=2.0) as r:
            return parse_master_metrics(r.read().decode())
    except Exception as e:
        logger.debug("Mooncake master metrics poll failed: %s", e)
        return None


class LeaseTracker:
    """Read leases the master granted one worker: key -> (expiry second, bytes).

    ``grant`` returns the change in leased bytes per expiry second. A key
    leased again before its lease expires moves its bytes to the new expiry.
    """

    def __init__(self, ttl_s: float):
        self._ttl_s = ttl_s
        self._leases: dict[str, tuple[int, int]] = {}
        self._next_prune = 0.0

    def grant(
        self, leased: list[tuple[str, int]], now: float | None = None
    ) -> dict[int, int]:
        now = time.time() if now is None else now
        expires_at = math.ceil(now + self._ttl_s)
        deltas: dict[int, int] = {}
        for key, nbytes in leased:
            old = self._leases.get(key)
            if old is not None and old[0] > now:
                deltas[old[0]] = deltas.get(old[0], 0) - old[1]
            self._leases[key] = (expires_at, nbytes)
            deltas[expires_at] = deltas.get(expires_at, 0) + nbytes
        if now >= self._next_prune:
            self._leases = {k: v for k, v in self._leases.items() if v[0] > now}
            self._next_prune = now + self._ttl_s
        return deltas


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
            if not records or operation in (LEASES_KEY, WRITES_KEY):
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

    def record_writes(self, nbytes: int) -> None:
        """Record bytes entering (+) or leaving (-) a put in flight."""
        self.data.setdefault(WRITES_KEY, []).append({"bytes": nbytes})

    def record_leases(self, deltas: dict[int, int]) -> None:
        """Record leased bytes gained (or moved away) per expiry second."""
        records = self.data.setdefault(LEASES_KEY, [])
        for expires_at, nbytes in deltas.items():
            if nbytes:
                records.append({"expires_at": expires_at, "bytes": nbytes})


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
            "reclaim now: puts in flight, data under a read lease, and "
            "displaced replicas awaiting deferred release.",
            "write_usage": "Fraction of Mooncake store memory held by puts in flight.",
            "read_usage": "Fraction of Mooncake store memory under a read lease "
            "this vLLM instance took, estimated from its reads and existence hits.",
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
        # Built only for a configured connector, so the config is present.
        assert self._kv_transfer_config is not None
        # e.g. "http://<master-host>:9003/metrics"; unset disables the
        # store usage gauges.
        master_metrics_url = self._kv_transfer_config.kv_connector_extra_config.get(
            "mooncake_master_metrics_url"
        )
        # Leased bytes per expiry second, from the workers' lease deltas.
        self._lease_lock = threading.Lock()
        self._leased_bytes_by_expiry: dict[int, int] | None = None
        self._writing_bytes = 0
        if master_metrics_url:
            self._leased_bytes_by_expiry = {}
            threading.Thread(
                target=self._poll_master,
                args=(master_metrics_url,),
                name="mooncake-master-usage",
                daemon=True,
            ).start()

    def _leased_bytes(self) -> int:
        assert self._leased_bytes_by_expiry is not None
        now = time.time()
        with self._lease_lock:
            buckets = self._leased_bytes_by_expiry
            for expires_at in [t for t in buckets if t <= now]:
                del buckets[expires_at]
            return sum(buckets.values())

    def _poll_master(self, url: str) -> None:
        while True:
            snapshot = poll_master_metrics(url)
            if snapshot is not None:
                fractions = master_usage_fractions(
                    snapshot, self._leased_bytes(), self._writing_bytes
                )
                # One store backs every engine, so every engine label gets it.
                for name, value in fractions.items():
                    for labelvalues in self.per_engine_labelvalues.values():
                        self._usage_gauges[name].labels(*labelvalues).set(value)
            time.sleep(1.0)

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
            if operation == LEASES_KEY:
                if self._leased_bytes_by_expiry is not None:
                    with self._lease_lock:
                        buckets = self._leased_bytes_by_expiry
                        for record in records:
                            expires_at = int(record["expires_at"])
                            buckets[expires_at] = (
                                buckets.get(expires_at, 0) + record["bytes"]
                            )
                continue
            if operation == WRITES_KEY:
                if self._leased_bytes_by_expiry is not None:
                    with self._lease_lock:
                        self._writing_bytes += sum(r["bytes"] for r in records)
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
