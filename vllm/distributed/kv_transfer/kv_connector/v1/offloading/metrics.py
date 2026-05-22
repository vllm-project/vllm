# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
)
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.worker.worker import TransferType

logger = init_logger(__name__)

LOAD_BYTES = "load_bytes"
LOAD_TIME = "load_time"
LOAD_SIZE = "load_size"
STORE_BYTES = "store_bytes"
STORE_TIME = "store_time"
STORE_SIZE = "store_size"
STORES_SKIPPED = "stores_skipped"

_LOAD_TRANSFER_TYPE = "CPU_to_GPU"
_STORE_TRANSFER_TYPE = "GPU_to_CPU"
_TRANSFER_TYPES = (_LOAD_TRANSFER_TYPE, _STORE_TRANSFER_TYPE)

_TRANSFER_SIZE_BUCKETS = (
    1e6,
    5e6,
    10e6,
    20e6,
    40e6,
    60e6,
    80e6,
    100e6,
    150e6,
    200e6,
)

_CONNECTOR_METRIC_DEFINITIONS: dict[str, OffloadingMetricMetadata] = {
    LOAD_BYTES: OffloadingCounterMetadata(
        name="vllm:kv_offload_load_bytes",
        documentation="Total bytes loaded from offload storage to GPU.",
    ),
    LOAD_TIME: OffloadingCounterMetadata(
        name="vllm:kv_offload_load_time",
        documentation="Total load time from offload storage to GPU, in seconds.",
    ),
    LOAD_SIZE: OffloadingHistogramMetadata(
        name="vllm:kv_offload_load_size",
        documentation="Histogram of KV offload load operation size, in bytes.",
        buckets=_TRANSFER_SIZE_BUCKETS,
    ),
    STORE_BYTES: OffloadingCounterMetadata(
        name="vllm:kv_offload_store_bytes",
        documentation="Total bytes stored from GPU to offload storage.",
    ),
    STORE_TIME: OffloadingCounterMetadata(
        name="vllm:kv_offload_store_time",
        documentation="Total store time from GPU to offload storage, in seconds.",
    ),
    STORE_SIZE: OffloadingHistogramMetadata(
        name="vllm:kv_offload_store_size",
        documentation="Histogram of KV offload store operation size, in bytes.",
        buckets=_TRANSFER_SIZE_BUCKETS,
    ),
}

# Deprecated legacy transfer metrics, kept during the migration to the flat
# metric names above. These stay in a separate definition block because they
# use a transfer_type label, but are emitted from the same flat stats payload
# for compatibility.
_DEPRECATED_CONNECTOR_METRIC_DEFINITIONS: dict[str, OffloadingMetricMetadata] = {
    "total_bytes": OffloadingCounterMetadata(
        name="vllm:kv_offload_total_bytes",
        documentation="Number of bytes offloaded by KV connector",
    ),
    "total_time": OffloadingCounterMetadata(
        name="vllm:kv_offload_total_time",
        documentation="Total time measured by all KV offloading operations",
    ),
    "size": OffloadingHistogramMetadata(
        name="vllm:kv_offload_size",
        documentation="Histogram of KV offload transfer size, in bytes.",
        buckets=_TRANSFER_SIZE_BUCKETS,
    ),
}

_DEFAULT_MANAGER_METRIC_DEFINITIONS: dict[str, OffloadingMetricMetadata] = {
    STORES_SKIPPED: OffloadingCounterMetadata(
        name="vllm:kv_offload_stores_skipped",
        documentation=(
            "Number of KV offload stores skipped because the reuse threshold "
            "was not reached."
        ),
    )
}


def _default_metric_metadata() -> dict[str, OffloadingMetricMetadata]:
    return {
        **_CONNECTOR_METRIC_DEFINITIONS,
        **_DEFAULT_MANAGER_METRIC_DEFINITIONS,
    }


@dataclass
class OffloadingConnectorStats(KVConnectorStats):
    """
    Offloading connector stats use flat metric names as keys.

    Counter values are aggregated by summing, gauge values use the latest
    snapshot, and histogram values are lists of observed samples.
    """

    metric_metadata: dict[str, OffloadingMetricMetadata] = field(
        default_factory=_default_metric_metadata
    )

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        self.data: dict[str, Any] = {}

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        if not other.is_empty():
            for key, value in other.data.items():
                metadata = self.metric_metadata.get(key)
                if metadata is None:
                    raise AssertionError(f"Unknown offloading stats key: {key}")
                if isinstance(metadata, OffloadingHistogramMetadata):
                    assert isinstance(value, list)
                    if key not in self.data:
                        self.data[key] = value
                    else:
                        assert isinstance(self.data[key], list)
                        self.data[key].extend(value)
                elif isinstance(metadata, OffloadingCounterMetadata):
                    assert isinstance(value, int | float)
                    self.data[key] = self.data.get(key, 0) + value
                elif isinstance(metadata, OffloadingGaugeMetadata):
                    assert isinstance(value, int | float)
                    self.data[key] = value
                else:
                    raise AssertionError(f"Unknown offloading metric metadata: {key}")
        return self

    def reduce(self) -> dict[str, int | float]:
        """
        Reduce the observations collected during a time interval to one or
        more representative values (eg avg/median/sum of the series).
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.
        """
        return_dict: dict[str, int | float] = {}
        for key, value in self.data.items():
            metadata = self.metric_metadata.get(key)
            if metadata is None:
                raise AssertionError(f"Unknown offloading stats key: {key}")
            if isinstance(metadata, OffloadingHistogramMetadata):
                assert isinstance(value, list)
                return_dict[f"{key}_count"] = len(value)
                return_dict[f"{key}_sum"] = sum(value)
            elif isinstance(
                metadata, OffloadingCounterMetadata | OffloadingGaugeMetadata
            ):
                assert isinstance(value, int | float)
                return_dict[key] = value
            else:
                raise AssertionError(f"Unknown offloading metric metadata: {key}")
        return return_dict

    def is_empty(self) -> bool:
        return not self.data

    def record_transfer(self, num_bytes: int, time: float, transfer_type: TransferType):
        src, dst = transfer_type
        if (src, dst) == ("CPU", "GPU"):
            bytes_key, time_key, size_key = LOAD_BYTES, LOAD_TIME, LOAD_SIZE
        elif (src, dst) == ("GPU", "CPU"):
            bytes_key, time_key, size_key = STORE_BYTES, STORE_TIME, STORE_SIZE
        else:
            raise AssertionError(f"Unknown offloading transfer type: {transfer_type}")
        self.set_counter(bytes_key, num_bytes)
        self.set_counter(time_key, time)
        self.observe_histogram(size_key, num_bytes)

    def set_counter(self, counter_name: str, counter_value: int | float) -> None:
        """Set a counter increment on the stats payload."""
        self.data[counter_name] = self.data.get(counter_name, 0) + counter_value

    def set_gauge(self, gauge_name: str, gauge_value: int | float) -> None:
        """Set a gauge snapshot on the stats payload."""
        self.data[gauge_name] = gauge_value

    def observe_histogram(
        self, histogram_name: str, histogram_value: int | float
    ) -> None:
        """Record a histogram observation on the stats payload."""
        self.data.setdefault(histogram_name, []).append(histogram_value)


class OffloadPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        # (engine_idx, transfer_type) -> (metric with bounded labels)
        self.histogram_transfer_size: dict[tuple[int, str], PromMetricT] = {}
        self.counter_kv_bytes: dict[tuple[int, str], PromMetricT] = {}
        self.counter_kv_transfer_time: dict[tuple[int, str], PromMetricT] = {}
        manager_metric_metadata = OffloadingSpecFactory.get_metric_definitions(
            vllm_config
        )
        self._offloading_metric_metadata: dict[str, OffloadingMetricMetadata] = {
            **_CONNECTOR_METRIC_DEFINITIONS,
            **manager_metric_metadata,
        }
        self._offloading_metric_defs: dict[str, PromMetricT] = {}
        self.offloading_metrics: dict[tuple[int, str], PromMetricT] = {}
        self._metric_observers: dict[
            str, Callable[[int | float | list[int | float], int], None]
        ] = {}

        self._counter_kv_bytes = self._counter_cls(
            name=_DEPRECATED_CONNECTOR_METRIC_DEFINITIONS["total_bytes"].name,
            documentation=_DEPRECATED_CONNECTOR_METRIC_DEFINITIONS[
                "total_bytes"
            ].documentation,
            labelnames=labelnames + ["transfer_type"],
        )

        self._counter_kv_transfer_time = self._counter_cls(
            name=_DEPRECATED_CONNECTOR_METRIC_DEFINITIONS["total_time"].name,
            documentation=_DEPRECATED_CONNECTOR_METRIC_DEFINITIONS[
                "total_time"
            ].documentation,
            labelnames=labelnames + ["transfer_type"],
        )

        deprecated_size_metadata = _DEPRECATED_CONNECTOR_METRIC_DEFINITIONS["size"]
        assert isinstance(deprecated_size_metadata, OffloadingHistogramMetadata)
        self._histogram_transfer_size = self._histogram_cls(
            name=deprecated_size_metadata.name,
            documentation=deprecated_size_metadata.documentation,
            buckets=deprecated_size_metadata.buckets,
            labelnames=labelnames + ["transfer_type"],
        )

        for engine_idx, labelvalues in per_engine_labelvalues.items():
            for transfer_type in _TRANSFER_TYPES:
                bounded_labelvalues = labelvalues + [transfer_type]
                self.histogram_transfer_size[(engine_idx, transfer_type)] = (
                    self._histogram_transfer_size.labels(*bounded_labelvalues)
                )
                self.counter_kv_bytes[(engine_idx, transfer_type)] = (
                    self._counter_kv_bytes.labels(*bounded_labelvalues)
                )
                self.counter_kv_transfer_time[(engine_idx, transfer_type)] = (
                    self._counter_kv_transfer_time.labels(*bounded_labelvalues)
                )

        for metric_name, metadata in self._offloading_metric_metadata.items():
            self._offloading_metric_defs[metric_name] = self._create_metric(metadata)
            for engine_idx, labelvalues in per_engine_labelvalues.items():
                self.offloading_metrics[(engine_idx, metric_name)] = (
                    self._offloading_metric_defs[metric_name].labels(*labelvalues)
                )
            self._metric_observers[metric_name] = self._make_observer(
                metric_name, metadata
            )

    def _create_metric(self, metadata: OffloadingMetricMetadata) -> PromMetricT:
        kwargs: dict[str, Any] = {
            "name": metadata.name,
            "documentation": metadata.documentation,
            "labelnames": self._labelnames,
        }
        if isinstance(metadata, OffloadingCounterMetadata):
            metric_cls = self._counter_cls
        elif isinstance(metadata, OffloadingGaugeMetadata):
            metric_cls = self._gauge_cls
        elif isinstance(metadata, OffloadingHistogramMetadata):
            metric_cls = self._histogram_cls
            if metadata.buckets is not None:
                kwargs["buckets"] = metadata.buckets
        else:
            raise AssertionError(f"Unknown offloading metric metadata: {metadata}")
        return metric_cls(**kwargs)

    def _make_observer(
        self, metric_name: str, metadata: OffloadingMetricMetadata
    ) -> Callable[[int | float | list[int | float], int], None]:
        if isinstance(metadata, OffloadingCounterMetadata):
            return lambda value, engine_idx: self._observe_counter(
                metric_name, value, engine_idx
            )
        if isinstance(metadata, OffloadingGaugeMetadata):
            return lambda value, engine_idx: self._observe_gauge(
                metric_name, value, engine_idx
            )
        if isinstance(metadata, OffloadingHistogramMetadata):
            return lambda value, engine_idx: self._observe_histogram(
                metric_name, value, engine_idx
            )
        raise AssertionError(f"Unknown offloading metric metadata: {metadata}")

    def _observe_counter(
        self, metric_name: str, value: int | float | list[int | float], engine_idx: int
    ) -> None:
        assert isinstance(value, int | float)
        self.offloading_metrics[(engine_idx, metric_name)].inc(value)
        if metric_name == LOAD_BYTES:
            self.counter_kv_bytes[(engine_idx, _LOAD_TRANSFER_TYPE)].inc(value)
        elif metric_name == LOAD_TIME:
            self.counter_kv_transfer_time[(engine_idx, _LOAD_TRANSFER_TYPE)].inc(value)
        elif metric_name == STORE_BYTES:
            self.counter_kv_bytes[(engine_idx, _STORE_TRANSFER_TYPE)].inc(value)
        elif metric_name == STORE_TIME:
            self.counter_kv_transfer_time[(engine_idx, _STORE_TRANSFER_TYPE)].inc(value)

    def _observe_gauge(
        self, metric_name: str, value: int | float | list[int | float], engine_idx: int
    ) -> None:
        assert isinstance(value, int | float)
        self.offloading_metrics[(engine_idx, metric_name)].set(value)

    def _observe_histogram(
        self, metric_name: str, value: int | float | list[int | float], engine_idx: int
    ) -> None:
        assert isinstance(value, list)
        for observation in value:
            assert isinstance(observation, int | float)
            self.offloading_metrics[(engine_idx, metric_name)].observe(observation)
            if metric_name == LOAD_SIZE:
                self.histogram_transfer_size[
                    (engine_idx, _LOAD_TRANSFER_TYPE)
                ].observe(observation)
            elif metric_name == STORE_SIZE:
                self.histogram_transfer_size[
                    (engine_idx, _STORE_TRANSFER_TYPE)
                ].observe(observation)

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Observe transfer statistics."""
        for key, value in transfer_stats_data.items():
            self._metric_observers[key](value, engine_idx)
