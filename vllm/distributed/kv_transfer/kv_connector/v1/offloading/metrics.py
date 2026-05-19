# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
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

TRANSFER_PREFIX = "xfer:"
COUNTER_PREFIX = "counter:"
GAUGE_PREFIX = "gauge:"
HISTOGRAM_PREFIX = "histogram:"
_TRANSFER_TYPES = ("CPU_to_GPU", "GPU_to_CPU")


@dataclass
class OffloadingConnectorStats(KVConnectorStats):
    """
    Offloading connector stats encode the stat type in each key.

    * ``xfer:<transfer_type>`` maps to a list of serialized operation metrics.
    * ``counter:<counter_name>`` maps to an int or float increment.
    * ``gauge:<gauge_name>`` maps to the latest int or float value.
    * ``histogram:<histogram_name>`` maps to observed int or float samples.
    """

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        self.data: dict[str, Any] = {}

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        if not other.is_empty():
            for key, value in other.data.items():
                if key.startswith(TRANSFER_PREFIX):
                    assert isinstance(value, list)
                    if key not in self.data:
                        self.data[key] = value
                    else:
                        assert isinstance(self.data[key], list)
                        self.data[key].extend(value)
                elif key.startswith(COUNTER_PREFIX):
                    assert isinstance(value, int | float)
                    self.data[key] = self.data.get(key, 0) + value
                elif key.startswith(GAUGE_PREFIX):
                    assert isinstance(value, int | float)
                    self.data[key] = value
                elif key.startswith(HISTOGRAM_PREFIX):
                    assert isinstance(value, list)
                    if key not in self.data:
                        self.data[key] = value
                    else:
                        assert isinstance(self.data[key], list)
                        self.data[key].extend(value)
                else:
                    raise AssertionError(f"Unknown offloading stats key: {key}")
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
            if key.startswith(TRANSFER_PREFIX):
                transfer_type = key.removeprefix(TRANSFER_PREFIX)
                assert isinstance(value, list)
                total_bytes = 0
                total_time = 0.0
                for op in value:
                    assert isinstance(op, dict)
                    total_bytes += op["op_size"]
                    total_time += op["op_time"]
                return_dict[f"{transfer_type}_total_bytes"] = total_bytes
                return_dict[f"{transfer_type}_total_time"] = total_time
            elif key.startswith(COUNTER_PREFIX):
                assert isinstance(value, int | float)
                return_dict[key.removeprefix(COUNTER_PREFIX)] = value
            elif key.startswith(GAUGE_PREFIX):
                assert isinstance(value, int | float)
                return_dict[key.removeprefix(GAUGE_PREFIX)] = value
            elif key.startswith(HISTOGRAM_PREFIX):
                histogram_name = key.removeprefix(HISTOGRAM_PREFIX)
                assert isinstance(value, list)
                return_dict[f"{histogram_name}_count"] = len(value)
                return_dict[f"{histogram_name}_sum"] = sum(value)
            else:
                raise AssertionError(f"Unknown offloading stats key: {key}")
        return return_dict

    def is_empty(self) -> bool:
        return not self.data

    def record_transfer(self, num_bytes: int, time: float, transfer_type: TransferType):
        src, dst = transfer_type
        transfer_type_key = TRANSFER_PREFIX + src + "_to_" + dst
        op = {"op_size": num_bytes, "op_time": time}
        self.data.setdefault(transfer_type_key, []).append(op)

    def set_counter(self, counter_name: str, counter_value: int | float) -> None:
        """Set a counter increment on the stats payload."""
        self.data[COUNTER_PREFIX + counter_name] = counter_value

    def set_gauge(self, gauge_name: str, gauge_value: int | float) -> None:
        """Set a gauge snapshot on the stats payload."""
        self.data[GAUGE_PREFIX + gauge_name] = gauge_value

    def observe_histogram(
        self, histogram_name: str, histogram_value: int | float
    ) -> None:
        """Record a histogram observation on the stats payload."""
        self.data.setdefault(HISTOGRAM_PREFIX + histogram_name, []).append(
            histogram_value
        )


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
        self._offloading_manager_metric_metadata: dict[
            str, OffloadingMetricMetadata
        ] = OffloadingSpecFactory.get_metric_definitions(vllm_config)
        self._offloading_manager_metric_defs: dict[str, PromMetricT] = {}
        self.offloading_manager_metrics: dict[tuple[int, str], PromMetricT] = {}
        buckets = [  # In bytes
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
        ]

        self._counter_kv_bytes = self._counter_cls(
            name="vllm:kv_offload_total_bytes",
            documentation="Number of bytes offloaded by KV connector",
            labelnames=labelnames + ["transfer_type"],
        )

        self._counter_kv_transfer_time = self._counter_cls(
            name="vllm:kv_offload_total_time",
            documentation="Total time measured by all KV offloading operations",
            labelnames=labelnames + ["transfer_type"],
        )

        self._histogram_transfer_size = self._histogram_cls(
            name="vllm:kv_offload_size",
            documentation="Histogram of KV offload transfer size, in bytes.",
            buckets=buckets[:],
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

    def _ensure_offloading_manager_metric(
        self, metric_name: str, engine_idx: int
    ) -> PromMetricT:
        assert metric_name in self._offloading_manager_metric_metadata
        if metric_name not in self._offloading_manager_metric_defs:
            metadata = self._offloading_manager_metric_metadata[metric_name]
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
            self._offloading_manager_metric_defs[metric_name] = metric_cls(**kwargs)
        if (engine_idx, metric_name) not in self.offloading_manager_metrics:
            metric = self._offloading_manager_metric_defs[metric_name]
            self.offloading_manager_metrics[(engine_idx, metric_name)] = metric.labels(
                *self.per_engine_labelvalues[engine_idx]
            )
        return self.offloading_manager_metrics[(engine_idx, metric_name)]

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Observe transfer statistics."""
        for key, value in transfer_stats_data.items():
            if key.startswith(COUNTER_PREFIX):
                metric_name = key.removeprefix(COUNTER_PREFIX)
                assert isinstance(value, int | float)
                metadata = self._offloading_manager_metric_metadata[metric_name]
                assert isinstance(metadata, OffloadingCounterMetadata)
                counter = self._ensure_offloading_manager_metric(metric_name, engine_idx)
                counter.inc(value)
                continue
            if key.startswith(GAUGE_PREFIX):
                metric_name = key.removeprefix(GAUGE_PREFIX)
                assert isinstance(value, int | float)
                metadata = self._offloading_manager_metric_metadata[metric_name]
                assert isinstance(metadata, OffloadingGaugeMetadata)
                gauge = self._ensure_offloading_manager_metric(metric_name, engine_idx)
                gauge.set(value)
                continue
            if key.startswith(HISTOGRAM_PREFIX):
                metric_name = key.removeprefix(HISTOGRAM_PREFIX)
                assert isinstance(value, list)
                metadata = self._offloading_manager_metric_metadata[metric_name]
                assert isinstance(metadata, OffloadingHistogramMetadata)
                histogram = self._ensure_offloading_manager_metric(
                    metric_name, engine_idx
                )
                for observation in value:
                    assert isinstance(observation, int | float)
                    histogram.observe(observation)
                continue

            if not key.startswith(TRANSFER_PREFIX):
                raise AssertionError(f"Unknown offloading stats key: {key}")

            ops = value
            transfer_type = key.removeprefix(TRANSFER_PREFIX)
            transfer_size = self.histogram_transfer_size[(engine_idx, transfer_type)]
            kv_bytes = self.counter_kv_bytes[(engine_idx, transfer_type)]
            transfer_time = self.counter_kv_transfer_time[(engine_idx, transfer_type)]

            assert isinstance(ops, list)
            for op in ops:
                assert isinstance(op, dict)
                transfer_size.observe(op["op_size"])
                kv_bytes.inc(op["op_size"])
                transfer_time.inc(op["op_time"])
