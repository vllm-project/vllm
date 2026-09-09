# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    PromMetric,
    PromMetricT,
    TypedKVConnectorPromMetrics,
    TypedKVConnectorStats,
)
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
)
from vllm.v1.kv_offload.factory import OffloadingSpecFactory


class _TransferMetricName:
    """Flat metric names for GPU↔offload-medium transfer operations."""

    LOAD_BYTES = "vllm:kv_offload_load_bytes"
    LOAD_TIME = "vllm:kv_offload_load_time"
    LOAD_SIZE = "vllm:kv_offload_load_size"
    STORE_BYTES = "vllm:kv_offload_store_bytes"
    STORE_TIME = "vllm:kv_offload_store_time"
    STORE_SIZE = "vllm:kv_offload_store_size"


class _ConnectorMetricName:
    """Connector-side metrics emitted by scheduler-side offloading code."""

    LOOKUP_SYNC_DELAY = "vllm:kv_offload_lookup_sync_delay_seconds"
    LOOKUP_ASYNC_DELAY = "vllm:kv_offload_lookup_async_delay_seconds"
    ALLOCATION_FAILURE = "vllm:kv_offload_allocation_failure"


class _TransferType:
    """Transfer direction labels for deprecated CPU offload metrics."""

    LOAD = "CPU_to_GPU"
    STORE = "GPU_to_CPU"
    ALL = (LOAD, STORE)


TRANSFER_SIZE_BUCKETS = (
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


def get_connector_metric_definitions() -> dict[str, OffloadingMetricMetadata]:
    return {
        _TransferMetricName.LOAD_BYTES: OffloadingCounterMetadata(
            documentation="Total bytes loaded from offload storage to GPU.",
        ),
        _TransferMetricName.LOAD_TIME: OffloadingCounterMetadata(
            documentation="Total load time from offload storage to GPU, in seconds.",
        ),
        _TransferMetricName.LOAD_SIZE: OffloadingHistogramMetadata(
            documentation="Histogram of KV offload load operation size, in bytes.",
            buckets=TRANSFER_SIZE_BUCKETS,
        ),
        _TransferMetricName.STORE_BYTES: OffloadingCounterMetadata(
            documentation="Total bytes stored from GPU to offload storage.",
        ),
        _TransferMetricName.STORE_TIME: OffloadingCounterMetadata(
            documentation="Total store time from GPU to offload storage, in seconds.",
        ),
        _TransferMetricName.STORE_SIZE: OffloadingHistogramMetadata(
            documentation="Histogram of KV offload store operation size, in bytes.",
            buckets=TRANSFER_SIZE_BUCKETS,
        ),
        _ConnectorMetricName.LOOKUP_SYNC_DELAY: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of the time spent in a single offload lookup call, "
                "in seconds."
            ),
            buckets=(
                0.00001,
                0.00005,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1,
            ),
        ),
        _ConnectorMetricName.LOOKUP_ASYNC_DELAY: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of time between a request's offload lookup first "
                "deferring and the following lookup resolving, or request "
                "finish, in seconds."
            ),
            buckets=(
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1,
                5,
                10,
            ),
        ),
        _ConnectorMetricName.ALLOCATION_FAILURE: OffloadingCounterMetadata(
            documentation=(
                "Number of KV offload store allocation attempts that failed."
            ),
        ),
    }


_DEPRECATED_TOTAL_BYTES = "vllm:kv_offload_total_bytes"
_DEPRECATED_TOTAL_TIME = "vllm:kv_offload_total_time"
_DEPRECATED_SIZE = "vllm:kv_offload_size"

# Deprecated legacy transfer metrics, kept during the migration to the flat
# metric names above. These stay in a separate definition block because they
# use a transfer_type label, but are emitted from the same flat stats payload
# for compatibility.
_DEPRECATED_CONNECTOR_METRIC_DEFINITIONS: dict[str, OffloadingMetricMetadata] = {
    _DEPRECATED_TOTAL_BYTES: OffloadingCounterMetadata(
        documentation="Number of bytes offloaded by KV connector",
    ),
    _DEPRECATED_TOTAL_TIME: OffloadingCounterMetadata(
        documentation="Total time measured by all KV offloading operations",
    ),
    _DEPRECATED_SIZE: OffloadingHistogramMetadata(
        documentation="Histogram of KV offload transfer size, in bytes.",
        buckets=TRANSFER_SIZE_BUCKETS,
    ),
}


# Offloading stats are plain typed connector stats; the name is kept for the
# offloading managers and specs that record into them.
OffloadingConnectorStats = TypedKVConnectorStats


class OffloadPromMetrics(TypedKVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        extra_config = kv_transfer_config.kv_connector_extra_config
        spec_cls = OffloadingSpecFactory.get_spec_cls(extra_config)
        super().__init__(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
            {
                **spec_cls.build_metric_definitions(extra_config),
                **get_connector_metric_definitions(),
            },
        )
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

        self._observe_deprecated_metrics = issubclass(spec_cls, CPUOffloadingSpec)
        # (engine_idx, transfer_type) -> (metric with bounded labels)
        self.histogram_transfer_size: dict[tuple[int, str], PromMetricT] = {}
        self.counter_kv_bytes: dict[tuple[int, str], PromMetricT] = {}
        self.counter_kv_transfer_time: dict[tuple[int, str], PromMetricT] = {}

        self._counter_kv_bytes = self._counter_cls(
            name=_DEPRECATED_TOTAL_BYTES,
            documentation=_DEPRECATED_CONNECTOR_METRIC_DEFINITIONS[
                _DEPRECATED_TOTAL_BYTES
            ].documentation,
            labelnames=labelnames + ["transfer_type"],
        )

        self._counter_kv_transfer_time = self._counter_cls(
            name=_DEPRECATED_TOTAL_TIME,
            documentation=_DEPRECATED_CONNECTOR_METRIC_DEFINITIONS[
                _DEPRECATED_TOTAL_TIME
            ].documentation,
            labelnames=labelnames + ["transfer_type"],
        )

        deprecated_size_metadata = _DEPRECATED_CONNECTOR_METRIC_DEFINITIONS[
            _DEPRECATED_SIZE
        ]
        assert isinstance(deprecated_size_metadata, OffloadingHistogramMetadata)
        self._histogram_transfer_size = self._histogram_cls(
            name=_DEPRECATED_SIZE,
            documentation=deprecated_size_metadata.documentation,
            buckets=deprecated_size_metadata.buckets,
            labelnames=labelnames + ["transfer_type"],
        )

        for engine_idx, labelvalues in per_engine_labelvalues.items():
            for transfer_type in _TransferType.ALL:
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

    def _increase_counter(
        self,
        metric_name: str,
        value: int | float,
        labelvalues: tuple[str, ...],
        engine_idx: int,
    ) -> None:
        super()._increase_counter(metric_name, value, labelvalues, engine_idx)
        if labelvalues or not self._observe_deprecated_metrics:
            return
        # Keep deprecated CPU offload transfer metrics updated during the
        # transition to flat metric names.
        if metric_name == _TransferMetricName.LOAD_BYTES:
            self.counter_kv_bytes[(engine_idx, _TransferType.LOAD)].inc(value)
        elif metric_name == _TransferMetricName.LOAD_TIME:
            self.counter_kv_transfer_time[(engine_idx, _TransferType.LOAD)].inc(value)
        elif metric_name == _TransferMetricName.STORE_BYTES:
            self.counter_kv_bytes[(engine_idx, _TransferType.STORE)].inc(value)
        elif metric_name == _TransferMetricName.STORE_TIME:
            self.counter_kv_transfer_time[(engine_idx, _TransferType.STORE)].inc(value)

    def _observe_histogram(
        self,
        metric_name: str,
        value: list[int | float],
        labelvalues: tuple[str, ...],
        engine_idx: int,
    ) -> None:
        super()._observe_histogram(metric_name, value, labelvalues, engine_idx)
        if labelvalues or not self._observe_deprecated_metrics:
            return
        # Keep deprecated CPU offload transfer metrics updated during the
        # transition to flat metric names.
        if metric_name == _TransferMetricName.LOAD_SIZE:
            deprecated = self.histogram_transfer_size[(engine_idx, _TransferType.LOAD)]
        elif metric_name == _TransferMetricName.STORE_SIZE:
            deprecated = self.histogram_transfer_size[(engine_idx, _TransferType.STORE)]
        else:
            return
        for observation in value:
            deprecated.observe(observation)
