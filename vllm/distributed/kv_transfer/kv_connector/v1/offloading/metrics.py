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
from vllm.v1.kv_offload.worker.worker import TransferType

logger = init_logger(__name__)


@dataclass
class OffloadingOperationMetrics:
    op_size: int
    op_time: float


@dataclass
class OffloadingConnectorStats(KVConnectorStats):
    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        self.data: dict[str, list[OffloadingOperationMetrics]] = {}

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                if k not in self.data:
                    self.data[k] = v
                else:
                    accumulator = self.data[k]
                    assert isinstance(accumulator, list)
                    accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        """
        Reduce the observations collected during a time interval to one or
        more representative values (eg avg/median/sum of the series).
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.
        """
        return_dict: dict[str, int | float] = {}
        for transfer_type, ops_list in self.data.items():
            assert isinstance(ops_list, list)
            total_bytes = 0
            total_time = 0.0
            for op in ops_list:
                assert isinstance(op, dict)
                total_bytes += op["op_size"]
                total_time += op["op_time"]
            return_dict[f"{transfer_type}_total_bytes"] = total_bytes
            return_dict[f"{transfer_type}_total_time"] = total_time
        return return_dict

    def is_empty(self) -> bool:
        return not self.data

    def record_transfer(self, num_bytes: int, time: float, transfer_type: TransferType):
        src, dst = transfer_type
        transfer_type_key = src + "_to_" + dst
        op = OffloadingOperationMetrics(num_bytes, time)
        if transfer_type_key in self.data:
            self.data[transfer_type_key].append(op)
        else:
            self.data[transfer_type_key] = [op]


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

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """
        Observe transfer statistics from the new data structure.
        transfer_stats_data is expected to be a dict where:
        - keys are transfer type strings (e.g., "cpu_to_gpu", "gpu_to_cpu")
        - values are lists of OffloadingOperationMetrics objects
        """

        for transfer_type, ops in transfer_stats_data.items():
            # Cache:
            if (engine_idx, transfer_type) not in self.histogram_transfer_size:
                self.histogram_transfer_size[(engine_idx, transfer_type)] = (
                    self._histogram_transfer_size.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [transfer_type])
                    )
                )
                self.counter_kv_bytes[(engine_idx, transfer_type)] = (
                    self._counter_kv_bytes.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [transfer_type])
                    )
                )
                self.counter_kv_transfer_time[(engine_idx, transfer_type)] = (
                    self._counter_kv_transfer_time.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [transfer_type])
                    )
                )

            # Process ops:
            assert isinstance(ops, list)
            for op in ops:  # ops is a list of serialized OffloadingOperationMetrics
                assert isinstance(op, dict)
                # Observe size histogram
                self.histogram_transfer_size[(engine_idx, transfer_type)].observe(
                    op["op_size"]
                )

                # Increment byte and time counters
                self.counter_kv_bytes[(engine_idx, transfer_type)].inc(op["op_size"])

                self.counter_kv_transfer_time[(engine_idx, transfer_type)].inc(
                    op["op_time"]
                )
