# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from prometheus_client import Histogram

from vllm.v1.metrics.stats import WorkerTimingStats

WORKER_TIME_BUCKETS = [
    0.0010,  # 1ms,   1000TPS
    0.0015,  # 1.5ms, 666TPS
    0.002,  # 2ms,   500TPS
    0.005,  # 5ms,   200TPS
    0.010,  # 10ms,  100TPS
    0.020,  # 20ms,  50TPS
    0.050,  # 50ms,  20TPS
    0.1,  # 100ms, 10TPS --- Above: mostly decodes. Below: mostly prefills. ---
    0.2,  # 200ms
    0.5,  # 500ms
    1.0,  # 1s
    2.0,  # 2s
    5.0,  # 5s
    10.0,  # 10s
]


class WorkerTimingProm:
    _histogram_cls = Histogram

    def __init__(
        self,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        self.per_engine_labelvalues = per_engine_labelvalues
        self.model_time = self._histogram_cls(
            name="vllm:worker_model_execute_time_seconds",
            documentation=(
                "Device time spent on a model step, including sampling and "
                "excluding speculative proposal time. The num_model_tokens "
                "label is the target-model input size after padding."
            ),
            buckets=WORKER_TIME_BUCKETS,
            labelnames=labelnames + ["phase", "num_model_tokens"],
        )
        self.proposer_time = self._histogram_cls(
            name="vllm:worker_spec_decode_proposer_time_seconds",
            documentation="Device time spent generating speculative proposals.",
            buckets=WORKER_TIME_BUCKETS,
            labelnames=labelnames + ["phase", "num_requests"],
        )
        self.total_time = self._histogram_cls(
            name="vllm:worker_step_time_seconds",
            documentation=(
                "Total model-runner device time, including speculative proposals. "
                "The num_model_tokens label is the target-model input size after "
                "padding."
            ),
            buckets=WORKER_TIME_BUCKETS,
            labelnames=labelnames + ["phase", "num_model_tokens"],
        )

    def observe(self, samples: list[WorkerTimingStats], engine_idx: int) -> None:
        labels: list[Any] = self.per_engine_labelvalues[engine_idx]
        for stat in samples:
            model_labels = labels + [stat.phase, str(stat.num_model_tokens)]
            self.model_time.labels(*model_labels).observe(stat.model_time_seconds)
            self.total_time.labels(*model_labels).observe(stat.total_time_seconds)
            if stat.proposer_time_seconds is not None:
                proposer_labels = labels + [stat.phase, str(stat.num_requests)]
                self.proposer_time.labels(*proposer_labels).observe(
                    stat.proposer_time_seconds
                )
