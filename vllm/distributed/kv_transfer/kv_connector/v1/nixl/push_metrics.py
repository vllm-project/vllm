# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import PromMetric, PromMetricT
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import NixlPromMetrics


class NixlPushPromMetrics(NixlPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self.unmatched_registrations = self._gauge_cls(
            name="vllm:nixl_push_unmatched_registrations",
            documentation="Decode registrations waiting for finished prefill requests "
            "on this producer rank.",
            labelnames=[*labelnames, "rank"],
            multiprocess_mode="mostrecent",
        )
        self.unmatched_finished_blocks = self._gauge_cls(
            name="vllm:nixl_push_unmatched_finished_blocks",
            documentation="Finished prefill requests whose blocks are waiting for a "
            "decode registration on this producer rank. Counts requests, not KV pages.",
            labelnames=[*labelnames, "rank"],
            multiprocess_mode="mostrecent",
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        super().observe(transfer_stats_data, engine_idx)
        for rank, (registrations, finished) in transfer_stats_data.get(
            "push_backlog", {}
        ).items():
            labels = self.per_engine_labelvalues[engine_idx]
            self.unmatched_registrations.labels(*labels, rank).set(registrations)
            self.unmatched_finished_blocks.labels(*labels, rank).set(finished)
