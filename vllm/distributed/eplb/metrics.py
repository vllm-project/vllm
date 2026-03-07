# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import prometheus_client

from vllm.config import ParallelConfig
from vllm.v1.metrics.stats import EplbMetricsStats


def _make_per_engine(
    metric: prometheus_client.Gauge | prometheus_client.Counter,
    per_engine_labelvalues: dict[int, list[object]],
) -> dict[int, prometheus_client.Gauge | prometheus_client.Counter]:
    return {
        idx: metric.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }


class EplbProm:
    """Record EPLB balancedness metrics in Prometheus.

    Balancedness is defined per MoE layer as:
        avg(tokens_per_rank) / max(tokens_per_rank)

    A value of 1.0 means perfectly balanced load across EP ranks.

    Example PromQL queries:

      # Alert on worst-layer imbalance
      vllm:eplb_balancedness_min < 0.7

      # Median vs worst-layer gap
      vllm:eplb_balancedness_p50 - vllm:eplb_balancedness_min
    """

    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        parallel_config: ParallelConfig,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.enabled = parallel_config.enable_eplb
        if not self.enabled:
            return

        gauge_min = self._gauge_cls(
            name="vllm:eplb_balancedness_min",
            documentation=(
                "Minimum per-layer EPLB balancedness "
                "(worst MoE layer, avg/max token load across EP ranks)."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_balancedness_min = _make_per_engine(
            gauge_min, per_engine_labelvalues
        )

        gauge_p50 = self._gauge_cls(
            name="vllm:eplb_balancedness_p50",
            documentation=("Median per-layer EPLB balancedness across MoE layers."),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_balancedness_p50 = _make_per_engine(
            gauge_p50, per_engine_labelvalues
        )

        gauge_p90 = self._gauge_cls(
            name="vllm:eplb_balancedness_p90",
            documentation=(
                "10th-percentile EPLB balancedness "
                "(90%% of MoE layers are at least this balanced)."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_balancedness_p90 = _make_per_engine(
            gauge_p90, per_engine_labelvalues
        )

        gauge_avg = self._gauge_cls(
            name="vllm:eplb_balancedness_avg",
            documentation=("Mean EPLB balancedness across MoE layers."),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_balancedness_avg = _make_per_engine(
            gauge_avg, per_engine_labelvalues
        )

        counter_rearrangements = self._counter_cls(
            name="vllm:eplb_rearrangements_total",
            documentation="Total number of EPLB expert rearrangements.",
            labelnames=labelnames,
        )
        self.counter_rearrangements = _make_per_engine(
            counter_rearrangements, per_engine_labelvalues
        )

        gauge_rearrangement_seconds = self._gauge_cls(
            name="vllm:eplb_rearrangement_seconds",
            documentation=(
                "Duration of the most recent EPLB expert rearrangement in seconds."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_rearrangement_seconds = _make_per_engine(
            gauge_rearrangement_seconds, per_engine_labelvalues
        )

    def observe(self, eplb_stats: EplbMetricsStats, engine_idx: int = 0):
        if not self.enabled:
            return
        self.gauge_balancedness_min[engine_idx].set(eplb_stats.min_balancedness)
        self.gauge_balancedness_p50[engine_idx].set(eplb_stats.p50_balancedness)
        self.gauge_balancedness_p90[engine_idx].set(eplb_stats.p90_balancedness)
        self.gauge_balancedness_avg[engine_idx].set(eplb_stats.avg_balancedness)
        if eplb_stats.rearrangements > 0:
            self.counter_rearrangements[engine_idx].inc(eplb_stats.rearrangements)
            self.gauge_rearrangement_seconds[engine_idx].set(
                eplb_stats.last_rearrangement_seconds
            )
