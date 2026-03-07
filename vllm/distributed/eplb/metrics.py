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
    """Record EPLB load metrics in Prometheus.

    Each EP rank independently reports how many tokens it routed to each
    destination EP rank, per MoE layer.  These gauges are directly summable
    across DP ranks in Prometheus/Grafana — no inter-rank synchronization:

        # Global load per EP rank per layer
        sum by (layer_idx, dst_ep_rank) (vllm:eplb_tokens_routed_to_ep_rank)

        # Imbalance ratio per layer
        max by (layer_idx) (sum by (...) (...))
        / avg by (layer_idx) (sum by (...) (...))

    The ``layer_idx`` and ``dst_ep_rank`` labels are added on top of the
    standard per-engine labels (model_name, etc.).
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

        # Per-layer, per-destination-rank token count gauge.
        # Extra labels: layer_idx, dst_ep_rank
        extended_labels = [*labelnames, "layer_idx", "dst_ep_rank"]
        self._tokens_gauge = self._gauge_cls(
            name="vllm:eplb_tokens_routed_to_ep_rank",
            documentation=(
                "Tokens routed to each destination EP rank per MoE layer "
                "(from this DP rank's perspective). Summable across DP ranks."
            ),
            multiprocess_mode="mostrecent",
            labelnames=extended_labels,
        )
        self._per_engine_labelvalues = per_engine_labelvalues

        # Rearrangement counter and timing (shared labels, no extra dims)
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

        # Cache of labeled gauge children keyed by
        # (engine_idx, layer_idx, dst_ep_rank) to avoid repeated .labels()
        self._tokens_children: dict[
            tuple[int, int, int],
            prometheus_client.Gauge,
        ] = {}

    def _get_tokens_child(
        self, engine_idx: int, layer_idx: int, dst_ep_rank: int
    ) -> prometheus_client.Gauge:
        key = (engine_idx, layer_idx, dst_ep_rank)
        child = self._tokens_children.get(key)
        if child is None:
            base_labels = self._per_engine_labelvalues[engine_idx]
            child = self._tokens_gauge.labels(
                *base_labels, str(layer_idx), str(dst_ep_rank)
            )
            self._tokens_children[key] = child
        return child

    def observe(self, eplb_stats: EplbMetricsStats, engine_idx: int = 0):
        if not self.enabled:
            return

        # Set per-layer, per-EP-rank token counts
        for layer_idx, rank_counts in enumerate(eplb_stats.tokens_per_ep_rank):
            for dst_rank, count in enumerate(rank_counts):
                self._get_tokens_child(engine_idx, layer_idx, dst_rank).set(count)

        # Rearrangement metrics
        if eplb_stats.rearrangements > 0:
            self.counter_rearrangements[engine_idx].inc(eplb_stats.rearrangements)
            self.gauge_rearrangement_seconds[engine_idx].set(
                eplb_stats.last_rearrangement_seconds
            )
