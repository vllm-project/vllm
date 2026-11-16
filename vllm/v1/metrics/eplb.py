# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import prometheus_client

from vllm.config import ParallelConfig
from vllm.distributed.eplb.metrics import EplbMetricsSnapshot


class EplbProm:
    """Export per-physical-expert EPLB topology and token load.

    Physical experts are assigned to EP ranks in contiguous blocks, so the
    owning rank is ``physical_expert // num_local_experts``. Token counts are
    per source EP rank; aggregate them across engines in PromQL:

        sum by (layer, physical_expert) (
            rate(vllm:eplb_expert_tokens_total[1m])
        )

    The topology gauge reports the logical expert currently held by each
    physical slot. It changes in place after a rearrangement, so no new time
    series are created when experts move. Token counters also carry the
    ``logical_expert`` that was active while the tokens were routed.
    """

    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        parallel_config: ParallelConfig,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.enabled = (
            parallel_config.enable_eplb
            and parallel_config.eplb_config.enable_expert_metrics
        )
        self.per_engine_labelvalues = per_engine_labelvalues
        if not self.enabled:
            return

        expert_labelnames = [
            *labelnames,
            "model",
            "layer",
            "ep_rank",
            "physical_expert",
        ]
        counter_tokens = self._counter_cls(
            name="vllm:eplb_expert_tokens_total",
            documentation=(
                "Tokens routed to each physical expert, summed over the last "
                "EPLB load window. Each source EP rank reports independently; "
                "sum across engines for global load."
            ),
            labelnames=[*expert_labelnames, "logical_expert"],
        )
        self.counter_tokens: dict[tuple[int, str, int, int, int, int], object] = {}

        gauge_topology = self._gauge_cls(
            name="vllm:eplb_expert_topology",
            documentation=(
                "Logical expert id held by each physical expert. Physical "
                "experts are partitioned across EP ranks; value -1 means the "
                "slot is empty."
            ),
            multiprocess_mode="mostrecent",
            labelnames=expert_labelnames,
        )
        self.gauge_topology: dict[tuple[int, str, int, int, int], object] = {}
        self._counter_metric = counter_tokens
        self._gauge_metric = gauge_topology

    def _labels(
        self,
        engine_idx: int,
        model_name: str,
        layer: int,
        ep_rank: int,
        physical_expert: int,
        logical_expert: int | None = None,
    ) -> list[object]:
        labels = [
            *self.per_engine_labelvalues[engine_idx],
            model_name,
            str(layer),
            str(ep_rank),
            str(physical_expert),
        ]
        if logical_expert is not None:
            labels.append(str(logical_expert))
        return labels

    def observe(self, snapshot: EplbMetricsSnapshot, engine_idx: int = 0):
        if not self.enabled:
            return

        for model in snapshot.tokens:
            layout = model.physical_to_logical
            for layer, token_counts in enumerate(model.tokens):
                for physical_expert, num_tokens in enumerate(token_counts):
                    if num_tokens <= 0:
                        continue
                    logical_expert = (
                        layout[layer][physical_expert] if layout is not None else -1
                    )
                    ep_rank = physical_expert // model.num_local_experts
                    key = (
                        engine_idx,
                        model.model_name,
                        layer,
                        ep_rank,
                        physical_expert,
                        logical_expert,
                    )
                    metric = self.counter_tokens.get(key)
                    if metric is None:
                        metric = self._counter_metric.labels(
                            *self._labels(
                                engine_idx,
                                model.model_name,
                                layer,
                                ep_rank,
                                physical_expert,
                                logical_expert,
                            )
                        )
                        self.counter_tokens[key] = metric
                    metric.inc(num_tokens)

        for model in snapshot.topology:
            for layer, logical_experts in enumerate(model.physical_to_logical):
                for physical_expert, logical_expert in enumerate(logical_experts):
                    ep_rank = physical_expert // model.num_local_experts
                    key = (
                        engine_idx,
                        model.model_name,
                        layer,
                        ep_rank,
                        physical_expert,
                    )
                    metric = self.gauge_topology.get(key)
                    if metric is None:
                        metric = self._gauge_metric.labels(
                            *self._labels(
                                engine_idx,
                                model.model_name,
                                layer,
                                ep_rank,
                                physical_expert,
                            )
                        )
                        self.gauge_topology[key] = metric
                    metric.set(logical_expert)
