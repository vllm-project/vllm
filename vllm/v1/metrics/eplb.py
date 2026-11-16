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
                "Tokens routed to each physical expert over the last EPLB "
                "sampling period, split by batch phase (prefill, decode, or "
                "mixed). Each source EP rank reports independently; sum across "
                "engines for global load."
            ),
            labelnames=[*expert_labelnames, "logical_expert", "phase"],
        )
        self.counter_tokens: dict[tuple[int, str, int, int, int, int, str], object] = {}

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

        model_labelnames = [*labelnames, "model"]
        counter_rearrangements = self._counter_cls(
            name="vllm:eplb_rearrangements_total",
            documentation="Completed EPLB rearrangements.",
            labelnames=model_labelnames,
        )
        gauge_rearrangement_timestamp = self._gauge_cls(
            name="vllm:eplb_rearrangement_timestamp_seconds",
            documentation=(
                "Unix timestamp of the last completed EPLB rearrangement and "
                "expert-load sample."
            ),
            multiprocess_mode="mostrecent",
            labelnames=model_labelnames,
        )
        gauge_topology_timestamp = self._gauge_cls(
            name="vllm:eplb_expert_topology_timestamp_seconds",
            documentation=(
                "Unix timestamp of the last committed physical-expert topology update."
            ),
            multiprocess_mode="mostrecent",
            labelnames=model_labelnames,
        )
        load_labelnames = [*labelnames, "model", "phase"]
        gauge_load_timestamp = self._gauge_cls(
            name="vllm:eplb_expert_load_timestamp_seconds",
            documentation="Unix timestamp of the last per-expert load sample.",
            multiprocess_mode="mostrecent",
            labelnames=load_labelnames,
        )
        gauge_window_steps = self._gauge_cls(
            name="vllm:eplb_expert_load_window_steps",
            documentation=(
                "Number of engine steps covered by the last per-expert load sample."
            ),
            multiprocess_mode="mostrecent",
            labelnames=load_labelnames,
        )
        self.counter_rearrangements: dict[tuple[int, str], object] = {}
        self.gauge_rearrangement_timestamp: dict[tuple[int, str], object] = {}
        self.gauge_topology_timestamp: dict[tuple[int, str], object] = {}
        self.gauge_load_timestamp: dict[tuple[int, str, str], object] = {}
        self.gauge_window_steps: dict[tuple[int, str, str], object] = {}
        self._counter_rearrangements_metric = counter_rearrangements
        self._gauge_rearrangement_timestamp_metric = gauge_rearrangement_timestamp
        self._gauge_topology_timestamp_metric = gauge_topology_timestamp
        self._gauge_load_timestamp_metric = gauge_load_timestamp
        self._gauge_window_steps_metric = gauge_window_steps

    def _labels(
        self,
        engine_idx: int,
        model_name: str,
        layer: int,
        ep_rank: int,
        physical_expert: int,
        logical_expert: int | None = None,
        phase: str | None = None,
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
        if phase is not None:
            labels.append(phase)
        return labels

    def _model_labels(self, engine_idx: int, model_name: str) -> list[object]:
        return [*self.per_engine_labelvalues[engine_idx], model_name]

    def _load_metric(
        self,
        cache: dict[tuple[int, str, str], object],
        metric,
        engine_idx: int,
        model_name: str,
        phase: str,
    ):
        key = (engine_idx, model_name, phase)
        child = cache.get(key)
        if child is None:
            child = metric.labels(
                *self.per_engine_labelvalues[engine_idx], model_name, phase
            )
            cache[key] = child
        return child

    def _model_metric(
        self,
        cache: dict[tuple[int, str], object],
        metric,
        engine_idx: int,
        model_name: str,
    ):
        key = (engine_idx, model_name)
        child = cache.get(key)
        if child is None:
            child = metric.labels(*self._model_labels(engine_idx, model_name))
            cache[key] = child
        return child

    def observe(self, snapshot: EplbMetricsSnapshot, engine_idx: int = 0):
        if not self.enabled:
            return

        rearranged_models: dict[str, float] = {}
        for model in snapshot.tokens:
            if model.rearranged:
                rearranged_models[model.model_name] = max(
                    rearranged_models.get(model.model_name, 0.0), model.timestamp
                )
            if model.timestamp > 0:
                self._load_metric(
                    self.gauge_load_timestamp,
                    self._gauge_load_timestamp_metric,
                    engine_idx,
                    model.model_name,
                    model.phase,
                ).set(model.timestamp)
            if model.steps > 0:
                self._load_metric(
                    self.gauge_window_steps,
                    self._gauge_window_steps_metric,
                    engine_idx,
                    model.model_name,
                    model.phase,
                ).set(model.steps)
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
                        model.phase,
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
                                model.phase,
                            )
                        )
                        self.counter_tokens[key] = metric
                    metric.inc(num_tokens)

        for model_name, timestamp in rearranged_models.items():
            self._model_metric(
                self.counter_rearrangements,
                self._counter_rearrangements_metric,
                engine_idx,
                model_name,
            ).inc()
            if timestamp > 0:
                self._model_metric(
                    self.gauge_rearrangement_timestamp,
                    self._gauge_rearrangement_timestamp_metric,
                    engine_idx,
                    model_name,
                ).set(timestamp)

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
            if model.timestamp > 0:
                self._model_metric(
                    self.gauge_topology_timestamp,
                    self._gauge_topology_timestamp_metric,
                    engine_idx,
                    model.model_name,
                ).set(model.timestamp)
