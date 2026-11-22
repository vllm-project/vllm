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
    _histogram_cls = prometheus_client.Histogram

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

        stall_labelnames = [*labelnames, "model", "phase", "async"]
        histogram_stall = self._histogram_cls(
            name="vllm:eplb_rearrangement_pause_seconds",
            documentation=(
                "Blocking time spent rearranging experts before the serving "
                "loop resumes. Compare its tail with the latency SLO headroom."
            ),
            labelnames=stall_labelnames,
            buckets=(
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
                30.0,
            ),
        )
        counter_blocked_requests = self._counter_cls(
            name="vllm:eplb_rearrangement_blocked_request_seconds_total",
            documentation=(
                "Request-seconds spent blocked in EPLB rearrangements. The "
                "increase rate divided by served request-seconds is the "
                "fraction of request time exposed to rearrangement stalls."
            ),
            labelnames=stall_labelnames,
        )
        counter_blocked_tokens = self._counter_cls(
            name="vllm:eplb_rearrangement_blocked_token_seconds_total",
            documentation=(
                "Token-seconds spent blocked in EPLB rearrangements, using the "
                "scheduled tokens of the triggering batch as the weight."
            ),
            labelnames=stall_labelnames,
        )
        load_labelnames = [*labelnames, "model", "phase", "layer"]
        skew_buckets = (1.0, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 5.0)
        excess_buckets = (
            0.0,
            10.0,
            50.0,
            100.0,
            250.0,
            500.0,
            1000.0,
            2500.0,
            5000.0,
            10000.0,
            50000.0,
        )
        histogram_load_rank_skew = self._histogram_cls(
            name="vllm:eplb_load_rank_skew",
            documentation=(
                "Max/mean token load across EP ranks for one MoE layer. "
                "1.0 is perfectly balanced."
            ),
            labelnames=load_labelnames,
            buckets=skew_buckets,
        )
        histogram_critical_rank_excess = self._histogram_cls(
            name="vllm:eplb_critical_rank_excess_tokens",
            documentation=(
                "Tokens on the busiest EP rank above the rank mean for one MoE layer."
            ),
            labelnames=load_labelnames,
            buckets=excess_buckets,
        )
        histogram_load_expert_skew = self._histogram_cls(
            name="vllm:eplb_load_expert_skew",
            documentation=(
                "Max/mean token load across physical experts for one MoE "
                "layer. 1.0 is perfectly balanced."
            ),
            labelnames=load_labelnames,
            buckets=skew_buckets,
        )
        histogram_critical_expert_excess = self._histogram_cls(
            name="vllm:eplb_critical_expert_excess_tokens",
            documentation=(
                "Tokens on the busiest physical expert above the expert mean "
                "for one MoE layer."
            ),
            labelnames=load_labelnames,
            buckets=excess_buckets,
        )
        concentration_buckets = (
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        )
        histogram_expert_gini = self._histogram_cls(
            name="vllm:eplb_expert_token_gini",
            documentation=(
                "Gini coefficient of per-expert token load for one MoE layer. "
                "0.0 is perfectly balanced and 1.0 is fully concentrated."
            ),
            labelnames=load_labelnames,
            buckets=concentration_buckets,
        )
        topk_labelnames = [*load_labelnames, "top_k"]
        histogram_expert_topk_share = self._histogram_cls(
            name="vllm:eplb_expert_topk_token_share",
            documentation=(
                "Share of layer tokens routed to the top-k hottest physical experts."
            ),
            labelnames=topk_labelnames,
            buckets=concentration_buckets,
        )
        gauge_active_experts = self._gauge_cls(
            name="vllm:eplb_active_experts_per_rank",
            documentation=(
                "Physical experts with nonzero token load on one EP rank in "
                "one MoE layer."
            ),
            multiprocess_mode="mostrecent",
            labelnames=[*load_labelnames, "ep_rank"],
        )
        self._histogram_stall_metric = histogram_stall
        self._counter_blocked_requests_metric = counter_blocked_requests
        self._counter_blocked_tokens_metric = counter_blocked_tokens
        self._histogram_load_rank_skew_metric = histogram_load_rank_skew
        self._histogram_critical_rank_excess_metric = histogram_critical_rank_excess
        self._histogram_load_expert_skew_metric = histogram_load_expert_skew
        self._histogram_critical_expert_excess_metric = histogram_critical_expert_excess
        self._histogram_expert_gini_metric = histogram_expert_gini
        self._histogram_expert_topk_share_metric = histogram_expert_topk_share
        self._gauge_active_experts_metric = gauge_active_experts
        self._histogram_stall: dict[tuple[int, str, str, str], object] = {}
        self._counter_blocked_requests: dict[tuple[int, str, str, str], object] = {}
        self._counter_blocked_tokens: dict[tuple[int, str, str, str], object] = {}
        self._histogram_load_rank_skew: dict[tuple[int, str, str, int], object] = {}
        self._histogram_critical_rank_excess: dict[
            tuple[int, str, str, int], object
        ] = {}
        self._histogram_load_expert_skew: dict[tuple[int, str, str, int], object] = {}
        self._histogram_critical_expert_excess: dict[
            tuple[int, str, str, int], object
        ] = {}
        self._histogram_expert_gini: dict[tuple[int, str, str, int], object] = {}
        self._histogram_expert_topk_share: dict[
            tuple[int, str, str, int, int], object
        ] = {}
        self._gauge_active_experts: dict[tuple[int, str, str, int, int], object] = {}

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

    def _layer_load_metric(
        self,
        cache: dict[tuple[int, str, str, int], object],
        metric,
        engine_idx: int,
        model_name: str,
        phase: str,
        layer: int,
    ):
        key = (engine_idx, model_name, phase, layer)
        child = cache.get(key)
        if child is None:
            child = metric.labels(
                *self.per_engine_labelvalues[engine_idx],
                model_name,
                phase,
                str(layer),
            )
            cache[key] = child
        return child

    def _topk_load_metric(
        self,
        cache: dict[tuple[int, str, str, int, int], object],
        metric,
        engine_idx: int,
        model_name: str,
        phase: str,
        layer: int,
        top_k: int,
    ):
        key = (engine_idx, model_name, phase, layer, top_k)
        child = cache.get(key)
        if child is None:
            child = metric.labels(
                *self.per_engine_labelvalues[engine_idx],
                model_name,
                phase,
                str(layer),
                str(top_k),
            )
            cache[key] = child
        return child

    def _active_experts_metric(
        self,
        cache: dict[tuple[int, str, str, int, int], object],
        metric,
        engine_idx: int,
        model_name: str,
        phase: str,
        layer: int,
        ep_rank: int,
    ):
        key = (engine_idx, model_name, phase, layer, ep_rank)
        child = cache.get(key)
        if child is None:
            child = metric.labels(
                *self.per_engine_labelvalues[engine_idx],
                model_name,
                phase,
                str(layer),
                str(ep_rank),
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

    def _stall_metric(
        self,
        cache: dict[tuple[int, str, str, str], object],
        metric,
        engine_idx: int,
        model_name: str,
        phase: str,
        is_async: bool,
    ):
        async_label = "true" if is_async else "false"
        key = (engine_idx, model_name, phase, async_label)
        child = cache.get(key)
        if child is None:
            child = metric.labels(
                *self.per_engine_labelvalues[engine_idx],
                model_name,
                phase,
                async_label,
            )
            cache[key] = child
        return child

    def observe(self, snapshot: EplbMetricsSnapshot, engine_idx: int = 0):
        if not self.enabled:
            return

        for stall in snapshot.stalls:
            pause = max(0.0, stall.pause_seconds)
            self._stall_metric(
                self._histogram_stall,
                self._histogram_stall_metric,
                engine_idx,
                stall.model_name,
                stall.phase,
                stall.is_async,
            ).observe(pause)
            self._stall_metric(
                self._counter_blocked_requests,
                self._counter_blocked_requests_metric,
                engine_idx,
                stall.model_name,
                stall.phase,
                stall.is_async,
            ).inc(pause * stall.blocked_requests)
            self._stall_metric(
                self._counter_blocked_tokens,
                self._counter_blocked_tokens_metric,
                engine_idx,
                stall.model_name,
                stall.phase,
                stall.is_async,
            ).inc(pause * stall.blocked_tokens)

        for load in snapshot.loads:
            self._layer_load_metric(
                self._histogram_load_rank_skew,
                self._histogram_load_rank_skew_metric,
                engine_idx,
                load.model_name,
                load.phase,
                load.layer,
            ).observe(load.rank_skew)
            self._layer_load_metric(
                self._histogram_critical_rank_excess,
                self._histogram_critical_rank_excess_metric,
                engine_idx,
                load.model_name,
                load.phase,
                load.layer,
            ).observe(load.rank_excess_tokens)
            self._layer_load_metric(
                self._histogram_load_expert_skew,
                self._histogram_load_expert_skew_metric,
                engine_idx,
                load.model_name,
                load.phase,
                load.layer,
            ).observe(load.expert_skew)
            self._layer_load_metric(
                self._histogram_critical_expert_excess,
                self._histogram_critical_expert_excess_metric,
                engine_idx,
                load.model_name,
                load.phase,
                load.layer,
            ).observe(load.expert_excess_tokens)
            self._layer_load_metric(
                self._histogram_expert_gini,
                self._histogram_expert_gini_metric,
                engine_idx,
                load.model_name,
                load.phase,
                load.layer,
            ).observe(load.expert_gini)
            self._topk_load_metric(
                self._histogram_expert_topk_share,
                self._histogram_expert_topk_share_metric,
                engine_idx,
                load.model_name,
                load.phase,
                load.layer,
                load.expert_top_k,
            ).observe(load.expert_topk_share)
            for ep_rank, active_experts in enumerate(load.active_experts_per_rank):
                self._active_experts_metric(
                    self._gauge_active_experts,
                    self._gauge_active_experts_metric,
                    engine_idx,
                    load.model_name,
                    load.phase,
                    load.layer,
                    ep_rank,
                ).set(active_experts)

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
