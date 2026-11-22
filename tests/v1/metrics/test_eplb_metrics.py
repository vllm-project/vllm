# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import prometheus_client

from vllm.distributed.eplb.metrics import (
    EPLB_PHASE_DECODE,
    EPLB_PHASE_PREFILL,
    EplbExpertTokens,
    EplbExpertTopology,
    EplbLoadSample,
    EplbMetricsSnapshot,
    EplbRearrangementStall,
)
from vllm.v1.metrics.eplb import EplbProm


def make_prom(monkeypatch, *, enabled: bool = True):
    registry = prometheus_client.CollectorRegistry()

    class TestGauge:
        def __new__(cls, *args, **kwargs):
            return prometheus_client.Gauge(*args, registry=registry, **kwargs)

    class TestCounter:
        def __new__(cls, *args, **kwargs):
            return prometheus_client.Counter(*args, registry=registry, **kwargs)

    class TestHistogram:
        def __new__(cls, *args, **kwargs):
            return prometheus_client.Histogram(*args, registry=registry, **kwargs)

    monkeypatch.setattr(EplbProm, "_gauge_cls", TestGauge)
    monkeypatch.setattr(EplbProm, "_counter_cls", TestCounter)
    monkeypatch.setattr(EplbProm, "_histogram_cls", TestHistogram)

    config = SimpleNamespace(
        enable_eplb=True,
        eplb_config=SimpleNamespace(enable_expert_metrics=enabled),
    )
    prom = EplbProm(
        config,
        ["model_name", "engine"],
        {0: ["served", "0"]},
    )
    return prom, registry


def test_eplb_prom_exports_topology_and_tokens(monkeypatch):
    prom, registry = make_prom(monkeypatch)
    snapshot = EplbMetricsSnapshot(
        tokens=[
            EplbExpertTokens(
                model_name="served",
                ep_rank=0,
                num_ep_ranks=2,
                num_local_experts=2,
                tokens=[[2, 0, 1, 3]],
                physical_to_logical=[[0, 1, 0, 1]],
                phase=EPLB_PHASE_PREFILL,
                timestamp=123.0,
                steps=5,
                rearranged=True,
            )
        ],
        topology=[
            EplbExpertTopology(
                model_name="served",
                ep_rank=0,
                num_ep_ranks=2,
                num_local_experts=2,
                physical_to_logical=[[0, 1, 0, 1]],
                timestamp=124.0,
            )
        ],
        stalls=[
            EplbRearrangementStall(
                model_name="served",
                pause_seconds=1.5,
                blocked_requests=8,
                blocked_tokens=512,
                phase=EPLB_PHASE_DECODE,
                is_async=False,
                timestamp=125.0,
            )
        ],
        loads=[
            EplbLoadSample(
                model_name="served",
                phase=EPLB_PHASE_DECODE,
                layer=0,
                num_ep_ranks=2,
                num_local_experts=2,
                expert_tokens=[4, 1, 2, 1],
                expert_gini=0.3125,
                expert_top_k=2,
                expert_topk_share=0.75,
                rank_tokens=[5, 3],
                active_experts_per_rank=[2, 2],
                expert_skew=2.0,
                expert_excess_tokens=2,
                rank_skew=1.25,
                rank_excess_tokens=1,
                timestamp=125.0,
                steps=10,
            )
        ],
    )

    prom.observe(snapshot, engine_idx=0)

    def labels(physical_expert: int, *, include_logical: bool = True) -> dict[str, str]:
        labels = {
            "model_name": "served",
            "engine": "0",
            "model": "served",
            "layer": "0",
            "ep_rank": str(physical_expert // 2),
            "physical_expert": str(physical_expert),
        }
        if include_logical:
            labels["logical_expert"] = str(physical_expert % 2)
            labels["phase"] = EPLB_PHASE_PREFILL
        return labels

    assert registry.get_sample_value("vllm:eplb_expert_tokens_total", labels(0)) == 2.0
    assert registry.get_sample_value("vllm:eplb_expert_tokens_total", labels(2)) == 1.0
    assert registry.get_sample_value("vllm:eplb_expert_tokens_total", labels(3)) == 3.0
    # Zero-token experts are not exported.
    assert registry.get_sample_value("vllm:eplb_expert_tokens_total", labels(1)) is None

    def topology_labels(physical_expert: int) -> dict[str, str]:
        return labels(physical_expert, include_logical=False)

    assert (
        registry.get_sample_value("vllm:eplb_expert_topology", topology_labels(0))
        == 0.0
    )
    assert (
        registry.get_sample_value("vllm:eplb_expert_topology", topology_labels(3))
        == 1.0
    )

    model_labels = {"model_name": "served", "engine": "0", "model": "served"}
    assert (
        registry.get_sample_value("vllm:eplb_rearrangements_total", model_labels) == 1.0
    )
    assert (
        registry.get_sample_value(
            "vllm:eplb_rearrangement_timestamp_seconds", model_labels
        )
        == 123.0
    )
    assert (
        registry.get_sample_value(
            "vllm:eplb_expert_topology_timestamp_seconds", model_labels
        )
        == 124.0
    )
    load_labels = {**model_labels, "phase": EPLB_PHASE_PREFILL}
    assert (
        registry.get_sample_value(
            "vllm:eplb_expert_load_timestamp_seconds", load_labels
        )
        == 123.0
    )
    assert (
        registry.get_sample_value("vllm:eplb_expert_load_window_steps", load_labels)
        == 5.0
    )

    stall_labels = {
        "model_name": "served",
        "engine": "0",
        "model": "served",
        "phase": EPLB_PHASE_DECODE,
        "async": "false",
    }
    assert (
        registry.get_sample_value(
            "vllm:eplb_rearrangement_blocked_request_seconds_total", stall_labels
        )
        == 12.0
    )
    assert (
        registry.get_sample_value(
            "vllm:eplb_rearrangement_blocked_token_seconds_total", stall_labels
        )
        == 768.0
    )
    pause_count = registry.get_sample_value(
        "vllm:eplb_rearrangement_pause_seconds_count", stall_labels
    )
    assert pause_count == 1.0
    pause_sum = registry.get_sample_value(
        "vllm:eplb_rearrangement_pause_seconds_sum", stall_labels
    )
    assert pause_sum == 1.5

    skew_labels = {
        "model_name": "served",
        "engine": "0",
        "model": "served",
        "phase": EPLB_PHASE_DECODE,
        "layer": "0",
    }
    assert (
        registry.get_sample_value("vllm:eplb_load_rank_skew_count", skew_labels) == 1.0
    )
    assert (
        registry.get_sample_value("vllm:eplb_load_rank_skew_sum", skew_labels) == 1.25
    )
    assert (
        registry.get_sample_value(
            "vllm:eplb_critical_rank_excess_tokens_sum", skew_labels
        )
        == 1.0
    )
    assert (
        registry.get_sample_value("vllm:eplb_load_expert_skew_count", skew_labels)
        == 1.0
    )
    assert (
        registry.get_sample_value("vllm:eplb_load_expert_skew_sum", skew_labels) == 2.0
    )
    assert (
        registry.get_sample_value(
            "vllm:eplb_critical_expert_excess_tokens_sum", skew_labels
        )
        == 2.0
    )
    assert (
        registry.get_sample_value("vllm:eplb_expert_token_gini_count", skew_labels)
        == 1.0
    )
    assert (
        registry.get_sample_value("vllm:eplb_expert_token_gini_sum", skew_labels)
        == 0.3125
    )
    topk_labels = {**skew_labels, "top_k": "2"}
    assert (
        registry.get_sample_value(
            "vllm:eplb_expert_topk_token_share_count", topk_labels
        )
        == 1.0
    )
    assert (
        registry.get_sample_value("vllm:eplb_expert_topk_token_share_sum", topk_labels)
        == 0.75
    )
    for ep_rank in range(2):
        active_labels = {**skew_labels, "ep_rank": str(ep_rank)}
        assert (
            registry.get_sample_value(
                "vllm:eplb_active_experts_per_rank", active_labels
            )
            == 2.0
        )


def test_eplb_prom_is_disabled_by_default(monkeypatch):
    prom, _ = make_prom(monkeypatch, enabled=False)
    prom.observe(
        EplbMetricsSnapshot(
            topology=[
                EplbExpertTopology(
                    model_name="served",
                    ep_rank=0,
                    num_ep_ranks=1,
                    num_local_experts=1,
                    physical_to_logical=[[0]],
                )
            ]
        ),
        engine_idx=0,
    )
    assert not hasattr(prom, "_gauge_metric")
