# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import prometheus_client

from vllm.distributed.eplb.metrics import (
    EPLB_PHASE_PREFILL,
    EplbExpertTokens,
    EplbExpertTopology,
    EplbMetricsSnapshot,
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

    monkeypatch.setattr(EplbProm, "_gauge_cls", TestGauge)
    monkeypatch.setattr(EplbProm, "_counter_cls", TestCounter)

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
