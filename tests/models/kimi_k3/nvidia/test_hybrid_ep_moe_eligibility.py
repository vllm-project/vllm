# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which Kimi K3 MoE layers take the hybrid DEP MoE schedule
(VLLM_KIMI_K3_HYBRID_EP_MOE, KimiMoE._hybrid_ep_layer): the env var, the layer
layout and every weight DepMoe reads directly being BF16. DepMoe itself is a
stand-in here, so no kernel runs."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia import model as kimi
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.ep.layer import (
    FRONT_MAX,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import dep

ENV = "VLLM_KIMI_K3_HYBRID_EP_MOE"


def _linear(dtype=torch.bfloat16):
    return SimpleNamespace(weight=torch.empty(1, dtype=dtype))


def _moe():
    """The attributes `_hybrid_ep_layer` reads, as on an eligible hybrid layer:
    sequence-parallel MegaMoE with a replicated BF16 shared expert and latent
    projections, routed by one-group sigmoid top-16."""
    return SimpleNamespace(
        _hybrid_ep=kimi._HYBRID_EP_UNBUILT,
        _down_proj_stream=object(),
        use_sequence_parallel=True,
        tp_size=8,
        use_mega_moe=True,
        experts=SimpleNamespace(top_k=16),
        shared_experts=SimpleNamespace(
            shard_sequence_parallel=False,
            gemm_rs_ar=None,
            gate_up_proj=_linear(),
            down_proj=_linear(),
        ),
        routed_expert_norm=object(),
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        moe_router_activation_func="sigmoid",
        gate=_linear(),
        routed_expert_down_proj=_linear(),
        routed_expert_up_proj=_linear(),
    )


@pytest.fixture
def built(monkeypatch):
    """The DepMoe constructions `_hybrid_ep_layer` makes."""
    calls = []

    class StandInDepMoe:
        def __init__(self, module, batches, side_stream, *, front_gemm=False):
            calls.append((module, batches, side_stream, front_gemm))

    monkeypatch.setattr(dep, "DepMoe", StandInDepMoe)
    return calls


def test_eligible_layer_builds_once(monkeypatch, built):
    monkeypatch.setenv(ENV, "1")
    moe = _moe()
    ep = kimi.KimiMoE._hybrid_ep_layer(moe)
    assert isinstance(ep, dep.DepMoe)
    assert built == [(moe, list(range(1, FRONT_MAX + 1)), moe._down_proj_stream, True)]
    assert kimi.KimiMoE._hybrid_ep_layer(moe) is ep
    assert len(built) == 1


def test_env_unset_keeps_stock_path(monkeypatch, built):
    monkeypatch.delenv(ENV, raising=False)
    moe = _moe()
    assert kimi.KimiMoE._hybrid_ep_layer(moe) is None
    assert moe._hybrid_ep is None
    assert built == []


def _quantized(name):
    def change(moe):
        holder = moe.shared_experts if name.startswith("shared.") else moe
        setattr(holder, name.removeprefix("shared."), _linear(torch.float8_e4m3fn))

    return change


INELIGIBLE = {
    "gate_fp8": _quantized("gate"),
    "shared_gate_up_fp8": _quantized("shared.gate_up_proj"),
    "shared_down_fp8": _quantized("shared.down_proj"),
    "latent_down_fp8": _quantized("routed_expert_down_proj"),
    "latent_up_fp8": _quantized("routed_expert_up_proj"),
    "latent_up_without_weight": lambda m: setattr(
        m, "routed_expert_up_proj", SimpleNamespace()
    ),
    "no_latent_moe": lambda m: setattr(m, "routed_expert_down_proj", None),
    "no_sequence_parallel": lambda m: setattr(m, "use_sequence_parallel", False),
    "tp1": lambda m: setattr(m, "tp_size", 1),
    "sharded_shared_expert": lambda m: setattr(
        m.shared_experts, "shard_sequence_parallel", True
    ),
    "shared_gemm_rs_ar": lambda m: setattr(m.shared_experts, "gemm_rs_ar", object()),
    "top_k_8": lambda m: setattr(m.experts, "top_k", 8),
}


@pytest.mark.parametrize("case", sorted(INELIGIBLE))
def test_ineligible_layer_keeps_stock_path(monkeypatch, built, case):
    monkeypatch.setenv(ENV, "1")
    moe = _moe()
    INELIGIBLE[case](moe)
    assert kimi.KimiMoE._hybrid_ep_layer(moe) is None
    assert built == []
