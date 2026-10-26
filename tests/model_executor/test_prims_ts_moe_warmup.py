# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.experts.prims_ts_nvfp4_moe import (
    PrimsTsNvFp4ExpertsModular,
)
from vllm.model_executor.warmup import prims_ts_moe_warmup


def _layer(hidden_size: int = 64, num_experts: int = 8) -> SimpleNamespace:
    """A Prims-TS MoE layer whose precompile kwargs hold fresh tensor data."""
    experts = object.__new__(PrimsTsNvFp4ExpertsModular)
    experts.untuned_precompile_kwargs = lambda layer: dict(
        topk_ids=(torch.randint(8, (4, 2)), torch.rand(4, 2)),
        hidden_states=torch.rand(4, hidden_size),
        num_experts=num_experts,
        routing_bias=None,
    )
    return SimpleNamespace(prims_ts_moe_warmup_provider=experts)


def _precompiled_calls(monkeypatch, modules, available: bool = True) -> list[dict]:
    calls: list[dict] = []

    def precompile_untuned(kwargs):
        calls.append(kwargs)
        return 5

    monkeypatch.setattr(
        prims_ts_moe_warmup, "has_flashinfer_prims_ts_moe_precompile", lambda: available
    )
    monkeypatch.setattr(
        PrimsTsNvFp4ExpertsModular,
        "precompile_untuned",
        staticmethod(precompile_untuned),
    )
    model = SimpleNamespace(modules=lambda: iter(modules))
    prims_ts_moe_warmup.prims_ts_moe_warmup(model)
    return calls


def test_layers_differing_only_in_tensor_data_precompile_once(monkeypatch):
    """Weights differ from layer to layer but select the same kernels."""
    modules = [
        _layer(),
        SimpleNamespace(),
        _layer(),
        _layer(hidden_size=128),
        _layer(num_experts=16),
    ]

    calls = _precompiled_calls(monkeypatch, modules)

    assert [call["hidden_states"].shape[1] for call in calls] == [64, 128, 64]
    assert [call["num_experts"] for call in calls] == [8, 8, 16]


def test_only_prims_ts_experts_are_precompiled(monkeypatch):
    stray = SimpleNamespace(prims_ts_moe_warmup_provider=object())

    assert _precompiled_calls(monkeypatch, [stray]) == []


def test_flashinfer_without_precompile_skips_warmup(monkeypatch):
    assert _precompiled_calls(monkeypatch, [_layer()], available=False) == []
