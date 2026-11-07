# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 MoE intermediate-size ownership tests."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    mxfp4_round_up_hidden_size_and_intermediate_size,
)

K3_MOE_INTERMEDIATE_SIZE = 3072
K3_ROUTED_EXPERT_HIDDEN_SIZE = 3584


def _make_kimi_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=7168,
        moe_intermediate_size=K3_MOE_INTERMEDIATE_SIZE,
        routed_expert_hidden_size=K3_ROUTED_EXPERT_HIDDEN_SIZE,
        num_experts=896,
        num_experts_per_token=16,
        num_shared_experts=None,
        moe_renormalize=True,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=1.0,
        hidden_act="situ",
        activation_situ_beta=1.0,
        activation_situ_linear_beta=1.0,
        latent_moe_use_norm=False,
        rms_norm_eps=1e-6,
        min_moe_intermediate_per_partition=256,
    )


class _StubLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.e_score_correction_bias = None


class _RecordingExperts(nn.Module):
    calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        type(self).calls.append(dict(kwargs))
        self.moe_config = SimpleNamespace()
        self.w13_weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.w2_weight = nn.Parameter(torch.ones(1), requires_grad=False)


def _stub_common_model_dependencies(monkeypatch, model_mod, tp_size: int) -> None:
    monkeypatch.setattr(
        model_mod, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(model_mod, "GateLinear", _StubLinear)
    monkeypatch.setattr(model_mod, "ReplicatedLinear", _StubLinear)
    monkeypatch.setattr(model_mod, "RMSNorm", _StubLinear)
    monkeypatch.setattr(model_mod, "KimiMLP", _StubLinear)
    monkeypatch.setattr(model_mod, "KimiRoutedOutputTransform", _StubLinear)


@pytest.mark.parametrize(
    ("tp_size", "expected_model_size"),
    [(8, 3072), (16, 4096), (32, 8192)],
)
def test_nvidia_regular_kimi_moe_delegates_logical_size(
    monkeypatch: pytest.MonkeyPatch, tp_size: int, expected_model_size: int
):
    model_mod = pytest.importorskip("vllm.models.kimi_k3.nvidia.model")
    _RecordingExperts.calls = []
    _stub_common_model_dependencies(monkeypatch, model_mod, tp_size)
    monkeypatch.setattr(model_mod, "FusedMoEFactory", _RecordingExperts)
    monkeypatch.setattr(model_mod, "aux_stream", lambda: None)
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace())
    monkeypatch.setattr(
        model_mod,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: False,
            is_device_capability_family=lambda *_: False,
        ),
    )
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="marlin"),
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
    )

    module = model_mod.KimiMoE(_make_kimi_config(), vllm_config)

    assert len(_RecordingExperts.calls) == 1
    assert _RecordingExperts.calls[0]["intermediate_size"] == 3072
    assert module.padded_moe_intermediate_size == expected_model_size
    assert torch.count_nonzero(module.experts.w13_weight) == 1
    assert torch.count_nonzero(module.experts.w2_weight) == 1
    assert not hasattr(
        module.experts.moe_config, "intermediate_size_per_partition_unpadded"
    )


@pytest.mark.parametrize(
    ("backend", "expected_intermediate"),
    [
        (Mxfp4MoeBackend.MARLIN, 128),
        (Mxfp4MoeBackend.BATCHED_MARLIN, 128),
        (Mxfp4MoeBackend.DEEPGEMM_MXFP4, 128),
        (Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8, 128),
        (Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16, 128),
        (Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8, 128),
        (Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16, 128),
        (Mxfp4MoeBackend.EMULATION, 96),
        (Mxfp4MoeBackend.CPU, 96),
    ],
)
def test_backend_rounding_of_tp32_partition(backend, expected_intermediate):
    _, rounded = mxfp4_round_up_hidden_size_and_intermediate_size(
        backend, K3_ROUTED_EXPERT_HIDDEN_SIZE, 96
    )
    assert rounded == expected_intermediate


@pytest.mark.parametrize(
    ("tp_size", "expected_local"), [(8, 384), (16, 256), (32, 128)]
)
def test_marlin_tp_sweep(tp_size: int, expected_local: int):
    raw_local = K3_MOE_INTERMEDIATE_SIZE // tp_size
    _, rounded = mxfp4_round_up_hidden_size_and_intermediate_size(
        Mxfp4MoeBackend.MARLIN, K3_ROUTED_EXPERT_HIDDEN_SIZE, raw_local
    )

    assert rounded == expected_local
