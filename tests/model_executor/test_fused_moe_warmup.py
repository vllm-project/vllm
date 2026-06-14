# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.model_executor.warmup.fused_moe_warmup as fused_moe_warmup
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_moe_kernel_gptq_awq,
)


def _fake_wna16_cuda_dispatch(
    num_valid_tokens: int,
    group_size: int,
    num_experts: int,
    bit: int,
) -> bool:
    return (
        bit == 4 and group_size in [32, 64, 128] and num_valid_tokens / num_experts <= 6
    )


def test_fused_moe_gptq_awq_kernel_does_not_specialize_token_counts():
    assert set(fused_moe_kernel_gptq_awq.do_not_specialize) >= {
        "EM",
        "num_valid_tokens",
    }


def test_wna16_warmup_m_values_cover_first_and_second_gemm(monkeypatch):
    monkeypatch.setattr(
        fused_moe_warmup,
        "should_moe_wna16_use_cuda",
        _fake_wna16_cuda_dispatch,
    )

    # 128 local experts, top_k=2:
    # - w13 uses top_k=2, first Triton M is floor(6 * 128 / 2) + 1 = 385
    # - w2 uses top_k=1, first Triton M is floor(6 * 128 / 1) + 1 = 769
    assert fused_moe_warmup._generate_wna16_triton_m_values(
        max_num_batched_tokens=4096,
        num_experts=128,
        top_k=2,
        group_size=128,
        weight_bits=4,
    ) == [385, 769]


def test_wna16_warmup_m_values_cover_reachable_default_buckets(monkeypatch):
    monkeypatch.setattr(
        fused_moe_warmup,
        "should_moe_wna16_use_cuda",
        _fake_wna16_cuda_dispatch,
    )

    assert fused_moe_warmup._generate_wna16_triton_m_values(
        max_num_batched_tokens=64,
        num_experts=4,
        top_k=2,
        group_size=128,
        weight_bits=4,
    ) == [13, 21, 25, 41]


def test_wna16_warmup_m_values_cover_int8_default_buckets(monkeypatch):
    monkeypatch.setattr(
        fused_moe_warmup,
        "should_moe_wna16_use_cuda",
        _fake_wna16_cuda_dispatch,
    )

    assert fused_moe_warmup._generate_wna16_triton_m_values(
        max_num_batched_tokens=64,
        num_experts=128,
        top_k=2,
        group_size=128,
        weight_bits=8,
    ) == [1, 21, 41]


class _FakeWNA16Method:
    def __init__(self):
        self.quant_config = SimpleNamespace(weight_bits=4, group_size=128)
        self.calls = []

    def apply(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "x_shape": tuple(x.shape),
                "topk_weights_shape": tuple(topk_weights.shape),
                "topk_ids": topk_ids.clone(),
                "shared_experts_input": shared_experts_input,
            }
        )
        return x


class _FakeFusedMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_method = _FakeWNA16Method()
        self.w13_qweight = torch.empty((2, 8, 2), dtype=torch.uint8)
        self.w2_qweight = torch.empty((2, 4, 4), dtype=torch.uint8)
        self.w13_scales = torch.empty((2, 8, 1))
        self.w2_scales = torch.empty((2, 4, 1))
        self.group_size = 128
        self.top_k = 2
        self.global_num_experts = 4
        self.expert_map = torch.tensor([-1, 0, -1, 1], dtype=torch.int32)
        self.rocm_aiter_fmoe_enabled = False
        self.activation = "silu"
        self.apply_router_weight_on_input = False
        self.moe_config = SimpleNamespace(
            in_dtype=torch.float16,
            hidden_dim=4,
        )
        self.ensure_moe_quant_config_init_called = False

    def ensure_moe_quant_config_init(self):
        self.ensure_moe_quant_config_init_called = True


def test_wna16_warmup_uses_local_experts_and_quant_config(monkeypatch):
    monkeypatch.setattr(fused_moe_warmup, "FusedMoE", _FakeFusedMoE)
    monkeypatch.setattr(fused_moe_warmup, "MoeWNA16Method", _FakeWNA16Method)
    monkeypatch.setattr(
        fused_moe_warmup,
        "should_moe_wna16_use_cuda",
        lambda *args, **kwargs: False,
    )

    layer = _FakeFusedMoE()
    model = torch.nn.Sequential(layer)

    fused_moe_warmup.fused_moe_wna16_warmup(
        model,
        max_num_batched_tokens=2,
    )

    assert layer.ensure_moe_quant_config_init_called
    assert len(layer.quant_method.calls) == 1
    call = layer.quant_method.calls[0]
    assert call["x_shape"] == (1, 4)
    assert call["topk_weights_shape"] == (1, 2)
    assert call["shared_experts_input"] is None
    assert call["topk_ids"].tolist() == [[1, 3]]


def test_wna16_warmup_rocm_aiter_expert_mask(monkeypatch):
    # ROCm AITER returns a binary expert mask from the expert_map property.
    monkeypatch.setattr(fused_moe_warmup, "FusedMoE", _FakeFusedMoE)
    monkeypatch.setattr(fused_moe_warmup, "MoeWNA16Method", _FakeWNA16Method)
    monkeypatch.setattr(
        fused_moe_warmup,
        "should_moe_wna16_use_cuda",
        lambda *args, **kwargs: False,
    )

    layer = _FakeFusedMoE()
    layer.rocm_aiter_fmoe_enabled = True
    layer.expert_map = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    model = torch.nn.Sequential(layer)

    fused_moe_warmup.fused_moe_wna16_warmup(
        model,
        max_num_batched_tokens=2,
    )

    assert len(layer.quant_method.calls) == 1
    call = layer.quant_method.calls[0]
    assert call["topk_ids"].tolist() == [[1, 3]]
