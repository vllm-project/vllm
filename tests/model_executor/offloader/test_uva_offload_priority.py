# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.offloader.uva import (
    UVAOffloader,
    _is_sparse_expert_param,
)


@pytest.fixture
def should_do_global_cleanup_after_test():
    return False


def _is_offloaded(p: nn.Parameter) -> bool:
    return getattr(p, "_vllm_is_uva_offloaded", False)


class MockParam(nn.Parameter):
    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0")


def _make_mock_param(tensor: torch.Tensor) -> nn.Parameter:
    return MockParam(tensor.clone())


class MockAttention(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.qkv_proj.weight = _make_mock_param(self.qkv_proj.weight)
        self.o_proj.weight = _make_mock_param(self.o_proj.weight)


class MockSparseMoE(nn.Module):
    def __init__(self, dim: int = 128, num_experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.gate.weight = _make_mock_param(self.gate.weight)
        self.shared_expert = nn.Linear(dim, dim * 2, bias=False)
        self.shared_expert.weight = _make_mock_param(self.shared_expert.weight)
        self.experts = _make_mock_param(torch.randn(num_experts, dim, dim * 2))


class MockMoELayer(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(dim)
        self.input_layernorm.weight = _make_mock_param(self.input_layernorm.weight)
        self.input_layernorm.bias = _make_mock_param(self.input_layernorm.bias)
        self.self_attn = MockAttention(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.post_attention_layernorm.weight = _make_mock_param(
            self.post_attention_layernorm.weight
        )
        self.post_attention_layernorm.bias = _make_mock_param(
            self.post_attention_layernorm.bias
        )
        self.mlp = MockSparseMoE(dim)


class MockDenseLayer(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(dim)
        self.input_layernorm.weight = _make_mock_param(self.input_layernorm.weight)
        self.input_layernorm.bias = _make_mock_param(self.input_layernorm.bias)
        self.self_attn = MockAttention(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.post_attention_layernorm.weight = _make_mock_param(
            self.post_attention_layernorm.weight
        )
        self.post_attention_layernorm.bias = _make_mock_param(
            self.post_attention_layernorm.bias
        )
        self.mlp_w1 = nn.Linear(dim, dim * 4, bias=False)
        self.mlp_w1.weight = _make_mock_param(self.mlp_w1.weight)
        self.mlp_w2 = nn.Linear(dim * 4, dim, bias=False)
        self.mlp_w2.weight = _make_mock_param(self.mlp_w2.weight)


@pytest.fixture(autouse=True)
def mock_uva_runtime(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.is_uva_available", lambda: True
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.should_pin_memory", lambda: False
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.get_accelerator_view_from_cpu_tensor",
        lambda x: x,
    )


def test_sparse_expert_param_classification():
    """Verify classification of sparse vs dense parameters."""
    # Real MoE sparse expert patterns
    assert _is_sparse_expert_param("model.layers.0.mlp.experts.w13_weight")
    assert _is_sparse_expert_param("model.layers.0.mlp.experts.w2_weight")
    assert _is_sparse_expert_param("model.layers.0.block_sparse_moe.experts.0")
    assert _is_sparse_expert_param("model.layers.0.mlp.fused_moe.w13_weight")
    assert _is_sparse_expert_param("model.layers.0.mlp.switch_mlp.w13_weight")
    assert _is_sparse_expert_param("model.layers.0.mlp.moe_experts.w13_weight")

    # Dense weights must NOT be classified as sparse experts
    assert not _is_sparse_expert_param("model.layers.0.self_attn.qkv_proj.weight")
    assert not _is_sparse_expert_param("model.layers.0.self_attn.o_proj.weight")
    assert not _is_sparse_expert_param("model.layers.0.input_layernorm.weight")
    assert not _is_sparse_expert_param("model.layers.0.mlp.gate.weight")
    assert not _is_sparse_expert_param("model.layers.0.mlp.shared_expert.weight")
    assert not _is_sparse_expert_param("model.layers.0.mlp.shared_experts.w1.weight")
    assert not _is_sparse_expert_param("model.layers.0.mlp.share_expert.weight")

    # Edge cases with misleading substrings
    assert not _is_sparse_expert_param("model.layers.0.mlp.expert_gate.weight")
    assert not _is_sparse_expert_param("model.layers.0.mlp.router_logits.weight")
    assert not _is_sparse_expert_param("visual.transformer.resblocks.0.expert_mode")
    assert not _is_sparse_expert_param("language_model.model.embed_tokens.weight")


def test_moe_expert_priority_offloading_single_layer():
    """Verify sparse experts are offloaded before dense attention/norm weights."""
    layer = MockMoELayer(dim=128)
    expert_bytes = layer.mlp.experts.numel() * layer.mlp.experts.element_size()

    # Allocate budget sufficient only for expert parameters
    offloader = UVAOffloader(cpu_offload_max_bytes=expert_bytes)
    offloader.wrap_modules([layer], prefix="model.layers.0")

    # Experts MUST be offloaded
    assert _is_offloaded(layer.mlp.experts)

    # Dense attention, router, and layernorms MUST NOT be offloaded
    assert not _is_offloaded(layer.self_attn.qkv_proj.weight)
    assert not _is_offloaded(layer.self_attn.o_proj.weight)
    assert not _is_offloaded(layer.input_layernorm.weight)
    assert not _is_offloaded(layer.post_attention_layernorm.weight)
    assert not _is_offloaded(layer.mlp.gate.weight)
    assert not _is_offloaded(layer.mlp.shared_expert.weight)


def test_moe_expert_priority_multi_layer_global_ordering():
    """Verify global expert prioritization across multiple layers."""
    layer0 = MockMoELayer(dim=128)
    layer1 = MockMoELayer(dim=128)

    expert_bytes_per_layer = (
        layer0.mlp.experts.numel() * layer0.mlp.experts.element_size()
    )

    # Budget covers exactly 2 layers of experts
    budget = expert_bytes_per_layer * 2
    offloader = UVAOffloader(cpu_offload_max_bytes=budget)
    offloader.wrap_modules([layer0, layer1], prefix="model.layers")

    # Both layers' experts MUST be offloaded
    assert _is_offloaded(layer0.mlp.experts)
    assert _is_offloaded(layer1.mlp.experts)

    # Neither layer's dense attention/norm should be offloaded
    assert not _is_offloaded(layer0.self_attn.qkv_proj.weight)
    assert not _is_offloaded(layer1.self_attn.qkv_proj.weight)
    assert not _is_offloaded(layer0.input_layernorm.weight)
    assert not _is_offloaded(layer1.input_layernorm.weight)


def test_budget_spillover_to_dense():
    """Verify that if budget exceeds total experts, dense parameters
    are also offloaded in declaration order.
    """
    layer = MockMoELayer(dim=128)
    expert_bytes = layer.mlp.experts.numel() * layer.mlp.experts.element_size()
    qkv_bytes = (
        layer.self_attn.qkv_proj.weight.numel()
        * layer.self_attn.qkv_proj.weight.element_size()
    )

    # Budget covers all experts + layernorm + qkv_proj
    budget = expert_bytes + qkv_bytes + 1024
    offloader = UVAOffloader(cpu_offload_max_bytes=budget)
    offloader.wrap_modules([layer], prefix="model.layers.0")

    # Experts offloaded first
    assert _is_offloaded(layer.mlp.experts)
    # Dense parameters consume remaining budget in declaration order
    assert _is_offloaded(layer.input_layernorm.weight)
    assert _is_offloaded(layer.self_attn.qkv_proj.weight)
    assert not _is_offloaded(layer.self_attn.o_proj.weight)


def test_explicit_params_filter_overrides_heuristic():
    """Verify explicit cpu_offload_params overrides the default heuristic."""
    layer = MockMoELayer(dim=128)
    offloader = UVAOffloader(
        cpu_offload_max_bytes=10 * 1024 * 1024,
        cpu_offload_params={"self_attn"},
    )
    offloader.wrap_modules([layer], prefix="model.layers.0")

    # Only self_attn matches explicit filter
    assert _is_offloaded(layer.self_attn.qkv_proj.weight)
    assert _is_offloaded(layer.self_attn.o_proj.weight)
    assert not _is_offloaded(layer.mlp.experts)


def test_dense_model_offload():
    """Verify dense models continue offloading in standard order."""
    layer = MockDenseLayer(dim=128)
    qkv_bytes = (
        layer.self_attn.qkv_proj.weight.numel()
        * layer.self_attn.qkv_proj.weight.element_size()
    )

    offloader = UVAOffloader(cpu_offload_max_bytes=qkv_bytes + 512)
    offloader.wrap_modules([layer], prefix="model.layers.0")

    # Layernorm and QKV offloaded in declaration order
    assert _is_offloaded(layer.input_layernorm.weight)
    assert _is_offloaded(layer.self_attn.qkv_proj.weight)
    assert not _is_offloaded(layer.self_attn.o_proj.weight)
