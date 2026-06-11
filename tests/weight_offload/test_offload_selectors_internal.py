# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.model_executor.offloader.selectors import select_module_parameters


class _ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class _ToyExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class _ToyMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 2, bias=False)
        self.shared_expert_gate = nn.Linear(4, 1, bias=False)
        self.experts = _ToyExperts()
        self.shared_expert = _ToyMLP()


class _ToyPackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.w13_weight = nn.Parameter(nn.Linear(4, 8, bias=False).weight.unsqueeze(0))
        self.w2_weight = nn.Parameter(nn.Linear(8, 4, bias=False).weight.unsqueeze(0))
        self.w13_weight_scale = nn.Parameter(
            nn.Linear(4, 8, bias=False).weight[:1, :1].clone()
        )
        self.w2_weight_scale = nn.Parameter(
            nn.Linear(4, 4, bias=False).weight[:1, :1].clone()
        )
        self.w13_weight_scale_inv = nn.Parameter(
            nn.Linear(4, 8, bias=False).weight[:1, :1].clone()
        )
        self.w2_weight_scale_inv = nn.Parameter(
            nn.Linear(4, 4, bias=False).weight[:1, :1].clone()
        )


class _ToyPackedMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 2, bias=False)
        self.shared_expert_gate = nn.Linear(4, 1, bias=False)
        self.experts = _ToyPackedExperts()
        self.shared_expert = _ToyMLP()


class _ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _ToyAttention()
        self.mlp = _ToyMLP()
        self.block_sparse_moe = _ToyMoeBlock()


class _ToyPackedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _ToyAttention()
        self.mlp = _ToyMLP()
        self.block_sparse_moe = _ToyPackedMoeBlock()


class _ToyQwenMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _ToyPackedExperts()


class _ToyQwenLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _ToyQwenMlp()


def test_attention_selector():
    selected = select_module_parameters(_ToyLayer(), selectors={"attention"})

    assert set(selected) == {
        "self_attn.qkv_proj.weight",
        "self_attn.o_proj.weight",
    }


def test_dense_mlp_selector():
    selected = select_module_parameters(_ToyLayer(), selectors={"dense_mlp"})

    assert set(selected) == {
        "mlp.gate_up_proj.weight",
        "mlp.down_proj.weight",
    }


def test_routed_experts_selector():
    selected = select_module_parameters(_ToyLayer(), selectors={"routed_experts"})

    assert set(selected) == {
        "block_sparse_moe.experts.gate_up_proj.weight",
        "block_sparse_moe.experts.down_proj.weight",
    }


def test_shared_experts_selector():
    selected = select_module_parameters(_ToyLayer(), selectors={"shared_experts"})

    assert set(selected) == {
        "block_sparse_moe.shared_expert.gate_up_proj.weight",
        "block_sparse_moe.shared_expert.down_proj.weight",
    }


def test_selectors_union_with_include_names():
    selected = select_module_parameters(
        _ToyLayer(),
        selectors={"routed_experts"},
        include_names={"o_proj"},
    )

    assert set(selected) == {
        "self_attn.o_proj.weight",
        "block_sparse_moe.experts.gate_up_proj.weight",
        "block_sparse_moe.experts.down_proj.weight",
    }


def test_routed_experts_selector_matches_packed_moe_weights_only():
    selected = select_module_parameters(_ToyPackedLayer(), selectors={"routed_experts"})

    assert set(selected) == {
        "block_sparse_moe.experts.w13_weight",
        "block_sparse_moe.experts.w2_weight",
    }


def test_routed_experts_selector_excludes_qwen_fp8_scales():
    selected = select_module_parameters(_ToyQwenLayer(), selectors={"routed_experts"})

    assert set(selected) == {
        "mlp.experts.w13_weight",
        "mlp.experts.w2_weight",
    }
