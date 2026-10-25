# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_ROCM_DSV4_BPRESHUFFLE=0 keeps DeepSeek-V4's FP8 attention linears loadable.

The B-preshuffle rewrites the loaded weights into aiter's shuffled layout once,
after loading. A later ``load_weights`` (an RL weight sync) writes checkpoint-layout
data into those parameters while the forward keeps reading them as shuffled.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v4.amd import rocm as dsv4_rocm


def _block_fp8_linear(out_features: int, in_features: int) -> nn.Module:
    linear = nn.Module()
    linear.weight = nn.Parameter(
        torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    linear.weight_scale_inv = nn.Parameter(
        torch.ones(out_features // 128, in_features // 128), requires_grad=False
    )
    return linear


def _attention() -> SimpleNamespace:
    return SimpleNamespace(
        fused_wqa_wkv=_block_fp8_linear(256, 512),
        wo_b=_block_fp8_linear(512, 256),
        _wqa_wkv_scale=None,
        _wo_b_scale=None,
    )


@pytest.fixture
def aiter(monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", classmethod(lambda cls: True))
    # A recognizable "shuffled" layout: the test weights are all zeros.
    monkeypatch.setattr(
        rocm_aiter_ops,
        "shuffle_weight",
        staticmethod(lambda w, layout: torch.ones_like(w)),
    )
    monkeypatch.setattr(dsv4_rocm, "on_gfx1250", lambda: False)
    monkeypatch.setattr(dsv4_rocm, "_ON_GFX950", False)


def test_attention_weights_are_preshuffled_by_default(aiter, monkeypatch):
    monkeypatch.setattr(dsv4_rocm.envs, "VLLM_ROCM_DSV4_BPRESHUFFLE", True)
    attn = _attention()
    loaded = (attn.fused_wqa_wkv.weight, attn.wo_b.weight)

    dsv4_rocm.DeepseekV4ROCMAiterMLAAttention.prepare_attn_preshuffle(attn)

    for linear, param in zip((attn.fused_wqa_wkv, attn.wo_b), loaded):
        assert linear.weight is param
        assert linear.weight.float().eq(1).all()
    assert attn._wqa_wkv_scale is attn.fused_wqa_wkv.weight_scale_inv
    assert attn._wo_b_scale is attn.wo_b.weight_scale_inv


def test_attention_weights_stay_loadable_without_preshuffle(aiter, monkeypatch):
    monkeypatch.setattr(dsv4_rocm.envs, "VLLM_ROCM_DSV4_BPRESHUFFLE", False)
    attn = _attention()
    loaded = (attn.fused_wqa_wkv.weight, attn.wo_b.weight)

    dsv4_rocm.DeepseekV4ROCMAiterMLAAttention.prepare_attn_preshuffle(attn)

    # The weights keep the checkpoint layout a reload writes, and the forward
    # takes the non-preshuffled GEMM because the scales stay unset.
    for linear, param in zip((attn.fused_wqa_wkv, attn.wo_b), loaded):
        assert linear.weight is param
        assert linear.weight.float().eq(0).all()
    assert attn._wqa_wkv_scale is None
    assert attn._wo_b_scale is None
