# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.minimax_m3.amd.mono.weights import (
    EXPERTS,
    HIDDEN,
    INTER,
    MonoLayerWeights,
)


def _parameter(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="meta")


def _layer():
    w13 = _parameter((EXPERTS, 2 * INTER, HIDDEN // 2), torch.float4_e2m1fn_x2)
    w2 = _parameter((EXPERTS, HIDDEN, INTER // 2), torch.float4_e2m1fn_x2)
    w13.is_shuffled = True
    w2.is_shuffled = True
    experts = SimpleNamespace(
        top_k=4,
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=_parameter((EXPERTS * 2 * INTER, HIDDEN // 32), torch.uint8),
        w2_weight_scale=_parameter((EXPERTS * HIDDEN, INTER // 32), torch.uint8),
    )
    moe = SimpleNamespace(
        is_fused_shared_expert_enabled=True,
        use_aiter_moe_fse=True,
        experts=SimpleNamespace(routed_experts=experts),
        gate=SimpleNamespace(weight=_parameter((128, HIDDEN), torch.float32)),
        e_score_correction_bias=_parameter((128,), torch.float32),
    )
    attn = SimpleNamespace(
        qkv_proj=SimpleNamespace(weight=_parameter((2560, HIDDEN), torch.bfloat16)),
        o_proj=SimpleNamespace(weight=_parameter((HIDDEN, 2048), torch.bfloat16)),
        q_norm=SimpleNamespace(weight=_parameter((128,), torch.bfloat16)),
        k_norm=SimpleNamespace(weight=_parameter((128,), torch.bfloat16)),
        index_q_norm=SimpleNamespace(weight=_parameter((128,), torch.bfloat16)),
        index_k_norm=SimpleNamespace(weight=_parameter((128,), torch.bfloat16)),
    )
    return SimpleNamespace(
        layer_id=3,
        self_attn=attn,
        block_sparse_moe=moe,
        input_layernorm=SimpleNamespace(weight=_parameter((HIDDEN,), torch.bfloat16)),
        post_attention_layernorm=SimpleNamespace(
            weight=_parameter((HIDDEN,), torch.bfloat16)
        ),
    )


def test_weight_views_preserve_checkpoint_formats():
    layer = _layer()
    weights = MonoLayerWeights.from_layer(layer)
    assert (
        weights.qkv.untyped_storage()._cdata
        == layer.self_attn.qkv_proj.weight.untyped_storage()._cdata
    )
    assert (
        weights.output.untyped_storage()._cdata
        == layer.self_attn.o_proj.weight.untyped_storage()._cdata
    )
    assert (
        weights.router.untyped_storage()._cdata
        == layer.block_sparse_moe.gate.weight.untyped_storage()._cdata
    )
    assert weights.router.dtype == torch.float32
    assert weights.w13.dtype == torch.float4_e2m1fn_x2


def test_bf16_router_is_refused():
    layer = _layer()
    layer.block_sparse_moe.gate.weight = _parameter((128, HIDDEN), torch.bfloat16)
    with pytest.raises(ValueError, match="router"):
        MonoLayerWeights.from_layer(layer)
