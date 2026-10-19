# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterMxfp4Experts,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.models.minimax_m3.amd.specialized_decode.runtime.weights import (
    EXPERTS,
    HIDDEN,
    INTER,
    EffectiveActivationMode,
    MonoLayerWeights,
    effective_activation_mode,
)


def _parameter(shape, dtype, *, meta=True):
    device = "meta" if meta else "cpu"
    return torch.empty(shape, dtype=dtype, device=device)


def _layer():
    w13 = _parameter((EXPERTS, 2 * INTER, HIDDEN // 2), torch.float4_e2m1fn_x2)
    w2 = _parameter((EXPERTS, HIDDEN, INTER // 2), torch.float4_e2m1fn_x2)
    w13.is_shuffled = True
    w2.is_shuffled = True
    fused_experts = object.__new__(AiterMxfp4Experts)
    fused_experts.moe_config = SimpleNamespace(
        use_mori_kernels=False,
        use_mxfp4_w4a4_dsv4=False,
    )
    experts = SimpleNamespace(
        top_k=4,
        renormalize=True,
        scoring_func="sigmoid",
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=7.0,
        quant_method=SimpleNamespace(
            mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
            moe_kernel=SimpleNamespace(fused_experts=fused_experts),
        ),
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=_parameter((EXPERTS, 2 * INTER, HIDDEN // 32), torch.uint8),
        w2_weight_scale=_parameter((EXPERTS, HIDDEN, INTER // 32), torch.uint8),
    )
    moe = SimpleNamespace(
        is_fused_shared_expert_enabled=True,
        use_aiter_moe_fse=True,
        experts=SimpleNamespace(routed_experts=experts),
        gate=SimpleNamespace(weight=_parameter((128, HIDDEN), torch.float32)),
        e_score_correction_bias=_parameter((128,), torch.float32),
        routed_scaling_factor=2.0,
        n_shared_experts=1,
    )
    qkv = type("MinimaxM3QKVParallelLinearWithIndexer", (), {})()
    qkv.weight = _parameter((2560, HIDDEN), torch.bfloat16, meta=False)
    output = type("RowParallelLinear", (), {})()
    output.weight = _parameter((HIDDEN, 2048), torch.bfloat16, meta=False)
    attn = SimpleNamespace(
        qkv_proj=qkv,
        o_proj=output,
        q_norm=SimpleNamespace(weight=_parameter((128,), torch.bfloat16)),
        k_norm=SimpleNamespace(weight=_parameter((128,), torch.bfloat16)),
        index_q_norm=SimpleNamespace(weight=_parameter((128,), torch.bfloat16)),
        index_k_norm=SimpleNamespace(weight=_parameter((128,), torch.bfloat16)),
    )
    input_norm = type("MiniMAXGemmaRMSNorm", (), {})()
    input_norm.weight = _parameter((HIDDEN,), torch.bfloat16)
    input_norm.variance_epsilon = 1e-6
    post_norm = type("MiniMAXGemmaRMSNorm", (), {})()
    post_norm.weight = _parameter((HIDDEN,), torch.bfloat16)
    post_norm.variance_epsilon = 1e-6
    return SimpleNamespace(
        layer_id=3,
        self_attn=attn,
        block_sparse_moe=moe,
        input_layernorm=input_norm,
        post_attention_layernorm=post_norm,
    )


def test_weight_views_preserve_checkpoint_formats():
    layer = _layer()
    layer.block_sparse_moe.gate.weight = _parameter(
        (128, HIDDEN), torch.float32, meta=False
    )
    weights = MonoLayerWeights.from_layer(layer, require_cuda=False)
    assert weights.qkv.data_ptr() == layer.self_attn.qkv_proj.weight.data_ptr()
    assert weights.output.data_ptr() == layer.self_attn.o_proj.weight.data_ptr()
    assert weights.router.data_ptr() == layer.block_sparse_moe.gate.weight.data_ptr()
    assert weights.router.dtype == torch.float32
    assert weights.w13.dtype == torch.float4_e2m1fn_x2
    assert weights.routed_scale == 2.0
    assert weights.shared_weight == 1.0


def test_effective_activation_mode_is_explicitly_bf16():
    experts = _layer().block_sparse_moe.experts.routed_experts
    assert effective_activation_mode(experts) is EffectiveActivationMode.BF16

    experts.quant_method.moe_kernel.fused_experts.moe_config.use_mori_kernels = True
    assert effective_activation_mode(experts) is EffectiveActivationMode.MXFP4


def test_other_mxfp4_shuffle_is_refused():
    layer = _layer()
    quant_method = layer.block_sparse_moe.experts.routed_experts.quant_method
    quant_method.mxfp4_backend = Mxfp4MoeBackend.AITER_MXFP4_BF16
    with pytest.raises(ValueError, match="AITER_MXFP4_MXFP4"):
        MonoLayerWeights.from_layer(layer, require_cuda=False)


def test_bf16_router_is_refused():
    layer = _layer()
    layer.block_sparse_moe.gate.weight = _parameter((128, HIDDEN), torch.bfloat16)
    with pytest.raises(ValueError, match="router"):
        MonoLayerWeights.from_layer(layer, require_cuda=False)


def test_routed_scale_is_independent_of_shared_weight():
    layer = _layer()
    layer.block_sparse_moe.routed_scaling_factor = 1.0
    with pytest.raises(ValueError, match="routed scale"):
        MonoLayerWeights.from_layer(layer, require_cuda=False)


def test_noncontiguous_router_is_refused():
    layer = _layer()
    layer.block_sparse_moe.gate.weight = torch.empty(
        HIDDEN, 128, dtype=torch.float32
    ).t()
    with pytest.raises(ValueError, match="router"):
        MonoLayerWeights.from_layer(layer, require_cuda=False)
