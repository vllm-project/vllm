# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Describe native tensors and verify backend semantics for the ATOM library."""

import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"MiniMax-M3 ATOM mono: {message}")


def _layer_specs(layer, cfg):
    attn = layer.self_attn
    moe = layer.block_sparse_moe
    experts = moe.experts.routed_experts
    label = f"layer {layer.layer_id}"
    require(
        moe.gate.weight.dtype == torch.bfloat16 and moe.gate.out_dtype == torch.float32,
        f"{label}: requires BF16 router weights and FP32 logits",
    )
    w_qkv, s_qkv = _projection_weights(attn.qkv_proj)
    w_o, s_o = _projection_weights(attn.o_proj)
    require(
        (attn.num_heads, attn.num_kv_heads, attn.num_idx_heads, attn.head_dim)
        == (16, 1, 1, 128),
        f"{label}: unsupported Q/KV/index heads",
    )
    require(
        attn.use_aiter_sparse_pa
        and not attn.skip_index_topk
        and attn.rotary_emb.rotary_dim == 64
        and attn.kv_cache_dtype == "fp8",
        f"{label}: attention/cache/RoPE configuration",
    )
    require(
        moe.is_fused_shared_expert_enabled
        and moe.use_aiter_moe_fse
        and moe.shared_experts is None
        and experts.w13_weight.shape[0] == 129,
        f"{label}: enable VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1",
    )
    require(
        str(getattr(experts.quant_method, "mxfp4_backend", None)).endswith(
            "AITER_MXFP4_MXFP4"
        )
        and getattr(experts.w13_weight, "is_shuffled", False)
        and getattr(experts.w2_weight, "is_shuffled", False),
        f"{label}: expected shuffled AITER MXFP4 expert weights",
    )

    k, v = attn.get_aiter_sparse_pa_kv_cache()
    require(
        attn._aiter_sparse_pa_block_page_stride == 16, f"{label}: native page stride"
    )
    from atom.models.minimax_m3.mono.library import (
        CacheSpec,
        LayerSpec,
        validate_layer,
    )

    spec = LayerSpec(
        layer_id=layer.layer_id,
        g_in=layer.input_layernorm.weight,
        w_qkv=w_qkv,
        s_qkv=s_qkv,
        g_q=attn.q_norm.weight,
        g_k=attn.k_norm.weight,
        g_iq=attn.index_q_norm.weight,
        g_ik=attn.index_k_norm.weight,
        cos_sin=attn.rotary_emb.cos_sin_cache,
        w_o=w_o,
        s_o=s_o,
        g_post=layer.post_attention_layernorm.weight,
        gate=moe.gate.weight,
        bias=moe.e_score_correction_bias,
        w13=experts.w13_weight,
        s13=experts.w13_weight_scale,
        w2=experts.w2_weight,
        s2=experts.w2_weight_scale,
        eps=cfg.rms_norm_eps,
        route_scale=cfg.routed_scaling_factor,
        swiglu_limit=cfg.swiglu_limit,
    )
    validate_layer(spec, w_qkv.device)
    cache = CacheSpec(
        layer_id=layer.layer_id,
        main=attn.kv_cache,
        k=k,
        v=v,
        index=attn.indexer.index_cache.kv_cache,
        k_scale=attn._k_scale,
        v_scale=attn._v_scale,
    )
    return spec, cache


def _projection_weights(linear):
    from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
        AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
        AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
    )

    kernel = getattr(linear.quant_method, "fp8_linear", None)
    require(
        type(kernel)
        in (
            AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
            AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
        ),
        "projections require an AITER preshuffled per-channel FP8 backend",
    )
    weight = linear.weight
    if isinstance(kernel, AiterHipbMMPerTokenFp8ScaledMMLinearKernel):
        weight = weight.t()
    scale = linear.weight_scale
    require(
        weight.dtype == torch.float8_e4m3fn
        and weight.is_contiguous()
        and scale.dtype == torch.float32
        and scale.is_contiguous(),
        "projections require contiguous E4M3 weights and FP32 channel scales",
    )
    return weight, scale.view(-1)


def layer_specs(model, layer_ids):
    layers, caches = zip(
        *[_layer_specs(model.layers[i], model.config) for i in layer_ids]
    )
    return list(layers), list(caches)
