# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use the ordinary path's loaded FP8 tensors without requantization."""


def layer_weights(layer):
    import torch

    attn, mlp = layer.self_attn, layer.mlp
    prefix = [
        layer.input_layernorm.weight,
        layer.post_attention_layernorm.weight,
        attn.qkv_proj.weight,
        attn.qkv_proj.weight_scale,
        attn.o_proj.weight,
        attn.o_proj.weight_scale,
    ]
    if hasattr(mlp, "experts"):
        routed = mlp.experts.routed_experts
        shared = mlp.shared_experts
        return prefix + [
            mlp.gate.weight,
            mlp.gate.e_score_correction_bias,
            routed.w13_weight,
            routed.w13_weight_scale,
            routed.w2_weight,
            routed.w2_weight_scale,
            shared.gate_up_proj.weight,
            shared.gate_up_proj.weight_scale,
            shared.down_proj.weight,
            shared.down_proj.weight_scale,
        ]
    empty = prefix[0].new_empty(0, dtype=torch.float32)
    return prefix + [
        empty,
        empty,
        mlp.gate_up_proj.weight,
        mlp.gate_up_proj.weight_scale,
        mlp.down_proj.weight,
        mlp.down_proj.weight_scale,
        empty,
        empty,
        empty,
        empty,
    ]


def cache_tensor(layer):
    cache = layer.self_attn.attn.kv_cache
    return cache[0] if isinstance(cache, (tuple, list)) else cache


def tiled_qkv_weight(weight):
    """Reorder raw FP8 bytes into contiguous 16-output by 16-K tiles."""
    import torch

    rows, columns = weight.t().shape
    raw = weight.t().view(torch.uint8)
    tiled = raw.reshape(rows // 16, 16, columns // 16, 16).permute(0, 2, 1, 3)
    return tiled.contiguous().reshape(rows, columns).view(weight.dtype).t()


def tiled_qkv_supported(ops, tp, dtype):
    """Each dtype opts in only to its exact native byte-layout version."""
    import torch

    if dtype == torch.bfloat16:
        if tp != 2:
            return False
        version = getattr(ops, "qkv_layout_version", None)
    elif dtype == torch.float16:
        supported = getattr(ops, "fp16_qkv_supported_tps", None)
        supported_tps = supported() if supported is not None else (2,)
        if tp not in supported_tps:
            return False
        version = getattr(ops, "fp16_qkv_layout_version", None)
    else:
        return False
    return version is not None and version() == 1
