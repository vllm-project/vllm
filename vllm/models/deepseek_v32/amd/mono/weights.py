# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Views and repacks of a loaded GLM-5 decoder layer in the MonoKernel's layout.

Experts are read in place (AITER-shuffled MXFP4, shared expert in the last
slot); attention projections and the dense MLP are repacked once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    FP8BlockParams,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.deepseek_v32.amd.model import DeepseekV32DecoderLayer
    from vllm.models.deepseek_v32.amd.ops.mono import LayerWeights


def split_kv_b(
    weight: torch.Tensor,
    scale: torch.Tensor,
    heads: int,
    nope_dim: int,
    v_dim: int,
    block: int = 128,
    uk_block_k: int = 64,
    uv_block_rows: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cut a block-FP8 ``kv_b_proj`` into the kernel's W_UK and W_UV.

    W_UK is ``[heads * kv_lora, nope_dim]`` (each head's nope rows transposed)
    with ``block x uk_block_k`` scales; W_UV is ``[heads * v_dim, kv_lora]`` with
    ``uv_block_rows x block`` scales. Each new scale block lies inside one
    checkpoint block, so the values are exact.
    """
    rows_per_head = nope_dim + v_dim
    kv_lora = weight.shape[1]
    w = weight.view(heads, rows_per_head, kv_lora)
    w_uk = w[:, :nope_dim].transpose(1, 2).reshape(heads * kv_lora, nope_dim)
    w_uv = w[:, nope_dim:].reshape(heads * v_dim, kv_lora)
    head = torch.arange(heads, device=scale.device)[:, None] * rows_per_head
    uk_rows = (
        head + torch.arange(0, nope_dim, uk_block_k, device=scale.device)
    ) // block
    uv_rows = (
        head + nope_dim + torch.arange(0, v_dim, uv_block_rows, device=scale.device)
    ) // block
    s_uk = scale[uk_rows].permute(0, 2, 1).reshape(heads * kv_lora // block, -1)
    s_uv = scale[uv_rows].reshape(heads * v_dim // uv_block_rows, -1)
    return (
        w_uk.contiguous(),
        s_uk.contiguous(),
        w_uv.contiguous(),
        s_uv.contiguous(),
    )


def _weight_preshuffled(linear: torch.nn.Module) -> bool:
    # The preshuffling kernel leaves these weights in the plain layout.
    if getattr(linear, "skip_weight_relayout", False) or getattr(
        linear, "is_bmm", False
    ):
        return False
    return any(
        getattr(getattr(method, "fp8_linear", None), "preshuffles_weight", False)
        for method in (
            getattr(linear, "quant_method", None),
            getattr(linear, "scheme", None),
        )
    )


def _block_fp8(linear: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.models.deepseek_v32.amd.ops.mono import unshuffle_linear_weight

    params = FP8BlockParams.from_layer(linear)
    scale = (
        params.weight_scale
        if params.weight_scale_inv is None
        else params.weight_scale_inv
    )
    weight = params.weight
    if (
        weight.dtype != current_platform.fp8_dtype()
        or scale is None
        or scale.dtype != torch.float32
        or scale.dim() != 2
    ):
        raise ValueError(
            "needs block-FP8 attention projections, "
            f"but {getattr(linear, 'prefix', linear)} is {weight.dtype}."
        )
    if _weight_preshuffled(linear):
        weight = unshuffle_linear_weight(weight)
    return weight.contiguous(), scale.contiguous()


def _computes_indices(layer: DeepseekV32DecoderLayer) -> bool:
    attn = layer.self_attn
    return attn.indexer is not None and not attn.skip_topk


def _indexer_weights(layer: DeepseekV32DecoderLayer) -> dict[str, torch.Tensor]:
    indexer = layer.self_attn.indexer
    wk_w = indexer.wk_weights_proj.weight
    if (
        wk_w.dtype != torch.bfloat16
        or wk_w.shape[0] != indexer.head_dim + indexer.n_head
    ):
        raise ValueError(
            f"needs a bf16 wk_weights_proj, got {wk_w.dtype} {tuple(wk_w.shape)}."
        )
    w_index_q, s_index_q = _block_fp8(indexer.wq_b)
    return {
        "w_index_k": wk_w[: indexer.head_dim].contiguous(),
        "w_index_w": wk_w[indexer.head_dim :].contiguous(),
        "w_index_q": w_index_q,
        "s_index_q": s_index_q,
        "g_index_k": indexer.k_norm.weight.float().contiguous(),
        "b_index_k": indexer.k_norm.bias.float().contiguous(),
    }


def _fp4_storage(tensor: torch.Tensor) -> torch.Tensor:
    view = tensor.view(torch.uint8)
    view.is_shuffled = True
    return view


def _moe_weights(moe: DeepseekV2MoE, cfg) -> tuple[dict[str, torch.Tensor], int]:
    experts = moe.experts.routed_experts
    physical = cfg.n_experts + cfg.num_shared_experts
    for name, weight, shape in (
        ("w13_weight", experts.w13_weight, (physical, 2 * cfg.inter, cfg.hidden // 2)),
        ("w2_weight", experts.w2_weight, (physical, cfg.hidden, cfg.inter // 2)),
    ):
        if (
            weight.element_size() != 1
            or tuple(weight.shape) != shape
            or not weight.is_contiguous()
        ):
            raise ValueError(
                f"needs {name} as contiguous MXFP4 bytes {shape} (the TP shard with "
                f"the shared expert last), got {weight.dtype} {tuple(weight.shape)}"
            )
    tensors = {
        "w_r": moe.gate.weight.to(torch.bfloat16).contiguous(),
        "bias": moe.gate.e_score_correction_bias.float().contiguous(),
        "w_ug": _fp4_storage(experts.w13_weight),
        "s_ug": experts.w13_weight_scale.view(torch.uint8),
        "w_dn": _fp4_storage(experts.w2_weight),
        "s_dn": experts.w2_weight_scale.view(torch.uint8),
    }
    return tensors, physical


def _mxfp4_rows(linear: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    weight = linear.weight.view(torch.uint8)
    rows, half_k = weight.shape
    scale = linear.weight_scale.view(torch.uint8)
    if scale.shape != (half_k * 2 // 32, rows):
        raise ValueError(
            "needs the dense MLP in the row-major "
            "MXFP4 layout of the AITER Triton GEMM, but "
            f"{getattr(linear, 'prefix', linear)} has weight_scale "
            f"{tuple(scale.shape)}."
        )
    return weight, scale.T.contiguous()


def _dense_mlp_weights(
    mlp: torch.nn.Module, cfg
) -> tuple[dict[str, torch.Tensor], int]:
    from vllm.models.deepseek_v32.amd.ops.mono import pack_dense_mlp

    gate_up = _mxfp4_rows(mlp.gate_up_proj)
    slices = gate_up[0].shape[0] // (2 * cfg.inter)
    if gate_up[0].shape[0] != 2 * cfg.inter * slices or not 0 < slices <= cfg.moe_slots:
        raise ValueError(
            "needs the dense MLP width to be a "
            f"multiple of the expert width {cfg.inter}, at most {cfg.moe_slots} "
            f"times, got {gate_up[0].shape[0] // 2}."
        )
    tensors = pack_dense_mlp(*gate_up, *_mxfp4_rows(mlp.down_proj), slices)
    device = gate_up[0].device
    tensors["w_r"] = torch.zeros(16, cfg.hidden, dtype=torch.bfloat16, device=device)
    tensors["bias"] = torch.zeros(cfg.n_experts, dtype=torch.float32, device=device)
    return tensors, slices


def _layer_weights(
    layer: DeepseekV32DecoderLayer,
    cfg,
    rank: int,
    tp: int,
    with_indexer: bool = False,
    dense: bool = False,
) -> LayerWeights:
    from vllm.models.deepseek_v32.amd.ops.mono import (
        LayerWeights,
        Mxfp4ScaleLayout,
        Mxfp4WeightLayout,
    )

    attn = layer.self_attn
    if dense:
        mlp, physical = _dense_mlp_weights(layer.mlp, cfg)
    else:
        mlp, physical = _moe_weights(layer.mlp, cfg)
    w_qkv_a, s_qkv_a = _block_fp8(attn.fused_qkv_a_proj)
    w_q_b, s_q_b = _block_fp8(attn.q_b_proj)
    w_o, s_o = _block_fp8(attn.o_proj)
    w_uk, s_uk, w_uv, s_uv = split_kv_b(
        *_block_fp8(attn.kv_b_proj), cfg.local_heads, cfg.nope_dim, cfg.v_dim
    )
    tensors = {
        "g_in": layer.input_layernorm.weight,
        "g_q": attn.q_a_layernorm.weight,
        "g_kv": attn.kv_a_layernorm.weight,
        "g_post": layer.post_attention_layernorm.weight,
        "w_qkv_a": w_qkv_a,
        "s_qkv_a": s_qkv_a,
        "w_q_b": w_q_b,
        "s_q_b": s_q_b,
        "w_uk": w_uk,
        "s_uk": s_uk,
        "w_uv": w_uv,
        "s_uv": s_uv,
        "w_o": w_o,
        "s_o": s_o,
        **mlp,
    }
    if with_indexer:
        tensors.update(_indexer_weights(layer))
    return LayerWeights(
        cfg.local_heads,
        tensors,
        cfg,
        rank,
        tp,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=physical,
    )
