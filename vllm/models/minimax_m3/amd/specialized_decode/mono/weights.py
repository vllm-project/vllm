# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checked weight views and projection packing for the full-layer kernel."""

from dataclasses import dataclass

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    RowWiseTorchFP8ScaledMMLinearKernel,
)

from ..runtime.weights import MonoLayerWeights
from .config import HEAD_DIM, HIDDEN, O_K, ONE_INDEX_HEAD, ROTARY_DIM


def _projection(linear, rows, cols):
    method = getattr(linear.quant_method, "fp8_linear", None)
    weight, scale = linear.weight, getattr(linear, "weight_scale", None)
    if (
        weight.dtype != torch.float8_e4m3fn
        or scale is None
        or scale.dtype != torch.float32
        or scale.numel() != rows
    ):
        raise ValueError("full-layer decode requires dynamic per-token FP8 projections")
    if isinstance(method, RowWiseTorchFP8ScaledMMLinearKernel):
        if tuple(weight.shape) != (cols, rows):
            raise ValueError("unexpected unpacked FP8 projection shape")
        weight = rocm_aiter_ops.shuffle_weight(weight.t().contiguous())
    elif isinstance(method, AiterPreshuffledPerTokenFp8ScaledMMLinearKernel):
        if weight.numel() != rows * cols or not weight.is_contiguous():
            raise ValueError("unexpected packed FP8 projection size")
    else:
        raise ValueError(f"unsupported FP8 projection kernel: {type(method).__name__}")
    return weight.detach(), scale.detach().view(rows)


@dataclass(frozen=True)
class LayerWeights:
    ffn: MonoLayerWeights
    input_norm: torch.Tensor
    qkv: torch.Tensor
    qkv_scale: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    index_q_norm: torch.Tensor
    index_k_norm: torch.Tensor
    cos_sin: torch.Tensor
    output: torch.Tensor
    output_scale: torch.Tensor
    router_bf16: torch.Tensor

    @classmethod
    def from_layer(cls, layer, shared_tables: dict[int, torch.Tensor]):
        attn = layer.self_attn
        if (attn.num_heads, attn.num_kv_heads, attn.num_idx_heads, attn.head_dim) != (
            16,
            1,
            1,
            HEAD_DIM,
        ) or attn.rotary_emb.rotary_dim != ROTARY_DIM:
            raise ValueError("unsupported full-layer attention dimensions")
        qkv, qkv_scale = _projection(attn.qkv_proj, ONE_INDEX_HEAD.rows, HIDDEN)
        output, output_scale = _projection(attn.o_proj, HIDDEN, O_K)
        table = attn.rotary_emb.cos_sin_cache
        if table.ndim != 2 or table.shape[1] != ROTARY_DIM:
            raise ValueError("unexpected rotary table shape")
        if table.data_ptr() not in shared_tables:
            shared_tables[table.data_ptr()] = table.to(torch.bfloat16).contiguous()
        ffn = MonoLayerWeights.from_layer(layer)
        return cls(
            ffn,
            layer.input_layernorm.weight.detach(),
            qkv,
            qkv_scale,
            attn.q_norm.weight.detach(),
            attn.k_norm.weight.detach(),
            attn.index_q_norm.weight.detach(),
            attn.index_k_norm.weight.detach(),
            shared_tables[table.data_ptr()],
            output,
            output_scale,
            ffn.router.to(torch.bfloat16),
        )
