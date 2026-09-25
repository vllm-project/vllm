# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4.1 output projection with a small-batch SM100/SM103 fusion."""

import torch
from torch import nn

from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj
from vllm.platforms import current_platform

_FUSED_WO_A_MAX_TOKENS = 32


def _can_fuse_wo_a(layer: nn.Module) -> bool:
    """Whether the attention layer matches the fused WO-A kernel's layout."""
    method = getattr(layer.wo_b, "scheme", layer.wo_b.quant_method)
    return (
        current_platform.is_device_capability_family(100)
        # WO-A is FP8 with per-32 scales, and WO-B takes MXFP8 input directly.
        and layer._einsum_recipe == (1, 1, 32)
        and get_input_quant_key(layer.wo_b) == kMxfp8Dynamic
        and isinstance(
            getattr(method, "kernel", None), FlashInferCutedslMxfp8LinearKernel
        )
        # A group's heads form one portable (<= 8 CTA) cluster.
        and layer.n_local_heads // layer.n_local_groups <= 8
        and (layer.nope_head_dim, layer.rope_head_dim) == (448, 64)
        and layer.o_lora_rank % 128 == 0
    )


def register_dsv41_o_proj_warmup(layer: nn.Module) -> None:
    """Warm the fused WO-A token counts ``dsv41_o_proj`` can dispatch."""
    # Draft models load outside the warmup registry; their layers share the
    # target's attention shapes, so the target's registration covers them too.
    if _can_fuse_wo_a(layer):
        from .fused_wo_a import _FUSED_WO_A_KERNEL

        _FUSED_WO_A_KERNEL.register_warmup(
            max_tokens=_FUSED_WO_A_MAX_TOKENS,
            n_groups=layer.n_local_groups,
            heads_per_group=layer.n_local_heads // layer.n_local_groups,
            o_lora_rank=layer.o_lora_rank,
        )


def dsv41_o_proj(
    layer: nn.Module, attn_out: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """``deep_gemm_fp8_o_proj`` with WO-A fused for small SM100/SM103 batches."""
    o = attn_out[:, : layer.n_local_heads, :]
    cos_sin_cache = layer.rotary_emb.cos_sin_cache
    if 1 <= o.shape[0] <= _FUSED_WO_A_MAX_TOKENS and _can_fuse_wo_a(layer):
        from .fused_wo_a import _FUSED_WO_A_KERNEL

        q, scales = _FUSED_WO_A_KERNEL(
            x=o,
            positions=positions,
            rope=cos_sin_cache,
            weight=layer.wo_a.weight,
            weight_scale=layer.wo_a.weight_scale,
        )
        return layer._wo_b_proj(
            QuantizedActivation(q, scales, o.dtype, q.shape, kMxfp8Dynamic)
        )
    return deep_gemm_fp8_o_proj(
        o,
        positions,
        cos_sin_cache,
        layer.wo_a,
        layer._wo_b_proj,
        n_groups=layer.n_local_groups,
        heads_per_group=layer.n_local_heads // layer.n_local_groups,
        nope_dim=layer.nope_head_dim,
        rope_dim=layer.rope_head_dim,
        o_lora_rank=layer.o_lora_rank,
        einsum_recipe=layer._einsum_recipe,
        tma_aligned_scales=layer._tma_aligned_scales,
    )
