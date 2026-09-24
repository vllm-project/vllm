# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4.1 output projection with a small-batch SM100 fusion."""

from collections.abc import Callable

import torch
from torch import nn

from vllm.config import get_current_vllm_config
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
from vllm.utils.flashinfer import has_flashinfer_cutedsl

# GB200: gains hold through 24 tokens; larger graphs have marginal benefit.
_FUSED_WO_A_MAX_TOKENS = 24


def _can_fuse_wo_a(
    wo_b_layer: nn.Module,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> bool:
    method = getattr(wo_b_layer, "scheme", getattr(wo_b_layer, "quant_method", None))
    return (
        (n_groups, heads_per_group, nope_dim, rope_dim, o_lora_rank)
        == (2, 8, 448, 64, 1024)
        and einsum_recipe == (1, 1, 32)
        and tma_aligned_scales
        and current_platform.is_device_capability_family(100)
        and has_flashinfer_cutedsl()
        and get_input_quant_key(wo_b_layer) == kMxfp8Dynamic
        and isinstance(
            getattr(method, "kernel", None), FlashInferCutedslMxfp8LinearKernel
        )
    )


def register_dsv41_o_proj_warmup(layer: nn.Module) -> None:
    """Warm the fused WO-A token counts ``dsv41_o_proj`` can dispatch."""
    vllm_config = get_current_vllm_config()
    if vllm_config.kernel_config.enable_jit_warmup and _can_fuse_wo_a(
        layer.wo_b,
        layer.n_local_groups,
        layer.n_local_heads // layer.n_local_groups,
        layer.nope_head_dim,
        layer.rope_head_dim,
        layer.o_lora_rank,
        layer._einsum_recipe,
        layer._tma_aligned_scales,
    ):
        from .fused_wo_a import _FUSED_WO_A_KERNEL

        _FUSED_WO_A_KERNEL.register_warmup(
            vllm_config,
            max_tokens=_FUSED_WO_A_MAX_TOKENS,
            x_stride=layer.padded_heads * layer.head_dim,
        )


def dsv41_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: Callable,
    *,
    wo_b_layer: nn.Module,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """Fuse the BF16 attention-output path; preserve other projection paths."""
    ws = getattr(wo_a, "weight_scale", None)
    if (
        1 <= o.shape[0] <= _FUSED_WO_A_MAX_TOKENS
        and ws is not None
        and _can_fuse_wo_a(
            wo_b_layer,
            n_groups,
            heads_per_group,
            nope_dim,
            rope_dim,
            o_lora_rank,
            einsum_recipe,
            tma_aligned_scales,
        )
        and (o.dtype, positions.dtype, cos_sin_cache.dtype, wo_a.weight.dtype, ws.dtype)
        == (
            torch.bfloat16,
            torch.int64,
            torch.float32,
            torch.float8_e4m3fn,
            torch.int32,
        )
        and (o.shape[1:], cos_sin_cache.shape[1:], wo_a.weight.shape, ws.shape)
        == ((16, 512), (64,), (2, 1024, 4096), (2, 1024, 32))
        and o.stride()[1:] == (512, 1)
        and o.stride(0) % 8 == 0
        and o.storage_offset() % 8 == 0
        and positions.stride() == (1,)
        and cos_sin_cache.is_contiguous()
        and wo_a.weight.is_contiguous()
        and ws.stride() == (32768, 1, 1024)
    ):
        from .fused_wo_a import _FUSED_WO_A_KERNEL

        q, scales = _FUSED_WO_A_KERNEL(
            x=o,
            positions=positions,
            rope=cos_sin_cache,
            weight=wo_a.weight,
            weight_scale=ws,
        )
        return wo_b(
            QuantizedActivation(
                q, scales, o.dtype, torch.Size((o.shape[0], 2048)), kMxfp8Dynamic
            )
        )
    return deep_gemm_fp8_o_proj(
        o,
        positions,
        cos_sin_cache,
        wo_a,
        wo_b,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        o_lora_rank=o_lora_rank,
        einsum_recipe=einsum_recipe,
        tma_aligned_scales=tma_aligned_scales,
    )
