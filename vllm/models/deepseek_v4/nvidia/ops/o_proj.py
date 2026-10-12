# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
import torch.nn as nn

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum, get_tma_aligned_size


def compute_fp8_einsum_recipe(
    block_size: int = 128,
) -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90 keeps block-row FP32 scales. SM100 uses packed per-row E8M0 scales.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, block_size)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor | QuantizedActivation,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + grouped wo_a + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. The attention
    layer selects the recipe at initialization. ``wo_b`` is any callable over
    the flattened ``z``: the projection module itself, or a wrapper that also
    reduce-scatters its output (DeepSeek-V4.1 GEMM-RS).

    A QuantizedActivation ``o`` was already rotated and cast by the attention
    kernel, one ``wo_a`` group per slot with the live groups first, so only
    the padding groups are sliced off.
    """
    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    if isinstance(o, QuantizedActivation):
        assert use_fp8
        o_proj_input, o_scale = o.data[:, :n_groups], o.scale[:, :n_groups]
    else:
        o_proj_input, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            quant_group_size=einsum_recipe[2],
            tma_aligned_scales=tma_aligned_scales,
            quantize=use_fp8,
        )
    z = torch.empty(
        (o_proj_input.shape[0], n_groups, o_lora_rank),
        device=o_proj_input.device,
        dtype=torch.bfloat16,
    )
    if use_fp8:
        weight_scale = (
            wo_a.weight_scale
            if hasattr(wo_a, "weight_scale")
            else wo_a.weight_scale_inv
        )
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_proj_input, o_scale),
            (wo_a.weight, weight_scale),
            z,
            recipe=einsum_recipe,
        )
    else:
        grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
        torch.bmm(
            o_proj_input.transpose(0, 1),
            grouped_weight.transpose(1, 2),
            out=z.transpose(0, 1),
        )
    return wo_b(z.flatten(1))


# TRTLLM-gen's DSv4 RopeQuant epilogue (flashinfer-ai/flashinfer#4918) has one
# fixed schedule: 128 query heads in groups of 8 512-wide heads, and FP8 with
# one packed UE8M0 scale per 128 elements.
_ROPE_QUANT_NUM_HEADS = 128
_ROPE_QUANT_HEADS_PER_GROUP = 8
_ROPE_QUANT_HEAD_DIM = 512
_ROPE_QUANT_BLOCK = 128


def rope_quant_unsupported_reason(layer: torch.nn.Module) -> str | None:
    """Why ``layer`` cannot take FlashInfer's fused inverse RoPE + FP8 output.

    DSv4 always has 8 heads per ``wo_a`` group at any TP size, and the rest of
    the kernel's fixed shape (head/RoPE dims, per-128 einsum scales, FP32 RoPE
    cache) is what the SM100 layer uses anyway. What varies is the local head
    count and the configs the layer also serves unquantized.
    """
    if layer.n_local_heads != _ROPE_QUANT_NUM_HEADS:
        # Padding fewer heads up to the kernel's fixed 128 costs more
        # attention than the fused epilogue saves.
        return f"RopeQuant needs {_ROPE_QUANT_NUM_HEADS} local query heads"
    if layer.kv_cache_torch_dtype != torch.float8_e4m3fn:
        return "RopeQuant needs the per-tensor FP8 KV cache"
    if layer.wo_a.weight.dtype != torch.float8_e4m3fn:
        return "RopeQuant needs an FP8 wo_a"
    return None


def rope_quant_attn_out(num_tokens: int, device: torch.device) -> QuantizedActivation:
    """Allocate the output pair the RopeQuant kernel writes.

    Values are group-major, ``[num_tokens, 16, 8 * 512]`` over physical
    ``[group, token, K]``. Scales are MN-major packed UE8M0, one int32 per
    head over its four 128-wide blocks. Both are what ``fp8_einsum`` consumes.
    """
    n_groups = _ROPE_QUANT_NUM_HEADS // _ROPE_QUANT_HEADS_PER_GROUP
    group_width = _ROPE_QUANT_HEADS_PER_GROUP * _ROPE_QUANT_HEAD_DIM
    data = torch.empty(
        (n_groups, num_tokens, group_width), dtype=torch.float8_e4m3fn, device=device
    ).transpose(0, 1)
    aligned = get_tma_aligned_size(num_tokens, torch.int32.itemsize)
    scale = torch.empty(
        (n_groups, group_width // (_ROPE_QUANT_BLOCK * 4), aligned),
        dtype=torch.int32,
        device=device,
    ).permute(2, 0, 1)[:num_tokens]
    return QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=data.shape,
        quant_key=kFp8Dynamic128Sym,
    )
