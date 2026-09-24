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
from vllm.utils.flashinfer import has_flashinfer_dsv4_rope_quant


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


def _wo_a_weight_scale(wo_a: nn.Module) -> torch.Tensor:
    return wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
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
    """
    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
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
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    if use_fp8:
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_proj_input, o_scale),
            (wo_a.weight, _wo_a_weight_scale(wo_a)),
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


def deep_gemm_prequantized_o_proj(
    attn_out: QuantizedActivation,
    wo_a: nn.Module,
    wo_b: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_groups: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
) -> torch.Tensor:
    """O projection over an attention output the kernel already rotated and cast.

    ``attn_out`` holds one ``wo_a`` group per slot along dim 1; the live groups
    come first, so the padding groups are sliced off here.
    """
    z = torch.empty(
        (attn_out.data.shape[0], n_groups, o_lora_rank),
        device=attn_out.data.device,
        dtype=torch.bfloat16,
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (attn_out.data[:, :n_groups], attn_out.scale[:, :n_groups]),
        (wo_a.weight, _wo_a_weight_scale(wo_a)),
        z,
        recipe=einsum_recipe,
    )
    return wo_b(z.flatten(1))


# TRTLLM-gen's DSv4 RopeQuant epilogue (flashinfer-ai/flashinfer#4918) has one
# fixed schedule: 128 query heads in groups of 8, the inverse GPT-J RoPE on the
# last 64 dims of each 512-wide head, and FP8 with one packed UE8M0 scale per
# 128 elements.
_ROPE_QUANT_NUM_HEADS = 128
_ROPE_QUANT_HEADS_PER_GROUP = 8
_ROPE_QUANT_HEAD_DIM = 512
_ROPE_QUANT_ROPE_DIM = 64
_ROPE_QUANT_BLOCK = 128


def rope_quant_unsupported_reason(
    layer: torch.nn.Module,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> str | None:
    """Why ``layer`` cannot take FlashInfer's fused inverse RoPE + FP8 output.

    The kernel emits ``wo_a``'s input directly, so everything it fixes has to
    match what the unfused ``deep_gemm_fp8_o_proj`` would have produced.
    """
    if not has_flashinfer_dsv4_rope_quant():
        return "FlashInfer predates DSv4 RopeQuant"
    if layer.kv_cache_torch_dtype != torch.float8_e4m3fn:
        return "RopeQuant needs the per-tensor FP8 KV cache"
    if layer.n_local_heads != _ROPE_QUANT_NUM_HEADS:
        # Padding fewer heads up to the kernel's fixed 128 costs more
        # attention than the fused epilogue saves.
        return f"RopeQuant needs {_ROPE_QUANT_NUM_HEADS} local query heads"
    if layer.head_dim != _ROPE_QUANT_HEAD_DIM or (
        layer.rope_head_dim != _ROPE_QUANT_ROPE_DIM
    ):
        return "RopeQuant needs 512-wide heads with a 64-dim RoPE tail"
    if layer.n_local_heads != layer.n_local_groups * _ROPE_QUANT_HEADS_PER_GROUP:
        return f"RopeQuant needs {_ROPE_QUANT_HEADS_PER_GROUP} heads per wo_a group"
    if layer.wo_a.weight.dtype != torch.float8_e4m3fn:
        return "RopeQuant needs an FP8 wo_a"
    if einsum_recipe != (1, 1, _ROPE_QUANT_BLOCK) or not tma_aligned_scales:
        return f"RopeQuant emits per-{_ROPE_QUANT_BLOCK} scales, wo_a wants others"
    cos_sin_cache = layer.rotary_emb.cos_sin_cache
    if cos_sin_cache.dtype != torch.float32 or not cos_sin_cache.is_contiguous():
        return "RopeQuant needs a contiguous FP32 cos/sin cache"
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
