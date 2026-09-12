# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum
from vllm.utils.math_utils import round_up

# DeepGEMM picks the kernel configuration of the fp8 einsum (and of wo_b's
# GEMM) from the token count and JIT-compiles a new kernel for every
# configuration it has not seen: ~3 s per compile, paid the first time a
# batch of that size shows up, in the middle of serving. Token counts up to
# this value are fixed CUDA graph shapes, compiled at capture; larger batches
# (eager prefill steps) take every value between two chunk boundaries, so
# they are padded to a multiple of the bucket size, which bounds the set of
# configurations to what the warm-up can pre-compile
# (vllm/model_executor/warmup/deepseek_v4_o_proj_warmup.py).
O_PROJ_EAGER_BUCKET = 1024


def o_proj_padded_num_tokens(num_tokens: int) -> int:
    if num_tokens <= O_PROJ_EAGER_BUCKET:
        return num_tokens
    return round_up(num_tokens, O_PROJ_EAGER_BUCKET)


def o_proj_warmup_num_tokens(max_num_tokens: int) -> list[int]:
    """Every token count the o-projection can be asked for, for warm-up.

    Below the bucket every count is possible (a prefill chunk of 324 tokens
    runs eagerly at 324); above it only the bucket multiples remain.
    """
    small = list(range(1, min(max_num_tokens, O_PROJ_EAGER_BUCKET) + 1))
    top = o_proj_padded_num_tokens(max_num_tokens)
    return small + list(range(2 * O_PROJ_EAGER_BUCKET, top + 1, O_PROJ_EAGER_BUCKET))


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
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
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
    layer selects the recipe at initialization.
    """
    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    num_tokens = o.shape[0]
    padded_num_tokens = o_proj_padded_num_tokens(num_tokens)
    if padded_num_tokens != num_tokens:
        # Zero rows: their projection is zero and is sliced off below; a
        # GEMM's rows are independent, so the real rows are unaffected.
        pad = padded_num_tokens - num_tokens
        o = F.pad(o, (0, 0, 0, 0, 0, pad))
        positions = F.pad(positions, (0, pad))
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
    out = wo_b(z.flatten(1))
    if padded_num_tokens != num_tokens:
        out = out[:num_tokens]
    return out
