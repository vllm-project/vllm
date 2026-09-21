# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8DynamicDeepGemm,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum, get_tma_aligned_size


def alloc_fp8_einsum_output(
    num_tokens: int, n_groups: int, o_lora_rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 ``z`` plus its packed UE8M0 scale for ``fp8_einsum('bhr,hdr->bhd')``.

    DeepGEMM quantizes the output per token over the flattened ``(h, d)`` row in
    groups of 32, writing 4 scales per int32 in an MN-major, TMA-aligned layout;
    that is exactly the A-operand layout of ``fp8_gemm_nt`` with recipe
    (1, 1, 32), so ``wo_b`` consumes the pair with no repack.
    """
    z = torch.empty(
        (num_tokens, n_groups, o_lora_rank), dtype=torch.float8_e4m3fn, device=device
    )
    aligned = get_tma_aligned_size(num_tokens, torch.int32.itemsize)
    z_sf = torch.empty_strided(
        (num_tokens, n_groups * o_lora_rank // 128),
        (1, aligned),
        dtype=torch.int32,
        device=device,
    )
    return z, z_sf


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
    fp8_z: bool = False,
) -> torch.Tensor:
    """O projection: inverse RoPE + grouped wo_a + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. The attention
    layer selects the recipe at initialization.

    With ``fp8_z`` the einsum emits ``z`` already quantized to MXFP8 with
    DeepGEMM packed scales and ``wo_b`` receives a ``QuantizedActivation``
    (its kernel must consume ``kMxfp8DynamicDeepGemm``), skipping the
    standalone activation quantization.
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
    if fp8_z:
        assert use_fp8, "fp8_z needs the FP8 wo_a einsum"
        z_fp8, z_sf = alloc_fp8_einsum_output(
            o.shape[0], n_groups, o_lora_rank, o.device
        )
        weight_scale = (
            wo_a.weight_scale
            if hasattr(wo_a, "weight_scale")
            else wo_a.weight_scale_inv
        )
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_proj_input, o_scale),
            (wo_a.weight, weight_scale),
            (z_fp8, z_sf),
            recipe=einsum_recipe,
        )
        z_flat = z_fp8.flatten(1)
        return wo_b(
            QuantizedActivation(
                z_flat, z_sf, torch.bfloat16, z_flat.shape, kMxfp8DynamicDeepGemm
            )
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
    return wo_b(z.flatten(1))
