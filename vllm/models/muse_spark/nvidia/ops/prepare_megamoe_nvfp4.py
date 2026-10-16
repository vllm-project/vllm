# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton input staging for MuseSpark1x on the NVFP4 MegaMoE kernel.

MegaMoE reads linear scales, not ``scaled_fp4_quant``'s swizzled 128x4 layout.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _quantize_nvfp4(hidden, BLOCK_K: tl.constexpr):
    num_groups: tl.constexpr = BLOCK_K // 16
    groups = tl.reshape(hidden, [num_groups, 16])
    amax = tl.max(tl.abs(groups), axis=1)

    # The divisor is the *quantized* scale, so round-trip it through E4M3. The
    # floor is E4M3's smallest subnormal, keeping an all-zero block off zero.
    sf = tl.maximum(amax * (1.0 / 6.0), 2.0**-9).to(tl.float8e4nv)
    # Compare against scaled grid boundaries rather than dividing: Triton's
    # division is approximate and rounds exact ties the wrong way. E4M3 scales
    # (4 significant bits) times boundaries (<= 3) are exact in FP32.
    sf_f32 = sf.to(tl.float32)[:, None]
    magnitude = tl.abs(groups)

    # Round to nearest on the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}. Strict `>`
    # on one set of midpoints and `>=` on the other sends every tie to an even
    # code, matching `cvt.rn.satfinite.e2m1x2.f32`.
    code = (
        (magnitude > 0.25 * sf_f32).to(tl.uint8)
        + (magnitude > 1.25 * sf_f32).to(tl.uint8)
        + (magnitude > 2.5 * sf_f32).to(tl.uint8)
        + (magnitude > 5.0 * sf_f32).to(tl.uint8)
        + (magnitude >= 0.75 * sf_f32).to(tl.uint8)
        + (magnitude >= 1.75 * sf_f32).to(tl.uint8)
        + (magnitude >= 3.5 * sf_f32).to(tl.uint8)
    )
    code |= ((groups < 0) & (code != 0)).to(tl.uint8) << 3
    code = tl.reshape(code, [BLOCK_K])

    lo, hi = tl.split(tl.reshape(code, [BLOCK_K // 2, 2]))
    packed = (lo & 0x0F) | (hi << 4)
    return packed, sf


@triton.jit
def _prepare_megamoe_nvfp4_kernel(
    hidden_states,
    x_packed,
    x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    x_packed_stride_m: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    USE_PDL: tl.constexpr,
) -> None:
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    hidden = tl.load(hidden_states + token_id * hidden_stride_m + k_offsets).to(
        tl.float32
    )

    num_groups: tl.constexpr = BLOCK_K // 16
    packed, sf = _quantize_nvfp4(hidden, BLOCK_K)
    tl.store(
        x_packed
        + token_id * x_packed_stride_m
        + k_block_id * (BLOCK_K // 2)
        + tl.arange(0, BLOCK_K // 2),
        packed,
    )
    tl.store(
        x_sf
        + token_id * x_sf_stride_m
        + k_block_id * num_groups
        + tl.arange(0, num_groups),
        sf.to(tl.uint8, bitcast=True),
    )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offsets < top_k
        token_is_padding = False
        if is_padding is not None:
            token_is_padding = tl.load(is_padding + token_id * is_padding_stride_m)

        ids = tl.load(
            topk_ids + token_id * topk_ids_stride_m + topk_offsets,
            mask=topk_mask,
            other=0,
        ).to(tl.int64)
        tl.store(
            topk_idx_out + token_id * topk_idx_stride_m + topk_offsets,
            tl.where(token_is_padding, -1, ids),
            mask=topk_mask,
        )

        weights = tl.load(
            topk_weights + token_id * topk_weights_stride_m + topk_offsets,
            mask=topk_mask,
            other=0.0,
        )
        tl.store(
            topk_weights_out + token_id * topk_weights_out_stride_m + topk_offsets,
            tl.where(token_is_padding, 0.0, weights),
            mask=topk_mask,
        )


def prepare_megamoe_nvfp4_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_packed: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> None:
    """Stage ``hidden_states`` and routing into MegaMoE's symmetric buffer."""
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 64 != 0:
        raise ValueError(
            "MuseSpark1x NVFP4 MegaMoE staging requires hidden_size to be a "
            f"multiple of 64, got {hidden_size}."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "MuseSpark1x NVFP4 MegaMoE staging requires topk_weights and "
            "topk_ids to have the same shape."
        )

    top_k = topk_ids.shape[1]
    num_elements = num_tokens * hidden_size
    if num_elements <= 512 * 1024:
        block_k, num_warps = 256, 4
    elif num_elements < 2 * 1024 * 1024:
        block_k, num_warps = 512, 2
    else:
        block_k, num_warps = 1024, 1
    block_k = min(block_k, hidden_size)
    while hidden_size % block_k != 0:
        block_k //= 2

    enable_pdl = current_platform.is_cuda() and current_platform.is_arch_support_pdl()
    _prepare_megamoe_nvfp4_kernel[(num_tokens, hidden_size // block_k)](
        hidden_states,
        x_packed,
        x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        x_packed.stride(0),
        x_sf.stride(0),
        topk_ids.stride(0),
        topk_weights.stride(0),
        is_padding.stride(0) if is_padding is not None else 0,
        topk_idx_out.stride(0),
        topk_weights_out.stride(0),
        hidden_size=hidden_size,
        top_k=top_k,
        BLOCK_K=block_k,
        BLOCK_TOPK=triton.next_power_of_2(top_k),
        USE_PDL=enable_pdl,
        num_warps=num_warps,
        **({"launch_pdl": True} if enable_pdl else {}),
    )


@triton.jit
def _norm_quant_megamoe_nvfp4_kernel(
    x,
    weight,
    packed_out,
    sf_out,
    x_stride: tl.constexpr,
    packed_stride: tl.constexpr,
    sf_stride: tl.constexpr,
    eps,
    gain_center,
    HIDDEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    token = tl.program_id(0)
    block = tl.program_id(1)
    d = tl.arange(0, HIDDEN)
    row = tl.load(x + token * x_stride + d).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(row * row, axis=0) * (1.0 / HIDDEN) + eps)
    k = block * BLOCK_K + tl.arange(0, BLOCK_K)
    hidden = tl.load(x + token * x_stride + k).to(tl.float32)
    gamma = tl.load(weight + k).to(tl.float32) + gain_center
    # Round to BF16 as the unfused norm -> quantize path does.
    hidden = ((hidden * rstd) * gamma).to(tl.bfloat16).to(tl.float32)
    packed, sf = _quantize_nvfp4(hidden, BLOCK_K)
    tl.store(
        packed_out
        + token * packed_stride
        + block * (BLOCK_K // 2)
        + tl.arange(0, BLOCK_K // 2),
        packed,
    )
    tl.store(
        sf_out
        + token * sf_stride
        + block * (BLOCK_K // 16)
        + tl.arange(0, BLOCK_K // 16),
        sf.to(tl.uint8, bitcast=True),
    )


def prepare_megamoe_nvfp4_with_norm(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    gain_center: float,
    x_packed: torch.Tensor,
    x_sf: torch.Tensor,
) -> None:
    """Normalize BF16 pooled states and write NVFP4 into MegaMoE's buffer.

    The caller must join the routing aux stream before launching MegaMoE.
    """
    num_tokens, hidden_size = hidden_states.shape
    assert hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous()
    assert hidden_size == 4096
    assert norm_weight.shape == (hidden_size,) and norm_weight.is_contiguous()
    if num_tokens == 0:
        return
    enable_pdl = current_platform.is_cuda() and current_platform.is_arch_support_pdl()
    _norm_quant_megamoe_nvfp4_kernel[(num_tokens, hidden_size // 1024)](
        hidden_states,
        norm_weight,
        x_packed,
        x_sf,
        hidden_states.stride(0),
        x_packed.stride(0),
        x_sf.stride(0),
        norm_eps,
        gain_center,
        hidden_size,
        1024,
        num_warps=8,
        enable_fp_fusion=False,
        USE_PDL=enable_pdl,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
