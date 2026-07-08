# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton input-staging kernel for the DSA (DeepSeek V3.2 / GLM-5.2) NVFP4 MegaMoE.

Quantizes hidden states to NVFP4 (packed E2M1 values with one E4M3 scale per
16 elements, 4 scales packed per int32 along K) and repacks the routing top-k
tensors into the int64/float32 layout that the DeepGEMM MegaMoE kernel consumes.

The E2M1 rounding matches ``cvt.rn.satfinite.e2m1x2.f32`` (round-to-nearest-even
on the {0, 0.5, 1, 1.5, 2, 3, 4, 6} grid), and the scale is
``e4m3(max(amax / 6, 2^-9))`` — the same recipe DeepGEMM's L1 epilogue uses for
the intermediate activations, so both GEMM inputs share one quantization scheme.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _quantize_to_e2m1_rne(x):
    # Round-to-nearest-even on the E2M1 grid; ties at 0.75 / 1.75 / 3.5 round up
    # (to even codes), ties at 0.25 / 1.25 / 2.5 / 5.0 round down. satfinite: >6 -> 6.
    ax = tl.abs(x)
    code = (
        (ax > 0.25).to(tl.int32)
        + (ax >= 0.75).to(tl.int32)
        + (ax > 1.25).to(tl.int32)
        + (ax >= 1.75).to(tl.int32)
        + (ax > 2.5).to(tl.int32)
        + (ax >= 3.5).to(tl.int32)
        + (ax > 5.0).to(tl.int32)
    )
    return tl.where(x < 0, code | 8, code)


@triton.jit
def _prepare_megamoe_nvfp4_inputs_kernel(
    hidden_states,
    x_fp4,
    x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
    ).to(tl.float32)

    # Per-16-element E4M3 scales: sf = e4m3(max(amax / 6, 2^-9))
    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(hidden, [num_groups, GROUP_K])
    amax = tl.max(tl.abs(hidden_groups), axis=1)
    sf_fp32 = tl.maximum(amax * (1.0 / 6.0), 0.001953125)
    sf_e4m3 = sf_fp32.to(tl.float8e4nv)

    # Quantize with the dequantized (rounded) scale
    # NOTES: triton's `/` (and even `tl.fdiv(ieee_rounding=True)`) lowers to the
    # approximate `div.full.f32`, whose ~2-ulp error tips round-to-nearest-even
    # tie values, so divide with correctly-rounded PTX `div.rn.f32`
    sf_bcast = tl.broadcast_to(sf_e4m3.to(tl.float32)[:, None], hidden_groups.shape)
    scaled = tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [hidden_groups, sf_bcast],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    codes = _quantize_to_e2m1_rne(tl.reshape(scaled, [BLOCK_K]))

    # Pack 2 E2M1 codes per byte (even element in the low nibble)
    lo, hi = tl.split(tl.reshape(codes, [BLOCK_K // 2, 2]))
    packed_fp4 = (lo | (hi << 4)).to(tl.uint8)
    byte_offsets = k_block_id * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    tl.store(
        x_fp4 + token_id * x_stride_m + byte_offsets * x_stride_k,
        packed_fp4,
    )

    # Pack 4 E4M3 scale bytes per int32 along K (little-endian: byte j = group 4q+j)
    sf_bytes = sf_e4m3.to(tl.uint8, bitcast=True).to(tl.uint32)
    a, b = tl.split(tl.reshape(sf_bytes, [num_groups // 4, 2, 2]))
    s0, s2 = tl.split(a)
    s1, s3 = tl.split(b)
    packed_sf = (s0 | (s1 << 8) | (s2 << 16) | (s3 << 24)).to(tl.int32, bitcast=True)
    sf_offsets = k_block_id * (num_groups // 4) + tl.arange(0, num_groups // 4)
    tl.store(
        x_sf + token_id * x_sf_stride_m + sf_offsets * x_sf_stride_k,
        packed_sf,
    )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offsets < top_k
        token_is_padding = False
        if is_padding is not None:
            token_is_padding = tl.load(is_padding + token_id * is_padding_stride_m)

        ids = tl.load(
            topk_ids + token_id * topk_ids_stride_m + topk_offsets * topk_ids_stride_k,
            mask=topk_mask,
            other=0,
        ).to(tl.int64)
        ids = tl.where(token_is_padding, -1, ids)
        tl.store(
            topk_idx_out
            + token_id * topk_idx_stride_m
            + topk_offsets * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )

        weights = tl.load(
            topk_weights
            + token_id * topk_weights_stride_m
            + topk_offsets * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        ).to(tl.float32)
        weights = tl.where(token_is_padding, 0.0, weights)
        tl.store(
            topk_weights_out
            + token_id * topk_weights_out_stride_m
            + topk_offsets * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def prepare_megamoe_nvfp4_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> None:
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    block_k = 256
    if hidden_size % block_k != 0:
        raise ValueError(
            "NVFP4 MegaMoE input staging requires hidden_size to be a "
            f"multiple of {block_k}."
        )
    top_k = topk_ids.shape[1]
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "NVFP4 MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )

    grid = (num_tokens, triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(top_k)
    padding_stride_m = is_padding.stride(0) if is_padding is not None else 0
    _prepare_megamoe_nvfp4_inputs_kernel[grid](
        hidden_states,
        x_fp4,
        x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x_fp4.stride(0),
        x_fp4.stride(1),
        x_sf.stride(0),
        x_sf.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        padding_stride_m,
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        hidden_size,
        top_k,
        BLOCK_K=block_k,
        GROUP_K=16,
        BLOCK_TOPK=block_topk,
        num_warps=4,
    )
