# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton input-staging kernel for DeepSeek V4 MegaMoE.

Quantizes hidden states to fp8 with E8M0 group scales and repacks the
routing top-k tensors into the int64/float32 layout that the DeepGEMM
MegaMoE kernels consume.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_megamoe_inputs_kernel(
    hidden_states,
    x_fp8,
    x_sf,
    shared_x_sf,
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
    shared_x_sf_stride_m: tl.constexpr,
    shared_x_sf_stride_k: tl.constexpr,
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
    SHARED_BLOCK_M: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < hidden_size
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [num_groups, GROUP_K])
    amax = tl.max(hidden_groups, axis=1)
    amax = tl.maximum(amax, 1.0e-4)

    scale = amax / 448.0
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        tl.uint32
    )
    scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
    rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

    hidden_groups = tl.reshape(hidden, [num_groups, GROUP_K])
    scaled = hidden_groups * (1.0 / rounded_scale)[:, None]
    scaled = tl.reshape(scaled, [BLOCK_K])
    fp8 = scaled.to(tl.float8e4nv)
    tl.store(
        x_fp8 + token_id * x_stride_m + k_offsets * x_stride_k,
        fp8,
        mask=k_mask,
    )

    scale_offsets = tl.arange(0, num_groups)
    packed_scale = tl.sum(scale_exp << (scale_offsets * 8), axis=0).to(tl.int32)
    tl.store(
        x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
        packed_scale,
    )

    # DeepGEMM's SM100 shared-expert TMA loads require the activation scales
    # in an MN-major layout whose row permutation depends on the MegaMoE
    # scheduler's runtime BLOCK_M. Write that view while the packed UE8M0 scale
    # is already resident, avoiding another kernel and temporary tensor.
    if shared_x_sf is not None:
        m_block_id = token_id // SHARED_BLOCK_M
        m_in_block = token_id % SHARED_BLOCK_M
        aligned_block_m: tl.constexpr = triton.cdiv(SHARED_BLOCK_M, 128) * 128
        transposed_m = (
            (m_in_block // 128) * 128 + (m_in_block % 32) * 4 + (m_in_block % 128) // 32
        )
        shared_row = m_block_id * aligned_block_m + transposed_m
        tl.store(
            shared_x_sf
            + shared_row * shared_x_sf_stride_m
            + k_block_id * shared_x_sf_stride_k,
            packed_scale,
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
        )
        weights = tl.where(token_is_padding, 0.0, weights)
        tl.store(
            topk_weights_out
            + token_id * topk_weights_out_stride_m
            + topk_offsets * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def prepare_megamoe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
    shared_x_sf: torch.Tensor | None = None,
    shared_block_m: int | None = None,
) -> None:
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires hidden_size to be "
            "a multiple of 128."
        )
    top_k = topk_ids.shape[1]
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )
    if (shared_x_sf is None) != (shared_block_m is None):
        raise ValueError(
            "DeepSeek V4 MegaMoE shared input staging requires both "
            "shared_x_sf and shared_block_m."
        )
    if shared_x_sf is not None:
        assert shared_block_m is not None
        if shared_block_m <= 0:
            raise ValueError("MegaMoE shared_block_m must be positive.")
        expected_sf_k = hidden_size // 128
        if shared_x_sf.ndim != 2 or shared_x_sf.shape[1] != expected_sf_k:
            raise ValueError(
                "MegaMoE shared_x_sf must have shape "
                f"(*, {expected_sf_k}), got {tuple(shared_x_sf.shape)}."
            )
        aligned_block_m = triton.cdiv(shared_block_m, 128) * 128
        required_rows = triton.cdiv(num_tokens, shared_block_m) * aligned_block_m
        if shared_x_sf.shape[0] < required_rows:
            raise ValueError(
                "MegaMoE shared_x_sf has insufficient rows: requires "
                f"{required_rows}, got {shared_x_sf.shape[0]}."
            )

    block_k = 128
    grid = (num_tokens, triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(top_k)
    padding_stride_m = is_padding.stride(0) if is_padding is not None else 0
    _prepare_megamoe_inputs_kernel[grid](
        hidden_states,
        x_fp8,
        x_sf,
        shared_x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_sf.stride(0),
        x_sf.stride(1),
        shared_x_sf.stride(0) if shared_x_sf is not None else 0,
        shared_x_sf.stride(1) if shared_x_sf is not None else 0,
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
        GROUP_K=32,
        BLOCK_TOPK=block_topk,
        SHARED_BLOCK_M=shared_block_m or 1,
        num_warps=4,
    )
