# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton input-staging kernels for DeepGEMM MegaMoE.

Quantize hidden states to fp8 with E8M0 group scales, to NVFP4 with per-token
global scales, or copy them as BF16, and repack the routing top-k tensors into
the int64/float32 layout that the DeepGEMM MegaMoE kernels consume.
"""

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    FP8_DTYPE,
    QuantKey,
    kMxfp8Dynamic,
)
from vllm.triton_utils import tl, triton

_E8M0_SCALE_DTYPES = (torch.uint8, torch.float8_e8m0fnu)
# Inverse of the FP8 E4M3 range.
_INV_FP8_MAX = tl.constexpr(1.0 / 448.0)
# Inverse of the NVFP4 range: E2M1 max (6) times E4M3 max (448).
_INV_NVFP4_MAX = tl.constexpr(1.0 / 2688.0)


@triton.jit
def _quantize_fp8_groups(
    hidden,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    """Return fp8 values, fp32 group scales and, for UE8M0, the scale exponents."""
    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [BLOCK_M, num_groups, GROUP_K])
    amax = tl.max(hidden_groups, axis=2)
    amax = tl.maximum(amax, 1.0e-4)

    # Multiply by 1/448 like DeepGEMM's cast, so FP32 scales match bit for bit.
    scale = amax * _INV_FP8_MAX
    if USE_UE8M0:
        scale_bits = scale.to(tl.uint32, bitcast=True)
        scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
            tl.uint32
        )
        scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
        quant_scale = (scale_exp << 23).to(tl.float32, bitcast=True)
    else:
        scale_exp = scale.to(tl.uint32, bitcast=True)
        quant_scale = scale

    hidden_groups = tl.reshape(hidden, [BLOCK_M, num_groups, GROUP_K])
    scaled = hidden_groups * tl.math.div_rn(1.0, quant_scale)[:, :, None]
    scaled = tl.reshape(scaled, [BLOCK_M, BLOCK_K])
    return scaled.to(tl.float8e4nv), quant_scale, scale_exp


@triton.jit
def _pack_bytes(values, NUM_BYTES: tl.constexpr):
    """Pack the last dim of ``[..., NUM_BYTES]`` byte values into int32 words."""
    offsets = tl.arange(0, NUM_BYTES)
    return tl.sum(values.to(tl.uint32) << (offsets * 8), axis=-1).to(tl.int32)


@triton.jit
def _shared_sf_row(token_id, SHARED_BLOCK_M: tl.constexpr):
    """Row of ``token_id`` in the shared-expert MN-major UTCCP scale pages."""
    m_block_id = token_id // SHARED_BLOCK_M
    m_in_block = token_id % SHARED_BLOCK_M
    aligned_block_m: tl.constexpr = triton.cdiv(SHARED_BLOCK_M, 128) * 128
    transposed_m = (
        (m_in_block // 128) * 128 + (m_in_block % 32) * 4 + (m_in_block % 128) // 32
    )
    return m_block_id * aligned_block_m + transposed_m


@triton.jit
def _store_topk(
    token_id,
    token_mask,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    """Repack top-k routing as int64 ids and fp32 weights; padding is unrouted."""
    topk_offsets = tl.arange(0, BLOCK_TOPK)
    topk_mask = token_mask[:, None] & (topk_offsets[None, :] < top_k)
    token_is_padding = tl.full((BLOCK_M,), False, tl.int1)
    if is_padding is not None:
        token_is_padding = tl.load(
            is_padding + token_id * is_padding_stride_m,
            mask=token_mask,
            other=True,
        )

    ids = tl.load(
        topk_ids
        + token_id[:, None] * topk_ids_stride_m
        + topk_offsets[None, :] * topk_ids_stride_k,
        mask=topk_mask,
        other=0,
    ).to(tl.int64)
    ids = tl.where(token_is_padding[:, None], -1, ids)
    tl.store(
        topk_idx_out
        + token_id[:, None] * topk_idx_stride_m
        + topk_offsets[None, :] * topk_idx_stride_k,
        ids,
        mask=topk_mask,
    )

    weights = tl.load(
        topk_weights
        + token_id[:, None] * topk_weights_stride_m
        + topk_offsets[None, :] * topk_weights_stride_k,
        mask=topk_mask,
        other=0.0,
    )
    weights = tl.where(token_is_padding[:, None], 0.0, weights)
    tl.store(
        topk_weights_out
        + token_id[:, None] * topk_weights_out_stride_m
        + topk_offsets[None, :] * topk_weights_out_stride_k,
        weights,
        mask=topk_mask,
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _prepare_megamoe_inputs_kernel(
    hidden_states,
    x_out,
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
    num_tokens,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    SHARED_BLOCK_M: tl.constexpr,
    QUANTIZE: tl.constexpr,
    USE_UE8M0: tl.constexpr,
) -> None:
    token_id = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    token_mask = token_id < num_tokens
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = token_mask[:, None] & (k_offsets[None, :] < hidden_size)
    hidden = tl.load(
        hidden_states
        + token_id[:, None] * hidden_stride_m
        + k_offsets[None, :] * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    if not QUANTIZE:
        tl.store(
            x_out + token_id[:, None] * x_stride_m + k_offsets[None, :] * x_stride_k,
            hidden.to(x_out.dtype.element_ty),
            mask=k_mask,
        )
    else:
        num_groups: tl.constexpr = BLOCK_K // GROUP_K
        fp8, quant_scale, scale_exp = _quantize_fp8_groups(
            hidden, BLOCK_M, BLOCK_K, GROUP_K, USE_UE8M0
        )
        tl.store(
            x_out + token_id[:, None] * x_stride_m + k_offsets[None, :] * x_stride_k,
            fp8,
            mask=k_mask,
        )

        if USE_UE8M0:
            packed_scale = _pack_bytes(scale_exp, num_groups)
            tl.store(
                x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
                packed_scale,
                mask=token_mask,
            )
        else:
            # FP32 scales as raw bits (int32 buffer on DeepGEMM SM90)
            sf_offset = tl.arange(0, num_groups)
            tl.store(
                x_sf
                + token_id[:, None] * x_sf_stride_m
                + k_block_id * x_sf_stride_k
                + sf_offset[None, :],
                quant_scale.to(x_sf.dtype.element_ty, bitcast=True),
                mask=token_mask[:, None],
            )

        # DeepGEMM's shared-expert TMA loads require the activation scales
        # in an MN-major layout whose row permutation depends on the MegaMoE
        # scheduler's runtime BLOCK_M. Write that view while the scale is
        # already resident, avoiding another kernel and temporary tensor.
        if shared_x_sf is not None:
            if USE_UE8M0:
                shared_scale = packed_scale
            else:
                shared_scale = tl.reshape(quant_scale, [BLOCK_M]).to(
                    shared_x_sf.dtype.element_ty, bitcast=True
                )
            shared_row = _shared_sf_row(token_id, SHARED_BLOCK_M)
            tl.store(
                shared_x_sf
                + shared_row * shared_x_sf_stride_m
                + k_block_id * shared_x_sf_stride_k,
                shared_scale,
                mask=token_mask,
            )

    if k_block_id == 0:
        _store_topk(
            token_id,
            token_mask,
            topk_ids,
            topk_weights,
            is_padding,
            topk_idx_out,
            topk_weights_out,
            topk_ids_stride_m,
            topk_ids_stride_k,
            topk_weights_stride_m,
            topk_weights_stride_k,
            is_padding_stride_m,
            topk_idx_stride_m,
            topk_idx_stride_k,
            topk_weights_out_stride_m,
            topk_weights_out_stride_k,
            top_k,
            BLOCK_M,
            BLOCK_TOPK,
        )


@triton.jit(do_not_specialize=["num_tokens", "l2_activation_scale"])
def _prepare_megamoe_nvfp4_inputs_kernel(
    hidden_states,
    x_out,
    x_sf,
    x_scales,
    shared_x,
    shared_x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    w2_global_scale,
    l2_alpha,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    shared_x_stride_m: tl.constexpr,
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
    num_tokens,
    l2_activation_scale,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    num_local_experts: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    SHARED_BLOCK_M: tl.constexpr,
    SHARED_MXFP8: tl.constexpr,
) -> None:
    # Grid (token, split): each program quantizes the BLOCK_K chunks split,
    # split + NUM_SPLITS, ... of one token's row.
    token = tl.program_id(0)
    split = tl.program_id(1)
    if l2_alpha is not None:
        experts = tl.arange(0, BLOCK_E)
        expert_mask = (experts < num_local_experts) & (token == 0) & (split == 0)
        alpha = tl.load(w2_global_scale + experts, mask=expert_mask)
        tl.store(l2_alpha + experts, alpha * l2_activation_scale, mask=expert_mask)
    if token >= num_tokens:
        return

    if split == 0:
        token_ids = token + tl.zeros([1], dtype=tl.int32)
        _store_topk(
            token_ids,
            token_ids < num_tokens,
            topk_ids,
            topk_weights,
            is_padding,
            topk_idx_out,
            topk_weights_out,
            topk_ids_stride_m,
            topk_ids_stride_k,
            topk_weights_stride_m,
            topk_weights_stride_k,
            is_padding_stride_m,
            topk_idx_stride_m,
            topk_idx_stride_k,
            topk_weights_out_stride_m,
            topk_weights_out_stride_k,
            top_k,
            1,
            BLOCK_TOPK,
        )

    # Every program reads the whole row for the token scale.
    row = hidden_states + token.to(tl.int64) * hidden_stride_m
    cols = tl.arange(0, BLOCK_H)
    hidden_row = tl.load(
        row + cols * hidden_stride_k, mask=cols < hidden_size, other=0.0
    )
    amax = tl.max(tl.abs(hidden_row.to(tl.float32)), axis=0)
    x_scale = tl.maximum(amax, 1.0e-12) * _INV_NVFP4_MAX
    if split == 0:
        tl.store(x_scales + token, x_scale)

    shared_sf_row = _shared_sf_row(token, SHARED_BLOCK_M)
    for k0 in tl.range(split * BLOCK_K, hidden_size, NUM_SPLITS * BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_ids < hidden_size
        hidden = tl.load(row + k_ids * hidden_stride_k, mask=k_mask, other=0.0)

        # E2M1 values with one E4M3 scale per 16, relative to the token scale.
        values = tl.reshape(
            tl.div_rn(hidden.to(tl.float32), x_scale), [BLOCK_K // 16, 16]
        )
        block_scale = tl.max(tl.abs(values), axis=1) * (1.0 / 6.0)
        block_scale = tl.minimum(tl.maximum(block_scale, 2.0**-9), 448.0)
        block_scale = block_scale.to(tl.float8e4nv)
        values = tl.div_rn(values, block_scale.to(tl.float32)[:, None])
        magnitude = tl.abs(values)
        # Round to nearest even on the grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
        code = (
            (magnitude > 0.25).to(tl.int32)
            + (magnitude >= 0.75).to(tl.int32)
            + (magnitude > 1.25).to(tl.int32)
            + (magnitude >= 1.75).to(tl.int32)
            + (magnitude > 2.5).to(tl.int32)
            + (magnitude >= 3.5).to(tl.int32)
            + (magnitude > 5.0).to(tl.int32)
        )
        code = tl.where((values < 0) & (code != 0), code | 8, code)
        low, high = tl.split(tl.reshape(code, [BLOCK_K // 2, 2]))
        packed_ids = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        tl.store(
            x_out + token * x_stride_m + packed_ids,
            (low | (high << 4)).to(tl.uint8),
            mask=packed_ids < hidden_size // 2,
        )
        # Four E4M3 scales per little-endian int32 word, one word per 64 values.
        sf_ids = k0 // 64 + tl.arange(0, BLOCK_K // 64)
        sf_bytes = tl.reshape(
            block_scale.to(tl.uint8, bitcast=True), [BLOCK_K // 64, 4]
        )
        tl.store(
            x_sf + token * x_sf_stride_m + sf_ids,
            _pack_bytes(sf_bytes, 4),
            mask=sf_ids < hidden_size // 64,
        )

        if shared_x is not None:
            shared_ptrs = shared_x + token * shared_x_stride_m + k_ids
            if SHARED_MXFP8:
                fp8, _, scale_exp = _quantize_fp8_groups(
                    tl.reshape(hidden.to(tl.float32), (1, BLOCK_K)),
                    1,
                    BLOCK_K,
                    32,
                    True,
                )
                tl.store(shared_ptrs, tl.reshape(fp8, [BLOCK_K]), mask=k_mask)
                word_ids = k0 // 128 + tl.arange(0, BLOCK_K // 128)
                tl.store(
                    shared_x_sf
                    + shared_sf_row * shared_x_sf_stride_m
                    + word_ids * shared_x_sf_stride_k,
                    _pack_bytes(tl.reshape(scale_exp, [BLOCK_K // 128, 4]), 4),
                    mask=word_ids < hidden_size // 128,
                )
            else:
                tl.store(shared_ptrs, hidden, mask=k_mask)


def _resolve_hidden_quant(hidden_quant: QuantKey, block_k: int) -> tuple[int, bool]:
    """Map a hidden-state ``QuantKey`` to the kernel's ``(GROUP_K, USE_UE8M0)``."""
    scale = hidden_quant.scale
    group_shape = scale.group_shape
    if (
        hidden_quant.dtype != FP8_DTYPE
        or not hidden_quant.symmetric
        or hidden_quant.scale2 is not None
        or scale.static
        or not group_shape.is_per_group()
        or block_k % group_shape.col != 0
        or scale.dtype not in (*_E8M0_SCALE_DTYPES, torch.float32)
    ):
        raise ValueError(
            "MegaMoE input staging requires symmetric dynamic fp8 "
            f"per-group quantization with a group size dividing {block_k} and "
            f"E8M0 or fp32 scales, got {hidden_quant}."
        )
    return group_shape.col, scale.dtype in _E8M0_SCALE_DTYPES


def prepare_megamoe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor | None,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
    shared_x_sf: torch.Tensor | None = None,
    shared_block_m: int | None = None,
    hidden_quant: QuantKey | None = kMxfp8Dynamic,
) -> None:
    """Quantize hidden states and repack top-k routing for DeepGEMM MegaMoE.

    Args:
        hidden_states: Input activations of shape ``[num_tokens, hidden_dim]``.
        topk_weights: Router top-k weights of shape ``[num_tokens, top_k]``.
        topk_ids: Router top-k expert ids of shape ``[num_tokens, top_k]``.
        x: Output buffer for the dispatched hidden states.
        x_sf: Output buffer for the hidden-state scale factors; ``None`` when
            ``hidden_quant`` is ``None``.
        topk_idx_out: Output buffer for the repacked top-k expert ids.
        topk_weights_out: Output buffer for the repacked top-k weights.
        is_padding: Optional per-token mask; padded tokens are not routed.
        shared_x_sf: Optional output buffer for shared-expert scale factors.
            Must be given together with ``shared_block_m``.
        shared_block_m: Block M used to lay out ``shared_x_sf``.
        hidden_quant: Hidden-state quantization scheme. E8M0 scales are packed
            four per int32 into ``x_sf``; fp32 scales are stored as their bits.
            ``None`` copies the hidden states into ``x`` unquantized.

    """
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError(
            "MegaMoE input staging requires hidden_size to be a multiple of 128."
        )
    block_k = 128
    quantize = hidden_quant is not None
    if quantize != (x_sf is not None):
        raise ValueError("MegaMoE input staging needs x_sf iff hidden_quant is set.")
    group_k, use_ue8m0 = (
        (block_k, False)
        if hidden_quant is None
        else _resolve_hidden_quant(hidden_quant, block_k)
    )
    top_k = topk_ids.shape[1]
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )
    if (shared_x_sf is None) != (shared_block_m is None):
        raise ValueError(
            "MegaMoE shared input staging requires both shared_x_sf and shared_block_m."
        )
    if shared_x_sf is not None and not use_ue8m0 and group_k != block_k:
        raise ValueError(
            "MegaMoE shared input staging currently requires "
            f"UE8M0-packed or per-{block_k} fp32 hidden scales."
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

    # On GB200, eight-row tiles win from 64 tokens; keep smaller batches untiled.
    block_m = 8 if num_tokens >= 64 else 1
    grid = (triton.cdiv(num_tokens, block_m), triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(top_k)
    padding_stride_m = is_padding.stride(0) if is_padding is not None else 0
    _prepare_megamoe_inputs_kernel[grid](
        hidden_states,
        x,
        x_sf,
        shared_x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x.stride(0),
        x.stride(1),
        x_sf.stride(0) if x_sf is not None else 0,
        x_sf.stride(1) if x_sf is not None else 0,
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
        num_tokens,
        hidden_size,
        top_k,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        GROUP_K=group_k,
        BLOCK_TOPK=block_topk,
        SHARED_BLOCK_M=shared_block_m or 1,
        QUANTIZE=quantize,
        USE_UE8M0=use_ue8m0,
        num_warps=4,
    )


def prepare_megamoe_nvfp4_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    x_scales: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
    shared_x: torch.Tensor | None = None,
    shared_x_sf: torch.Tensor | None = None,
    shared_block_m: int | None = None,
    w2_global_scale: torch.Tensor | None = None,
    l2_alpha: torch.Tensor | None = None,
    l2_activation_scale: float = 1.0,
) -> None:
    """Stage DeepGEMM NVFP4 MegaMoE inputs (``kNvfp4DynamicToken``).

    Each token is scaled by its own fp32 ``x_scales`` (row amax / 2688) and
    quantized to E2M1 with one E4M3 scale per 16 values.

    Args:
        hidden_states: BF16 activations of shape ``[num_tokens, hidden_dim]``.
        topk_weights: Router top-k weights of shape ``[num_tokens, top_k]``.
        topk_ids: Router top-k expert ids of shape ``[num_tokens, top_k]``.
        x: Packed E2M1 output, ``[num_tokens, hidden_dim // 2]``.
        x_sf: E4M3 scales packed four per int32, ``[num_tokens, hidden_dim // 64]``.
        x_scales: Per-token fp32 dequantization scales, ``[num_tokens]``.
        topk_idx_out: Output buffer for the repacked top-k expert ids.
        topk_weights_out: Output buffer for the repacked top-k weights.
        is_padding: Optional per-token mask; padded tokens are not routed.
        shared_x: Optional shared-expert input, fp8 (MXFP8 with ``shared_x_sf``)
            or a BF16 copy of the hidden states.
        shared_x_sf: UE8M0 scales of an fp8 ``shared_x`` in the shared-expert
            TMA layout for ``shared_block_m``.
        shared_block_m: Block M used to lay out ``shared_x_sf``.
        w2_global_scale: Per-local-expert L2 weight global scales.
        l2_alpha: Output ``w2_global_scale * l2_activation_scale``, the L2
            dequantization scale of each local expert. Written even without
            local tokens, as other ranks' tokens still reach these experts.
        l2_activation_scale: Global scale the kernel quantizes L2 inputs with.

    """
    num_tokens, hidden_size = hidden_states.shape
    if hidden_size % 128 != 0:
        raise ValueError(
            "MegaMoE input staging requires hidden_size to be a multiple of 128."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )
    shared_mxfp8 = shared_x is not None and shared_x.dtype == FP8_DTYPE
    if shared_mxfp8 != (shared_x_sf is not None) or (shared_x_sf is None) != (
        shared_block_m is None
    ):
        raise ValueError(
            "MegaMoE NVFP4 staging needs shared_x_sf and shared_block_m exactly "
            "for an fp8 shared_x."
        )
    if (w2_global_scale is None) != (l2_alpha is None):
        raise ValueError("MegaMoE NVFP4 staging needs w2_global_scale with l2_alpha.")
    for name, tensor in (("x", x), ("x_sf", x_sf), ("shared_x", shared_x)):
        if tensor is not None and tensor.stride(-1) != 1:
            raise ValueError(f"MegaMoE NVFP4 staging requires a row-major {name}.")

    top_k = topk_ids.shape[1]
    num_local_experts = 0 if l2_alpha is None else l2_alpha.numel()
    # Up to 512 tokens each row is split over programs; larger batches fill the
    # GPU with one program per row looping over wider chunks.
    block_h = triton.next_power_of_2(hidden_size)
    if num_tokens <= 512:
        block_k = min(1024, block_h)
        num_splits = triton.cdiv(hidden_size, block_k)
    else:
        block_k = min(2048, block_h)
        num_splits = 1
    # One program even without tokens, to write l2_alpha.
    grid = (max(num_tokens, 1), num_splits)
    _prepare_megamoe_nvfp4_inputs_kernel[grid](
        hidden_states,
        x.view(torch.uint8),
        x_sf,
        x_scales,
        shared_x,
        shared_x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        w2_global_scale,
        l2_alpha,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x.stride(0),
        x_sf.stride(0),
        shared_x.stride(0) if shared_x is not None else 0,
        shared_x_sf.stride(0) if shared_x_sf is not None else 0,
        shared_x_sf.stride(1) if shared_x_sf is not None else 0,
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        is_padding.stride(0) if is_padding is not None else 0,
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        num_tokens,
        l2_activation_scale,
        hidden_size,
        top_k,
        num_local_experts,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        BLOCK_TOPK=triton.next_power_of_2(top_k),
        BLOCK_E=triton.next_power_of_2(max(num_local_experts, 1)),
        SHARED_BLOCK_M=shared_block_m or 1,
        SHARED_MXFP8=shared_mxfp8,
        num_warps=4,
    )
