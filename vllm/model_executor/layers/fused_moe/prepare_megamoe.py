# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input staging kernels for DeepGEMM MegaMoE."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _reciprocal_approximate_ftz(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 packed;
            cvt.rn.satfinite.e2m1x2.f32 packed, $1, $2;
            cvt.u32.u8 $0, packed;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _prepare_nvfp4_megamoe_inputs_kernel(
    hidden_states,
    input_global_scale,
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
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)
    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k
    ).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(hidden, [num_groups, GROUP_K])
    amax = tl.max(tl.abs(hidden_groups), axis=1)
    global_scale = tl.load(input_global_scale).to(tl.float32)
    sf = global_scale * amax * _reciprocal_approximate_ftz(6.0)
    sf_fp8 = sf.to(tl.float8e4nv)
    rounded_sf = sf_fp8.to(tl.float32)
    quant_scale = _reciprocal_approximate_ftz(
        rounded_sf * _reciprocal_approximate_ftz(global_scale)
    )
    quant_scale = tl.where(rounded_sf == 0.0, 0.0, quant_scale)

    scaled = tl.reshape(hidden_groups * quant_scale[:, None], [BLOCK_K])
    scaled_pairs = tl.reshape(scaled, [BLOCK_K // 2, 2])
    x_lo, x_hi = tl.split(scaled_pairs)
    packed_fp4 = _fp32x2_to_fp4x2(x_lo, x_hi)
    fp4_offsets = k_block_id * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    tl.store(x_fp4 + token_id * x_stride_m + fp4_offsets * x_stride_k, packed_fp4)

    sf_bytes = sf_fp8.to(tl.uint8, bitcast=True)
    sf_quads = tl.reshape(sf_bytes, [num_groups // 4, 4]).to(tl.uint32)
    byte_offsets = tl.arange(0, 4)
    packed_sf = tl.sum(sf_quads << (byte_offsets[None, :] * 8), axis=1)
    sf_offsets = k_block_id * (num_groups // 4) + tl.arange(0, num_groups // 4)
    tl.store(x_sf + token_id * x_sf_stride_m + sf_offsets * x_sf_stride_k, packed_sf)

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
        )
        weights = tl.load(
            topk_weights
            + token_id * topk_weights_stride_m
            + topk_offsets * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        )
        ids = tl.where(token_is_padding, -1, ids)
        weights = tl.where(token_is_padding, 0.0, weights)
        tl.store(
            topk_idx_out
            + token_id * topk_idx_stride_m
            + topk_offsets * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )
        tl.store(
            topk_weights_out
            + token_id * topk_weights_out_stride_m
            + topk_offsets * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def prepare_nvfp4_megamoe_inputs(
    hidden_states: torch.Tensor,
    input_global_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> None:
    """Quantize and stage one rank's NVFP4 MegaMoE inputs."""
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be a 2D tensor.")
    num_tokens, hidden_size = hidden_states.shape
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("hidden_states must have BF16 dtype.")
    if hidden_size % 128 != 0:
        raise ValueError("NVFP4 MegaMoE hidden_size must be a multiple of 128.")
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same 2D shape.")
    if topk_ids.shape[0] != num_tokens:
        raise ValueError("topk tensors must have one row per input token.")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must have int32 or int64 dtype.")
    if topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must have float32 dtype.")
    if x_fp4.shape != (num_tokens, hidden_size // 2):
        raise ValueError(f"Unexpected NVFP4 output shape: {x_fp4.shape}.")
    if x_fp4.dtype != torch.uint8:
        raise TypeError("NVFP4 output must have uint8 dtype.")
    if x_sf.shape != (num_tokens, hidden_size // 64):
        raise ValueError(f"Unexpected NVFP4 scale output shape: {x_sf.shape}.")
    if x_sf.dtype != torch.int32:
        raise TypeError("NVFP4 scale output must have int32 dtype.")
    if topk_idx_out.shape != topk_ids.shape or topk_idx_out.dtype != torch.int64:
        raise ValueError("topk_idx_out must match topk shape and have int64 dtype.")
    if (
        topk_weights_out.shape != topk_weights.shape
        or topk_weights_out.dtype != torch.float32
    ):
        raise ValueError(
            "topk_weights_out must match topk shape and have float32 dtype."
        )
    if input_global_scale.numel() != 1 or input_global_scale.dtype != torch.float32:
        raise ValueError("input_global_scale must be a scalar float32 tensor.")
    tensors = (
        input_global_scale,
        topk_weights,
        topk_ids,
        x_fp4,
        x_sf,
        topk_idx_out,
        topk_weights_out,
    )
    if any(t.device != hidden_states.device for t in tensors):
        raise ValueError("All MegaMoE staging tensors must be on the same device.")
    if is_padding is not None and (
        is_padding.shape != (num_tokens,)
        or is_padding.dtype != torch.bool
        or is_padding.device != hidden_states.device
    ):
        raise ValueError("is_padding must be a same-device bool tensor of shape [M].")
    if num_tokens == 0:
        return

    top_k = topk_ids.shape[1]
    block_k = 128
    grid = (num_tokens, triton.cdiv(hidden_size, block_k))
    padding_stride_m = is_padding.stride(0) if is_padding is not None else 0
    _prepare_nvfp4_megamoe_inputs_kernel[grid](
        hidden_states,
        input_global_scale,
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
        top_k,
        BLOCK_K=block_k,
        GROUP_K=16,
        BLOCK_TOPK=triton.next_power_of_2(top_k),
        num_warps=4,
    )
