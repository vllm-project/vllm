# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton kernel for fused batched matrix multiply + FP8 static quantization.

Used by MLA's _v_up_proj to fuse the V up-projection BMM with the
post-attention FP8 quantization into a single kernel, avoiding a
memory round-trip through bf16.

Operation:
    For each batch n in [0, N):
        output[n] = fp8_quant(input[n] @ weight[n], scale)

    input:  (N, B, L) - N heads, B tokens, L = kv_lora_rank
    weight: (N, L, V) - N heads, L -> V projection
    output: (B, N * V) - flattened across heads, in FP8
    scale:  scalar     - static per-tensor quantization scale
"""

import torch

from vllm.triton_utils import tl, triton



@triton.jit
def _bmm_fp8_kernel(
    # Pointers
    input_ptr,
    weight_ptr,
    output_ptr,
    scale_ptr,
    # Dimensions
    N,  # num heads (batch dim of BMM)
    B,  # num tokens
    L,  # kv_lora_rank (K dim)
    V,  # v_head_dim (N dim of matmul)
    # Strides for input (N, B, L)
    stride_in_n,
    stride_in_b,
    stride_in_l,
    # Strides for weight (N, L, V)
    stride_w_n,
    stride_w_l,
    stride_w_v,
    # Strides for output (B, N * V) - stored as (B, N, V) logically
    stride_out_b,
    stride_out_v,  # inner stride (1 for contiguous)
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    """Fused BMM + FP8 static quantization kernel.

    Each program instance computes a tile of the output for one head.
    Grid: (cdiv(B, BLOCK_SIZE_B) * cdiv(V, BLOCK_SIZE_V), N)
    """
    # Program IDs
    pid_tile = tl.program_id(0)  # tile index within a single head
    pid_n = tl.program_id(1)  # head index

    # Compute tile position within the (B, V) output matrix
    num_tiles_v = tl.cdiv(V, BLOCK_SIZE_V)
    pid_b = pid_tile // num_tiles_v
    pid_v = pid_tile % num_tiles_v

    # Offsets for this tile
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_v = pid_v * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    offs_l = tl.arange(0, BLOCK_SIZE_L)

    # Pointers into input[pid_n, :, :] and weight[pid_n, :, :]
    input_base = input_ptr + pid_n * stride_in_n
    weight_base = weight_ptr + pid_n * stride_w_n

    # input tile: (BLOCK_SIZE_B, BLOCK_SIZE_L)
    a_ptrs = input_base + (offs_b[:, None] * stride_in_b +
                           offs_l[None, :] * stride_in_l)
    # weight tile: (BLOCK_SIZE_L, BLOCK_SIZE_V)
    b_ptrs = weight_base + (offs_l[:, None] * stride_w_l +
                            offs_v[None, :] * stride_w_v)

    # Accumulate in fp32
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_V), dtype=tl.float32)

    for k in range(0, tl.cdiv(L, BLOCK_SIZE_L)):
        k_offs = k * BLOCK_SIZE_L + offs_l
        # Mask out-of-bounds along L dimension
        a_mask = (offs_b[:, None] < B) & (k_offs[None, :] < L)
        b_mask = (k_offs[:, None] < L) & (offs_v[None, :] < V)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_L * stride_in_l
        b_ptrs += BLOCK_SIZE_L * stride_w_l

    # Load scale and quantize to FP8
    scale = tl.load(scale_ptr)
    # FP8 quant: clamp(value * scale, -max, max)
    # 448.0 is the max representable value in FP8 E4M3
    acc = acc * scale
    acc = tl.where(acc > 448.0, 448.0, acc)
    acc = tl.where(acc < -448.0, -448.0, acc)
    result = acc.to(output_ptr.type.element_ty)

    # Store output at (B, N * V) layout
    # For head pid_n, output offset is: batch * stride_out_b + pid_n * V + v
    out_ptrs = (output_ptr + offs_b[:, None] * stride_out_b +
                (pid_n * V + offs_v[None, :]) * stride_out_v)
    out_mask = (offs_b[:, None] < B) & (offs_v[None, :] < V)
    tl.store(out_ptrs, result, mask=out_mask)


def bmm_fp8_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Fused batched matrix multiply + FP8 static quantization.

    Args:
        input: (N, B, L) input tensor in bf16/fp16
        weight: (N, L, V) weight tensor in bf16/fp16
        scale: scalar tensor - static quantization scale (1/max_val)
        output: (B, N*V) pre-allocated output tensor in FP8
    """
    assert input.ndim == 3
    assert weight.ndim == 3
    N, B, L = input.shape
    N_w, L_w, V = weight.shape
    assert N == N_w and L == L_w

    assert output.shape == (B, N * V)
    assert scale.numel() == 1

    # Grid: one program per (tile_within_head, head)
    def grid(META):
        return (
            triton.cdiv(B, META["BLOCK_SIZE_B"])
            * triton.cdiv(V, META["BLOCK_SIZE_V"]),
            N,
        )

    _bmm_fp8_kernel[grid](
        input,
        weight,
        output,
        scale,
        N,
        B,
        L,
        V,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_B=32,
        BLOCK_SIZE_V=64,
        BLOCK_SIZE_L=64,
    )
