# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utilities for Punica kernel construction.
"""

from vllm.model_executor.layers.fused_moe.utils import mm_k
from vllm.triton_utils import tl, triton


@triton.jit
def do_expand_kernel(
    pid_n,
    lora_index,
    slice_id,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    ram,  # array identifying the rows of Input ptr to operate on
    slice_start_loc,
    # input ptr strides
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,
    # lora ptr strides
    ls_d0_ptr,
    ls_d1_ptr,
    ls_d2_ptr,
    # out ptr strides
    output_d0_stride,
    output_d1_stride,
    # constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    EVEN_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    """
    Given an array of integers that identifies the rows of A, ram,
    a lora index that identifies which LoRA to use from lora_ptr, lora_index,
    a slice_id that identifies the input/output slice,
    compute the matrix product and store in the appropriate output location.
    Given that this is an expand kernel, we don't perform any split-K reduction
    as the K dimension is assumed to be small.
    """

    # ls_d*_ptr can be either an integer or a pointer
    if SAME_STRIDE:
        # integer
        cur_lora_d0_stride = ls_d0_ptr
        cur_lora_d1_stride = ls_d1_ptr
        cur_lora_d2_stride = ls_d2_ptr
    else:
        # pointer
        cur_lora_d0_stride = tl.load(ls_d0_ptr + slice_id)
        cur_lora_d1_stride = tl.load(ls_d1_ptr + slice_id)
        cur_lora_d2_stride = tl.load(ls_d2_ptr + slice_id)

    # Identify the input_ptr and lora_ptr from slice_id.
    if SLICE_NUM == 1:
        cur_input_ptr = input_ptr
        cur_lora_ptr = lora_ptr
    else:
        cur_input_ptr = input_ptr + slice_id * input_d0_stride
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(out_ptr.dtype.element_ty)
        )

    # Identify the column indices of B to process.
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    # Identify A and B block pointers
    offset_k = tl.arange(0, BLOCK_K)
    a_ptrs = (
        cur_input_ptr
        + ram[:, None] * input_d1_stride
        + offset_k[None, :] * input_d2_stride
    )
    b_ptrs = (
        cur_lora_ptr
        + cur_lora_d0_stride * lora_index
        + offset_k[:, None] * cur_lora_d2_stride
        + rbn[None, :] * cur_lora_d1_stride
    )

    # Compute the block matrix product.
    SPLIT_K = 1

    accumulator = mm_k(
        a_ptrs=a_ptrs,
        b_ptrs=b_ptrs,
        ak_stride=input_d2_stride,
        bk_stride=cur_lora_d2_stride,
        token_mask=tl.full((BLOCK_M,), 1, tl.int1),  # no masking needed
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        EVEN_K=EVEN_K,
        SPLIT_K=SPLIT_K,
        CAST_TYPE=CAST_TYPE,
        b_dtype=cur_lora_ptr.dtype.element_ty,
        USE_GDC=USE_GDC,
        IS_PRIMARY=True,  # IS_PRIMARY is always True in expand kernel
        base_k=0,
    )

    tiled_c = accumulator.to(cur_lora_ptr.dtype.element_ty)
    if SLICE_NUM == 1:
        cur_slice_start = slice_start_loc
    else:
        cur_slice_start = tl.load(slice_start_loc + slice_id)

    # Identify the C output pointers to store the results of the accumulator.
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + cur_slice_start
    offset_cm = tl.arange(0, BLOCK_M)
    c_ptrs = (
        out_ptr
        + ram[:, None] * output_d0_stride
        + offset_cn[None, :] * output_d1_stride
    )
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < (cur_slice_start + N))

    if ADD_INPUTS:
        tiled_out = tl.load(c_ptrs, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptrs, tiled_c, mask=c_mask)


@triton.jit
def do_shrink_kernel(
    pid_n,
    pid_sk,
    slice_id,
    lora_index,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    ram,
    # input strides
    input_d0_stride,
    input_d1_stride,
    # lora strides
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    # output strides
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    """
    Given an array of integers that identifies the rows of A, ram,
    a lora index that identifies which LoRA to use from lora_ptr, lora_index,
    a slice_id that identifies the input/output slice, compute the
    matrix product and store in the appropriate output location.
    """

    # Identify the lora_ptr from slice_id.
    if SLICE_NUM == 1:
        # current lora ptr
        cur_lora_ptr = lora_ptr
    else:
        # current lora ptr
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(input_ptr.dtype.element_ty)
        )

    # Identify the column indices of B to process.
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    # Identify A and B block pointers
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptrs = (
        input_ptr + ram[:, None] * input_d0_stride + offset_k[None, :] * input_d1_stride
    )
    b_ptrs = (
        cur_lora_ptr
        + lora_d0_stride * lora_index
        + rbn[None, :] * lora_d1_stride
        + offset_k[:, None] * lora_d2_stride
    )

    # Compute partial/complete block matrix product.
    accumulator = mm_k(
        a_ptrs=a_ptrs,
        b_ptrs=b_ptrs,
        ak_stride=input_d1_stride,
        bk_stride=lora_d2_stride,
        token_mask=tl.full((BLOCK_M,), 1, tl.int1),  # no masking needed
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        EVEN_K=EVEN_K,
        SPLIT_K=SPLIT_K,
        CAST_TYPE=False,
        b_dtype=cur_lora_ptr.dtype.element_ty,
        USE_GDC=USE_GDC,
        IS_PRIMARY=False,  # IS_PRIMARY is always False in shrink kernel
        base_k=pid_sk * BLOCK_K,
    )
    # GDC launch dependents hints the runtime system to launch dependent kernels.
    # Identify the C output pointers to store the results of the accumulator.
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_cm = tl.arange(0, BLOCK_M)
    cur_out_ptr = out_ptr if SLICE_NUM == 1 else out_ptr + slice_id * output_d0_stride
    c_ptrs = (
        cur_out_ptr
        + ram[:, None] * output_d1_stride
        + offset_cn[None, :] * output_d2_stride
    )
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < N)
    accumulator *= scaling

    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")
