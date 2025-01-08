"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

from typing import List

import torch
import triton
import triton.language as tl

from vllm.utils import direct_register_custom_op

from .utils import _get_lora_a_ptr


@triton.jit
def _sgmv_shrink_kernel(
        input_ptr,
        lora_ptr,  #1-3
        out_ptr,
        N,
        K,
        b_seq_start_loc,
        seq_lens,
        lora_indices,
        scaling,
        input_d0_stride,
        input_d1_stride,  # 1
        lora_d0_stride,
        lora_d1_stride,
        lora_d2_stride,  # 1
        output_d0_stride,
        output_d1_stride,
        output_d2_stride,  # 1 
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        SLICE_NUM: tl.constexpr):
    """
    The sgmv's shrink triton kernel is based on GroupGEMM+SPLIT-K.
    The GEMM of Multi-LoRA can be considered as GroupGEMM. Additionally,
    introducing SPLIT-K can improve performance
    """
    pid = tl.program_id(axis=0)
    pid_mix = tl.program_id(axis=1)
    cur_batch = tl.program_id(axis=2)
    cta_n_num = tl.cdiv(N, BLOCK_N)
    pid_m = pid // cta_n_num
    pid_n = pid % cta_n_num
    if SLICE_NUM == 1:
        slice_id: tl.constexpr = 0
        pid_sk = tl.program_id(axis=1)
    else:
        pid_mix = tl.program_id(axis=1)
        slice_id = pid_mix // SPLIT_K
        pid_sk = pid_mix % SPLIT_K

    M = tl.load(seq_lens + cur_batch)
    if pid_m * BLOCK_M > M:
        return
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return
    cur_seq_start = tl.load(b_seq_start_loc + cur_batch)
    offset_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(offset_m % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    # input ptr
    a_ptr = (input_ptr + cur_seq_start * input_d0_stride +
             ram[:, None] * input_d0_stride +
             offset_k[None, :] * input_d1_stride)

    if SLICE_NUM == 1:
        # current lora ptr
        cur_lora_ptr = lora_ptr
    else:
        # current lora ptr
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(input_ptr.dtype.element_ty))

    b_ptr = (cur_lora_ptr + lora_d0_stride * lora_index +
             rbn[None, :] * lora_d1_stride +
             offset_k[:, None] * lora_d2_stride)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            tiled_a = tl.load(a_ptr,
                              mask=offset_k[None, :] < k_remaining,
                              other=0.0)
            tiled_b = tl.load(b_ptr,
                              mask=offset_k[:, None] < k_remaining,
                              other=0.0)
        accumulator += tl.dot(tiled_a, tiled_b)

        a_ptr += BLOCK_K * SPLIT_K * input_d1_stride
        b_ptr += BLOCK_K * SPLIT_K * lora_d2_stride
    offset_cm = cur_seq_start + tl.arange(0, BLOCK_M) + pid_m * BLOCK_M

    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    cur_out_ptr = (out_ptr if SLICE_NUM == 1 else out_ptr +
                   slice_id * output_d0_stride)
    c_ptr = cur_out_ptr + offset_cm[:, None] * output_d1_stride + offset_cn[
        None, :] * output_d2_stride
    c_mask = (offset_cm[:, None] <
              (cur_seq_start + M)) & (offset_cn[None, :] < N)
    accumulator *= scaling
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask)


@torch.inference_mode()
def _sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
) -> None:
    """
    Args:
        inputs (torch.Tensor): input tensor
        lora_a_weights (List[torch.Tensor]): lora'a weight
        output_tensor (torch.Tensor): output tensor
        b_seq_start_loc (torch.Tensor): (batch_size,). The cumulative
            sequence lengths of the sequences in the batch, used to index
            into sequence. E.g., if the sequence length is [4, 6], it is
            [0, 4].
        seq_len_tensor (torch.Tensor): (batch_size,). Record the sequence
            length of the sequences in the batch.
        lora_indices_tensor (torch.Tensor): (batch_size,). The LoRA index
            corresponding to each batch. An index of -1 means no lora should be
            applied.
        batches (int): batch size
        max_seq_length (int): The max sequence lengths of the sequences in the 
            batch.
        token_nums (int): The token numbers in the batch. Used to verify if the 
            token numbers in the inputs matches the one in the metadata.
        scaling (float): Scaling factor.
    """
    assert inputs.dtype == lora_a_weights[0].dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    for weight in lora_a_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]

    assert inputs.size(0) == token_nums
    assert inputs.size(1) == lora_a_weights[0].size(-1)
    assert b_seq_start_loc.size(0) == batches
    assert lora_indices_tensor.size(0) == batches
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()
    (lora_ptr_tensor, lora_strides_d0, lora_strides_d1,
     lora_strides_d2) = _get_lora_a_ptr(lora_a_weights, b_seq_start_loc.device)
    # TODO tuning this config
    N, K = lora_a_weights[0].shape[-2:]  # K=hidden_size,N=rank
    BLOCK_M = 32
    BLOCK_N = 16
    BLOCK_K = 32
    SPLIT_K = 8
    EVEN_K = K % (BLOCK_K * SPLIT_K) == 0
    grid = (
        triton.cdiv(max_seq_length, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        SPLIT_K * len(lora_a_weights),
        batches,
    )
    _sgmv_shrink_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        N,
        K,
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        scaling,
        inputs.stride(0),
        inputs.stride(1),
        lora_strides_d0,
        lora_strides_d1,
        lora_strides_d2,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor.stride(2),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        len(lora_a_weights),
    )
    return


def sgmv_shrink_fake(
    inputs: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="sgmv_shrink",
        op_func=_sgmv_shrink,
        mutates_args=["output_tensor"],
        fake_impl=sgmv_shrink_fake,
    )
    sgmv_shrink = torch.ops.vllm.sgmv_shrink

except AttributeError:
    sgmv_shrink = _sgmv_shrink
