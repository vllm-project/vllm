# SPDX-License-Identifier: Apache-2.0
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

from .kernel_utils import do_shrink_kernel
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
    if pid_m * BLOCK_M >= M:
        return
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return

    m_offset = tl.load(b_seq_start_loc + cur_batch)

    cta_m_len = min(BLOCK_M, M - (pid_m * BLOCK_M))
    cta_m_offset = m_offset + (pid_m * BLOCK_M)
    offset_m = tl.arange(0, BLOCK_M)
    ram = cta_m_offset + tl.max_contiguous(
        tl.multiple_of(offset_m % cta_m_len, BLOCK_M), BLOCK_M)

    do_shrink_kernel(
        pid_n,
        pid_sk,
        slice_id,
        lora_index,
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        cta_m_len,
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
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        SLICE_NUM)


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
