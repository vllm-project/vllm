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

from .utils import _get_lora_a_ptr, get_v1_op_configs
from .kernel_utils import do_shrink_kernel



@triton.jit
def _v1_shrink_kernel(
        input_ptr,
        lora_ptr,  #1-3
        out_ptr,
        M,
        N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
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


    pid_lmn = tl.program_id(axis=0)
    pid_sk_slice = tl.program_id(axis=1)

    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)
    lora_idx = pid_lmn // (cta_m_num * cta_n_num)
    pid_n = (pid_lmn // cta_m_num) % cta_n_num
    pid_m = pid_lmn % cta_m_num

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        # early exit for the no-lora case.
        return

    if SLICE_NUM == 1:
        slice_id: tl.constexpr = 0
        pid_sk = pid_sk_slice 
    else:
        slice_id = pid_sk_slice // SPLIT_K
        pid_sk = pid_sk_slice % SPLIT_K


    # lora m indices offsets
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        # early exit CTA
        return

    CTA_M_LEN = min(BLOCK_M, lora_m_size - cta_m_offset)
    offset_m = tl.arange(0, BLOCK_M) % CTA_M_LEN 

    cta_lora_seq_indices = (token_indices_sorted_by_lora_ids +
                            lora_m_indices_start + cta_m_offset)
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_shrink_kernel(pid_m,
                     pid_n,
                     pid_sk,
                     slice_id,
                     lora_id,
                     input_ptr,
                     lora_ptr,
                     out_ptr,
                     N,
                     K,
                     CTA_M_LEN,
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
def _v1_shrink(
    inputs: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,  # inputs.size(0)
    num_tokens_per_lora: torch.Tensor,  # max-loras
    lora_token_start_loc: torch.Tensor,  # max-loras
    lora_ids: torch.Tensor,  # max-loras
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

    assert inputs.size(1) == lora_a_weights[0].size(-1)
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()
    (lora_ptr_tensor, lora_strides_d0, lora_strides_d1,
     lora_strides_d2) = _get_lora_a_ptr(lora_a_weights, inputs.device)
    # TODO tuning this config
    N, K = lora_a_weights[0].shape[-2:]  # K=hidden_size,N=rank
    M = inputs.size(0)
    NUM_SLICES = len(lora_a_weights)

    kernel_config = get_v1_op_configs("shrink",
                                        batch = M, 
                                        hidden_size = K,
                                        rank = N,
                                        num_slices = NUM_SLICES)
    BLOCK_M = kernel_config['block_m']
    BLOCK_N = kernel_config['block_n']
    BLOCK_K = kernel_config['block_k']
    SPLIT_K = kernel_config['split_k']

    EVEN_K = K % (BLOCK_K * SPLIT_K) == 0
    MAX_LORAS = lora_ids.size(0)
    grid = (
        MAX_LORAS * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        SPLIT_K * len(lora_a_weights),
    )

    _v1_shrink_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        N,
        K,
        # New additions 
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        # ----
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
        NUM_SLICES,
        num_warps = kernel_config['num_warps'],
        num_ctas = kernel_config['num_ctas'],
        num_stages = kernel_config['num_stages'],
        maxnreg = kernel_config['max_nreg'],
    )

    return


def _v1_shrink_fake(
    inputs: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,  # inputs.size(0)
    num_tokens_per_lora: torch.Tensor,  # max-loras
    lora_token_start_loc: torch.Tensor,  # max-loras
    lora_ids: torch.Tensor,  # max-loras
    scaling: float,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="v1_shrink",
        op_func=_v1_shrink,
        mutates_args=["output_tensor"],
        fake_impl=_v1_shrink_fake,
    )
    v1_shrink = torch.ops.vllm.v1_shrink

except AttributeError:
    v1_shrink = _v1_shrink
