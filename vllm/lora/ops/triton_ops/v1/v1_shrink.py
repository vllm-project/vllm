# SPDX-License-Identifier: Apache-2.0
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

from typing import List
from itertools import product

import torch
import triton
import triton.language as tl

from vllm.lora.ops.triton_ops.kernel_utils import do_shrink_kernel
from vllm.lora.ops.triton_ops.utils import _get_lora_a_ptr
from vllm.utils import direct_register_custom_op

def split_k_ranges():
    return [8, 32, 64, 128]


def block_m_ranges():
    return [16, 32, 64, 128, 256, 512]


def block_n_ranges():
    return [16]


def block_k_ranges():
    return [32, 64, 128, 256, 512, 1024]


def warp_ranges():
    return [4, 8]


def cta_ranges():
    return [1]


def cta_num_stages():
    return [2, 4]


def autotune_configs():
    return [
        triton.Config(kwargs={
            'BLOCK_M': bm,
            'BLOCK_N': bn,
            'BLOCK_K': bk,
            'SPLIT_K': sk
        },
                      num_warps=nw,
                      num_ctas=nc,
                      num_stages=ns)
        for bm, bn, bk, sk, nw, nc, ns in product(
            block_m_ranges(), block_n_ranges(), block_k_ranges(),
            split_k_ranges(), warp_ranges(), cta_ranges(), cta_num_stages())
    ]


def prune_fn(*args, **kwargs):
    configs_list, kernel_kwargs = args

    # prune such that EVEN_K is true

    def is_even_k_good(config, kkwargs):
        return kkwargs['K'] % (config.kwargs['BLOCK_K'] *
                               config.kwargs['SPLIT_K']) == 0

    def is_m_good(config, kkwargs):
        return config.kwargs['BLOCK_M'] == 16 or config.kwargs[
            'BLOCK_M'] <= kkwargs['M']

    pruned = filter(
        lambda x: is_even_k_good(x, kernel_kwargs) and is_m_good(
            x, kernel_kwargs), configs_list)
    pruned = list(pruned)
    print(f"Trying #configs {len(pruned)}")
    return pruned


@triton.autotune(configs=autotune_configs(),
                 key=['M', 'N', 'K', 'SLICE_NUM', 'MAX_LORAS'],
                 restore_value=["out_ptr"],
                 prune_configs_by={'early_config_prune': prune_fn})
@triton.jit
def _v1_shrink_kernel(input_ptr, lora_ptr, out_ptr, M, N, K,
                      token_indices_sorted_by_lora_ids, num_tokens_per_lora,
                      lora_token_start_loc, lora_ids, scaling, input_d0_stride,
                      input_d1_stride, lora_d0_stride, lora_d1_stride,
                      lora_d2_stride, output_d0_stride, output_d1_stride,
                      output_d2_stride,
                      EVEN_K: tl.constexpr,
                      SLICE_NUM: tl.constexpr,
                      MAX_LORAS: tl.constexpr,
                      BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr,
                      SPLIT_K: tl.constexpr):


    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)

    pid_sk_m_n = tl.program_id(axis=0)
    pid_sk = pid_sk_m_n % SPLIT_K
    pid_m = (pid_sk_m_n // SPLIT_K) % cta_m_num
    pid_n = pid_sk_m_n // (SPLIT_K * cta_m_num) % cta_n_num

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        # Early exit for the no-lora case.
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        # Early exit CTA.
        return

    # num rows this CTA should process.
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)

    # Identify all rows that this CTA should process.
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (token_indices_sorted_by_lora_ids +
                            lora_m_indices_start + cta_m_offset)

    # Load all relevant row indices.
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_shrink_kernel(
        pid_n,
        pid_sk,
        slice_id,
        lora_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        cta_m_len,
        ram,  # array identifying the rows of Input ptr to operate on
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
    inputs: torch.Tensor,  #  shape [num_tokens, hidden_size]
    lora_a_weights: List[
        torch.Tensor],  # shape [num_loras, lora_rank, hidden_size]
    output_tensor: torch.Tensor,  # shape [num_slices, num_tokens, lora_rank]
    token_lora_mapping: torch.Tensor,  # shape [num_tokens]
    token_indices_sorted_by_lora_ids: torch.Tensor,  # shape [num_tokens] 
    num_tokens_per_lora: torch.Tensor,  # shape [max-loras + 1]
    lora_token_start_loc: torch.Tensor,  # shape [max-loras + 2]
    lora_ids: torch.Tensor,  # shape [max-loras + 1]
    scaling: float,
) -> None:
    """
    Args:
        inputs (torch.Tensor): Input tensor
        lora_a_weights (List[torch.Tensor]): LoRA weights
        output_tensor (torch.Tensor): output tensor
        token_lora_mapping (torch.Tensor): A tensor mapping each input token
            to the lora-id related to that token. A value of -1 indicates that
            LoRA doesn't apply to that token.
        token_indices_sorted_by_lora_ids (torch.Tensor): Row/Token indices from
            the A matrix grouped by LoRA IDs.
        num_tokens_per_lora (torch.Tensor): num_tokens_per_lora[i] is the number
            of tokens that are to be processed by LoRA ID lora_ids[i] 
        lora_token_start_loc (torch.Tensor): A cumulative sum of
            num_tokens_per_lora. lora_token_start_loc[0] is always 0 so that
            lora_token_start_loc[i], along with num_tokens_per_lora[i]
            identifies the region in token_indices_sorted_by_lora_ids that
            LoRA lora_ids[i] should process.
        lora_ids (torch.Tensor): LoRA ids to process.
        scaling (float): Scaling factor.
    """
    assert inputs.dtype == lora_a_weights[0].dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    for weight in lora_a_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]

    assert inputs.size(1) == lora_a_weights[0].size(-1)
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()

    # metadata sanity check
    assert token_lora_mapping.size(0) == token_indices_sorted_by_lora_ids.size(
        0)
    assert lora_ids.size(0) == num_tokens_per_lora.size(0)
    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1

    (lora_ptr_tensor, lora_strides_d0, lora_strides_d1,
     lora_strides_d2) = _get_lora_a_ptr(lora_a_weights, inputs.device)
    N, K = lora_a_weights[0].shape[-2:]  # K=hidden_size,N=rank
    M = inputs.size(0)
    NUM_SLICES = len(lora_a_weights)
    MAX_LORAS = lora_ids.size(0)

    grid = lambda meta: (meta['SPLIT_K'] * triton.cdiv(M, meta[
        'BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), NUM_SLICES, MAX_LORAS)

    _v1_shrink_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        scaling,
        inputs.stride(0),
        inputs.stride(1),
        lora_strides_d0,
        lora_strides_d1,
        lora_strides_d2,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor.stride(2),
        True, # EVEN_K - make sure to invoke kernels with EVEN_K=true only
        NUM_SLICES,
        MAX_LORAS)

    return


def _v1_shrink_fake(
    inputs: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
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
