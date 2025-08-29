# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from vllm.distributed import get_cp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _correct_attn_cp_out_kernel(outputs_ptr, new_output_ptr, lses_ptr,
                                vlse_ptr, outputs_stride_B, outputs_stride_H,
                                outputs_stride_D, lses_stride_N, lses_stride_B,
                                lses_stride_H, lse_idx, HEAD_DIM: tl.constexpr,
                                N_ROUNDED: tl.constexpr):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        output: [ B, H, D ]
        lses   : [ N, B, H ]
        cp, batch, q_heads, v_head_dim
    Return:
        output: [ B, H, D ]
        lse   : [ B, H ]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)

    # shape = [N]
    lse_offsets = num_n_offsets * lses_stride_N + batch_idx * \
        lses_stride_B + head_idx * lses_stride_H

    # calc final lse
    lse = tl.load(lses_ptr + lse_offsets)
    lse = tl.where(lse != lse or lse == float('inf'), -float('inf'), lse)
    lse_max = tl.max(lse, axis=0)
    lse -= lse_max
    lse_exp = tl.exp(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    lse = tl.log(lse_acc)
    lse += lse_max

    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)

    # shape = [D]
    output_offsets = batch_idx * outputs_stride_B + \
                    head_idx * outputs_stride_H + \
                    d_offsets * outputs_stride_D

    # correct output
    lse_offset = lse_idx * lses_stride_N + batch_idx * \
        lses_stride_B + head_idx * lses_stride_H
    lse_tmp = tl.load(lses_ptr + lse_offset)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        lse_finally != lse_finally or lse_finally == float('inf'),
        -float('inf'), lse_finally)
    factor = tl.exp(lse_finally)
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor

    tl.store(new_output_ptr + output_offsets, output)


class CPTritonContext:
    """ The CPTritonContext is used to avoid recompilation of the Triton JIT.
    """

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


def correct_attn_out(out: torch.Tensor,
                     lses: torch.Tensor,
                     cp_rank,
                     ctx: CPTritonContext = None):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        output: [ B, H, D ]
        lses   : [ N, B, H ]
    Return:
        output: [ B, H, D ]
        lse   : [ B, H ]
    """
    if ctx is None:
        ctx = CPTritonContext()

    lse = torch.empty_like(lses[0])

    grid = (out.shape[0], out.shape[1], 1)
    regular_args = (out, out, lses, lse, *out.stride(), *lses.stride(),
                    cp_rank)
    const_args = {
        "HEAD_DIM": out.shape[-1],
        "N_ROUNDED": lses.shape[0],
    }

    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args,
                    **const_args)
    return out, lse


def cp_lse_ag_out_rs(cp_attn_out,
                     cp_attn_lse,
                     cp_group,
                     ctx: CPTritonContext = None):
    if cp_group.world_size == 1:
        return cp_attn_out

    if ctx is None:
        ctx = CPTritonContext()

    lses = torch.empty((cp_group.world_size, ) + cp_attn_lse.shape,
                       dtype=cp_attn_lse.dtype,
                       device=cp_attn_lse.device)

    cp_attn_lse = cp_attn_lse.contiguous()
    lses = cp_group.all_gather(cp_attn_lse, dim=0).view_as(lses)
    out, _ = correct_attn_out(cp_attn_out, lses,
                              dist.get_rank(cp_group.device_group), ctx)
    assert out.is_contiguous()
    out = cp_group.reduce_scatter(out, dim=1)
    return out


if current_platform.is_cuda():
    try:
        import vllm._flashmla_C  # noqa: F401
        _flashmla_C_AVAILABLE = True
    except ImportError:
        _flashmla_C_AVAILABLE = False
else:
    _flashmla_C_AVAILABLE = False


def is_flashmla_supported() -> Tuple[bool, Optional[str]]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    if not current_platform.is_cuda():
        return False, "FlashMLA is only supported on CUDA devices."
    if current_platform.get_device_capability()[0] != 9:
        return False, "FlashMLA is only supported on Hopper devices."
    if not _flashmla_C_AVAILABLE:
        return False, "vllm._flashmla_C is not available, likely was not "\
            "compiled due to insufficient nvcc version or a supported arch "\
            "(only sm90a currently) was not in the list of target arches to "\
            "compile for."
    return True, None


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Return:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), 
                                 dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    return torch.ops._flashmla_C.get_mla_metadata(cache_seqlens,
                                                  num_heads_per_head_k,
                                                  num_heads_k)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), 
                                 torch.int32, return by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
        softmax_scale: float. The scaling of QK^T before applying softmax. 
                       Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.
        descale_q: (batch_size), torch.float32. Descaling factors for Q.
        descale_k: (batch_size), torch.float32. Descaling factors for K.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    out, softmax_lse = torch.ops._flashmla_C.fwd_kvcache_mla(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
    )
    return out, softmax_lse


def flash_mla_with_kvcache_for_cp(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), 
                                 torch.int32, return by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
        softmax_scale: float. The scaling of QK^T before applying softmax. 
                       Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    assert descale_q is None and descale_k is None, \
        "cp not support scaled kv now."
    assert q.shape[1] == 1, "cp not support seq_len_q gt 1 now."
    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    q = get_cp_group().all_gather(q, dim=2)
    out, softmax_lse = torch.ops._flashmla_C.fwd_kvcache_mla(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
    )
    out = out.squeeze(1)
    softmax_lse = softmax_lse.squeeze(-1)
    assert out.is_contiguous()
    out = cp_lse_ag_out_rs(out, softmax_lse, get_cp_group())
    return out, softmax_lse


#
# TODO: Add fake functions
#
# @register_fake("_flashmla_C::get_mla_metadata")
# def _get_mla_metadata_fake(....) -> Tuple[torch.Tensor, torch.Tensor]:
#     return ....
#
# @register_fake("_flashmla_C::fwd_kvcache_mla")
# def _fwd_kvcache_mla_fake(....) -> Tuple[torch.Tensor, torch.Tensor]:
#     return ....
#
