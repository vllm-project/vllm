# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py
from typing import Optional, Tuple

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_cuda():
    try:
        import vllm._flashmla_C  # noqa: F401
        import vllm._flashmla_sparse_C  # noqa: F401
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

    # Note(hc): need revisit when we support DCP with decode query_len > 1.
    return out.squeeze(1), softmax_lse.squeeze(-1)


# ------------------------ Sparse FlashMLA bindings -------------------------


def get_sparse_mla_metadata(
    cache_seqlens: torch.Tensor,
    q_seq_per_hk: int,
    num_heads_k: int,
    topk: int,
    q_heads_per_hk: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        q_seq_per_hk: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.
        topk: topk
        q_heads_per_hk: equals to num_heads_q // num_heads_k. Only need to be
            specified when topk is not None.

    Return:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize),
            dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    return torch.ops._flashmla_sparse_C.get_mla_metadata(
        cache_seqlens, q_seq_per_hk, num_heads_k, topk, q_heads_per_hk)


def flash_mla_sparse_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    indices_in_kvcache: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize),
            torch.int32, returned by get_sparse_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, returned by
            get_sparse_mla_metadata.
        indices_in_kvcache: (batch_size, seq_len_q, topk). KV indices when
            sparse attention is enabled. Note that
            indices_in_kvcache[i][j][k] =
              (the index of the page block where token t resides) *
              page_block_size + (the offset of token t within that page block),
            where t is the k-th token of the j-th q-sequence in the i-th batch.
        softmax_scale: float. Scaling of QK^T before softmax.
            Defaults to 1 / sqrt(head_dim).

    Explanation of K/V cache layout:
        We quantize the NoPE part of each token (in 1x128 granularity),
        yielding 512 float8_e4m3 values and 4 float32 scale factors. For the
        RoPE part, we keep it as 64 bfloat16. Each token occupies 656 bytes:
        - First 512 bytes: quantized NoPE (512 x float8_e4m3)
        - Next 16 bytes: scale factors (4 x float32)
        - Last 128 bytes: RoPE (64 x bfloat16)

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    # Strict shape checks like the reference implementation
    assert k_cache.shape[-1] == 656, (
        "The last dim of k_cache must be 656 (=512+2*16+4*4) when "
        "is_fp8_kvcache is True")
    assert k_cache.shape[-2] == 1, (
        "The number of K heads must be 1 when is_fp8_kvcache is True")

    out, softmax_lse = torch.ops._flashmla_sparse_C.fwd_kvcache_mla(
        q, k_cache, head_dim_v, cache_seqlens, softmax_scale,
        tile_scheduler_metadata, num_splits, indices_in_kvcache)
    return out, softmax_lse


def flash_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention forward operator, for prefill

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1, 
                 or to a number >= s_kv
        sm_scale: float, scaling factor for the attention scores
        d_v: dimension of the value, default (and only supported) is 512

    Returns:
        Returns (output, max_logits, lse)
        - output: [s_q, h_q, d_v], bfloat16, the result of attention
        - max_logits: [s_q, h_q], float
        - lse: [s_q, h_q], float, base-2
    """
    results = torch.ops._flashmla_sparse_C.sparse_topk_attn_fwd(
        q, kv, indices, sm_scale, d_v)
    return results


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
