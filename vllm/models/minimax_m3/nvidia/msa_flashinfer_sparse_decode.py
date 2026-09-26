# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer sparse decode adapter for MiniMax M3 on SM100."""

from __future__ import annotations

import torch

from vllm.config.attention import MiniMaxM3MSADecodeBackend
from vllm.platforms import current_platform

_HEAD_DIM = 128
_PAGE_SIZE = 128
_TOPK = 16


def supports_flashinfer_sparse_decode(
    *,
    decode_backend: MiniMaxM3MSADecodeBackend,
    num_q_heads: int,
    num_kv_heads: int,
    kv_cache_dtype: str,
    page_size: int,
    topk_blocks: int,
) -> bool:
    """Return whether static geometry supports packed FlashInfer MSA decode."""

    return (
        decode_backend == "flashinfer"
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and kv_cache_dtype in ("fp8", "fp8_e4m3")
        and 0 < num_kv_heads <= num_q_heads
        and num_q_heads % num_kv_heads == 0
        and num_q_heads // num_kv_heads <= 16
        and page_size == _PAGE_SIZE
        and topk_blocks == _TOPK
    )


def _validate_inputs(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    q2k_indices: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
) -> None:
    if query.ndim != 3 or query.shape[-1] != _HEAD_DIM:
        raise ValueError("query must have shape (total_q, num_q_heads, 128)")
    if query.dtype != torch.float8_e4m3fn:
        raise ValueError("FlashInfer packed MSA decode requires an FP8 E4M3 query")
    if query.shape[0] % decode_query_len != 0:
        raise ValueError("query rows must be divisible by decode_query_len")
    if kv_cache.ndim != 4 or kv_cache.shape[2:] != (
        _PAGE_SIZE,
        2 * _HEAD_DIM,
    ):
        raise ValueError("kv_cache must have shape (num_pages, num_kv_heads, 128, 256)")
    if kv_cache.dtype != torch.float8_e4m3fn or not kv_cache.is_contiguous():
        raise ValueError("kv_cache must be a contiguous FP8 E4M3 packed cache")

    total_q, num_q_heads, _ = query.shape
    num_kv_heads = kv_cache.shape[1]
    if num_q_heads % num_kv_heads != 0 or num_q_heads // num_kv_heads > 16:
        raise ValueError("FlashInfer MSA requires a GQA group size of at most 16")
    if q2k_indices.shape != (num_kv_heads, total_q, _TOPK):
        raise ValueError("q2k_indices must have shape (num_kv_heads, total_q, 16)")
    if q2k_indices.dtype != torch.int32 or (
        not q2k_indices.is_contiguous()
        and q2k_indices.stride() != (16, num_kv_heads * 16, 1)
    ):
        raise ValueError(
            "q2k_indices must be compact head-major int32 or an exact compact "
            "token-major transpose"
        )

    batch_size = total_q // decode_query_len
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch_size
        or block_table.dtype != torch.int32
        or not block_table.is_contiguous()
    ):
        raise ValueError(
            "block_table must be contiguous int32 with one row per request"
        )
    if (
        seq_lens.shape != (batch_size,)
        or seq_lens.dtype != torch.int32
        or not seq_lens.is_contiguous()
    ):
        raise ValueError("seq_lens must be contiguous int32 with one entry per request")

    tensors = (
        kv_cache,
        q2k_indices,
        block_table,
        seq_lens,
    )
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("all FlashInfer MSA inputs must be on the query device")


@torch.no_grad()
def msa_flashinfer_sparse_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    q2k_indices: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    *,
    scale: float,
    q_scale_float: float = 1.0,
    k_scale_float: float = 1.0,
    v_scale_float: float = 1.0,
    out: torch.Tensor,
    lse_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run paged MSA decode directly on vLLM's packed K/V cache."""
    if decode_query_len <= 0:
        raise ValueError("decode_query_len must be positive")
    _validate_inputs(
        query,
        kv_cache,
        q2k_indices,
        block_table,
        seq_lens,
        decode_query_len,
    )

    from flashinfer.msa_ops import msa_sparse_decode_attention

    key_cache, value_cache = kv_cache.split(_HEAD_DIM, dim=-1)

    result = msa_sparse_decode_attention(
        query,
        key_cache,
        value_cache,
        q2k_indices,
        page_table=block_table,
        seqused_k=seq_lens,
        seqlen_q=decode_query_len,
        causal=True,
        softmax_scale=scale * q_scale_float,
        return_softmax_lse=True,
        k_global_scale=k_scale_float,
        v_global_scale=v_scale_float,
        force_fused=True,
        out=out,
        lse_out=lse_out,
    )
    assert isinstance(result, tuple)
    output, lse = result
    return output, lse
