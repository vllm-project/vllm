# ruff: ignore
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

if TYPE_CHECKING:
    import torch

def get_scheduler_metadata(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    cache_seqlens: torch.Tensor,
    qkv_dtype: torch.dtype = ...,
    headdim_v: int | None = ...,
    cu_seqlens_q: torch.Tensor | None = ...,
    cu_seqlens_k_new: torch.Tensor | None = ...,
    cache_leftpad: torch.Tensor | None = ...,
    page_size: int = ...,
    max_seqlen_k_new: int = ...,
    causal: bool = ...,
    window_size: tuple[int, int] = ...,  # -1 means infinite context window
    has_softcap: bool = ...,
    num_splits: int = ...,  # Can be tuned for speed
    pack_gqa: Any | None = ...,  # Can be tuned for speed
    sm_margin: int = ...,  # Can be tuned if some SMs are used for communication
): ...
@overload
def flash_attn_varlen_func(
    q: tuple[int, int, int],
    k: tuple[int, int, int],
    v: tuple[int, int, int],
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor | None,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor | None = ...,
    seqused_k: Any | None = ...,
    q_v: Any | None = ...,
    dropout_p: float = ...,
    causal: bool = ...,
    window_size: list[int] | None = ...,
    softmax_scale: float = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    deterministic: bool = ...,
    return_attn_probs: bool = ...,
    block_table: Any | None = ...,
    return_softmax_lse: Literal[False] = ...,
    out: Any = ...,
    # FA3 Only
    scheduler_metadata: Any | None = ...,
    q_descale: Any | None = ...,
    k_descale: Any | None = ...,
    v_descale: Any | None = ...,
    # Version selector
    fa_version: int = ...,
) -> tuple[int, int, int]: ...
@overload
def flash_attn_varlen_func(
    q: tuple[int, int, int],
    k: tuple[int, int, int],
    v: tuple[int, int, int],
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor | None,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor | None = ...,
    seqused_k: Any | None = ...,
    q_v: Any | None = ...,
    dropout_p: float = ...,
    causal: bool = ...,
    window_size: list[int] | None = ...,
    softmax_scale: float = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    deterministic: bool = ...,
    return_attn_probs: bool = ...,
    block_table: Any | None = ...,
    return_softmax_lse: Literal[True] = ...,
    out: Any = ...,
    # FA3 Only
    scheduler_metadata: Any | None = ...,
    q_descale: Any | None = ...,
    k_descale: Any | None = ...,
    v_descale: Any | None = ...,
    # Version selector
    fa_version: int = ...,
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """

@overload
def flash_attn_with_kvcache(
    q: tuple[int, int, int, int],
    k_cache: tuple[int, int, int, int],
    v_cache: tuple[int, int, int, int],
    k: tuple[int, int, int, int] | None = ...,
    v: tuple[int, int, int, int] | None = ...,
    rotary_cos: tuple[int, int] | None = ...,
    rotary_sin: tuple[int, int] | None = ...,
    cache_seqlens: int | torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = ...,
    block_table: torch.Tensor | None = ...,
    softmax_scale: float = ...,
    causal: bool = ...,
    window_size: tuple[int, int] = ...,  # -1 means infinite context window
    softcap: float = ...,
    rotary_interleaved: bool = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    num_splits: int = ...,
    return_softmax_lse: Literal[False] = ...,
    *,
    out: Any = ...,
    # FA3 Only
    scheduler_metadata: Any | None = ...,
    q_descale: Any | None = ...,
    k_descale: Any | None = ...,
    v_descale: Any | None = ...,
    # Version selector
    fa_version: int = ...,
) -> tuple[int, int, int, int]: ...
@overload
def flash_attn_with_kvcache(
    q: tuple[int, int, int, int],
    k_cache: tuple[int, int, int, int],
    v_cache: tuple[int, int, int, int],
    k: tuple[int, int, int, int] | None = ...,
    v: tuple[int, int, int, int] | None = ...,
    rotary_cos: tuple[int, int] | None = ...,
    rotary_sin: tuple[int, int] | None = ...,
    cache_seqlens: int | torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = ...,
    block_table: torch.Tensor | None = ...,
    softmax_scale: float = ...,
    causal: bool = ...,
    window_size: tuple[int, int] = ...,  # -1 means infinite context window
    softcap: float = ...,
    rotary_interleaved: bool = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    num_splits: int = ...,
    return_softmax_lse: Literal[True] = ...,
    *,
    out: Any = ...,
    # FA3 Only
    scheduler_metadata: Any | None = ...,
    q_descale: Any | None = ...,
    k_descale: Any | None = ...,
    v_descale: Any | None = ...,
    # Version selector
    fa_version: int = ...,
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """

@overload
def sparse_attn_func(
    q: tuple[int, int, int, int],
    k: tuple[int, int, int, int],
    v: tuple[int, int, int, int],
    block_count: tuple[int, int, float],
    block_offset: tuple[int, int, float, int],
    column_count: tuple[int, int, float],
    column_index: tuple[int, int, float, int],
    dropout_p: float = ...,
    softmax_scale: float = ...,
    causal: bool = ...,
    softcap: float = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    deterministic: bool = ...,
    return_attn_probs: bool = ...,
    *,
    return_softmax_lse: Literal[False] = ...,
    out: Any = ...,
) -> tuple[int, int, int]: ...
@overload
def sparse_attn_func(
    q: tuple[int, int, int, int],
    k: tuple[int, int, int, int],
    v: tuple[int, int, int, int],
    block_count: tuple[int, int, float],
    block_offset: tuple[int, int, float, int],
    column_count: tuple[int, int, float],
    column_index: tuple[int, int, float, int],
    dropout_p: float = ...,
    softmax_scale: float = ...,
    causal: bool = ...,
    softcap: float = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    deterministic: bool = ...,
    return_attn_probs: bool = ...,
    *,
    return_softmax_lse: Literal[True] = ...,
    out: Any = ...,
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """

@overload
def sparse_attn_varlen_func(
    q: tuple[int, int, int],
    k: tuple[int, int, int],
    v: tuple[int, int, int],
    block_count: tuple[int, int, float],
    block_offset: tuple[int, int, float, int],
    column_count: tuple[int, int, float],
    column_index: tuple[int, int, float, int],
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = ...,
    softmax_scale: float = ...,
    causal: bool = ...,
    softcap: float = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    deterministic: bool = ...,
    return_attn_probs: bool = ...,
    *,
    return_softmax_lse: Literal[False] = ...,
    out: Any = ...,
) -> tuple[int, int, int]: ...
@overload
def sparse_attn_varlen_func(
    q: tuple[int, int, int],
    k: tuple[int, int, int],
    v: tuple[int, int, int],
    block_count: tuple[int, int, float],
    block_offset: tuple[int, int, float, int],
    column_count: tuple[int, int, float],
    column_index: tuple[int, int, float, int],
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = ...,
    softmax_scale: float = ...,
    causal: bool = ...,
    softcap: float = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    deterministic: bool = ...,
    return_attn_probs: bool = ...,
    *,
    return_softmax_lse: Literal[True] = ...,
    out: Any = ...,
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_varlen_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """

def is_fa_version_supported(
    fa_version: int, device: torch.device | None = None
) -> bool: ...
def fa_version_unsupported_reason(
    fa_version: int, device: torch.device | None = None
) -> str | None: ...
