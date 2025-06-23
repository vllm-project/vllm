# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch
# Use relative import to support build-from-source installation in vLLM

try:
    from . import _vllm_fa2_C  # noqa: F401
    FA2_UNAVAILABLE_REASON = None
    FA2_AVAILABLE = True
except ImportError as e:
    FA2_UNAVAILABLE_REASON = str(e)
    FA2_AVAILABLE = False

try:
    from . import _vllm_fa3_C  # noqa: F401
    FA3_UNAVAILABLE_REASON = None
    FA3_AVAILABLE = True
except ImportError as e:
    FA3_UNAVAILABLE_REASON = str(e)
    FA3_AVAILABLE = False

# isort: on

DEFAULT_FA_VERSION = 2

def _is_fa2_supported(device = None) -> Tuple[bool, Optional[str]]:
    if not FA2_AVAILABLE:
        return False, f"FA2 is unavaible due to: {FA2_UNAVAILABLE_REASON}"
    if torch.cuda.get_device_capability(device)[0] < 8:
        return False, \
            "FA2 is only supported on devices with compute capability >= 8"
    return True, None
    
def _is_fa3_supported(device = None) -> Tuple[bool, Optional[str]]:
    if not FA3_AVAILABLE:
        return False, f"FA3 is unavaible due to: {FA3_UNAVAILABLE_REASON}"
    if torch.cuda.get_device_capability(device)[0] < 8 \
        or torch.cuda.get_device_capability(device)[0] >= 10 \
        or torch.cuda.get_device_capability(device) == (8, 6) \
        or torch.cuda.get_device_capability(device) == (8, 9):
        return False, \
            "FA3 is only supported on devices with compute capability >= 8" \
            " excluding 8.6 and 8.9 and Blackwell archs (>=10)"
    return True, None

def is_fa_version_supported(fa_version: int, device = None) -> bool:
    assert fa_version in [2, 3], f"Unsupported FA version: {fa_version}"
    if fa_version == 2:
        return _is_fa2_supported(device)[0]
    elif fa_version == 3:
        return _is_fa3_supported(device)[0]

def fa_version_unsupported_reason(fa_version: int, device = None) \
    -> Optional[str]:
    assert fa_version in [2, 3], f"Unsupported FA version: {fa_version}"
    if fa_version == 2:
        return _is_fa2_supported(device)[1]
    elif fa_version == 3:
        return _is_fa3_supported(device)[1]

#
#  For vLLM we only care about `flash_attn_varlen_func` and 
#   `flash_attn_with_kvcache` so we only maintain wrappers for these two.
#


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

# NOTE only used in FA3
def get_scheduler_metadata(
    batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim,
    cache_seqlens: torch.Tensor,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    has_softcap=False,
    num_splits=0,    # Can be tuned for speed
    pack_gqa=None,   # Can be tuned for speed
    sm_margin=0,     # Can be tuned if some SMs are used for communication
):
    cache_seqlens = maybe_contiguous(cache_seqlens)
    if headdim_v is None:
        headdim_v = headdim
    scheduler_metadata = torch.ops._vllm_fa3_C.get_scheduler_metadata(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim, headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_leftpad,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size[0], window_size[1],
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
    )

    return scheduler_metadata


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None, # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: Optional[List[int]] = None,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # FA3 Only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    # Version selector
    fa_version: int = DEFAULT_FA_VERSION,
):
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
    assert cu_seqlens_k is not None or seqused_k is not None, \
        "cu_seqlens_k or seqused_k must be provided"
    assert cu_seqlens_k is None or seqused_k is None, \
        "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert block_table is None or seqused_k is not None, \
        "seqused_k must be provided if block_table is provided"
    
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    # custom op does not support non-tuple input
    real_window_size: Tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    
    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)
    
    if fa_version == 2:
        if scheduler_metadata is not None and q_descale is not None \
            and k_descale is not None and v_descale is not None:
                raise NotImplementedError(
                    "FA2 does not support scheduler_metadata, q_descale, "
                    "k_descale, v_descale"
                )
        out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(
            q, k, v,
            out,
            cu_seqlens_q,
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp 
            # still wants it so we pass all zeros
            dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
            seqused_k,
            None,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            return_softmax_lse and dropout_p > 0,
            None,
        )
    elif fa_version == 3:
        assert alibi_slopes is None, "Alibi is not supported in FA3"
        out, softmax_lse, _, _ = torch.ops._vllm_fa3_C.fwd(
            q, k, v,
            None, None,       # k_new, v_new
            q_v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,     # cu_seqlens_k
            None,             # cu_seqlens_k_new
            None, seqused_k,  # seqused_q, seqused_k
            max_seqlen_q, max_seqlen_k,
            block_table,
            None,             # kv_batch_idx
            None,             # leftpad_k
            None, None, None, # rotary_cos, rotary_sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal,
            real_window_size[0], real_window_size[1],
            softcap,
            True,             # rotary_interleaved
            scheduler_metadata,
            0,                # num_splits
            None,             # pack_gqa
            0,                # sm_margin
        )
    else:
        raise ValueError(f"Unsupported FA version: {fa_version}")
    return (out, softmax_lse) if return_softmax_lse else out


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
    *,
    out=None,
    # FA3 Only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    # Version selector
    fa_version: int = DEFAULT_FA_VERSION,
):
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
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)

    if fa_version == 2:
        if scheduler_metadata is not None and q_descale is not None \
            and k_descale is not None and v_descale is not None:
                raise NotImplementedError(
                    "FA2 does not support scheduler_metadata, q_descale, "
                    "k_descale, v_descale"
                )
        out, softmax_lse = torch.ops._vllm_fa2_C.fwd_kvcache(
            q, k_cache, v_cache,
            k, v,             # k_new, v_new
            cache_seqlens,
            rotary_cos,
            rotary_sin,
            cache_batch_idx,
            cache_leftpad,
            block_table,
            alibi_slopes,
            out,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
            softcap,
            rotary_interleaved,
            num_splits,
        )
    elif fa_version == 3:
        assert alibi_slopes is None, "Alibi is not supported in FA3"
        out, softmax_lse, _, _ = torch.ops._vllm_fa3_C.fwd(
            q, k_cache, v_cache, # q, k, v
            k, v,                # k_new, v_new
            None,                # q_v
            out,
            None, None,          # cu_seqlens_q, cu_seqlens_k
            None,                # cu_seqlens_k_new
            None, cache_seqlens, # seqused_q, seqused_k
            None, None,          # max_seqlen_q, max_seqlen_k
            block_table,
            cache_batch_idx,     # kv_batch_idx
            None,                # leftpad_k
            None, None, None,    # rotary_cos, rotary_sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal,
            window_size[0], window_size[1],
            softcap,
            rotary_interleaved,  # rotary_interleaved
            scheduler_metadata,
            num_splits,          # num_splits
            None,                # pack_gqa
            0,                   # sm_margin
        )
    else:
        raise ValueError(f"Unsupported FA version: {fa_version}")
    return (out, softmax_lse) if return_softmax_lse else out


def sparse_attn_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
):
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
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse = torch.ops._vllm_fa2_C.fwd_sparse(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        out,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        softcap,
        return_attn_probs and dropout_p > 0,
        None,
    )
    return (out, softmax_lse) if return_softmax_lse else out


def sparse_attn_varlen_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
):
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
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
        
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd_sparse(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        softcap,
        return_attn_probs and dropout_p > 0,
        None,
    )
    return (out, softmax_lse) if return_softmax_lse else out
