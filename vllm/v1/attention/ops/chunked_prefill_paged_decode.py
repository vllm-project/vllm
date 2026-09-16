# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import math
from functools import cache, lru_cache

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import get_kv_quant_mode
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

from .prefix_prefill import context_attention_fwd
from .rdna4_splitkv import (
    _can_use_rdna4_splitkv_decode,
    try_rdna4_splitkv_paged_attention,
)

logger = init_logger(__name__)

float8_info = torch.finfo(current_platform.fp8_dtype())

_MAX_SPLITS = 16
_DEFAULT_COMPUTE_BLOCK_SIZE = 32


def _choose_compute_block_size(physical_block_size: int) -> int:
    """Choose the logical attention tile size inside a physical KV block."""
    is_pow2 = physical_block_size > 0 and not (
        physical_block_size & (physical_block_size - 1)
    )
    if is_pow2:
        return min(physical_block_size, _DEFAULT_COMPUTE_BLOCK_SIZE)
    # Stride-based addressing permits a logical tile to cross a physical page.
    return _DEFAULT_COMPUTE_BLOCK_SIZE


def _choose_fallback_block_size(physical_block_size: int) -> int:
    """Match the incumbent non-split Triton launch tile."""
    is_pow2 = physical_block_size > 0 and not (
        physical_block_size & (physical_block_size - 1)
    )
    return min(physical_block_size, 128) if is_pow2 else 32


def has_native_kv_cache_layout(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> bool:
    """Return whether KV cache blocks can use the native ROCm pairing.

    The native reshape_and_cache writer assumes packed blocks. If cache update
    needs reshape_and_cache_flash for a stride-padded hybrid layout, decode
    should use the matching Triton path too.
    """
    return (
        key_cache.stride(0) == key_cache.shape[1:].numel()
        and value_cache.stride(0) == value_cache.shape[1:].numel()
    )


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


@cache
def _get_num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def kernel_paged_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale_inv,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    num_queries_per_kv_padded: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    PHYSICAL_BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    x: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.int64,  # int
    stride_k_cache_4: tl.int64,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.int64,  # int
    filter_by_query_len: tl.constexpr,  # bool
    query_start_len_ptr,  # [num_seqs+1]
    USE_SINKS: tl.constexpr,  # bool
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len != 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0
        + query_head_idx[:, None] * query_stride_1
    )

    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # Q : (num_queries_per_kv, HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
        L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_head_idx,
            mask=head_mask,
            other=float("-inf"),
        ).to(dtype=tl.float32)
        L = tl.where(float("-inf") < M, 1.0, 0.0)

    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_head_idx, mask=head_mask, other=0.0
        )

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    # iterate through tiles
    for j in range(0, num_blocks):
        start_n = j * BLOCK_SIZE
        # Calculate the logical location within a non-standard physical block,
        # such as 544 in Qwen/Qwen3-Next-80B-A3B-Thinking.
        # Supports non-contiguous mapping
        # from logical blocks to physical blocks
        abs_token_idx = start_n + offs_n
        l_block_idx = abs_token_idx // PHYSICAL_BLOCK_SIZE
        # Vectorized loading of physical block IDs
        p_block_idx = tl.load(block_tables_ptr + block_table_offset + l_block_idx)
        internal_offsets = abs_token_idx % PHYSICAL_BLOCK_SIZE

        # 5D addressing logic of K
        k_offset = (
            p_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + internal_offsets[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )

        # 4D addressing logic of V (Slot is innermost)
        v_offset = (
            p_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[None, :] * stride_v_cache_2
            + internal_offsets[:, None] * stride_v_cache_3
        )

        # Only the final tile can straddle seq_len. Slots >= seq_len are
        # unwritten KV cache that may hold NaN/garbage; they are score-masked
        # below, but 0 * NaN = NaN would still poison the output, so mask them
        # out of the K/V loads too. Earlier tiles are fully written, so they
        # use the cheaper token-uniform dim_mask (matching the pre-0.25.0 fast
        # path) and skip the per-token predicate entirely.
        # K : (HEAD_SIZE, BLOCK_SIZE), V : (BLOCK_SIZE, HEAD_SIZE)
        if j == num_blocks - 1:
            kv_load_mask = abs_token_idx < seq_len
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & kv_load_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & kv_load_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
        else:
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )

        # Keep FP8 scales in FP32; the intermediate also preserves FNUZ decoding.
        K = K_load.to(tl.float32).to(Q.dtype) if K_load.dtype.is_fp8() else K_load

        V = V_load.to(tl.float32).to(Q.dtype) if V_load.dtype.is_fp8() else V_load

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary

        # First calculate the dot, then apply the mask.
        qk = scale * tl.dot(Q, K)
        if K_load.dtype.is_fp8():
            qk *= tl.load(k_scale)
        S = tl.where(head_mask[:, None] & seq_mask, qk, float("-inf"))

        context_len = seq_len - 1

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S, -10000)

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        # compute running maximum
        # m_j : (num_queries_per_kv,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # P : (num_queries_per_kv, BLOCK_SIZE,)
        p = tl.exp(S - m_j[:, None])
        p = tl.where(m_j[:, None] == float("-inf"), 0.0, p)

        # l_j : (num_queries_per_kv,)
        l_j = tl.sum(p, axis=1)

        # alpha : (num_queries_per_kv, )
        alpha = tl.exp(M - m_j)
        alpha = tl.where(float("-inf") == M, 0.0, alpha)

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc += tl.dot(p.to(V.dtype), V)

    # epilogue
    acc = acc / (L[:, None] + 1e-10)
    if value_cache_ptr.dtype.element_ty.is_fp8():
        acc *= tl.load(v_scale)
    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
    )

    tl.store(
        output_ptr + output_offset[:, None] + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )


@triton.jit
def kernel_paged_attention_2d_splitkv(
    mid_out_ptr,  # [num_seqs, num_query_heads, max_num_splits, head_size]
    mid_lse_ptr,  # [num_seqs, num_query_heads, max_num_splits]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, page_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, page_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    scale,
    k_scale,
    v_scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    num_queries_per_kv_padded: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    mid_out_stride_0: tl.int64,
    mid_out_stride_1: tl.int64,
    mid_out_stride_2: tl.int64,
    mid_lse_stride_0: tl.int64,
    mid_lse_stride_1: tl.int64,
    mid_lse_stride_2: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    x: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_k_cache_4: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    filter_by_query_len: tl.constexpr,
    query_start_len_ptr,  # [num_seqs + 1]
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    if filter_by_query_len:
        query_start = tl.load(query_start_len_ptr + seq_idx)
        query_stop = tl.load(query_start_len_ptr + seq_idx + 1)
        if query_stop - query_start != 1:
            return
    else:
        query_start = seq_idx

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_splits = tl.num_programs(2)
    split_len = cdiv_fn(cdiv_fn(seq_len, num_splits), BLOCK_SIZE) * BLOCK_SIZE
    split_start = split_idx * split_len
    split_end = tl.minimum(split_start + split_len, seq_len)

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )
    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE
    query_offset = (
        query_start * query_stride_0 + query_head_idx[:, None] * query_stride_1
    )
    Q = tl.load(
        query_ptr + query_offset + offs_d[None, :],
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    block_table_offset = seq_idx * block_table_stride
    offs_n = tl.arange(0, BLOCK_SIZE)
    for start_n in tl.range(split_start, split_end, BLOCK_SIZE):
        abs_token_idx = start_n + offs_n
        token_mask = abs_token_idx < split_end
        logical_block_idx = abs_token_idx // PHYSICAL_BLOCK_SIZE
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + logical_block_idx,
            mask=token_mask,
            other=0,
        )
        internal_offsets = abs_token_idx % PHYSICAL_BLOCK_SIZE

        key_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + internal_offsets[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )
        K_load = tl.load(
            key_cache_ptr + key_offset,
            mask=dim_mask[:, None] & token_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )

        value_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[None, :] * stride_v_cache_2
            + internal_offsets[:, None] * stride_v_cache_3
        )
        V_load = tl.load(
            value_cache_ptr + value_offset,
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )

        # Keep FP8 scales in FP32; the intermediate also preserves FNUZ decoding.
        K = K_load.to(tl.float32).to(Q.dtype) if K_load.dtype.is_fp8() else K_load
        V = V_load.to(tl.float32).to(Q.dtype) if V_load.dtype.is_fp8() else V_load

        scores = scale * tl.dot(Q, K)
        if K_load.dtype.is_fp8():
            scores *= tl.load(k_scale)
        scores = tl.where(
            head_mask[:, None] & token_mask[None, :], scores, float("-inf")
        )
        new_max = tl.maximum(M, tl.max(scores, axis=1))
        probabilities = tl.exp(scores - new_max[:, None])
        probabilities = tl.where(new_max[:, None] == float("-inf"), 0.0, probabilities)
        block_sum = tl.sum(probabilities, axis=1)
        alpha = tl.exp(M - new_max)
        alpha = tl.where(float("-inf") == M, 0.0, alpha)

        acc *= alpha[:, None]
        L = L * alpha + block_sum
        M = new_max
        acc += tl.dot(probabilities.to(V.dtype), V)

    if value_cache_ptr.dtype.element_ty.is_fp8():
        acc *= tl.load(v_scale)

    mid_out_offset = (
        seq_idx * mid_out_stride_0
        + query_head_idx[:, None] * mid_out_stride_1
        + split_idx * mid_out_stride_2
        + offs_d[None, :]
    )
    mid_lse_offset = (
        seq_idx * mid_lse_stride_0
        + query_head_idx * mid_lse_stride_1
        + split_idx * mid_lse_stride_2
    )
    has_tokens = split_end > split_start
    tl.store(
        mid_out_ptr + mid_out_offset,
        acc / (L[:, None] + 1e-10),
        mask=has_tokens & head_mask[:, None] & dim_mask[None, :],
    )
    tl.store(
        mid_lse_ptr + mid_lse_offset,
        M + tl.log(L),
        mask=has_tokens & head_mask,
    )


@triton.jit
def kernel_paged_attention_2d_splitkv_reduce(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    mid_out_ptr,  # [num_seqs, num_query_heads, max_num_splits, head_size]
    mid_lse_ptr,  # [num_seqs, num_query_heads, max_num_splits]
    seq_lens_ptr,  # [num_seqs]
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    mid_out_stride_0: tl.int64,
    mid_out_stride_1: tl.int64,
    mid_out_stride_2: tl.int64,
    mid_lse_stride_0: tl.int64,
    mid_lse_stride_1: tl.int64,
    mid_lse_stride_2: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    MAX_NUM_SPLITS: tl.constexpr,
    filter_by_query_len: tl.constexpr,
    query_start_len_ptr,  # [num_seqs + 1]
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    if filter_by_query_len:
        query_start = tl.load(query_start_len_ptr + seq_idx)
        query_stop = tl.load(query_start_len_ptr + seq_idx + 1)
        if query_stop - query_start != 1:
            return
    else:
        query_start = seq_idx

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    split_len = cdiv_fn(cdiv_fn(seq_len, MAX_NUM_SPLITS), BLOCK_SIZE) * BLOCK_SIZE
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE
    M = -float("inf")
    L = 0.0
    acc = tl.zeros([HEAD_SIZE_PADDED], dtype=tl.float32)

    for split_idx in tl.range(0, MAX_NUM_SPLITS, num_stages=2):
        split_start = split_idx * split_len
        split_end = tl.minimum(split_start + split_len, seq_len)
        if split_end > split_start:
            partial_lse = tl.load(
                mid_lse_ptr
                + seq_idx * mid_lse_stride_0
                + query_head_idx * mid_lse_stride_1
                + split_idx * mid_lse_stride_2
            )
            partial_out = tl.load(
                mid_out_ptr
                + seq_idx * mid_out_stride_0
                + query_head_idx * mid_out_stride_1
                + split_idx * mid_out_stride_2
                + offs_d,
                mask=dim_mask,
                other=0.0,
            )
            new_max = tl.maximum(M, partial_lse)
            alpha = tl.exp(M - new_max)
            beta = tl.exp(partial_lse - new_max)
            acc = acc * alpha + partial_out * beta
            L = L * alpha + beta
            M = new_max

    tl.store(
        output_ptr
        + query_start * output_stride_0
        + query_head_idx * output_stride_1
        + offs_d,
        acc / (L + 1e-10),
        mask=dim_mask,
    )


@lru_cache(maxsize=256)
def _num_splits_heuristic(
    batch_nheads: int,
    num_sms: int,
    num_n_blocks: int,
    max_splits: int,
) -> int:
    """Choose the smallest split count near peak wave efficiency."""
    target_workgroups = 2 * num_sms
    if batch_nheads >= 0.8 * target_workgroups:
        return 1

    max_splits = min(max_splits, num_sms, num_n_blocks)
    if max_splits <= 1:
        return 1

    def is_split_eligible(num_splits: int) -> bool:
        return num_splits == 1 or _cdiv(num_n_blocks, num_splits) != _cdiv(
            num_n_blocks, num_splits - 1
        )

    max_efficiency = 0.0
    efficiencies = []
    for num_splits in range(1, max_splits + 1):
        if not is_split_eligible(num_splits):
            efficiencies.append(0.0)
            continue
        num_waves = batch_nheads * num_splits / target_workgroups
        efficiency = num_waves / math.ceil(num_waves)
        max_efficiency = max(max_efficiency, efficiency)
        efficiencies.append(efficiency)

    for num_splits, efficiency in enumerate(efficiencies, start=1):
        if is_split_eligible(num_splits) and efficiency >= 0.85 * max_efficiency:
            return num_splits
    return 1


def _get_num_splits(
    batch_size: int,
    num_kv_heads: int,
    head_size: int,
    physical_block_size: int,
    max_seq_len: int,
    max_num_splits: int = _MAX_SPLITS,
    num_sms: int | None = None,
    allow_short_context: bool = False,
) -> int:
    """Choose a capture-safe split count from host metadata."""
    if num_sms is None:
        num_sms = _get_num_sms(torch.accelerator.current_device_index())

    block_size = _choose_compute_block_size(physical_block_size)
    if head_size <= 64 and max_seq_len < 4096:
        return 1
    if max_seq_len <= block_size:
        return 1

    num_n_blocks = _cdiv(max_seq_len, block_size)
    if not allow_short_context and num_n_blocks < 2 * num_sms:
        return 1

    max_splits = min(max_num_splits, num_sms, num_n_blocks)
    num_splits = _num_splits_heuristic(
        batch_size * num_kv_heads, num_sms, num_n_blocks, max_splits
    )
    # Bound the reduction kernel to a small, capture-warmable specialization set.
    return min(max_num_splits, 1 << (num_splits - 1).bit_length())


def _splitkv_workspace_shapes(
    max_batch_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    physical_block_size: int,
    max_seq_len: int,
    allow_short_context: bool,
    num_sms: int,
    max_num_splits: int = _MAX_SPLITS,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return a conservative scratch envelope for every splitting batch."""
    block_size = _choose_compute_block_size(physical_block_size)
    num_n_blocks = _cdiv(max_seq_len, block_size)
    if not allow_short_context and num_n_blocks < 2 * num_sms:
        return None
    raw_max_splits = min(max_num_splits, num_sms, num_n_blocks)
    max_possible_splits = (
        1
        if raw_max_splits <= 1
        else min(max_num_splits, 1 << (raw_max_splits - 1).bit_length())
    )
    best_batch_size = 0
    for batch_size in range(1, max_batch_size + 1):
        if batch_size * num_kv_heads < 0.8 * (2 * num_sms):
            best_batch_size = batch_size
    if best_batch_size == 0 or max_possible_splits <= 1:
        return None

    return (
        (best_batch_size, num_query_heads, max_possible_splits, head_size),
        (best_batch_size, num_query_heads, max_possible_splits),
    )


def reserve_splitkv_workspace(
    max_batch_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    physical_block_size: int,
    max_seq_len: int,
    allow_short_context: bool,
    max_num_splits: int = _MAX_SPLITS,
) -> None:
    """Reserve graph-stable SplitKV scratch in the shared workspace."""
    if not is_workspace_manager_initialized():
        return
    shapes = _splitkv_workspace_shapes(
        max_batch_size,
        num_query_heads,
        num_kv_heads,
        head_size,
        physical_block_size,
        max_seq_len,
        allow_short_context,
        _get_num_sms(torch.accelerator.current_device_index()),
        max_num_splits,
    )
    if shapes is None:
        return
    mid_out_shape, mid_lse_shape = shapes

    current_workspace_manager()._reserve_simultaneous(
        (mid_out_shape, torch.float32),
        (mid_lse_shape, torch.float32),
    )


def _paged_attention_2d_splitkv_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    k_scale: torch.Tensor | float = 1.0,
    v_scale: torch.Tensor | float = 1.0,
    output: torch.Tensor | None = None,
    actual_max_splits: int | None = None,
    max_seq_len: int | None = None,
    mid_out: torch.Tensor | None = None,
    mid_lse: torch.Tensor | None = None,
    max_num_splits: int = _MAX_SPLITS,
    query_start_loc: torch.Tensor | None = None,
    filter_by_query_len: bool = False,
) -> torch.Tensor:
    """Run paged decode with optional KV splitting."""
    if output is None:
        output = torch.empty_like(query)

    batch_size = seq_lens.shape[0] if filter_by_query_len else query.shape[0]
    num_query_heads = query.shape[1]
    head_size = query.shape[2]
    num_kv_heads = key_cache.shape[1]
    physical_block_size = key_cache.shape[3]
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    is_fp8_kv = key_cache.dtype in fp8_dtypes
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads != 0:
        raise ValueError("Query heads must be divisible by non-zero KV heads.")
    if key_cache.dtype != value_cache.dtype:
        raise ValueError("Key and value caches must have the same dtype.")
    if not is_fp8_kv and key_cache.dtype != query.dtype:
        raise TypeError(
            "SplitKV requires E4M3 FP8 caches or caches matching the query dtype."
        )
    if value_cache.shape[3] != physical_block_size:
        raise ValueError("Key and value caches must use the same physical page size.")
    if is_fp8_kv:
        for name, value in (("K", k_scale), ("V", v_scale)):
            if not (
                isinstance(value, torch.Tensor)
                and value.numel() == 1
                and value.dtype == torch.float32
                and value.device == query.device
            ):
                raise TypeError(
                    f"FP8 SplitKV {name} scale must be a scalar float32 tensor "
                    "on the query device."
                )
    block_size = _choose_compute_block_size(physical_block_size)
    x = key_cache.shape[4]
    num_queries_per_kv = num_query_heads // num_kv_heads
    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv), 16)
    head_size_padded = triton.next_power_of_2(head_size)

    if max_seq_len is None:
        max_seq_len = block_tables.shape[1] * physical_block_size
    if actual_max_splits is None:
        actual_max_splits = _get_num_splits(
            batch_size,
            num_kv_heads,
            head_size,
            physical_block_size,
            max_seq_len,
            max_num_splits,
            allow_short_context=is_fp8_kv,
        )
    if not 1 <= actual_max_splits <= max_num_splits:
        raise ValueError(
            f"actual_max_splits ({actual_max_splits}) must be between 1 and "
            f"max_num_splits ({max_num_splits})."
        )

    if actual_max_splits == 1:
        from vllm.platforms.rocm import on_gfx12x

        fallback_block_size = _choose_fallback_block_size(physical_block_size)
        kernel_paged_attention_2d[(batch_size, num_kv_heads)](
            output_ptr=output,
            query_ptr=query,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            sink_ptr=None,
            block_tables_ptr=block_tables,
            seq_lens_ptr=seq_lens,
            alibi_slopes_ptr=None,
            scale=scale,
            k_scale=k_scale,
            v_scale=v_scale,
            out_scale_inv=1.0,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            num_queries_per_kv_padded=num_queries_per_kv_padded,
            block_table_stride=block_tables.stride(0),
            query_stride_0=query.stride(0),
            query_stride_1=query.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            BLOCK_SIZE=fallback_block_size,
            PHYSICAL_BLOCK_SIZE=physical_block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            USE_ALIBI_SLOPES=False,
            SLIDING_WINDOW=0,
            x=x,
            stride_k_cache_0=key_cache.stride(0),
            stride_k_cache_1=key_cache.stride(1),
            stride_k_cache_2=key_cache.stride(2),
            stride_k_cache_3=key_cache.stride(3),
            stride_k_cache_4=key_cache.stride(4),
            stride_v_cache_0=value_cache.stride(0),
            stride_v_cache_1=value_cache.stride(1),
            stride_v_cache_2=value_cache.stride(2),
            stride_v_cache_3=value_cache.stride(3),
            filter_by_query_len=filter_by_query_len,
            query_start_len_ptr=query_start_loc,
            USE_SINKS=False,
            USE_FP8=False,
            num_warps=8 if is_fp8_kv and on_gfx12x() else 4,
        )
        return output

    if (mid_out is None) != (mid_lse is None):
        raise ValueError("mid_out and mid_lse must be provided together.")
    if mid_out is None:
        scratch_shapes = (
            (batch_size, num_query_heads, actual_max_splits, head_size),
            (batch_size, num_query_heads, actual_max_splits),
        )
        if is_workspace_manager_initialized():
            mid_out, mid_lse = current_workspace_manager().get_simultaneous(
                (scratch_shapes[0], torch.float32),
                (scratch_shapes[1], torch.float32),
            )
        else:
            mid_out = torch.empty(
                scratch_shapes[0], device=query.device, dtype=torch.float32
            )
            mid_lse = torch.empty(
                scratch_shapes[1], device=query.device, dtype=torch.float32
            )
    assert mid_out is not None and mid_lse is not None
    if (
        mid_out.ndim != 4
        or mid_lse.ndim != 3
        or mid_out.dtype != torch.float32
        or mid_lse.dtype != torch.float32
        or mid_out.device != query.device
        or mid_lse.device != query.device
        or mid_out.shape[0] < batch_size
        or mid_lse.shape[0] < batch_size
        or mid_out.shape[1] < num_query_heads
        or mid_lse.shape[1] < num_query_heads
        or mid_out.shape[2] < actual_max_splits
        or mid_lse.shape[2] < actual_max_splits
        or mid_out.shape[3] < head_size
        or mid_out.stride(3) != 1
    ):
        raise ValueError(
            "SplitKV scratch tensors have incompatible shape, dtype, device, or layout."
        )

    if try_rdna4_splitkv_paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        output=output,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        k_scale=k_scale,
        v_scale=v_scale,
        scale=scale,
        actual_max_splits=actual_max_splits,
        max_seq_len=max_seq_len,
        mid_out=mid_out,
        mid_lse=mid_lse,
        filter_by_query_len=filter_by_query_len,
    ):
        return output

    kernel_paged_attention_2d_splitkv[
        (
            batch_size,
            num_kv_heads,
            actual_max_splits,
        )
    ](
        mid_out,
        mid_lse,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=num_queries_per_kv_padded,
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        mid_out_stride_0=mid_out.stride(0),
        mid_out_stride_1=mid_out.stride(1),
        mid_out_stride_2=mid_out.stride(2),
        mid_lse_stride_0=mid_lse.stride(0),
        mid_lse_stride_1=mid_lse.stride(1),
        mid_lse_stride_2=mid_lse.stride(2),
        BLOCK_SIZE=block_size,
        PHYSICAL_BLOCK_SIZE=physical_block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        x=x,
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=filter_by_query_len,
        query_start_len_ptr=query_start_loc,
        num_warps=(
            8 if is_fp8_kv and (head_size == 256 or num_queries_per_kv >= 4) else 4
        ),
        num_stages=1,
        waves_per_eu=1,
    )
    kernel_paged_attention_2d_splitkv_reduce[(batch_size, num_query_heads)](
        output,
        mid_out,
        mid_lse,
        seq_lens,
        output.stride(0),
        output.stride(1),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        MAX_NUM_SPLITS=actual_max_splits,
        filter_by_query_len=filter_by_query_len,
        query_start_len_ptr=query_start_loc,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    return output


def chunked_prefill_paged_decode(
    query,
    key,
    value,
    output,
    kv_cache_dtype,
    key_cache,
    value_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_seq_len,
    max_query_len,
    k_scale,
    v_scale,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    output_scale=None,
    # Optional tensor for sinks
    sinks=None,
    is_block_table_ptr: bool = False,
    causal: bool = True,
):
    if sm_scale is None:
        sm_scale = 1.0 / (query.shape[2] ** 0.5)

    use_alibi_slopes = alibi_slopes is not None

    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if max_query_len > 1:
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            kv_cache_dtype=kv_cache_dtype,
            k_cache=key_cache,
            v_cache=value_cache,
            b_loc=block_table,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_seq_len=max_seq_len,
            max_input_len=max_query_len,
            k_scale=k_scale,
            v_scale=v_scale,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            skip_decode=True,
            fp8_out_scale=output_scale,
            sinks=sinks,
            causal=causal,
        )

    block_size = value_cache.shape[3]
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    # key may be None in cross-attention decode (already cached from encoder)
    num_kv_heads = key.shape[1] if key is not None else key_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = query.shape[2]

    # Conversion of FP8 Tensor from uint8 storage to
    # appropriate torch.dtype for interpretation by Triton
    if "fp8" in kv_cache_dtype:
        assert key_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]
        assert value_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = current_platform.fp8_dtype()
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError(
                f"Unsupported FP8 kv_cache_dtype {kv_cache_dtype}: "
                f"should be one of 'fp8', 'fp8_e4m3', 'fp8_e5m2'."
            )

        key_cache = key_cache.view(target_dtype)
        value_cache = value_cache.view(target_dtype)

    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv), 16)

    from vllm.platforms.rocm import (
        on_gfx1x,
        on_gfx12x,
        use_rocm_custom_paged_attention,
    )

    use_custom = use_rocm_custom_paged_attention(
        query.dtype,
        head_size,
        block_size,
        num_queries_per_kv,
        max_seq_len,
        sliding_window,
        kv_cache_dtype,
        alibi_slopes,
        sinks,
    )
    has_native_layout = has_native_kv_cache_layout(key_cache, value_cache)
    # Force Triton for non-standard blocks like Qwen3's 544 and for
    # stride-padded hybrid layouts. The latter use reshape_and_cache_flash
    # during cache update, so keep decode on the matching stride-aware path.
    is_pow2 = block_size > 0 and (block_size & (block_size - 1) == 0)
    if not is_pow2 or not has_native_layout:
        use_custom = False

    if use_custom:
        _PARTITION_SIZE_ROCM = 256
        max_num_partitions = (
            max_seq_len + _PARTITION_SIZE_ROCM - 1
        ) // _PARTITION_SIZE_ROCM
        assert _PARTITION_SIZE_ROCM % block_size == 0
        total_num_seq = block_table.shape[0]
        tmp_output = torch.empty(
            size=(total_num_seq, num_query_heads, max_num_partitions, head_size),
            dtype=query.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(total_num_seq, num_query_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)

        ops.paged_attention_rocm(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale=sm_scale,
            block_tables=block_table,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_size=block_size,
            max_seq_len=max_seq_len,
            alibi_slopes=alibi_slopes,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            fp8_out_scale=output_scale,
        )
    else:
        logger.warning_once(
            "Cannot use ROCm custom paged attention kernel,"
            " falling back to Triton implementation."
        )
        real_block_size = value_cache.shape[3]
        # The standard model directly uses the original block_size.
        # Non-standard 544 uses 32 to accommodate integer division logic.
        # Cap at 128 to avoid exceeding GPU shared memory limits
        # (e.g. hybrid Mamba models inflate block_size to 2048).
        # The kernel handles TRITON_BLOCK_SIZE != PHYSICAL_BLOCK_SIZE
        # via the l_block_idx/internal_offsets addressing logic.
        TRITON_BLOCK_SIZE = _choose_fallback_block_size(block_size)
        if is_block_table_ptr:
            # Using the physical base address of tensors
            kv_element_size = key_cache.element_size()
            block_byte_stride = key_cache.stride(0) * kv_element_size
            # Get the starting physical address of the KV Cache
            base_addr = key_cache.data_ptr()

            # Normalization: Directly calculate the block offset
            # of the pointer relative to the base address
            processed_block_table = ((block_table - base_addr) // block_byte_stride).to(
                torch.int32
            )
        else:
            processed_block_table = block_table.to(torch.int32)

        kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)
        use_splitkv_decode = _can_use_rdna4_splitkv_decode(
            query_dtype=query.dtype,
            key_cache_dtype=key_cache.dtype,
            value_cache_dtype=value_cache.dtype,
            kv_quant_mode=kv_quant_mode,
            is_e4m3_kv_cache=kv_cache_dtype in ("fp8", "fp8_e4m3"),
            head_size=head_size,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            use_alibi_slopes=use_alibi_slopes,
            sliding_window=sliding_window,
            has_sinks=sinks is not None,
            has_output_scale=output_scale is not None,
            is_gfx1x=on_gfx1x(),
            is_gfx12x=on_gfx12x(),
        )
        actual_max_splits = 1
        if use_splitkv_decode:
            actual_max_splits = _get_num_splits(
                num_seqs,
                num_kv_heads,
                head_size,
                real_block_size,
                max_seq_len,
                max_num_splits=_MAX_SPLITS,
                allow_short_context=key_cache.dtype
                in (torch.float8_e4m3fn, torch.float8_e4m3fnuz),
            )
        if actual_max_splits > 1:
            _paged_attention_2d_splitkv_decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=processed_block_table,
                seq_lens=seq_lens,
                scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                output=output,
                actual_max_splits=actual_max_splits,
                max_seq_len=max_seq_len,
                max_num_splits=_MAX_SPLITS,
                query_start_loc=query_start_loc,
                filter_by_query_len=True,
            )
            return

        kernel_paged_attention_2d[
            (
                num_seqs,
                num_kv_heads,
            )
        ](
            output_ptr=output,
            query_ptr=query,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            sink_ptr=sinks,
            block_tables_ptr=processed_block_table,
            seq_lens_ptr=seq_lens,
            alibi_slopes_ptr=alibi_slopes,
            scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            out_scale_inv=1.0 / output_scale if output_scale is not None else 1.0,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            num_queries_per_kv_padded=num_queries_per_kv_padded,
            block_table_stride=processed_block_table.stride(0),
            query_stride_0=query.stride(0),
            query_stride_1=query.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            BLOCK_SIZE=TRITON_BLOCK_SIZE,
            PHYSICAL_BLOCK_SIZE=real_block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            SLIDING_WINDOW=sliding_window,
            x=key_cache.shape[4],
            stride_k_cache_0=key_cache.stride(0),
            stride_k_cache_1=key_cache.stride(1),
            stride_k_cache_2=key_cache.stride(2),
            stride_k_cache_3=key_cache.stride(3),
            stride_k_cache_4=key_cache.stride(4),
            stride_v_cache_0=value_cache.stride(0),
            stride_v_cache_1=value_cache.stride(1),
            stride_v_cache_2=value_cache.stride(2),
            stride_v_cache_3=value_cache.stride(3),
            filter_by_query_len=True,
            query_start_len_ptr=query_start_loc,
            USE_SINKS=sinks is not None,
            USE_FP8=output_scale is not None,
            num_warps=(8 if key_cache.element_size() == 1 and on_gfx12x() else 4),
        )
