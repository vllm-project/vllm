# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import functools

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .prefix_prefill import context_attention_fwd

logger = init_logger(__name__)

float8_info = torch.finfo(current_platform.fp8_dtype())


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
        if cur_batch_query_len > 1:
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

        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary

        # First calculate the dot, then apply the mask.
        qk = scale * tl.dot(Q, K)
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


# ---------------------------------------------------------------------------
# Flash-decoding sequence partitioning for the Triton paged-attention fallback.
#
# The unpartitioned kernel launches a grid of (num_seqs, num_kv_heads). During
# decode that is a handful of programs -- on qwen38 (num_kv_heads small, batch
# often 1) it is single digits, against 104 CUs per MI210 die. The whole
# sequence is walked by one program, so long context serialises and decode
# collapses. The ROCm HIP kernel avoids this with _PARTITION_SIZE_ROCM = 256
# and a second reduction pass; it is unavailable here because the free kernel
# is numerically wrong for block_size > 64 on gfx90a (see
# vllm/platforms/rocm.py, use_rocm_custom_paged_attention).
#
# So do the same thing in Triton: split the sequence into partitions, run one
# program per (seq, kv_head, partition) accumulating an UNNORMALISED softmax
# numerator plus its running max/denominator, then combine the partials.
#
# Env:
#   VLLM_TRITON_PA_SEQ_PARTITION=0   disable (default: enabled)
#   VLLM_TRITON_PA_TARGET_PROGRAMS   occupancy target (default 4 * #CUs)
#   VLLM_TRITON_PA_SCRATCH_BUDGET_MB scratch cap (default 128)
#   VLLM_TRITON_PA_TARGET_PROGRAMS   programs to aim for (default 4 * #CUs)
# ---------------------------------------------------------------------------

_PA_PARTITION_MIN = 512
_PA_MAX_PARTITIONS = 1024
def _pa_scratch_budget() -> int:
    return envs.VLLM_TRITON_PA_SCRATCH_BUDGET_MB * 1024 * 1024


@functools.lru_cache(maxsize=1)
def _seq_partition_enabled() -> bool:
    return envs.VLLM_TRITON_PA_SEQ_PARTITION


@functools.lru_cache(maxsize=1)
def _seq_partition_target_programs() -> int:
    if envs.VLLM_TRITON_PA_TARGET_PROGRAMS:
        return max(1, envs.VLLM_TRITON_PA_TARGET_PROGRAMS)
    try:
        cus = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        cus = 64
    return 4 * cus


def _choose_partition_size(
    num_seqs: int, num_kv_heads: int, seq_len_bound: int, block_size: int,
    num_query_heads: int = 32, head_size_padded: int = 256,
) -> int:
    """Partition size in tokens, or 0 to keep the single-program-per-seq path.

    CUDAGRAPH SAFETY -- read before changing `seq_len_bound`.

    Decode runs inside full cudagraphs, and RocmAttentionMetadataBuilder.
    build_for_cudagraph_capture() does `seq_lens.fill_(1)`. So the *runtime*
    max_seq_len seen at capture time is tiny and bears no relation to the
    sequence lengths the captured graph will later replay against. Sizing the
    partition grid from it would bake in too few partitions and silently drop
    the tail of every long sequence -- wrong answers, not a slowdown.

    Callers therefore pass a bound that is fixed for the life of a captured
    graph: the block table's width times the physical block size, which by
    construction covers max_model_len. An OVERSIZED grid is harmless (surplus
    partitions store their -inf/0 sentinel and the reducer, which recomputes
    num_parts from the true per-sequence length, ignores them). An UNDERSIZED
    grid is not, so the bound must never shrink.

    Within that constraint the partition count is chosen to fill the device
    rather than to be maximal: it falls out of how far the (num_seqs,
    num_kv_heads) grid is from the target occupancy. A large decode batch
    already saturates and gets 0, which keeps scratch small by construction.
    """
    if not _seq_partition_enabled():
        return 0
    if seq_len_bound < 2 * _PA_PARTITION_MIN:
        return 0
    base = max(1, num_seqs * num_kv_heads)
    if base >= _seq_partition_target_programs():
        return 0  # the (seq, kv_head) grid already saturates the device
    # Take the finest partition the scratch budget affords rather than sizing
    # for occupancy AT THE BOUND: the bound is max_model_len, so occupancy
    # sizing leaves most partitions empty at realistic context lengths. Surplus
    # partitions cost one early-exiting program each.
    bytes_per_part = max(1, num_seqs * num_query_heads * head_size_padded * 4)
    max_parts = max(1, _pa_scratch_budget() // bytes_per_part)
    max_parts = min(max_parts, _PA_MAX_PARTITIONS)
    part_size = max(_PA_PARTITION_MIN, -(-seq_len_bound // max_parts))
    # Quantise to a power of two: PARTITION_SIZE is a constexpr, so an
    # unquantised value would trigger a fresh Triton compile per batch shape.
    part_size = triton.next_power_of_2(part_size)
    if part_size % block_size != 0:
        part_size = -(-part_size // block_size) * block_size
    parts = -(-seq_len_bound // part_size)
    if parts <= 1:
        return 0
    while parts > _PA_MAX_PARTITIONS:
        part_size *= 2
        parts = -(-seq_len_bound // part_size)
    return part_size


@triton.jit
def kernel_paged_attention_2d_partitioned(
    tmp_acc_ptr,  # [num_seqs, num_query_heads, max_parts, HEAD_SIZE_PADDED] f32
    tmp_m_ptr,  # [num_seqs, num_query_heads, max_parts] f32
    tmp_l_ptr,  # [num_seqs, num_query_heads, max_parts] f32
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    scale,
    k_scale,
    v_scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    num_queries_per_kv_padded: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    tmp_acc_stride_0: tl.int64,
    tmp_acc_stride_1: tl.int64,
    tmp_acc_stride_2: tl.int64,
    tmp_ml_stride_0: tl.int64,
    tmp_ml_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
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
    query_start_len_ptr,  # [num_seqs+1]
    PARTITION_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    part_idx = tl.program_id(2)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )
    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)

    ml_offset = seq_idx * tmp_ml_stride_0 + query_head_idx * tmp_ml_stride_1 + part_idx
    acc_offset = (
        seq_idx * tmp_acc_stride_0
        + query_head_idx[:, None] * tmp_acc_stride_1
        + part_idx * tmp_acc_stride_2
    )

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    part_start = part_idx * PARTITION_SIZE

    # A partition writes iff part_idx < cdiv(seq_len, PARTITION_SIZE), which is
    # exactly the range the reducer reads (it recomputes num_parts from the true
    # seq_len and masks). Surplus partitions are therefore never read and can
    # exit without storing anything -- worth doing, because the grid is sized
    # from a fixed bound, so at short context MOST partitions are surplus and
    # their sentinel stores were the dominant cost.
    if part_start >= seq_len:
        return

    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0
        + query_head_idx[:, None] * query_stride_1
    )
    Q = tl.load(
        query_ptr + query_offset + offs_d[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_head_idx, mask=head_mask, other=0.0
        )

    part_end = tl.minimum(seq_len, part_start + PARTITION_SIZE)
    num_blocks = cdiv_fn(part_end - part_start, BLOCK_SIZE)

    offs_n = tl.arange(0, BLOCK_SIZE)
    for j in range(0, num_blocks):
        start_n = part_start + j * BLOCK_SIZE
        abs_token_idx = start_n + offs_n
        kv_load_mask = abs_token_idx < seq_len
        l_block_idx = abs_token_idx // PHYSICAL_BLOCK_SIZE
        p_block_idx = tl.load(block_tables_ptr + block_table_offset + l_block_idx)
        internal_offsets = abs_token_idx % PHYSICAL_BLOCK_SIZE

        k_offset = (
            p_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + internal_offsets[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )
        v_offset = (
            p_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[None, :] * stride_v_cache_2
            + internal_offsets[:, None] * stride_v_cache_3
        )

        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & kv_load_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & kv_load_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = abs_token_idx
        seq_mask = seq_offset[None, :] < seq_len

        qk = scale * tl.dot(Q, K)
        S = tl.where(head_mask[:, None] & seq_mask, qk, float("-inf"))

        context_len = seq_len - 1
        if SLIDING_WINDOW > 0:
            S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S, -10000)
        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        m_j = tl.maximum(M, tl.max(S, axis=1))
        p = tl.exp(S - m_j[:, None])
        p = tl.where(m_j[:, None] == float("-inf"), 0.0, p)
        l_j = tl.sum(p, axis=1)

        alpha = tl.exp(M - m_j)
        alpha = tl.where(float("-inf") == M, 0.0, alpha)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(p.to(V.dtype), V)

    # Store the UNNORMALISED numerator plus (max, denominator). The division
    # and the sink/fp8 epilogue happen once, in the reduce kernel.
    tl.store(tmp_m_ptr + ml_offset, M, mask=head_mask)
    tl.store(tmp_l_ptr + ml_offset, L, mask=head_mask)
    tl.store(
        tmp_acc_ptr + acc_offset + offs_d[None, :],
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )


@triton.jit
def kernel_paged_attention_2d_reduce(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    tmp_acc_ptr,
    tmp_m_ptr,
    tmp_l_ptr,
    sink_ptr,  # [num_query_heads]
    seq_lens_ptr,  # [num_seqs]
    out_scale_inv,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    tmp_acc_stride_0: tl.int64,
    tmp_acc_stride_1: tl.int64,
    tmp_acc_stride_2: tl.int64,
    tmp_ml_stride_0: tl.int64,
    tmp_ml_stride_1: tl.int64,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    MAX_PARTS_PADDED: tl.constexpr,
    filter_by_query_len: tl.constexpr,
    query_start_len_ptr,
    USE_SINKS: tl.constexpr,
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if filter_by_query_len:
        start = tl.load(query_start_len_ptr + seq_idx)
        stop = tl.load(query_start_len_ptr + seq_idx + 1)
        if stop - start > 1:
            return
        out_row = start
    else:
        out_row = seq_idx

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_parts = cdiv_fn(seq_len, PARTITION_SIZE)

    offs_p = tl.arange(0, MAX_PARTS_PADDED)
    part_mask = offs_p < num_parts

    ml_base = seq_idx * tmp_ml_stride_0 + head_idx * tmp_ml_stride_1
    m_p = tl.load(tmp_m_ptr + ml_base + offs_p, mask=part_mask, other=float("-inf"))
    l_p = tl.load(tmp_l_ptr + ml_base + offs_p, mask=part_mask, other=0.0)

    m = tl.max(m_p, axis=0)
    if USE_SINKS:
        sink = tl.load(sink_ptr + head_idx).to(tl.float32)
        m = tl.maximum(m, sink)

    # exp(-inf - -inf) is NaN, so mask empty partitions out explicitly rather
    # than relying on the arithmetic.
    alpha = tl.exp(m_p - m)
    alpha = tl.where(part_mask & (m_p > float("-inf")), alpha, 0.0)

    denom = tl.sum(l_p * alpha, axis=0)
    if USE_SINKS:
        denom += tl.exp(sink - m)

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE
    acc_base = seq_idx * tmp_acc_stride_0 + head_idx * tmp_acc_stride_1
    partials = tl.load(
        tmp_acc_ptr + acc_base + offs_p[:, None] * tmp_acc_stride_2 + offs_d[None, :],
        mask=part_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    out = tl.sum(partials * alpha[:, None], axis=0)
    out = out / (denom + 1e-10)

    if USE_FP8:
        out = out * tl.load(out_scale_inv)
        out = tl.clamp(out, FP8_MIN, FP8_MAX)

    tl.store(
        output_ptr + out_row * output_stride_0 + head_idx * output_stride_1 + offs_d,
        out,
        mask=dim_mask,
    )


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

    from vllm.platforms.rocm import use_rocm_custom_paged_attention

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
        MAX_TRITON_BLOCK_SIZE = 128
        TRITON_BLOCK_SIZE = min(block_size, MAX_TRITON_BLOCK_SIZE) if is_pow2 else 32
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

        # See _choose_partition_size: the bound must be fixed for the life of
        # a captured cudagraph, so derive it from the block table's width
        # rather than from the runtime max_seq_len (which capture sets to 1).
        seq_len_bound = max(
            max_seq_len, processed_block_table.shape[1] * real_block_size
        )
        part_size = _choose_partition_size(
            num_seqs, num_kv_heads, seq_len_bound, TRITON_BLOCK_SIZE,
            num_query_heads=num_query_heads,
            head_size_padded=triton.next_power_of_2(head_size),
        )
        if part_size:
            max_parts = (seq_len_bound + part_size - 1) // part_size
            head_size_padded = triton.next_power_of_2(head_size)
            tmp_acc = torch.empty(
                (num_seqs, num_query_heads, max_parts, head_size_padded),
                dtype=torch.float32,
                device=query.device,
            )
            tmp_m = torch.empty(
                (num_seqs, num_query_heads, max_parts),
                dtype=torch.float32,
                device=query.device,
            )
            tmp_l = torch.empty_like(tmp_m)
            logger.warning_once(
                "Triton paged decode: sequence partitioning ON "
                "(%d seqs x %d kv heads x %d partitions of %d tokens).",
                num_seqs,
                num_kv_heads,
                max_parts,
                part_size,
            )
            kernel_paged_attention_2d_partitioned[
                (
                    num_seqs,
                    num_kv_heads,
                    max_parts,
                )
            ](
                tmp_acc_ptr=tmp_acc,
                tmp_m_ptr=tmp_m,
                tmp_l_ptr=tmp_l,
                query_ptr=query,
                key_cache_ptr=key_cache,
                value_cache_ptr=value_cache,
                block_tables_ptr=processed_block_table,
                seq_lens_ptr=seq_lens,
                alibi_slopes_ptr=alibi_slopes,
                scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                num_queries_per_kv_padded=num_queries_per_kv_padded,
                block_table_stride=processed_block_table.stride(0),
                query_stride_0=query.stride(0),
                query_stride_1=query.stride(1),
                tmp_acc_stride_0=tmp_acc.stride(0),
                tmp_acc_stride_1=tmp_acc.stride(1),
                tmp_acc_stride_2=tmp_acc.stride(2),
                tmp_ml_stride_0=tmp_m.stride(0),
                tmp_ml_stride_1=tmp_m.stride(1),
                BLOCK_SIZE=TRITON_BLOCK_SIZE,
                PHYSICAL_BLOCK_SIZE=real_block_size,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=head_size_padded,
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
                PARTITION_SIZE=part_size,
            )
            kernel_paged_attention_2d_reduce[
                (
                    num_seqs,
                    num_query_heads,
                )
            ](
                output_ptr=output,
                tmp_acc_ptr=tmp_acc,
                tmp_m_ptr=tmp_m,
                tmp_l_ptr=tmp_l,
                sink_ptr=sinks,
                seq_lens_ptr=seq_lens,
                out_scale_inv=1.0 / output_scale
                if output_scale is not None
                else 1.0,
                output_stride_0=output.stride(0),
                output_stride_1=output.stride(1),
                tmp_acc_stride_0=tmp_acc.stride(0),
                tmp_acc_stride_1=tmp_acc.stride(1),
                tmp_acc_stride_2=tmp_acc.stride(2),
                tmp_ml_stride_0=tmp_m.stride(0),
                tmp_ml_stride_1=tmp_m.stride(1),
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=head_size_padded,
                PARTITION_SIZE=part_size,
                MAX_PARTS_PADDED=triton.next_power_of_2(max_parts),
                filter_by_query_len=True,
                query_start_len_ptr=query_start_loc,
                USE_SINKS=sinks is not None,
                USE_FP8=output_scale is not None,
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
        )
