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
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .prefix_prefill import context_attention_fwd

logger = init_logger(__name__)

float8_info = torch.finfo(current_platform.fp8_dtype())

# Upper bound on KV splits. Past this the reduction and the extra partial
# traffic cost more than the added parallelism returns.
_MAX_KV_SPLITS = 32
# Workgroups we aim to have resident before splitting stops helping.
_TARGET_BLOCKS_PER_SM = 4


@functools.lru_cache(maxsize=16)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _kv_split_supported() -> bool:
    """KV splitting is restricted to the AMD architectures it was tuned on.

    Importing vllm.platforms.rocm resolves the GCN arch at module load, so it
    must stay behind the is_rocm() check.
    """
    if not current_platform.is_rocm():
        return False

    from vllm.platforms.rocm import on_gfx1100, on_gfx1201

    return on_gfx1100() or on_gfx1201()


def get_num_kv_splits(num_seqs: int, num_kv_heads: int, device: torch.device) -> int:
    """Pick a KV split count for the Triton paged-decode kernel.

    The unsplit grid is (num_seqs, num_kv_heads). For a small batch with few KV
    heads that leaves most of the GPU idle while each workgroup walks the whole
    context serially, so decode cost grows with context length. Splitting the
    KV range adds a third grid dimension and restores parallelism.

    Derived from batch shape only, never from sequence length: vLLM captures
    decode into CUDA graphs, and a grid that moved with max_seq_len would
    invalidate the capture. Over-splitting a short sequence is harmless because
    the surplus splits run zero tiles and emit neutral partials.

    Returns 1 (the unsplit path, byte-identical to before) on every platform
    other than gfx1100/gfx1201, which are the only ones this was measured on.
    """
    if not _kv_split_supported():
        return 1
    base = max(1, num_seqs * num_kv_heads)
    try:
        target = _TARGET_BLOCKS_PER_SM * _num_sms(device.index or 0)
    except (AssertionError, RuntimeError):
        return 1
    return max(1, min(-(-target // base), _MAX_KV_SPLITS))


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
    partial_out_ptr,  # [num_seqs, num_query_heads, NUM_SPLITS, head_size_padded]
    partial_m_ptr,  # [num_seqs, num_query_heads, NUM_SPLITS]
    partial_l_ptr,  # [num_seqs, num_query_heads, NUM_SPLITS]
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
    po_stride_0: tl.int64,  # int
    po_stride_1: tl.int64,  # int
    po_stride_2: tl.int64,  # int
    pm_stride_0: tl.int64,  # int
    pm_stride_1: tl.int64,  # int
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
    NUM_SPLITS: tl.constexpr = 1,  # int, KV splits per (seq, kv_head)
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

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
        # The sink is a single extra logit for the whole sequence, so only one
        # split may seed it; otherwise it would be counted NUM_SPLITS times.
        if split_idx == 0:
            M = tl.load(
                sink_ptr + query_head_idx,
                mask=head_mask,
                other=float("-inf"),
            ).to(dtype=tl.float32)
            L = tl.where(float("-inf") < M, 1.0, 0.0)
        else:
            M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
            L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)

    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_head_idx, mask=head_mask, other=0.0
        )

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    # Each split walks a contiguous range of tiles. The range is derived from
    # seq_lens on the device, so the grid stays independent of sequence length
    # and remains valid under CUDA graph capture. Splits that fall past the end
    # of a short sequence run zero tiles and emit a neutral (m=-inf, l=0)
    # partial, which the reduction drops.
    if NUM_SPLITS == 1:
        j_start = 0
        j_end = num_blocks
    else:
        blocks_per_split = cdiv_fn(num_blocks, NUM_SPLITS)
        j_start = split_idx * blocks_per_split
        j_end = tl.minimum(j_start + blocks_per_split, num_blocks)

    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    # iterate through tiles
    for j in range(j_start, j_end):
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

        # Slots >= seq_len are unwritten KV cache that may hold NaN/garbage;
        # they are score-masked below, but 0 * NaN = NaN would still poison the
        # output, so mask them out of the K/V loads too. Tiles that cannot
        # straddle seq_len use the cheaper token-uniform dim_mask and skip the
        # per-token predicate entirely.
        # Test straddling directly rather than comparing j to num_blocks - 1:
        # with NUM_SPLITS > 1 a split's last tile is generally not the
        # sequence's last tile.
        # K : (HEAD_SIZE, BLOCK_SIZE), V : (BLOCK_SIZE, HEAD_SIZE)
        if start_n + BLOCK_SIZE > seq_len:
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
    if NUM_SPLITS == 1:
        acc = acc / (L[:, None] + 1e-10)
        if USE_FP8:
            acc = acc * tl.load(out_scale_inv)
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

        output_offset = (
            cur_batch_in_all_start_index * output_stride_0
            + query_head_idx * output_stride_1
        )

        tl.store(
            output_ptr
            + output_offset[:, None]
            + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
            acc,
            mask=dim_mask[None, :] & head_mask[:, None],
        )
    else:
        # Store the UNNORMALISED accumulator plus (m, l); the reduction
        # rescales by the merged l and applies any output scaling.
        # Partials are indexed by seq_idx, not by token index: in a mixed
        # chunked-prefill batch the token index ranges over all prefill tokens
        # too, which would make this scratch orders of magnitude larger.
        po_offset = (
            seq_idx * po_stride_0
            + query_head_idx[:, None] * po_stride_1
            + split_idx * po_stride_2
        )
        tl.store(
            partial_out_ptr + po_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
            acc,
            mask=dim_mask[None, :] & head_mask[:, None],
        )
        pm_offset = seq_idx * pm_stride_0 + query_head_idx * pm_stride_1 + split_idx
        tl.store(partial_m_ptr + pm_offset, M, mask=head_mask)
        tl.store(partial_l_ptr + pm_offset, L, mask=head_mask)


@triton.jit
def kernel_paged_attention_2d_reduce(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    partial_out_ptr,  # [num_seqs, num_query_heads, NUM_SPLITS, head_size_padded]
    partial_m_ptr,  # [num_seqs, num_query_heads, NUM_SPLITS]
    partial_l_ptr,  # [num_seqs, num_query_heads, NUM_SPLITS]
    out_scale_inv,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    po_stride_0: tl.int64,
    po_stride_1: tl.int64,
    po_stride_2: tl.int64,
    pm_stride_0: tl.int64,
    pm_stride_1: tl.int64,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_SPLITS_PADDED: tl.constexpr,
    filter_by_query_len: tl.constexpr,
    query_start_len_ptr,  # [num_seqs+1]
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    """Merge the per-split partials with the standard log-sum-exp rescale."""
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        if cur_batch_in_all_stop_index - cur_batch_in_all_start_index > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    offs_s = tl.arange(0, NUM_SPLITS_PADDED)
    split_mask = offs_s < NUM_SPLITS
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE

    # Partials are indexed by seq_idx; only the output uses the token index.
    pm_offset = seq_idx * pm_stride_0 + head_idx * pm_stride_1 + offs_s
    m_i = tl.load(partial_m_ptr + pm_offset, mask=split_mask, other=float("-inf"))
    l_i = tl.load(partial_l_ptr + pm_offset, mask=split_mask, other=0.0)

    m = tl.max(m_i, axis=0)
    # exp(-inf - -inf) is NaN, so empty splits are forced to contribute 0.
    alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m))
    alpha = tl.where(m == float("-inf"), 0.0, alpha)
    l_merged = tl.sum(l_i * alpha, axis=0)

    po_offset = (
        seq_idx * po_stride_0
        + head_idx * po_stride_1
        + offs_s[:, None] * po_stride_2
        + offs_d[None, :]
    )
    acc = tl.load(
        partial_out_ptr + po_offset,
        mask=split_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    acc = tl.sum(acc * alpha[:, None], axis=0) / (l_merged + 1e-10)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        cur_batch_in_all_start_index * output_stride_0 + head_idx * output_stride_1
    )
    tl.store(output_ptr + output_offset + offs_d, acc, mask=dim_mask)


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

        head_size_padded = triton.next_power_of_2(head_size)
        num_kv_splits = get_num_kv_splits(num_seqs, num_kv_heads, query.device)
        if num_kv_splits > 1:
            # Allocated per call on purpose. A tensor allocated during CUDA
            # graph capture belongs to that graph's private pool, so caching
            # these across calls faults once the graph is replayed. The ROCm
            # custom path above allocates its tmp_output/exp_sums the same way.
            partial_out = torch.empty(
                (num_seqs, num_query_heads, num_kv_splits, head_size_padded),
                dtype=torch.float32,
                device=query.device,
            )
            partial_m = torch.empty(
                (num_seqs, num_query_heads, num_kv_splits),
                dtype=torch.float32,
                device=query.device,
            )
            partial_l = torch.empty_like(partial_m)
            po_strides = partial_out.stride()
            pm_strides = partial_m.stride()
        else:
            partial_out = partial_m = partial_l = None
            po_strides = (0, 0, 0)
            pm_strides = (0, 0)

        kernel_paged_attention_2d[
            (
                num_seqs,
                num_kv_heads,
                num_kv_splits,
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
            partial_out_ptr=partial_out,
            partial_m_ptr=partial_m,
            partial_l_ptr=partial_l,
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
            po_stride_0=po_strides[0],
            po_stride_1=po_strides[1],
            po_stride_2=po_strides[2],
            pm_stride_0=pm_strides[0],
            pm_stride_1=pm_strides[1],
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
            USE_SINKS=sinks is not None,
            USE_FP8=output_scale is not None,
            NUM_SPLITS=num_kv_splits,
        )

        if num_kv_splits > 1:
            kernel_paged_attention_2d_reduce[(num_seqs, num_query_heads)](
                output_ptr=output,
                partial_out_ptr=partial_out,
                partial_m_ptr=partial_m,
                partial_l_ptr=partial_l,
                out_scale_inv=1.0 / output_scale if output_scale is not None else 1.0,
                output_stride_0=output.stride(0),
                output_stride_1=output.stride(1),
                po_stride_0=po_strides[0],
                po_stride_1=po_strides[1],
                po_stride_2=po_strides[2],
                pm_stride_0=pm_strides[0],
                pm_stride_1=pm_strides[1],
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=head_size_padded,
                NUM_SPLITS=num_kv_splits,
                NUM_SPLITS_PADDED=triton.next_power_of_2(num_kv_splits),
                filter_by_query_len=True,
                query_start_len_ptr=query_start_loc,
                USE_FP8=output_scale is not None,
            )
