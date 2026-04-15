# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared ``@triton.jit`` helpers used across the core kernel and the
per-mode backends.

Two groups live here:

  * **Attention-loop helpers** — mask building, ALiBi / QQ-bias score
    post-processing, and the online-softmax bookkeeping step.  The core
    kernel and the INT4 / INT2 plugins all run the same online softmax
    over a tiled key/value sequence; keeping this logic in one place
    ensures a fix in one (e.g. a sliding-window edge case) lands in
    every kernel.

  * **Sub-byte packing helpers** — two helpers per layout (pack /
    unpack) for the INT4-nibble and INT2-quartet formats, shared by
    the reshape (write) and attention (read) kernels to prevent drift.
"""

from __future__ import annotations

from vllm.triton_utils import tl, triton

# ===========================================================================
# Scalar helpers (reused by every kernel + reduce_segments)
# ===========================================================================


@triton.jit
def cdiv_fn(x, y):
    """Ceiling division.  Kept as a helper to keep kernel bodies terse."""
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    """Softcap (aka tanh-style clamp) used to bound attention scores.

    ``x * tanh(S / x)`` rewritten to avoid a direct ``tanh`` call.
    """
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


# ===========================================================================
# Attention loop
# ===========================================================================


@triton.jit
def resolve_seq_and_query_len(
    query_start_len_ptr,
    seq_lens_ptr,
    q_block_global_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
):
    """Resolve the (sequence, q-block-within-sequence) pair and load the
    per-sequence lengths.

    Shared across every attention kernel — the ``q_block_global_idx``
    program id indexes into the flattened ``(seq, q_block_in_seq)``
    space, and a binary search over ``query_start_len_ptr`` recovers
    the (seq, local-q-block) pair.

    Returns ``(seq_idx, q_block_local_idx, cur_batch_in_all_start_index,
    cur_batch_query_len, seq_len)``.  Callers must still early-return
    when ``q_block_local_idx * BLOCK_Q >= cur_batch_query_len`` (Triton
    helpers cannot return from the caller).
    """
    # find_seq_idx is defined below; forward use is fine inside @triton.jit.
    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_start = tl.load(query_start_len_ptr + seq_idx)
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_stop - cur_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    return seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    """Binary search over the cumulative query-length prefix.

    When ``use_q_block_mode`` is True, the prefix values are reshaped
    into units of ``BLOCK_Q`` plus one entry per boundary — matching
    the q-block grid laid out by the attention kernels.  When False
    we search the plain cumulative-length prefix (used by
    ``reduce_segments`` which iterates over raw query tokens).
    """
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def init_softmax_M(
    sink_ptr,
    query_offset_1,
    query_mask_1,
    segm_idx_or_0,
    BLOCK_M: tl.constexpr,
    USE_SINKS: tl.constexpr,
    IS_3D: tl.constexpr,
):
    """Initial row-max ``M`` for the online softmax.

    Without sinks: ``-inf``.  With sinks: load the per-head sink bias
    once.  In 3D mode only segment 0 loads — ``reduce_segments`` adds
    the sink contribution exactly once across segments, so other
    segments must start from ``-inf``.

    ``segm_idx_or_0`` is the 3D segment index or 0 for 2D (caller
    passes ``0`` when ``IS_3D`` is False).
    """
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    if USE_SINKS:
        load_sinks = (not IS_3D) or (segm_idx_or_0 == 0)
        if load_sinks:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(tl.float32)
    return M


@triton.jit
def compute_tile_loop_bounds(
    context_len,
    seq_len,
    cur_batch_query_len,
    q_block_local_idx,
    segm_idx_or_0,
    tiles_per_segment_or_0,
    TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    IS_3D: tl.constexpr,
):
    """Compute the tile-loop bounds ``(loop_lo, loop_hi)`` and the
    derived ``max_seq_prefix_len`` used for per-tile masking.

    Combines three concerns into one helper:

    1. Longest prefix spanned by any query token in this q-block.
       Clamped to ``seq_len`` (causal) or extended to it when
       mm_prefix is active (bidirectional ranges can reach past the
       causal prefix).
    2. Sliding-window pruning: narrows ``[tile_start, tile_end)`` to
       only tiles that can contain an allowed key under SWA.
    3. 3D scoping: when ``IS_3D`` is True, further narrows to the
       segment's slice via ``(segm_idx * tiles_per_segment,
       (segm_idx + 1) * tiles_per_segment)``.
    """
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    if USE_MM_PREFIX:
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    if IS_3D:
        loop_lo = max(segm_idx_or_0 * tiles_per_segment_or_0, tile_start)
        loop_hi = min((segm_idx_or_0 + 1) * tiles_per_segment_or_0, tile_end)
    else:
        loop_lo = tile_start
        loop_hi = tile_end

    return loop_lo, loop_hi, max_seq_prefix_len


@triton.jit
def store_segm_reduce_scalars(
    segm_max_ptr,
    segm_expsum_ptr,
    query_offset_0,
    query_offset_1,
    segm_idx,
    M,
    L,
    query_mask_0,
    query_mask_1,
    num_query_heads: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    """Store per-segment ``M`` and ``L`` for ``reduce_segments`` to
    combine into the final softmax.

    Shared across every 3D attention epilogue; the per-token output
    stripes are mode-specific (flat / 2-stream split / 4-stream split)
    and stay inlined.
    """
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def compute_kv_seq_mask(
    query_abs_pos,
    seq_offset,
    seq_idx,
    mm_prefix_range_ptr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
):
    """Build the KV mask for one tile.

    Causal (key <= query) by default; AND-ed with the sliding window when
    enabled; OR-ed with the bidirectional ranges from ``mm_prefix_range``
    when PrefixLM / multimodal attention is active.  The order matches
    FlexAttention: ``(causal AND sliding_window) OR mm_prefix``.
    """
    seq_mask = seq_offset[None, :] <= query_abs_pos
    if SLIDING_WINDOW > 0:
        seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
    if USE_MM_PREFIX:
        for i in range(MAX_MM_RANGES):
            range_start = tl.load(
                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
            )
            range_end = tl.load(
                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
            )
            is_valid = range_start < range_end
            q_in_range = (
                (query_abs_pos >= range_start) & (query_abs_pos <= range_end) & is_valid
            )
            k_in_range = (
                (seq_offset[None, :] >= range_start)
                & (seq_offset[None, :] <= range_end)
                & is_valid
            )
            seq_mask |= q_in_range & k_in_range
    return seq_mask


@triton.jit
def apply_alibi_to_score(
    S,
    alibi_slope,
    seq_offset,
    context_len,
    query_pos,
    USE_ALIBI_SQRT: tl.constexpr,
):
    """Add the ALiBi positional bias (linear or sqrt variant) to S in-place."""
    if USE_ALIBI_SQRT:
        relative_pos = seq_offset - (context_len + query_pos[:, None])
        alibi_offset = tl.where(
            relative_pos <= 0,
            -tl.sqrt((-relative_pos).to(tl.float32)),
            0.0,
        )
    else:
        alibi_offset = seq_offset - context_len
    return S + alibi_slope[:, None] * alibi_offset


@triton.jit
def load_qq_bias_tile(
    qq_bias_row_ptrs,
    seq_offset,
    context_len,
    qq_bias_stride_0,
):
    """Load the qq-bias slice for keys that correspond to query rows."""
    key_rel_pos = seq_offset - context_len
    is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
    return tl.load(
        qq_bias_row_ptrs + key_rel_pos[None, :],
        mask=is_query_key[None, :],
        other=0.0,
    )


@triton.jit
def softmax_step(S, M, L):
    """Online softmax update for one tile.

    Returns ``(M_new, L_new, P, alpha)``.  Caller is responsible for
    rescaling its accumulator(s) by ``alpha[:, None]`` — done outside
    because the number / shape of accumulators varies between kernels
    (1 in the core, 2 in INT4 split-dot, 4 in INT2 quartet-dot).
    """
    m_j = tl.maximum(M, tl.max(S, axis=1))
    # Sliding-window may mask the entire row → max == -inf.  Clamp to 0
    # so ``acc * exp(M - m_j)`` doesn't produce NaN.
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    P = tl.exp(S - m_j[:, None])
    l_j = tl.sum(P, axis=1)
    alpha = tl.exp(M - m_j)
    L_new = L * alpha + l_j
    return m_j, L_new, P, alpha


# ===========================================================================
# Sub-byte packing
# ===========================================================================
# INT4 layout: two 4-bit values per uint8 — low nibble = even index,
# high nibble = odd index.  INT2 layout: four 2-bit values per uint8 in
# little-endian quartet order (q0 in bits [0:2], q3 in [6:8]).


@triton.jit
def pack_int4_nibbles(lo, hi):
    """Pack two uint8 values (each in [0, 15]) into one byte."""
    return (lo & 0xF) | ((hi & 0xF) << 4)


@triton.jit
def unpack_int4_nibbles(packed):
    """Split one packed byte into the (low, high) nibble pair as uint8."""
    return packed & 0xF, (packed >> 4) & 0xF


@triton.jit
def pack_int2_quartet(q0, q1, q2, q3):
    """Pack four uint8 values (each in [0, 3]) into one byte."""
    return (q0 & 0x3) | ((q1 & 0x3) << 2) | ((q2 & 0x3) << 4) | ((q3 & 0x3) << 6)


@triton.jit
def unpack_int2_quartet(packed):
    """Split one packed byte into the (q0, q1, q2, q3) quartet as uint8."""
    return (
        packed & 0x3,
        (packed >> 2) & 0x3,
        (packed >> 4) & 0x3,
        (packed >> 6) & 0x3,
    )
