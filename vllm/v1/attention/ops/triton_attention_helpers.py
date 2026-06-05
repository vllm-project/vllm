# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared ``@triton.jit`` helpers used by the unified attention kernel
and ``reduce_segments``.

These are plain attention-loop helpers — mask building, ALiBi / QQ-bias
score post-processing, online-softmax bookkeeping, tile-loop bounds,
sequence lookup — extracted so the 2D and 3D paths of the unified
kernel (and any future consumer) share a single implementation.
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
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    USE_TD: tl.constexpr = False,
):
    """Compute ``(loop_lo, loop_hi, max_seq_prefix_len, tile_base)`` for the
    KV-tile loop, folding in: (1) the longest prefix any query in this q-block
    spans (clamped to ``seq_len``, or extended to it under mm_prefix);
    (2) sliding-window / chunked pruning; (3) 3D segment scoping when ``IS_3D``.

    ``tile_base`` is the absolute KV position the loop starts from
    (``seq_offset = tile_base + j * TILE_SIZE + offs_t``); it is 0 on every
    non-shifted path. The 2D pointer and 3D-segmented SWA paths both shift it
    to the exact window lower bound instead of the floor-rounded tile boundary,
    so the window's keys occupy the minimal ``ceil(W / TILE_SIZE)`` tiles; the
    floor-rounded start, by contrast, lets the masked leading slots (positions
    before the window in the first tile) push the window across up to one extra
    tile. (It also makes the online-softmax reduction order independent of the
    window offset mod ``TILE_SIZE``, so output is byte-identical across batch
    shapes - now on the 3D path as well.)

    Only the ``USE_TD`` tensor-descriptor path keeps ``tile_base = 0``: its load
    fetches a whole tile from one physical block (relying on
    ``BLOCK_SIZE % TILE_SIZE == 0``), which an arbitrary base would violate. On
    the 3D path the per-segment slice below runs in window-tile coordinates
    while ``tiles_per_segment`` stays ``seq_len``-derived, so the segments that
    do work remain a subset of those ``reduce_segments`` treats as valid and the
    extra high-index segments are harmlessly empty (``M = -inf``).
    """
    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    if USE_MM_PREFIX:
        # image bidirectional attention ranges require a full range
        # including q_block padding to make sure doc mask is correct
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # Default: iterate from absolute origin, all tiles.
    tile_base: tl.int32 = 0
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block.
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # Allowed keys for this Q-block: [first_allowed_key, last_allowed_key],
        # where each query q attends [q_abs - SLIDING_WINDOW + 1, q_abs].
        q_abs = context_len + qpos_lo
        if CHUNK_LOOKBACK > -1:
            # Chunked attention: lower bound is the lookback'th prior chunk.
            first_allowed_key = ((q_abs // CHUNK_SIZE) - CHUNK_LOOKBACK) * CHUNK_SIZE
        else:
            first_allowed_key = q_abs - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        if USE_TD:
            # Tensor-descriptor KV load fetches a whole tile from one physical
            # block (relies on BLOCK_SIZE % TILE_SIZE == 0); an arbitrary base
            # would straddle a block boundary, so keep the floor-rounded,
            # TILE_SIZE-aligned iteration.
            tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
            tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
        else:
            # 2D pointer and 3D-segmented paths: base-shift to the exact window
            # lower bound (see docstring). Each slot resolves its own physical
            # block, so a base-shifted tile straddling two KV blocks is fine; for
            # 3D the segment slice below runs in window-tile coordinates while
            # tiles_per_segment stays seq_len-derived, keeping the active
            # segments a subset of reduce_segments' valid range.
            tile_base = tl.maximum(0, first_allowed_key)
            tile_start = 0
            tile_end = cdiv_fn(
                tl.maximum(0, last_allowed_key + 1 - tile_base), TILE_SIZE
            )

    if IS_3D:
        loop_lo = max(segm_idx_or_0 * tiles_per_segment_or_0, tile_start)
        loop_hi = min((segm_idx_or_0 + 1) * tiles_per_segment_or_0, tile_end)
    else:
        loop_lo = tile_start
        loop_hi = tile_end

    return loop_lo, loop_hi, max_seq_prefix_len, tile_base


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
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
    """Build the KV mask for one tile.

    Causal (key <= query) by default; AND-ed with either chunked
    attention (``CHUNK_LOOKBACK >= 0``) or sliding window
    (``SLIDING_WINDOW > 0``); OR-ed with the bidirectional ranges from
    ``mm_prefix_range`` when PrefixLM / multimodal attention is active.
    Order matches FlexAttention: ``(causal AND window) OR mm_prefix``.
    Chunked attention takes precedence over sliding window when both
    are non-default — the launcher zeros ``CHUNK_LOOKBACK`` whenever
    sliding window is disabled.
    """
    # Compute attention mask: causal by default (key <= query)
    seq_mask = seq_offset[None, :] <= query_abs_pos

    # Apply sliding window / chunked attention to base mask
    # BEFORE mm_prefix OR.
    # Order must match FlexAttention:
    #   (causal AND sliding_window) OR mm_prefix
    if CHUNK_LOOKBACK > -1:
        seq_mask = seq_mask & (
            (query_abs_pos // CHUNK_SIZE - seq_offset[None, :] // CHUNK_SIZE)
            <= CHUNK_LOOKBACK
        )
    elif SLIDING_WINDOW > 0:
        seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

    # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
    # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
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
    rescaling its accumulator(s) by ``alpha[:, None]`` — done outside so
    kernels with a different number / shape of accumulators can reuse
    the same step.
    """
    # compute running maximum
    # m_j : (BLOCK_M,)
    m_j = tl.maximum(M, tl.max(S, axis=1))
    # For sliding window there's a chance the max is -inf due to masking of
    # the entire row. In this case we need to set m_j 0 to avoid NaN
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    # P : (BLOCK_M, TILE_SIZE)
    P = tl.exp(S - m_j[:, None])
    # l_j : (BLOCK_M,)
    l_j = tl.sum(P, axis=1)
    # alpha : (BLOCK_M, )
    alpha = tl.exp(M - m_j)
    # update constants
    L_new = L * alpha + l_j
    return m_j, L_new, P, alpha
