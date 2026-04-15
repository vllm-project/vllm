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
# Attention loop
# ===========================================================================


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
