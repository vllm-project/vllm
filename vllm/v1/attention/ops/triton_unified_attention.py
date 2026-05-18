# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score,
    apply_softcap,
    cdiv_fn,
    compute_kv_seq_mask,
    compute_tile_loop_bounds,
    find_seq_idx,
    init_softmax_M,
    load_qq_bias_tile,
    resolve_seq_and_query_len,
    softmax_step,
    store_segm_reduce_scalars,
)
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)
is_batch_invariant = envs.VLLM_BATCH_INVARIANT
float8_info = torch.finfo(current_platform.fp8_dtype())


@triton.jit
def _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    """Cast a loaded KV tile to Q's dtype, dequantizing if needed.

    Modes handled inside the core kernel:

    - ``KV_QUANT_MODE == 0`` (NONE) and ``2`` (INT8 per-token-head) and
      ``3`` (FP8 per-token-head): plain cast.  Per-token-head modes apply
      their scales separately on S/P inside the loop.
    - ``KV_QUANT_MODE == 1`` (FP8 per-tensor): dequantize using the
      tensor-wide scale.
    """
    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
    return data.to(Q.dtype)


# ---------------------------------------------------------------------------
# Tensor-descriptor (TD) helpers
#
# Used when the caller enables ``USE_TD`` (Intel Xe2/Xe3 HW 2D block reads).
# When ``USE_TD`` is False the helpers are dead-code-eliminated at Triton
# compile time, leaving the pointer-arithmetic load/store path untouched.
# ---------------------------------------------------------------------------


@triton.jit
def _load_q_td(
    query_ptr,
    q_block_local_len,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    cur_batch_in_all_start_index,
    q_block_local_idx,
    kv_head_idx,
    num_queries_per_kv: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Load Q via a 2D tensor descriptor.

    Caller guarantees (via the wrapper's ``use_td_qo`` gate):
      * ``HEAD_SIZE == HEAD_SIZE_PADDED`` (head_size is a power of 2),
      * ``num_queries_per_kv`` is a power of 2,
      * the ``num_queries_per_kv`` heads of the current KV group are
        contiguous in memory (``query_stride_1 == HEAD_SIZE``, which is
        the default vLLM query layout).

    Under those preconditions the inner two axes are flattened into one
    row of size ``num_queries_per_kv * HEAD_SIZE`` with stride 1, which
    avoids the non-power-of-2 ``block_shape`` error from the Triton
    tensor-descriptor validator.  Returns (BLOCK_M, HEAD_SIZE_PADDED).
    """
    q_base = (
        query_ptr
        + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * query_stride_0
        + (kv_head_idx * num_queries_per_kv) * query_stride_1
    )
    q_desc = tl.make_tensor_descriptor(
        base=q_base,
        shape=(q_block_local_len, num_queries_per_kv * HEAD_SIZE),
        strides=(query_stride_0, 1),
        block_shape=(BLOCK_Q, num_queries_per_kv * HEAD_SIZE_PADDED),
    )
    return q_desc.load([0, 0]).reshape(BLOCK_M, HEAD_SIZE_PADDED)


@triton.jit
def _load_kv_tile_td(
    cache_ptr,
    physical_block_idx_scalar,
    kv_head_idx,
    offset_in_block,
    stride_cache_0: tl.int64,
    stride_cache_1: tl.int64,
    stride_cache_2: tl.int64,
    stride_cache_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Load a KV cache tile via tensor descriptor.

    Returns shape (TILE_SIZE, HEAD_SIZE_PADDED). Caller transposes for K.
    Tensor descriptors zero-pad reads beyond the shape boundary, so
    ``HEAD_SIZE_PADDED > HEAD_SIZE`` is handled correctly.
    """
    base = (
        cache_ptr
        + physical_block_idx_scalar * stride_cache_0
        + kv_head_idx * stride_cache_2
    )
    desc = tl.make_tensor_descriptor(
        base=base,
        shape=(BLOCK_SIZE, HEAD_SIZE),
        strides=(stride_cache_1, stride_cache_3),
        block_shape=(TILE_SIZE, HEAD_SIZE_PADDED),
    )
    return desc.load([offset_in_block, 0])


@triton.jit
def _store_output_td(
    base_ptr,
    acc,
    q_block_local_len,
    stride_token: tl.int64,
    stride_head: tl.int64,
    num_queries_per_kv: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Store an output tile via a tensor descriptor.

    The 2D and 3D epilogues differ only in ``base_ptr`` and the
    ``(stride_token, stride_head)`` pair: 2D writes directly to the
    flat output buffer, 3D writes to a single per-segment slice of
    ``segm_output_ptr``.  Descriptor shape / block_shape / reshape
    are the same in both modes, so share one helper.
    """
    acc = acc.to(base_ptr.dtype.element_ty)
    output_desc = tl.make_tensor_descriptor(
        base=base_ptr,
        shape=(q_block_local_len, num_queries_per_kv, HEAD_SIZE),
        strides=(stride_token, stride_head, 1),
        block_shape=(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED),
    )
    output_desc.store(
        [0, 0, 0],
        acc.reshape(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED),
    )


@triton.jit
def kernel_unified_attention(
    # Output destination for the 2D path.  In 3D mode per-segment partials
    # go to the ``segm_*`` tensors (see bottom of signature) and
    # ``output_ptr`` is unused (callers may pass any non-null pointer).
    output_ptr,
    # Inputs
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    # Scalars
    scale,
    k_scale,
    v_scale,
    out_scale,
    softcap,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    # Toggles 2D vs 3D layout.  The 2D path runs the full sequence in one
    # tile loop and writes to ``output_ptr``.  The 3D path scopes the loop
    # to ``[segm_idx, segm_idx+1) × tiles_per_segment`` and writes
    # per-segment partials, finalized by ``reduce_segments``.
    IS_3D: tl.constexpr,
    # Parameters below default to None so Triton can skip materialising them
    # on call sites where the corresponding constexpr branch is dead.
    # Credit: @quinnlp identified this as a perf regression source in
    # intel/intel-xpu-backend-for-triton#6758 (review comment r3204641104).
    # Per-segment outputs: used in 3D mode; unused in 2D (IS_3D=False).
    segm_output_ptr=None,
    segm_max_ptr=None,
    segm_expsum_ptr=None,
    # Per-(token, head) scale caches: used iff KV_QUANT_MODE in {2, 3}.
    k_scale_cache_ptr=None,
    v_scale_cache_ptr=None,
    stride_ks_blk: tl.int64 = None,
    stride_ks_slot: tl.int64 = None,
    stride_ks_head: tl.int64 = None,
    stride_vs_blk: tl.int64 = None,
    stride_vs_slot: tl.int64 = None,
    stride_vs_head: tl.int64 = None,
    # KV cache quantization mode handled inside this kernel via constexpr
    # branches: NONE (0), FP8_PER_TENSOR (1), INT8_PER_TOKEN_HEAD (2),
    # FP8_PER_TOKEN_HEAD (3).
    KV_QUANT_MODE: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    # Chunked / block-local attention.  ``CHUNK_LOOKBACK >= 0`` enables
    # chunked masking (used by Gemma3 block-local layers); takes precedence
    # over ``SLIDING_WINDOW`` inside the helpers.  ``-1`` disables.
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    # Tensor-descriptor load/store for HW 2D block reads on Intel Xe2/Xe3.
    # ``USE_TD`` gates KV tile loads; ``USE_TD_QO`` separately gates Q/output
    # (see ``unified_attention`` wrapper for the gating rules).
    USE_TD: tl.constexpr = False,
    USE_TD_QO: tl.constexpr = False,
):
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2

    if USE_TD:
        tl.static_assert(
            BLOCK_SIZE % TILE_SIZE == 0,
            "USE_TD requires BLOCK_SIZE to be a multiple of TILE_SIZE",
        )

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    # Number of valid query rows in this block (used by TD descriptor
    # shapes, but always computed so the variable stays in scope).
    q_block_local_len = tl.minimum(
        BLOCK_Q, cur_batch_query_len - q_block_local_idx * BLOCK_Q
    )

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    if USE_TD_QO:
        Q = _load_q_td(
            query_ptr,
            q_block_local_len,
            query_stride_0,
            query_stride_1,
            cur_batch_in_all_start_index,
            q_block_local_idx,
            kv_head_idx,
            num_queries_per_kv,
            BLOCK_Q,
            BLOCK_M,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
        )
    else:
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=0.0,
        )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
    )

    # iterate through tiles (now limited to the sliding window range)
    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        if USE_TD:
            # All TILE_SIZE slots within a single KV tile map to one
            # physical block (guaranteed by ``BLOCK_SIZE % TILE_SIZE == 0``
            # from the static_assert above), so load the block index as
            # a scalar instead of a broadcast reduction.
            offset_in_block = (j * TILE_SIZE) % BLOCK_SIZE
            physical_block_scalar = tl.load(
                block_tables_ptr + block_table_offset + (j * TILE_SIZE) // BLOCK_SIZE
            ).to(tl.int64)
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = _load_kv_tile_td(
                key_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            ).T
            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = _load_kv_tile_td(
                value_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            )
            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )
        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)

        # Per-(token, head) scales for INT8 / FP8 per-token-head modes.
        if USE_PER_TOKEN_HEAD_SCALES:
            scale_idx = (
                physical_block_idx * stride_ks_blk
                + (seq_offset % BLOCK_SIZE) * stride_ks_slot
                + kv_head_idx * stride_ks_head
            )
            k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )
            v_scale_idx = (
                physical_block_idx * stride_vs_blk
                + (seq_offset % BLOCK_SIZE) * stride_vs_slot
                + kv_head_idx * stride_vs_head
            )
            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
            )

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
        )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: fuse softmax_scale with per-head k_scale
            # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
            S += tl.dot(Q, K) * (scale * k_token_head_scales[None, :])
        else:
            S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale to P instead of V.
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue ---------------------------------------------------------
    if IS_3D:
        # Store per-segment partials; finalized by ``reduce_segments``.
        if USE_TD_QO:
            # 3D target: segm_output[token, head, segm_idx, :].  Advance
            # the base to the correct (token-start, head-start, segm)
            # slice; strides step between tokens / heads of the flattened
            # (T, H, SEGS, PAD) layout.
            segm_base = (
                segm_output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q).to(
                    tl.int64
                )
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + (kv_head_idx * num_queries_per_kv)
                * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
            )
            _store_output_td(
                segm_base,
                acc,
                q_block_local_len,
                num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            segm_output_offset = (
                query_offset_0[:, None].to(tl.int64)
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
                + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
            )
            tl.store(
                segm_output_ptr + segm_output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        if USE_FP8:
            acc = acc * tl.load(out_scale)
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
        if USE_TD_QO:
            # 2D target: flat output[token, head, :].  Strides come
            # straight from the caller (``output_stride_0`` per token,
            # ``output_stride_1`` per head).
            output_base = (
                output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q)
                * output_stride_0
                + (kv_head_idx * num_queries_per_kv) * output_stride_1
            )
            _store_output_td(
                output_base,
                acc,
                q_block_local_len,
                output_stride_0,
                output_stride_1,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            output_offset = (
                query_offset_0[:, None] * output_stride_0
                + query_offset_1[:, None] * output_stride_1
                + offs_d[None, :]
            )
            tl.store(
                output_ptr + output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization."""
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    return 16 if element_size >= 2 else 32


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    use_alibi_sqrt=False,
    # KV cache quantization mode and per-token-head scale caches.
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE,
    k_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    v_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    # Chunked attention: restrict attention to aligned blocks with lookback.
    chunk_lookback=-1,
    # Tensor-descriptor mode: use ``tl.make_tensor_descriptor`` for Q/K/V
    # loads and output stores.  Enables HW 2D block reads on Intel Xe2/Xe3.
    # The non-TD branch is dead-code-eliminated at Triton compile time so
    # disabling this flag costs nothing.
    use_td: bool = False,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_per_token_head_scales = kv_quant_mode in (
        KVQuantMode.INT8_PER_TOKEN_HEAD,
        KVQuantMode.FP8_PER_TOKEN_HEAD,
    )
    if use_per_token_head_scales:
        assert k_scale_cache is not None and v_scale_cache is not None, (
            f"{kv_quant_mode.name} requires k_scale_cache / v_scale_cache"
        )

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Compute chunked block size from sliding window if needed.
    chunk_size = -1
    if sliding_window_val > 0 and chunk_lookback > -1:
        chunk_size = sliding_window_val // (chunk_lookback + 1)
        assert chunk_size > 0, "sliding_window must be > chunk_lookback+1"
    elif sliding_window_val <= 0:
        chunk_lookback = -1

    TILE_SIZE_PREFILL = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=True
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )

    # USE_TD requires BLOCK_SIZE % TILE_SIZE == 0 (enforced by a
    # ``tl.static_assert`` in the kernel).  The default prefill tile
    # size (32) is larger than a common ``block_size=16``, so clamp it
    # down when TD is enabled.  Zero overhead when disabled.
    if use_td:
        TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, block_size)
        TILE_SIZE_DECODE = min(TILE_SIZE_DECODE, block_size)

    # Tensor descriptors for Q load / output store require every element
    # of ``block_shape`` to be a power of 2.  ``num_queries_per_kv`` is
    # not always pow2 (e.g. Qwen2-7B: 28 / 4 = 7), so gate the Q/O paths
    # separately from the KV tile loads (whose ``block_shape`` does not
    # include ``num_queries_per_kv``).
    #
    # The Q/O descriptors also encode ``HEAD_SIZE_PADDED`` on the inner
    # axis while the backing buffers (both flat output and per-segment
    # output) are laid out with ``HEAD_SIZE``.  When they differ (e.g.
    # Phi-3's head_size=96 → HEAD_SIZE_PADDED=128) the store would spill
    # padded lanes into neighbouring heads because tensor-descriptor
    # stores don't mask the padded tail.  Fall back to the pointer path
    # for Q/O in that case — KV tile loads are unaffected because their
    # ``shape`` already matches ``block_shape`` on the inner axis.
    head_size_padded = triton.next_power_of_2(head_size)
    _is_pow2_nq = (num_queries_per_kv & (num_queries_per_kv - 1)) == 0
    _is_pow2_hs = head_size == head_size_padded
    use_td_qo = use_td and _is_pow2_nq and _is_pow2_hs

    # ``_load_q_td`` / ``_store_output_td`` flatten ``(num_queries_per_kv,
    # HEAD_SIZE)`` into a single contiguous inner axis.  That's only
    # equivalent to the pointer path when the ``num_queries_per_kv`` heads
    # for this KV group start at ``kv_head_idx * num_queries_per_kv`` and
    # lie exactly HEAD_SIZE apart — i.e. ``query_stride_1 == HEAD_SIZE``
    # and ``output_stride_1 == head_size``.  This is the default vLLM
    # query/output layout; assert it explicitly so we fail fast if a
    # future caller passes a non-contiguous query tensor.
    if use_td_qo:
        assert q.stride(1) == head_size, (
            f"USE_TD_QO requires contiguous query heads "
            f"(q.stride(1) = {q.stride(1)} != head_size = {head_size}); "
            f"set VLLM_TRITON_ATTN_USE_TD=0 or pad the query layout."
        )
        assert out.stride(1) == head_size, (
            f"USE_TD_QO requires contiguous output heads "
            f"(out.stride(1) = {out.stride(1)} != head_size = {head_size})."
        )

    # Launch the 2D kernel if
    # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # 2. The batch includes at least one prefill request, or
    # 3. The number of sequences exceeds the configured threshold, or
    # 4. Batch invariance is enabled
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    # The kernel signature is the same for 2D and 3D — only the launch
    # grid + a handful of constexpr toggles differ.  Per-token-head scale
    # caches and their strides are required arguments; non-per-token-head
    # modes pass dummy zeros (the code path is dead-code eliminated by
    # the ``USE_PER_TOKEN_HEAD_SCALES`` constexpr branch in the kernel).
    if use_per_token_head_scales:
        ks_strides = k_scale_cache.stride()
        vs_strides = v_scale_cache.stride()
        ks_blk, ks_slot, ks_head = ks_strides[0], ks_strides[1], ks_strides[2]
        vs_blk, vs_slot, vs_head = vs_strides[0], vs_strides[1], vs_strides[2]
        k_scale_ptr = k_scale_cache
        v_scale_ptr = v_scale_cache
    else:
        ks_blk = ks_slot = ks_head = 0
        vs_blk = vs_slot = vs_head = 0
        # Pass the K cache as a stand-in pointer; never dereferenced.
        k_scale_ptr = k
        v_scale_ptr = v

    # 3D needs real segm tensors; 2D never touches them but Triton wants
    # a non-null pointer.  Reuse ``out`` as the placeholder.
    segm_output_ptr = softmax_segm_output if use_3d else out
    segm_max_ptr = softmax_segm_max if use_3d else out
    segm_expsum_ptr = softmax_segm_expsum if use_3d else out
    num_segments = num_par_softmax_segments if use_3d else 1

    grid: tuple[Any, ...]
    if not use_3d:
        grid = (total_num_q_blocks, num_kv_heads)
        tile_size = TILE_SIZE_PREFILL
    else:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        tile_size = TILE_SIZE_DECODE

    kernel_unified_attention[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        k_scale_cache_ptr=k_scale_ptr,
        v_scale_cache_ptr=v_scale_ptr,
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        out_scale=1 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        mm_prefix_range_ptr=mm_prefix_range,
        SLIDING_WINDOW=(1 + window_size[0]),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        stride_ks_blk=ks_blk,
        stride_ks_slot=ks_slot,
        stride_ks_head=ks_head,
        stride_vs_blk=vs_blk,
        stride_vs_slot=vs_slot,
        stride_vs_head=vs_head,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=output_scale is not None,
        IS_3D=use_3d,
        KV_QUANT_MODE=kv_quant_mode,
        CHUNK_LOOKBACK=chunk_lookback,
        CHUNK_SIZE=chunk_size,
        USE_TD=use_td,
        USE_TD_QO=use_td_qo,
    )

    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )
