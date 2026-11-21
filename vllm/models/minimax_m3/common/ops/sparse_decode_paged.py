# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page-split and direct single-page kernels for MiniMax-M3 decode."""

from vllm.triton_utils import tl, triton


@triton.jit
def _single_page_decode(
    q_ptr,
    kv_ptr,
    topk_ptr,
    block_table_ptr,
    seq_lens_ptr,
    k_scale_ptr,
    v_scale_ptr,
    output_ptr,
    NUM_KV_HEADS: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    STRIDE_BT_REQ: tl.constexpr,
    STRIDE_TOPK_HEAD: tl.constexpr,
    STRIDE_TOPK_TOKEN: tl.constexpr,
    STRIDE_TOPK_SLOT: tl.constexpr,
    STRIDE_K_SCALE_HEAD: tl.constexpr,
    STRIDE_K_SCALE_TOKEN: tl.constexpr,
    STRIDE_V_SCALE_HEAD: tl.constexpr,
    STRIDE_V_SCALE_TOKEN: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_OUTPUT_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    USE_FP8: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    sh, output_part = tl.program_id(0), tl.program_id(1)
    t, h = sh // NUM_KV_HEADS, sh % NUM_KV_HEADS
    req, local = t // DECODE_QUERY_LEN, t % DECODE_QUERY_LEN
    visible = tl.maximum(tl.load(seq_lens_ptr + req) - DECODE_QUERY_LEN + local + 1, 0)
    n, d, g = tl.arange(0, BLOCK_TOKENS), tl.arange(0, 128), tl.arange(0, 16)
    od = output_part * BLOCK_OUTPUT_DIM + tl.arange(0, BLOCK_OUTPUT_DIM)
    slots, offset = n // 128, n % 128
    valid = (slots < NUM_SLOTS) & (slots < tl.minimum(16, tl.cdiv(visible, 128)))
    logical = tl.load(
        topk_ptr
        + h * STRIDE_TOPK_HEAD
        + t * STRIDE_TOPK_TOKEN
        + slots * STRIDE_TOPK_SLOT,
        valid,
        other=0,
    )
    physical = tl.load(
        block_table_ptr + req * STRIDE_BT_REQ + logical, valid, other=0
    ).to(tl.int64)
    mask = valid & (logical * 128 + offset < visible)
    base = (physical * NUM_KV_HEADS + h) * 128 * 256 + offset * 256
    q = tl.load(q_ptr + (sh * 16 + g[:, None]) * 128 + d[None, :])
    k = tl.load(kv_ptr + base[None, :] + d[:, None], mask[None, :], other=0.0).to(
        q.dtype
    )
    if USE_FP8:
        if SCALE_MODE == 1:
            k = (k * tl.load(k_scale_ptr)).to(q.dtype)
        elif SCALE_MODE == 2:
            ks = tl.load(
                k_scale_ptr
                + h * STRIDE_K_SCALE_HEAD
                + (physical * 128 + offset) * STRIDE_K_SCALE_TOKEN,
                mask,
                other=1.0,
            )
            k = (k * ks[None, :]).to(q.dtype)
    scores = tl.dot(q, k) * (SM_SCALE * 1.4426950408889634)
    scores = tl.where(mask[None, :], scores, float("-inf"))
    maximum = tl.max(scores, 1)
    maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    prob = tl.exp2(scores - maximum[:, None])
    denom = tl.sum(prob, 1)
    # Load V after QK so the compiler may reuse K's shared-memory lifetime.
    v = tl.load(
        kv_ptr + base[:, None] + 128 + od[None, :], mask[:, None], other=0.0
    ).to(q.dtype)
    if USE_FP8:
        if SCALE_MODE == 1:
            v = (v * tl.load(v_scale_ptr)).to(q.dtype)
        elif SCALE_MODE == 2:
            vs = tl.load(
                v_scale_ptr
                + h * STRIDE_V_SCALE_HEAD
                + (physical * 128 + offset) * STRIDE_V_SCALE_TOKEN,
                mask,
                other=1.0,
            )
            v = (v * vs[:, None]).to(q.dtype)
    accum = tl.dot(prob.to(q.dtype), v)
    out = accum / tl.where(denom > 0, denom, 1.0)[:, None]
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(output_ptr + (sh * 16 + g[:, None]) * 128 + od[None, :], out)


@triton.jit
def _page_split_decode(
    q_ptr,
    kv_ptr,
    topk_ptr,
    block_table_ptr,
    seq_lens_ptr,
    k_scale_ptr,
    v_scale_ptr,
    partial_ptr,
    lse_ptr,
    NUM_KV_HEADS: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    STRIDE_BT_REQ: tl.constexpr,
    STRIDE_TOPK_HEAD: tl.constexpr,
    STRIDE_TOPK_TOKEN: tl.constexpr,
    STRIDE_TOPK_SLOT: tl.constexpr,
    STRIDE_K_SCALE_HEAD: tl.constexpr,
    STRIDE_K_SCALE_TOKEN: tl.constexpr,
    STRIDE_V_SCALE_HEAD: tl.constexpr,
    STRIDE_V_SCALE_TOKEN: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SM_SCALE: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    USE_FP8: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    sh = tl.program_id(0)
    t, h = sh // NUM_KV_HEADS, sh % NUM_KV_HEADS
    piece = tl.program_id(1)
    slot, sub = piece // (128 // TOKENS_PER_SPLIT), piece % (128 // TOKENS_PER_SPLIT)
    req, local = t // DECODE_QUERY_LEN, t % DECODE_QUERY_LEN
    visible = tl.maximum(tl.load(seq_lens_ptr + req) - DECODE_QUERY_LEN + local + 1, 0)
    valid = slot < tl.minimum(16, tl.cdiv(visible, 128))
    logical = tl.load(
        topk_ptr
        + h * STRIDE_TOPK_HEAD
        + t * STRIDE_TOPK_TOKEN
        + slot * STRIDE_TOPK_SLOT,
        valid,
        other=0,
    )
    active = valid & (logical * 128 + sub * TOKENS_PER_SPLIT < visible)
    d, g = tl.arange(0, 128), tl.arange(0, 16)
    po = ((sh * NUM_SPLITS + piece) * 16 + g[:, None]) * 128 + d[None, :]
    lo = (sh * NUM_SPLITS + piece) * 16 + g
    if active:
        page = tl.load(
            block_table_ptr + req * STRIDE_BT_REQ + logical, valid, other=0
        ).to(tl.int64)
        n = sub * TOKENS_PER_SPLIT + tl.arange(0, TOKENS_PER_SPLIT)
        mask = valid & (logical * 128 + n < visible)
        q = tl.load(
            q_ptr + ((t * NUM_KV_HEADS + h) * 16 + g[:, None]) * 128 + d[None, :]
        )
        base = (page * NUM_KV_HEADS + h) * 128 * 256
        k = tl.load(
            kv_ptr + base + n[None, :] * 256 + d[:, None], mask[None, :], other=0.0
        ).to(q.dtype)
        v = tl.load(
            kv_ptr + base + n[:, None] * 256 + 128 + d[None, :],
            mask[:, None],
            other=0.0,
        ).to(q.dtype)
        if USE_FP8:
            if SCALE_MODE == 1:
                k = (k * tl.load(k_scale_ptr)).to(q.dtype)
                v = (v * tl.load(v_scale_ptr)).to(q.dtype)
            elif SCALE_MODE == 2:
                ks = tl.load(
                    k_scale_ptr
                    + h * STRIDE_K_SCALE_HEAD
                    + (page * 128 + n) * STRIDE_K_SCALE_TOKEN,
                    mask,
                    other=1.0,
                )
                vs = tl.load(
                    v_scale_ptr
                    + h * STRIDE_V_SCALE_HEAD
                    + (page * 128 + n) * STRIDE_V_SCALE_TOKEN,
                    mask,
                    other=1.0,
                )
                k = (k * ks[None, :]).to(q.dtype)
                v = (v * vs[:, None]).to(q.dtype)
        score = tl.dot(q, k) * (SM_SCALE * 1.4426950408889634)
        score = tl.where(mask[None, :], score, float("-inf"))
        m = tl.max(score, 1)
        safe_m = tl.where(m == float("-inf"), 0.0, m)
        prob = tl.exp2(score - safe_m[:, None])
        z = tl.sum(prob, 1)
        accum = tl.dot(prob.to(q.dtype), v)
        out = accum / tl.where(z > 0, z, 1.0)[:, None]
        lse = tl.where(z > 0, safe_m + tl.log2(z), float("-inf"))
        tl.store(partial_ptr + po, out)
        tl.store(lse_ptr + lo, lse)
    else:
        # Every slot is overwritten on every replay: no stale scratch on shrink.
        tl.store(partial_ptr + po, tl.full((16, 128), 0.0, tl.float32))
        tl.store(lse_ptr + lo, tl.full((16,), float("-inf"), tl.float32))

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _merge_page_splits(
    partial_ptr,
    lse_ptr,
    output_ptr,
    NUM_SPLITS: tl.constexpr,
    PADDED_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    sh, gd = tl.program_id(0), tl.program_id(1)
    g, part = gd // (128 // BLOCK_D), gd % (128 // BLOCK_D)
    ss = tl.arange(0, PADDED_SPLITS)
    d = part * BLOCK_D + tl.arange(0, BLOCK_D)
    lo = (sh * NUM_SPLITS + ss) * 16 + g
    po = ((sh * NUM_SPLITS + ss[:, None]) * 16 + g) * 128 + d[None, :]
    lse_values = tl.load(lse_ptr + lo, ss < NUM_SPLITS, other=float("-inf"))
    m = tl.max(lse_values, 0)
    m = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp2(lse_values - m)
    z = tl.sum(w, 0)
    w = w / tl.where(z > 0, z, 1.0)
    p = tl.load(partial_ptr + po, (ss < NUM_SPLITS)[:, None], other=0.0).to(tl.float32)
    out = tl.sum(p * w[:, None], 0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(output_ptr + (sh * 16 + g) * 128 + d, out)
