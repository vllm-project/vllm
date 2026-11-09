# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-pass row statistics for the DiffusionGemma denoise sampler.

The step needs, for every canvas position, the argmax of the
temperature-scaled logits, a Gumbel-max sample from them, the entropy of
their softmax, and the softmax itself in the model dtype for the
self-conditioning matmul. As PyTorch ops these are several passes over a
`[rows, vocab]` fp32 tensor plus a same-sized noise tensor. The Triton
kernel reads each row once with an online max and sum, draws the noise
inline, and writes the probabilities once.

The logprobs of a converging slot are written to a per-slot stash by
`stash_converged_logprobs`, which skips the rows of slots that did not
converge.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _row_stats_kernel(
    logits_ptr,
    logits_stride,
    temps_ptr,
    argmax_ptr,
    sample_ptr,
    entropy_ptr,
    probs_ptr,
    probs_stride,
    seed,
    V,
    CL,
    WRITE_PROBS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    temp = tl.load(temps_ptr + row // CL).to(tl.float32)
    inv = 1.0 / tl.maximum(temp, 1e-10)
    noise = (temp > 0).to(tl.float32)
    base = logits_ptr + row.to(tl.int64) * logits_stride
    rng_base = row.to(tl.int64) * V

    # The running max stays in raw logit units and the temperature is applied
    # to differences from it. Scaling the row first breaks at a zero
    # temperature: the row is multiplied by 1e10, fp32 spacing there is about
    # a thousand, and exp(x - max) becomes 0 or inf for the max element itself
    # whenever the two sides round differently.
    m = float("-inf")  # running max of the raw row
    z = 0.0  # sum exp(d), d = (x - m) * inv
    s = 0.0  # sum d exp(d)
    best = float("-inf")
    best_idx = 0
    best_noisy = float("-inf")
    best_noisy_idx = 0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)

        block_max = tl.max(x, 0)
        new_m = tl.maximum(m, block_max)
        # Until a block holds a finite logit the running max stays -inf, and
        # top_k/top_p leave whole blocks masked. x - new_m would then be
        # -inf - -inf = NaN and poison z for the rest of the row, so subtract
        # 0 instead: every d is still -inf and every e 0.
        safe_m = tl.where(new_m > float("-inf"), new_m, 0.0)
        # Rescale the running sums to the new max. While m is -inf, z and s
        # are 0, so alpha and the shift are zeroed rather than computed from
        # -inf - -inf.
        alpha = tl.where(m > float("-inf"), tl.exp((m - safe_m) * inv), 0.0)
        shift = tl.where(z > 0, (m - safe_m) * inv * z, 0.0)
        d = (x - safe_m) * inv
        e = tl.exp(d)
        z = z * alpha + tl.sum(e, 0)
        # A -inf logit (top_k/top_p, or the row's padding) has d = -inf and
        # e = 0, and -inf * 0 is NaN. The where drops it from the sum.
        s = (s + shift) * alpha + tl.sum(tl.where(e > 0, d * e, 0.0), 0)
        m = new_m

        block_idx = tl.argmax(x, 0)
        take = block_max > best
        best_idx = tl.where(take, start + block_idx, best_idx)
        best = tl.where(take, block_max, best)

        u = tl.rand(seed, rng_base + offs)
        u = tl.maximum(u, 1e-20)
        gumbel = -tl.log(-tl.log(u))
        noisy = tl.where(mask, x * inv + gumbel * noise, float("-inf"))
        noisy_max = tl.max(noisy, 0)
        noisy_idx = tl.argmax(noisy, 0)
        take_n = noisy_max > best_noisy
        best_noisy_idx = tl.where(take_n, start + noisy_idx, best_noisy_idx)
        best_noisy = tl.where(take_n, noisy_max, best_noisy)

    # With d = (x - m) * inv: p = e^d / z, log Z = log z in d units, and
    # H = -sum p (d - log z) = log z - s / z.
    # tl.store casts to the pointee dtype. mypy type-checks the kernel body
    # as Python and rejects .to() on the int accumulators.
    tl.store(argmax_ptr + row, best_idx)
    tl.store(sample_ptr + row, best_noisy_idx)
    tl.store(entropy_ptr + row, tl.log(z) - s / z)

    if WRITE_PROBS:
        out = probs_ptr + row.to(tl.int64) * probs_stride
        for start in range(0, V, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < V
            x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
            p = tl.exp((x - m) * inv) / z
            tl.store(out + offs, p.to(probs_ptr.dtype.element_ty), mask=mask)


def sample_row_stats(
    logits: torch.Tensor,
    temps: torch.Tensor,
    canvas_len: int,
    seed: int,
    probs_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Argmax, Gumbel-max sample, entropy and (optionally) softmax per row.

    `logits` is `[rows, vocab]` and row `i` uses temperature
    `temps[i // canvas_len]`. A zero temperature is greedy: the sample is
    the argmax, and the entropy is the reference's, which clamps the
    temperature at 1e-10.
    """
    rows, vocab = logits.shape
    device = logits.device
    argmax = torch.empty(rows, dtype=torch.int64, device=device)
    sample = torch.empty(rows, dtype=torch.int64, device=device)
    entropy = torch.empty(rows, dtype=torch.float32, device=device)
    if probs_dtype is not None:
        probs = torch.empty(rows, vocab, dtype=probs_dtype, device=device)
        probs_arg, probs_stride = probs, probs.stride(0)
    else:
        probs = None
        probs_arg, probs_stride = entropy, 0
    if rows == 0:
        return argmax, sample, entropy, probs
    _row_stats_kernel[(rows,)](
        logits,
        logits.stride(0),
        temps,
        argmax,
        sample,
        entropy,
        probs_arg,
        probs_stride,
        seed,
        vocab,
        canvas_len,
        WRITE_PROBS=probs is not None,
        BLOCK=4096,
        num_warps=8,
    )
    return argmax, sample, entropy, probs


def sample_row_stats_reference(
    logits: torch.Tensor,
    temps: torch.Tensor,
    canvas_len: int,
    probs_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The PyTorch form of the same statistics, for tests and non-CUDA runs."""
    temp = temps.repeat_interleave(canvas_len).float()[:, None]
    scaled = logits.float() / temp.clamp(min=1e-10)
    u = torch.rand_like(scaled).clamp(min=1e-20)
    noisy = scaled + (-torch.log(-torch.log(u))) * (temp > 0).float()
    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()
    # Masked (-inf) columns: 0 * -inf is NaN, so they are dropped from the sum.
    entropy = -torch.where(probs > 0, probs * log_probs, 0.0).sum(dim=-1)
    out_probs = probs.to(probs_dtype) if probs_dtype is not None else None
    return scaled.argmax(dim=-1), noisy.argmax(dim=-1), entropy, out_probs


# The stash kernels reproduce compute_topk_scores' values (tied top-k ids
# aside) only while each block is summed as _topk_log_softmax_kernel sums it:
# 1024 wide, 4 warps, in block order.
_LP_BLOCK = 1024
_LP_BLOCKS_PER_CHUNK = 16
_NO_KEY = tl.constexpr(-(2**63))


@triton.jit
def _lp_skip(keep_ptr, valid_ptr, j, pos):
    return (tl.load(keep_ptr + j) == 0) | (pos >= tl.load(valid_ptr + j))


@triton.jit
def _lp_skip_topk(keep_ptr, valid_ptr, slots_ptr, num_ids_ptr, j, pos, HAS_IDS):
    skip = _lp_skip(keep_ptr, valid_ptr, j, pos)
    if HAS_IDS:
        # Requests with their own token ids report those instead.
        skip = skip | (tl.load(num_ids_ptr + tl.load(slots_ptr + j)) > 0)
    return skip


@triton.jit
def _lp_load(logits_ptr, logits_stride, src_ptr, temps_ptr, j, pos, W, offs, V, other):
    row = (tl.load(src_ptr + j) * W + pos).to(tl.int64)
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=offs < V, other=other)
    # Round-to-nearest fp32 division, as torch divides.
    return tl.div_rn(x.to(tl.float32), tl.load(temps_ptr + j))


@triton.jit
def _lp_key32(x):
    # fp32 -> int32 with the same order.
    bits = x.to(tl.int32, bitcast=True)
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


@triton.jit
def _lp_max_kernel(
    logits_ptr,
    logits_stride,
    src_ptr,
    keep_ptr,
    valid_ptr,
    temps_ptr,
    chunk_max_ptr,
    chunk_arg_ptr,
    block_max_ptr,
    W,
    V,
    NUM_CHUNKS,
    NUM_BLOCKS,
    BLOCK: tl.constexpr,
    BLOCKS_PER_CHUNK: tl.constexpr,
    STORE_BLOCK_MAX: tl.constexpr,
):
    j = tl.program_id(0)
    pos = tl.program_id(1)
    chunk = tl.program_id(2)
    if _lp_skip(keep_ptr, valid_ptr, j, pos):
        return
    r = (j * W + pos).to(tl.int64)
    max_val = float("-inf")
    best_idx = 0
    for b in range(BLOCKS_PER_CHUNK):
        block_id = chunk * BLOCKS_PER_CHUNK + b
        offs = block_id * BLOCK + tl.arange(0, BLOCK)
        x = _lp_load(
            logits_ptr,
            logits_stride,
            src_ptr,
            temps_ptr,
            j,
            pos,
            W,
            offs,
            V,
            float("-inf"),
        )
        block_max, block_idx = tl.max(x, 0, return_indices=True)
        if STORE_BLOCK_MAX:
            tl.store(
                block_max_ptr + r * NUM_BLOCKS + block_id,
                block_max,
                mask=block_id < NUM_BLOCKS,
            )
        take = block_max > max_val
        best_idx = tl.where(take, block_id * BLOCK + block_idx, best_idx)
        max_val = tl.where(take, block_max, max_val)
    tl.store(chunk_max_ptr + r * NUM_CHUNKS + chunk, max_val)
    tl.store(chunk_arg_ptr + r * NUM_CHUNKS + chunk, best_idx)


@triton.jit
def _lp_sum_kernel(
    logits_ptr,
    logits_stride,
    src_ptr,
    keep_ptr,
    valid_ptr,
    temps_ptr,
    chunk_max_ptr,
    block_sum_ptr,
    block_cnt_ptr,
    W,
    V,
    NUM_CHUNKS,
    NUM_BLOCKS,
    BLOCK: tl.constexpr,
    BLOCKS_PER_CHUNK: tl.constexpr,
    CHUNKS_POW2: tl.constexpr,
):
    j = tl.program_id(0)
    pos = tl.program_id(1)
    chunk = tl.program_id(2)
    if _lp_skip(keep_ptr, valid_ptr, j, pos):
        return
    r = (j * W + pos).to(tl.int64)
    c = tl.arange(0, CHUNKS_POW2)
    maxes = tl.load(
        chunk_max_ptr + r * NUM_CHUNKS + c, mask=c < NUM_CHUNKS, other=float("-inf")
    )
    max_val = tl.max(maxes, 0)
    for b in range(BLOCKS_PER_CHUNK):
        block_id = chunk * BLOCKS_PER_CHUNK + b
        block = block_id * BLOCK + tl.arange(0, BLOCK)
        x = _lp_load(
            logits_ptr, logits_stride, src_ptr, temps_ptr, j, pos, W, block, V, 0.0
        )
        e = tl.exp(x - max_val)
        e = tl.where(block < V, e, 0.0)
        out = r * NUM_BLOCKS + block_id
        in_row = block_id < NUM_BLOCKS
        tl.store(block_sum_ptr + out, tl.sum(e), mask=in_row)
        rank = tl.sum(((x >= max_val) & (block < V)).to(tl.int32))
        tl.store(block_cnt_ptr + out, rank, mask=in_row)


@triton.jit
def _lp_topk_kernel(
    logits_ptr,
    logits_stride,
    src_ptr,
    keep_ptr,
    valid_ptr,
    temps_ptr,
    slots_ptr,
    num_ids_ptr,
    block_max_ptr,
    cand_ptr,
    K,
    W,
    V,
    NUM_CHUNKS,
    NUM_BLOCKS,
    HAS_IDS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCKS_PER_CHUNK: tl.constexpr,
    BLOCKS_POW2: tl.constexpr,
    KP: tl.constexpr,
):
    """The KP largest keys of one chunk of a row, unordered.

    A key orders by value, then by lower index: the fp32 bits mapped to an
    order-preserving int32 in the high word, the inverted index in the low.
    Keys are distinct, so the top-k under them is unique.
    """
    j = tl.program_id(0)
    pos = tl.program_id(1)
    chunk = tl.program_id(2)
    if _lp_skip_topk(keep_ptr, valid_ptr, slots_ptr, num_ids_ptr, j, pos, HAS_IDS):
        return
    r = (j * W + pos).to(tl.int64)
    # K block maxima are K distinct elements at or above the K-th largest
    # block max, so every top-k element is too: the rest of the row is
    # skipped without a reduction over it.
    b = tl.arange(0, BLOCKS_POW2)
    block_max = tl.load(
        block_max_ptr + r * NUM_BLOCKS + b, mask=b < NUM_BLOCKS, other=float("-inf")
    )
    sorted_max = tl.sort(_lp_key32(block_max), descending=True)
    kth = tl.max(tl.where(b == K - 1, sorted_max, -0x7FFFFFFF - 1), 0)
    threshold = kth.to(tl.int64) << 32
    # Distinct placeholders below every real key.
    best = _NO_KEY + 1 + tl.arange(0, KP).to(tl.int64)
    best_min = tl.min(best, 0)
    for blk in range(BLOCKS_PER_CHUNK):
        offs = (chunk * BLOCKS_PER_CHUNK + blk) * BLOCK + tl.arange(0, BLOCK)
        x = _lp_load(
            logits_ptr,
            logits_stride,
            src_ptr,
            temps_ptr,
            j,
            pos,
            W,
            offs,
            V,
            float("-inf"),
        )
        key = (_lp_key32(x).to(tl.int64) << 32) | (0x7FFFFFFF - offs).to(tl.int64)
        key = tl.where((offs < V) & (key >= threshold), key, _NO_KEY)
        key_max = tl.max(key, 0)
        while key_max > best_min:
            best = tl.where(best == best_min, key_max, best)
            best_min = tl.min(best, 0)
            key = tl.where(key == key_max, _NO_KEY, key)
            key_max = tl.max(key, 0)
    out = (r * NUM_CHUNKS + chunk) * KP + tl.arange(0, KP)
    tl.store(cand_ptr + out, best)


@triton.jit
def _lp_topk_merge_kernel(
    keep_ptr,
    valid_ptr,
    slots_ptr,
    num_ids_ptr,
    cand_ptr,
    topk_ptr,
    K,
    W,
    NUM_CAND,
    HAS_IDS: tl.constexpr,
    CAND_POW2: tl.constexpr,
):
    j = tl.program_id(0)
    pos = tl.program_id(1)
    if _lp_skip_topk(keep_ptr, valid_ptr, slots_ptr, num_ids_ptr, j, pos, HAS_IDS):
        return
    r = (j * W + pos).to(tl.int64)
    c = tl.arange(0, CAND_POW2)
    keys = tl.load(cand_ptr + r * NUM_CAND + c, mask=c < NUM_CAND, other=_NO_KEY)
    keys = tl.sort(keys, descending=True)
    ids = 0x7FFFFFFF - (keys & 0x7FFFFFFF)
    tl.store(topk_ptr + r * K + c, ids.to(tl.int32), mask=c < K)


@triton.jit
def _lp_write_kernel(
    logits_ptr,
    logits_stride,
    src_ptr,
    slots_ptr,
    keep_ptr,
    valid_ptr,
    temps_ptr,
    chunk_max_ptr,
    chunk_arg_ptr,
    block_sum_ptr,
    block_cnt_ptr,
    topk_ptr,
    num_ids_ptr,
    ids_ptr,
    ids_stride,
    out_ids_ptr,
    out_lp_ptr,
    out_rank_ptr,
    num_topk,
    W,
    CL,
    V,
    STASH_W,
    NUM_CHUNKS,
    NUM_BLOCKS,
    HAS_IDS: tl.constexpr,
    LOGITS_MODE: tl.constexpr,
    CHUNKS_POW2: tl.constexpr,
    BLOCKS_POW2: tl.constexpr,
    SUM_GROUP: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    j = tl.program_id(0)
    pos = tl.program_id(1)
    if _lp_skip(keep_ptr, valid_ptr, j, pos):
        return
    r = (j * W + pos).to(tl.int64)
    c = tl.arange(0, CHUNKS_POW2)
    in_chunks = c < NUM_CHUNKS
    maxes = tl.load(
        chunk_max_ptr + r * NUM_CHUNKS + c, mask=in_chunks, other=float("-inf")
    )
    max_val = tl.max(maxes, 0)
    # The first chunk holding the max holds the first-index argmax.
    first = tl.min(tl.where(maxes == max_val, c, CHUNKS_POW2), 0)
    best_idx = tl.load(chunk_arg_ptr + r * NUM_CHUNKS + first).to(tl.int64)
    blocks = tl.arange(0, BLOCKS_POW2)
    rank = tl.sum(
        tl.load(block_cnt_ptr + r * NUM_BLOCKS + blocks, mask=blocks < NUM_BLOCKS)
    )
    # Add the block sums one by one in block order. A group is loaded at once
    # and each sum picked out by a masked sum, which adds only zeros to it, so
    # the loop does not wait on a load per block.
    lanes = tl.arange(0, SUM_GROUP)
    se = 0.0
    for g in range(0, NUM_BLOCKS, SUM_GROUP):
        sums = tl.load(
            block_sum_ptr + r * NUM_BLOCKS + g + lanes,
            mask=g + lanes < NUM_BLOCKS,
            other=0.0,
        )
        for i in tl.static_range(SUM_GROUP):
            se += tl.sum(tl.where(lanes == i, sums, 0.0))
    lse = tl.log(se)

    row = (tl.load(src_ptr + j) * W + pos).to(tl.int64)
    base = logits_ptr + row * logits_stride
    temp = tl.load(temps_ptr + j)
    slot = tl.load(slots_ptr + j).to(tl.int64)
    out = (slot * CL + pos) * STASH_W
    tl.store(out_rank_ptr + slot * CL + pos, rank)
    num_ids = 0
    if HAS_IDS:
        num_ids = tl.load(num_ids_ptr + slot)
    for c0 in range(0, STASH_W, COL_BLOCK):
        col = c0 + tl.arange(0, COL_BLOCK)
        k = col - 1
        # Column 0 is the argmax; then the request's own token ids, or else
        # the top-k. The rest pads with token 0 at -inf.
        in_topk = (k >= 0) & (k < num_topk) & (num_ids == 0)
        tok = tl.load(topk_ptr + r * num_topk + k, mask=in_topk, other=0)
        tok = tok.to(tl.int64)
        valid = (col == 0) | in_topk
        if HAS_IDS:
            in_ids = (k >= 0) & (k < num_ids)
            from_ids = tl.load(ids_ptr + slot * ids_stride + k, mask=in_ids, other=0)
            tok = tl.where(in_ids, from_ids.to(tl.int64), tok)
            valid = valid | in_ids
        tok = tl.where(col == 0, best_idx, tok)
        x = tl.load(base + tok, mask=valid, other=0.0).to(tl.float32)
        x = tl.div_rn(x, temp)
        if not LOGITS_MODE:
            x = x - max_val - lse
        in_stash = col < STASH_W
        tl.store(out_ids_ptr + out + col, tl.where(valid, tok, 0), mask=in_stash)
        tl.store(
            out_lp_ptr + out + col, tl.where(valid, x, float("-inf")), mask=in_stash
        )


def stash_converged_logprobs(
    logits: torch.Tensor,
    canvas_len: int,
    src: torch.Tensor,
    slots: torch.Tensor,
    keep: torch.Tensor,
    valid: torch.Tensor,
    temps: torch.Tensor,
    num_topk: int,
    num_token_ids: torch.Tensor,
    token_ids: torch.Tensor | None,
    out_ids: torch.Tensor,
    out_logprobs: torch.Tensor,
    out_ranks: torch.Tensor,
    logits_mode: bool,
) -> None:
    """Write the logprobs of the converged requests' rows to the stash.

    Request `j` owns logits rows `src[j] * canvas_len + p` for
    `p < valid[j]`, divided by `temps[j]`. Where `keep[j]`, they go to
    `out_*[slots[j], p]`, laid out as compute_topk_scores lays out a row:
    the argmax, then the request's `token_ids` if it has any, else the
    `num_topk` largest (tied ids by lower id), padded with token 0 at -inf.
    Rows not kept exit at once; the others are split into chunks so a few
    kept rows still fill the GPU.
    """
    m = slots.shape[0]
    vocab = logits.shape[-1]
    assert logits.stride(-1) == 1
    assert temps.dtype == torch.float32
    assert num_topk + 1 <= out_logprobs.shape[-1]
    block = _LP_BLOCK
    num_blocks = triton.cdiv(vocab, block)
    blocks_per_chunk = min(_LP_BLOCKS_PER_CHUNK, num_blocks)
    num_chunks = triton.cdiv(num_blocks, blocks_per_chunk)
    rows = m * canvas_len
    device = logits.device
    chunk_max = torch.empty(rows, num_chunks, dtype=torch.float32, device=device)
    chunk_arg = torch.empty(rows, num_chunks, dtype=torch.int64, device=device)
    block_sum = torch.empty(rows, num_blocks, dtype=torch.float32, device=device)
    block_cnt = torch.empty(rows, num_blocks, dtype=torch.int32, device=device)
    block_max = (
        torch.empty(rows, num_blocks, dtype=torch.float32, device=device)
        if num_topk > 0
        else chunk_max
    )
    has_ids = token_ids is not None
    ids_arg = token_ids if token_ids is not None else num_token_ids
    common = (logits, logits.stride(0), src, keep, valid, temps)
    grid = (m, canvas_len, num_chunks)
    _lp_max_kernel[grid](
        *common,
        chunk_max,
        chunk_arg,
        block_max,
        canvas_len,
        vocab,
        num_chunks,
        num_blocks,
        BLOCK=block,
        BLOCKS_PER_CHUNK=blocks_per_chunk,
        STORE_BLOCK_MAX=num_topk > 0,
        num_warps=4,
    )
    chunks_pow2 = triton.next_power_of_2(num_chunks)
    _lp_sum_kernel[grid](
        *common,
        chunk_max,
        block_sum,
        block_cnt,
        canvas_len,
        vocab,
        num_chunks,
        num_blocks,
        BLOCK=block,
        BLOCKS_PER_CHUNK=blocks_per_chunk,
        CHUNKS_POW2=chunks_pow2,
        num_warps=4,
    )
    topk = chunk_arg
    if num_topk > 0:
        kp = triton.next_power_of_2(num_topk)
        cand = torch.empty(rows, num_chunks, kp, dtype=torch.int64, device=device)
        _lp_topk_kernel[grid](
            *common,
            slots,
            num_token_ids,
            block_max,
            cand,
            num_topk,
            canvas_len,
            vocab,
            num_chunks,
            num_blocks,
            HAS_IDS=has_ids,
            BLOCK=block,
            BLOCKS_PER_CHUNK=blocks_per_chunk,
            BLOCKS_POW2=triton.next_power_of_2(max(num_blocks, num_topk)),
            KP=kp,
            num_warps=4,
        )
        topk = torch.empty(rows, num_topk, dtype=torch.int32, device=device)
        _lp_topk_merge_kernel[(m, canvas_len)](
            keep,
            valid,
            slots,
            num_token_ids,
            cand,
            topk,
            num_topk,
            canvas_len,
            num_chunks * kp,
            HAS_IDS=has_ids,
            CAND_POW2=triton.next_power_of_2(num_chunks * kp),
        )
    stash_width = out_logprobs.shape[-1]
    _lp_write_kernel[(m, canvas_len)](
        logits,
        logits.stride(0),
        src,
        slots,
        keep,
        valid,
        temps,
        chunk_max,
        chunk_arg,
        block_sum,
        block_cnt,
        topk,
        num_token_ids,
        ids_arg,
        ids_arg.stride(0) if has_ids else 0,
        out_ids,
        out_logprobs,
        out_ranks,
        num_topk,
        canvas_len,
        out_logprobs.shape[1],
        vocab,
        stash_width,
        num_chunks,
        num_blocks,
        HAS_IDS=has_ids,
        LOGITS_MODE=logits_mode,
        CHUNKS_POW2=chunks_pow2,
        BLOCKS_POW2=triton.next_power_of_2(num_blocks),
        SUM_GROUP=16,
        COL_BLOCK=min(triton.next_power_of_2(stash_width), 1024),
    )
