# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k/top-p masking of FP32 logits on x86 CPU through Triton, with the same exact
contract as topk_topp_triton. tl.sort does not compile at vocab size on CPU, so group
maxima bound the best logits and a radix select cuts them instead."""

import functools
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.sample.ops.topk_topp_triton import (
    _at_least,
    _bounded_mass,
    _key,
    _ordered,
    _row_params,
)

BLOCK = 2048  # logits per streaming step of the levels and fill, a multiple of 256
SELECT_BLOCK = 128  # items per step of a radix select, times its 16 digit bins
GATHER = 2  # blocks of 256 logits scanned per step of the gather
CAND = 16384  # logits gathered per row at most
TOP_P_TIERS = (64, 512, 4096)
MAX_SPLITS = 32
BIN_ROUNDS = tl.constexpr(4)  # bins of 1 nat down to 1/4096 for a top-p cut
_MAX_VOCAB_SIZE = 1 << 20  # keys hold the index in 32 bits
INTS = ["has_k", "has_p", "splits", "chunk"]


def is_supported(logits: torch.Tensor) -> bool:
    """FP32 logits on x86 (ARM, PowerPC and s390x use the C++ kernels)."""
    vocab_ok = 1 < logits.shape[-1] < _MAX_VOCAB_SIZE
    return _x86() and logits.dtype == torch.float32 and vocab_ok


@functools.cache
def _x86() -> bool:
    return (
        HAS_TRITON
        and current_platform.is_cpu()
        and current_platform.get_cpu_architecture() == CpuArchEnum.X86
    )


def apply_top_k_top_p_triton_cpu(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Mask logits outside the per-row top-k, then outside the top-p nucleus, in
    place. k >= vocab_size or p >= 1 disables it for the row."""
    assert logits.dtype == torch.float32 and logits.ndim == 2
    if (k is None and p is None) or logits.shape[0] == 0:
        return logits
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    rows, vocab = logits.shape
    has_k, has_p = k is not None, p is not None
    k = _EMPTY_I32 if k is None else k.to(torch.int32).contiguous()
    p = _EMPTY_F32 if p is None else p.to(torch.float32).contiguous()
    levels, select, fill, fused = _kernel_args(logits, k, p, has_k, has_p)
    splits = levels["splits"]
    # One split per row: a program per row runs all three phases while its row is
    # still in cache, instead of reading the row again from memory.
    if splits == 1:
        _fused_kernel[(rows,)](**fused)
    else:
        _levels_kernel[(rows, splits)](**levels)
        _select_kernel[(rows,)](**select)
        _fill_kernel[(rows, splits)](**fill)
    return logits


def register_warmups() -> None:
    if _x86():
        _levels_warmup.register_warmup()
        _select_warmup.register_warmup()
        _fill_warmup.register_warmup()
        _fused_warmup.register_warmup()


_EMPTY_I32 = torch.empty(0, dtype=torch.int32)
_EMPTY_F32 = torch.empty(0, dtype=torch.float32)
_BUFFERS: dict[str, torch.Tensor] = {}


def reset_buffer_cache() -> None:
    _BUFFERS.clear()


def _buffers(rows: int, vocab: int) -> dict[str, torch.Tensor]:
    """Grow-only workspace, shared by all calls."""
    sizes = dict(
        gmax=(rows * triton.cdiv(vocab, 16), torch.float32),
        smax=(rows * triton.cdiv(vocab, 256), torch.float32),
        stats=(rows * MAX_SPLITS * 2, torch.float64),
        counts=(rows * MAX_SPLITS, torch.int32),
        cand=(rows * CAND, torch.float32),
        cand_idx=(rows * CAND, torch.int32),
        cut=(rows * 2, torch.int64),
    )
    for name, (size, dtype) in sizes.items():
        if name not in _BUFFERS or _BUFFERS[name].numel() < size:
            _BUFFERS[name] = torch.empty(size, dtype=dtype)
    return _BUFFERS


@functools.cache
def _split(rows: int, vocab: int, threads: int) -> tuple[int, int]:
    """Splits per row that fill the threads, in chunks of whole BLOCKs."""
    splits = max(1, min(MAX_SPLITS, triton.cdiv(threads, rows)))
    chunk = triton.cdiv(triton.cdiv(vocab, splits), BLOCK) * BLOCK
    return triton.cdiv(vocab, chunk), chunk


def _kernel_args(logits, k, p, has_k, has_p) -> tuple[dict, dict, dict, dict]:
    rows, vocab = logits.shape
    splits, chunk = _split(rows, vocab, torch.get_num_threads())
    b = _buffers(rows, vocab)
    row = dict(
        logits_ptr=logits, k_ptr=k, p_ptr=p, stride=logits.stride(0), vocab=vocab
    )
    levels_ = dict(gmax_ptr=b["gmax"], smax_ptr=b["smax"])
    stats = dict(stats_ptr=b["stats"], counts_ptr=b["counts"], splits=splits)
    levels = dict(
        **row,
        **levels_,
        **stats,
        has_p=has_p,
        chunk=chunk,
        MAX_SPLITS=MAX_SPLITS,
        BLOCK=BLOCK,
    )
    select = dict(
        **row,
        **levels_,
        **stats,
        has_k=has_k,
        has_p=has_p,
        cand_ptr=b["cand"],
        cand_idx_ptr=b["cand_idx"],
        cut_ptr=b["cut"],
        MAX_SPLITS=MAX_SPLITS,
        CAND=CAND,
        GATHER=GATHER,
        SELECT_BLOCK=SELECT_BLOCK,
        P_SMALL=TOP_P_TIERS[0],
        P_MID=TOP_P_TIERS[1],
        P_LARGE=TOP_P_TIERS[2],
    )
    fill = dict(**row, cut_ptr=b["cut"], chunk=chunk, BLOCK=BLOCK)
    fused = dict(**select, chunk=chunk, BLOCK=BLOCK)
    return levels, select, fill, fused


@triton.jit(do_not_specialize=INTS)
def _levels_kernel(
    logits_ptr,
    k_ptr,
    p_ptr,
    stride,
    vocab,
    gmax_ptr,
    smax_ptr,
    stats_ptr,
    counts_ptr,
    splits,
    has_p,
    chunk,
    MAX_SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _levels(
        tl.program_id(0), tl.program_id(1), logits_ptr, stride, vocab, gmax_ptr,
        smax_ptr, stats_ptr, counts_ptr, has_p, chunk, MAX_SPLITS, BLOCK,
    )  # fmt: skip


@triton.jit(do_not_specialize=INTS)
def _select_kernel(
    logits_ptr,
    k_ptr,
    p_ptr,
    stride,
    vocab,
    has_k,
    has_p,
    gmax_ptr,
    smax_ptr,
    stats_ptr,
    counts_ptr,
    cand_ptr,
    cand_idx_ptr,
    cut_ptr,
    splits,
    MAX_SPLITS: tl.constexpr,
    CAND: tl.constexpr,
    GATHER: tl.constexpr,
    SELECT_BLOCK: tl.constexpr,
    P_SMALL: tl.constexpr,
    P_MID: tl.constexpr,
    P_LARGE: tl.constexpr,
):
    _select(
        tl.program_id(0), logits_ptr, k_ptr, p_ptr, stride, vocab, has_k, has_p,
        gmax_ptr, smax_ptr, stats_ptr, counts_ptr, cand_ptr, cand_idx_ptr, cut_ptr,
        splits, MAX_SPLITS, CAND, GATHER, SELECT_BLOCK, P_SMALL, P_MID, P_LARGE,
    )  # fmt: skip


@triton.jit(do_not_specialize=INTS)
def _fill_kernel(
    logits_ptr, k_ptr, p_ptr, stride, vocab, cut_ptr, chunk, BLOCK: tl.constexpr
):
    _fill(
        tl.program_id(0), tl.program_id(1), logits_ptr, stride, vocab, cut_ptr,
        chunk, BLOCK,
    )  # fmt: skip


@triton.jit(do_not_specialize=INTS)
def _fused_kernel(
    logits_ptr,
    k_ptr,
    p_ptr,
    stride,
    vocab,
    has_k,
    has_p,
    gmax_ptr,
    smax_ptr,
    stats_ptr,
    counts_ptr,
    cand_ptr,
    cand_idx_ptr,
    cut_ptr,
    splits,
    chunk,
    MAX_SPLITS: tl.constexpr,
    CAND: tl.constexpr,
    GATHER: tl.constexpr,
    SELECT_BLOCK: tl.constexpr,
    P_SMALL: tl.constexpr,
    P_MID: tl.constexpr,
    P_LARGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    _levels(
        row, 0, logits_ptr, stride, vocab, gmax_ptr, smax_ptr, stats_ptr,
        counts_ptr, has_p, chunk, MAX_SPLITS, BLOCK,
    )  # fmt: skip
    _select(
        row, logits_ptr, k_ptr, p_ptr, stride, vocab, has_k, has_p, gmax_ptr,
        smax_ptr, stats_ptr, counts_ptr, cand_ptr, cand_idx_ptr, cut_ptr, splits,
        MAX_SPLITS, CAND, GATHER, SELECT_BLOCK, P_SMALL, P_MID, P_LARGE,
    )  # fmt: skip
    _fill(row, 0, logits_ptr, stride, vocab, cut_ptr, chunk, BLOCK)


@triton.jit
def _levels(
    row,
    split,
    logits_ptr,
    stride,
    vocab,
    gmax_ptr,
    smax_ptr,
    stats_ptr,
    counts_ptr,
    has_p,
    chunk,
    MAX_SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_ptr = logits_ptr + row.to(tl.int64) * stride
    gmax_row = gmax_ptr + row.to(tl.int64) * tl.cdiv(vocab, 16)
    smax_row = smax_ptr + row.to(tl.int64) * tl.cdiv(vocab, 256)
    offs = tl.arange(0, BLOCK)
    group = tl.arange(0, BLOCK // 16)
    supergroup = tl.arange(0, BLOCK // 256)
    lo = split * chunk
    hi = tl.minimum(lo + chunk, vocab)
    row_max = tl.full([], -float("inf"), tl.float32)
    total = tl.full([], 0.0, tl.float64)
    finite = 0
    for start in range(lo, hi, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < hi, other=-float("inf"))
        # NaN is never kept, so it counts as -inf everywhere.
        x = tl.where(x == x, x, -float("inf"))
        gmax = tl.max(tl.reshape(x, [BLOCK // 16, 16]), 1)
        gi = start // 16 + group
        tl.store(gmax_row + gi, gmax, mask=gi * 16 < hi)
        smax = tl.max(tl.reshape(gmax, [BLOCK // 256, 16]), 1)
        si = start // 256 + supergroup
        tl.store(smax_row + si, smax, mask=si * 256 < hi)
        new_max = tl.maximum(row_max, tl.max(smax))
        # Only top-p reads the mass.
        if has_p & (new_max > -float("inf")):
            scale = tl.exp((row_max - new_max).to(tl.float64))
            # FP32 block sums lose a tail of ~1e-8 weights next to the max.
            total = total * scale + tl.sum(tl.exp(x - new_max).to(tl.float64))
        row_max = new_max
        finite += tl.sum((x > -float("inf")).to(tl.int32))
    slot = row * MAX_SPLITS + split
    tl.store(stats_ptr + 2 * slot, row_max.to(tl.float64))
    tl.store(stats_ptr + 2 * slot + 1, total)
    tl.store(counts_ptr + slot, finite)


@triton.jit
def _select(
    row,
    logits_ptr,
    k_ptr,
    p_ptr,
    stride,
    vocab,
    has_k,
    has_p,
    gmax_ptr,
    smax_ptr,
    stats_ptr,
    counts_ptr,
    cand_ptr,
    cand_idx_ptr,
    cut_ptr,
    splits,
    MAX_SPLITS: tl.constexpr,
    CAND: tl.constexpr,
    GATHER: tl.constexpr,
    SELECT_BLOCK: tl.constexpr,
    P_SMALL: tl.constexpr,
    P_MID: tl.constexpr,
    P_LARGE: tl.constexpr,
):
    row_ptr = logits_ptr + row.to(tl.int64) * stride
    groups = tl.cdiv(vocab, 16)
    supergroups = tl.cdiv(vocab, 256)
    gmax_row = gmax_ptr + row.to(tl.int64) * groups
    smax_row = smax_ptr + row.to(tl.int64) * supergroups
    cand_row = cand_ptr + row.to(tl.int64) * CAND
    cand_idx_row = cand_idx_ptr + row.to(tl.int64) * CAND
    split = tl.arange(0, MAX_SPLITS)
    slot = row * MAX_SPLITS + split
    in_row = split < splits
    split_max = tl.load(stats_ptr + 2 * slot, mask=in_row, other=-float("inf"))
    split_sum = tl.load(stats_ptr + 2 * slot + 1, mask=in_row, other=0.0)
    row_max = tl.max(split_max)
    weight = tl.exp(split_max - row_max)
    total = tl.sum(tl.where(split_max > -float("inf"), split_sum * weight, 0.0))
    finite = tl.sum(tl.load(counts_ptr + slot, mask=in_row, other=0))
    k, p = _row_params(k_ptr, p_ptr, row, vocab, has_k, has_p)
    use_k = k < finite
    use_p = p < 1.0
    cut_hi = row_max.to(tl.int64) * 0 - 1  # -1: the row is left as is
    cut_idx = cut_hi * 0
    if (finite > 0) & (use_k | use_p):
        row_max32 = row_max.to(tl.float32)
        n = tl.where(use_k, k, P_SMALL)
        done = 0
        for tier in tl.static_range(3):
            if tier == 1:
                n = n * 0 + P_MID
            if tier == 2:
                n = n * 0 + P_LARGE
            # A logit weighs at most the row max's 1, so n logits short of the
            # nucleus mass cannot hold it.
            fits = use_k | (n.to(tl.float64) >= p.to(tl.float64) * total)
            if (done == 0) & ((tier == 0) | (use_k == 0)) & fits:
                # The coarser level is enough while it has a few times n maxima.
                if n * 4 <= supergroups:
                    t = _nth_largest(smax_row, supergroups, n, SELECT_BLOCK)
                else:
                    t = _nth_largest(gmax_row, groups, n, SELECT_BLOCK)
                # At most all finite logits: then the gather holds every candidate.
                lowest = _ordered(tl.full([], -3.4028234663852886e38, tl.float32))
                all_finite = t <= lowest
                t = tl.maximum(t, lowest)
                ncand = _gather(
                    row_ptr, vocab, smax_row, supergroups, t, cand_row,
                    cand_idx_row, CAND, GATHER,
                )  # fmt: skip
                if ncand > 0:
                    bound_hi, bound_lo = _cut(
                        cand_row, ncand, k, p, use_k, use_p, total, row_max32,
                        SELECT_BLOCK,
                    )  # fmt: skip
                    kept, last = _last_kept(
                        cand_row, ncand, bound_hi, bound_lo, SELECT_BLOCK
                    )
                    # Top-p alone: the gather holds the n best logits, so a nucleus
                    # of at most n is exact if it is short of the whole gather
                    # (which an unreachable target also keeps).
                    if use_k | ((kept <= n) & (kept < ncand)) | all_finite:
                        cut_hi = 0x7FFFFFFF - (last >> 32)
                        last_pos = last & 0xFFFFFFFF
                        cut_idx = tl.load(cand_idx_row + last_pos).to(tl.int64)
                        done = 1
        if (done == 0) & (use_k == 0):
            ncand, rest = _bin_gather(
                row_ptr, vocab, p.to(tl.float64) * total, row_max32, cand_row,
                cand_idx_row, CAND, SELECT_BLOCK,
            )  # fmt: skip
            if ncand > 0:
                zero = ncand.to(tl.int64) * 0
                bound_hi, bound_lo = _radix_select(
                    cand_row, ncand, zero, zero, rest, row_max32, True, 16,
                    SELECT_BLOCK,
                )  # fmt: skip
                _, last = _last_kept(cand_row, ncand, bound_hi, bound_lo, SELECT_BLOCK)
                cut_hi = 0x7FFFFFFF - (last >> 32)
                cut_idx = tl.load(cand_idx_row + (last & 0xFFFFFFFF)).to(tl.int64)
                done = 1
        if done == 0:
            bound_hi, bound_lo = _cut(
                row_ptr, vocab, k, p, use_k, use_p, total, row_max32, SELECT_BLOCK
            )
            cut_hi = bound_hi
            cut_idx = 0xFFFFFFFF - bound_lo
    tl.store(cut_ptr + 2 * row, cut_hi)
    tl.store(cut_ptr + 2 * row + 1, cut_idx)


@triton.jit
def _nth_largest(ptr, count, n, BLOCK: tl.constexpr):
    """An ordered value t such that ptr[:count] has the same items at or above t as
    at or above its n-th largest (all of them if n > count). Values only: 8 digits."""
    zero = count.to(tl.int64) * 0
    need = tl.minimum(n, count).to(tl.float64)
    bound_hi, _ = _radix_select(ptr, count, zero, zero, need, 0.0, False, 8, BLOCK)
    return bound_hi


@triton.jit
def _cut(ptr, n, k, p, use_k, use_p, total, row_max, BLOCK: tl.constexpr):
    """The (ordered logit, ~position) bound of the kept items of ptr[:n]: top-k,
    then top-p of the top-k mass (of total if no top-k)."""
    bound_hi = n.to(tl.int64) * 0
    bound_lo = bound_hi
    if use_k:
        bound_hi, bound_lo = _radix_select(
            ptr, n, bound_hi, bound_lo, k.to(tl.float64), 0.0, False, 16, BLOCK
        )
    if use_p:
        mass = total
        if use_k:
            mass = _bounded_mass(ptr, n, bound_hi, bound_lo, row_max, BLOCK)
        bound_hi, bound_lo = _radix_select(
            ptr, n, bound_hi, bound_lo, p.to(tl.float64) * mass, row_max, True, 16,
            BLOCK,
        )  # fmt: skip
    return bound_hi, bound_lo


@triton.jit
def _gather(
    row_ptr,
    vocab,
    smax_row,
    supergroups,
    t,
    cand_row,
    cand_idx_row,
    CAND: tl.constexpr,
    GATHER: tl.constexpr,
):
    """Copy the logits at or above t in index order, so a position order is an
    index order. Returns how many, or -1 past CAND."""
    block = tl.arange(0, GATHER)
    lane = tl.arange(0, 256)
    taken = 0
    for start in range(0, supergroups, GATHER):
        si = start + block
        smax = tl.load(smax_row + si, mask=si < supergroups, other=-float("inf"))
        hit = (si < supergroups) & (_ordered(smax) >= t)
        if (tl.sum(hit.to(tl.int32)) > 0) & (taken <= CAND):
            idx = si[:, None] * 256 + lane[None, :]
            x = tl.load(
                row_ptr + idx, mask=hit[:, None] & (idx < vocab), other=-float("inf")
            )
            x = tl.where(x == x, x, -float("inf"))
            sel = tl.reshape(hit[:, None] & (_ordered(x) >= t), [GATHER * 256])
            count = tl.sum(sel.to(tl.int32))
            if taken + count <= CAND:
                dst = taken + tl.cumsum(sel.to(tl.int32), 0) - 1
                tl.store(cand_row + dst, tl.reshape(x, [GATHER * 256]), mask=sel)
                tl.store(cand_idx_row + dst, tl.reshape(idx, [GATHER * 256]), mask=sel)
            taken += count
    return tl.where(taken <= CAND, taken, -1)


@triton.jit
def _bin_gather(
    row_ptr,
    vocab,
    target,
    row_max,
    cand_row,
    cand_idx_row,
    CAND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather the bin of the top-p cut in index order: how many (0 if none fits) and
    the mass still needed there. Bins are base-16 digits of row_max - x from 1 nat
    down; scaling by 16 is exact in FP32, so each round splits the last bin exactly."""
    offs = tl.arange(0, BLOCK)
    bins = tl.arange(0, 16)
    prefix = row_max * 0.0
    scale = row_max * 0.0 + 1.0
    rest = target
    count = vocab * 0 + CAND + 1
    rounds = vocab * 0
    while (rounds < BIN_ROUNDS) & (count > CAND):
        counts = tl.zeros([16], tl.int32)
        masses = tl.zeros([16], tl.float64)
        for start in range(0, vocab, BLOCK):
            idx = start + offs
            x = tl.load(row_ptr + idx, mask=idx < vocab, other=-float("inf"))
            u = row_max - tl.where(x == x, x, -float("inf"))
            sel = (u < 16.0) & (tl.floor(u * scale / 16.0) == prefix)
            digit = tl.where(sel, tl.floor(u * scale) - 16.0 * prefix, -1.0)
            onehot = digit[:, None] == bins[None, :].to(tl.float32)
            counts += tl.sum(onehot.to(tl.int32), 0)
            weight = tl.exp(-u).to(tl.float64)
            masses += tl.sum(tl.where(onehot, weight[:, None], 0.0), 0)
        reach = tl.cumsum(masses, 0)
        chosen = tl.min(tl.where(reach >= rest, bins, 16))
        if chosen < 16:
            rest -= tl.sum(tl.where(bins < chosen, masses, 0.0))
            count = tl.sum(tl.where(bins == chosen, counts, 0))
            prefix = prefix * 16.0 + chosen.to(tl.float32)
            scale = scale * 16.0
            rounds += 1
        else:
            rounds = rounds * 0 + BIN_ROUNDS
    taken = 0
    if count <= CAND:
        for start in range(0, vocab, BLOCK):
            idx = start + offs
            x = tl.load(row_ptr + idx, mask=idx < vocab, other=-float("inf"))
            u = row_max - tl.where(x == x, x, -float("inf"))
            sel = (u < 16.0) & (tl.floor(u * scale / 16.0) == prefix)
            dst = taken + tl.cumsum(sel.to(tl.int32), 0) - 1
            tl.store(cand_row + dst, x, mask=sel)
            tl.store(cand_idx_row + dst, idx, mask=sel)
            taken += tl.sum(sel.to(tl.int32))
    return taken, rest


@triton.jit
def _radix_select(
    row_ptr,
    vocab,
    bound_hi,
    bound_lo,
    target,
    row_max,
    MASS: tl.constexpr,
    DIGITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The largest key whose keys at or above it, within the bound, reach target
    elements (or mass). topk_topp_triton's, but with FP64 mass sums (in FP32 a
    nucleus far below the max misses its 1e-7 bound); DIGITS=8 selects on values."""
    bins = tl.arange(0, 16)
    prefix_hi = bound_hi * 0
    prefix_lo = bound_hi * 0
    out_hi = bound_hi
    out_lo = bound_lo
    remaining = target
    digit = 0
    done = 0
    while (digit < DIGITS) & (done == 0):
        counts, masses = _digit_histogram(
            row_ptr,
            vocab,
            bound_hi,
            bound_lo,
            prefix_hi,
            prefix_lo,
            digit,
            row_max,
            MASS,
            BLOCK,
        )
        hist = masses if MASS else counts
        from_top = tl.sum(hist) - tl.cumsum(hist, 0) + hist
        chosen = tl.maximum(tl.max(tl.where(from_top >= remaining, bins, -1)), 0)
        remaining -= tl.sum(tl.where(bins > chosen, hist, 0.0))
        in_bucket = tl.sum(tl.where(bins == chosen, counts, 0.0))
        if digit < 8:
            prefix_hi = prefix_hi * 16 + chosen
        else:
            prefix_lo = prefix_lo * 16 + chosen
        stop = (in_bucket <= 1) | (digit == DIGITS - 1)
        if not MASS:
            stop = stop | (in_bucket == remaining)
        if stop:
            done = 1
            pad_hi = tl.where(
                digit < 8, prefix_hi << (28 - 4 * tl.minimum(digit, 7)), prefix_hi
            )
            pad_lo = tl.where(
                digit < 8, 0, prefix_lo << (28 - 4 * tl.maximum(digit - 8, 0))
            )
            above = (pad_hi > bound_hi) | ((pad_hi == bound_hi) & (pad_lo > bound_lo))
            out_hi = tl.where(above, pad_hi, bound_hi)
            out_lo = tl.where(above, pad_lo, bound_lo)
        digit += 1
    return out_hi, out_lo


@triton.jit
def _digit_histogram(
    row_ptr,
    vocab,
    bound_hi,
    bound_lo,
    prefix_hi,
    prefix_lo,
    digit,
    row_max,
    MASS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    bins = tl.arange(0, 16)
    in_hi = digit < 8
    match_hi_shift = tl.where(in_hi, 32 - 4 * digit, 0)
    match_lo_shift = tl.where(in_hi, 32, 32 - 4 * (digit - 8))
    digit_shift = tl.where(
        in_hi, 28 - 4 * tl.minimum(digit, 7), 28 - 4 * tl.maximum(digit - 8, 0)
    )
    counts = tl.zeros([16], tl.float64)
    masses = tl.zeros([16], tl.float64)
    for start in range(0, vocab, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < vocab, other=-float("inf"))
        hi = _ordered(x)
        lo = 0xFFFFFFFF - idx.to(tl.int64)
        match = ((hi >> match_hi_shift) == prefix_hi) & (
            (lo >> match_lo_shift) == prefix_lo
        )
        sel = (idx < vocab) & match & _at_least(hi, lo, bound_hi, bound_lo)
        value = tl.where(in_hi, hi, lo)
        onehot = (((value >> digit_shift) & 15)[:, None] == bins[None, :]) & sel[
            :, None
        ]
        counts += tl.sum(onehot.to(tl.float32), 0).to(tl.float64)
        if MASS:
            weight = tl.exp(x - row_max).to(tl.float64)
            masses += tl.sum(tl.where(onehot, weight[:, None], 0.0), 0)
    return counts, masses


@triton.jit
def _last_kept(ptr, n, bound_hi, bound_lo, BLOCK: tl.constexpr):
    """How many of ptr[:n] the bound keeps, and the key of the last one."""
    offs = tl.arange(0, BLOCK)
    kept = 0
    last = bound_hi * 0 - 0x7FFFFFFFFFFFFFFF
    for start in range(0, n, BLOCK):
        pos = start + offs
        x = tl.load(ptr + pos, mask=pos < n, other=-float("inf"))
        keep = (pos < n) & _at_least(
            _ordered(x), 0xFFFFFFFF - pos.to(tl.int64), bound_hi, bound_lo
        )
        kept += tl.sum(keep.to(tl.int32))
        last = tl.maximum(last, tl.max(tl.where(keep, _key(x, pos), last)))
    return kept, last


@triton.jit
def _fill(row, split, logits_ptr, stride, vocab, cut_ptr, chunk, BLOCK: tl.constexpr):
    cut_hi = tl.load(cut_ptr + 2 * row)
    cut_idx = tl.load(cut_ptr + 2 * row + 1)
    if cut_hi >= 0:
        row_ptr = logits_ptr + row.to(tl.int64) * stride
        offs = tl.arange(0, BLOCK)
        lo = split * chunk
        hi = tl.minimum(lo + chunk, vocab)
        for start in range(lo, hi, BLOCK):
            idx = start + offs
            x = tl.load(row_ptr + idx, mask=idx < hi, other=-float("inf"))
            x = tl.where(x == x, x, -float("inf"))
            ordered = _ordered(x)
            keep = (ordered > cut_hi) | ((ordered == cut_hi) & (idx <= cut_idx))
            tl.store(row_ptr + idx, -float("inf"), mask=(idx < hi) & ~keep)


_KERNELS = (_levels_kernel, _select_kernel, _fill_kernel, _fused_kernel)


# JIT warmup: compile the kernels at startup for the model's vocab (one variant each,
# every batch size and k/p mix share it).


def _warmup_inputs(vllm_config: Any) -> dict[str, Any]:
    vocab = vllm_config.model_config.get_vocab_size()
    return dict(
        logits=TritonWarmupTensor(torch.float32, shape=(1, vocab)),
        k=TritonWarmupTensor(torch.int32),
        p=TritonWarmupTensor(torch.float32),
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_levels_kernel, warmup_inputs=_warmup_inputs
)
def _levels_warmup(logits: Any, k: Any, p: Any) -> DispatchSpec:
    return (1,), _kernel_args(logits, k, p, True, True)[0]


@triton_kernel_dispatcher_with_warmup(
    kernel=_select_kernel, warmup_inputs=_warmup_inputs
)
def _select_warmup(logits: Any, k: Any, p: Any) -> DispatchSpec:
    return (1,), _kernel_args(logits, k, p, True, True)[1]


@triton_kernel_dispatcher_with_warmup(kernel=_fill_kernel, warmup_inputs=_warmup_inputs)
def _fill_warmup(logits: Any, k: Any, p: Any) -> DispatchSpec:
    return (1,), _kernel_args(logits, k, p, True, True)[2]


@triton_kernel_dispatcher_with_warmup(
    kernel=_fused_kernel, warmup_inputs=_warmup_inputs
)
def _fused_warmup(logits: Any, k: Any, p: Any) -> DispatchSpec:
    return (1,), _kernel_args(logits, k, p, True, True)[3]
