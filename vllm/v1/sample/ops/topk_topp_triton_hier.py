# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k/top-p masking of FP32 logits through group maxima, in two launches.

1. Levels (rows x splits): stream each row once and write the max of every group of
   16 logits (level 1) and of every 256 (level 2), the row max, and the sum of exp for
   top-p-only rows. No compaction, so it runs close to the streaming bandwidth.
2. One persistent cooperative kernel, with phases separated by grid barriers:
   a. Select (per row): exact selection through the levels. The supergroups at or
      above the n-th best supergroup max hold the best n groups, and the groups at or
      above the n-th best group max hold every logit at or above it. Tiers of 32, 64,
      128 and 256 logits (top-p only: 32 and 128), with tl.topk at each level.
   b. Only if a row missed the tiers (k > 256, a nucleus beyond the tiers, ties at a
      cut): an 8192-bin histogram of (row max - logit) in 1/512 nat bins, then per row
      an exact cut from the histogram: the bins that can hold the cut (counts for k,
      mass bounds for p) are gathered (at most 4096 logits) and sorted, or refined
      over their value range. An exact radix select is the last resort.
   c. Fill (rows x splits): -inf, then the kept logits (or every logit up to the cut).
Exactly k logits are kept, ties broken by the lowest index, then the smallest prefix
whose mass reaches p.
"""

import functools

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

BLOCK = 4096  # elements per streaming step, a multiple of 256
MAX_SPLITS = 32
HIST_SPLITS = 16  # splits per row of the histogram path (few rows, so many splits)
FILL_SPLITS = 8  # at least this many splits per row for the fill (balances cut rows)
P_TIERS = 2  # tiers for top-p-only rows: 1 (32 logits) or 2 (32, then 128)
BINS = 8192
BIN_SCALE = 512.0  # bins per nat below the row max (16 nats)
WINDOW = 4096  # logits gathered to resolve a histogram cut
KEPT = 256  # kept logits stored per row by the tiers, also the largest tier
WARPS = 8  # of the persistent kernel
# Keys hold the index in 32 bits, and the levels fit tl.topk for these vocabularies.
_MAX_VOCAB_SIZE = 1 << 20
NOOP_C = tl.constexpr(-1)
HIST_C = tl.constexpr(-2)  # row waits for the histogram path
CUT_C = tl.constexpr(-3)  # row keeps every logit whose key is at most cut[row]
SELF_C = tl.constexpr(-4)  # row already masked by the exact radix select
GATHER_C = tl.constexpr(-5)  # row's cut bins are gathered by every program


def can_use_top_k_top_p_triton_hier(logits: torch.Tensor) -> bool:
    return (
        envs.VLLM_USE_TRITON_HIER_TOPK_TOPP
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.shape[-1] < _MAX_VOCAB_SIZE
        and _supported_device()
    )


@functools.cache
def _supported_device() -> bool:
    """Tested on SM90 and SM100, and needs cooperative launches."""
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major >= 9


def apply_top_k_top_p_triton_hier(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Mask logits outside the per-row top-k, then outside the top-p nucleus.

    Args:
        logits: [batch_size, vocab_size] float32 CUDA tensor.
        k: [batch_size] top-k per row, or None. k >= vocab_size disables it.
        p: [batch_size] top-p per row, or None. p >= 1 disables it.

    Returns:
        The masked logits. They alias the input unless its rows were not
        contiguous.

    """
    if k is None and p is None:
        return logits
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    rows, vocab = logits.shape
    device = logits.device
    splits = num_splits(rows, device)
    chunk = split_chunk(vocab, splits)
    groups, supergroups = triton.cdiv(vocab, 16), triton.cdiv(vocab, 256)
    b = buffers(device, rows, vocab)
    k_arg = k if k is not None else b["state"]
    p_arg = p if p is not None else b["row_stats"]
    shape = dict(
        vocab=vocab,
        chunk=chunk,
        groups=groups,
        supergroups=supergroups,
        splits=splits,
        MAX_SPLITS=MAX_SPLITS,
    )
    params = dict(k_ptr=k_arg, p_ptr=p_arg, HAS_K=k is not None, HAS_P=p is not None)
    _levels_kernel[(rows, splits)](
        logits,
        logits.stride(0),
        b["level1"],
        b["level2"],
        b["stats"],
        **shape,
        **params,
        BLOCK=BLOCK,
        num_warps=4,
    )
    args = (
        logits,
        logits.stride(0),
        b["level1"],
        b["level2"],
        b["stats"],
        b["row_stats"],
        b["state"],
        b["cut"],
        b["kept"],
        b["hist"],
        b["scratch"],
        b["window"],
        b["found"],
        b["more"],
        b["sync"],
        rows,
    )
    fill_splits = max(splits, FILL_SPLITS)
    meta = dict(
        **shape,
        **params,
        hchunk=split_chunk(vocab, HIST_SPLITS),
        fchunk=split_chunk(vocab, fill_splits),
        fsplits=fill_splits,
        HSPLITS=HIST_SPLITS,
        P_TIERS=P_TIERS,
        SUPERGROUPS_P2=max(triton.next_power_of_2(supergroups), KEPT),
        BLOCK=BLOCK,
        BINS=BINS,
        BIN_SCALE=BIN_SCALE,
        WINDOW=WINDOW,
        KEPT=KEPT,
    )
    programs = min(capacity(device, args, meta), rows * splits)
    _filter_kernel[(programs,)](
        *args, **meta, num_warps=WARPS, launch_cooperative_grid=True
    )
    return logits


def split_chunk(vocab: int, splits: int) -> int:
    return triton.cdiv(triton.cdiv(vocab, splits), BLOCK) * BLOCK


def num_splits(rows: int, device: torch.device) -> int:
    """Split rows for streaming until the grid has about 4 programs per SM."""
    return min(
        MAX_SPLITS, triton.next_power_of_2(triton.cdiv(4 * num_sms(device), rows))
    )


@functools.cache
def num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


_CAPACITY: dict[tuple, int] = {}


def capacity(device: torch.device, args: tuple, meta: dict) -> int:
    """Programs of the persistent kernel that fit on the GPU at once, so that the
    grid barriers cannot deadlock (the cooperative launch fails otherwise)."""
    key = (
        device,
        *(v for v in meta.values() if isinstance(v, (bool, float))),
        meta["SUPERGROUPS_P2"],
    )
    if key not in _CAPACITY:
        kernel = _filter_kernel.warmup(
            *args, **meta, num_warps=WARPS, launch_cooperative_grid=True, grid=(1,)
        )
        kernel._init_handles()
        props = torch.cuda.get_device_properties(device)
        threads = 32 * WARPS
        regs = getattr(props, "regs_per_multiprocessor", 65536)
        smem = getattr(props, "shared_memory_per_multiprocessor", 228 * 1024)
        by_regs = regs // (max(kernel.n_regs, 1) * threads)
        by_smem = smem // (kernel.metadata.shared + 1024)
        by_threads = props.max_threads_per_multi_processor // threads
        _CAPACITY[key] = (
            max(1, min(by_regs, by_smem, by_threads)) * props.multi_processor_count
        )
    return _CAPACITY[key]


_BUFFERS: dict[tuple[torch.device, int], dict[str, torch.Tensor]] = {}
_SYNC: dict[torch.device, torch.Tensor] = {}


def buffers(device: torch.device, rows: int, vocab: int) -> dict[str, torch.Tensor]:
    """Per-row workspace for at least rows rows, grown to a power of 2."""
    key = (device, vocab)
    if key not in _BUFFERS or _BUFFERS[key]["state"].numel() < rows:
        _BUFFERS[key] = allocate(device, triton.next_power_of_2(rows), vocab)
    return _BUFFERS[key]


def allocate(device: torch.device, rows: int, vocab: int) -> dict[str, torch.Tensor]:
    """Level 1 and 2 maxima, (max, sum of exp) per split and per row, row states, cut
    keys, kept keys, histograms, per-row gather scratch, gather windows and counts, and
    the grid barrier (barrier count, exit count, histogram flag), which is zeroed once
    and reset by each call."""
    empty = functools.partial(torch.empty, device=device)
    if device not in _SYNC:
        _SYNC[device] = torch.zeros(4, dtype=torch.int32, device=device)
    return dict(
        level1=empty(rows * triton.cdiv(vocab, 16), dtype=torch.float32),
        level2=empty(rows * triton.cdiv(vocab, 256), dtype=torch.float32),
        stats=empty(rows * MAX_SPLITS * 2, dtype=torch.float64),
        row_stats=empty(rows * 2, dtype=torch.float64),
        state=empty(rows, dtype=torch.int32),
        cut=empty(rows, dtype=torch.int64),
        kept=empty(rows * KEPT, dtype=torch.int64),
        hist=empty(rows * BINS, dtype=torch.int32),
        scratch=empty(rows * WINDOW, dtype=torch.int64),
        window=empty(rows * 2, dtype=torch.int32),
        found=empty(rows, dtype=torch.int32),
        more=empty(rows, dtype=torch.float64),
        sync=_SYNC[device],
    )


def reset_buffer_cache() -> None:
    _BUFFERS.clear()


@triton.jit
def _levels_kernel(
    logits_ptr,
    stride,
    level1_ptr,
    level2_ptr,
    stats_ptr,
    vocab,
    chunk,
    groups,
    supergroups,
    splits,
    k_ptr,
    p_ptr,
    HAS_K: tl.constexpr,
    HAS_P: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    k, p = _row_params(k_ptr, p_ptr, row, vocab, HAS_K, HAS_P)
    if (k >= vocab) & (p >= 1.0):
        return
    row_ptr = logits_ptr + tl.cast(row, tl.int64) * stride
    level1_row = level1_ptr + tl.cast(row, tl.int64) * groups
    level2_row = level2_ptr + tl.cast(row, tl.int64) * supergroups
    with_sum = (p < 1.0) & (k >= vocab)
    offs = tl.arange(0, BLOCK)
    lo = split * chunk
    hi = tl.minimum(lo + chunk, vocab)
    row_max = tl.full([], -3.0e38, tl.float32)
    row_sum = (lo * 0).to(tl.float64)
    for start in range(lo, hi, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < vocab, other=-float("inf"))
        group_max = tl.max(tl.reshape(x, [BLOCK // 16, 16]), 1)
        group = start // 16 + tl.arange(0, BLOCK // 16)
        tl.store(level1_row + group, group_max, mask=group < groups)
        supergroup = start // 256 + tl.arange(0, BLOCK // 256)
        supergroup_max = tl.max(tl.reshape(group_max, [BLOCK // 256, 16]), 1)
        tl.store(level2_row + supergroup, supergroup_max, mask=supergroup < supergroups)
        new_max = tl.maximum(row_max, tl.max(supergroup_max))
        if with_sum:
            rescale = tl.exp(row_max - new_max).to(tl.float64)
            row_sum = row_sum * rescale + tl.sum(tl.exp(x - new_max)).to(tl.float64)
        row_max = new_max
    program = row * MAX_SPLITS + split
    tl.store(stats_ptr + 2 * program, row_max.to(tl.float64))
    tl.store(stats_ptr + 2 * program + 1, row_sum)


# rows is not specialized, so batch sizes share one compiled kernel.
@triton.jit(do_not_specialize=["rows"])
def _filter_kernel(
    logits_ptr,
    stride,
    level1_ptr,
    level2_ptr,
    stats_ptr,
    row_stats_ptr,
    state_ptr,
    cut_ptr,
    kept_ptr,
    hist_ptr,
    scratch_ptr,
    window_ptr,
    found_ptr,
    more_ptr,
    sync_ptr,
    rows,
    vocab,
    chunk,
    groups,
    supergroups,
    splits,
    k_ptr,
    p_ptr,
    hchunk,
    fchunk,
    fsplits,
    HAS_K: tl.constexpr,
    HAS_P: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    HSPLITS: tl.constexpr,
    P_TIERS: tl.constexpr,
    SUPERGROUPS_P2: tl.constexpr,
    BLOCK: tl.constexpr,
    BINS: tl.constexpr,
    BIN_SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
    KEPT: tl.constexpr,
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    for row in range(pid, rows, programs):
        _select_row(
            logits_ptr,
            stride,
            level1_ptr,
            level2_ptr,
            stats_ptr,
            row_stats_ptr,
            state_ptr,
            kept_ptr,
            hist_ptr,
            sync_ptr,
            row,
            vocab,
            groups,
            supergroups,
            k_ptr,
            p_ptr,
            splits,
            HAS_K,
            HAS_P,
            MAX_SPLITS,
            SUPERGROUPS_P2,
            BINS,
            KEPT,
            P_TIERS,
        )
    _grid_sync(sync_ptr, programs)
    if tl.load(sync_ptr + 2, volatile=True) != 0:  # some row needs the histogram path
        for item in range(pid, rows * HSPLITS, programs):
            row = item // HSPLITS
            if tl.load(state_ptr + row, volatile=True) == HIST_C:
                _histogram(
                    logits_ptr + tl.cast(row, tl.int64) * stride,
                    hist_ptr + tl.cast(row, tl.int64) * BINS,
                    tl.load(row_stats_ptr + 2 * row, volatile=True).to(tl.float32),
                    item % HSPLITS * hchunk,
                    vocab,
                    hchunk,
                    BINS,
                    BIN_SCALE,
                    BLOCK,
                )
        _grid_sync(sync_ptr, 2 * programs)
        # Rows with one cut (k or p) whose bins hold at most WINDOW logits: all programs
        # gather them, then one program per row sorts and cuts. Others: one program.
        for row in range(pid, rows, programs):
            if tl.load(state_ptr + row, volatile=True) == HIST_C:
                k, p = _row_params(k_ptr, p_ptr, row, vocab, HAS_K, HAS_P)
                _plan_row(
                    hist_ptr + tl.cast(row, tl.int64) * BINS,
                    state_ptr + row,
                    window_ptr + 2 * row,
                    found_ptr + row,
                    more_ptr + row,
                    k,
                    p,
                    tl.load(row_stats_ptr + 2 * row, volatile=True).to(tl.float32),
                    tl.load(row_stats_ptr + 2 * row + 1, volatile=True),
                    vocab,
                    BINS,
                    BIN_SCALE,
                    WINDOW,
                )
        _grid_sync(sync_ptr, 3 * programs)
        for item in range(pid, rows * HSPLITS, programs):
            row = item // HSPLITS
            if tl.load(state_ptr + row, volatile=True) == GATHER_C:
                k, p = _row_params(k_ptr, p_ptr, row, vocab, HAS_K, HAS_P)
                _gather_part(
                    logits_ptr + tl.cast(row, tl.int64) * stride,
                    scratch_ptr + tl.cast(row, tl.int64) * WINDOW,
                    found_ptr + row,
                    more_ptr + row,
                    tl.load(row_stats_ptr + 2 * row, volatile=True).to(tl.float32),
                    tl.load(window_ptr + 2 * row, volatile=True),
                    tl.load(window_ptr + 2 * row + 1, volatile=True),
                    k >= vocab,
                    item % HSPLITS * hchunk,
                    vocab,
                    hchunk,
                    BINS,
                    BIN_SCALE,
                    WINDOW,
                    BLOCK,
                )
        _grid_sync(sync_ptr, 4 * programs)
        for row in range(pid, rows, programs):
            state = tl.load(state_ptr + row, volatile=True)
            if (state == HIST_C) | (state == GATHER_C):
                k, p = _row_params(k_ptr, p_ptr, row, vocab, HAS_K, HAS_P)
                _resolve_row(
                    logits_ptr + tl.cast(row, tl.int64) * stride,
                    hist_ptr + tl.cast(row, tl.int64) * BINS,
                    scratch_ptr + tl.cast(row, tl.int64) * WINDOW,
                    state_ptr + row,
                    cut_ptr + row,
                    k,
                    p,
                    tl.load(row_stats_ptr + 2 * row, volatile=True).to(tl.float32),
                    tl.load(row_stats_ptr + 2 * row + 1, volatile=True),
                    vocab,
                    state == GATHER_C,
                    tl.load(found_ptr + row, volatile=True),
                    tl.load(more_ptr + row, volatile=True),
                    BINS,
                    BIN_SCALE,
                    WINDOW,
                    BLOCK,
                )
        _grid_sync(sync_ptr, 5 * programs)
    for item in range(pid, rows * fsplits, programs):
        row = item // fsplits
        _fill(
            logits_ptr + tl.cast(row, tl.int64) * stride,
            tl.load(state_ptr + row, volatile=True),
            tl.load(cut_ptr + row, volatile=True),
            kept_ptr + tl.cast(row, tl.int64) * KEPT,
            item % fsplits * fchunk,
            vocab,
            fchunk,
            BLOCK,
        )
    _grid_exit(sync_ptr, programs)


@triton.jit
def _grid_sync(sync_ptr, target):
    """Wait until target arrivals (every program, once per barrier so far)."""
    tl.debug_barrier()
    tl.atomic_add(sync_ptr, 1, sem="release", scope="gpu")
    while tl.atomic_add(sync_ptr, 0, sem="acquire", scope="gpu") < target:
        pass
    tl.debug_barrier()


@triton.jit
def _grid_exit(sync_ptr, programs):
    """The last program to leave resets the barrier, the exit count and the flag."""
    tl.debug_barrier()
    if tl.atomic_add(sync_ptr + 1, 1, sem="acq_rel", scope="gpu") == programs - 1:
        tl.store(sync_ptr, 0)
        tl.store(sync_ptr + 1, 0)
        tl.store(sync_ptr + 2, 0)


@triton.jit
def _select_row(
    logits_ptr,
    stride,
    level1_ptr,
    level2_ptr,
    stats_ptr,
    row_stats_ptr,
    state_ptr,
    kept_ptr,
    hist_ptr,
    sync_ptr,
    row,
    vocab,
    groups,
    supergroups,
    k_ptr,
    p_ptr,
    splits,
    HAS_K: tl.constexpr,
    HAS_P: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    SUPERGROUPS_P2: tl.constexpr,
    BINS: tl.constexpr,
    KEPT: tl.constexpr,
    P_TIERS: tl.constexpr,
):
    k, p = _row_params(k_ptr, p_ptr, row, vocab, HAS_K, HAS_P)
    no_k = k >= vocab
    done = no_k & (p >= 1.0)
    kept = -1
    if not done:
        row_max, row_sum = _row_stats(stats_ptr, row, splits, MAX_SPLITS)
        tl.store(row_stats_ptr + 2 * row, row_max.to(tl.float64))
        tl.store(row_stats_ptr + 2 * row + 1, row_sum)
        s = tl.arange(0, SUPERGROUPS_P2)
        level2 = tl.load(
            level2_ptr + tl.cast(row, tl.int64) * supergroups + s,
            mask=s < supergroups,
            other=-float("inf"),
            volatile=True,
        )
        args = (
            logits_ptr + tl.cast(row, tl.int64) * stride,
            level1_ptr + tl.cast(row, tl.int64) * groups,
            kept_ptr + tl.cast(row, tl.int64) * KEPT,
            _key(level2, s),
            level2,
            k,
            p,
            row_max,
            row_sum,
            vocab,
            groups,
        )
        if no_k | (k <= 32):
            done, kept = _tier(*args, 32)
        if (not done) & (k <= 64):
            done, kept = _tier(*args, 64)
        if (not done) & ((no_k & (P_TIERS > 1)) | (k <= 128)):
            done, kept = _tier(*args, 128)
        if (not done) & (k <= 256):
            done, kept = _tier(*args, 256)
        if not done:
            zeros = tl.zeros([BINS], tl.int32)
            tl.store(
                hist_ptr + tl.cast(row, tl.int64) * BINS + tl.arange(0, BINS), zeros
            )
            tl.store(sync_ptr + 2, 1)
    tl.store(
        state_ptr + row,
        tl.where(done, kept, tl.where(no_k & (p >= 1.0), NOOP_C, HIST_C)),
    )


@triton.jit
def _tier(
    row_ptr,
    level1_row,
    kept_row,
    keys2,
    level2,
    k,
    p,
    row_max,
    row_sum,
    vocab,
    groups,
    C: tl.constexpr,
):
    """Store the kept keys if the row's best C logits decide the cutoff, return
    whether they did and how many are kept."""
    use_k = k < vocab
    use_p = p < 1.0
    rank = tl.arange(0, C)
    # Top-p only needs C / 2, leaving room for ties at the cuts.
    need = tl.where(use_k, k, C // 2)
    ok = need <= C
    # The supergroups at or above the need-th best max hold the best need groups.
    top2 = tl.topk(keys2, C, descending=False)
    cut2 = tl.sum(tl.where(rank == need - 1, _key_value(top2), 0.0))
    count2 = tl.sum((level2 >= cut2).to(tl.int32))
    ok = ok & (count2 <= C)
    group = (top2 & 0xFFFFFFFF)[:, None] * 16 + tl.arange(0, 16)[None, :]
    level1 = tl.load(
        level1_row + group,
        mask=(rank < count2)[:, None] & (group < groups),
        other=-float("inf"),
        volatile=True,
    )
    # The groups at or above the need-th best group max hold every logit at or above
    # it (at least need logits).
    top1 = tl.topk(tl.reshape(_key(level1, group), [C * 16]), C, descending=False)
    cut1 = tl.sum(tl.where(rank == need - 1, _key_value(top1), 0.0))
    count1 = tl.sum((tl.reshape(level1, [C * 16]) >= cut1).to(tl.int32))
    ok = ok & (count1 <= C)
    index = (top1 & 0xFFFFFFFF)[:, None] * 16 + tl.arange(0, 16)[None, :]
    x = tl.load(
        row_ptr + index,
        mask=(rank < count1)[:, None] & (index < vocab),
        other=-float("inf"),
    )
    # The best logits in exact order (logit descending, index ascending), exact up to
    # the last one at or above cut1.
    top = tl.topk(tl.reshape(_key(x, index), [C * 16]), C, descending=False)
    value = _key_value(top)
    keep = (rank < k) & (value >= cut1)
    if use_p:
        best = tl.max(value)
        weight = tl.where(keep, tl.exp(value - best), 0.0)
        kept_mass = tl.sum(weight)
        total = tl.where(
            use_k, kept_mass, row_sum.to(tl.float32) * tl.exp(row_max - best)
        )
        keep = keep & (tl.cumsum(weight, 0) - weight < p * total)
        ok = ok & (use_k | (kept_mass >= p * total))
    tl.store(kept_row + rank, top, mask=keep & ok)
    return ok, tl.sum(keep.to(tl.int32))


@triton.jit
def _bin(x, top, scale, BINS: tl.constexpr):
    """Histogram bin of a logit: bins of 1/scale nats below top, the last bin holds
    everything further down (and -inf)."""
    return tl.minimum((top - x) * scale, BINS - 1.0).to(tl.int32)


@triton.jit
def _histogram(
    row_ptr,
    hist_row,
    row_max,
    lo,
    vocab,
    chunk,
    BINS: tl.constexpr,
    BIN_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    hi = tl.minimum(lo + chunk, vocab)
    counts = tl.zeros([BINS], tl.int32)
    for start in range(lo, hi, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < hi, other=-float("inf"))
        counts += tl.histogram(_bin(x, row_max, BIN_SCALE, BINS), BINS, mask=idx < hi)
    tl.atomic_add(hist_row + tl.arange(0, BINS), counts, sem="relaxed", scope="gpu")


@triton.jit
def _resolve_row(
    row_ptr,
    hist_row,
    scratch,
    state_row,
    cut_row,
    k,
    p,
    row_max,
    row_sum,
    vocab,
    gathered,
    found,
    more,
    BINS: tl.constexpr,
    BIN_SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Exact cut of a row from its histogram: the key of the last kept logit."""
    use_k = k < vocab
    every = vocab.to(tl.int64) * 0 + 0x7FFFFFFFFFFFFFFF  # a cut that keeps every key
    done = vocab < 0
    cut = every
    if gathered & (found <= WINDOW):
        if use_k:
            done, cut = _finish_any(
                scratch, found, k.to(tl.float64), more, row_max, False, WINDOW
            )
        else:
            done, cut = _finish_any(
                scratch, found, p.to(tl.float64) * row_sum, more, row_max, True, WINDOW
            )
    if done:
        tl.store(cut_row, cut)
        tl.store(state_row, CUT_C)
    else:
        _resolve_alone(
            row_ptr,
            hist_row,
            scratch,
            state_row,
            cut_row,
            k,
            p,
            row_max,
            row_sum,
            vocab,
            BINS,
            BIN_SCALE,
            WINDOW,
            BLOCK,
        )


@triton.jit
def _plan_row(
    hist_row,
    state_row,
    window_row,
    found_row,
    more_row,
    k,
    p,
    row_max,
    row_sum,
    vocab,
    BINS: tl.constexpr,
    BIN_SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
):
    """Mark a row with one cut whose bins hold at most WINDOW logits for the
    shared gather."""
    use_k = k < vocab
    use_p = p < 1.0
    if use_k ^ use_p:
        counts = tl.load(hist_row + tl.arange(0, BINS), volatile=True)
        floor = row_max * 0.0 - float("inf")
        scale = row_max * 0.0 + BIN_SCALE
        zero = row_sum * 0.0
        if use_k:
            lo, hi, n = _window(
                counts, k.to(tl.float64), row_max, floor, scale, row_max, False, BINS
            )
        else:
            lo, hi, n = _window(
                counts,
                p.to(tl.float64) * row_sum - zero,
                row_max,
                floor,
                scale,
                row_max,
                True,
                BINS,
            )
        if n <= WINDOW:
            tl.store(window_row, lo)
            tl.store(window_row + 1, hi)
            tl.store(found_row, 0)
            tl.store(more_row, zero)
            tl.store(state_row, GATHER_C)


@triton.jit
def _gather_part(
    row_ptr,
    scratch,
    found_row,
    more_row,
    row_max,
    lo,
    hi,
    mass,
    start_at,
    vocab,
    chunk,
    BINS: tl.constexpr,
    BIN_SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One split's share of a shared gather: keys of the logits in bins lo to hi,
    and the count (or mass) of the logits in bins above lo."""
    offs = tl.arange(0, BLOCK)
    end = tl.minimum(start_at + chunk, vocab)
    more = (vocab * 0).to(tl.float64)
    for start in range(start_at, end, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < end, other=-float("inf"))
        b = _bin(x, row_max, BIN_SCALE, BINS)
        weight = tl.where(mass, tl.exp(x - row_max), 1.0)
        more += tl.sum(tl.where((idx < end) & (b < lo), weight, 0.0)).to(tl.float64)
        hit = ((idx < end) & (b >= lo) & (b <= hi)).to(tl.int32)
        hits = tl.sum(hit)
        if hits > 0:
            base = tl.atomic_add(found_row, hits, sem="relaxed", scope="gpu")
            pos = base + tl.cumsum(hit, 0) - 1
            tl.store(scratch + pos, _key(x, idx), mask=(hit > 0) & (pos < WINDOW))
    tl.atomic_add(more_row, more, sem="relaxed", scope="gpu")


@triton.jit
def _finish_any(
    scratch, found, target, above, row_max, MASS: tl.constexpr, WINDOW: tl.constexpr
):
    if found <= 512:
        ok, cut = _finish(scratch, found, target, above, row_max, MASS, 512)
    elif found <= 1024:
        ok, cut = _finish(scratch, found, target, above, row_max, MASS, 1024)
    elif found <= 2048:
        ok, cut = _finish(scratch, found, target, above, row_max, MASS, 2048)
    else:
        ok, cut = _finish(scratch, found, target, above, row_max, MASS, WINDOW)
    return ok, cut


@triton.jit
def _window(
    counts, need, top, floor, scale, row_max, MASS: tl.constexpr, BINS: tl.constexpr
):
    """The bins lo to hi that can hold the need-th logit (or the logit where the running
    mass reaches need), and how many logits they hold."""
    j = tl.arange(0, BINS)
    c = counts.to(tl.float32)
    if MASS:
        lower = tl.where(j == BINS - 1, floor, top - (j + 1.0) / scale)
        low = tl.cumsum(c * tl.exp(lower - row_max), 0)
        high = tl.cumsum(c * tl.exp(top - j / scale - row_max), 0)
        lo = tl.min(tl.where(high >= need * (1 - 1e-4), j, BINS - 1))
        hi = tl.min(tl.where(low >= need * (1 + 1e-4), j, BINS - 1))
    else:
        lo = tl.min(tl.where(tl.cumsum(c, 0) >= need, j, BINS - 1))
        hi = lo
    return lo, hi, tl.sum(tl.where((j >= lo) & (j <= hi), counts, 0))


@triton.jit
def _resolve_alone(
    row_ptr,
    hist_row,
    scratch,
    state_row,
    cut_row,
    k,
    p,
    row_max,
    row_sum,
    vocab,
    BINS: tl.constexpr,
    BIN_SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program cuts the row (two cuts, refinement, ties)."""
    use_k = k < vocab
    use_p = p < 1.0
    j = tl.arange(0, BINS)
    counts = tl.load(hist_row + j, volatile=True)
    ok = vocab > 0
    cut = vocab.to(tl.int64) * 0 + 0x7FFFFFFFFFFFFFFF  # keeps every key
    if use_k:
        ok, cut = _cut(
            row_ptr,
            scratch,
            counts,
            k.to(tl.float64),
            cut,
            row_max,
            vocab,
            False,
            BINS,
            BIN_SCALE,
            WINDOW,
            BLOCK,
        )
        # Only the top-k logits count toward top-p.
        b = tl.min(tl.where(tl.cumsum(counts, 0) >= k, j, BINS))
        counts = tl.where(
            j < b, counts, tl.where(j == b, k - tl.sum(tl.where(j < b, counts, 0)), 0)
        )
    if use_p & ok:
        total = row_sum
        if use_k:
            total = _mass_upto(row_ptr, row_max, cut, vocab, BLOCK)
        ok, cut = _cut(
            row_ptr,
            scratch,
            counts,
            p.to(tl.float64) * total,
            cut,
            row_max,
            vocab,
            True,
            BINS,
            BIN_SCALE,
            WINDOW,
            BLOCK,
        )
    if ok:
        tl.store(cut_row, cut)
        tl.store(state_row, CUT_C)
    else:
        _radix_row(row_ptr, vocab, k, p, use_k, use_p, BLOCK // 4)
        tl.store(state_row, SELF_C)


@triton.jit
def _cut(
    row_ptr,
    scratch,
    counts,
    target,
    limit,
    row_max,
    vocab,
    MASS: tl.constexpr,
    BINS: tl.constexpr,
    BIN_SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Among the keys at most limit, the key of the target-th logit (or of the first
    whose running mass reaches target). The bins that can hold it are gathered and
    sorted once they hold at most WINDOW logits, else the histogram is refined over
    their value range, and a range of one value is cut by index."""
    top = row_max
    floor = row_max * 0.0 - float("inf")
    scale = row_max * 0.0 + BIN_SCALE
    above = target * 0.0  # count (or mass) of the kept logits above the range
    ok = vocab < 0
    cut = limit
    rounds = 0
    done = 0
    while (done == 0) & (rounds < 4):
        lo, hi, n = _window(
            counts, target - above, top, floor, scale, row_max, MASS, BINS
        )
        if n <= WINDOW:
            found, more = _gather(
                row_ptr,
                scratch,
                top,
                floor,
                scale,
                lo,
                hi,
                limit,
                row_max,
                vocab,
                MASS,
                BINS,
                WINDOW,
                BLOCK,
            )
            ok, cut = _finish_any(
                scratch, found, target, above + more, row_max, MASS, WINDOW
            )
            done = 1
        else:
            low_v, high_v, more = _window_range(
                row_ptr,
                top,
                floor,
                scale,
                lo,
                hi,
                limit,
                row_max,
                vocab,
                MASS,
                BINS,
                BLOCK,
            )
            above += more
            if low_v >= high_v:
                # One value left (-inf if the window holds no finite logit): by index.
                value = tl.where(low_v > high_v, -float("inf"), low_v)
                ok, cut = _tie_cut(
                    row_ptr, value, target, above, limit, row_max, vocab, MASS, BLOCK
                )
                done = 1
            else:
                top = high_v
                floor = low_v
                scale = (BINS - 0.01) / (high_v - low_v)
                counts = _range_histogram(
                    row_ptr, top, floor, scale, limit, vocab, BINS, BLOCK
                )
        rounds += 1
    return ok & (done == 1), cut


@triton.jit
def _finish(
    scratch, found, target, above, row_max, MASS: tl.constexpr, N: tl.constexpr
):
    """Sort the gathered keys, return the key where the running count (or mass) from
    above reaches target."""
    rank = tl.arange(0, N)
    key = tl.sort(tl.load(scratch + rank, mask=rank < found, other=0x7FFFFFFFFFFFFFFF))
    if MASS:
        weight = tl.where(rank < found, tl.exp(_key_value(key) - row_max), 0.0).to(
            tl.float64
        )
        reached = above + tl.cumsum(weight, 0) >= target
    else:
        reached = above + rank + 1.0 >= target
    last = tl.min(tl.where(reached & (rank < found), rank, N))
    return last < found, tl.sum(tl.where(rank == last, key, 0))


@triton.jit
def _gather(
    row_ptr,
    scratch,
    top,
    floor,
    scale,
    lo,
    hi,
    limit,
    row_max,
    vocab,
    MASS: tl.constexpr,
    BINS: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Store the keys of the range's logits in bins lo to hi (keys at most limit),
    return how many there are and the count (or mass) of the range's logits in bins
    above lo."""
    offs = tl.arange(0, BLOCK)
    found = 0
    more = (vocab * 0).to(tl.float64)
    for start in range(0, vocab, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < vocab, other=-float("inf"))
        key = _key(x, idx)
        in_range = (idx < vocab) & (x >= floor) & (x <= top) & (key <= limit)
        b = _bin(x, top, scale, BINS)
        weight = tl.exp(x - row_max) if MASS else 1.0
        more += tl.sum(tl.where(in_range & (b < lo), weight, 0.0)).to(tl.float64)
        hit = (in_range & (b >= lo) & (b <= hi)).to(tl.int32)
        hits = tl.sum(hit)
        if hits > 0:
            pos = found + tl.cumsum(hit, 0) - 1
            tl.store(scratch + pos, key, mask=(hit > 0) & (pos < WINDOW))
        found += hits
    tl.debug_barrier()
    return found, more


@triton.jit
def _window_range(
    row_ptr,
    top,
    floor,
    scale,
    lo,
    hi,
    limit,
    row_max,
    vocab,
    MASS: tl.constexpr,
    BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Smallest and largest finite logit in bins lo to hi of the range, and the count
    (or mass) of the range's logits in bins above lo."""
    offs = tl.arange(0, BLOCK)
    low_v = tl.full([BLOCK], float("inf"), tl.float32)
    high_v = tl.full([BLOCK], -float("inf"), tl.float32)
    more = (vocab * 0).to(tl.float64)
    for start in range(0, vocab, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < vocab, other=-float("inf"))
        in_range = (idx < vocab) & (x >= floor) & (x <= top) & (_key(x, idx) <= limit)
        b = _bin(x, top, scale, BINS)
        window = in_range & (b >= lo) & (b <= hi) & (x > -float("inf"))
        low_v = tl.minimum(low_v, tl.where(window, x, float("inf")))
        high_v = tl.maximum(high_v, tl.where(window, x, -float("inf")))
        weight = tl.exp(x - row_max) if MASS else 1.0
        more += tl.sum(tl.where(in_range & (b < lo), weight, 0.0)).to(tl.float64)
    return tl.min(low_v), tl.max(high_v), more


@triton.jit
def _range_histogram(
    row_ptr, top, floor, scale, limit, vocab, BINS: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    counts = tl.zeros([BINS], tl.int32)
    for start in range(0, vocab, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < vocab, other=-float("inf"))
        in_range = (idx < vocab) & (x >= floor) & (x <= top) & (_key(x, idx) <= limit)
        counts += tl.histogram(_bin(x, top, scale, BINS), BINS, mask=in_range)
    return counts


@triton.jit
def _tie_cut(
    row_ptr,
    value,
    target,
    above,
    limit,
    row_max,
    vocab,
    MASS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The key of the tie (logit equal to value, key at most limit) that reaches target,
    ties in index order."""
    if MASS:
        weight = tl.exp(value - row_max).to(tl.float64)
        needed = tl.where(weight > 0, tl.ceil((target - above) / weight), 3.0e38)
    else:
        needed = target - above
    needed = tl.maximum(needed, 1.0)
    offs = tl.arange(0, BLOCK)
    seen = 0.0
    cut = vocab.to(tl.int64) * 0
    hits = 0
    for start in range(0, vocab, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < vocab, other=float("nan"))
        key = _key(x, idx)
        tie = (idx < vocab) & (x == value) & (key <= limit)
        ties = tl.sum(tie.to(tl.float32))
        if (seen < needed) & (seen + ties >= needed):
            chosen = tie & (seen + tl.cumsum(tie.to(tl.float32), 0) == needed)
            cut += tl.sum(tl.where(chosen, key, 0))
            hits += tl.sum(chosen.to(tl.int32))
        seen += ties
    return hits == 1, cut


@triton.jit
def _mass_upto(row_ptr, row_max, cut, vocab, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    total = (vocab * 0).to(tl.float64)
    for start in range(0, vocab, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < vocab, other=-float("inf"))
        keep = (_key(x, idx) <= cut) & (idx < vocab)
        total += tl.sum(tl.where(keep, tl.exp(x - row_max), 0.0)).to(tl.float64)
    return total


@triton.jit
def _fill(row_ptr, state, cut, kept_row, lo, vocab, chunk, BLOCK: tl.constexpr):
    """Mask the split's chunk: from the kept keys (state >= 0) or from the cut key."""
    hi = tl.minimum(lo + chunk, vocab)
    offs = tl.arange(0, BLOCK)
    if state == CUT_C:
        for start in range(lo, hi, BLOCK):
            idx = start + offs
            x = tl.load(row_ptr + idx, mask=idx < hi)
            keep = _key(x, idx) <= cut
            tl.store(row_ptr + idx, tl.where(keep, x, -float("inf")), mask=idx < hi)
    if state >= 0:
        for start in range(lo, hi, BLOCK):
            tl.store(row_ptr + start + offs, -float("inf"), mask=start + offs < hi)
        tl.debug_barrier()
        slots = tl.arange(0, 256)
        key = tl.load(kept_row + slots, mask=slots < state, other=0, volatile=True)
        index = (key & 0xFFFFFFFF).to(tl.int32)
        valid = (slots < state) & (index >= lo) & (index < hi)
        tl.store(row_ptr + index, _key_value(key), mask=valid)


# Last resort (cut bins with more than WINDOW logits, e.g. huge exact ties): radix
# select over the 64-bit key (ordered logit, inverted index), 4 bits per pass, by one
# program over the row. Slow, but exact.


@triton.jit
def _radix_row(row_ptr, vocab, k, p, use_k, use_p, BLOCK: tl.constexpr):
    bound_hi = vocab.to(tl.int64) * 0
    bound_lo = vocab.to(tl.int64) * 0
    if use_k:
        bound_hi, bound_lo = _radix_select(
            row_ptr, vocab, bound_hi, bound_lo, k.to(tl.float64), 0.0, False, BLOCK
        )
    if use_p:
        row_max = _plain_max(row_ptr, vocab, BLOCK)
        mass = _bounded_mass(row_ptr, vocab, bound_hi, bound_lo, row_max, BLOCK)
        bound_hi, bound_lo = _radix_select(
            row_ptr,
            vocab,
            bound_hi,
            bound_lo,
            p.to(tl.float64) * mass,
            row_max,
            True,
            BLOCK,
        )
    offs = tl.arange(0, BLOCK)
    for start in range(0, vocab, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < vocab)
        keep = _at_least(_ordered(x), 0xFFFFFFFF - idx.to(tl.int64), bound_hi, bound_lo)
        tl.store(row_ptr + idx, tl.where(keep, x, -float("inf")), mask=idx < vocab)


@triton.jit
def _radix_select(
    row_ptr,
    vocab,
    bound_hi,
    bound_lo,
    target,
    row_max,
    MASS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The largest key whose keys at or above it, within the bound, reach target
    elements (or mass)."""
    bins = tl.arange(0, 16)
    prefix_hi = bound_hi * 0
    prefix_lo = bound_hi * 0
    out_hi = bound_hi
    out_lo = bound_lo
    remaining = target
    digit = 0
    done = 0
    while (digit < 16) & (done == 0):
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
        stop = (in_bucket <= 1) | (digit == 15)
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
            weight = tl.exp(x - row_max)
            masses += tl.sum(tl.where(onehot, weight[:, None], 0.0), 0).to(tl.float64)
    return counts, masses


@triton.jit
def _at_least(hi, lo, bound_hi, bound_lo):
    return (hi > bound_hi) | ((hi == bound_hi) & (lo >= bound_lo))


@triton.jit
def _plain_max(row_ptr, vocab, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    best = tl.full([BLOCK], -float("inf"), tl.float32)
    for start in range(0, vocab, BLOCK):
        idx = start + offs
        best = tl.maximum(
            best, tl.load(row_ptr + idx, mask=idx < vocab, other=-float("inf"))
        )
    return tl.max(best)


@triton.jit
def _bounded_mass(row_ptr, vocab, bound_hi, bound_lo, row_max, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    total = tl.zeros([BLOCK], tl.float64)
    for start in range(0, vocab, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < vocab, other=-float("inf"))
        keep = _at_least(_ordered(x), 0xFFFFFFFF - idx.to(tl.int64), bound_hi, bound_lo)
        total += tl.where(keep, tl.exp(x - row_max), 0.0).to(tl.float64)
    return tl.sum(total)


@triton.jit
def _row_params(k_ptr, p_ptr, row, vocab, HAS_K: tl.constexpr, HAS_P: tl.constexpr):
    k = vocab
    p = vocab.to(tl.float32) * 0.0 + 1.0
    if HAS_K:
        k = tl.load(k_ptr + row).to(tl.int32)
    if HAS_P:
        p = tl.load(p_ptr + row).to(tl.float32)
    return k, p


@triton.jit
def _row_stats(stats_ptr, row, splits, MAX_SPLITS: tl.constexpr):
    """The row's max and sum of exp(logit - max), from the per-split partials."""
    split = tl.arange(0, MAX_SPLITS)
    program = row * MAX_SPLITS + split
    split_max = tl.load(stats_ptr + 2 * program, mask=split < splits, other=-3.0e38)
    split_max = split_max.to(tl.float32)
    split_sum = tl.load(stats_ptr + 2 * program + 1, mask=split < splits, other=0.0)
    row_max = tl.max(split_max)
    return row_max, tl.sum(split_sum * tl.exp(split_max - row_max).to(tl.float64))


@triton.jit
def _key(x, idx):
    """Ascending order is logit descending, then index ascending."""
    return ((0x7FFFFFFF - _ordered(x)) << 32) | idx.to(tl.int64)


@triton.jit
def _ordered(x):
    """Order-preserving uint32 of the logit, in an int64."""
    bits = tl.where(x == 0, 0.0, x).to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    return tl.where(bits >= 0x80000000, 0xFFFFFFFF - bits, bits + 0x80000000)


@triton.jit
def _key_value(key):
    ordered = 0x7FFFFFFF - (key >> 32)
    bits = tl.where(ordered >= 0x80000000, ordered - 0x80000000, 0xFFFFFFFF - ordered)
    return bits.to(tl.uint32).to(tl.float32, bitcast=True)
