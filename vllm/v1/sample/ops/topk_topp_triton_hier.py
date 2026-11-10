# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k/top-p masking of FP32 logits through group maxima, in two launches.

1. Levels (rows x splits): stream each row once and write the max of every group of
   16 logits (level 1) and of every 256 (level 2), the row max, and the sum of exp.
   No compaction, so it runs close to the streaming bandwidth.
2. One persistent cooperative kernel, with phases separated by grid barriers:
   a. Select (per row): exact selection through the levels, in one tier of 32, 64,
      128 or 256 logits (top-p only: 128). At each level a histogram of the maxima
      picks the best bins that hold enough items, at most the tier's: they hold the
      best items of the next level. Only the tier's logits are sorted.
   b. Only if a row missed its tier (k > 256, a larger nucleus, crowded ties): an
      8192-bin histogram of (row max - logit) in 1/512 nat bins by all programs, a
      plan per row (the bins that can hold the k and p cuts, from counts and mass
      bounds), one pass that masks the bins below and gathers the cut bins (at most
      4096 logits), and an exact cut on the sorted window. A row with a larger
      window is cut by one program, by refinement or an exact radix select.
   c. Fill (rows x splits): -inf, then the kept logits (or every logit up to the cut).
Exactly k logits are kept, ties broken by the lowest index, then the smallest prefix
whose mass reaches p. Runtime ints are not specialized, so one compiled variant serves
every batch size and k/p mix, and the launches are cached per shape.
"""

import functools

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

BLOCK = 4096  # elements per streaming step, a multiple of 256
MAX_SPLITS = 32
HIST_ITEMS = 4  # items per program of the histogram passes (each adds BINS atomics)
FILL_SPLITS = 8  # at least this many splits per row for the fill (balances cut rows)
BINS = 8192
BIN_SCALE = 512.0  # bins per nat below the row max (16 nats)
WINDOW = 4096  # logits gathered to resolve a histogram cut
KEPT = 256  # kept logits stored per row by the tiers, also the largest tier
WARPS = 16  # of the persistent kernel: more warps hide the latency of its block syncs
# Keys hold the index in 32 bits.
_MAX_VOCAB_SIZE = 1 << 20
NOOP_C = tl.constexpr(-1)
HIST_C = tl.constexpr(-2)  # row waits for the histogram path
CUT_C = tl.constexpr(-3)  # row keeps every logit whose key is at most cut[row]
SELF_C = tl.constexpr(-4)  # row already masked by its cut
GATHER_C = tl.constexpr(-5)  # row's cut bins are gathered by every program
# stride and vocab stay specialized (aligned rows load vectorized), so a vocab that is
# not a multiple of 16 compiles a second variant. Chunks are multiples of BLOCK.
LEVELS_INTS = ["groups", "supergroups", "has_k", "has_p", "splits"]
FILTER_INTS = LEVELS_INTS + ["rows", "fsplits"]


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
    key = (logits.device, *logits.shape, logits.stride(0), k is not None, p is not None)
    plan = _PLANS.get(key)
    if plan is None:
        return launch_jit(logits, k, p, key)
    stream = torch.cuda.current_stream(logits.device).cuda_stream
    for kernel, grid, args, (at_logits, at_k, at_p) in plan:
        args[at_logits] = logits
        args[at_k] = k if k is not None else args[at_k]  # else a stand-in buffer
        args[at_p] = p if p is not None else logits
        kernel.run(
            *grid,
            stream,
            kernel.function,
            kernel.packed_metadata,
            None,
            None,
            None,
            *args,
        )
        args[at_logits] = args[at_p] = None  # do not keep the caller's tensors alive
    return logits


# Launches cached per shape: Triton's launcher with ~40 keyword arguments costs tens
# of µs of host time per call, which shows at small batch.
_PLANS: dict[tuple, list] = {}


def launch_jit(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None, key: tuple
) -> torch.Tensor:
    """Launch through Triton's JIT (compiling if needed) and cache the launches."""
    rows, vocab = logits.shape
    device = logits.device
    sms = num_sms(device)
    has_k, has_p = k is not None, p is not None
    _, ints, levels_meta, levels_opts = launch_args(
        "levels", rows, vocab, has_k, has_p, sms
    )
    _, fints, meta, opts = launch_args("filter", rows, vocab, has_k, has_p, sms)
    ints["stride"] = fints["stride"] = logits.stride(0)
    b = buffers(device, rows, vocab)
    # Unused pointers get buffers of the same dtype, so the signature does not vary.
    params = dict(k_ptr=k if has_k else b["state"], p_ptr=p if has_p else logits)
    level_ptrs = dict(
        logits_ptr=logits,
        level1_ptr=b["level1"],
        level2_ptr=b["level2"],
        stats_ptr=b["stats"],
    )
    levels_grid = (rows, ints["splits"], 1)
    levels_args = dict(**level_ptrs, **params, **ints, **levels_meta)
    levels = _levels_kernel[levels_grid](**levels_args, **levels_opts)
    names = (
        "row_stats",
        "state",
        "cut",
        "kept",
        "hist",
        "scratch",
        "window",
        "found",
        "more",
        "sync",
    )
    capacity_key = (device, *meta.items())
    meta = dict(
        **meta, **params, **fints, **level_ptrs, **{f"{n}_ptr": b[n] for n in names}
    )
    programs = min(capacity(capacity_key, meta, opts), rows * ints["splits"])
    filter_ = _filter_kernel[(programs,)](**meta, **opts)
    _PLANS[key] = [
        plan_entry(levels, _levels_kernel, levels_grid, levels_args),
        plan_entry(filter_, _filter_kernel, (programs, 1, 1), meta),
    ]
    return logits


def plan_entry(compiled, jit, grid: tuple, kwargs: dict) -> tuple:
    """A compiled kernel's launch: its arguments in order, and where logits, k and p
    go."""
    defaults = {param.name: param.default for param in jit.params}
    values = [kwargs.get(name, defaults[name]) for name in jit.arg_names]
    slots = [jit.arg_names.index(n) for n in ("logits_ptr", "k_ptr", "p_ptr")]
    return compiled, grid, values, slots


def launch_args(
    name: str, rows: int, vocab: int, has_k: bool, has_p: bool, sms: int
) -> tuple:
    """(kernel, runtime ints, constexprs, launch options) of one launch."""
    splits = num_splits(rows, sms)
    groups, supergroups = triton.cdiv(vocab, 16), triton.cdiv(vocab, 256)
    ints = dict(
        stride=vocab,
        vocab=vocab,
        chunk=split_chunk(vocab, splits),
        groups=groups,
        supergroups=supergroups,
        has_k=int(has_k),
        has_p=int(has_p),
        splits=splits,
    )
    common = dict(MAX_SPLITS=MAX_SPLITS, BLOCK=BLOCK)
    if name == "levels":
        return _levels_kernel, ints, common, dict(num_warps=4)
    fill_splits = max(splits, FILL_SPLITS)
    ints.update(rows=rows, fchunk=split_chunk(vocab, fill_splits), fsplits=fill_splits)
    # At least 1024 supergroups (vocab up to 256k), so common vocabs share one variant.
    meta = dict(
        **common,
        HIST_ITEMS=HIST_ITEMS,
        SUPERGROUPS_P2=max(1024, triton.next_power_of_2(supergroups)),
        BINS=BINS,
        BIN_SCALE=BIN_SCALE,
        WINDOW=WINDOW,
        KEPT=KEPT,
    )
    return (
        _filter_kernel,
        ints,
        meta,
        dict(num_warps=WARPS, launch_cooperative_grid=True),
    )


def split_chunk(vocab: int, splits: int) -> int:
    return triton.cdiv(triton.cdiv(vocab, splits), BLOCK) * BLOCK


def num_splits(rows: int, sms: int) -> int:
    """Split rows for streaming until the grid has about 4 programs per SM."""
    return min(MAX_SPLITS, triton.next_power_of_2(triton.cdiv(4 * sms, rows)))


@functools.cache
def num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


_CAPACITY: dict[tuple, int] = {}


def capacity(key: tuple, meta: dict, opts: dict) -> int:
    """Programs of the persistent kernel that fit on the GPU at once, so that the
    grid barriers cannot deadlock (the cooperative launch fails otherwise)."""
    if key not in _CAPACITY:
        kernel = _filter_kernel.warmup(**meta, **opts, grid=(1,))
        kernel._init_handles()
        props = torch.cuda.get_device_properties(key[0])
        threads = 32 * WARPS
        regs = getattr(props, "regs_per_multiprocessor", 65536)
        smem = getattr(props, "shared_memory_per_multiprocessor", 228 * 1024)
        by_regs = regs // (max(kernel.n_regs, 1) * threads)
        by_smem = smem // (kernel.metadata.shared + 1024)
        by_threads = props.max_threads_per_multi_processor // threads
        fit = max(1, min(by_regs, by_smem, by_threads))
        _CAPACITY[key] = fit * props.multi_processor_count
    return _CAPACITY[key]


_BUFFERS: dict[tuple[torch.device, int], dict[str, torch.Tensor]] = {}
_SYNC: dict[torch.device, torch.Tensor] = {}


def buffers(device: torch.device, rows: int, vocab: int) -> dict[str, torch.Tensor]:
    """Per-row workspace for at least rows rows, grown to a power of 2. Growing it
    drops the cached launches, which hold the old one."""
    key = (device, vocab)
    if key not in _BUFFERS or _BUFFERS[key]["state"].numel() < rows:
        _BUFFERS[key] = allocate(device, triton.next_power_of_2(rows), vocab)
        _PLANS.clear()
    return _BUFFERS[key]


def allocate(device: torch.device, rows: int, vocab: int) -> dict[str, torch.Tensor]:
    """Level 1 and 2 maxima, (max, sum of exp) per split and per row, row states, cut
    keys, kept keys, histograms, per-row gather scratch, windows, gather counts and
    masses, and the grid barrier (barrier count, exit count, histogram rows), which
    is zeroed once and reset by each call."""
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
        window=empty(rows * 4, dtype=torch.int32),
        found=empty(rows, dtype=torch.int32),
        more=empty(rows * 2, dtype=torch.float64),
        sync=_SYNC[device],
    )


def reset_buffer_cache() -> None:
    _BUFFERS.clear()
    _PLANS.clear()


@triton.jit(do_not_specialize=LEVELS_INTS)
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
    k_ptr,
    p_ptr,
    has_k,
    has_p,
    splits,
    MAX_SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    k, p = _row_params(k_ptr, p_ptr, row, vocab, has_k, has_p)
    if (k >= vocab) & (p >= 1.0):
        return
    row_ptr = logits_ptr + row.to(tl.int64) * stride
    level1_row = level1_ptr + row.to(tl.int64) * groups
    level2_row = level2_ptr + row.to(tl.int64) * supergroups
    with_sum = (p < 1.0) & (k >= vocab)
    offs = tl.arange(0, BLOCK)
    lo = split * chunk
    hi = tl.minimum(lo + chunk, vocab)
    row_max = -3.0e38
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
    tl.store(stats_ptr + 2 * program, tl.cast(row_max, tl.float64))
    tl.store(stats_ptr + 2 * program + 1, row_sum)


@triton.jit(do_not_specialize=FILTER_INTS)
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
    k_ptr,
    p_ptr,
    fchunk,
    has_k,
    has_p,
    splits,
    MAX_SPLITS: tl.constexpr,
    fsplits,
    HIST_ITEMS: tl.constexpr,
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
            scratch_ptr,
            sync_ptr,
            row,
            vocab,
            groups,
            supergroups,
            k_ptr,
            p_ptr,
            has_k,
            has_p,
            splits,
            MAX_SPLITS,
            SUPERGROUPS_P2,
            BINS,
            KEPT,
            WINDOW,
        )
    _grid_sync(sync_ptr, programs)
    hist_rows = tl.load(sync_ptr + 2, volatile=True)  # rows on the histogram path
    if hist_rows != 0:
        # Split those rows until there are about HIST_ITEMS items per program.
        hsplits = tl.minimum(MAX_SPLITS, tl.cdiv(HIST_ITEMS * programs, hist_rows))
        hchunk = tl.cdiv(tl.cdiv(vocab, hsplits), BLOCK) * BLOCK
        for item in range(pid, rows * hsplits, programs):
            row = item // hsplits
            if tl.load(state_ptr + row, volatile=True) == HIST_C:
                _histogram(
                    logits_ptr + row.to(tl.int64) * stride,
                    hist_ptr + row.to(tl.int64) * BINS,
                    tl.load(row_stats_ptr + 2 * row, volatile=True).to(tl.float32),
                    item % hsplits * hchunk,
                    vocab,
                    hchunk,
                    BINS,
                    BIN_SCALE,
                    BLOCK,
                )
        _grid_sync(sync_ptr, 2 * programs)
        # Rows whose cut bins hold at most WINDOW logits: all programs mask and gather
        # them, then one program per row sorts and cuts. Others: one program per row.
        for row in range(pid, rows, programs):
            if tl.load(state_ptr + row, volatile=True) == HIST_C:
                k, p = _row_params(k_ptr, p_ptr, row, vocab, has_k, has_p)
                _plan_row(
                    hist_ptr + tl.cast(row, tl.int64) * BINS,
                    state_ptr + row,
                    window_ptr + 4 * row,
                    found_ptr + row,
                    more_ptr + 2 * row,
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
        for item in range(pid, rows * hsplits, programs):
            row = item // hsplits
            if tl.load(state_ptr + row, volatile=True) == GATHER_C:
                k, p = _row_params(k_ptr, p_ptr, row, vocab, has_k, has_p)
                _gather_part(
                    logits_ptr + row.to(tl.int64) * stride,
                    scratch_ptr + row.to(tl.int64) * WINDOW,
                    found_ptr + row,
                    more_ptr + 2 * row,
                    tl.load(row_stats_ptr + 2 * row, volatile=True).to(tl.float32),
                    window_ptr + 4 * row,
                    item % hsplits * hchunk,
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
                k, p = _row_params(k_ptr, p_ptr, row, vocab, has_k, has_p)
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
                    more_ptr + 2 * row,
                    window_ptr + 4 * row,
                    BINS,
                    BIN_SCALE,
                    WINDOW,
                    BLOCK,
                )
        _grid_sync(sync_ptr, 5 * programs)
    for item in range(pid, rows * fsplits, programs):
        row = item // fsplits
        _fill(
            logits_ptr + row.to(tl.int64) * stride,
            tl.load(state_ptr + row, volatile=True),
            tl.load(cut_ptr + row, volatile=True),
            kept_ptr + row.to(tl.int64) * KEPT,
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
    scratch_ptr,
    sync_ptr,
    row,
    vocab,
    groups,
    supergroups,
    k_ptr,
    p_ptr,
    has_k,
    has_p,
    splits,
    MAX_SPLITS: tl.constexpr,
    SUPERGROUPS_P2: tl.constexpr,
    BINS: tl.constexpr,
    KEPT: tl.constexpr,
    WINDOW: tl.constexpr,
):
    k, p = _row_params(k_ptr, p_ptr, row, vocab, has_k, has_p)
    no_k = k >= vocab
    done = no_k & (p >= 1.0)
    kept = -1
    if not done:
        row_max, row_sum = _row_stats(stats_ptr, row, splits, MAX_SPLITS)
        tl.store(row_stats_ptr + 2 * row, row_max.to(tl.float64))
        tl.store(row_stats_ptr + 2 * row + 1, row_sum)
        args = (
            logits_ptr + row.to(tl.int64) * stride,
            level1_ptr + row.to(tl.int64) * groups,
            kept_ptr + row.to(tl.int64) * KEPT,
            level2_ptr + row.to(tl.int64) * supergroups,
            scratch_ptr + row.to(tl.int64) * WINDOW,
            supergroups,
            k,
            p,
            row_max,
            row_sum,
            vocab,
            groups,
        )
        # Top-p only rows try one tier: each tier's code is fetched cold, and a second
        # one costs about as much as the fallback saves.
        if k <= 32:
            done, kept = _tier(*args, 32, SUPERGROUPS_P2)
        if (not done) & (k <= 64):
            done, kept = _tier(*args, 64, SUPERGROUPS_P2)
        if (not done) & (no_k | (k <= 128)):
            done, kept = _tier(*args, 128, SUPERGROUPS_P2)
        if (not done) & (k <= 256):
            done, kept = _tier(*args, 256, SUPERGROUPS_P2)
        if not done:
            zeros = tl.zeros([BINS], tl.int32)
            tl.store(hist_ptr + row.to(tl.int64) * BINS + tl.arange(0, BINS), zeros)
            tl.atomic_add(sync_ptr + 2, 1, sem="relaxed", scope="gpu")
    tl.store(
        state_ptr + row,
        tl.where(done, kept, tl.where(no_k & (p >= 1.0), NOOP_C, HIST_C)),
    )


@triton.jit(noinline=True)  # a call instead of 4 inlined copies: compiles much faster
def _tier(
    row_ptr,
    level1_row,
    kept_row,
    level2_row,
    scratch,
    supergroups,
    k,
    p,
    row_max,
    row_sum,
    vocab,
    groups,
    C: tl.constexpr,
    SUPERGROUPS_P2: tl.constexpr,
):
    """Store the kept keys if the row's best C logits decide the cutoff, return whether
    they did and how many are kept.
    Each level (supergroup maxima, group maxima, logits) picks the at most C items in
    its best bins that hold at least need: they hold the best need items of the next
    level. Histograms and one sort of C keep the code small (it runs cold, once per
    row)."""
    use_k = k < vocab
    use_p = p < 1.0
    need = tl.where(use_k, k, C // 2)  # top-p only: room for crowded bins at the cut
    s = tl.arange(0, SUPERGROUPS_P2)
    in_row = s < supergroups
    level2 = tl.load(level2_row + s, mask=in_row, other=-float("inf"), volatile=True)
    ok, top2, count2 = _pick(level2, s, in_row, need, row_max, scratch, C)
    rank = tl.arange(0, C)
    group = tl.reshape(top2[:, None] * 16 + tl.arange(0, 16)[None, :], [C * 16])
    in_row = tl.reshape(
        (rank < count2)[:, None] & (tl.arange(0, 16) >= 0)[None, :], [C * 16]
    ) & (group < groups)
    level1 = tl.load(
        level1_row + group, mask=in_row, other=-float("inf"), volatile=True
    )
    ok1, top1, count1 = _pick(level1, group, in_row, need, row_max, scratch + C, C)
    index = tl.reshape(top1[:, None] * 16 + tl.arange(0, 16)[None, :], [C * 16])
    in_row = tl.reshape(
        (rank < count1)[:, None] & (tl.arange(0, 16) >= 0)[None, :], [C * 16]
    ) & (index < vocab)
    x = tl.load(row_ptr + index, mask=in_row, other=-float("inf"))
    ok0, top0, count0 = _pick(x, index, in_row, need, row_max, scratch + 2 * C, C)
    ok = ok & ok1 & ok0
    valid = rank < count0
    xs = tl.load(row_ptr + top0, mask=valid, other=-float("inf"))
    # Every logit in the picked bins, in exact order (logit descending, then index).
    top = tl.sort(tl.where(valid, _key(xs, top0), 0x7FFFFFFFFFFFFFFF))
    value = _key_value(top)
    keep = valid & (rank < k)
    if use_p:
        best = tl.max(tl.where(valid, value, -float("inf")))
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
def _pick(values, items, mask, need, row_max, slots, C: tl.constexpr):
    """The items in the best bins (1/32 nat below the row max, the last bin refined over
    its own value range if crowded) that hold at least need of them, if at most C:
    whether they fit, the items (compacted through slots) and their count."""
    j = tl.arange(0, 512)
    b = _bin(values, row_max, 32.0, 512)
    hist = tl.histogram(b, 512, mask=mask)
    cum = tl.cumsum(hist, 0)
    last = tl.min(tl.where(cum >= need, j, 511))
    count = tl.sum(tl.where(j == last, cum, 0))
    take = mask & (b <= last)
    if count > C:  # crowded bin (or the open last one): refine it
        above = count - tl.sum(tl.where(j == last, hist, 0))
        inside = mask & (b == last)
        top = tl.max(tl.where(inside, values, -float("inf")))
        low = tl.min(tl.where(inside & (values > -float("inf")), values, float("inf")))
        scale = 511.0 / tl.maximum(top - low, 1e-30)
        b2 = tl.minimum((top - values) * scale, 511.0).to(tl.int32)
        cum2 = tl.cumsum(tl.histogram(b2, 512, mask=inside), 0)
        last2 = tl.min(tl.where(above + cum2 >= need, j, 511))
        count = above + tl.sum(tl.where(j == last2, cum2, 0))
        take = mask & ((b < last) | (inside & (b2 <= last2)))
    pos = tl.cumsum(take.to(tl.int32), 0) - 1
    tl.store(slots + pos, items.to(tl.int64), mask=take & (pos < C))
    tl.debug_barrier()
    rank = tl.arange(0, C)
    picked = tl.load(slots + rank, mask=rank < count, other=0).to(tl.int32)
    tl.debug_barrier()
    return (count >= need) & (count <= C), picked, count


@triton.jit
def _bin(x, top, scale, BINS: tl.constexpr):
    """Histogram bin of a logit: bins of 1/scale nats below top, the last bin holds
    everything further down (and -inf)."""
    return tl.minimum((top - x) * scale, BINS - 1.0).to(tl.int32)


@triton.jit(noinline=True)
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
    x_next = tl.load(row_ptr + lo + offs, mask=lo + offs < hi, other=-float("inf"))
    for start in range(lo, hi, BLOCK):
        idx = start + offs
        x = x_next
        x_next = tl.load(
            row_ptr + idx + BLOCK, mask=idx + BLOCK < hi, other=-float("inf")
        )
        counts += tl.histogram(_bin(x, row_max, BIN_SCALE, BINS), BINS, mask=idx < hi)
    tl.atomic_add(
        hist_row + tl.arange(0, BINS),
        counts,
        mask=counts > 0,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit(noinline=True)
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
    more_row,
    window_row,
    BINS: tl.constexpr,
    BIN_SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """A gathered row: the exact cut among its window, which masks the row. Else one
    program cuts the row from the histogram."""
    if gathered:
        args = (
            row_ptr,
            scratch,
            found,
            tl.load(more_row, volatile=True),
            tl.load(more_row + 1, volatile=True),
            tl.load(window_row + 3, volatile=True),
            k,
            p,
            row_max,
            row_sum,
            vocab,
        )
        if found <= 512:
            _cut_window(*args, 512, WINDOW)
        elif found <= 1024:
            _cut_window(*args, 1024, WINDOW)
        elif found <= 2048:
            _cut_window(*args, 2048, WINDOW)
        else:
            _cut_window(*args, WINDOW, WINDOW)
        tl.store(state_row, SELF_C)
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


@triton.jit(noinline=True)
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
    """From the histogram: bin kb holds the k-th logit, bins lo to hi the top-p cut
    (mass bounds from the bin edges, the top-k mass bounded through bin kb). If the
    window and bin kb hold at most WINDOW logits, the row is gathered by every program.
    Two levels: sums over blocks of F bins, then the bins of the one block that holds
    each cut."""
    F: tl.constexpr = 128
    NC: tl.constexpr = BINS // F
    use_k = k < vocab
    use_p = p < 1.0
    c = tl.arange(0, NC)
    j = c[:, None] * F + tl.arange(0, F)[None, :]
    counts = tl.load(hist_row + j, volatile=True)
    per_block = tl.sum(counts, 1)
    before = tl.cumsum(per_block, 0) - per_block  # logits in the blocks above
    total = tl.sum(per_block)
    kb = BINS  # no k cut
    if use_k:
        block = tl.min(tl.where(before + per_block >= k, c, NC - 1))
        kb = _find(
            hist_row,
            block,
            tl.sum(tl.where(c == block, before, 0)).to(tl.float32),
            k.to(tl.float32),
            0,
            F,
            BINS,
            BIN_SCALE,
        )
    lo = tl.minimum(kb, BINS - 1)
    hi = lo
    if use_p:
        upper = tl.sum(_weights(counts, j, 1, BINS, BIN_SCALE), 1)
        lower = tl.sum(_weights(counts, j, 2, BINS, BIN_SCALE), 1)
        total_lo = row_sum.to(tl.float32)
        total_hi = total_lo
        if use_k:  # the top-k mass: bins above kb, and the best of bin kb
            need = (k - _above(hist_row, before, total, kb, F, NC, BINS)).to(tl.float32)
            least = tl.where(kb == BINS - 1, 0.0, tl.exp(-(kb + 1.0) / BIN_SCALE))
            total_lo = (
                _mass_above(hist_row, lower, kb, 2, F, NC, BINS, BIN_SCALE)
                + need * least
            )
            total_hi = _mass_above(
                hist_row, upper, kb, 1, F, NC, BINS, BIN_SCALE
            ) + need * tl.exp(-kb / BIN_SCALE)
        lo = _reach(
            hist_row, upper, p * total_lo * (1 - 1e-4), 1, F, NC, BINS, BIN_SCALE
        )
        hi = tl.minimum(
            _reach(
                hist_row, lower, p * total_hi * (1 + 1e-4), 2, F, NC, BINS, BIN_SCALE
            ),
            kb,
        )
        lo = tl.minimum(lo, hi)
    apart = use_k & (kb > hi)
    above_lo = _above(hist_row, before, total, lo, F, NC, BINS)
    after_hi = _above(hist_row, before, total, hi + 1, F, NC, BINS)
    at_kb = _above(hist_row, before, total, kb, F, NC, BINS)
    in_kb = _above(hist_row, before, total, kb + 1, F, NC, BINS) - at_kb
    n = after_hi - above_lo + tl.where(apart, in_kb, 0)
    between = tl.where(apart, at_kb - after_hi, 0)
    if n <= WINDOW:
        tl.store(window_row, lo)
        tl.store(window_row + 1, hi)
        tl.store(window_row + 2, kb)
        # Position of the k-th logit among the gathered ones, in key order.
        tl.store(window_row + 3, tl.minimum(k - 1 - above_lo - between, WINDOW))
        tl.store(found_row, 0)
        tl.store(more_row, row_sum * 0.0)
        tl.store(more_row + 1, row_sum * 0.0)
        tl.store(state_row, GATHER_C)


@triton.jit
def _weights(
    counts, j, KIND: tl.constexpr, BINS: tl.constexpr, BIN_SCALE: tl.constexpr
):
    """Per bin: the count (KIND 0), or bounds of its mass relative to the row max, from
    above (1) or below (2, the open last bin unbounded below)."""
    c = counts.to(tl.float32)
    if KIND == 1:
        c = c * tl.exp(-j / BIN_SCALE)
    if KIND == 2:
        c = tl.where(j == BINS - 1, 0.0, c * tl.exp(-(j + 1.0) / BIN_SCALE))
    return c


@triton.jit
def _find(
    hist_row,
    block,
    before,
    target,
    KIND: tl.constexpr,
    F: tl.constexpr,
    BINS: tl.constexpr,
    BIN_SCALE: tl.constexpr,
):
    """The first bin of the block where the running weight, from before, reaches target
    (else the block's last bin)."""
    j = block * F + tl.arange(0, F)
    w = _weights(tl.load(hist_row + j, volatile=True), j, KIND, BINS, BIN_SCALE)
    return tl.min(tl.where(before + tl.cumsum(w, 0) >= target, j, block * F + F - 1))


@triton.jit
def _reach(
    hist_row,
    per_block,
    target,
    KIND: tl.constexpr,
    F: tl.constexpr,
    NC: tl.constexpr,
    BINS: tl.constexpr,
    BIN_SCALE: tl.constexpr,
):
    """The first bin where the running weight reaches target (else the last bin)."""
    c = tl.arange(0, NC)
    before = tl.cumsum(per_block, 0) - per_block
    block = tl.min(tl.where(before + per_block >= target, c, NC - 1))
    return _find(
        hist_row,
        block,
        tl.sum(tl.where(c == block, before, 0.0)),
        target,
        KIND,
        F,
        BINS,
        BIN_SCALE,
    )


@triton.jit
def _above(
    hist_row, before, total, at, F: tl.constexpr, NC: tl.constexpr, BINS: tl.constexpr
):
    """Logits in the bins before bin at (at most BINS)."""
    block = tl.minimum(at, BINS - 1) // F
    f = tl.arange(0, F)
    fine = tl.load(
        hist_row + block * F + f, mask=f < at - block * F, other=0, volatile=True
    )
    inside = tl.sum(tl.where(tl.arange(0, NC) == block, before, 0)) + tl.sum(fine)
    return tl.where(at >= BINS, total, inside)


@triton.jit
def _mass_above(
    hist_row,
    per_block,
    at,
    KIND: tl.constexpr,
    F: tl.constexpr,
    NC: tl.constexpr,
    BINS: tl.constexpr,
    BIN_SCALE: tl.constexpr,
):
    """Bound of the mass in the bins before bin at."""
    block = tl.minimum(at, BINS - 1) // F
    j = block * F + tl.arange(0, F)
    w = _weights(tl.load(hist_row + j, volatile=True), j, KIND, BINS, BIN_SCALE)
    inside = tl.sum(tl.where(tl.arange(0, NC) < block, per_block, 0.0)) + tl.sum(
        tl.where(j < at, w, 0.0)
    )
    return tl.where(at >= BINS, tl.sum(per_block), inside)


@triton.jit(noinline=True)
def _gather_part(
    row_ptr,
    scratch,
    found_row,
    more_row,
    row_max,
    window_row,
    start_at,
    vocab,
    chunk,
    BINS: tl.constexpr,
    BIN_SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One split's share of the window pass: -inf over the bins below the window, the
    keys of the window and of bin kb gathered, and the mass of the bins above the window
    and of those between it and bin kb."""
    lo = tl.load(window_row, volatile=True)
    hi = tl.load(window_row + 1, volatile=True)
    kb = tl.load(window_row + 2, volatile=True)
    offs = tl.arange(0, BLOCK)
    end = tl.minimum(start_at + chunk, vocab)
    above = tl.zeros([BLOCK], tl.float32)
    between = tl.zeros([BLOCK], tl.float32)
    # The next block is loaded before this one is processed: its layout conversions sync
    # the warps, which would otherwise expose the load latency once per block.
    x_next = tl.load(
        row_ptr + start_at + offs, mask=start_at + offs < end, other=-float("inf")
    )
    for start in range(start_at, end, BLOCK):
        idx = start + offs
        valid = idx < end
        x = x_next
        x_next = tl.load(
            row_ptr + idx + BLOCK, mask=idx + BLOCK < end, other=-float("inf")
        )
        b = _bin(x, row_max, BIN_SCALE, BINS)
        weight = tl.exp(x - row_max)
        above += tl.where(valid & (b < lo), weight, 0.0)
        between += tl.where(valid & (b > hi) & (b < kb), weight, 0.0)
        hit = valid & (((b >= lo) & (b <= hi)) | (b == kb))
        # One slot per hit: no block-wide scan, so warps never wait on each other.
        pos = tl.atomic_add(
            found_row + offs * 0, 1, mask=hit, sem="relaxed", scope="gpu"
        )
        # 32-bit logit and index: the layout conversions of a 64-bit key sync the warps.
        tl.store(
            scratch.to(tl.pointer_type(tl.float32)) + pos, x, mask=hit & (pos < WINDOW)
        )
        tl.store(
            scratch.to(tl.pointer_type(tl.int32)) + WINDOW + pos,
            idx,
            mask=hit & (pos < WINDOW),
        )
        tl.store(
            row_ptr + idx, -float("inf"), mask=valid & (b > hi) & (x > -float("inf"))
        )
    tl.atomic_add(more_row, tl.sum(above).to(tl.float64), sem="relaxed", scope="gpu")
    tl.atomic_add(
        more_row + 1, tl.sum(between).to(tl.float64), sem="relaxed", scope="gpu"
    )


@triton.jit(noinline=True)  # one sort per size
def _cut_window(
    row_ptr,
    scratch,
    found,
    above,
    between,
    k_at,
    k,
    p,
    row_max,
    row_sum,
    vocab,
    N: tl.constexpr,
    WINDOW: tl.constexpr,
):
    """Sort the gathered keys (the window, then bin kb), keep those up to the k-th and
    up to where the mass from the top reaches p of the kept mass, -inf over the rest."""
    use_k = k < vocab
    use_p = p < 1.0
    rank = tl.arange(0, N)
    valid = rank < found
    # Volatile: other programs gathered these, and this program's L1 may hold stale
    # lines of the same scratch (its tier's slots).
    x = tl.load(
        scratch.to(tl.pointer_type(tl.float32)) + rank,
        mask=valid,
        other=0.0,
        volatile=True,
    )
    idx = tl.load(
        scratch.to(tl.pointer_type(tl.int32)) + WINDOW + rank,
        mask=valid,
        other=0,
        volatile=True,
    )
    key = tl.sort(tl.where(valid, _key(x, idx), 0x7FFFFFFFFFFFFFFF))
    last = tl.where(use_k, k_at, N)
    if use_p:
        weight = tl.where(valid, tl.exp(_key_value(key) - row_max), 0.0).to(tl.float64)
        total = row_sum
        if use_k:
            total = above + between + tl.sum(tl.where(rank <= k_at, weight, 0.0))
        reached = valid & (above + tl.cumsum(weight, 0) >= p.to(tl.float64) * total)
        last = tl.minimum(last, tl.min(tl.where(reached, rank, N)))
    index = (key & 0xFFFFFFFF).to(tl.int32)
    tl.store(row_ptr + index, -float("inf"), mask=valid & (rank > last))


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
            ok, cut = _finish(
                scratch,
                found,
                target,
                above + more,
                row_max,
                (vocab * 0 + MASS) > 0,
                WINDOW,
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


@triton.jit(noinline=True)  # a call: one copy of the sort (slow path only)
def _finish(scratch, found, target, above, row_max, mass, N: tl.constexpr):
    """Sort the gathered keys, return the key where the running count (or mass) from
    above reaches target."""
    rank = tl.arange(0, N)
    key = tl.sort(tl.load(scratch + rank, mask=rank < found, other=0x7FFFFFFFFFFFFFFF))
    weight = tl.where(
        rank < found, tl.where(mass, tl.exp(_key_value(key) - row_max), 1.0), 0.0
    )
    reached = above + tl.cumsum(weight.to(tl.float64), 0) >= target
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


@triton.jit(noinline=True)
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
def _row_params(k_ptr, p_ptr, row, vocab, has_k, has_p):
    k = vocab
    p = vocab.to(tl.float32) * 0.0 + 1.0
    if has_k:
        k = tl.load(k_ptr + row).to(tl.int32)
    if has_p:
        p = tl.load(p_ptr + row).to(tl.float32)
    return k, p


@triton.jit
def _row_stats(stats_ptr, row, splits, MAX_SPLITS: tl.constexpr):
    """The row's max and sum of exp(logit - max), from the per-split partials."""
    split = tl.arange(0, MAX_SPLITS)
    program = row * MAX_SPLITS + split
    split_max = tl.load(stats_ptr + 2 * program, mask=split < splits, other=-3.0e38).to(
        tl.float32
    )
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
