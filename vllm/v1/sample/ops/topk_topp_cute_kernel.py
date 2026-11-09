# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL port of the core path of csrc/topk_topp_filter.cu.

In-place top-k/top-p masking of FP32 logits with per-row device k and p. Each
row is handled by one CTA (large batches, persistent over rows) or by a thread
block cluster of up to 16 CTAs that split the row (small batches):
  1. stream the row chunk once from HBM through a TMA ring in shared memory:
     optional sum-exp, and gather the values above a sampled threshold into a
     candidate store (shared memory, then a per-CTA overflow region in L2).
  2. exact radix selection over the candidates, with histograms and scalars
     combined across the cluster: top-k rank, then top-p fixed-point mass,
     ties broken by lowest index.
  3. write -inf over the chunk and scatter the kept candidates back, or mask
     the chunk in place when many are kept.
Rows the candidates cannot answer (store overflow, too few candidates for k,
a nucleus below the candidates) use an exact radix selection over the row,
whose cost does not depend on where the cut is (as in AIR top-p): rounds read
the row until the boundary bucket fits the store, one more read gathers it,
and the selection finishes on the gathered bucket.

Semantics: exactly k tokens by logit, ties broken by lowest index, then top-p
on the renormalized top-k distribution, keeping the smallest prefix of the
(logit desc, index asc) order whose mass reaches p. Masses are FP32 exp()
rounded to 2^-44 fixed point, so sums are deterministic.
"""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Float64, Int32, Int64, Uint32, Uint64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

THREADS = 256
WARPS = THREADS // 32
STAGE = 4096  # elements per streaming round
VALS = STAGE // THREADS  # values per thread per round, four contiguous per vector
MAX_STAGES = 10
SHALLOW = 3  # ring stages when two CTAs share an SM
MAX_SPLITS = 16
CAP = 6912  # candidates in shared memory per CTA
GCAP = 32768  # overflow candidates in global memory per CTA
STORE = CAP + GCAP
MAX_CAND_K = STORE * 5 // 8
TARGET_P = CAP * 3 // 4  # expected candidates of top-p-only rows
KEY_NEGINF = 0x007FFFFF
ALL_IDX = 0xFFFFFFFF
MAX_VOCAB = 1 << 20
MASS_SCALE = float(1 << 44)
NEG_INF = float("-inf")
# Shared scalars.
TOTAL, PREFIX, COUNT_AT, FOUND, XTURN = range(5)
# A fixed-point mass (< 2^64) in 12-bit pieces: a bin sums at most 2^20 of each.
LIMB_BITS, LIMBS = 12, 4


# ---------------------------------------------------------------------------
# Kernels.


@cute.jit
def launch(
    x: cute.Tensor,
    ks: cute.Tensor,
    ps: cute.Tensor,
    ws: cute.Tensor,
    flags: Int32,
    grid: Int32,
    splits: Int32,
    depth: Int32,
    cluster: cutlass.Constexpr[bool],
    stream,
):
    smem = RING_OFFSET + depth * (STAGE * 4)
    if const_expr(cluster):
        cluster_kernel(x, ks, ps, ws, flags, depth).launch(
            grid=[grid, 1, 1],
            block=[THREADS, 1, 1],
            cluster=[splits, 1, 1],
            smem=smem,
            stream=stream,
            min_blocks_per_mp=2,
            preferred_smem_carveout=100,  # else 135 KiB for runtime smem sizes
        )
    else:
        row_kernel(x, ks, ps, ws, flags, depth).launch(
            grid=[grid, 1, 1],
            block=[THREADS, 1, 1],
            smem=smem,
            stream=stream,
            min_blocks_per_mp=2,
            preferred_smem_carveout=100,  # else 135 KiB for runtime smem sizes
        )


@cute.kernel
def row_kernel(
    x: cute.Tensor,
    ks: cute.Tensor,
    ps: cute.Tensor,
    ws: cute.Tensor,
    flags: Int32,
    depth: Int32,
):
    """One CTA per row, persistent over rows."""
    batch, vocab = x.shape
    a = make_launch(x, ks, ps, ws, flags, vocab)
    s = make_smem(depth, a.store)
    s.init()
    row = next_active(a, Int32(cute.arch.block_idx()[0]), batch)
    while row < batch:
        process_row(s, a, row, Int32(0), vocab, min(STAGE, vocab), False)
        cute.arch.sync_threads()
        row = next_active(a, row + cute.arch.grid_dim()[0], batch)


@cute.kernel
def cluster_kernel(
    x: cute.Tensor,
    ks: cute.Tensor,
    ps: cute.Tensor,
    ws: cute.Tensor,
    flags: Int32,
    depth: Int32,
):
    """Clusters of CTAs that split one row each, persistent over rows."""
    batch, vocab = x.shape
    a = make_launch(x, ks, ps, ws, flags, vocab)
    s = make_smem(depth, a.store)
    s.init()
    cluster_sync()
    splits = Int32(cute.arch.cluster_size())
    rank = Int32(cute.arch.block_idx_in_cluster())
    chunk = ((vocab + splits - 1) // splits + 3) // 4 * 4
    begin = min(vocab, rank * chunk)
    end = min(vocab, begin + chunk)
    sample_size = Int32(0)
    for r in range(splits):
        sample_size += min(STAGE, max(0, min(vocab, (r + 1) * chunk) - r * chunk))
    clusters = cute.arch.grid_dim()[0] // splits
    for row in range(cute.arch.block_idx()[0] // splits, batch, clusters):
        if row_active(a, row):
            process_row(s, a, row, begin, end, sample_size, True)
        # Peers finish reading this CTA's shared memory before it is reused.
        cluster_sync()


# ---------------------------------------------------------------------------
# Row processing.


@cute.jit
def process_row(
    s,
    a,
    row: Int32,
    begin: Int32,
    end: Int32,
    sample_size: Int32,
    cl: cutlass.Constexpr[bool],
):
    use_k, kk, use_p, p = row_params(a, row)
    vocab = a.vocab
    xr = make_row(a.x, row, begin, end, a.vec)
    gather = (not use_k) or kk <= MAX_CAND_K
    need_sum = use_p and not use_k
    target = Uint32(TARGET_P)
    if use_k:
        target = max(
            min(Uint32(CAP // 2), max(Uint32(1024), 8 * kk)),
            min(Uint32(MAX_CAND_K), 2 * kk),
        )
    maximum, total, tk, n_row, over = (
        Float32(NEG_INF),
        Float64(0.0),
        Uint32(KEY_NEGINF),
        Uint32(0),
        Boolean(True),
    )
    if gather:
        maximum, total, tk, n_row, over = stream(
            s, xr, vocab, sample_size, target, need_sum, vocab > STAGE, cl
        )
    # The candidates hold the row's answer: no overflow, and at least k of them
    # unless they are every finite value.
    fallback = over or (use_k and n_row < kk and tk > KEY_NEGINF)
    cut_key, cut_idx = Uint32(0), Uint32(ALL_IDX)
    write = Int32(0)  # 1: fill and scatter the candidates, 2: mask the chunk
    cands = Candidates(s, min(s.sc[TOTAL], Uint32(STORE)))
    if not fallback and maximum != NEG_INF:
        exact_k = (not use_k) or n_row <= kk
        # Without top-p, a complete candidate set needs no masking at all.
        if use_p or tk != KEY_NEGINF or not exact_k:
            prefix, start = common_start(tk + 1, ordered(maximum))
            if not exact_k:
                cut_key, cut_idx = select(
                    s,
                    cands,
                    cut_key,
                    cut_idx,
                    Uint64(kk),
                    prefix,
                    start,
                    Float32(0.0),
                    vocab,
                    False,
                    cl,
                )
            if use_p:
                mass = total * MASS_SCALE
                if use_k or tk == KEY_NEGINF:
                    mass = Float64(set_mass(s, cands, cut_key, cut_idx, maximum, cl))
                goal = mass_target(p, mass)
                # The nucleus may extend below the candidates of top-p-only rows.
                if not use_k and tk != KEY_NEGINF:
                    fallback = set_mass(s, cands, cut_key, cut_idx, maximum, cl) < goal
                if not fallback:
                    cut_key, cut_idx = select(
                        s,
                        cands,
                        cut_key,
                        cut_idx,
                        goal,
                        prefix,
                        start,
                        maximum,
                        vocab,
                        True,
                        cl,
                    )
            write = 1
            # Large kept sets: a masked read-modify-write beats writing twice.
            factor = Uint32(4)
            if const_expr(not cl):  # noqa: SIM102
                if vocab >= 16 * STAGE:
                    factor = Uint32(8)
            if factor * n_row > Uint32(vocab):
                write = 2
    if fallback:
        if not need_sum:
            maximum = row_max(s, xr, cl)
        if maximum != NEG_INF:
            cut_key, cut_idx = global_path(
                s, xr, use_k, kk, use_p, p, total, maximum, vocab, cl
            )
            write = 2
    cute.arch.sync_threads()
    if write == 2:
        mask_row(xr, cut_key, cut_idx)
    if write == 1:
        fill_and_scatter(xr, cands, cut_key, cut_idx)


@cute.jit
def global_path(
    s,
    xr,
    use_k: Boolean,
    kk: Uint32,
    use_p: Boolean,
    p: Float32,
    total: Float64,
    maximum: Float32,
    vocab: Int32,
    cl: cutlass.Constexpr[bool],
):
    """Exact radix selection over the row in global memory."""
    cut_key, cut_idx = Uint32(0), Uint32(ALL_IDX)
    if use_k:
        cut_key, cut_idx = select_row(
            s, xr, cut_key, cut_idx, Uint64(kk), maximum, vocab, False, cl
        )
    if use_p:
        mass = total * MASS_SCALE
        if use_k:
            mass = Float64(set_mass(s, RowSource(xr), cut_key, cut_idx, maximum, cl))
        cut_key, cut_idx = select_row(
            s, xr, cut_key, cut_idx, mass_target(p, mass), maximum, vocab, True, cl
        )
    return cut_key, cut_idx


@cute.jit
def select_row(
    s,
    xr,
    set_key: Uint32,
    set_idx: Uint32,
    target: Uint64,
    maximum: Float32,
    vocab: Int32,
    mass: cutlass.Constexpr[bool],
    cl: cutlass.Constexpr[bool],
):
    """Like select over the row, at a cost independent of where the cut is
    (AIR top-p): radix rounds read the row until the boundary bucket fits
    every CTA's candidate store, then one more read gathers that bucket and
    the selection finishes on it."""
    row = Values(RowSource(xr), set_key, set_idx)
    prefix, start, above = Uint32(0), Int32(32), Uint64(0)
    count_at = Uint32(0)
    found, fits = Boolean(True), Boolean(False)
    while found and not fits and start > 0:
        key, part, count_at, found = radix_select(
            s, row, target - above, prefix, start, maximum, mass, cl, 1
        )
        if found:
            start -= min(8, start)
            prefix = key
            above += part
            local = s.count[Int32((prefix >> Uint32(start)) & Uint32(255))]
            fits = group_reduce(s, local, cutlass.max, cl) <= STORE
    cut_key, cut_idx = set_key, set_idx
    if found:
        if fits:
            gather_row(s, xr, set_key, set_idx, prefix, start)
            cands = Candidates(s, s.sc[TOTAL])
            cut_key, cut_idx = select(
                s,
                cands,
                set_key,
                set_idx,
                target - above,
                prefix,
                start,
                maximum,
                vocab,
                mass,
                cl,
            )
        else:  # more than a store of ties at one key
            cut_key, cut_idx = finish(
                s,
                RowSource(xr),
                set_key,
                set_idx,
                prefix,
                count_at,
                target - above,
                maximum,
                vocab,
                mass,
                cl,
            )
    return cut_key, cut_idx


@cute.jit
def gather_row(s, xr, set_key: Uint32, set_idx: Uint32, prefix: Uint32, start: Int32):
    """Stores the set's elements whose keys share prefix above bit start."""
    cute.arch.sync_threads()
    if cute.arch.thread_idx()[0] == 0:
        put(s.sc, TOTAL, Uint32(0))
    cute.arch.sync_threads()
    fold_row(xr, Gather(s, set_key, set_idx, prefix >> Uint32(start), start), Int32(0))
    cute.arch.sync_threads()


@dataclass(frozen=True)
class Gather:
    s: Any
    set_key: Uint32
    set_idx: Uint32
    high: Uint32
    start: Int32

    @cute.jit
    def __call__(self, y: Float32, i: Int32, acc):
        key = ordered(y)
        if (key >> Uint32(self.start)) == self.high and (
            key > self.set_key or (key == self.set_key and Uint32(i) <= self.set_idx)
        ):
            at = cute.arch.atomic_add(
                self.s.sc.iterator + TOTAL, Uint32(1), scope="cta"
            )
            self.s.store(at, key, Uint32(i))
        return acc


# ---------------------------------------------------------------------------
# Streaming phase.


@cute.jit
def stream(
    s,
    xr,
    vocab: Int32,
    sample_size: Int32,
    target: Uint32,
    need_sum: Boolean,
    sample: Boolean,
    cl: cutlass.Constexpr[bool],
):
    """Streams the chunk once: sum-exp if needed, and the candidates above a
    sampled threshold. Returns (max, sum-exp, threshold key, candidates,
    overflow), combined over the cluster."""
    cute.arch.sync_threads()
    if cute.arch.thread_idx()[0] == 0:
        put(s.sc, TOTAL, Uint32(0))
    cute.arch.sync_threads()
    begin, end = xr.begin, xr.end
    rounds = max(0, end - begin + STAGE - 1) // STAGE
    ring = Ring(s, xr)
    ring.open()
    if xr.vec:
        ring.pump()
    tk = Uint32(KEY_NEGINF)
    # Clusters sample collectively, so every CTA samples, even an empty chunk.
    if sample:
        tk = sample_threshold(s, xr, vocab, sample_size, target, cl)
    tv = from_ordered(tk)
    m, total, comp = Float32(NEG_INF), Float32(0.0), Float32(0.0)
    v = cute.make_rmem_tensor((VALS,), Float32)
    for r in range(rounds):
        base = begin + r * STAGE
        if xr.vec:
            ring.pump()
            ring.consume(base, end, v)
        else:
            load_scalar(xr, base, v)
        if need_sum:
            m, total, comp = sum_exp(v, m, total, comp)
        append(s, v, base, tv)
        cute.arch.sync_threads()  # also frees this round's ring slot
    count = s.sc[TOTAL]
    if not need_sum:
        top = Candidates(s, min(count, Uint32(STORE))).fold(
            lambda key, idxs, i, acc: cutlass.max(acc, key), Uint32(0)
        )
        m = Float32(NEG_INF)
        if top != 0:
            m = from_ordered(top)
    maximum = block_reduce(s, m, cute.arch.fmax)
    part = rescale(Float64(total) + Float64(comp), m, maximum)
    total64 = block_reduce(s, part, add)
    if const_expr(cl):
        n = Int32(cute.arch.cluster_size())
        base_slot = group_gather(
            s, (pack(maximum) | (Uint64(count) << 32), pack(total64))
        )
        peak = Float32(NEG_INF)
        for r in range(n):
            peak = cute.arch.fmax(peak, unpack(s.xin[base_slot + r * 3], Float32))
        total64 = Float64(0.0)
        count = Uint32(0)
        over = Boolean(False)
        for r in range(n):
            w0 = s.xin[base_slot + r * 3]
            c = Uint32(w0 >> 32)
            total64 += rescale(
                unpack(s.xin[base_slot + r * 3 + 1], Float64), unpack(w0, Float32), peak
            )
            count += c
            over = over or c > STORE
        return peak, total64, tk, count, over
    return maximum, total64, tk, count, count > STORE


@cute.jit
def sum_exp(v: cute.Tensor, m: Float32, total: Float32, comp: Float32):
    """Running max and compensated FP32 sum of exp(x - max)."""
    mr = v[0]
    for b in cutlass.range_constexpr(1, VALS):
        mr = cute.arch.fmax(mr, v[b])
    if mr > m:
        scale = Float32(0.0)
        if m != NEG_INF:
            scale = exp(m - mr)
        total *= scale
        comp *= scale
        m = mr
    if m != NEG_INF:
        for b in cutlass.range_constexpr(VALS):
            e = exp(v[b] - m)
            t = total + e
            if total >= e:
                comp += (total - t) + e
            else:
                comp += (e - t) + total
            total = t
    return m, total, comp


@cute.jit
def append(s, v: cute.Tensor, base: Int32, tv: Float32):
    """Appends the values above tv as (ordered key, index), with one shared
    atomic per warp. Positions past the store only count."""
    mask = Uint32(0)
    for b in cutlass.range_constexpr(VALS):
        if v[b] > tv:
            mask |= Uint32(1 << b)
    if cute.arch.vote_any_sync(mask != 0):
        lane = cute.arch.lane_idx()
        n = Uint32(cute.arch.popc(mask))
        incl = warp_inclusive_scan(n)
        wbase = Uint32(0)
        if lane == 31:
            wbase = cute.arch.atomic_add(s.sc.iterator + TOTAL, incl, scope="cta")
        pos = cute.arch.shuffle_sync(wbase, 31) + incl - n
        tid = cute.arch.thread_idx()[0]
        for b in cutlass.range_constexpr(VALS):
            if ((mask >> b) & 1) != 0:
                at = pos + Uint32(cute.arch.popc(mask & Uint32((1 << b) - 1)))
                i = base + ((b // 4) * THREADS + tid) * 4 + b % 4
                s.store(at, ordered(v[b]), Uint32(i))


@cute.jit
def sample_threshold(
    s, xr, vocab: Int32, sample_size: Int32, target: Uint32, cl: cutlass.Constexpr[bool]
):
    """Key threshold with about `target` of the row above it, from up to STAGE
    values strided over each chunk (token order is not random)."""
    v = cute.make_rmem_tensor((VALS,), Float32)
    load_sample(xr, v)
    lo, hi = Uint32(0xFFFFFFFF), Uint32(0)
    for b in cutlass.range_constexpr(VALS):
        key = ordered(v[b])
        if key > KEY_NEGINF:
            lo = min(lo, key)
            hi = max(hi, key)
    lo = block_reduce(s, lo, cutlass.min)
    hi = block_reduce(s, hi, cutlass.max)
    if const_expr(cl):
        base_slot = group_gather(s, (Uint64(lo) | (Uint64(hi) << 32),))
        for r in range(Int32(cute.arch.cluster_size())):
            w = s.xin[base_slot + r * 3]
            lo = min(lo, Uint32(w & 0xFFFFFFFF))
            hi = max(hi, Uint32(w >> 32))
    tk = Uint32(KEY_NEGINF)
    if hi > KEY_NEGINF:
        prefix, start = common_start(lo, hi)
        scaled = (Uint64(target) * Uint64(sample_size) + Uint64(vocab - 1)) // Uint64(
            vocab
        )
        rank = min(Uint64(sample_size), max(Uint64(16), scaled))
        key, _, _, found = radix_select(
            s, SampleSource(v), rank, prefix, start, Float32(0.0), False, cl, 2
        )
        if found:
            tk = max(key - 1, Uint32(KEY_NEGINF))
    return tk


# ---------------------------------------------------------------------------
# Radix selection.


@cute.jit
def select(
    s,
    src,
    set_key: Uint32,
    set_idx: Uint32,
    target: Uint64,
    prefix: Uint32,
    start: Int32,
    maximum: Float32,
    vocab: Int32,
    mass: cutlass.Constexpr[bool],
    cl: cutlass.Constexpr[bool],
):
    """The cut keeping the `target` largest of the set or, by mass, its
    smallest prefix whose fixed-point mass reaches target."""
    key, above, count_at, found = radix_select(
        s, Values(src, set_key, set_idx), target, prefix, start, maximum, mass, cl
    )
    cut_key, cut_idx = set_key, set_idx
    if found:
        cut_key, cut_idx = finish(
            s,
            src,
            set_key,
            set_idx,
            key,
            count_at,
            target - above,
            maximum,
            vocab,
            mass,
            cl,
        )
    return cut_key, cut_idx


@cute.jit
def finish(
    s,
    src,
    set_key: Uint32,
    set_idx: Uint32,
    key: Uint32,
    count_at: Uint32,
    need: Uint64,
    maximum: Float32,
    vocab: Int32,
    mass: cutlass.Constexpr[bool],
    cl: cutlass.Constexpr[bool],
):
    """The cut at the boundary key, given the rank or mass still needed there."""
    rank = need
    if const_expr(mass):
        w = fixed_mass(key, maximum)
        rank = Uint64(count_at)
        if w != 0:
            rank = (need + w - 1) // w
        rank = max(Uint64(1), min(rank, Uint64(count_at)))
    return tie_cut(s, src, set_key, set_idx, key, count_at, rank, vocab, cl)


@cute.jit
def tie_cut(
    s,
    src,
    set_key: Uint32,
    set_idx: Uint32,
    key: Uint32,
    count_at: Uint32,
    rank: Uint64,
    vocab: Int32,
    cl: cutlass.Constexpr[bool],
):
    """Among the ties at key inside the set, keep the `rank` lowest indices."""
    cut_idx = Uint32(ALL_IDX)
    if key == set_key:
        cut_idx = set_idx
    if rank < Uint64(count_at):
        prefix, start = common_start(~Uint32(vocab - 1), Uint32(0xFFFFFFFF))
        tie, _, _, _ = radix_select(
            s,
            Ties(src, set_key, set_idx, key),
            rank,
            prefix,
            start,
            Float32(0.0),
            False,
            cl,
        )
        cut_idx = ~tie
    return key, cut_idx


@cute.jit
def radix_select(
    s,
    src,
    target: Uint64,
    prefix: Uint32,
    start: Int32,
    maximum: Float32,
    mass: cutlass.Constexpr[bool],
    cl: cutlass.Constexpr[bool],
    max_rounds: cutlass.Constexpr[int] = 8,
):
    """Key K with weight(> K) < target <= weight(>= K) over the visited set,
    8 bits per round from bit `start` down. All select keys share `prefix`
    above `start`. Returns (key, weight above, count at key, found)."""
    tid = cute.arch.thread_idx()[0]
    above = Uint64(0)
    rounds = Int32(0)
    found = Boolean(True)
    go = Boolean(True)
    while go:
        width = min(8, start)
        shift = start - width
        nb = Int32(1) << width
        if tid < nb:
            put(s.count, tid, Uint32(0))
            for li in cutlass.range_constexpr(LIMBS):
                put(s.limb, li * 256 + tid, Uint32(0))
        if const_expr(cl):
            open_bins(s, nb)
        cute.arch.sync_threads()
        hist = Histogram(s, prefix, start, shift, nb, maximum, mass)
        hist.flush(src.visit(hist, hist.empty()))
        cute.arch.sync_threads()
        if const_expr(cl):
            push_bins(s, nb, mass)
        if tid < 32:
            scan_bins(s, nb, shift, target, above, prefix, mass, cl)
        cute.arch.sync_threads()
        found = s.sc[FOUND] != 0
        prefix = s.sc[PREFIX]
        above = s.above[0]
        start = shift
        cute.arch.sync_threads()
        if tid == 0:
            put(s.sc, FOUND, Uint32(1))
        rounds += 1
        go = found and start > 0 and rounds < max_rounds
    return prefix, above, s.sc[COUNT_AT], found


@dataclass(frozen=True)
class Histogram:
    """Visitor adding each selected key's digit to the shared bins. Each
    thread merges runs of one digit before its atomics: in the first rounds
    most keys share a few digits, and same-address atomics serialize."""

    s: Any
    prefix: Uint32
    start: Int32
    shift: Int32
    nb: Int32
    maximum: Float32
    mass: bool

    @staticmethod
    def empty():
        return Int32(-1), Uint32(0), Uint64(0)

    @cute.jit
    def __call__(self, key: Uint32, value: Uint32, acc):
        # Shifts by 32 are undefined: start == 32 matches every key.
        matches = self.start >= 32 or (key >> Uint32(self.start & 31)) == (
            self.prefix >> Uint32(self.start & 31)
        )
        if matches:
            d = Int32((key >> Uint32(self.shift)) & Uint32(self.nb - 1))
            w = Uint64(0)
            if const_expr(self.mass):
                w = fixed_mass(value, self.maximum)
            last, n, total = acc
            if d == last:
                n, total = n + 1, total + w
            else:
                self.flush(acc)
                last, n, total = d, Uint32(1), w
            acc = (last, n, total)
        return acc

    @cute.jit
    def flush(self, acc):
        last, n, total = acc
        if n != 0:
            cute.arch.atomic_add(self.s.count.iterator + last, n, scope="cta")
            if const_expr(self.mass):
                for li in cutlass.range_constexpr(LIMBS):
                    piece = total >> (LIMB_BITS * li)
                    if const_expr(li < LIMBS - 1):
                        piece = piece & ((1 << LIMB_BITS) - 1)
                    cute.arch.atomic_add(
                        self.s.limb.iterator + li * 256 + last,
                        Uint32(piece),
                        scope="cta",
                    )


@cute.jit
def scan_bins(
    s,
    nb: Int32,
    shift: Int32,
    target: Uint64,
    above: Uint64,
    prefix: Uint32,
    mass: cutlass.Constexpr[bool],
    cl: cutlass.Constexpr[bool],
):
    """Warp 0 finds the digit bin, from the top, where the cumulative weight
    first reaches target."""
    lane = cute.arch.lane_idx()
    w = cute.make_rmem_tensor((8,), Uint64)
    total = Uint64(0)
    for j in cutlass.range_constexpr(8):
        b = nb - 1 - (lane * 8 + j)
        w[j] = Uint64(0)
        if b >= 0:
            w[j] = s.weight(b, mass, cl)
        total += w[j]
    incl = warp_inclusive_scan(total)
    acc = above + incl - total
    hit = acc < target and target <= above + incl
    if cute.arch.vote_ballot_sync(hit) == 0 and lane == 0:
        put(s.sc, FOUND, Uint32(0))
    if hit:
        done = Boolean(False)
        for j in cutlass.range_constexpr(8):
            b = nb - 1 - (lane * 8 + j)
            if not done and acc + w[j] >= target:
                put(s.sc, PREFIX, prefix | (Uint32(b) << Uint32(shift)))
                put(s.above, 0, acc)
                put(s.sc, COUNT_AT, s.bin_count(b, cl))
                put(s.sc, FOUND, Uint32(1))
                done = Boolean(True)
            acc += w[j]


@cute.jit
def open_bins(s, nb: Int32):
    """Clears this CTA's cluster-combined bins, then arrives: push_bins waits
    until every CTA cleared its bins."""
    tid = cute.arch.thread_idx()[0]
    if tid < nb:
        put(s.gcount, tid, Uint32(0))
        put(s.gwide, tid, Uint64(0))
    cluster_arrive()


@cute.jit
def push_bins(s, nb: Int32, mass: cutlass.Constexpr[bool]):
    """Adds this CTA's nonzero bins into every CTA's combined bins."""
    cluster_wait()
    tid = cute.arch.thread_idx()[0]
    n = Int32(cute.arch.cluster_size())
    rank = Int32(cute.arch.block_idx_in_cluster())
    if tid < nb:
        count = s.count[tid]
        if count != 0:
            weight = s.weight(tid, True, False)
            for i in range(n):
                r = (rank + i) & (n - 1)
                red_add(s.gcount.iterator + tid, r, count)
                if const_expr(mass):
                    red_add(s.gwide.iterator + tid, r, weight)
    cluster_sync()


# ---------------------------------------------------------------------------
# Element sources and visitors. visit(f, acc) folds f(select key, value key,
# acc) over the owned elements.


@dataclass(frozen=True)
class Candidates:
    """This CTA's stored candidates: shared memory, then the overflow store."""

    s: Any
    n: Uint32

    @property
    def g(self):
        return self.s.g

    def values(self, set_key: Uint32, set_idx: Uint32, f, acc):
        return self.fold(InStore(set_key, set_idx, Uint32(0), f, False), acc)

    def ties(self, set_key: Uint32, set_idx: Uint32, tie: Uint32, f, acc):
        return self.fold(InStore(set_key, set_idx, tie, f, True), acc)

    @cute.jit
    def fold(self, g, acc):
        """Folds g(key, index tensor, position, acc) over the shared part,
        then over the overflow store (in L2) with more loads in flight."""
        acc = fold_keys(self.s.key, self.s.idx, min(self.n, Uint32(CAP)), g, acc, 2)
        if self.n > CAP:
            gidx = cute.make_tensor(self.g.iterator + GCAP, cute.make_layout(GCAP))
            acc = fold_keys(self.g, gidx, self.n - CAP, g, acc, 8)
        return acc


@cute.jit
def fold_keys(
    keys: cute.Tensor,
    idxs: cute.Tensor,
    n: Uint32,
    g,
    acc,
    unroll: cutlass.Constexpr[int],
):
    k = cute.make_rmem_tensor((unroll,), Uint32)
    for i0 in range(cute.arch.thread_idx()[0], n, THREADS * unroll):
        for q in cutlass.range_constexpr(unroll):
            k[q] = Uint32(0)
            if i0 + q * THREADS < n:
                k[q] = Uint32(keys[i0 + q * THREADS])
        for q in cutlass.range_constexpr(unroll):
            acc = g(k[q], idxs, i0 + q * THREADS, acc)
    return acc


@dataclass(frozen=True)
class InStore:
    """Store visitor passing the elements inside the set (only those at key
    `tie`, by ~index, for ties) to f. Key 0 is padding: no stored key is 0."""

    set_key: Uint32
    set_idx: Uint32
    tie: Uint32
    f: Any
    ties: bool

    @cute.jit
    def __call__(self, key: Uint32, idxs: cute.Tensor, i: Int32, acc):
        # The index matters only for a tie with the set's boundary key.
        if const_expr(self.ties):
            if key == self.tie and key != 0:
                idx = Uint32(idxs[i])
                if key > self.set_key or idx <= self.set_idx:
                    acc = self.f(~idx, key, acc)
        else:
            if key >= self.set_key and key != 0:
                keep = key > self.set_key
                if not keep:
                    keep = Uint32(idxs[i]) <= self.set_idx
                if keep:
                    acc = self.f(key, key, acc)
        return acc


@dataclass(frozen=True)
class RowSource:
    """The CTA's chunk of the row in global memory."""

    xr: Any

    def values(self, set_key: Uint32, set_idx: Uint32, f, acc):
        return fold_row(self.xr, InSet(set_key, set_idx, Uint32(0), f, False), acc)

    def ties(self, set_key: Uint32, set_idx: Uint32, tie: Uint32, f, acc):
        return fold_row(self.xr, InSet(set_key, set_idx, tie, f, True), acc)


@dataclass(frozen=True)
class InSet:
    """Row visitor passing the elements inside the set (only those at key
    `tie`, by ~index, for ties) to f."""

    set_key: Uint32
    set_idx: Uint32
    tie: Uint32
    f: Any
    ties: bool

    @cute.jit
    def __call__(self, y: Float32, i: Int32, acc):
        key = ordered(y)
        if key > self.set_key or (key == self.set_key and Uint32(i) <= self.set_idx):
            if const_expr(self.ties):
                if key == self.tie:
                    acc = self.f(~Uint32(i), key, acc)
            else:
                acc = self.f(key, key, acc)
        return acc


@cute.jit
def fold_row(xr, g, acc):
    """Folds g(value, index, acc) over the chunk, with four vector loads per
    thread in flight when the row is aligned."""
    tid = cute.arch.thread_idx()[0]
    if xr.vec:
        v = cute.make_rmem_tensor((16,), Float32)
        for i0 in range(xr.begin + tid * 4, xr.end, THREADS * 16):
            for q in cutlass.range_constexpr(4):
                i = i0 + q * THREADS * 4
                a, b, c, d = (Float32(NEG_INF),) * 4
                if i < xr.end:
                    a, b, c, d = ld_global_v4(xr.addr(i))
                v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3] = a, b, c, d
            for q in cutlass.range_constexpr(4):
                i = i0 + q * THREADS * 4
                if i < xr.end:
                    for e in cutlass.range_constexpr(4):
                        acc = g(v[4 * q + e], i + e, acc)
    else:
        for i in range(xr.begin + tid, xr.end, THREADS):
            acc = g(xr.x[xr.row, i], i, acc)
    return acc


@dataclass(frozen=True)
class SampleSource:
    """The thread's sampled values with finite keys."""

    v: cute.Tensor

    @cute.jit
    def visit(self, f, acc):
        for b in cutlass.range_constexpr(VALS):
            key = ordered(self.v[b])
            if key > KEY_NEGINF:
                acc = f(key, key, acc)
        return acc


@dataclass(frozen=True)
class Values:
    """The elements of src inside the set (key desc, index asc) cut."""

    src: Any
    set_key: Uint32
    set_idx: Uint32

    def visit(self, f, acc):
        return self.src.values(self.set_key, self.set_idx, f, acc)


@dataclass(frozen=True)
class Ties:
    """The elements of src inside the set at key `tie`, by ~index."""

    src: Any
    set_key: Uint32
    set_idx: Uint32
    tie: Uint32

    def visit(self, f, acc):
        return self.src.ties(self.set_key, self.set_idx, self.tie, f, acc)


@dataclass(frozen=True)
class MassSum:
    maximum: Float32

    @cute.jit
    def __call__(self, key: Uint32, value: Uint32, acc: Uint64):
        return acc + fixed_mass(value, self.maximum)


@cute.jit
def set_mass(
    s,
    src,
    set_key: Uint32,
    set_idx: Uint32,
    maximum: Float32,
    cl: cutlass.Constexpr[bool],
):
    part = src.values(set_key, set_idx, MassSum(maximum), Uint64(0))
    return group_reduce(s, part, add, cl)


@cute.jit
def row_max(s, xr, cl: cutlass.Constexpr[bool]):
    m = fold_row(xr, lambda y, i, acc: cute.arch.fmax(acc, y), Float32(NEG_INF))
    return group_reduce(s, m, cute.arch.fmax, cl)


# ---------------------------------------------------------------------------
# Output.


@cute.jit
def fill_and_scatter(xr, cands, cut_key: Uint32, cut_idx: Uint32):
    tid = cute.arch.thread_idx()[0]
    if xr.vec:
        for i in range(xr.begin + tid * 4, xr.end, THREADS * 4):
            st_global_v4(
                xr.addr(i),
                Float32(NEG_INF),
                Float32(NEG_INF),
                Float32(NEG_INF),
                Float32(NEG_INF),
            )
    else:
        for i in range(xr.begin + tid, xr.end, THREADS):
            put(xr.x, (xr.row, i), Float32(NEG_INF))
    cute.arch.sync_threads()
    cands.fold(Scatter(xr, cut_key, cut_idx), Int32(0))


@dataclass(frozen=True)
class Scatter:
    """Store visitor writing back the candidates the cut keeps."""

    xr: Any
    cut_key: Uint32
    cut_idx: Uint32

    @cute.jit
    def __call__(self, key: Uint32, idxs: cute.Tensor, i: Int32, acc):
        if key >= self.cut_key and key != 0:
            idx = Uint32(idxs[i])
            if key > self.cut_key or idx <= self.cut_idx:
                put(self.xr.x, (self.xr.row, Int32(idx)), from_ordered(key))
        return acc


@cute.jit
def mask_row(xr, cut_key: Uint32, cut_idx: Uint32):
    """Read-modify-write of the chunk; a vector is stored only if it changes."""
    tid = cute.arch.thread_idx()[0]
    if xr.vec:
        for i in range(xr.begin + tid * 4, xr.end, THREADS * 4):
            a, b, c, d = ld_global_v4(xr.addr(i))
            ma = masked(a, i, cut_key, cut_idx)
            mb = masked(b, i + 1, cut_key, cut_idx)
            mc = masked(c, i + 2, cut_key, cut_idx)
            md = masked(d, i + 3, cut_key, cut_idx)
            if ma != a or mb != b or mc != c or md != d:
                st_global_v4(xr.addr(i), ma, mb, mc, md)
    else:
        for i in range(xr.begin + tid, xr.end, THREADS):
            put(xr.x, (xr.row, i), masked(xr.x[xr.row, i], i, cut_key, cut_idx))


@cute.jit
def masked(y: Float32, i: Int32, cut_key: Uint32, cut_idx: Uint32):
    key = ordered(y)
    out = Float32(NEG_INF)
    if key > cut_key or (key == cut_key and Uint32(i) <= cut_idx):
        out = y
    return out


# ---------------------------------------------------------------------------
# Row data, loads and the TMA ring.


@dataclass(frozen=True)
class Launch:
    """Kernel arguments, and this CTA's overflow store in the workspace."""

    x: cute.Tensor
    ks: cute.Tensor
    ps: cute.Tensor
    vocab: Int32
    has_k: Boolean
    has_p: Boolean
    vec: Boolean  # 16-byte aligned rows: TMA and vector loads
    store: cute.Tensor


def make_launch(x, ks, ps, ws, flags: Int32, vocab: Int32) -> Launch:
    """flags: bit 0 k given, bit 1 p given, bit 2 aligned rows."""
    offset = Int32(cute.arch.block_idx()[0]) * (2 * GCAP)
    store = cute.make_tensor(ws.iterator + offset, cute.make_layout(2 * GCAP))
    return Launch(
        x, ks, ps, vocab, (flags & 1) != 0, (flags & 2) != 0, (flags & 4) != 0, store
    )


@dataclass(frozen=True)
class RowPtr:
    """This CTA's chunk [begin, end) of a row."""

    x: cute.Tensor
    row: Int32
    begin: Int32
    end: Int32
    vec: Boolean
    base: Int64

    def addr(self, i):
        return self.base + Int64(i) * 4


def make_row(x, row: Int32, begin: Int32, end: Int32, vec: Boolean) -> RowPtr:
    base = x.iterator.toint() + Int64(row) * Int64(x.stride[0]) * 4
    return RowPtr(x, row, begin, end, vec, base)


@cute.jit
def row_params(a, row: Int32):
    k = Int32(a.vocab)
    if a.has_k:
        k = a.ks[row]
    p = Float32(1.0)
    if a.has_p:
        p = a.ps[row]
    return k < a.vocab, Uint32(max(k, 1)), p < 1.0, p


@cute.jit
def row_active(a, row: Int32):
    use_k, _, use_p, _ = row_params(a, row)
    return use_k or use_p


@cute.jit
def next_active(a, row: Int32, batch: Int32):
    """First row at or after `row`, in this CTA's stride, that needs filtering."""
    done = row >= batch
    while not done:
        done = row_active(a, row)
        if not done:
            row += cute.arch.grid_dim()[0]
            done = row >= batch
    return min(row, batch)


@cute.jit
def load_sample(xr, v: cute.Tensor):
    begin, end = xr.begin, xr.end
    step = max(4, (end - begin) // VALS // THREADS // 4 * 4)
    tid = cute.arch.thread_idx()[0]
    for j in cutlass.range_constexpr(VALS // 4):
        i = begin + (j * THREADS + tid) * step
        if xr.vec and i + 3 < end:
            v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3] = ld_global_v4(
                xr.addr(i)
            )
        else:
            for c in cutlass.range_constexpr(4):
                v[4 * j + c] = Float32(NEG_INF)
                if i + c < end:
                    v[4 * j + c] = xr.x[xr.row, i + c]


@cute.jit
def load_scalar(xr, base: Int32, v: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    for b in cutlass.range_constexpr(VALS):
        i = base + ((b // 4) * THREADS + tid) * 4 + b % 4
        v[b] = Float32(NEG_INF)
        if i < xr.end:
            v[b] = xr.x[xr.row, i]


@dataclass(frozen=True)
class Ring:
    """FIFO of chunk stages. All threads track the same state, thread 0
    issues the bulk copies. The state, in s.ring_state: read slot, write
    slot, stages in flight, read parity, next element to load."""

    s: Any
    xr: RowPtr

    def open(self):
        self.s.ring_state[4] = self.xr.begin

    @cute.jit
    def pump(self):
        """Fills every free slot: pump runs after the previous round's
        barrier, so the slot read last round is free too."""
        s, xr, rs = self.s, self.xr, self.s.ring_state
        while rs[2] < s.depth and rs[4] < xr.end:
            slot = rs[1]
            if cute.arch.thread_idx()[0] == 0:
                nbytes = min(STAGE, xr.end - rs[4]) * 4
                tma_load(s.stage_addr(slot), xr.addr(rs[4]), nbytes, s.bar_addr(slot))
            rs[1] = slot + 1
            if slot + 1 == s.depth:
                rs[1] = 0
            rs[4] = rs[4] + STAGE
            rs[2] = rs[2] + 1

    @cute.jit
    def consume(self, base: Int32, stop: Int32, v: cute.Tensor):
        s, rs = self.s, self.s.ring_state
        slot = rs[0]
        cute.arch.mbarrier_wait(s.bar.iterator + slot, rs[3])
        tid = cute.arch.thread_idx()[0]
        for j in cutlass.range_constexpr(VALS // 4):
            i = (j * THREADS + tid) * 4
            a, b, c, d = ld_shared_v4(s.stage_addr(slot) + i * 4)
            if base + i >= stop:
                a, b, c, d = (Float32(NEG_INF),) * 4
            v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3] = a, b, c, d
        rs[0] = slot + 1
        if slot + 1 == s.depth:
            rs[0] = 0
            rs[3] = rs[3] ^ 1
        rs[2] = rs[2] - 1


@dataclass(frozen=True)
class Smem:
    """Shared memory: candidates, bins, reduction scratch and the TMA ring,
    plus this CTA's overflow store g and the ring state (registers)."""

    depth: Int32
    g: cute.Tensor
    ring: cute.Tensor  # the TMA ring, in dynamic shared memory at RING_OFFSET
    key: cute.Tensor
    idx: cute.Tensor
    count: cute.Tensor
    gcount: cute.Tensor  # cluster-combined bins
    limb: cute.Tensor  # mass bins as LIMBS u32 limbs: 64-bit shared atomics spin
    gwide: cute.Tensor
    bar: cute.Tensor
    scratch: cute.Tensor
    xin: cute.Tensor  # group_gather slots
    above: cute.Tensor
    sc: cute.Tensor
    ring_state: cute.Tensor

    @cute.jit
    def init(self):
        tid = cute.arch.thread_idx()[0]
        if tid == 0:
            put(self.sc, FOUND, Uint32(1))
            put(self.sc, XTURN, Uint32(0))
            for i in cutlass.range_constexpr(MAX_STAGES):
                cute.arch.mbarrier_init(self.bar.iterator + i, 1)
            cute.arch.inline_ptx("fence.mbarrier_init.release.cluster;")
        for i in cutlass.range_constexpr(4):
            self.ring_state[i] = Int32(0)
        cute.arch.sync_threads()

    def stage_addr(self, slot):
        return smem_addr(self.ring.iterator) + slot * (STAGE * 4)

    def bar_addr(self, slot):
        return smem_addr(self.bar.iterator) + slot * 8

    @cute.jit
    def store(self, at: Uint32, key: Uint32, idx: Uint32):
        if at < CAP:
            put(self.key, at, key)
            put(self.idx, at, idx)
        elif at < STORE:
            put(self.g, at - CAP, key)
            put(self.g, (GCAP + at - CAP), idx)

    @cute.jit
    def weight(
        self, b: Int32, mass: cutlass.Constexpr[bool], cl: cutlass.Constexpr[bool]
    ):
        if const_expr(not mass):
            return Uint64(self.bin_count(b, cl))
        if const_expr(cl):
            return Uint64(self.gwide[b])
        w = Uint64(0)
        for li in cutlass.range_constexpr(LIMBS):
            w += Uint64(self.limb[li * 256 + b]) << (LIMB_BITS * li)
        return w

    @cute.jit
    def bin_count(self, b: Int32, cl: cutlass.Constexpr[bool]):
        c = self.count[b]
        if const_expr(cl):
            c = self.gcount[b]
        return c


# The Smem arrays below the ring: field, dtype, elements.
SMEM_ARRAYS = [
    ("key", Uint32, CAP),
    ("idx", Uint32, CAP),
    ("count", Uint32, 256),
    ("gcount", Uint32, 256),
    ("limb", Uint32, LIMBS * 256),
    ("gwide", Uint64, 256),
    ("bar", Uint64, MAX_STAGES),
    ("scratch", Uint64, WARPS),
    ("xin", Uint64, 2 * MAX_SPLITS * 3),
    ("above", Uint64, 1),
    ("sc", Uint32, 8),
]
# The TMA ring, of runtime depth, follows the 16-byte aligned arrays. Two CTAs
# per SM need 2 * (RING_OFFSET + SHALLOW * STAGE * 4 + 3 KiB) <= 228 KiB, where
# 3 KiB covers the DSL's static shared memory and the per-CTA reservation
# (measured on B300). Past that, one CTA per SM costs large batches up to 50%.
RING_OFFSET = sum((n * t.width // 8 + 15) // 16 * 16 for _, t, n in SMEM_ARRAYS)
assert 2 * (RING_OFFSET + SHALLOW * STAGE * 4 + 3072) <= 228 * 1024


def make_smem(depth: Int32, g: cute.Tensor) -> Smem:
    alloc = cutlass.utils.SmemAllocator()
    arrays = {
        name: alloc.allocate_tensor(t, cute.make_layout(n), byte_alignment=16)
        for name, t, n in SMEM_ARRAYS
    }
    ring = cute.arch.get_dyn_smem(Float32, alignment=16) + RING_OFFSET // 4
    ring = cute.make_tensor(ring, cute.make_layout(MAX_STAGES * STAGE))
    rs = cute.make_rmem_tensor((5,), Int32)
    return Smem(depth=depth, g=g, ring=ring, ring_state=rs, **arrays)


# ---------------------------------------------------------------------------
# Block and cluster reductions.


@cute.jit
def block_reduce(s, v, op):
    warp = cute.arch.warp_idx()  # a shuffle: outside divergent code
    for o in cutlass.range_constexpr(5):
        v = op(v, cute.arch.shuffle_sync_bfly(v, offset=1 << (4 - o)))
    cute.arch.sync_threads()
    if cute.arch.lane_idx() == 0:
        put(s.scratch, warp, pack(v))
    cute.arch.sync_threads()
    out = unpack(s.scratch[0], type(v))
    for i in cutlass.range_constexpr(1, WARPS):
        out = op(out, unpack(s.scratch[i], type(v)))
    return out


@cute.jit
def group_reduce(s, v, op, cl: cutlass.Constexpr[bool]):
    v = block_reduce(s, v, op)
    if const_expr(cl):
        base_slot = group_gather(s, (pack(v),))
        acc = unpack(s.xin[base_slot], type(v))
        for r in range(1, Int32(cute.arch.cluster_size())):
            acc = op(acc, unpack(s.xin[base_slot + r * 3], type(v)))
        v = acc
    return v


@cute.jit
def group_gather(s, words):
    """Every CTA's words (uniform within each CTA, at most 3 Uint64), by
    cluster rank, in this CTA's s.xin: each CTA pushes its words into every
    peer, then one cluster barrier. The two slot sets alternate, so a slot is
    rewritten only after the next gather's barrier, when its owner has read
    it. Call after a __syncthreads. Returns the base slot of rank 0."""
    tid = cute.arch.thread_idx()[0]
    turn = Int32(s.sc[XTURN])
    rank = Int32(cute.arch.block_idx_in_cluster())
    if tid < Int32(cute.arch.cluster_size()):
        for w in cutlass.range_constexpr(len(words)):
            st_cluster_u64(
                s.xin.iterator + (turn * MAX_SPLITS + rank) * 3 + w, tid, words[w]
            )
    cluster_sync()
    if tid == 0:
        put(s.sc, XTURN, Uint32(turn ^ 1))
    return turn * MAX_SPLITS * 3


def add(a, b):
    return a + b


# ---------------------------------------------------------------------------
# Scalar helpers.


@cute.jit
def common_start(lo: Uint32, hi: Uint32):
    """Common high bits of all keys in [lo, hi]: (prefix, number of unknown
    low bits)."""
    diff = lo ^ hi
    start = Int32(0)
    if diff != 0:
        start = 32 - Int32(cute.arch.clz(diff))
    prefix = Uint32(0)
    if start < 32:
        prefix = hi & ~((Uint32(1) << Uint32(start)) - 1)
    return prefix, start


@cute.jit
def ordered(x: Float32):
    y = x
    if x == 0.0:
        y = Float32(0.0)  # -0 -> +0
    bits = bitcast(y, Uint32)
    out = bits | Uint32(0x80000000)
    if (bits & Uint32(0x80000000)) != 0:
        out = ~bits
    return out


@cute.jit
def from_ordered(key: Uint32):
    bits = ~key
    if (key & Uint32(0x80000000)) != 0:
        bits = key & Uint32(0x7FFFFFFF)
    return bitcast(bits, Float32)


@cute.jit
def fixed_mass(key: Uint32, maximum: Float32):
    return f32_to_u64_rn(exp(from_ordered(key) - maximum) * MASS_SCALE)


@cute.jit
def mass_target(p: Float32, total: Float64):
    return max(Uint64(1), f64_to_u64_ceil(Float64(p) * total))


@cute.jit
def rescale(total: Float64, frm: Float32, to: Float32):
    """FP64 exp is slow on some GPUs: FP32 exp keeps the relative error of
    each rescaled part below 2^-22."""
    out = Float64(0.0)
    if frm != NEG_INF:
        out = total * Float64(exp(frm - to))
    return out


@cute.jit
def warp_inclusive_scan(v):
    lane = cute.arch.lane_idx()
    for o in cutlass.range_constexpr(5):
        t = cute.arch.shuffle_sync_up(v, offset=1 << o, mask_and_clamp=0)
        if lane >= (1 << o):
            v += t
    return v


@cute.jit
def exp(x: Float32):
    return cute.math.exp(x, fastmath=False)


@dsl_user_op
def bitcast(x, dtype, *, loc=None, ip=None):
    return dtype(llvm.bitcast(dtype.mlir_type, x.ir_value(loc=loc, ip=ip)))


@dsl_user_op
def pack(v, *, loc=None, ip=None) -> Uint64:
    if isinstance(v, Float64):
        return bitcast(v, Uint64, loc=loc, ip=ip)
    if isinstance(v, Float32):
        v = bitcast(v, Uint32, loc=loc, ip=ip)
    return Uint64(v)


@dsl_user_op
def unpack(w, dtype, *, loc=None, ip=None):
    if dtype is Float64:
        return bitcast(w, Float64, loc=loc, ip=ip)
    if dtype is Float32:
        return bitcast(Uint32(w), Float32, loc=loc, ip=ip)
    return dtype(w)


def smem_addr(ptr) -> Int32:
    return Int32(ptr.toint())


def f32_to_u64_rn(x: Float32) -> Uint64:
    return cute.arch.inline_ptx(
        "cvt.rni.u64.f32 {$w0}, {$r0};", write_only_types=[Uint64], read_only_args=[x]
    )


def f64_to_u64_ceil(x: Float64) -> Uint64:
    return cute.arch.inline_ptx(
        "cvt.rpi.u64.f64 {$w0}, {$r0};", write_only_types=[Uint64], read_only_args=[x]
    )


def ld_global_v4(addr: Int64):
    return cute.arch.inline_ptx(
        "ld.global.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Float32] * 4,
        read_only_args=[addr],
    )


def st_global_v4(addr: Int64, a, b, c, d):
    cute.arch.inline_ptx(
        "st.global.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr, a, b, c, d],
    )


def ld_shared_v4(addr: Int32):
    return cute.arch.inline_ptx(
        "ld.shared.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Float32] * 4,
        read_only_args=[addr],
    )


def tma_load(dst: Int32, src: Int64, nbytes: Int32, bar: Int32):
    cute.arch.inline_ptx("fence.proxy.async.shared::cta;")
    cute.arch.inline_ptx(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [{$r0}], {$r1};",
        read_only_args=[bar, nbytes],
    )
    cute.arch.inline_ptx(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[{$r0}], [{$r1}], {$r2}, [{$r3}];",
        read_only_args=[dst, src, nbytes, bar],
    )


def remote_addr(ptr, rank) -> Int32:
    return Int32(cute.arch.map_dsmem_ptr(ptr, rank).toint())


def red_add(ptr, rank, v):
    """Fire-and-forget add into CTA rank's copy of ptr (generic atomics on
    DSMEM would wait for the old value)."""
    suffix = "u64" if isinstance(v, Uint64) else "u32"
    cute.arch.inline_ptx(
        f"red.relaxed.cluster.shared::cluster.add.{suffix} [{{$r0}}], {{$r1}};",
        read_only_args=[remote_addr(ptr, rank), v],
    )


def st_cluster_u64(ptr, rank, v: Uint64):
    cute.arch.inline_ptx(
        "st.shared::cluster.u64 [{$r0}], {$r1};",
        read_only_args=[remote_addr(ptr, rank), v],
    )


def cluster_arrive():
    cute.arch.inline_ptx("barrier.cluster.arrive.release.aligned;")


def cluster_wait():
    cute.arch.inline_ptx("barrier.cluster.wait.acquire.aligned;")


def cluster_sync():
    cluster_arrive()
    cluster_wait()


def put(tensor, index, value):
    """tensor[index] = value inside dynamic control flow, where the DSL would
    otherwise carry the tensor's owner as a loop or branch value."""
    tensor[index] = value
