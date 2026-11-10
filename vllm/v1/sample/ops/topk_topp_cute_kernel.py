# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL top-k/top-p filter over hierarchical maxima.

In-place top-k/top-p masking of FP32 logits with per-row device k and p. Each
row is handled by one CTA, or at small batch by a thread block cluster of up
to 16 CTAs that split the row:
  1. stream the chunk once: the max of every 16 logits (groups, in shared
     memory), of every 256 (supergroups, pushed into the leader CTA), the row
     max, and the sum of exp for top-p-only rows.
  2. the leader selects exactly through the levels. The supergroups whose max
     reaches the bin of the n-th best supergroup max hold the groups whose max
     reaches the bin of the n-th best group max, and those hold every logit at
     or above it. n is k, or for top-p-only rows a tier of 64, then 1024
     logits whose mass must reach the nucleus. Then the exact cut among the
     gathered candidates.
  3. write -inf over the chunk and scatter the kept candidates back.
Rows the candidates cannot answer (huge nuclei, k in the thousands) use an
exact radix selection over the row by all CTAs, whose cost does not depend on
where the cut is (as in AIR top-p): rounds read the row until the boundary
bucket fits the store, one more read gathers it, and the selection finishes
on the gathered bucket.

Semantics: exactly k tokens by logit, ties broken by lowest index, then top-p
on the renormalized top-k distribution, keeping the smallest prefix of the
(logit desc, index asc) order whose mass reaches p. Masses are FP32 exp()
rounded to 2^-44 fixed point, so sums are deterministic.
"""

from dataclasses import dataclass, replace
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import (
    Boolean,
    Float32,
    Float64,
    Int32,
    Int64,
    Uint16,
    Uint32,
    Uint64,
    const_expr,
)
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

THREADS = 256
WARPS = THREADS // 32
U = 8  # vector loads per thread per streaming round
ROUND = THREADS * U * 4
GROUP, SUPER = 16, 256  # logits per level-1 and level-2 maximum
GL = 8192  # candidate groups
MIN_CAP = 4096  # candidates in shared memory, at least
STORE1 = 512  # candidates of the first top-p tier
TIER_P, TIER_P2 = 64, 1024  # top-p-only rows try the nucleus among their best
MIN_WIDE = 8  # wide clusters bin the row early for the fallback, skip TIER_P2
FINE, FINE_SCALE = 2048, 64.0  # fallback bins per row (the last open) and nat
MAX_SPLITS = 16
BIN_SCALE = 8.0  # level bins per nat below the row max
LAST_BIN = 255  # open-ended
KEY_NEGINF = 0x007FFFFF
ALL_IDX = 0xFFFFFFFF
MAX_VOCAB = 1 << 20
MASS_SCALE = float(1 << 44)
NEG_INF = float("-inf")
# Shared scalars.
TOTAL, PREFIX, COUNT_AT, FOUND, XTURN, KEPT = range(6)
# A fixed-point mass (< 2^64) in 12-bit pieces: a bin sums at most 2^20 of each.
LIMB_BITS, LIMBS = 12, 4
BIG = 2 * STORE1  # offset of the region store in the candidate words


# ---------------------------------------------------------------------------
# Kernels.


@cute.jit
def launch(
    x: cute.Tensor,
    ks: cute.Tensor,
    ps: cute.Tensor,
    flags: Int32,
    splits: Int32,
    smem: Int32,
    cluster: cutlass.Constexpr[bool],
    stream,
):
    grid = [x.shape[0] * splits, 1, 1]
    if const_expr(cluster):
        cluster_kernel(x, ks, ps, flags).launch(
            grid=grid,
            block=[THREADS, 1, 1],
            cluster=[splits, 1, 1],
            smem=smem,
            stream=stream,
            min_blocks_per_mp=2,
            preferred_smem_carveout=100,  # else 135 KiB for runtime smem sizes
        )
    else:
        row_kernel(x, ks, ps, flags).launch(
            grid=grid,
            block=[THREADS, 1, 1],
            smem=smem,
            stream=stream,
            min_blocks_per_mp=2,
            preferred_smem_carveout=100,
        )


@cute.kernel
def row_kernel(x: cute.Tensor, ks: cute.Tensor, ps: cute.Tensor, flags: Int32):
    filter_row(x, ks, ps, flags, False)


@cute.kernel
def cluster_kernel(x: cute.Tensor, ks: cute.Tensor, ps: cute.Tensor, flags: Int32):
    filter_row(x, ks, ps, flags, True)


@cute.jit
def filter_row(x, ks, ps, flags: Int32, cl: cutlass.Constexpr[bool]):
    if const_expr(cl):
        cluster_arrive_relaxed()  # the matching wait precedes any peer access
    c = make_chunk(x, flags, cl)
    s = make_smem(c, cl)
    use_k, kk, use_p, p = row_params(ks, ps, flags, c.row, c.vocab)
    if use_k or use_p:
        filter_chunk(s, c, use_k, kk, use_p, p, cl)
    elif const_expr(cl):
        cluster_wait()


@cute.jit
def filter_chunk(
    s,
    c,
    use_k: Boolean,
    kk: Uint32,
    use_p: Boolean,
    p: Float32,
    cl: cutlass.Constexpr[bool],
):
    s.init()
    stream(s, c, use_p and not use_k, cl)
    # Peers of wide clusters wait for the leader's selection: they bin the row
    # meanwhile when the fallback is likely (top-p only, or k beyond half the
    # store).
    binned = Boolean(False)
    if const_expr(cl):
        binned = c.splits >= MIN_WIDE and (not use_k or kk > Uint32(s.cap // 2))
    if c.rank == 0:
        select_leader(s, c, use_k, kk, use_p, p, cl)
    if const_expr(cl):  # noqa: SIM102 (peer_slice divides by splits - 1)
        if binned and c.rank != 0:
            bin_slice(s, peer_slice(c), leader_max(s, c), use_p)
    group_sync(cl)
    _, _, kept, off, fallback, maximum, total = s.published()
    if fallback:
        fine_fallback(s, c, use_k, kk, use_p, p, total, maximum, binned, cl)
    else:
        fill(c)
        cute.arch.sync_threads()
        scatter(s, c, kept, off)


# ---------------------------------------------------------------------------
# Phase 1: stream the chunk once.


@cute.jit
def stream(s, c, need_sum: Boolean, cl: cutlass.Constexpr[bool]):
    """Group and supergroup maxima of the chunk, its max and sum of exp, into
    the leader."""
    tid = cute.arch.thread_idx()[0]
    lane, warp = tid % 32, tid // 32
    rounds = (c.end - c.begin + ROUND - 1) // ROUND
    v = cute.make_rmem_tensor((4 * U,), Float32)
    nv = cute.make_rmem_tensor((4 * U,), Float32)
    mj = cute.make_rmem_tensor((U,), Float32)
    if rounds > 0:
        load_round(c, c.begin, v)
    m, total = Float32(NEG_INF), Float64(0.0)
    for r in range(rounds):
        base = c.begin + r * ROUND
        if r + 1 < rounds:
            load_round(c, base + ROUND, nv)
        mr = Float32(NEG_INF)
        for j in cutlass.range_constexpr(U):
            mj[j] = cute.arch.fmax(
                cute.arch.fmax(v[4 * j], v[4 * j + 1]),
                cute.arch.fmax(v[4 * j + 2], v[4 * j + 3]),
            )
            mr = cute.arch.fmax(mr, mj[j])
        reduce_round(s, c, base - c.begin + warp * U * 128, mj, lane)
        if need_sum:
            if mr > m:
                total = rescale(total, m, mr)
            m = cute.arch.fmax(m, mr)
            if m != NEG_INF:
                total += Float64(round_sum(v, m))
        else:
            m = cute.arch.fmax(m, mr)
        for i in cutlass.range_constexpr(4 * U):
            v[i] = nv[i]
    rmax = block_reduce(s, m, cute.arch.fmax)
    rsum = Float64(0.0)
    if need_sum:
        rsum = block_reduce(s, rescale(total, m, rmax), add)
    if const_expr(cl):
        cluster_wait()
        if tid == 0:
            st_cluster(s.rstat.iterator + 2 * c.rank, 0, pack(rmax))
            st_cluster(s.rstat.iterator + 2 * c.rank + 1, 0, pack(rsum))
        dst = s.l2all.iterator + c.rank * (c.chunk // SUPER)
        for i in range(tid, c.nsl, THREADS):
            st_cluster(dst + i, 0, Float32(s.l2loc[i]))
    else:
        if tid == 0:
            put(s.rstat, 0, pack(rmax))
            put(s.rstat, 1, pack(rsum))
    group_sync(cl)


@cute.jit
def load_round(c, base: Int32, v: cute.Tensor):
    """Warp w loads U * 128 consecutive logits, float4 j of lane l at
    j * 128 + 4 l."""
    tid = cute.arch.thread_idx()[0]
    i = base + (tid // 32) * U * 128 + (tid % 32) * 4
    for j in cutlass.range_constexpr(U):
        a, b, d, z = load4(c, i + j * 128, c.end)
        v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3] = a, b, d, z


@cute.jit
def reduce_round(s, c, offset: Int32, mj: cute.Tensor, lane: Int32):
    """A group of 16 is 4 lanes of one float4 j, a supergroup of 256 is the j
    pair (2u, 2u + 1). Transposing butterflies leave each lane two group
    maxima, then four lanes one supergroup max each."""
    b0, b1 = lane & 1, (lane >> 1) & 1
    r = cute.make_rmem_tensor((4,), Float32)
    q = cute.make_rmem_tensor((2,), Float32)
    for i in cutlass.range_constexpr(4):
        keep, send = mj[2 * i], mj[2 * i + 1]
        if b0 != 0:
            keep, send = mj[2 * i + 1], mj[2 * i]
        r[i] = cute.arch.fmax(keep, cute.arch.shuffle_sync_bfly(send, offset=1))
    for i in cutlass.range_constexpr(2):
        keep, send = r[2 * i], r[2 * i + 1]
        if b1 != 0:
            keep, send = r[2 * i + 1], r[2 * i]
        q[i] = cute.arch.fmax(keep, cute.arch.shuffle_sync_bfly(send, offset=2))
        put(s.l1, offset // GROUP + (4 * i + 2 * b1 + b0) * 8 + (lane >> 2), q[i])
    keep, send = q[0], q[1]
    if b0 != 0:
        keep, send = q[1], q[0]
    z = cute.arch.fmax(keep, cute.arch.shuffle_sync_bfly(send, offset=1))
    for o in cutlass.range_constexpr(3):
        z = cute.arch.fmax(z, cute.arch.shuffle_sync_bfly(z, offset=4 << o))
    sg = offset // SUPER + 2 * b0 + b1
    if lane < 4 and sg < c.nsl:
        put(s.l2loc, sg, z)


@cute.jit
def round_sum(v: cute.Tensor, m: Float32):
    """Pairwise sum of exp(x - m) over the thread's round."""
    e = cute.make_rmem_tensor((U,), Float32)
    for j in cutlass.range_constexpr(U):
        e[j] = (fast_exp(v[4 * j] - m) + fast_exp(v[4 * j + 1] - m)) + (
            fast_exp(v[4 * j + 2] - m) + fast_exp(v[4 * j + 3] - m)
        )
    for t in cutlass.range_constexpr(3):
        for j in cutlass.range_constexpr((U // 2) >> t):
            e[j] = e[j] + e[j + ((U // 2) >> t)]
    return e[0]


# ---------------------------------------------------------------------------
# Phase 2: selection by the leader.


@cute.jit
def select_leader(
    s,
    c,
    use_k: Boolean,
    kk: Uint32,
    use_p: Boolean,
    p: Float32,
    cl: cutlass.Constexpr[bool],
):
    """The cut over the row's candidates, published to every CTA with the
    kept candidates of each peer's chunk."""
    tid = cute.arch.thread_idx()[0]
    part = Float32(NEG_INF)
    if tid < c.splits:
        part = unpack(s.rstat[2 * tid], Float32)
    maximum = block_reduce(s, part, cute.arch.fmax)
    total = Float64(0.0)
    if not use_k:
        sums = Float64(0.0)
        if tid < c.splits:
            sums = rescale(unpack(s.rstat[2 * tid + 1], Float64), part, maximum)
        total = block_reduce(s, sums, add)
    kept, off, fallback = Uint32(0), Int32(STORE1), Uint32(0)
    cut_key, cut_idx = Uint32(0), Uint32(ALL_IDX)
    if maximum != NEG_INF:
        goal = mass_target(p, total * MASS_SCALE)
        count, off = candidates(s, c, use_k, kk, goal, Bins(maximum), cl)
        if count > floor_pow2(cand_cap(off, s.cap)):
            fallback = Uint32(1)
        else:
            kept, cut_key, cut_idx, _ = sort_cut(
                s, off, count, use_k, kk, use_p, p, goal, maximum, Uint64(0), Uint64(0)
            )
            if const_expr(cl):
                push_kept(s, c, off, kept)
    if fallback != 0:  # the peers add their bins into the leader's
        for i in range(tid, 2 * FINE, THREADS):
            put(s.fine, i, Uint32(0))
    cute.arch.sync_threads()
    words = (
        Uint64(cut_key) | (Uint64(cut_idx) << 32),
        Uint64(kept) | (Uint64(off) << 32) | (Uint64(fallback) << 48),
        pack(maximum),
        pack(total),
    )
    if const_expr(cl):
        if tid < c.splits:
            for w in cutlass.range_constexpr(4):
                st_cluster(s.pub.iterator + w, tid, words[w])
    elif tid == 0:
        for w in cutlass.range_constexpr(4):
            put(s.pub, w, words[w])


@cute.jit
def candidates(
    s, c, use_k: Boolean, kk: Uint32, goal: Uint64, bins, cl: cutlass.Constexpr[bool]
):
    """The candidates, packed into a store: the logits reaching the bin of
    the n-th best group max, for n = k or, for top-p only, TIER_P when their
    exact mass reaches the nucleus. Rows of one CTA (large batches, where the
    fallback is the most expensive) then try TIER_P2, unless the missing mass
    bounds the nucleus size beyond it (it comes from logits below the lowest
    candidate). Returns their number (beyond the store for the fallback) and
    the store."""
    n, off = Uint64(TIER_P), Int32(STORE1)
    if c.splits < MIN_WIDE:
        off = Int32(0)  # the region keeps the group maxima for TIER_P2
    if use_k:
        n, off = Uint64(kk), Int32(STORE1)
    count, at = Uint32(ALL_IDX), Int32(STORE1)
    b1, ng = candidate_groups(s, c, n, bins, cl)
    second = Boolean(False)
    if ng <= GL:
        got, low, mass = gather(s, c, off, ng, bins, b1)
        if use_k or mass >= goal:
            count, at = got, off
            if off == 0 and got > STORE1:  # the first tier's store overflows
                count, _, _ = gather(s, c, Int32(STORE1), ng, bins, b1)
                at = Int32(STORE1)
        elif c.splits < MIN_WIDE:
            unit = max(fixed_mass(low, bins.maximum), Uint64(1))
            second = Uint64(got) + (goal - mass + unit - 1) // unit <= TIER_P2 // 4
    if second:
        b1, ng = candidate_groups(s, c, Uint64(TIER_P2), bins, cl)
        if ng <= GL:
            got, _, mass = gather(s, c, Int32(STORE1), ng, bins, b1)
            if mass >= goal:
                count = got
    return count, at


@cute.jit
def candidate_groups(s, c, n: Uint64, bins, cl: cutlass.Constexpr[bool]):
    """The candidate groups for the best n logits: those whose max reaches bin
    b1, the bin of the n-th best group max, found within the supergroups whose
    max reaches the bin of the n-th best supergroup max. Per level: a
    histogram of the maxima by bin, the bin where their count reaches n, then
    the indices of the maxima at or above it by count, scan and write (atomic
    latency chains dominate long sparse lists). Returns b1 and the number of
    groups (s.lists holds GL)."""
    tid = cute.arch.thread_idx()[0]
    b, picked = Int32(LAST_BIN), c.ns
    for lvl in range(2):
        items = Levels(s, c, lvl, picked, cl)
        out, cap = Int32(GL), c.ns  # supergroups, after the group list
        if lvl == 1:
            out, cap = Int32(0), Int32(GL)
        acc, total = Uint32(0), Uint32(0)
        for mode in range(3):
            if mode == 0:
                put(s.count, tid, Uint32(0))
                cute.arch.sync_threads()
            if mode == 2:
                incl, total = block_scan(s, acc)
                acc = incl - acc
            acc = items.visit(Pass(s, bins, mode, b, out, cap), acc)
            cute.arch.sync_threads()
            if mode == 0:
                b = level_bin(s, n)
                acc = Uint32(0)
        picked = Int32(total)
    return b, picked


@cute.jit
def level_bin(s, n: Uint64):
    """The first bin from the top where the histogram's count reaches n,
    LAST_BIN if it never does (the open bin only matters then)."""
    tid = cute.arch.thread_idx()[0]
    if tid < 32:
        scan_bins(s, Int32(256), Int32(0), n, Uint64(0), Uint32(0), False, False)
    cute.arch.sync_threads()
    b = Int32(LAST_BIN)
    if s.sc[FOUND] != 0:
        b = LAST_BIN - Int32(s.sc[PREFIX])
    cute.arch.sync_threads()
    if tid == 0:
        put(s.sc, FOUND, Uint32(1))
    return b


@cute.jit
def gather(s, c, off: Int32, ng: Int32, bins, b1: Int32):
    """The logits reaching bin b1 of the candidate groups, from global memory
    (L2), packed into the store at off. Returns their number (also those
    beyond the store), their lowest key and their mass."""
    tid = cute.arch.thread_idx()[0]
    cap = cand_cap(off, s.cap)
    cute.arch.sync_threads()
    if tid == 0:
        put(s.sc, TOTAL, Uint32(0))
    cute.arch.sync_threads()
    low, mass = Uint32(0xFFFFFFFF), Uint64(0)
    n4 = ng * 4
    v = cute.make_rmem_tensor((16,), Float32)
    at = cute.make_rmem_tensor((4,), Int32)
    for j0 in range(0, n4, THREADS * 4):  # uniform over the warp: append
        for q in cutlass.range_constexpr(4):
            j = j0 + q * THREADS + tid
            at[q] = Int32(0)
            a, b, d, z = (Float32(NEG_INF),) * 4
            if j < n4:
                at[q] = Int32(s.lists[j >> 2]) * GROUP + (j & 3) * 4
                a, b, d, z = load4(c, at[q], c.vocab)
            v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3] = a, b, d, z
        for q in cutlass.range_constexpr(4):
            j = j0 + q * THREADS + tid
            take = Uint32(0)
            for e in cutlass.range_constexpr(4):
                inside = j < n4 and at[q] + e < c.vocab
                if inside and bins.of(ordered(v[4 * q + e])) <= b1:
                    take |= Uint32(1 << e)
            pos = append_n(s, Uint32(cute.arch.popc(take)))
            for e in cutlass.range_constexpr(4):
                if ((take >> e) & 1) != 0:
                    key = ordered(v[4 * q + e])
                    low = min(low, key)
                    mass += fixed_mass(key, bins.maximum)
                    if pos < Uint32(cap):
                        put(s.cand, off + Int32(pos), packed(key, Uint32(at[q] + e)))
                    pos += 1
    cute.arch.sync_threads()
    count = s.sc[TOTAL]
    low = block_reduce(s, low, cutlass.min)
    mass = block_reduce(s, mass, add)
    return count, low, mass


@cute.jit
def append_n(s, n: Uint32):
    """N slots of the store for each thread, one atomic per warp."""
    lane = cute.arch.lane_idx()
    incl = warp_inclusive_scan(n)
    base = Uint32(0)
    if lane == 31 and incl != 0:
        base = cute.arch.atomic_add(s.sc.iterator + TOTAL, incl, scope="cta")
    return cute.arch.shuffle_sync(base, 31) + incl - n


@cute.jit
def sort_cut(
    s,
    off: Int32,
    m: Uint32,
    use_k: Boolean,
    kk: Uint32,
    use_p: Boolean,
    p: Float32,
    goal: Uint64,
    maximum: Float32,
    above_mass: Uint64,
    between_mass: Uint64,
):
    """Sorts the m packed candidates (key desc, index asc) by a bitonic
    network, then cuts at the kk-th, then at the smallest prefix of the top-k
    whose mass reaches p (goal for top-p only). Logits of mass above_mass rank
    above the candidates, and the top-k also holds between_mass outside them.
    Returns (kept candidates, cut)."""
    tid = cute.arch.thread_idx()[0]
    size = Int32(32)
    while size < Int32(m):
        size *= 2
    for i in range(Int32(m) + tid, size, THREADS):
        put(s.cand, off + i, Uint64(0))
    cute.arch.sync_threads()
    k2 = Int32(2)
    while k2 <= size:
        j = k2 // 2
        while j > 0:
            for t in range(tid, size // 2, THREADS):
                i = off + 2 * t - (t & (j - 1))
                a, b = Uint64(s.cand[i]), Uint64(s.cand[i + j])
                ah, bh = Uint32(a >> 32), Uint32(b >> 32)
                swap = ah < bh or (ah == bh and Uint32(a) < Uint32(b))
                if ((i - off) & k2) != 0:
                    swap = not swap
                if swap:
                    put(s.cand, i, b)
                    put(s.cand, i + j, a)
            cute.arch.sync_threads()
            j //= 2
        k2 *= 2
    n = m
    if use_k:
        n = min(kk, m)
    r = n - 1
    if use_p:
        if use_k:
            mass = Uint64(0)
            for i in range(tid, n, THREADS):
                mass += fixed_mass(Uint32(s.cand[off + Int32(i)] >> 32), maximum)
            mass = block_reduce(s, mass, add) + above_mass + between_mass
            goal = mass_target(p, Float64(mass))
        below, carry = Uint32(0), above_mass
        for i0 in range(0, n, THREADS):  # uniform: block_scan
            i = i0 + tid
            w = Uint64(0)
            if i < n:
                w = fixed_mass(Uint32(s.cand[off + Int32(i)] >> 32), maximum)
            incl, part = block_scan(s, w)
            if i < n and carry + incl < goal:
                below += 1
            carry += part
        r = min(r, block_reduce(s, below, add))
    w = s.cand[off + Int32(r)]
    return r + 1, Uint32(w >> 32), ~Uint32(w & 0xFFFFFFFF), goal


@cute.jit
def push_kept(s, c, off: Int32, kept: Uint32):
    """The kept candidates of the peers' chunks, into the peers' region store
    (free by now), so every CTA scatters its own chunk after its fill."""
    for i in range(cute.arch.thread_idx()[0], kept, THREADS):
        w = Uint64(s.cand[off + Int32(i)])
        owner = Int32(~Uint32(w & 0xFFFFFFFF)) // c.chunk
        if owner != 0:
            pos = Int32(atom_cluster_add(s.sc.iterator + KEPT, owner, Uint32(1)))
            st_cluster(s.cand.iterator + (STORE1 + pos), owner, w)


# ---------------------------------------------------------------------------
# Phase 3: output.


@cute.jit
def fill(c):
    tid = cute.arch.thread_idx()[0]
    if c.vec:
        ninf = Float32(NEG_INF)
        for i in range(c.begin + tid * 4, c.end, THREADS * 4):
            st_global_v4(c.addr(i), ninf, ninf, ninf, ninf)
    else:
        for i in range(c.begin + tid, c.end, THREADS):
            put(c.x, (c.row, i), Float32(NEG_INF))


@cute.jit
def scatter(s, c, kept: Uint32, off: Int32):
    """The kept logits of this CTA's chunk: the leader's from its sorted
    store, the peers' from the lists the leader pushed."""
    leader = c.rank == 0
    n, o = s.sc[KEPT], Int32(STORE1)
    if leader:
        n, o = kept, off
    for i in range(cute.arch.thread_idx()[0], n, THREADS):
        w = s.cand[o + Int32(i)]
        idx = Int32(~Uint32(w & 0xFFFFFFFF))
        if not leader or idx < c.end:
            put(c.x, (c.row, idx), from_ordered(Uint32(w >> 32)))


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
    out = Float32(NEG_INF)
    if keeps(ordered(y), Uint32(i), cut_key, cut_idx):
        out = y
    return out


@cute.jit
def keeps(key: Uint32, idx: Uint32, cut_key: Uint32, cut_idx: Uint32):
    """(key, idx) ranks at or above the cut in (key desc, idx asc)."""
    return key > cut_key or (key == cut_key and idx <= cut_idx)


# Exact fallback over the row by all CTAs.
assert 2 * FINE * 4 <= GL * 2  # the fallback bins live in the lists


@cute.jit
def fine_fallback(
    s,
    c,
    use_k: Boolean,
    kk: Uint32,
    use_p: Boolean,
    p: Float32,
    total: Float64,
    maximum: Float32,
    binned: Boolean,
    cl: cutlass.Constexpr[bool],
):
    """Exact fallback in a fixed number of passes over the row: a histogram
    of fine value bins (counts, and in-bin offsets that bound the masses)
    locates the top-k bin and a window of bins holding the top-p cut. One
    read-modify-write pass then masks the bins below the window, sums the
    exact mass above it and gathers the window and the top-k bin into the
    leader, which cuts exactly among them. Windows beyond the leader's store
    (heavy ties) take radix passes instead. binned: the peers binned the row
    already, while the leader selected."""
    tid = cute.arch.thread_idx()[0]
    if not binned:
        bin_slice(s, c, maximum, use_p)
    if const_expr(cl):
        if not binned:
            cluster_sync()  # the leader's own bins are complete
        if c.rank != 0:
            for b in range(tid, 2 * FINE, THREADS):
                if s.fine[b] != 0:
                    red_add(s.fine.iterator + b, 0, Uint32(s.fine[b]))
        cluster_sync()
    if c.rank == 0:
        lo, hi, kb, rank, count, keep_all = plan(s, use_k, kk, use_p, p, total)
        if tid == 0:
            put(s.sc, TOTAL, Uint32(0))
        words = (
            Uint64(lo) | (Uint64(hi) << 16) | (Uint64(kb) << 32)
            | (Uint64(keep_all) << 48),
            Uint64(rank) | (Uint64(count) << 32),
        )  # fmt: skip
        for w in cutlass.range_constexpr(2):
            if const_expr(cl):
                if tid < c.splits:
                    st_cluster(s.pub.iterator + w, tid, words[w])
            elif tid == 0:
                put(s.pub, w, words[w])
    group_sync(cl)
    w0, w1 = Uint64(s.pub[0]), Uint64(s.pub[1])
    lo, hi, kb = (
        Int32(w0 & 0xFFFF),
        Int32((w0 >> 16) & 0xFFFF),
        Int32((w0 >> 32) & 0xFFFF),
    )
    rank, count = Uint32(w1 & 0xFFFFFFFF), Uint32(w1 >> 32)
    if (w0 >> 48) == 0:  # else the cut lies beyond every logit
        if count > floor_pow2(s.cap):
            cut_key, cut_idx = global_path(
                s, c, use_k, kk, use_p, p, total, maximum, cl
            )
            mask_row(c, cut_key, cut_idx)
        else:
            above, between = window_pass(s, c, lo, hi, kb, maximum, use_p, cl)
            above = group_reduce(s, above, add, cl)
            between = group_reduce(s, between, add, cl)
            if c.rank == 0:
                cut_window(
                    s, c, count, use_k, rank, use_p, p, total, maximum, above, between
                )


@cute.jit
def bin_slice(s, part, maximum: Float32, use_p: Boolean):
    """The fine bin counts and sums of in-bin offsets (in 1/256 bin) of a part
    of the row, in this CTA's s.fine."""
    for i in range(cute.arch.thread_idx()[0], 2 * FINE, THREADS):
        put(s.fine, i, Uint32(0))
    cute.arch.sync_threads()
    fold_row(part, FineCount(s, maximum, use_p), Int32(0))
    cute.arch.sync_threads()


def peer_slice(c):
    """The peers' share of the whole row, binned while the leader selects."""
    workers = c.splits - 1
    per = (c.vocab + 4 * workers - 1) // (4 * workers) * 4
    begin = cutlass.min(c.vocab, (c.rank - 1) * per)
    return replace(c, begin=begin, end=cutlass.min(c.vocab, begin + per))


@cute.jit
def leader_max(s, c):
    """The row max from the leader's per-CTA maxima."""
    m = Float32(NEG_INF)
    for r in range(c.splits):
        w = ld_cluster_u64(remote_addr(s.rstat.iterator + 2 * r, 0))
        m = cute.arch.fmax(m, unpack(w, Float32))
    return m


@dataclass(frozen=True)
class FineCount:
    s: Any
    maximum: Float32
    use_p: Boolean

    @cute.jit
    def __call__(self, y: Float32, i: Int32, acc):
        at = fine_at(y, self.maximum)
        b = Int32(at >> 8)
        cute.arch.atomic_add(self.s.fine.iterator + b, Uint32(1), scope="cta")
        if self.use_p and (at & 255) != 0:
            cute.arch.atomic_add(
                self.s.fine.iterator + (FINE + b), at & 255, scope="cta"
            )
        return acc


@cute.jit
def fine_at(y: Float32, maximum: Float32):
    """Fallback bins of 1/FINE_SCALE nat below the row max, the last one open,
    in 256 positions each: round(256 FINE_SCALE (maximum - y)) through the
    float's mantissa, so bin b starts 1/512 before b."""
    z = (maximum - y) * Float32(256 * FINE_SCALE) + Float32(8388608.0)
    return min(bitcast(z, Uint32) - Uint32(0x4B000000), Uint32(256 * (FINE - 1)))


@cute.jit
def plan(s, use_k: Boolean, kk: Uint32, use_p: Boolean, p: Float32, total: Float64):
    """The leader's plan from the combined bins, thread t holding bins PER t
    to PER t + PER - 1: the top-k bin kb, and the top-p window [lo, hi] from
    the first bin where the upper bounds of the cumulative mass reach the
    target to the first where the lower bounds do. With top-k, the target is
    bounded through the top-k bin. Returns (lo, hi, kb, the k-th logit's rank
    among the gathered, their number, keep all)."""
    per = FINE // THREADS
    first = per * cute.arch.thread_idx()[0]
    n = cute.make_rmem_tensor((per,), Uint64)
    low = cute.make_rmem_tensor((per,), Uint64)
    high = cute.make_rmem_tensor((per,), Uint64)
    ns, ls, hs = Uint64(0), Uint64(0), Uint64(0)
    for j in cutlass.range_constexpr(per):
        n[j] = Uint64(s.fine[first + j])
        low[j], high[j] = Uint64(0), Uint64(0)
        if use_p:
            offsets = Uint32(s.fine[FINE + first + j])
            low[j] = mass_bound(Uint32(n[j]), offsets, first + j, False)
            high[j] = mass_bound(Uint32(n[j]), offsets, first + j, True)
        ns, ls, hs = ns + n[j], ls + low[j], hs + high[j]
    incl, everything = block_scan(s, ns)
    n0 = incl - ns
    low0, _ = block_scan(s, ls)
    high0, _ = block_scan(s, hs)
    low0, high0 = low0 - ls, high0 - hs
    acc = n0
    for j in cutlass.range_constexpr(per):  # then the counts above each bin
        put(s.fine, FINE + first + j, Uint32(acc))
        acc += n[j]
    cute.arch.sync_threads()
    lo, hi, kb, keep_all = Int32(0), Int32(FINE - 1), Int32(FINE), Uint32(0)
    if use_k:
        kb = reach(s, n, n0, Uint64(kk), first)
        lo, hi = kb, kb
    if use_p:
        least = mass_target(p, total * MASS_SCALE)
        most = least
        if use_k:
            need = kk - count_above(s, kb, everything)
            bottom = mass_bound(need, 255 * need, kb, False)
            least = mass_target(p, Float64(sum_above(s, low, kb, first) + bottom))
            top = sum_above(s, high, kb, first) + mass_bound(need, Uint32(0), kb, True)
            most = mass_target(p, Float64(top))
        lo = reach(s, high, high0, least, first)
        hi = min(reach(s, low, low0, most, first), min(kb, FINE - 1))
        if lo == FINE:
            keep_all = Uint32(1)
    lo_at = count_above(s, lo, everything)
    count = count_above(s, hi + 1, everything) - lo_at
    between = Uint32(0)
    if kb > hi and kb < FINE:
        between = count_above(s, kb, everything) - count_above(s, hi + 1, everything)
        count += count_above(s, kb + 1, everything) - count_above(s, kb, everything)
    return lo, hi, kb, kk - lo_at - between, count, keep_all


@cute.jit
def count_above(s, b: Int32, everything: Uint64):
    """The logits in bins before b (planned counts)."""
    out = Uint32(everything)
    if b < FINE:
        out = Uint32(s.fine[FINE + b])
    return out


@cute.jit
def reach(s, w: cute.Tensor, start: Uint64, need: Uint64, first: Int32):
    """The first bin where the cumulative weight reaches need (FINE if never)."""
    hit, acc = Int32(FINE), start
    for j in cutlass.range_constexpr(FINE // THREADS):
        acc += w[j]
        if hit == FINE and acc >= need:
            hit = first + j
    return block_reduce(s, hit, cutlass.min)


@cute.jit
def sum_above(s, w: cute.Tensor, b: Int32, first: Int32):
    part = Uint64(0)
    for j in cutlass.range_constexpr(FINE // THREADS):
        if first + j < b:
            part += w[j]
    return block_reduce(s, part, add)


@cute.jit
def mass_bound(n: Uint32, offsets: Uint32, b: Int32, upper: cutlass.Constexpr[bool]):
    """Bounds of the fixed-point mass of n logits in fine bin b (from b - 1/512
    to b + 1 - 1/512) from the sum of their in-bin offsets: the chord of exp
    above, Jensen at the mean below, with slack for the rounding of expf and
    of the fixed point."""
    out = Uint64(0)
    if n != 0 and (const_expr(upper) or b != FINE - 1):
        start = Float32(b) - Float32(1.0 / 512)
        top = exp(-start / Float32(FINE_SCALE))
        bottom = top
        if b != FINE - 1:
            bottom = exp(-(start + Float32(1.0)) / Float32(FINE_SCALE))
        if const_expr(upper):
            m = Float64(n) * Float64(top) - Float64(offsets) / 256.0 * Float64(
                top - bottom
            )
            out = Uint64(m * (MASS_SCALE * 1.00001)) + Uint64(n)
        else:
            mean = (Float32(offsets) + Float32(n)) / (Float32(256.0) * Float32(n))
            m = Float64(n) * Float64(exp(-(start + mean) / Float32(FINE_SCALE)))
            m = m * (MASS_SCALE * 0.99999) - Float64(n)
            if m > 0.0:
                out = Uint64(m)
    return out


@cute.jit
def window_pass(
    s,
    c,
    lo: Int32,
    hi: Int32,
    kb: Int32,
    maximum: Float32,
    use_p: Boolean,
    cl: cutlass.Constexpr[bool],
):
    """One read-modify-write of the chunk: -inf over the bins below the window
    (bin kb included), the window and bin kb into the leader's region store.
    Returns this thread's exact mass above the window, and that of the bins between it
    and bin kb."""
    tid = cute.arch.thread_idx()[0]
    above, between = Uint64(0), Uint64(0)
    apart = kb > hi and kb < FINE
    v = cute.make_rmem_tensor((16,), Float32)
    for i0 in range(c.begin, c.end, THREADS * 16):  # uniform over the warp
        for q in cutlass.range_constexpr(4):
            x0, x1, x2, x3 = load4(c, i0 + (q * THREADS + tid) * 4, c.end)
            v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3] = x0, x1, x2, x3
        take, cut = Uint32(0), Uint32(0)
        for e in cutlass.range_constexpr(16):
            i = i0 + ((e // 4) * THREADS + tid) * 4 + e % 4
            b = Int32(fine_at(v[e], maximum) >> 8)
            if i < c.end:
                if (b >= lo and b <= hi) or b == kb:
                    take |= Uint32(1 << e)
                if b > hi and v[e] != NEG_INF:
                    cut |= Uint32(1 << e)
                if use_p and b < lo:
                    above += fixed_mass(ordered(v[e]), maximum)
                if use_p and apart and b > hi and b < kb:
                    between += fixed_mass(ordered(v[e]), maximum)
        if cute.arch.vote_any_sync(take != 0):
            pos = append_leader(s, Uint32(cute.arch.popc(take)), cl)
            for e in cutlass.range_constexpr(16):
                if ((take >> e) & 1) != 0:
                    i = i0 + ((e // 4) * THREADS + tid) * 4 + e % 4
                    w = packed(ordered(v[e]), Uint32(i))
                    if const_expr(cl):
                        st_cluster(s.cand.iterator + (STORE1 + pos), 0, w)
                    else:
                        put(s.cand, STORE1 + pos, w)
                    pos += 1
        for q in cutlass.range_constexpr(4):
            if ((cut >> (4 * q)) & 15) != 0:
                for e in cutlass.range_constexpr(4):
                    if ((cut >> (4 * q + e)) & 1) != 0:
                        v[4 * q + e] = Float32(NEG_INF)
                i = i0 + (q * THREADS + tid) * 4
                store4(c, i, c.end, v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3])
    return above, between  # per thread


@cute.jit
def append_leader(s, n: Uint32, cl: cutlass.Constexpr[bool]):
    """N slots of the leader's store for each thread, one atomic per warp."""
    lane = cute.arch.lane_idx()
    incl = warp_inclusive_scan(n)
    base = Uint32(0)
    if lane == 31 and incl != 0:
        if const_expr(cl):
            base = atom_cluster_add(s.sc.iterator + TOTAL, 0, incl)
        else:
            base = cute.arch.atomic_add(s.sc.iterator + TOTAL, incl, scope="cta")
    return Int32(cute.arch.shuffle_sync(base, 31) + incl - n)


@cute.jit
def cut_window(
    s,
    c,
    m: Uint32,
    use_k: Boolean,
    rank: Uint32,
    use_p: Boolean,
    p: Float32,
    total: Float64,
    maximum: Float32,
    above: Uint64,
    between: Uint64,
):
    """The leader's exact cut among the gathered logits, then -inf over those
    it cuts."""
    goal = mass_target(p, total * MASS_SCALE)
    kept, _, _, _ = sort_cut(
        s, Int32(STORE1), m, use_k, rank, use_p, p, goal, maximum, above, between
    )
    for i in range(kept + cute.arch.thread_idx()[0], m, THREADS):
        idx = Int32(~Uint32(Uint64(s.cand[STORE1 + Int32(i)]) & 0xFFFFFFFF))
        put(c.x, (c.row, idx), Float32(NEG_INF))


@cute.jit
def store4(c, i: Int32, end: Int32, a: Float32, b: Float32, d: Float32, z: Float32):
    """Logits i to i + 3 of the row, up to end."""
    if c.vec and i + 3 < end:
        st_global_v4(c.addr(i), a, b, d, z)
    else:
        if i < end:
            put(c.x, (c.row, i), a)
        if i + 1 < end:
            put(c.x, (c.row, i + 1), b)
        if i + 2 < end:
            put(c.x, (c.row, i + 2), d)
        if i + 3 < end:
            put(c.x, (c.row, i + 3), z)


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
    cl: cutlass.Constexpr[bool],
):
    cut_key, cut_idx = Uint32(0), Uint32(ALL_IDX)
    if use_k:
        cut_key, cut_idx = select_row(
            s, xr, cut_key, cut_idx, Uint64(kk), maximum, False, cl
        )
    if use_p:
        mass = total * MASS_SCALE
        if use_k:
            mass = Float64(set_mass(s, RowSource(xr), cut_key, cut_idx, maximum, cl))
        cut_key, cut_idx = select_row(
            s, xr, cut_key, cut_idx, mass_target(p, mass), maximum, True, cl
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
    mass: cutlass.Constexpr[bool],
    cl: cutlass.Constexpr[bool],
):
    """Like select over the row, at a cost independent of where the cut is
    (AIR top-p): radix rounds read the row until the boundary bucket fits
    every CTA's store, then one more read gathers that bucket and the
    selection finishes on it."""
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
            fits = group_reduce(s, local, cutlass.max, cl) <= Uint32(s.cap)
    cut_key, cut_idx = set_key, set_idx
    if found:
        if fits:
            gather_row(s, xr, set_key, set_idx, prefix, start)
            cands = Stored(s.store(BIG), s.sc[TOTAL])
            cut_key, cut_idx = select(
                s, cands, set_key, set_idx, target - above, prefix, start, maximum,
                xr.vocab, mass, cl,
            )  # fmt: skip
        else:  # more than a store of ties at one key
            cut_key, cut_idx = finish(
                s, RowSource(xr), set_key, set_idx, prefix, count_at, target - above,
                maximum, xr.vocab, mass, cl,
            )  # fmt: skip
    return cut_key, cut_idx


@cute.jit
def gather_row(s, xr, set_key: Uint32, set_idx: Uint32, prefix: Uint32, start: Int32):
    """Stores the set's elements whose keys share prefix above bit start."""
    cute.arch.sync_threads()
    if cute.arch.thread_idx()[0] == 0:
        put(s.sc, TOTAL, Uint32(0))
    cute.arch.sync_threads()
    high = prefix >> Uint32(start)
    fold_row(xr, Gather(s, s.store(BIG), set_key, set_idx, high, start), Int32(0))
    cute.arch.sync_threads()


@dataclass(frozen=True)
class Gather:
    s: Any
    st: Any
    set_key: Uint32
    set_idx: Uint32
    high: Uint32
    start: Int32

    @cute.jit
    def __call__(self, y: Float32, i: Int32, acc):
        key = ordered(y)
        if (key >> Uint32(self.start)) == self.high and keeps(
            key, Uint32(i), self.set_key, self.set_idx
        ):
            at = cute.arch.atomic_add(
                self.s.sc.iterator + TOTAL, Uint32(1), scope="cta"
            )
            self.st.put(at, key, Uint32(i))
        return acc


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
            s, src, set_key, set_idx, key, count_at, target - above, maximum, vocab,
            mass, cl,
        )  # fmt: skip
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
            s, Ties(src, set_key, set_idx, key), rank, prefix, start, Float32(0.0),
            False, cl,
        )  # fmt: skip
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
# acc) over the owned elements, items.visit(f, acc) f(key, item index, acc).


@dataclass(frozen=True)
class Store:
    """Candidate keys at off, their indices at off + cap, in s.words."""

    w: cute.Tensor
    off: Int32
    cap: Int32

    def key(self, i):
        return Uint32(self.w[self.off + Int32(i)])

    def idx(self, i):
        return Uint32(self.w[self.off + self.cap + Int32(i)])

    @cute.jit
    def put(self, at: Uint32, key: Uint32, idx: Uint32):
        if at < Uint32(self.cap):
            put(self.w, self.off + Int32(at), key)
            put(self.w, self.off + self.cap + Int32(at), idx)


@dataclass(frozen=True)
class Stored:
    """The first n candidates of a store."""

    st: Store
    n: Uint32

    def values(self, set_key: Uint32, set_idx: Uint32, f, acc):
        return self.fold(InStore(set_key, set_idx, Uint32(0), f, False), acc)

    def ties(self, set_key: Uint32, set_idx: Uint32, tie: Uint32, f, acc):
        return self.fold(InStore(set_key, set_idx, tie, f, True), acc)

    @cute.jit
    def fold(self, g, acc):
        """Folds g(key, store, position, acc), two loads in flight."""
        n = min(self.n, Uint32(self.st.cap))
        k = cute.make_rmem_tensor((2,), Uint32)
        for i0 in range(cute.arch.thread_idx()[0], n, THREADS * 2):
            for q in cutlass.range_constexpr(2):
                k[q] = Uint32(0)
                if i0 + q * THREADS < n:
                    k[q] = self.st.key(i0 + q * THREADS)
            for q in cutlass.range_constexpr(2):
                acc = g(k[q], self.st, i0 + q * THREADS, acc)
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
    def __call__(self, key: Uint32, st, i: Int32, acc):
        # The index matters only for a tie with the set's boundary key.
        if const_expr(self.ties):
            if key == self.tie and key != 0:
                idx = st.idx(i)
                if key > self.set_key or idx <= self.set_idx:
                    acc = self.f(~idx, key, acc)
        else:
            if key >= self.set_key and key != 0:
                keep = key > self.set_key
                if not keep:
                    keep = st.idx(i) <= self.set_idx
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
        if keeps(key, Uint32(i), self.set_key, self.set_idx):
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


@dataclass(frozen=True)
class Bins:
    """Value bins of 1/BIN_SCALE nat below the row max, the last open-ended.
    Monotone in the logit: "bin <= b" is "logit >= some threshold"."""

    maximum: Float32

    @cute.jit
    def of(self, key: Uint32):
        d = (self.maximum - from_ordered(key)) * Float32(BIN_SCALE)
        b = Int32(LAST_BIN)
        if d < Float32(LAST_BIN):
            b = Int32(d)
        return b


@dataclass(frozen=True)
class Levels:
    """Level 0: the row's supergroup maxima (in the leader). Level 1: the
    group maxima of the first n supergroups in the supergroup list (all if n
    is every supergroup), from the CTAs that own them, four per load."""

    s: Any
    c: Any
    lvl: Int32
    n: Int32
    cl: bool

    @cute.jit
    def visit(self, f, acc):
        tid = cute.arch.thread_idx()[0]
        if self.lvl == 0:
            for i in range(tid, self.c.ns, THREADS):
                acc = f(ordered(Float32(self.s.l2all[i])), Uint32(i), acc)
        else:
            n4 = self.n * 4
            v = cute.make_rmem_tensor((16,), Float32)
            for j0 in range(tid, n4, THREADS * 4):
                for q in cutlass.range_constexpr(4):
                    j = j0 + q * THREADS
                    a, b, d, z = (Float32(NEG_INF),) * 4
                    if j < n4:
                        a, b, d, z = self.load(self.group(j))
                    v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3] = a, b, d, z
                for q in cutlass.range_constexpr(4):
                    j = j0 + q * THREADS
                    if j < n4:
                        g = self.group(j)
                        for e in cutlass.range_constexpr(4):
                            acc = f(ordered(v[4 * q + e]), Uint32(g + e), acc)
        return acc

    @cute.jit
    def group(self, j: Int32):
        g = 4 * j
        if self.n != self.c.ns:
            g = Int32(self.s.lists[GL + (j >> 2)]) * (SUPER // GROUP) + (j & 3) * 4
        return g

    @cute.jit
    def load(self, g: Int32):
        if const_expr(self.cl):
            per = self.c.chunk // GROUP
            owner = g // per
            addr = remote_addr(self.s.l1.iterator + (g - owner * per), owner)
            v = ld_cluster_v4(addr)
        else:
            v = ld_shared_v4(smem_addr(self.s.l1.iterator + g))
        return v


@dataclass(frozen=True)
class Pass:
    """Level visitor. Mode 0: the histogram of the maxima by bin. Mode 1:
    count those at or above bin b. Mode 2: write their indices from position
    acc of the list at out."""

    s: Any
    bins: Bins
    mode: Int32
    b: Int32
    out: Int32
    cap: Int32

    @cute.jit
    def __call__(self, key: Uint32, i: Uint32, acc: Uint32):
        b = self.bins.of(key)
        if self.mode == 0:
            if b < LAST_BIN:
                cute.arch.atomic_add(
                    self.s.count.iterator + (LAST_BIN - b), Uint32(1), scope="cta"
                )
        elif b <= self.b:
            if self.mode == 2 and acc < Uint32(self.cap):
                put(self.s.lists, self.out + Int32(acc), Uint16(i))
            acc += 1
        return acc


# ---------------------------------------------------------------------------
# Row data and shared memory.


@dataclass(frozen=True)
class Chunk:
    """This CTA's chunk [begin, end) of a row."""

    x: cute.Tensor
    row: Int32
    rank: Int32
    splits: Int32
    begin: Int32
    end: Int32
    chunk: Int32
    vocab: Int32
    ns: Int32  # supergroups of the row
    nsl: Int32  # supergroups of the chunk
    vec: Boolean  # 16-byte aligned rows: vector loads
    base: Int64

    def addr(self, i):
        return self.base + Int64(i) * 4


def make_chunk(x, flags: Int32, cl: bool) -> Chunk:
    """flags: bit 0 k given, bit 1 p given, bit 2 aligned rows."""
    vocab = Int32(x.shape[1])
    splits, rank = Int32(1), Int32(0)
    if cl:
        splits = Int32(cute.arch.cluster_size())
        rank = Int32(cute.arch.block_idx_in_cluster())
    row = Int32(cute.arch.block_idx()[0]) // splits
    chunk = chunk_size(vocab, splits)
    begin = cutlass.min(vocab, rank * chunk)
    end = cutlass.min(vocab, begin + chunk)
    base = x.iterator.toint() + Int64(row) * Int64(x.stride[0]) * 4
    ns = (vocab + SUPER - 1) // SUPER
    nsl = (end - begin + SUPER - 1) // SUPER
    return Chunk(
        x, row, rank, splits, begin, end, chunk, vocab, ns, nsl, (flags & 4) != 0, base
    )


def chunk_size(vocab, splits):
    return ((vocab + splits - 1) // splits + SUPER - 1) // SUPER * SUPER


@cute.jit
def row_params(ks, ps, flags: Int32, row: Int32, vocab: Int32):
    k = vocab
    if (flags & 1) != 0:
        k = ks[row]
    p = Float32(1.0)
    if (flags & 2) != 0:
        p = ps[row]
    return k < vocab, Uint32(max(k, 1)), p < 1.0, p


@cute.jit
def load4(c, i: Int32, end: Int32):
    """Logits i to i + 3 of the row, -inf from end."""
    a, b, d, z = (Float32(NEG_INF),) * 4
    if c.vec and i + 3 < end:
        a, b, d, z = ld_global_v4(c.addr(i))
    else:
        if i < end:
            a = c.x[c.row, i]
        if i + 1 < end:
            b = c.x[c.row, i + 1]
        if i + 2 < end:
            d = c.x[c.row, i + 2]
        if i + 3 < end:
            z = c.x[c.row, i + 3]
    return a, b, d, z


@dataclass(frozen=True)
class Smem:
    """Shared memory: bins, reduction scratch, the leader's inputs and
    decision, then the dynamic arrays (see layout)."""

    count: cute.Tensor
    gcount: cute.Tensor  # cluster-combined bins
    limb: cute.Tensor  # mass bins as LIMBS u32 limbs: 64-bit shared atomics spin
    gwide: cute.Tensor
    scratch: cute.Tensor
    xin: cute.Tensor  # group_gather slots
    above: cute.Tensor
    sc: cute.Tensor
    rstat: cute.Tensor  # per CTA: chunk max, sum of exp (leader)
    pub: cute.Tensor  # the leader's decision
    l2all: cute.Tensor
    l2loc: cute.Tensor
    lists: cute.Tensor  # GL candidate groups, then the candidate supergroups
    fine: cute.Tensor  # lists as the fallback bins: counts, then offsets
    words: cute.Tensor  # the first tier's store, then the region
    cand: Any  # words as packed candidates (key << 32 | ~index)
    l1: cute.Tensor  # the region before selection: the chunk's group maxima
    cap: Int32  # candidates of the region store

    @cute.jit
    def init(self):
        tid = cute.arch.thread_idx()[0]
        if tid == 0:
            put(self.sc, FOUND, Uint32(1))
            put(self.sc, XTURN, Uint32(0))
            put(self.sc, KEPT, Uint32(0))
        cute.arch.sync_threads()

    def store(self, off: int) -> Store:
        """The region as (keys, indices), for the fallback."""
        return Store(self.words, Int32(off), self.cap)

    @cute.jit
    def published(self):
        w0, w1 = self.pub[0], self.pub[1]
        return (
            Uint32(w0 & 0xFFFFFFFF),
            Uint32(w0 >> 32),
            Uint32(w1 & 0xFFFFFFFF),
            Int32((w1 >> 32) & 0xFFFF),
            (w1 >> 48) != 0,
            unpack(self.pub[2], Float32),
            unpack(self.pub[3], Float64),
        )

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


@cute.jit
def cand_cap(off: Int32, cap: Int32):
    """Packed candidates of the store at off: the first tier's, or the
    region's."""
    out = cap
    if off == 0:
        out = Int32(STORE1)
    return out


@cute.jit
def floor_pow2(n: Int32):
    return Uint32(1) << Uint32(31 - Int32(cute.arch.clz(Uint32(n))))


@cute.jit
def packed(key: Uint32, idx: Uint32):
    """Descending order of (key desc, index asc)."""
    return (Uint64(key) << 32) | Uint64(~idx)


# The static Smem arrays: field, dtype, elements.
SMEM_ARRAYS = [
    ("count", Uint32, 256),
    ("gcount", Uint32, 256),
    ("limb", Uint32, LIMBS * 256),
    ("gwide", Uint64, 256),
    ("scratch", Uint64, WARPS),
    ("xin", Uint64, 2 * MAX_SPLITS * 3),
    ("above", Uint64, 1),
    ("sc", Uint32, 8),
    ("rstat", Uint64, 2 * MAX_SPLITS),
    ("pub", Uint64, 4),
]
FIXED = sum((n * t.width // 8 + 15) // 16 * 16 for _, t, n in SMEM_ARRAYS)


def align16(n):
    return (n + 15) // 16 * 16


def layout(vocab, splits, cl: bool, mx=max):
    """Byte offsets of the dynamic arrays and the shared memory size: the
    row's supergroup maxima (and the chunk's, for clusters), the group and
    supergroup lists, then the candidate words: the first tier's store, then
    the region of the chunk's group maxima, later the region store."""
    chunk = chunk_size(vocab, splits)
    ns = (vocab + SUPER - 1) // SUPER
    l2all = FIXED
    l2loc = l2all
    lists = l2all + align16(ns * 4)
    if cl:
        l2loc = lists
        lists = l2loc + align16(chunk // SUPER * 4)
    words = lists + align16((GL + ns) * 2)
    region = mx((chunk + ROUND - 1) // ROUND * ROUND // GROUP, 2 * MIN_CAP)
    return l2all, l2loc, lists, words, region, words + (BIG + region) * 4


def make_smem(c: Chunk, cl: bool) -> Smem:
    alloc = cutlass.utils.SmemAllocator()
    arrays = {
        name: alloc.allocate_tensor(t, cute.make_layout(n), byte_alignment=16)
        for name, t, n in SMEM_ARRAYS
    }
    l2all, l2loc, lists, words, region, _ = layout(c.vocab, c.splits, cl, cutlass.max)
    f32 = cute.arch.get_dyn_smem(Float32, alignment=16)
    u64 = cute.arch.get_dyn_smem(Uint64, alignment=16)
    u32 = cute.arch.get_dyn_smem(Uint32, alignment=16)
    u16 = cute.arch.get_dyn_smem(Uint16, alignment=16)
    return Smem(
        l2all=cute.make_tensor(f32 + l2all // 4, cute.make_layout(c.ns)),
        l2loc=cute.make_tensor(f32 + l2loc // 4, cute.make_layout(c.ns)),
        lists=cute.make_tensor(u16 + lists // 2, cute.make_layout(GL + c.ns)),
        fine=cute.make_tensor(u32 + lists // 4, cute.make_layout(2 * FINE)),
        words=cute.make_tensor(u32 + words // 4, cute.make_layout(BIG + region)),
        cand=cute.make_tensor(u64 + words // 8, cute.make_layout((BIG + region) // 2)),
        l1=cute.make_tensor(f32 + words // 4 + BIG, cute.make_layout(region)),
        cap=region // 2,
        **arrays,
    )


# ---------------------------------------------------------------------------
# Block and cluster reductions.


@cute.jit
def group_sync(cl: cutlass.Constexpr[bool]):
    if const_expr(cl):
        cluster_sync()
    else:
        cute.arch.sync_threads()


@cute.jit
def block_reduce(s, v, op):
    warp = cute.arch.thread_idx()[0] // 32
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
def block_scan(s, v):
    """Inclusive prefix sum over the CTA's threads, and the total."""
    warp = cute.arch.thread_idx()[0] // 32
    v = warp_inclusive_scan(v)
    cute.arch.sync_threads()
    if cute.arch.lane_idx() == 31:
        put(s.scratch, warp, Uint64(v))
    cute.arch.sync_threads()
    before, total = type(v)(0), type(v)(0)
    for w in cutlass.range_constexpr(WARPS):
        t = type(v)(s.scratch[w])
        if w < warp:
            before += t
        total += t
    return before + v, total


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
            st_cluster(
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


@cute.jit
def fast_exp(x: Float32):
    return cute.math.exp(x, fastmath=True)


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


def ld_cluster_u64(addr: Int32) -> Uint64:
    return cute.arch.inline_ptx(
        "ld.shared::cluster.u64 {$w0}, [{$r0}];",
        write_only_types=[Uint64],
        read_only_args=[addr],
    )


def ld_cluster_v4(addr: Int32):
    return cute.arch.inline_ptx(
        "ld.shared::cluster.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Float32] * 4,
        read_only_args=[addr],
    )


def remote_addr(ptr, rank) -> Int32:
    return Int32(cute.arch.map_dsmem_ptr(ptr, rank).toint())


def suffix(v) -> str:
    if isinstance(v, Float32):
        return "f32"
    return "u64" if type(v).width == 64 else "u32"


def red_add(ptr, rank, v):
    """Fire-and-forget add into CTA rank's copy of ptr (generic atomics on
    DSMEM would wait for the old value)."""
    cute.arch.inline_ptx(
        f"red.relaxed.cluster.shared::cluster.add.{suffix(v)} [{{$r0}}], {{$r1}};",
        read_only_args=[remote_addr(ptr, rank), v],
    )


def atom_cluster_add(ptr, rank, v: Uint32) -> Uint32:
    return cute.arch.inline_ptx(
        "atom.relaxed.cluster.shared::cluster.add.u32 {$w0}, [{$r0}], {$r1};",
        write_only_types=[Uint32],
        read_only_args=[remote_addr(ptr, rank), v],
    )


def st_cluster(ptr, rank, v):
    cute.arch.inline_ptx(
        f"st.shared::cluster.{suffix(v)} [{{$r0}}], {{$r1}};",
        read_only_args=[remote_addr(ptr, rank), v],
    )


def cluster_arrive_relaxed():
    cute.arch.inline_ptx("barrier.cluster.arrive.relaxed.aligned;")


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
