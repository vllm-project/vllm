# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed experts plus the shared expert of 8-64 decode tokens as one persistent
kernel, grouped by expert, on the TRTLLM-Gen FP8 weights (see ``mega_moe``).

One CTA per SM. CTA ``t`` routes and quantizes token ``t`` into a global plan;
every CTA then cuts the routes into units of up to 16 tokens of one expert (the
shared expert's first, then the routed experts in expert order), and splits the
weight stages (two 128 x 128 K blocks of one row tile) evenly over the CTAs:

- FC1: the stages of every unit (4 pairs of an up and a gate row tile) in one
  list, cut into one contiguous range per CTA. A pair cut between CTAs is
  finished by its owner, the CTA holding its first stage, which reaches it at
  the end of its range: the others publish FP32 partials as they start theirs.
  The owner sums them in CTA order, applies the clamped SwiGLU, quantizes the
  hidden block to FP8 per 128 and publishes it.
- FC2: (unit, output row tile) jobs, claimed in chunks from a counter as each
  CTA's FC1 ends (so the CTAs that finish FC1 first take more); a job waits
  for its unit's four hidden blocks and writes each route's BF16 output row
  (finalize-ready, like ``mega_moe``).

Above 40 units both lists are cut in halves A and B and every CTA runs FC1(A),
FC1(B), FC2(A), FC2(B), so FC2(A)'s hidden blocks are ready by the time it runs.

Block scales are not powers of two, so each K block lands in its own TMEM column
group (the unit's tokens on the UMMA N side) and the epilogue scales (weight
block scale times the token's activation scale, which the B gatherer warps
compute beside the B tiles) and sums them in FP32. The weight producer posts a
packed descriptor of every stage to the other roles through a shared memory
ring, so a stage costs each a few shared memory loads.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    acquire_tensormap_asm,
    alloc,
    asm,
    bf16x2,
    bits_float,
    copy_g2s,
    div_rn,
    fence_tmem,
    float_bits,
    load_bf16x8,
    load_tma3,
    load_tmem,
    named_barrier,
    pack_fp8,
    prefetch_bytes,
    redux,
    release_tmem,
    sigmoid,
    smem_desc,
    store_words,
    view,
    wait,
    wait_tmem,
)

from .mega_moe import (
    FP8_MAX,
    STAGE_BYTES,
    TILE,
    _commit,
    _logical_rows,
    _ordered,
    _store_bf16,
    _weight_map,
)

GROUP = 16  # tokens per UMMA column group
END = 0xFFFFFFFF  # stage descriptor past a CTA's last stage
DEPTH = 2  # K blocks per pipeline stage, each its own TMEM column group
MAX_TOKENS = 64
SMEM_LIMIT = 227 * 1024
TMEM_COLUMNS = 512
GATHERERS = 2  # B gatherer warps, taking alternate stages
EPILOGUE = 2 + GATHERERS  # first of the four epilogue warps
THREADS = 32 * (EPILOGUE + 4)  # weight producer, MMA, gatherers, epilogue
# Units above which FC1 and FC2 run as halves A and B.
SPLIT_UNITS = 40
# UMMA instruction: D FP32, A and B E4M3, both K-major, N = 16, M = 128.
IDESC = (1 << 4) | ((GROUP >> 3) << 17) | ((TILE >> 4) << 24)
# Shared CTA scalars (Int32 words).
(
    S_PARITY, S_UNITS, S_SPLIT, S_N1A, S_N1B, S_N2A, S_N2B,
    S_LO1A, S_LO1B, S_J2A, S_J2B, S_S1A, S_S1B, S_FLAG,
) = range(14)  # fmt: skip


@asm()
def _umma(d, a, b, instruction):
    """Four K32 FP8 UMMAs of one 128-K block into ``d`` (the first overwrites)
    from the elected lane."""
    mma = "tcgen05.mma.cta_group::1.kind::f8f6f4"
    code = "{ .reg .pred e,pf,pt; .reg .b64 a1,a2,a3,b1,b2,b3; "
    code += "elect.sync _|e,0xffffffff; setp.eq.b32 pf,0,1; setp.eq.b32 pt,0,0; "
    code += "add.s64 a1,$1,2; add.s64 a2,$1,4; add.s64 a3,$1,6; "
    code += "add.s64 b1,$2,2; add.s64 b2,$2,4; add.s64 b3,$2,6; "
    code += f"@e {mma} [$0],$1,$2,$3,pf; @e {mma} [$0],a1,b1,$3,pt; "
    code += f"@e {mma} [$0],a2,b2,$3,pt; @e {mma} [$0],a3,b3,$3,pt; }}"
    operands = (Uint32(d), Uint64(a), Uint64(b), Uint32(instruction))
    return code, "r,l,l,r", operands


@asm(Uint32)
def _atomic_add(pointer, value):
    return "atom.add.acq_rel.gpu.global.u32 $0,[$1],$2;", "=r,l,r", (pointer, value)


@asm()
def _red_add(pointer, value):
    return "red.relaxed.gpu.global.add.u32 [$0],$1;", "l,r", (pointer, value)


@asm(Uint32)
def _atomic_inc(pointer, limit):
    """``old >= limit ? 0 : old + 1``; returns ``old``."""
    return "atom.inc.relaxed.gpu.global.u32 $0,[$1],$2;", "=r,l,r", (pointer, limit)


@asm(Uint32)
def _load_acquire(pointer):
    return "ld.acquire.gpu.global.b32 $0,[$1];", "=r,l", (pointer,)


@asm(Uint32)
def _load_l2_u32(pointer):
    return "ld.global.cg.b32 $0,[$1];", "=r,l", (pointer,)


@asm()
def _fence():
    return "fence.acq_rel.gpu;", "~{memory}", ()


@asm(Float32)
def _load_l2(pointer):
    return "ld.global.cg.f32 $0,[$1];", "=f,l", (pointer,)


@asm(Float32, pure=True)
def _load_nc(pointer):
    return "ld.global.nc.f32 $0,[$1];", "=f,l", (pointer,)


@asm()
def _store_u32(pointer, value):
    return "st.global.b32 [$0],$1;", "l,r", (pointer, Uint32(value))


@asm()
def _store_f32(pointer, value):
    return "st.global.f32 [$0],$1;", "l,f", (pointer, value)


@asm()
def _store_u8(pointer, value):
    return "st.global.u8 [$0],$1;", "l,r", (pointer, Uint32(value))


@asm(Uint32, pure=True)
def _popc(value):
    return "popc.b32 $0,$1;", "=r,r", (Uint32(value),)


def unit_capacity(experts: int, top_k: int, tokens: int = MAX_TOKENS) -> int:
    """Most units of a call: the shared expert's, then one per started GROUP of
    each routed expert's tokens."""
    routes = tokens * top_k
    return -(-tokens // GROUP) + min(routes, experts) + routes // GROUP


def counter_words(experts: int, top_k: int, pairs: int) -> int:
    """Counters of one call parity: ready tokens, pair arrivals, unit readiness,
    the next FC2 job of halves A and B, finished CTAs."""
    return 4 + unit_capacity(experts, top_k) * (pairs + 1)


class _GroupedKernel:
    def __init__(
        self, *, hidden, inter, experts, top_k, limit, scale, ue8m0, tokens, grid,
        finalize,
    ):  # fmt: skip
        assert tokens <= MAX_TOKENS and experts % 32 == 0
        self.hidden, self.inter, self.experts = hidden, inter, experts
        self.top_k, self.limit, self.scale, self.ue8m0 = top_k, limit, scale, ue8m0
        self.tokens, self.grid, self.finalize = tokens, grid, finalize
        self.nk1, self.nk2 = hidden // TILE, inter // TILE
        self.pairs = self.nk2
        self.tiles2 = hidden // TILE
        self.span = 2 * self.nk1 // DEPTH  # FC1 stages of a pair (up, gate)
        self.unit1 = self.pairs * self.span  # FC1 stages of a unit
        self.job = self.nk2 // DEPTH  # FC2 stages of a (unit, row tile) job
        # FC2 jobs per claim: fewer unit switches against a finer tail balance
        # (8 tokens 59 vs 65 us at 2 vs 8; 64 tokens 174 vs 181 at 8 vs 2).
        self.chunk = 2 if tokens <= 16 else 8
        # Units of up to GROUP tokens: one UMMA column group, accumulators in
        # registers. Experts with more tokens run as several units.
        self.groups = 1
        self.width = w = GROUP
        self.routes = tokens * top_k
        self.shared_units = -(-tokens // GROUP)
        self.max_units = unit_capacity(experts, top_k, tokens)
        self.columns = DEPTH * w  # TMEM columns of a stage
        self.slots = min(32, TMEM_COLUMNS // self.columns)
        self.b_bytes = DEPTH * w * TILE
        self.a_bytes = DEPTH * STAGE_BYTES
        # As many weight stages as shared memory holds.
        for stages in range(12, 3, -1):
            self.layout(stages)
            if self.smem + 1024 <= SMEM_LIMIT:
                break
        assert self.smem + 1024 <= SMEM_LIMIT

    def layout(self, stages):
        w, experts, tokens = self.width, self.experts, self.tokens
        offset = 0

        def take(size, align=16):
            nonlocal offset
            offset = (offset + align - 1) // align * align
            start = offset
            offset += size
            return start

        self.stages = stages
        # Column scales of the stages in flight (see ``gather``).
        self.ring = self.slots + stages + 1
        # Stage descriptors: the producer posts one before waiting for its weight
        # slot, and the epilogue reads it up to ``stages + slots`` stages later.
        self.posts = self.slots + stages + 2
        self.off_w = take(stages * self.a_bytes, 1024)
        self.off_b = take(stages * self.b_bytes, 1024)
        self.off_ring = take(self.ring * self.columns * 4)
        self.off_hs = take(GATHERERS * w * self.nk2 * 4)
        self.off_amax = take(4 * w * 4)
        self.off_picks = take(self.routes * 4)
        self.off_mask = take(experts * 8)
        self.off_eoff = take(experts * 4)
        self.off_uexp = take(self.max_units * 4)
        self.off_uoff = take(self.max_units * 4 + 4)
        self.off_ucnt = take(self.max_units * 4)
        self.off_flat = take((self.routes + tokens) * 4)
        self.off_scal = take(64)
        self.off_desc = take(self.posts * 4)
        self.bars = 2 * stages + 2 * self.slots + self.posts
        self.off_bar = take(self.bars * 8, 8)
        self.off_tmem = take(16)
        self.smem = offset

    @cute.jit
    def __call__(
        self, maps: cute.Tensor, w13_scale: cute.Tensor, w2_scale: cute.Tensor,
        s13_scale: cute.Tensor, s2_scale: cute.Tensor, s13: cute.Tensor,
        s2: cute.Tensor, x: cute.Tensor, logits: cute.Tensor, bias: cute.Tensor,
        rows: cute.Tensor, out: cute.Tensor,
        shared_out: cute.Tensor, final: cute.Tensor, ids: cute.Tensor,
        weights: cute.Tensor, plan_x: cute.Tensor, plan_s: cute.Tensor,
        hidden: cute.Tensor, hscale: cute.Tensor, partial: cute.Tensor,
        counters: cute.Tensor, st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(
            maps,
            w13_scale,
            w2_scale,
            s13_scale,
            s2_scale,
            s13,
            s2,
            x,
            logits,
            bias,
            rows,
            out,
            shared_out,
            final,
            ids,
            weights,
            plan_x,
            plan_s,
            hidden,
            hscale,
            partial,
            counters,
        ).launch(  # fmt: skip
            grid=(self.grid, 1, 1),
            block=(THREADS, 1, 1),
            smem=self.smem + 1024,
            stream=st,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self, maps: cute.Tensor, w13_scale: cute.Tensor, w2_scale: cute.Tensor,
        s13_scale: cute.Tensor, s2_scale: cute.Tensor, s13: cute.Tensor,
        s2: cute.Tensor, x: cute.Tensor, logits: cute.Tensor, bias: cute.Tensor,
        rows: cute.Tensor, out: cute.Tensor,
        shared_out: cute.Tensor, final: cute.Tensor, ids: cute.Tensor,
        weights: cute.Tensor, plan_x: cute.Tensor, plan_s: cute.Tensor,
        hidden: cute.Tensor, hscale: cute.Tensor, partial: cute.Tensor,
        counters: cute.Tensor,
    ):  # fmt: skip
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        cta, _, _ = cute.arch.block_idx()
        stages, slots = self.stages, self.slots

        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem, 1024)
        bars = view(sm.iterator, self.off_bar, Uint64, self.bars)
        full = bars.iterator
        empty = full + stages
        acc_full = empty + stages
        acc_empty = acc_full + slots
        posted = acc_empty + slots
        tmem_base = view(sm.iterator, self.off_tmem, Uint32, 1)
        scal = view(sm.iterator, self.off_scal, Int32, 16)

        if tid == 0:
            for s in cutlass.range_constexpr(stages):
                # TMA, the 32 gather lanes' copies and their scales.
                cute.arch.mbarrier_init(full + s, 34)
                cute.arch.mbarrier_init(empty + s, 1)
            for s in cutlass.range_constexpr(slots):
                cute.arch.mbarrier_init(acc_full + s, 1)
                cute.arch.mbarrier_init(acc_empty + s, 4)
            for s in cutlass.range_constexpr(self.posts):
                cute.arch.mbarrier_init(posted + s, 1)
            cute.arch.mbarrier_init_fence()
            for m in cutlass.range_constexpr(4):
                acquire_tensormap_asm(maps.iterator + 128 * m)
        if tid == 64:
            # The shared expert's weights do not depend on the routing: pull this
            # CTA's slice into L2 while the router runs.
            for which in cutlass.range_constexpr(2):
                weight = s13 if which == 0 else s2
                size = cute.size(weight.layout)
                piece = (size // self.grid + 4095) // 4096 * 4096
                for k in cutlass.range_constexpr(piece // 4096):
                    if cta * piece + k * 4096 < size:
                        prefetch_bytes(weight.iterator + cta * piece + k * 4096, 4096)
        if tid == 96:
            # And the routed experts' block scales, which the B gatherers load
            # on their critical path.
            for which in cutlass.range_constexpr(2):
                scales = w13_scale if which == 0 else w2_scale
                size = cute.size(scales.layout)
                piece = (size // self.grid + 63) // 64 * 64
                if cta * piece < size:
                    prefetch_bytes(
                        scales.iterator + cta * piece,
                        cutlass.min(piece, size - cta * piece) * 4,
                    )  # fmt: skip
        if warp == 1:
            cute.arch.alloc_tmem(TMEM_COLUMNS, tmem_base.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = tmem_base[0]

        # No early launch_dependents: the finalize publish reads the routing
        # weights this kernel writes before its own PDL wait.
        cute.arch.griddepcontrol_wait()
        words = counter_words(self.experts, self.top_k, self.pairs)
        calls = counters.iterator + 2 * words
        if tid == 32:
            scal[S_PARITY] = Int32(_load_l2_u32(calls)) // self.grid
        if cta < self.tokens:
            if warp == 0:
                self.route(logits, bias, ids, weights, cta, lane)
            elif warp >= EPILOGUE:
                self.quantize(x, plan_x, plan_s, cta, tid - EPILOGUE * 32)
            _fence()
        cute.arch.sync_threads()
        parity = scal[S_PARITY]
        mine = counters.iterator + parity * words
        # The other parity's counters were the previous call's; the next call
        # finds them zeroed.
        other = counters.iterator + (1 - parity) * words
        per_cta = -(-words // self.grid)
        if tid < per_cta and cta * per_cta + tid < words:
            _store_u32(other + cta * per_cta + tid, 0)
        if cta < self.tokens and tid == 0:
            _atomic_add(mine, Uint32(1))
        if tid == 0:
            while _load_acquire(mine) < Uint32(self.tokens):
                pass
        cute.arch.sync_threads()
        _fence()
        self.number_units(sm, ids, scal, tid, warp, lane, cta)
        pair_count = mine + 1
        unit_ready = pair_count + unit_capacity(self.experts, self.top_k) * self.pairs
        jobs = mine + words - 3  # next FC2 job of halves A and B

        if warp == 0:
            if lane == 0:
                self.produce(sm, maps, full, empty, posted, scal, jobs)
        elif warp == 1:
            self.issue(sm, tmem, full, empty, acc_full, acc_empty)
        elif warp < EPILOGUE:
            self.gather(
                sm, full, empty, posted, plan_x, plan_s, hidden, hscale, unit_ready,
                lane, w13_scale, w2_scale, s13_scale, s2_scale, warp - 2,
            )  # fmt: skip
        else:
            self.epilogue(
                sm, tmem, warp, lane, cta, tid - EPILOGUE * 32, acc_full, acc_empty,
                scal, rows, out, shared_out, hidden, hscale, partial, pair_count,
                unit_ready,
            )  # fmt: skip
        release_tmem(tmem, TMEM_COLUMNS, warp, 1)
        if cutlass.const_expr(self.finalize):
            # Every route's row is written once all CTAs are through FC2.
            finished = mine + words - 1
            if tid == 0:
                _fence()
                _red_add(finished, Uint32(1))
                while _load_acquire(finished) < Uint32(self.grid):
                    pass
            cute.arch.sync_threads()
            self.finish_tokens(out, shared_out, final, weights, cta, tid)
        if tid == 0:
            _atomic_inc(calls, Uint32(2 * self.grid - 1))

    @cute.jit
    def finish_tokens(self, out, shared_out, final, weights, cta, tid):
        """``final`` = the routes' rows weighted by their routing weights (FP32,
        in top-k order) plus the shared expert's output, eight columns a thread."""
        per_token = self.hidden // 8
        for v in cutlass.range(
            cta * THREADS + tid, self.tokens * per_token, self.grid * THREADS, unroll=1
        ):
            token = v // per_token
            column = (v % per_token) * 8
            rows = cute.make_rmem_tensor(cute.make_layout((self.top_k, 8)), Float32)
            for k in cutlass.range_constexpr(self.top_k):
                route = token * self.top_k + k
                got = load_bf16x8(
                    out.iterator + Int64(route) * self.hidden + column, mutable=True
                )
                for c in cutlass.range_constexpr(8):
                    rows[k, c] = got[c]
            shared = load_bf16x8(
                shared_out.iterator + Int64(token) * self.hidden + column, mutable=True
            )
            total = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
            for c in cutlass.range_constexpr(8):
                total[c] = Float32(0.0)
            for k in cutlass.range_constexpr(self.top_k):
                w = _load_l2(weights.iterator + token * self.top_k + k)
                for c in cutlass.range_constexpr(8):
                    total[c] = total[c] + w * rows[k, c]
            words = tuple(
                bf16x2(
                    total[2 * c] + shared[2 * c], total[2 * c + 1] + shared[2 * c + 1]
                )
                for c in range(4)
            )
            store_words(final.iterator + Int64(token) * self.hidden + column, words)

    @cute.jit
    def route(self, logits, bias, ids, weights, token, lane):
        """Top-k of ``token`` (sigmoid scores plus bias, ties to the lower expert,
        weights renormalized and scaled); lane l holds experts l + 32 j."""
        per = self.experts // 32
        scores = cute.make_rmem_tensor(cute.make_layout((per,)), Float32)
        keys = cute.make_rmem_tensor(cute.make_layout((per,)), Uint32)
        for j in cutlass.range_constexpr(per):
            scores[j] = Float32(logits[token, j * 32 + lane])
        for j in cutlass.range_constexpr(per):
            scores[j] = sigmoid(scores[j], fast_math=True)
            keys[j] = _ordered(scores[j] + bias[j * 32 + lane])
        total = Float32(0.0)
        mine = Int32(0)
        my_score = Float32(0.0)
        for r in cutlass.range_constexpr(self.top_k):
            best = Uint32(0)
            for j in cutlass.range_constexpr(per):
                if keys[j] > best:
                    best = keys[j]
            top = redux(best, "max")
            candidate = Uint32(0xFFFFFFFF)
            for j in cutlass.range_constexpr(per - 1, -1, -1):
                if keys[j] == top:
                    candidate = Uint32(j * 32) + Uint32(lane)
            winner = Int32(redux(candidate, "min"))
            score = Float32(0.0)
            for j in cutlass.range_constexpr(per):
                if winner == j * 32 + lane:
                    score = scores[j]
                    keys[j] = Uint32(0)
            score = cute.arch.shuffle_sync(score, winner % 32)
            total = total + score
            if lane == r:
                mine = winner
                my_score = score
        if lane < self.top_k:
            route = token * self.top_k + lane
            ids[route] = mine
            weights[route] = div_rn(my_score, total) * Float32(self.scale)

    @cute.jit
    def quantize(self, x, plan_x, plan_s, token, e):
        """FP8 per 128 of ``token`` like per_token_group_quant_fp8 (with its UE8M0
        scale rounding when the stock path uses it); thread e quantizes 32
        values."""
        kb = e // 4
        source = x.iterator + Int64(token) * self.hidden + e * 32
        values = cute.make_rmem_tensor(cute.make_layout((32,)), Float32)
        for h in cutlass.range_constexpr(4):
            got = load_bf16x8(source + h * 8, mutable=True)
            for v in cutlass.range_constexpr(8):
                values[h * 8 + v] = got[v]
        amax = Float32(1e-10)
        for v in cutlass.range_constexpr(32):
            amax = cute.arch.fmax(amax, cute.arch.fmax(values[v], -values[v]))
        for offset in cutlass.range_constexpr(2):
            amax = cute.arch.fmax(
                amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << offset)
            )
        x_scale = div_rn(amax, Float32(FP8_MAX))
        if cutlass.const_expr(self.ue8m0):
            bits = Uint32(float_bits(cute.arch.fmax(x_scale, Float32(1e-10))))
            exponent = (bits >> 23) + Uint32((bits & Uint32(0x7FFFFF)) != 0)
            x_scale = bits_float(exponent << 23)
        inverse = div_rn(Float32(1.0), x_scale)
        target = plan_x.iterator + Int64(token) * self.hidden + e * 32
        for h in cutlass.range_constexpr(2):
            packed = tuple(
                pack_fp8(tuple(values[16 * h + 4 * w + v] * inverse for v in range(4)))
                for w in range(4)
            )
            store_words(target + 16 * h, packed)
        if e % 4 == 0:
            plan_s[token * self.nk1 + kb] = x_scale

    @cute.jit
    def number_units(self, sm, ids, scal, tid, warp, lane, cta):
        """Unit tables: unit 0 is the shared expert (every token), then one unit
        per routed expert in expert order with its routes in token order; and
        this CTA's stage ranges."""
        experts, routes = self.experts, self.routes
        picks = view(sm.iterator, self.off_picks, Int32, routes)
        mask = view(sm.iterator, self.off_mask, Uint32, 2 * experts)
        eoff = view(sm.iterator, self.off_eoff, Int32, experts)
        uexp = view(sm.iterator, self.off_uexp, Int32, self.max_units)
        uoff = view(sm.iterator, self.off_uoff, Int32, self.max_units + 1)
        ucnt = view(sm.iterator, self.off_ucnt, Int32, self.max_units)
        flat = view(sm.iterator, self.off_flat, Int32, routes + self.tokens)
        for i in cutlass.range(tid, 2 * experts, THREADS, unroll=1):
            mask[i] = Uint32(0)
        for r in cutlass.range(tid, routes, 2 * THREADS, unroll=1):
            got = Int32(_load_l2_u32(ids.iterator + r))
            other = Int32(0)
            if r + THREADS < routes:
                other = Int32(_load_l2_u32(ids.iterator + r + THREADS))
            picks[r] = got
            if r + THREADS < routes:
                picks[r + THREADS] = other
        cute.arch.sync_threads()
        for r in cutlass.range(tid, routes, THREADS, unroll=1):
            token = r // self.top_k
            cute.arch.atomic_or(
                mask.iterator + 2 * picks[r] + token // 32,
                Uint32(1) << Uint32(token % 32),
            )  # fmt: skip
        cute.arch.sync_threads()
        if warp == 0:
            per = experts // 32
            tokens = cute.make_rmem_tensor(cute.make_layout((per,)), Int32)
            distinct = Int32(0)
            count = Int32(0)
            for j in cutlass.range_constexpr(per):
                e = lane * per + j
                tokens[j] = Int32(_popc(mask[2 * e]) + _popc(mask[2 * e + 1]))
                count = count + tokens[j]
                distinct = distinct + (tokens[j] + GROUP - 1) // GROUP
            units = distinct
            routed = count
            for shift in cutlass.range_constexpr(5):
                a = cute.arch.shuffle_sync_up(units, offset=1 << shift)
                b = cute.arch.shuffle_sync_up(routed, offset=1 << shift)
                if lane >= (1 << shift):
                    units = units + a
                    routed = routed + b
            shared = self.shared_units
            unit = units - distinct + shared
            start = routed - count + self.tokens
            for j in cutlass.range_constexpr(per):
                e = lane * per + j
                n = tokens[j]
                eoff[e] = start
                if n > 0:
                    uexp[unit] = e
                    uoff[unit] = start
                    ucnt[unit] = cutlass.min(n, GROUP)
                    unit = unit + 1
                # Experts of more than GROUP tokens run as several units.
                for i in cutlass.range(GROUP, n, GROUP, unroll=1):
                    uexp[unit] = e
                    uoff[unit] = start + i
                    ucnt[unit] = cutlass.min(n - i, GROUP)
                    unit = unit + 1
                start = start + n
            if lane < shared:
                uexp[lane] = -1
                uoff[lane] = lane * GROUP
                ucnt[lane] = cutlass.min(self.tokens - lane * GROUP, GROUP)
            if lane == 31:
                uoff[units + shared] = start
                self.ranges(scal, units + shared, cta)
        for t in cutlass.range(tid, self.tokens, THREADS, unroll=1):
            flat[t] = t
        # The weight producer (warp 0) starts on the unit tables; the other
        # warps rank the routes.
        if warp == 0:
            cute.arch.sync_warp()
            named_barrier(2, THREADS, arrive=True)
        else:
            named_barrier(2, THREADS)
            self.rank_routes(picks, mask, eoff, flat, tid - 32)
            named_barrier(3, THREADS - 32)

    @cute.jit
    def rank_routes(self, picks, mask, eoff, flat, i):
        """``flat``: each expert's routes in token order (rows ``eoff[e]``..)."""
        for r in cutlass.range(i, self.routes, THREADS - 32, unroll=1):
            e = picks[r]
            token = r // self.top_k
            low = mask[2 * e]
            rank = Int32(0)
            if token < 32:
                rank = Int32(_popc(low & ((Uint32(1) << Uint32(token)) - Uint32(1))))
            else:
                below = (Uint32(1) << Uint32(token - 32)) - Uint32(1)
                rank = Int32(_popc(low) + _popc(mask[2 * e + 1] & below))
            flat[eoff[e] + rank] = r

    @cute.jit
    def ranges(self, scal, units, cta):
        """This CTA's FC1 stage ranges of halves A and B, and their FC2 job
        counts (claimed at run time)."""
        split = units
        if units > SPLIT_UNITS:
            split = (units + 1) // 2
        scal[S_UNITS] = units
        scal[S_SPLIT] = split
        s1a = split * self.unit1
        s1b = (units - split) * self.unit1
        j2a = split * self.tiles2
        j2b = (units - split) * self.tiles2
        scal[S_S1A] = s1a
        scal[S_S1B] = s1b
        lo1a = cta * s1a // self.grid
        lo1b = cta * s1b // self.grid
        scal[S_LO1A] = lo1a
        scal[S_LO1B] = lo1b
        scal[S_J2A] = j2a
        scal[S_J2B] = j2b
        scal[S_N1A] = (cta + 1) * s1a // self.grid - lo1a
        scal[S_N1B] = (cta + 1) * s1b // self.grid - lo1b

    @cute.jit
    def locate(self, scal, n):
        """(half, unit, row tile, K block, stage index in its half) of this
        CTA's n-th FC1 stage. A unit's FC1 stage r holds K blocks
        DEPTH * ((r % span) // 2) on of pair r // span, up tile then gate tile."""
        n1a = scal[S_N1A]
        half = Int32(0)
        unit = Int32(0)
        s = Int32(0)
        if n < n1a:
            s = scal[S_LO1A] + n
            unit = s // self.unit1
        else:
            half = Int32(1)
            s = scal[S_LO1B] + n - n1a
            unit = scal[S_SPLIT] + s // self.unit1
        r = s % self.unit1
        tile = r // self.span + self.pairs * (r % 2)
        kb = DEPTH * ((r % self.span) // 2)
        return half, unit, tile, kb, s

    @cute.jit
    def describe(self, scal, n):
        """FC1 stage n packed: unit (bits 0-9), row tile (10-15), K block
        (16-21), FC2 (22), half (23), first (24) and last (25) stage of a pair
        piece or an FC2 job; END marks the end of the stages."""
        half, unit, tile, kb, s = self.locate(scal, n)
        n1a = scal[S_N1A]
        n1 = n1a + scal[S_N1B]
        opens = (n == 0) | (n == n1a) | (s % self.span == 0)
        closes = (n == n1a - 1) | (n == n1 - 1) | (s % self.span == self.span - 1)
        word = Uint32(unit) | (Uint32(tile) << 10) | (Uint32(kb) << 16)
        word = word | (Uint32(half) << 23)
        return word | (Uint32(opens) << 24) | (Uint32(closes) << 25)

    @cute.jit
    def unpack(self, word):
        """(FC2, half, unit, row tile, K block, opens, closes) of a stage."""
        return (
            ((word >> 22) & 1) != 0,
            Int32((word >> 23) & 1),
            Int32(word & 0x3FF),
            Int32((word >> 10) & 0x3F),
            Int32((word >> 16) & 0x3F),
            ((word >> 24) & 1) != 0,
            ((word >> 25) & 1) != 0,
        )

    @cute.jit
    def produce(self, sm, maps, full, empty, posted, scal, jobs):
        """Weights of this CTA's FC1 stages, then of the FC2 jobs it claims from
        the per-half job counters (FC2 balances FC1's uneven finish), then one
        END stage per gatherer."""
        n1 = scal[S_N1A] + scal[S_N1B]
        for n in cutlass.range(n1, unroll=1):
            self.post(sm, maps, full, empty, posted, n, self.describe(scal, n))
        n = n1
        half = Int32(0)
        claim = Int32(_atomic_add(jobs, Uint32(self.chunk)))
        while half < 2:
            count = scal[S_J2A]
            first = Int32(0)
            if half == 1:
                count = scal[S_J2B]
                first = scal[S_SPLIT]
            if claim < count:
                start = claim
                # The next claim is in flight while these jobs' stages load.
                claim = Int32(_atomic_add(jobs + half, Uint32(self.chunk)))
                for job in cutlass.range(
                    start, cutlass.min(start + self.chunk, count), unroll=1
                ):
                    unit = first + job // self.tiles2
                    tile = job % self.tiles2
                    for k in cutlass.range_constexpr(self.job):
                        word = Uint32(unit) | (Uint32(tile) << 10)
                        word = word | (Uint32(DEPTH * k) << 16) | (Uint32(1) << 22)
                        if cutlass.const_expr(k == 0):
                            word = word | (Uint32(1) << 24)
                        if cutlass.const_expr(k == self.job - 1):
                            word = word | (Uint32(1) << 25)
                        self.post(sm, maps, full, empty, posted, n, word)
                        n = n + 1
            else:
                half = half + 1
                if half < 2:
                    claim = Int32(_atomic_add(jobs + half, Uint32(self.chunk)))
        for g in cutlass.range_constexpr(GATHERERS):
            self.post(sm, maps, full, empty, posted, n + g, Uint32(END))

    @cute.jit
    def post(self, sm, maps, full, empty, posted, n, word):
        """Post stage n's descriptor, then load its weights (END: arrive only)."""
        desc = view(sm.iterator, self.off_desc, Uint32, self.posts)
        slot = n % self.stages
        if n >= self.stages:
            wait(empty + slot, ((n // self.stages) - 1) % 2, scope="cta")
        desc[n % self.posts] = word
        cute.arch.mbarrier_arrive(posted + n % self.posts)
        if word == Uint32(END):
            cute.arch.mbarrier_arrive(full + slot)
        else:
            index, tile, chunk = self.weight_box(sm, word)
            cute.arch.mbarrier_arrive_and_expect_tx(full + slot, self.a_bytes)
            for d in cutlass.range_constexpr(DEPTH):
                load_tma3(
                    sm.iterator + self.off_w + slot * self.a_bytes + d * STAGE_BYTES,
                    maps.iterator + 128 * index, full + slot, Int32(tile * TILE),
                    chunk + d,
                )  # fmt: skip

    @cute.jit
    def weight_box(self, sm, word):
        """(tensor map, row tile, first K block chunk) of a stage's weights."""
        uexp = view(sm.iterator, self.off_uexp, Int32, self.max_units)
        fc2, _, unit, tile, kb, _, _ = self.unpack(word)
        expert = uexp[unit]
        index = Int32(fc2)
        if expert < 0:  # the shared expert's maps
            index = index + 2
            expert = Int32(0)
        chunk = expert * self.nk1 + kb
        if fc2:
            chunk = expert * self.nk2 + kb
        return index, tile, chunk

    @cute.jit
    def weight_scale(self, sm, word, d, w13_scale, w2_scale, s13_scale, s2_scale):
        """The FP32 weight block scale of K block d of a stage."""
        uexp = view(sm.iterator, self.off_uexp, Int32, self.max_units)
        fc2, _, unit, tile, kb, _, _ = self.unpack(word)
        expert = uexp[unit]
        safe = cutlass.max(expert, Int32(0))
        value = Float32(0.0)
        if fc2:
            index = (Int64(safe) * self.tiles2 + tile) * self.nk2 + kb + d
            if expert < 0:
                value = _load_nc(s2_scale.iterator + index)
            else:
                value = _load_nc(w2_scale.iterator + index)
        else:
            index = (Int64(safe) * 2 * self.pairs + tile) * self.nk1 + kb + d
            if expert < 0:
                value = _load_nc(s13_scale.iterator + index)
            else:
                value = _load_nc(w13_scale.iterator + index)
        return value

    @cute.jit
    def gather(
        self, sm, full, empty, posted, plan_x, plan_s, hidden, hscale, unit_ready,
        lane, w13_scale, w2_scale, s13_scale, s2_scale, gatherer,
    ):  # fmt: skip
        """Stages ``gatherer``, ``gatherer + GATHERERS``, ... up to END: each
        stage's B tile, the unit's token rows of K blocks kb.. (FC1: the
        quantized tokens, FC2: the unit's hidden), 128B-swizzled (16B chunk c of
        row j at c ^ (j % 8)); and its column scales (weight block scale times
        the token's activation or hidden scale) into the scale ring, which the
        epilogue reads at most ``stages + slots`` stages later."""
        uoff = view(sm.iterator, self.off_uoff, Int32, self.max_units + 1)
        ucnt = view(sm.iterator, self.off_ucnt, Int32, self.max_units)
        flat = view(sm.iterator, self.off_flat, Int32, self.routes + self.tokens)
        hs = view(
            sm.iterator, self.off_hs + gatherer * self.width * self.nk2 * 4, Float32,
            self.width * self.nk2,
        )  # fmt: skip
        ring = view(sm.iterator, self.off_ring, Float32, self.ring * self.columns)
        desc = view(sm.iterator, self.off_desc, Uint32, self.posts)
        # Per unit: the global row of this lane's 16-byte chunks lane + 32 k (row
        # (lane + 32 k) // 8 of the unit, -1 past its tokens) and the token of
        # its column ``lane`` (scales).
        chunks = GROUP * 8 // 32
        rows = cute.make_rmem_tensor(cute.make_layout((chunks,)), Int64)
        current = Int32(-1)  # unit (FC2: + max_units) of ``rows``
        token_scales = Int32(0)
        count = Int32(0)
        n = Int32(gatherer)
        word = Uint32(0)
        while word != Uint32(END):
            slot = n % self.stages
            if n >= self.stages:
                wait(empty + slot, ((n // self.stages) - 1) % 2, scope="cta")
            wait(posted + n % self.posts, (n // self.posts) % 2, scope="cta")
            word = Uint32(desc[n % self.posts])
            if word != Uint32(END):
                fc2, _, unit, _, kb, _, _ = self.unpack(word)
                # Lane d < DEPTH: the weight block scale of K block kb + d.
                ws = Float32(0.0)
                if lane < DEPTH:
                    ws = self.weight_scale(
                        sm, word, lane, w13_scale, w2_scale, s13_scale, s2_scale
                    )  # fmt: skip
                key = unit
                if fc2:
                    key = unit + self.max_units
                if key != current:
                    count = ucnt[unit]
                    first = uoff[unit]
                    if fc2:
                        if lane == 0:
                            while _load_acquire(unit_ready + unit) < Uint32(
                                self.pairs
                            ):  # fmt: skip
                                pass
                        cute.arch.sync_warp()
                        _fence()
                        for i in cutlass.range(lane, count * self.nk2, 32, unroll=1):
                            hs[i] = _load_l2(
                                hscale.iterator
                                + Int64(unit) * self.width * self.nk2 + i
                            )  # fmt: skip
                        cute.arch.sync_warp()
                    for k in cutlass.range_constexpr(chunks):
                        j = (lane + 32 * k) // 8
                        rows[k] = Int64(-1)
                        if j < count:
                            rows[k] = (Int64(unit) * self.width + j) * self.inter
                            if not fc2:
                                token = flat[first + j]
                                if unit >= self.shared_units:
                                    token = token // self.top_k
                                rows[k] = Int64(token) * self.hidden
                    if lane < count:
                        token = flat[first + lane]
                        if unit >= self.shared_units:
                            token = token // self.top_k
                        token_scales = token * self.nk1
                    current = key
                # The tokens' activation scales (FC1), in flight with ``ws``.
                xs = cute.make_rmem_tensor(cute.make_layout((DEPTH,)), Float32)
                for d in cutlass.range_constexpr(DEPTH):
                    xs[d] = Float32(0.0)
                    if not fc2 and lane < count:
                        xs[d] = _load_l2(plan_s.iterator + token_scales + kb + d)
                dst = sm.iterator + self.off_b + slot * self.b_bytes
                base = plan_x.iterator
                if fc2:
                    base = hidden.iterator
                for k in cutlass.range_constexpr(chunks):
                    if rows[k] >= 0:
                        i = lane + 32 * k
                        j = i // 8
                        c = i % 8
                        for d in cutlass.range_constexpr(DEPTH):
                            copy_g2s(
                                dst + (d * self.width + j) * TILE + (c ^ (j % 8)) * 16,
                                base + rows[k] + (kb + d) * TILE + c * 16,
                            )
                for d in cutlass.range_constexpr(DEPTH):
                    w_d = cute.arch.shuffle_sync(ws, d)
                    if lane < count:
                        scale = xs[d]
                        if fc2:
                            scale = hs[lane * self.nk2 + kb + d]
                        index = (n % self.ring) * self.columns + d * self.width + lane
                        ring[index] = w_d * scale
            cute.arch.cp_async_mbarrier_arrive_noinc(full + slot)
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(full + slot)
            n = n + GATHERERS

    @cute.jit
    def issue(self, sm, tmem, full, empty, acc_full, acc_empty):
        desc = view(sm.iterator, self.off_desc, Uint32, self.posts)
        adesc = smem_desc(sm.iterator + self.off_w, True)
        bdesc = smem_desc(sm.iterator + self.off_b, True)
        n = Int32(0)
        word = Uint32(0)
        while word != Uint32(END):
            slot = n % self.stages
            acc = n % self.slots
            wait_tmem(full + slot, (n // self.stages) % 2, scope="cta")
            word = Uint32(desc[n % self.posts])
            if n >= self.slots:
                wait_tmem(acc_empty + acc, ((n // self.slots) - 1) % 2, scope="cta")
            if word == Uint32(END):
                if cute.arch.lane_idx() == 0:
                    cute.arch.mbarrier_arrive(acc_full + acc)
            else:
                # The gathered B rows are generic-proxy writes.
                cute.arch.fence_proxy("async.shared", space="cta")
                for d in cutlass.range_constexpr(DEPTH):
                    _umma(
                        tmem + acc * self.columns + d * self.width,
                        adesc + ((slot * self.a_bytes + d * STAGE_BYTES) >> 4),
                        bdesc + ((slot * self.b_bytes + d * self.width * TILE) >> 4),
                        Uint32(IDESC),
                    )  # fmt: skip
                _commit(empty + slot)
                cute.arch.sync_warp()
                _commit(acc_full + acc)
            n = n + 1

    @cute.jit
    def epilogue(
        self, sm, tmem, warp, lane, cta, e, acc_full, acc_empty, scal, rows, out,
        shared_out, hidden, hscale, partial, pair_count, unit_ready,
    ):  # fmt: skip
        w = self.width
        quadrant = warp % 4
        row = quadrant * 32 + lane
        logical = rows[row]  # this thread's output row of every row tile
        lanes = tmem + (Uint32(quadrant * 32) << 16)
        uexp = view(sm.iterator, self.off_uexp, Int32, self.max_units)
        uoff = view(sm.iterator, self.off_uoff, Int32, self.max_units + 1)
        ucnt = view(sm.iterator, self.off_ucnt, Int32, self.max_units)
        flat = view(sm.iterator, self.off_flat, Int32, self.routes + self.tokens)
        ring = view(sm.iterator, self.off_ring, Float32, self.ring * self.columns)
        desc = view(sm.iterator, self.off_desc, Uint32, self.posts)
        up = cute.make_rmem_tensor(cute.make_layout((w,)), Float32)
        gate = cute.make_rmem_tensor(cute.make_layout((w,)), Float32)
        start = Int32(0)  # first FC1 stage of the current piece
        n = Int32(0)
        word = Uint32(0)
        while word != Uint32(END):
            acc = n % self.slots
            wait_tmem(acc_full + acc, (n // self.slots) % 2, scope="cta")
            word = Uint32(desc[n % self.posts])
            if word != Uint32(END):
                fc2, half, unit, tile, kb, opens, closes = self.unpack(word)
                count = ucnt[unit]
                expert = uexp[unit]
                shared = expert < 0
                first = uoff[unit]
                into_gate = tile >= self.pairs
                if fc2:
                    into_gate = tile < 0
                if opens:
                    start = n
                    for j in cutlass.range_constexpr(w):
                        up[j] = Float32(0.0)
                        gate[j] = Float32(0.0)
                parts = load_tmem(lanes + acc * self.columns, count=self.columns)
                scales = ring.iterator + (n % self.ring) * self.columns
                if into_gate:
                    for c in cutlass.range_constexpr(self.columns):
                        gate[c % w] = gate[c % w] + parts[c] * scales[c]
                else:
                    for c in cutlass.range_constexpr(self.columns):
                        up[c % w] = up[c % w] + parts[c] * scales[c]
                fence_tmem(before=True)
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(acc_empty + acc)
                if closes:
                    if fc2:
                        target = out.iterator + tile * TILE + logical
                        if shared:
                            target = shared_out.iterator + tile * TILE + logical
                        for j in cutlass.range_constexpr(w):
                            if j < count:
                                _store_bf16(
                                    target + Int64(flat[first + j]) * self.hidden,
                                    up[j],
                                )
                    else:
                        self.close_pair(
                            sm, cta, e, row, lane, quadrant, half, unit, n, start,
                            count, up, gate, scal, logical, hidden, hscale,
                            partial, pair_count, unit_ready,
                        )  # fmt: skip
            n = n + 1

    @cute.jit
    def owner(self, s, size):
        """The CTA whose range of a ``size``-stage half holds stage ``s``."""
        return ((s + 1) * self.grid - 1) // size

    @cute.jit
    def close_pair(
        self, sm, cta, e, row, lane, quadrant, half, unit, n, opened, count, up,
        gate, scal, logical, hidden, hscale, partial, pair_count, unit_ready,
    ):  # fmt: skip
        """End of this CTA's piece (stages ``opened``..``n``) of a pair: finish
        the pair, or publish the partial and finish the pair if it is the pair's
        last piece."""
        w = self.width
        _, _, _, _, s = self.locate(scal, n)
        start = s - (n - opened)
        base = s - s % self.span  # the pair's first stage in its half
        pair = s % self.unit1 // self.span
        if start == base and s == base + self.span - 1:
            self.finish(sm, e, row, lane, quadrant, unit, pair, count, up, gate,
                        logical, hidden, hscale, unit_ready)  # fmt: skip
        else:
            size = scal[S_S1A]
            if half == 1:
                size = scal[S_S1B]
            first = self.owner(base, size)
            last = self.owner(base + self.span - 1, size)
            arrivals = pair_count + unit * self.pairs + pair
            if cta != first:
                # The pair is the first of this CTA's range: publish the partial
                # for the owner, the CTA holding the pair's first stage, which
                # reaches it last (at the end of its range).
                piece = (
                    partial.iterator + (Int64(half) * self.grid + cta) * 2 * w * TILE
                )
                for j in cutlass.range_constexpr(w):
                    if j < count:
                        _store_f32(piece + j * TILE + row, up[j])
                        _store_f32(piece + (w + j) * TILE + row, gate[j])
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                if e == 0:
                    # Cumulative: releases the stores the barrier ordered first.
                    _fence()
                    _red_add(arrivals, Uint32(1))
            else:
                if e == 0:
                    while _load_acquire(arrivals) < Uint32(last - first):
                        pass
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                # Pieces in CTA order, the owner's first.
                for c in cutlass.range(first + 1, last + 1, unroll=1):
                    peer = (
                        partial.iterator + (Int64(half) * self.grid + c) * 2 * w * TILE
                    )
                    for j in cutlass.range_constexpr(w):
                        if j < count:
                            up[j] = up[j] + _load_l2(peer + j * TILE + row)
                            gate[j] = gate[j] + _load_l2(peer + (w + j) * TILE + row)
                self.finish(sm, e, row, lane, quadrant, unit, pair, count, up, gate,
                            logical, hidden, hscale, unit_ready)  # fmt: skip

    @cute.jit
    def finish(self, sm, e, row, lane, quadrant, unit, pair, count, up, gate,
               logical, hidden, hscale, unit_ready):  # fmt: skip
        """Clamped SwiGLU of hidden block ``pair``, FP8 per 128 per token (scale
        amax / 448), published for FC2."""
        w = self.width
        amax_buf = view(sm.iterator, self.off_amax, Float32, 4 * w)
        limit = Float32(self.limit)
        # Every column (past ``count``: computed, never stored), so the
        # tokens' chains interleave.
        for j in cutlass.range_constexpr(w):
            g = cute.arch.fmin(gate[j], limit)
            u = cute.arch.fmax(cute.arch.fmin(up[j], limit), -limit)
            up[j] = g / (Float32(1.0) + cute.math.exp(-g)) * u
            bits = redux(float_bits(cute.arch.fmax(up[j], -up[j])), "max")
            if lane == 0:
                amax_buf[quadrant * w + j] = bits_float(bits)
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        rows = hidden.iterator + Int64(unit) * w * self.inter + pair * TILE + logical
        scales = hscale.iterator + Int64(unit) * w * self.nk2 + pair
        for j in cutlass.range_constexpr(w):
            if j < count:
                amax = cute.arch.fmax(
                    cute.arch.fmax(amax_buf[j], amax_buf[w + j]),
                    cute.arch.fmax(amax_buf[2 * w + j], amax_buf[3 * w + j]),
                )
                scale = cute.arch.fmax(amax, Float32(1e-10)) / Float32(FP8_MAX)
                byte = pack_fp8(
                    (up[j] / scale, Float32(0.0), Float32(0.0), Float32(0.0))
                )
                _store_u8(rows + j * self.inter, byte & Uint32(0xFF))
                if e == 0:
                    _store_f32(scales + j * self.nk2, scale)
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        if e == 0:
            _fence()
            _red_add(unit_ready + unit, Uint32(1))


_workspaces: dict = {}
# Compiled kernels by shape, shared by every layer.
_compiled: dict = {}


def _workspace(device, experts, top_k, hidden, inter, grid):
    """Scratch every layer's calls share (they run one after another): the token
    plan, hidden blocks, FC1 partials and the counters, sized for 64 tokens."""
    if device.index not in _workspaces:
        w, nk1, nk2 = GROUP, hidden // TILE, inter // TILE
        units = unit_capacity(experts, top_k)
        sizes = {
            "plan_x": MAX_TOKENS * hidden,
            "plan_s": MAX_TOKENS * nk1 * 4,
            "hidden": units * w * inter,
            "hscale": units * w * nk2 * 4,
            "partial": 2 * grid * 2 * w * TILE * 4,
        }
        _workspaces[device.index] = {
            **{k: torch.empty(v, dtype=torch.uint8, device=device)
               for k, v in sizes.items()},
            "counters": torch.zeros(
                2 * counter_words(experts, top_k, nk2) + 1, dtype=torch.int32,
                device=device,
            ),
        }  # fmt: skip
    return _workspaces[device.index]


class GroupedExperts:
    """Routed experts of 8-64 tokens and the shared expert: routing, GEMM1,
    clamped SwiGLU, FP8 hidden and GEMM2 of every distinct expert in one
    persistent kernel."""

    def __init__(
        self, w13, w13_scale, w2, w2_scale, top_k, limit, scale, ue8m0, shared
    ):
        """``shared``: the shared expert's TRTLLM-Gen-layout (w13, FP32 scales, w2,
        FP32 scales) with one expert each (see ``shared_weights``)."""
        experts, nk1, n1, _ = w13.shape
        self.hidden, self.inter, self.experts = nk1 * TILE, n1 // 2, experts
        self.top_k, self.limit, self.scale, self.ue8m0 = top_k, limit, scale, ue8m0
        self.w13, self.w2 = w13, w2
        self.w13_scale = w13_scale.contiguous()
        self.w2_scale = w2_scale.contiguous()
        self.shared = shared
        self.maps = torch.cat(
            [
                _weight_map(w13, TILE),
                _weight_map(w2, TILE),
                _weight_map(shared[0], TILE),
                _weight_map(shared[2], TILE),
            ]
        )
        self.shared_scales = (shared[1].contiguous(), shared[3].contiguous())
        self.shared_bytes = tuple(shared[i].view(torch.uint8).view(-1) for i in (0, 2))
        # Every row tile of both GEMMs shares one in-tile row shuffle.
        self.rows = _logical_rows(TILE, w13.device)
        for n in (n1, w2.shape[2]):
            tiles = _logical_rows(n, w13.device).view(-1, TILE)
            offsets = torch.arange(0, n, TILE, device=w13.device, dtype=torch.int32)
            assert torch.equal(tiles - offsets[:, None], self.rows.expand_as(tiles))
        self.grid = torch.cuda.get_device_properties(w13.device).multi_processor_count
        self.compiled = _compiled.setdefault(
            (self.hidden, self.inter, experts, top_k, limit, scale, ue8m0, self.grid),
            {},
        )

    def kernel(self, tokens, finalize):
        return _GroupedKernel(
            hidden=self.hidden, inter=self.inter, experts=self.experts,
            top_k=self.top_k, limit=self.limit, scale=self.scale, ue8m0=self.ue8m0,
            tokens=tokens, grid=self.grid, finalize=finalize,
        )  # fmt: skip

    def prepare(
        self, tokens: list[int], logits_dtype: torch.dtype, finalized=()
    ) -> None:
        """Compiles the kernel for these token counts before graph capture;
        ``finalized``: the token counts that run with ``finalize``."""
        device = self.w13.device
        bias = torch.zeros(self.experts, device=device)
        for m in tokens:
            x = torch.zeros(m, self.hidden, dtype=torch.bfloat16, device=device)
            logits = torch.zeros(m, self.experts, dtype=logits_dtype, device=device)
            self(x, logits, bias, finalize=m in finalized)

    def __call__(self, x, logits, bias, finalize=False):
        """BF16 [M * top_k, hidden] routed expert outputs (one row per route),
        int32 [M, top_k] experts, FP32 [M, top_k] routing weights and the shared
        expert's BF16 [M, hidden] output; with ``finalize``, the BF16 [M, hidden]
        MoE output (routing-weighted routes plus the shared expert) instead.

        Args:
            x: [M, hidden] BF16 tokens.
            logits: [M, experts] router logits.
            bias: [experts] FP32 score correction bias.
            finalize: whether to reduce the routes in the kernel.

        """
        m = x.shape[0]
        device = x.device
        ws = _workspace(
            device, self.experts, self.top_k, self.hidden, self.inter, self.grid
        )
        out = torch.empty(m * self.top_k, self.hidden, dtype=torch.bfloat16,
                          device=device)  # fmt: skip
        shared = torch.empty(m, self.hidden, dtype=torch.bfloat16, device=device)
        ids = torch.empty(m, self.top_k, dtype=torch.int32, device=device)
        weights = torch.empty(m, self.top_k, dtype=torch.float32, device=device)
        final = torch.empty_like(shared) if finalize else shared
        args = (
            self.maps, self.w13_scale, self.w2_scale, *self.shared_scales,
            *self.shared_bytes, x, logits, bias, self.rows, out, shared,
            final, ids.view(-1), weights.view(-1), ws["plan_x"],
            ws["plan_s"].view(torch.float32), ws["hidden"],
            ws["hscale"].view(torch.float32), ws["partial"].view(torch.float32),
            ws["counters"],
        )  # fmt: skip
        st = stream(device)
        key = (m, logits.dtype, finalize)
        if key not in self.compiled:
            self.compiled[key] = cute.compile(
                self.kernel(m, finalize), *(to_cute(t, 16) for t in args), st
            )
        self.compiled[key](*(to_cute(t, 16) for t in args), st)
        if finalize:
            return final
        return out, ids, weights, shared


def shared_weights(w13, w13_scale, w2, w2_scale):
    """The shared expert ([gate; up] FP8 [2I, H] and [H, I] with FP32 block
    scales) as a one-expert TRTLLM-Gen BlockMajorK layout."""
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        _shuffle_deepseek_fp8_moe_weights,
        swap_w13_to_w31,
    )

    w31 = swap_w13_to_w31(w13[None])
    w31_shuffled, w2_shuffled = _shuffle_deepseek_fp8_moe_weights(w31, w2[None])
    return (
        w31_shuffled,
        swap_w13_to_w31(w13_scale[None]).contiguous(),
        w2_shuffled,
        w2_scale[None].contiguous(),
    )
