# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent routed experts: TMA producer, MXFP8 activations, FP4 x MXFP8 FC1/FC2 MMAs,
SiTU hidden exchange and output epilogue. Capacity <= 8 keeps the batch's activations in
SMEM; capacity > 8 (grouped) runs (expert, <= 8 tokens) units whose FC1 stages gather
8-row slices of the grid-quantized batch."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int8, Int16, Int32, Int64, Uint8, Uint32, Uint64

from ..common.primitives import (
    acquire_tensormap,
    alloc,
    asm,
    bf16,
    bf16x2,
    bits_float,
    commit,
    copy_evict,
    copy_g2s,
    fence_tmem,
    float_bits,
    fma,
    fp8_bits,
    group_barrier,
    launch,
    load_tma3,
    load_tmem,
    load_volatile,
    load_words,
    multicast_words,
    mx_exponent,
    mx_rescale,
    pack_fp8,
    pack_pair,
    release_tmem,
    shared_address,
    situ,
    smem_desc,
    smem_words,
    store_shared_words,
    store_words,
    thread,
    unpack_bf16,
    view,
    wait,
    wait_tmem,
)
from ..common.twoshot import push

shuffle = cute.arch.shuffle_sync


class ExpertsGeometry:
    """Static plain-input geometry. No cluster-local or multicast storage. The workspace
    is capacity-independent; capacity > 8 swaps the batch's SMEM activation image for
    per-slot 8-row images, which frees a sixth weight stage, and keeps per-CTA windows
    of the unit tables (expert, 8 pair rows) in the misc block."""

    def __init__(self, tp, capacity=8):
        self.tp = tp
        self.tiles, self.w13_rows, self.h_chunks, self.fc2_stages = (
            (6, 768, 3, 2) if tp == 8 else (3, 512, 2, 1)
        )
        self.grouped = capacity > 8
        self.stages = stages = 6 if self.grouped else 5
        self.units = 16 * capacity if self.grouped else 128
        self.w13_scale_bytes = self.w13_rows * 112
        self.w2_scale_columns = 4 * self.h_chunks
        self.w2_scale_bytes = 3584 * self.w2_scale_columns
        self.w2_tile_scale_bytes = 128 * self.w2_scale_columns
        self.h_image_bytes = self.h_chunks * 1024
        self.h_scale_bytes = self.h_chunks * 128 + 384
        self.h_compact_bytes = self.h_image_bytes + 128
        self.off_sfa = stages * 32768
        self.off_x = self.off_sfa + stages * 1024
        self.off_h = self.off_x + (stages * 2048 if self.grouped else 28672)
        self.off_xsf = self.off_h + 2 * self.h_image_bytes
        self.off_hsf = self.off_xsf + (stages * 256 if self.grouped else 28 * 128) + 384
        self.off_misc = self.off_hsf + 2 * self.h_scale_bytes
        self.smem_bytes = self.off_misc + 5120
        # Misc barriers (byte offsets): weight full/empty [stages], accumulator
        # full/empty [16], hidden full/empty [2], x ready [14]. Grouped stage metadata
        # and unit windows (4 x 16 units) use the B<=8 routing tables' space.
        self.bar_full, self.bar_empty, self.bar_accfull = (0, 8 * stages, 16 * stages)
        acc = self.bar_accfull
        self.bar_accempty, self.bar_hfull = (acc + 128, acc + 256)
        self.bar_hempty, self.bar_xready = (acc + 272, acc + 288)
        self.off_meta = self.off_misc + (1152 if self.grouped else 520)
        self.off_window = self.off_misc + 1280
        # Stealing: FC2 unit queue (producer -> hidden loader) [8], full/empty [8, 8].
        self.off_queue, self.bar_queue = (self.off_misc + 4608, 4640)
        self.ws_partial = 2 * 128 * self.h_compact_bytes
        self.ws_counter = self.ws_partial + 2 * 160 * 8 * 128 * 4
        self.ws_xflag = self.ws_counter + 160 * 4
        self.ws_stage = (self.ws_xflag + 160 * 4 + 255) // 256 * 256
        # Publish stage rows [64 tokens x 16 slots, 3584] BF16, then grouped: call
        # counters [160] and unit counts [2], hidden images [2, 1024].
        self.ws_calls = self.ws_stage + 1024 * 3584 * 2
        self.ws_units = self.ws_calls + 1024
        # FC2(B) tail claim counters [2 parities, 160] (``ExpertsKernel.steal``).
        self.ws_steal = self.ws_units + 2 * 1024 * self.h_compact_bytes
        self.workspace_bytes = self.ws_steal + 2 * 160 * 4

    def image(self, workspace, parity, unit):
        """Compact hidden image of ``unit`` in the ``parity`` half of the workspace."""
        if self.grouped:
            base = workspace.iterator + self.ws_units
            return base + (Int64(parity) * 1024 + unit) * self.h_compact_bytes
        return workspace.iterator + (Int64(parity) * 128 + unit) * self.h_compact_bytes


@cute.jit
def swizzled(chunk, row, base=0):
    """Byte offset of 16-byte ``chunk`` of ``row`` in a 128B-swizzled 8-row image."""
    return base + chunk // 8 * 1024 + row * 128 + (chunk % 8 ^ row) * 16


@cute.jit
def has_ff_byte(word: Uint32):
    value = ~word
    return value - Uint32(16843009) & ~value & Uint32(2155905152) != 0


@cute.jit
def mx_block(words):
    """MXFP8 of one 32-BF16 block (16 words): eight e4m3x4 words and the E8M0 scale."""
    values = cute.make_rmem_tensor(32, Float32)
    for i in cutlass.range_constexpr(16):
        pair = unpack_bf16(words[i])
        values[2 * i], values[2 * i + 1] = (pair[0], pair[1])
    maxima = cute.make_rmem_tensor(16, Float32)
    for j in cutlass.range_constexpr(16):
        maxima[j] = cute.arch.fmax(abs(values[j]), abs(values[j + 16]))
    for width in (8, 4, 2, 1):
        for j in cutlass.range_constexpr(width):
            maxima[j] = cute.arch.fmax(maxima[j], maxima[j + width])
    scale = mx_exponent(maxima[0])
    reciprocal = mx_rescale(scale)
    packed = [
        pack_fp8(tuple(values[4 * j + i] * reciprocal for i in range(4)))
        for j in range(8)
    ]
    return packed, scale


# Grouped expert plan, written before the experts run (by the routing kernel's copy
# role, or ExpertsPlanKernel): MXFP8 rows [64, 3584] then scale words [64, 28, 16 B]
# (bytes 0-3 used), the layout of the experts' per-slot SMEM images.
PLAN_XSF = 64 * 3584
PLAN_BYTES = PLAN_XSF + 64 * 448


@cute.jit
def plan_chunk(plan, token, chunk, words):
    """Write 8-element ``chunk`` (four BF16x2 ``words``) of ``token`` as MXFP8 into
    the plan; lanes 4k..4k+3 hold one 32-element block, 16 lanes one scale word.
    Bit-identical to ``mx_block`` (an exact max, the same scaling and packing)."""
    lane, values = (thread()[2], cute.make_rmem_tensor(8, Float32))
    for i in cutlass.range_constexpr(4):
        pair = unpack_bf16(words[i])
        values[2 * i], values[2 * i + 1] = (pair[0], pair[1])
    top = abs(values[0])
    for i in cutlass.range_constexpr(1, 8):
        top = cute.arch.fmax(top, abs(values[i]))
    for width in (1, 2):
        top = cute.arch.fmax(top, cute.arch.shuffle_sync_bfly(top, width))
    scale = mx_exponent(top)
    reciprocal = mx_rescale(scale)
    packed = [
        pack_fp8(tuple(values[4 * j + i] * reciprocal for i in range(4)))
        for j in range(2)
    ]
    store_words(plan.iterator + token * 3584 + chunk * 8, packed)
    word = Uint32(scale)
    for s in cutlass.range_constexpr(1, 4):
        word |= shuffle(Uint32(scale), lane + 4 * s) << 8 * s
    if chunk % 16 == 0:
        dst = plan.iterator + PLAN_XSF + token * 448 + chunk // 16 * 16
        store_words(dst, (word, Uint32(0), Uint32(0), Uint32(0)))


class ExpertsPlanKernel:
    """Standalone plan writer (no routing kernel in front): one CTA per token."""

    def __init__(self, batch):
        self.batch = batch

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.batch, 256)

    @cute.kernel
    def kernel(self, args):
        x, plan = args
        tid, _, _, token = thread()
        cute.arch.griddepcontrol_wait()
        for i in cutlass.range_constexpr(2):
            chunk = tid + 256 * i
            if chunk < 448:  # warp-uniform: chunks 256-447 are warps 0-5
                words = load_words(
                    x.iterator + token * 3584 + chunk * 8, count=4, mutable=True
                )
                plan_chunk(plan, token, chunk, words)
        cute.arch.griddepcontrol_launch_dependents()


@asm()
def stage(mma, weight_scales, activation_scales, release, done, *, kind):
    """One FC1/FC2 stage. ``mma``: (TMEM accumulator, weight desc, activation desc);
    scales: (TMEM column, desc[, activation copy flag]); ``release``: (FC1 accumulate
    or FC2 prior-H release, weight empty barrier); ``done``: accumulator barrier."""
    half = kind in ("fc2b", "fc2p")
    steps = 6 if kind == "fc2" else 4 if half else 8
    cp = "tcgen05.cp.cta_group::1.32x128b.warpx4"
    arrive = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = [
        "{ .reg .pred e,p0,pt,pc,pa,ph; .reg .b32 i<4>; .reg .b64 as1,bs1,bs2;",
        f".reg .b32 af1,bf1,bf2; .reg .b64 xa<{steps}>,xb<{steps}>;",
        "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 pc,$7,0;",
        "and.pred pc,pc,e; setp.ne.b32 pa,$10,0; and.pred pa,pa,e;",
    ]
    if kind == "fc1":
        code.append("setp.ne.b32 p0,$8,0;")
    elif half:
        code.append("setp.eq.b32 p0,0,0;")
    else:
        code.append("setp.eq.b32 p0,0,1; setp.ne.b32 ph,$8,0; and.pred ph,ph,e;")
        code.append(f"@ph {arrive} [$8];")
    code += [f"mov.b32 i{k},{142738048 | k << 29 | k << 4};" for k in range(4)]
    if not half:
        code.append("add.s64 as1,$4,32; add.s64 bs1,$6,8;")
        code.append("add.u32 af1,$3,4; add.u32 bf1,$5,4;")
        if kind == "fc2a":
            code.append("add.s64 bs2,$6,16; add.u32 bf2,$5,8;")
        code += [f"@pc {cp} [$5],$6;", f"@pc {cp} [bf1],bs1;"]
        if kind == "fc2a":
            code.append(f"@pc {cp} [bf2],bs2;")
    code.append(f"@e {cp} [$3],$4;")
    if not half:
        code.append(f"@e {cp} [af1],as1;")
    for k in range(steps):
        a = k // 4 * 1024 + k % 4 * 2 + (4 if kind == "fc2" and k >= 4 else 0)
        b = k // 4 * 64 + k % 4 * 2
        code.append(f"add.s64 xa{k},$1,{a}; add.s64 xb{k},$2,{b};")
    for k in range(steps):
        af, bf = ("$3", "$5") if k < 4 else ("af1", "bf1")
        code.append(
            "@e tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
            f"[$0],xa{k},xb{k},i{k % 4},[{af}],[{bf}],{'p0' if k == 0 else 'pt'};"
        )
    if kind != "fc2p":
        code.append(f"@e {arrive} [$9];")
    if kind != "fc2a":
        code.append(f"@pa {arrive} [$10];")
    code.append("}")
    kinds = "rllrlrlrrrr"
    operands = (*mma, *weight_scales, *activation_scales, *release, done)
    values = [(Uint64 if k == "l" else Uint32)(v) for k, v in zip(kinds, operands)]
    return "\n".join(code), ",".join(kinds) + ",~{memory}", values


@asm(Uint32)
def claim(pointer, value):
    """Relaxed GPU-scope fetch-add (FC2 tail claims): the counter before the add."""
    code = "atom.relaxed.gpu.global.add.u32 $0,[$1],$2;"
    return code, "=r,l,r,~{memory}", (pointer, Uint32(value))


def fc2_pairs(tp, capacity, streamed):
    """TP8 FC2 jobs are tile pairs on the incremental producer from capacity 4 (the
    grouped engine always streams): chunks 0-1 of tiles t and t + 1, then both tiles'
    chunk 2 through one 256-row box (map 5), so every stage is 16 KB."""
    return tp == 8 and streamed and capacity >= 4


class ExpertsKernel(ExpertsGeometry):
    """The engine for one capacity and persistent grid. CTAs wait on other clusters'
    partials (``load_hidden``, ``partials``, ``finalize``), so
    ``PersistentExperts.prepare`` checks the grid is co-resident. It is not launched
    cooperatively: that forfeits the PDL prologue overlap (1-2% decode at B4-B8).
    ``publish`` stages FC2 rows for ``finalize``: routing weights replace the output
    arg, and (latent multicast, parity, rank) follow the SiTU scalars; ``"twoshot"``:
    (receive slots [3 x TP, 64, 3584/TP], their owners' bases, parity x TP + rank) of
    ``common.twoshot`` instead. Capacity > 8 (grouped) takes the plan (``PLAN_BYTES``:
    MXFP8 x) in place of x. ``stream`` runs capacity <= 8 on the incremental producer
    and CTA-scoped waits (grouped always).
    ``steal`` (grouped) balances the end: each CTA's last ``zone`` FC2(B) grains (TP8 a
    tile pair, TP16 three tiles) are claimed by the owner from the front and by the
    ``victims`` CTAs before it from the back, through one counter per CTA re-armed by
    call parity; a stolen grain is computed exactly as by its owner."""

    def __init__(
        self, tp, capacity, experts, grid, publish=False,
        stream=True, steal=False,
    ):  # fmt: skip
        super().__init__(tp, capacity)
        self.steal = steal and self.grouped
        self.streamed = stream or self.grouped
        self.paired = fc2_pairs(tp, capacity, self.streamed)
        self.jobs = 14 if self.paired else 28  # FC2 jobs per unit
        self.grain, self.zone, self.victims = (1 if self.paired else 3, 8, 3)
        self.threads = 384
        self.capacity, self.experts, self.grid = (capacity, experts, grid)
        self.publish = publish
        assert not self.grouped or grid >= capacity <= 64
        self.width = min(capacity, 8)
        self.dense = capacity <= 4
        self.rounds = (112 * capacity + 319) // 320 if self.dense else 3

    def sync(self, barrier, phase=0, tmem=False):
        """Parity wait (then tcgen05 ordering if ``tmem``). Every barrier here is
        CTA-local, so grouped and streamed waits use CTA scope."""
        scope = "cta" if self.streamed else "cluster"
        (wait_tmem if tmem else wait)(barrier, phase, scope=scope)

    @cute.jit
    def route(self, sm, ids, counters, batch: Int32):
        """Dedupe after the dependency wait. Capacity <= 2: one work item per routing
        pair; else first occurrences across tokens in order, keeping token slots."""
        tid, warp, lane, cta = thread()
        misc = self.off_misc
        table = view(sm.iterator, 0, Int32, self.experts)
        expert, uof = (view(sm.iterator, misc + o, Int32, 128) for o in (1140, 1652))
        slots = view(sm.iterator, misc + 2176, Int8, 1024)
        slot_words = view(sm.iterator, misc + 2176, Uint32, 256)
        counts = view(sm.iterator, misc + 868, Int32, 4)
        distinct, parity = (view(sm.iterator, misc + o, Int32, 1) for o in (860, 864))
        for i in range(tid, self.experts, 384):
            table[i] = Int32(2147483647)
        myid, pairs = (Int32(-1), batch * 16)
        if tid < 128:
            lo, hi = (Uint32(4294967295), Uint32(4294967295))
            if tid < pairs:
                myid = ids[tid]
                if cutlass.const_expr(self.capacity <= 2):
                    expert[tid] = myid
                    if tid < 16:
                        lo = lo & Uint32(4294967040) | Uint32(tid % 16)
                    else:
                        lo = lo & Uint32(4294902015) | Uint32(tid % 16) << 8
            slot_words[2 * tid], slot_words[2 * tid + 1] = (lo, hi)
        if tid == 160:
            parity[0] = counters[cta]
        if cutlass.const_expr(self.capacity <= 2) and tid == 0:
            distinct[0] = pairs
        cute.arch.sync_threads()
        if cutlass.const_expr(self.capacity > 2):
            if tid < pairs:
                cute.arch.atomic_min(
                    table.iterator + myid, tid, sem="relaxed", scope="cta"
                )
            cute.arch.sync_threads()
            if tid < 128:
                firstk = Int32(-1)
                if tid < pairs:
                    firstk = table[myid]
                first = (tid < pairs) & (firstk == tid)
                ballot = Uint32(cute.arch.vote_ballot_sync(first))
                if lane == 0:
                    counts[warp] = Int32(cute.arch.popc(ballot))
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                base = Int32(0)
                for w in cutlass.range_constexpr(3):
                    if warp > w:
                        base += counts[w]
                rank = base + Int32(cute.arch.popc(ballot & (Uint32(1) << lane) - 1))
                if first:
                    uof[tid] = rank
                    expert[rank] = myid
                if tid == 0:
                    distinct[0] = counts[0] + counts[1] + counts[2] + counts[3]
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                if tid < pairs:
                    slots[uof[firstk] * 8 + tid // 16] = Int8(tid % 16)
        # Retire every thread's hash accesses before async weight-ring reuse.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        return (distinct[0], parity[0], expert)

    def tables(self, sm, window=True):
        """Grouped unit tables: (expert, eight pair rows (Int16, -1 padded), the rows
        as four words per unit) of every unit in the weight-ring scratch during
        routing, or of this CTA's windows. Views all start from the byte pointer."""
        base, units = (self.off_window, 64) if window else (12288, self.units)
        experts = view(sm.iterator, base, Int32, units)
        rows = view(sm.iterator, base + 4 * units, Int16, 8 * units)
        return experts, rows, view(sm.iterator, base + 4 * units, Uint32, 4 * units)

    @cute.jit
    def first_unit(self, ranges, fc2: cutlass.Constexpr, group):
        """First unit of this CTA's FC1 or FC2 range in ``group``."""
        first = ranges[group, 4] // self.jobs
        if cutlass.const_expr(not fc2):
            first = ranges[group, 0] + ranges[group, 2] // 14 // self.tiles
        return first

    @cute.jit
    def local(self, ranges, unit, fc2: cutlass.Constexpr, group):
        """Window index of ``unit`` (windows: FC1 A, FC1 B, FC2 A, FC2 B)."""
        return (2 * fc2 + group) * 16 + unit - self.first_unit(ranges, fc2, group)

    @cute.jit
    def windows(self, sm, ranges):
        """Copy this CTA's four unit windows (<= 16 units each) out of the scratch."""
        tid, _, _, cta = thread()
        experts, _, words = self.tables(sm, False)
        window_experts, _, window_words = self.tables(sm)
        for w in cutlass.range_constexpr(4):
            fc2, group = (w // 2, w % 2)
            lo, hi = (ranges[group, 2 + 2 * fc2], ranges[group, 3 + 2 * fc2])
            if cutlass.const_expr(self.steal and w == 3):
                hi = self.stolen_end(ranges, cta)
            last = (hi - 1) // self.jobs
            if cutlass.const_expr(not fc2):
                last = ranges[group, 0] + (hi - 1) // 14 // self.tiles
            unit = self.first_unit(ranges, fc2, group) + tid % 16
            if (tid // 16 == w) & (lo < hi) & (unit <= last):
                window_experts[tid] = experts[unit]
                for i in cutlass.range_constexpr(4):
                    window_words[4 * tid + i] = words[4 * unit + i]

    @cute.jit
    def stolen_end(self, ranges, cta):
        """End of the FC2(B) jobs of CTAs up to ``cta + victims`` (window 3's reach)."""
        stop = cta + self.victims + 1
        if stop > self.grid:
            stop = Int32(self.grid)
        count = ranges[1, 1] // (self.tiles * 14)
        return self.jobs * ranges[1, 0] + stop * self.jobs * count // self.grid

    @cute.jit
    def group(self, sm, ids, workspace, batch: Int32, scalars=None):
        """Grouped routing: per-expert token masks; expert e's
        tokens, in token order, fill ceil(n / 8) units numbered in expert order. Unit
        rows hold pair indices (token * 16 + slot), -1 when padded. Misc scalars:
        units, call, grouped call (``scalars`` given: the units only)."""
        tid, warp, lane, cta = thread()
        masks = view(sm.iterator, 0, Uint32, 2 * self.experts)
        base = view(sm.iterator, 8 * self.experts, Int32, self.experts)
        totals = view(sm.iterator, 12 * self.experts, Int32, self.threads // 32)
        experts, rows, row_words = self.tables(sm, False)
        counted = scalars is None
        if cutlass.const_expr(counted):
            scalars = view(sm.iterator, self.off_misc + 1140, Int32, 3)
        threads: cutlass.Constexpr = self.threads
        rounds: cutlass.Constexpr = (16 * self.capacity + threads - 1) // threads
        mine = cute.make_rmem_tensor(rounds, Int32)
        for k in cutlass.range_constexpr(rounds):
            mine[k] = Int32(0)
            if tid + self.threads * k < batch * 16:
                mine[k] = ids[tid + self.threads * k]
        if cutlass.const_expr(counted):
            if tid == 0:
                scalars[1] = view(workspace.iterator, self.ws_counter, Int32, 160)[cta]
            if tid == 1:
                scalars[2] = view(workspace.iterator, self.ws_calls, Int32, 160)[cta]
        for i in range(tid, 2 * self.experts, self.threads):
            masks[i] = Uint32(0)
        for i in range(tid, 4 * self.units, self.threads):
            row_words[i] = Uint32(4294967295)
        cute.arch.sync_threads()
        for k in cutlass.range_constexpr(rounds):
            pair = tid + self.threads * k
            if pair < batch * 16:
                token = pair >> 4
                bit = Uint32(1) << (token & 31)
                word = masks.iterator + 2 * mine[k] + (token >> 5)
                cute.arch.atomic_or(word, bit, sem="relaxed", scope="cta")
        cute.arch.sync_threads()
        span: cutlass.Constexpr = (self.experts + self.threads - 1) // self.threads
        units = cute.make_rmem_tensor(span, Int32)
        count = Int32(0)
        for j in cutlass.range_constexpr(span):
            units[j], e = (Int32(0), span * tid + j)
            if e < self.experts:
                n = cute.arch.popc(masks[2 * e]) + cute.arch.popc(masks[2 * e + 1])
                units[j] = Int32(n + 7) >> 3
            count += units[j]
        total = count
        for shift in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(total, 1 << shift)
            if lane >= 1 << shift:
                total += other
        if lane == 31:
            totals[warp] = total
        cute.arch.sync_threads()
        start = total - count
        for w in cutlass.range_constexpr(self.threads // 32 - 1):
            if warp > w:
                start += totals[w]
        if tid == self.threads - 1:
            scalars[0] = start + count
        for j in cutlass.range_constexpr(span):
            e = span * tid + j
            if e < self.experts:
                base[e] = start
                for v in range(units[j]):
                    experts[start + v] = e
            start += units[j]
        cute.arch.sync_threads()
        for k in cutlass.range_constexpr(rounds):
            pair = tid + self.threads * k
            if pair < batch * 16:
                e, token = (mine[k], pair >> 4)
                low, word = (masks[2 * e], masks[2 * e + 1])
                rank = Int32(0)
                if token < 32:
                    word = low
                else:
                    rank = Int32(cute.arch.popc(low))
                below = (Uint32(1) << (token & 31)) - 1
                rank += Int32(cute.arch.popc(word & below))
                rows[(base[e] + (rank >> 3)) * 8 + (rank & 7)] = Int16(pair)
        cute.arch.sync_threads()

    @cute.jit
    def gather_x(self, sm, plan, ranges, batch):
        """Warp 3 (grouped): cp.async each FC1 stage's 8-row MXFP8 slice and scale
        rows out of the plan (complete before the dependency wait returns) into its
        slot; full[slot] counts the 32 lanes besides the producer (which stands in for
        them on FC2 stages and the end). Padded rows repeat the unit's first token
        (their outputs are dropped)."""
        lane, ws = (thread()[2], plan.iterator)
        stages, rows = (self.stages, self.tables(sm)[1])
        full, empty = (
            view(sm.iterator, self.off_misc + o, Uint64, stages)
            for o in (self.bar_full, self.bar_empty)
        )
        chunk, lanes = (lane & 15, lane // 16)
        image = [self.off_x + swizzled(chunk, 2 * j + lanes) for j in range(4)]
        scale = self.off_xsf + lane // 8 * 128 + (lane & 7) * 16
        sources = cute.make_rmem_tensor(5, Int32)
        slot, phase, issued = (Int32(0), Int32(0), Int32(0))
        for group in cutlass.range_constexpr(2):
            lo, hi = (ranges[group, 2], ranges[group, 3])
            ks, r, index = (lo % 14, lo // 14 % self.tiles, Int32(16 * group))
            if lo < hi:
                self.row_sources(rows, index, sources, batch)
            for s in cutlass.range(lo, hi, unroll=1):
                if issued >= stages:
                    self.sync(empty.iterator + slot, phase ^ 1)
                for j in cutlass.range_constexpr(4):
                    src = sources[j] + ks * 256 + chunk * 16
                    copy_g2s(sm.iterator + image[j] + slot * 2048, ws + src)
                if lane < 16:
                    src = PLAN_XSF + sources[4] + (2 * ks + lane // 8) * 16
                    copy_g2s(sm.iterator + scale + slot * 256, ws + src)
                cute.arch.cp_async_mbarrier_arrive_noinc(full.iterator + slot)
                slot, issued = (slot + 1, issued + 1)
                if slot == stages:
                    slot, phase = (Int32(0), phase ^ 1)
                ks += 1
                if ks == 14:
                    ks, r = (Int32(0), r + 1)
                    if r == self.tiles:
                        r, index = (Int32(0), index + 1)
                        if s + 1 < hi:
                            self.row_sources(rows, index, sources, batch)

    @cute.jit
    def row_sources(self, rows, index, sources, batch):
        """Byte offsets of this lane's four MXFP8 rows and scale row of window unit
        ``index`` (padded rows repeat its first token)."""
        lane = thread()[2]
        for j in cutlass.range_constexpr(5):
            row = 2 * j + lane // 16 if j < 4 else lane & 7
            pair = Int32(rows[index * 8 + row])
            if pair < 0:
                pair = Int32(rows[index * 8])
            sources[j] = (pair >> 4) * (3584 if j < 4 else 448)

    @cute.jit
    def partition(self, distinct: Int32, cta: Int32):
        """FC1(A), FC1(B), FC2(A), FC2(B) work partitions."""
        middle = distinct
        if distinct > 40:
            middle = (distinct * 50 + 99) // 100
        ranges = cute.make_rmem_tensor((2, 6), Int32)
        for group in cutlass.range_constexpr(2):
            start, end = (Int32(0), middle) if group == 0 else (middle, distinct)
            count = end - start
            size = self.tiles * 14 * count
            ranges[group, 0], ranges[group, 1] = (start, size)
            ranges[group, 2] = cta * size // self.grid
            ranges[group, 3] = (cta + 1) * size // self.grid
            n = self.jobs
            ranges[group, 4] = n * start + cta * n * count // self.grid
            ranges[group, 5] = n * start + (cta + 1) * n * count // self.grid
        return ranges

    @cute.jit
    def schedule(self, ranges, experts, n: Int32):
        """Returns the stage's (kind, a, b, c) metadata and its TMA arguments."""
        n1a = ranges[0, 3] - ranges[0, 2]
        n1b = ranges[1, 3] - ranges[1, 2]
        n2a = ranges[0, 5] - ranges[0, 4]
        kind, a, b, c = (Int32(0), Int32(0), Int32(0), Int32(0))
        row, chunk, scale_bytes, scale_offset = (Int32(0), Int32(0), Int32(0), Int64(0))
        if n < n1a + n1b:
            group = Int32(n >= n1a)
            s = ranges[group, 2] + (n - n1a if group == 1 else n)
            tile = self.tiles * ranges[group, 0] + s // 14
            ks = s % 14
            expert, r = (experts[tile // self.tiles], tile % self.tiles)
            a, b = (tile, ks)
            opens = Int32((s == ranges[group, 2]) | (ks == 0))
            c = opens | Int32((s == ranges[group, 3] - 1) | (ks == 13)) << 1
            row, chunk = (expert * self.w13_rows + r * 128, 2 * ks)
            scale_offset = Int64(expert) * self.w13_scale_bytes + r * 14336 + ks * 1024
            scale_bytes = Int32(1024)
        else:
            step = n - n1a - n1b
            unit, half = (step // self.fc2_stages, step % self.fc2_stages)
            group = Int32(unit >= n2a)
            job = ranges[group, 4] + (unit - n2a if group == 1 else unit)
            u, tile = (job // 28, job % 28)
            expert = experts[u]
            kind = Int32(1) if half == 0 else Int32(3)
            a, b = (u, tile)
            row, chunk = (expert * 3584 + tile * 128, 2 * half)
            scale_offset = Int64(expert) * self.w2_scale_bytes
            scale_offset += tile * self.w2_tile_scale_bytes
            scale_offset += half * 1024
            scale_bytes = Int32(1024 if half == 0 else self.w2_tile_scale_bytes - 1024)
        return (kind, a, b, c), (row, chunk, scale_offset, scale_bytes)

    @cute.jit
    def poll_hidden(self, sm, source, buffer: Int32):
        """Poll real K data and scale bytes, preserving TP16 padding semantics."""
        lane = thread()[2]
        chunks: cutlass.Constexpr = self.h_image_bytes // 512
        values, scale = (cute.make_rmem_tensor((chunks, 4), Uint32), Uint32(0))
        valid_bytes = min(4, max(0, self.tiles * 2 - 4 * (lane & 3)))
        mask = Uint32(4294967295)
        if valid_bytes < 4:
            mask = (Uint32(1) << 8 * valid_bytes) - 1
        ready = False
        while not ready:
            bad = False
            for v in cutlass.range_constexpr(chunks):
                index = lane + 32 * v
                words = load_volatile(source + index * 16, count=4)
                for w in cutlass.range_constexpr(4):
                    values[v, w] = words[w]
                atom, token, position = (index >> 6, index >> 3 & 7, index & 7)
                if 8 * atom + (position ^ token) < self.tiles * 4:
                    for w in cutlass.range_constexpr(4):
                        bad |= has_ff_byte(words[w])
            scale = load_volatile(source + self.h_image_bytes + lane * 4)[0]
            if valid_bytes > 0:
                bad |= has_ff_byte(scale & mask)
            ready = cute.arch.vote_all_sync(not bad)
        image = sm.iterator + self.off_h + buffer * self.h_image_bytes
        for v in cutlass.range_constexpr(chunks):
            words = tuple(values[v, w] for w in range(4))
            store_shared_words(image + (lane + 32 * v) * 16, words)
        word = cute.arch.shuffle_sync(scale, 4 * (lane & 7) + (lane >> 3))
        if lane < 8 * self.h_chunks:
            sf = sm.iterator + self.off_hsf + buffer * self.h_scale_bytes
            store_shared_words(sf + (lane >> 3) * 128 + (lane & 7) * 16, (word,) * 4)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()

    @cute.jit
    def load_hidden(self, sm, workspace, ranges, parity):
        lane = thread()[2]
        full = view(sm.iterator, self.off_misc + self.bar_hfull, Uint64, 2)
        empty = view(sm.iterator, self.off_misc + self.bar_hempty, Uint64, 2)
        ordinal = Int32(-1)
        if cutlass.const_expr(self.steal):
            unit = self.pop_unit(sm, Int32(0))
            while unit >= 0:
                ordinal += 1
                buffer = ordinal & 1
                if ordinal >= 2:
                    self.sync(empty.iterator + buffer, (ordinal >> 1) - 1 & 1)
                    cute.arch.fence_proxy("async.shared", space="cta")  # MMA read it
                image = self.image(workspace, parity, unit)
                self.poll_hidden(sm, image, buffer)
                if lane == 0:
                    cute.arch.mbarrier_arrive(full.iterator + buffer)
                unit = self.pop_unit(sm, ordinal + 1)
        for g in cutlass.range_constexpr(2 * (not self.steal)):
            if ranges[g, 5] > ranges[g, 4]:
                lo, hi = (ranges[g, 4] // self.jobs, (ranges[g, 5] - 1) // self.jobs)
                for u in range(lo, hi + 1):
                    ordinal += 1
                    buffer = ordinal & 1
                    if ordinal >= 2:
                        self.sync(empty.iterator + buffer, (ordinal >> 1) - 1 & 1)
                        cute.arch.fence_proxy("async.shared", space="cta")
                    image = self.image(workspace, parity, u)
                    self.poll_hidden(sm, image, buffer)
                    if lane == 0:
                        cute.arch.mbarrier_arrive(full.iterator + buffer)

    @cute.jit
    def queue(self, sm):
        """The unit queue: (units [8], full [8], empty [8] barriers)."""
        bars = self.off_misc + self.bar_queue
        units = view(sm.iterator, self.off_queue, Int32, 8)
        full = view(sm.iterator, bars, Uint64, 8)
        return units, full, view(sm.iterator, bars + 64, Uint64, 8)

    @cute.jit
    def push_unit(self, sm, pushed, unit):
        """Producer: queue ``unit`` (-1 ends) for the hidden loader."""
        units, full, empty = self.queue(sm)
        slot = pushed % 8
        if pushed >= 8:
            self.sync(empty.iterator + slot, (pushed // 8 - 1) & 1)
        units[slot] = unit
        cute.arch.mbarrier_arrive(full.iterator + slot)
        return pushed + 1

    @cute.jit
    def pop_unit(self, sm, popped):
        """Hidden loader warp: the ``popped``-th queued unit."""
        units, full, empty = self.queue(sm)
        slot = popped % 8
        self.sync(full.iterator + slot, popped // 8 & 1)
        unit = units[slot]
        cute.arch.sync_warp()
        if thread()[2] == 0:
            cute.arch.mbarrier_arrive(empty.iterator + slot)
        return unit

    @cute.jit
    def fc2_job(self, ring, state, maps, experts, ranges, j, group: cutlass.Constexpr):
        """FC2 job ``j`` (absolute) of window ``group``: queue a new unit, then issue
        its stages (TP8 a tile pair, TP16 a tile) exactly as the static walk does."""
        slot, phase, issued, pushed, current = state
        unit, job = (j // self.jobs, j % self.jobs)
        if unit != current:
            pushed = self.push_unit(ring[0], pushed, unit)
            current = unit
        index = 32 + 16 * group + unit - self.first_unit(ranges, 1, group)
        expert = experts[index]
        if cutlass.const_expr(self.paired):
            work = (maps.iterator, expert, unit, 2 * job)
            slot, phase, issued = self.fc2_pair(ring, slot, phase, issued, *work)
        else:
            row = expert * 3584 + job * 128
            offset = Int64(expert) * self.w2_scale_bytes
            offset += job * self.w2_tile_scale_bytes
            tma = (maps.iterator + 128, ring[3], row, Int32(0), offset, Int32(1024))
            meta = (Int32(1), unit, job, Int32(0))
            slot, phase, issued = self.issue(ring, slot, phase, issued, meta, tma, True)
        return slot, phase, issued, pushed, current

    @cute.jit
    def fc2_jobs(self, ring, state, maps, experts, ranges, lo, hi, group):
        for j in cutlass.range(lo, hi, unroll=1):
            state = self.fc2_job(ring, state, maps, experts, ranges, j, group)
        return state

    @cute.jit
    def claims(self, ring, state, maps, experts, ranges, counter, base, grains, back):
        """Grains of one zone (``grains`` from job ``base``) claimed through
        ``counter`` (baseline -1: low half front claims, high half back claims) from
        the front or the ``back``; the next claim is in flight while one issues.
        Returns the state and the grains taken."""
        step = (1 << 16) if back else 1
        pending = claim(counter, step)
        active, taken = (grains > 0, Int32(0))
        while active:
            value = pending + 1
            front, rear = (Int32(value & 65535), Int32(value >> 16))
            active = front + rear < grains
            if active:
                ahead = claim(counter, step)
                index = front
                if back:
                    index = grains - 1 - rear
                lo = base + index * self.grain
                hi = lo + self.grain
                state = self.fc2_jobs(ring, state, maps, experts, ranges, lo, hi, 1)
                pending, taken = (ahead, taken + 1)
        return state, taken

    @cute.jit
    def fc2_steal(self, ring, state, maps, experts, ranges, workspace, parity):
        """FC2 with tail stealing: FC2(A) and the static head of FC2(B), the owner's
        zone from the front, then the zones of CTAs cta + 1 .. cta + victims from the
        back while their units fit window 3; the queue ends with -1."""
        cta = thread()[3]
        n, g = (self.jobs, self.grain)
        head = (ranges[0, 4], ranges[0, 5], 0)
        state = self.fc2_jobs(ring, state, maps, experts, ranges, *head)
        counters = workspace.iterator + self.ws_steal + 640 * parity
        lo, hi = (ranges[1, 4], ranges[1, 5])
        grains = (hi - lo) // g
        if grains > self.zone:
            grains = Int32(self.zone)
        base = hi - grains * g
        state = self.fc2_jobs(ring, state, maps, experts, ranges, lo, base, 1)
        own = (counters + 4 * cta, base, grains, False)
        state, _ = self.claims(ring, state, maps, experts, ranges, *own)
        start, count = (ranges[1, 0], ranges[1, 1] // (self.tiles * 14))
        first = ranges[1, 4] // n
        victim = cta + 1
        while (victim <= cta + self.victims) & (victim < self.grid):
            lo = n * start + victim * n * count // self.grid
            hi = n * start + (victim + 1) * n * count // self.grid
            grains = (hi - lo) // g
            if grains > self.zone:
                grains = Int32(self.zone)
            if (hi - 1) // n - first > 15:
                victim = cta + self.victims + 1  # beyond window 3: stop
            else:
                counter = counters + 4 * victim
                work = (counter, hi - grains * g, grains, True)
                state, taken = self.claims(ring, state, maps, experts, ranges, *work)
                victim = victim + 1
        slot, phase, issued, pushed, current = state
        pushed = self.push_unit(ring[0], pushed, Int32(-1))
        return slot, phase, issued

    @cute.jit
    def fetch(self, sm, desc, scales, full, slot, row, chunk, offset, size, more=None):
        """Weight box and scale bytes (``more``: a second ``size``-byte scale run)."""
        load_tma3(sm.iterator + slot * 32768, desc, full.iterator + slot, row, chunk)
        dst = sm.iterator + self.off_sfa + slot * 1024
        copy_evict(dst, scales.iterator + offset, size, full.iterator + slot)
        if cutlass.const_expr(more is not None):
            copy_evict(dst + 512, scales.iterator + more, size, full.iterator + slot)

    @cute.jit
    def produce(self, sm, maps, w13_scales, w2_scales, ranges, experts):
        """Producer lane; dynamic scale bytes: 1024, or 512 for the TP8 FC2 tail."""
        stages = self.stages
        barriers = view(sm.iterator, self.off_misc, Uint64, 2 * stages)
        metadata = view(sm.iterator, self.off_meta, Int32, 4 * stages)
        fc1 = ranges[0, 3] - ranges[0, 2] + ranges[1, 3] - ranges[1, 2]
        fc2 = ranges[0, 5] - ranges[0, 4] + ranges[1, 5] - ranges[1, 4]
        total = fc1 + self.fc2_stages * fc2
        for n in cutlass.range(total, unroll=1):
            slot, phase = (n % stages, n // stages % 2)
            meta, tma = self.schedule(ranges, experts, n)
            if n >= stages:
                empty = barriers.iterator + stages + slot
                self.sync(empty, phase ^ 1)
            for i in cutlass.range_constexpr(4):
                metadata[4 * slot + i] = meta[i]
            cute.arch.mbarrier_arrive_and_expect_tx(
                barriers.iterator + slot, 16384 + tma[3]
            )
            if meta[0] == 0:
                self.fetch(sm, maps.iterator, w13_scales, barriers, slot, *tma)
            else:
                self.fetch(sm, maps.iterator + 128, w2_scales, barriers, slot, *tma)
        slot, phase = (total % stages, total // stages % 2)
        if total >= stages:
            self.sync(barriers.iterator + stages + slot, phase ^ 1)
        for i in cutlass.range_constexpr(4):
            metadata[4 * slot + i] = Int32(2 if i == 0 else 0)
        cute.arch.mbarrier_arrive(barriers.iterator + slot)

    @cute.jit
    def stream(self, sm, maps, w13_scales, w2_scales, ranges, experts, steal=None):
        """Producer lane walking its FC1 then FC2 stages incrementally (no per-stage
        divisions; the expert is re-read only when the unit changes). Grouped expert
        tables are this CTA's windows; else they hold every distinct expert."""
        tiles, w13, w2 = (self.tiles, maps.iterator, maps.iterator + 128)
        barriers = view(sm.iterator, self.off_misc, Uint64, 2 * self.stages)
        metadata = view(sm.iterator, self.off_meta, Int32, 4 * self.stages)
        ring = (sm, barriers, metadata, w2_scales)
        slot, phase, issued = (Int32(0), Int32(0), Int32(0))
        for group in cutlass.range_constexpr(2):
            lo, hi = (ranges[group, 2], ranges[group, 3])
            ks, tile = (lo % 14, tiles * ranges[group, 0] + lo // 14)
            r, index = (tile % tiles, self.first_unit(ranges, 0, group))
            if cutlass.const_expr(self.grouped):
                index = Int32(16 * group)
            expert = experts[index]
            for s in cutlass.range(lo, hi, unroll=1):
                opens, closes = ((s == lo) | (ks == 0), (s == hi - 1) | (ks == 13))
                c = Int32(opens) | Int32(closes) << 1
                row = expert * self.w13_rows + r * 128
                offset = Int64(expert) * self.w13_scale_bytes + r * 14336 + ks * 1024
                tma = (w13, w13_scales, row, 2 * ks, offset, Int32(1024))
                slot, phase, issued = self.issue(
                    ring, slot, phase, issued, (Int32(0), tile, ks, c), tma, False
                )
                ks += 1
                if ks == 14:
                    ks, tile, r = (Int32(0), tile + 1, r + 1)
                    if r == tiles:
                        r, index = (Int32(0), index + 1)
                        expert = experts[index]
        if cutlass.const_expr(self.steal):
            state = (slot, phase, issued, Int32(0), Int32(-1))
            work = (ring, state, maps, experts, ranges, *steal)
            slot, phase, issued = self.fc2_steal(*work)
        for group in cutlass.range_constexpr(2 * (not self.steal)):
            lo, hi = (ranges[group, 4], ranges[group, 5])
            unit, tile, index = (lo // self.jobs, lo % self.jobs, lo // self.jobs)
            if cutlass.const_expr(self.grouped):
                index = Int32(32 + 16 * group)
            expert = experts[index]
            for _ in cutlass.range(lo, hi, unroll=1):
                if cutlass.const_expr(self.paired):
                    job = (maps.iterator, expert, unit, 2 * tile)
                    slot, phase, issued = self.fc2_pair(ring, slot, phase, issued, *job)
                else:
                    for half in cutlass.range_constexpr(self.fc2_stages):
                        size = 1024 if half == 0 else self.w2_tile_scale_bytes - 1024
                        row = expert * 3584 + tile * 128
                        offset = Int64(expert) * self.w2_scale_bytes + half * 1024
                        offset += tile * self.w2_tile_scale_bytes
                        tma = (w2, w2_scales, row, 2 * half, offset, Int32(size))
                        meta = (Int32(1 if half == 0 else 3), unit, tile, Int32(0))
                        slot, phase, issued = self.issue(
                            ring, slot, phase, issued, meta, tma, True
                        )
                tile += 1
                if tile == self.jobs:
                    unit, tile, index = (unit + 1, Int32(0), index + 1)
                    expert = experts[index]
        if issued >= self.stages:
            self.sync(barriers.iterator + self.stages + slot, phase ^ 1)
        for i in cutlass.range_constexpr(4):
            metadata[4 * slot + i] = Int32(2 if i == 0 else 0)
        arrivals = 33 if self.grouped else 1
        cute.arch.mbarrier_arrive(barriers.iterator + slot, arrive_count=arrivals)

    @cute.jit
    def fc2_pair(self, ring, slot, phase, issued, maps, expert, unit, tile):
        """The three 16 KB stages of FC2 tiles ``tile`` and ``tile + 1``: chunks 0-1 of
        each (1 KB scales), then chunk 2 of both (map 5, two 512 B scale runs)."""
        w2, scales, step = (maps + 128, ring[3], self.w2_tile_scale_bytes)
        row = expert * 3584 + tile * 128
        offset = Int64(expert) * self.w2_scale_bytes + tile * step
        for t in cutlass.range_constexpr(2):
            tma = (w2, scales, row + 128 * t, Int32(0), offset + t * step, Int32(1024))
            meta = (Int32(1), unit, tile + t, Int32(0))
            slot, phase, issued = self.issue(ring, slot, phase, issued, meta, tma, True)
        tma = (maps + 640, scales, row, Int32(2), offset + 1024, Int32(512))
        meta = (Int32(3), unit, tile, Int32(0))
        more = offset + step + 1024
        return self.issue(ring, slot, phase, issued, meta, tma, True, more)

    @cute.jit
    def issue(
        self, ring, slot, phase, issued, meta, tma, fc2: cutlass.Constexpr, more=None
    ):
        """One grouped stage: wait for its slot, publish the metadata, expect and issue
        the weight box and scales (FC2: standing in for the gather warp's 32
        arrivals); returns the advanced (slot, phase, issued)."""
        sm, barriers, metadata = ring[:3]
        desc, scales, row, chunk, offset, size = tma
        if issued >= self.stages:
            empty = barriers.iterator + self.stages + slot
            self.sync(empty, phase ^ 1)
        for i in cutlass.range_constexpr(4):
            metadata[4 * slot + i] = meta[i]
        full = barriers.iterator + slot
        total = 16384 + size if more is None else 16384 + 2 * size
        cute.arch.mbarrier_arrive_and_expect_tx(full, total)
        self.fetch(sm, desc, scales, barriers, slot, row, chunk, offset, size, more)
        if cutlass.const_expr(fc2 and self.grouped):
            cute.arch.mbarrier_arrive(full, arrive_count=32)
        slot += 1
        if slot == self.stages:
            slot, phase = (Int32(0), phase ^ 1)
        return slot, phase, issued + 1

    @cute.jit
    def ownership(self, index: cutlass.Constexpr[int], batch):
        xtid = thread()[0] - 64
        if cutlass.const_expr(self.dense):
            block = xtid + 320 * index
            q = block // 8
            stage = q // batch
            token = q - stage * batch
            kblock = block & 7
            valid = block < 112 * batch
        else:
            stage = 5 * index + xtid // 64
            token = (xtid & 63) // 8
            kblock = xtid & 7
            valid = (stage < 14) & (token < batch)
        return (stage, token, kblock, valid)

    @cute.jit
    def load_x(self, x, raw, batch):
        """Warps 2..11 preload their 32-BF16 x blocks before the dependent launch."""
        raw.fill(0)
        for index in cutlass.range_constexpr(self.rounds):
            stage, token, kblock, valid = self.ownership(index, batch)
            if valid:
                source = x.iterator + token * 3584 + stage * 256 + kblock * 32
                for fragment in cutlass.range_constexpr(4):
                    words = load_words(
                        source + fragment * 8, count=4, cache_policy="", mutable=True
                    )
                    for j in cutlass.range_constexpr(4):
                        raw[index, fragment, j] = words[j]

    @cute.jit
    def quantize_x(self, sm, raw, barriers, batch):
        """Each thread quantizes its block (two uint4 fragments, one scale byte) and
        arrives once at xready[stage]: eight arrivals per active token, no DSMEM."""
        for index in cutlass.range_constexpr(self.rounds):
            stage, token, kblock, valid = self.ownership(index, batch)
            if valid:
                packed, scale = mx_block([raw[index, i // 4, i % 4] for i in range(16)])
                for half, words in enumerate((packed[:4], packed[4:])):
                    offset = swizzled(16 * stage + 2 * kblock + half, token, self.off_x)
                    store_shared_words(sm.iterator + offset, words)
                kb = 8 * stage + kblock
                sm[self.off_xsf + kb // 4 * 128 + token * 16 + kb % 4] = Uint8(scale)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.mbarrier_arrive(barriers + stage)

    @cute.jit
    def acquire(self, empty, acc: Int32, uses: Int32):
        if uses >= 16:
            self.sync(empty.iterator + acc, uses // 16 - 1 & 1, True)

    @cute.jit
    def signal(self, metadata, full, acc, values, arrivals: cutlass.Constexpr = 1):
        if thread()[2] == 0:
            for i in cutlass.range_constexpr(4):
                metadata[4 * acc + i] = values[i]
            for _ in cutlass.range_constexpr(arrivals):
                cute.arch.mbarrier_arrive(full.iterator + acc)

    @cute.jit
    def consume(self, sm, tmem):
        lane = thread()[2]
        stages = self.stages
        bars: tuple = (self.bar_full, self.bar_empty, self.bar_accfull)
        bars += (self.bar_accempty, self.bar_hfull, self.bar_hempty, self.bar_xready)
        full, empty, accfull, accempty, hfull, hempty, xready = (
            view(sm.iterator, self.off_misc + o, Uint64, n)
            for o, n in zip(bars, (stages, stages, 16, 16, 2, 2, 14))
        )
        meta = view(sm.iterator, self.off_meta, Int32, 4 * stages)
        accmeta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        offsets = (0, self.off_sfa, self.off_x, self.off_xsf, self.off_h, self.off_hsf)
        a0, asf0, x0, xsf0, h0, hsf0 = (
            smem_desc(sm.iterator + o, i % 2 == 0) for i, o in enumerate(offsets)
        )
        xmask, ended = (Uint32(0), False)
        slot, uses, acc, count = (Int32(0), Int32(0), Int32(0), Int32(0))
        current, ordinal, hbuffer = (Int32(-1), Int32(-1), Int32(0))
        while not ended:
            slot = count % stages
            ready = (full.iterator + slot, count // stages & 1, True)
            self.sync(*ready)
            kind, a, b, c = (meta[4 * slot + i] for i in range(4))
            if kind == 2:
                if (lane == 0) & (current >= 0) & (hbuffer < 2):
                    commit(hempty.iterator + hbuffer)
                for _ in cutlass.range(2, unroll=1):
                    self.acquire(accempty, acc, uses)
                    self.signal(accmeta, accfull, acc, (kind, a, b, c), 2)
                    uses += 1
                    acc = uses % 16
                ended = True
            else:
                publish = False
                if kind == 0:
                    first, last = (c & 1 != 0, c & 2 != 0)
                    if first:
                        self.acquire(accempty, acc, uses)
                    copy_scales = Uint32(0)
                    if cutlass.const_expr(self.grouped):
                        # The slot's own image (its cp.async completed full[slot]).
                        cute.arch.fence_proxy("async.shared", space="cta")
                        copy_scales = Uint32(1)
                    elif xmask >> b & 1 == 0:
                        self.sync(xready.iterator + b, 0, True)
                        xmask |= Uint32(1) << b
                        copy_scales = Uint32(1)
                    completion = Uint32(0)
                    if last:
                        completion = shared_address(accfull.iterator + acc)
                    xi = slot if self.grouped else b
                    stage(
                        (tmem + acc * 8, a0 + slot * 2048, x0 + xi * 128),
                        (tmem + 128 + slot * 8, asf0 + slot * 64),
                        (tmem + 256 + xi * 8, xsf0 + xi * 16, copy_scales),
                        (Uint32(not first), shared_address(empty.iterator + slot)),
                        completion,
                        kind="fc1",
                    )
                    cute.arch.sync_warp()
                    publish = last
                else:
                    copy_scales, release = (Uint32(0), Uint32(0))
                    if a != current:
                        if (current >= 0) & (hbuffer < 2):
                            release = shared_address(hempty.iterator + hbuffer)
                        current = a
                        ordinal += 1
                        hbuffer = ordinal & 1
                        ready = (hfull.iterator + hbuffer, ordinal >> 1 & 1, True)
                        self.sync(*ready)
                        copy_scales = Uint32(1)
                    hd = h0 + hbuffer * (self.h_image_bytes // 16)
                    hs = hsf0 + hbuffer * (self.h_scale_bytes // 16)
                    sfbc = tmem + 128 + 8 * stages + hbuffer * self.w2_scale_columns
                    tp16 = cutlass.const_expr(self.tp == 16)
                    if cutlass.const_expr(self.paired):
                        weights = (tmem + 128 + slot * 8, asf0 + slot * 64)
                        released = shared_address(empty.iterator + slot)
                        if kind == 1:  # chunks 0-1 of the pair's tile b
                            target, owner = (acc, uses)
                            if b % 2 == 1:
                                target, owner = ((uses + 1) % 16, uses + 1)
                            self.acquire(accempty, target, owner)
                            stage(
                                (tmem + target * 8, a0 + slot * 2048, hd),
                                weights,
                                (sfbc, hs, copy_scales),
                                (release, released),
                                0,
                                kind="fc2a",
                            )
                        else:  # chunk 2 of tiles b (rows 0-127) and b + 1 (128-255)
                            for t in cutlass.range_constexpr(2):
                                rows = a0 + slot * 2048 + 1024 * t
                                stage(
                                    (tmem + acc * 8, rows, hd + 128),
                                    (weights[0] + 4 * t, weights[1] + 32 * t),
                                    (sfbc + 8, 0, 0),
                                    (0, released),
                                    shared_address(accfull.iterator + acc),
                                    kind="fc2b" if t else "fc2p",
                                )
                                if cutlass.const_expr(t == 0):
                                    cute.arch.sync_warp()
                                    first = (Int32(1), a, b, c)
                                    self.signal(accmeta, accfull, acc, first)
                                    uses += 1
                                    acc = uses % 16
                                    cute.arch.sync_warp()
                            publish, kind, b = (True, Int32(1), b + 1)
                    else:
                        if tp16 or kind == 1:
                            self.acquire(accempty, acc, uses)
                            stage(
                                (tmem + acc * 8, a0 + slot * 2048, hd),
                                (tmem + 128 + slot * 8, asf0 + slot * 64),
                                (sfbc, hs, copy_scales),
                                (release, shared_address(empty.iterator + slot)),
                                shared_address(accfull.iterator + acc) if tp16 else 0,
                                kind="fc2" if tp16 else "fc2a",
                            )
                            publish = tp16
                        else:
                            stage(
                                (tmem + acc * 8, a0 + slot * 2048, hd + 128),
                                (tmem + 128 + slot * 8, asf0 + slot * 64),
                                (sfbc + 8, 0, 0),
                                (0, shared_address(empty.iterator + slot)),
                                shared_address(accfull.iterator + acc),
                                kind="fc2b",
                            )
                            publish = True
                            kind = Int32(1)
                    cute.arch.sync_warp()
                if publish:
                    self.signal(accmeta, accfull, acc, (kind, a, b, c))
                    uses += 1
                    acc = uses % 16
                count += 1

    @cute.jit
    def partials(self, workspace, values, tile, ranges, batch):
        """Publish non-owner fragments, or reduce the owner's remaining pieces."""
        ws, live = (workspace.iterator, 8 if self.grouped else batch)
        _, warp, lane, cta = thread()
        row = 32 * (warp & 3) + lane
        group = Int32(tile >= self.tiles * ranges[1, 0])
        start = (tile - self.tiles * ranges[group, 0]) * 14
        owner = start >= ranges[group, 2]
        if not owner:
            offset = self.ws_partial + ((Int64(group) * 160 + cta) * 1024 + row) * 4
            for t in cutlass.range_constexpr(self.width):
                if t < live:
                    store_words(ws + offset + t * 512, (float_bits(values[t]),))
        else:
            peer, size = (cta + 1, ranges[group, 1])
            while (peer < self.grid) & (peer * size // self.grid < start + 14):
                if (peer + 1) * size // self.grid != peer * size // self.grid:
                    source = ws + self.ws_partial
                    source += ((Int64(group) * 160 + peer) * 1024 + row) * 4
                    words = cute.make_rmem_tensor(self.width, Uint32)
                    ready = False
                    while not ready:
                        bad = False
                        for t in cutlass.range_constexpr(self.width):
                            words[t] = Uint32(0)
                            if t < live:
                                words[t] = load_volatile(source + t * 512)[0]
                                bad |= words[t] == Uint32(4294967295)
                        ready = cute.arch.vote_all_sync(not bad)
                    for t in cutlass.range_constexpr(self.width):
                        if t < live:
                            values[t] += bits_float(words[t])
                            store_words(source + t * 512, (Uint32(4294967295),))
                peer += 1
        return owner

    @cute.jit
    def store_hidden(self, sm, workspace, values, tile, parity, scalars):
        _, warp, lane, _ = thread()
        ew, eg = (warp & 3, (warp - 4) // 4)
        expert, chunk = (tile // self.tiles, tile % self.tiles)
        up = lane & 8 == 0
        h, maxima = (cute.make_rmem_tensor(self.width, Float32) for _ in range(2))
        shared_max = view(sm.iterator, self.off_misc + 884, Float32, 64)
        for t in cutlass.range_constexpr(self.width):
            gate = cute.arch.shuffle_sync_bfly(values[t], 8)
            h[t] = situ(gate, values[t], *scalars) if up else Float32(0)
            maxima[t] = abs(h[t])
        for t in cutlass.range_constexpr(self.width):
            for off in cutlass.range_constexpr(5):
                other = cute.arch.shuffle_sync_bfly(maxima[t], 1 << off)
                maxima[t] = cute.arch.fmax(maxima[t], other)
            if lane == 0:
                shared_max[(eg * 4 + ew) * 8 + t] = maxima[t]
        group_barrier(eg)
        stage_base, scale_token = (self.off_misc + 3200 + (eg * 4 + ew) * 128, Int32(0))
        for t in cutlass.range_constexpr(self.width):
            peer = shared_max[(eg * 4 + (ew ^ 1)) * 8 + t]
            scale = mx_exponent(cute.arch.fmax(maxima[t], peer), floor=self.tp == 8)
            if lane == t:
                scale_token = scale
            if up:
                bits = fp8_bits(h[t] * mx_rescale(scale))
                sm[stage_base + t * 16 + 2 * (lane % 8) + lane // 16] = bits
        cute.arch.sync_warp()
        if lane < 8:
            kc, data = (4 * chunk + ew, (Uint32(0), Uint32(0), Uint32(0), Uint32(0)))
            if lane < self.width:
                data = smem_words(sm.iterator + stage_base + lane * 16)
            image = self.image(workspace, parity, expert)
            store_words(image + swizzled(kc, lane), data)
            if ew & 1 == 0:
                byte = view(image, 0, Uint8, self.h_compact_bytes)
                value = Uint8(scale_token)
                byte[self.h_image_bytes + lane * 16 + 2 * chunk + ew // 2] = value
        group_barrier(eg)

    def pair(self, t, slot):
        """Expanded row of unit row ``t``: its pair (grouped), or token t's ``slot``."""
        return Int64(slot) if self.grouped else Int64(t) * 16 + slot

    @cute.jit
    def fc2(self, sm, workspace, output, values, unit, tile, ranges):
        """Rows of ``unit``: slots of tokens 0..7, or grouped pair indices."""
        _, warp, lane, _ = thread()
        if cutlass.const_expr(self.grouped):
            slots = self.tables(sm)[1]
            unit = self.local(ranges, unit, 1, Int32(unit >= ranges[1, 0]))
        else:
            slots = view(sm.iterator, self.off_misc + 2176, Int8, 1024)
        for t in cutlass.range_constexpr(self.width):
            slot = Int32(slots[unit * 8 + t])
            if slot >= 0:
                other = cute.arch.shuffle_sync_down(values[t], 8)
                if cutlass.const_expr(self.publish):
                    # Stage row [token, slot]: lane q < 4 stores this warp's 8q..8q+7.
                    word, q = pack_pair(values[t], other, policy="stage"), lane & 3
                    words = [shuffle(word, s + 2 * q) for s in (0, 16, 1, 17)]
                    row = self.pair(t, slot) * 1792 + tile * 64 + (warp & 3) * 16
                    stage = workspace.iterator + self.ws_stage + (row + 4 * q) * 4
                    if lane < 4:
                        store_words(stage, words)
                elif lane & 8 == 0:
                    column = tile * 128 + (warp & 3) * 32 + 4 * (lane % 8) + lane // 8
                    dst = output.iterator + self.pair(t, slot) * 3584 + column
                    store_words(dst, (bf16x2(values[t], other),))

    @cute.jit
    def finalize(self, workspace, weights, batch, box):
        """Each epilogue warp polls (and re-arms) the 16 staged rows of its 64-column
        lines and multicasts their weighted sum into latent slot (parity, rank) of the
        [3, TP, 64, 3584] mailbox (two-shot: pushes every 8 columns to their owner);
        lines stride over every CTA's epilogue warps."""
        _, warp, lane, cta = thread()
        for line in range(cta * 8 + warp - 4, batch * 56, self.grid * 8):
            token, column = line // 56, line % 56 * 64
            source = workspace.iterator + self.ws_stage
            source += (Int64(token) * 16 * 1792 + column // 2 + lane) * 4
            words = cute.make_rmem_tensor(16, Uint32)
            ready = False
            while not ready:
                bad = False
                for slot in cutlass.range_constexpr(16):
                    words[slot] = load_volatile(source + slot * 7168)[0]
                    bad |= words[slot] == Uint32(4294967295)
                ready = cute.arch.vote_all_sync(not bad)
            low, high = Float32(0), Float32(0)
            for slot in cutlass.range_constexpr(16):
                store_words(source + slot * 7168, (Uint32(4294967295),))
                weight = Float32(weights[token * 16 + slot])
                low = fma(bf16(words[slot]), weight, low)
                high = fma(bf16(words[slot], True), weight, high)
            word = pack_pair(low, high, policy="latent")
            packed = [shuffle(word, 4 * (lane & 7) + i) for i in range(4)]
            if lane < 8:
                if cutlass.const_expr(self.publish == "twoshot"):
                    push(*box, token, column + 8 * lane, packed)
                else:
                    mc, parity, rank = box
                    offset = ((parity * self.tp + rank) * 64 + token) * 3584 + column
                    offset += 8 * lane
                    multicast_words(mc + Int64(offset) * 2, packed)

    @cute.jit
    def epilogue(self, sm, workspace, output, tmem, ranges, batch, parity, scalars):
        _, warp, lane, _ = thread()
        ew, eg = (warp & 3, (warp - 4) // 4)
        full = view(sm.iterator, self.off_misc + self.bar_accfull, Uint64, 16)
        empty = view(sm.iterator, self.off_misc + self.bar_accempty, Uint64, 16)
        meta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        acc, phase, ended = (eg, Int32(0), False)
        while not ended:
            self.sync(full.iterator + acc, phase, True)
            kind, a, b = (meta[4 * acc], meta[4 * acc + 1], meta[4 * acc + 2])
            if kind == 2:
                ended = True
            else:
                raw = load_tmem(tmem + (Uint32(32 * ew) << 16) + acc * 8, 8)
                values = cute.make_rmem_tensor(self.width, Float32)
                for t in cutlass.range_constexpr(self.width):
                    values[t] = raw[t]
                fence_tmem(before=True)
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(empty.iterator + acc)
                acc += 2
                if acc >= 16:
                    acc -= 16
                    phase ^= 1
                if kind == 0:
                    if self.partials(workspace, values, a, ranges, batch):
                        self.store_hidden(sm, workspace, values, a, parity, scalars[:3])
                else:
                    self.fc2(sm, workspace, output, values, a, b, ranges)
        if cutlass.const_expr(self.publish):
            self.finalize(workspace, output, batch, scalars[3:])

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 384, cluster=1)

    @cute.jit
    def initialize(self, sm, maps, batch):
        tid, _, _ = cute.arch.thread_idx()
        barriers = view(sm.iterator, self.off_misc, Uint64, 65)
        tmem_base = view(sm.iterator, self.off_misc + 856, Uint32, 1)
        if tid == 0:
            receive = range(528, 533) if self.capacity == 4 else ()
            for i in (*range(65), *receive):
                acc = 2 * self.stages
                count = 1 if not acc <= i < acc + 32 else 2 if i < acc + 16 else 4
                count = batch * 8 if acc + 36 <= i < acc + 50 else count
                if self.grouped and i < self.stages:
                    count = 33
                cute.arch.mbarrier_init(barriers.iterator + i, count)
            if cutlass.const_expr(self.steal):
                queue = view(sm.iterator, self.off_misc + self.bar_queue, Uint64, 16)
                for i in cutlass.range_constexpr(16):
                    cute.arch.mbarrier_init(queue.iterator + i, 1)
            cute.arch.mbarrier_init_fence()
            acquire_tensormap(maps.iterator)
            acquire_tensormap(maps.iterator + 128)
            if cutlass.const_expr(self.paired):
                acquire_tensormap(maps.iterator + 640)
        if tid // 32 == 3:
            cute.arch.alloc_tmem(512, tmem_base.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        if cutlass.const_expr(not self.grouped):
            self.clear_x(sm, tid, batch)
        cute.arch.fence_proxy("async.shared", space="cta")
        return tmem_base

    @cute.jit
    def clear_x(self, sm, tid, batch):
        """Zero the padded token rows and the scale image (capacity <= 8)."""
        zero = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
        for i in range(tid, (8 - batch) * 224, 384):
            token, chunk = (batch + i // 224, i % 224)
            store_shared_words(sm.iterator + swizzled(chunk, token, self.off_x), zero)
        xsf = sm.iterator + self.off_xsf
        for i in range(tid, 224, 384):
            store_shared_words(xsf + (i >> 3) * 128 + (i & 7) * 16, zero)

    @cute.jit
    def rearm(self, workspace, call_count):
        """Re-arm the other parity's images: all 128, or the last grouped call's."""
        _, _, lane, cta = thread()
        items: cutlass.Constexpr = 128 * self.h_compact_bytes // 16
        lo, hi = (cta * items // self.grid, (cta + 1) * items // self.grid)
        if cutlass.const_expr(self.grouped):
            last = view(workspace.iterator, self.ws_calls + 640, Int32, 2)
            used = max(last[call_count & 1 ^ 1], 0) * (self.h_compact_bytes // 16)
            lo, hi = (cta * used // self.grid, (cta + 1) * used // self.grid)
        other = self.image(workspace, call_count & 1 ^ 1, 0)
        for i in range(lo + lane, hi, 32):
            store_words(other + i * 16, (Uint32(4294967295),) * 4)
        if cutlass.const_expr(self.steal):
            # Every one of the 160 counters: grids change with the batch.
            base = self.ws_steal + 640 * (call_count & 1 ^ 1)
            counters = view(workspace.iterator, base, Int32, 160)
            if lane == 0:
                counters[cta] = Int32(-1)
                if cta < 160 - self.grid:
                    counters[self.grid + cta] = Int32(-1)

    @cute.kernel
    def kernel(self, args):
        x, ids, maps, w13_scales, w2_scales, workspace, output, batch, *scalars = args
        tid, warp, lane, cta = thread()
        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem_bytes, 1024)
        tmem_base = self.initialize(sm, maps, batch)
        cute.arch.griddepcontrol_wait()
        counters = view(workspace.iterator, self.ws_counter, Int32, 160)
        if cutlass.const_expr(self.grouped):
            # Images and rearm alternate over grouped calls; ``x`` is the plan.
            routed = view(sm.iterator, self.off_misc + 1140, Int32, 3)
            self.group(sm, ids, workspace, batch)
            self.windows(sm, self.partition(routed[0], cta))
            # Retire the routing scratch's generic accesses before weight-ring reuse.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            distinct, call_count, generation = (routed[0], routed[1], routed[2])
            experts = self.tables(sm)[0]
        else:
            distinct, call_count, experts = self.route(sm, ids, counters, batch)
            generation = call_count
            layout = cute.make_layout((self.rounds, 4, 4), stride=(16, 4, 1))
            raw_x = cute.make_rmem_tensor(layout, Uint32)
            if warp >= 2:
                self.load_x(x, raw_x, batch)
        fence_tmem()
        tmem = tmem_base[0]
        cute.arch.griddepcontrol_launch_dependents()
        ranges = self.partition(distinct, cta)
        if warp == 0:
            if lane == 0:
                weights = (sm, maps, w13_scales, w2_scales, ranges, experts)
                if cutlass.const_expr(self.steal):
                    self.stream(*weights, (workspace, generation & 1))
                elif cutlass.const_expr(self.streamed):
                    self.stream(*weights)
                else:
                    self.produce(*weights)
        else:
            if cutlass.const_expr(not self.grouped):
                ready = view(sm.iterator, self.off_misc + self.bar_xready, Uint64, 14)
                if warp >= 2:
                    self.quantize_x(sm, raw_x, ready.iterator, batch)
            if warp == 1:
                self.consume(sm, tmem)
            elif warp == 2:
                self.load_hidden(sm, workspace, ranges, generation & 1)
            elif warp == 3:
                if cutlass.const_expr(self.grouped):
                    self.gather_x(sm, x, ranges, batch)
                self.rearm(workspace, generation)
            else:
                self.epilogue(
                    sm, workspace, output, tmem, ranges, batch, generation & 1, scalars
                )
        release_tmem(tmem, 512, warp, 3)
        if tid == 0:
            counters[cta] = call_count + 1
            if cta < 160 - self.grid:
                counters[self.grid + cta] = call_count + 1
            if cutlass.const_expr(self.grouped):
                grouped = view(workspace.iterator, self.ws_calls, Int32, 162)
                grouped[cta] = generation + 1
                if cta < 160 - self.grid:
                    grouped[self.grid + cta] = generation + 1
                if cta == 0:
                    grouped[160 + (generation & 1)] = distinct
