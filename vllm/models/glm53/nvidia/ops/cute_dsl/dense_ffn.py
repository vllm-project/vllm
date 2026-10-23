# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense FFN of GLM-5.3's first three layers (TP8, four decode rows): ``Final = x +
allreduce(down(silu(gate(n)) * up(n)))`` with ``n = rmsnorm(x) * gamma``.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, Uint64

from . import _rt as rt

TOKENS = 4
HIDDEN = rt.Core.HIDDEN
INTERMEDIATE = 12288 // rt.Core.RANKS
PEERS = rt.Core.RANKS - 1
BLOCK, BLOCK_SHIFT = 128, 7
SWIZZLE_ROW = 128
CHUNK = 16
TMA_HINT = 0x12F0000000000000
NEG_LOG2E = -1.4426950408889634
GRID, GRID_SHIFT = 128, 7
COMPUTE_WARPS = 8
COMPUTE_THREADS = 32 * COMPUTE_WARPS
LOADER_THREADS = 128
THREADS = COMPUTE_THREADS + LOADER_THREADS
PRODUCER_WARP, TICKET_WARP = 8, 9
BAR_COMPUTE = (8, COMPUTE_THREADS)
BAR_LOADER = (9, LOADER_THREADS)
BAR_ALL = (11, THREADS)
SMEM_BYTES = 214016


class UpWeights:
    """DenseUp: gate, up E4M3 [1536][6144], FP32 scales [2][12][48]; 16 rows per CTA."""

    CTAS = 96
    ROWS, ROW_SHIFT = INTERMEDIATE // CTAS, 4
    PAGES = 6
    PAGE_K = HIDDEN // PAGES
    SLICE_K = PAGE_K // COMPUTE_WARPS
    PAGE_SLICES = PAGE_K // SLICE_K
    CTA_BLOCK_SHIFT = 3
    SCALE_COLS = HIDDEN // BLOCK
    SCALE_COUNT = (INTERMEDIATE // BLOCK) * SCALE_COLS
    SCALES = 2 * INTERMEDIATE * HIDDEN
    SLICE_BYTES = ROWS * SWIZZLE_ROW
    BOX_BYTES = PAGE_SLICES * SLICE_BYTES
    SCALE_BYTES = PAGE_SLICES * 4
    SLOT_SCALES = 2 * BOX_BYTES
    PAGE_BYTES = SLOT_SCALES + 2 * SCALE_BYTES

    @staticmethod
    def chunk(base, warp, group, chunk):
        """Ring byte (word-aligned) of a swizzled gate-box chunk."""
        addr = (
            base
            + warp * UpWeights.SLICE_BYTES
            + group * SWIZZLE_ROW
            + (chunk ^ group) * CHUNK
        )
        return (addr >> 2) * 4

    @staticmethod
    def chunk_pair(q):
        """The two 16-byte chunks of a 128-K slice that MMA lane column ``q`` reads."""
        first = 2 * q + ((q ^ (q >> 1)) & 1)
        return first, first ^ 1

    @staticmethod
    def scale_offset(block, proj):
        """Byte offset in DenseUp of scale ``block`` of ``proj``."""
        return (block + UpWeights.SCALE_COUNT * proj) * 4 + UpWeights.SCALES


class DownWeights:
    """DenseDown: FP32 scales [48][12], E4M3 [6144][1536]; 48 rows per CTA."""

    ROWS = HIDDEN // GRID
    PAGES = 3
    PAGE_K = INTERMEDIATE // PAGES
    BOXES = PAGE_K // BLOCK
    BOX_BYTES = ROWS * SWIZZLE_ROW
    PAGE_BYTES = BOXES * BOX_BYTES
    SCALES = 0
    SCALE_ROW_BYTES = (INTERMEDIATE // BLOCK) * 4
    WEIGHTS = SCALES + (HIDDEN // BLOCK) * SCALE_ROW_BYTES
    SCALE_BYTES = 2 * SCALE_ROW_BYTES
    TILES = ROWS // 16
    ROW_GROUPS = ROWS // 8

    @staticmethod
    def lane_word(warp, lane):
        """u64 word, inside a box pair, of the lane's swizzled weight word."""
        group = lane >> 2
        chunk = ((warp & 3) + 4 * ((lane >> 1) & 1)) ^ group
        return (
            ((warp >> 2) * DownWeights.BOX_BYTES + group * SWIZZLE_ROW + chunk * CHUNK)
            >> 3
        ) + (lane & 1)

    @staticmethod
    def word(base, half, lane_word, row_group):
        """u64 ring word of the lane's weight word in box pair ``half``, row group
        ``row_group``.
        """
        return (
            ((base + 2 * DownWeights.BOX_BYTES * half) >> 3)
            + lane_word
            + (8 * SWIZZLE_ROW // 8) * row_group
        )

    @staticmethod
    def scale_word(bid, tile, page, half, box):
        """Smem word of the scale for MMA row tile ``tile`` and K block ``page * 4 +
        half * 2 + box``.
        """
        first_row = bid * DownWeights.ROWS
        row_block = ((first_row + 16 * tile) >> BLOCK_SHIFT) - (
            first_row >> BLOCK_SHIFT
        )
        return (
            row_block * (INTERMEDIATE // BLOCK)
            + DownWeights.BOXES * page
            + 2 * half
            + box
        )

    @staticmethod
    def scale_offset(bid):
        """Byte offset in DenseDown of the scale row of CTA ``bid``'s first row."""
        return (
            DownWeights.SCALES
            + ((bid * DownWeights.ROWS) >> BLOCK_SHIFT) * DownWeights.SCALE_ROW_BYTES
        )


class Activations:
    """Activation bf16 [6 groups][TOKENS][256 permuted columns]; smem rows padded to 528
    B.
    """

    GROUP_COLS = 256
    GROUP_CTAS, GROUP_SHIFT = GROUP_COLS // UpWeights.ROWS, 4
    ROWS = (INTERMEDIATE // GROUP_COLS) * TOKENS
    ROW_CHUNKS, ROW_CHUNK_SHIFT = GROUP_COLS * 2 // CHUNK, 5
    ROW_BYTES = GROUP_COLS * 2 + CHUNK
    ROW_WORDS = ROW_BYTES // 4
    GROUP_WORDS = TOKENS * ROW_WORDS
    CHUNKS_PER_LANE = ROWS * ROW_CHUNKS // LOADER_THREADS

    @staticmethod
    def col(c):
        """Stored column of ``c``: bits 2<->3, 5<->6 swapped so one 8-byte load matches
        a weight word.
        """
        return (
            (c & 0b10010011)
            | ((c & 64) >> 1)
            | ((c & 32) << 1)
            | ((c & 8) >> 1)
            | ((c & 4) << 1)
        )

    @staticmethod
    def index(group, token, col):
        """Element index in Activation."""
        return (group * TOKENS + token) * Activations.GROUP_COLS + col

    @staticmethod
    def smem_addr(act, chunk):
        """Smem address of 16-byte chunk ``chunk`` (padded rows)."""
        return (
            act
            + (chunk >> Activations.ROW_CHUNK_SHIFT) * Activations.ROW_BYTES
            + (chunk & (Activations.ROW_CHUNKS - 1)) * CHUNK
        )

    @staticmethod
    def b_word(warp, lane):
        """Smem u32 word (inside a group) of a down lane's B fragment."""
        return ((lane >> 2) & 3) * Activations.ROW_WORDS + (
            (warp >> 2) * 64
            + ((warp >> 1) & 1) * 32
            + ((lane >> 1) & 1) * 16
            + (warp & 1) * 8
            + (lane & 1) * 2
        )


@dataclass(frozen=True)
class DenseFfnParams(rt.Params):
    """Kernel parameters of ``dense_ffn_q4_vllm``."""

    X: cute.Pointer = rt.ptr("bfloat16")
    Gamma: cute.Pointer = rt.ptr("bfloat16")
    DenseUp: cute.Pointer = rt.ptr("float8_e4m3")
    DenseDown: cute.Pointer = rt.ptr("float8_e4m3")
    Activation: cute.Pointer = rt.ptr("bfloat16")
    LocalO: cute.Pointer = rt.ptr("bfloat16")
    Final: cute.Pointer = rt.ptr("bfloat16")
    FullEpoch: cute.Pointer = rt.ptr("uint32")
    Flags: cute.Pointer = rt.ptr("uint32")
    peer_rank: Int32
    peer_base0: Int64
    peer_base1: Int64
    peer_base2: Int64
    peer_base3: Int64
    peer_base4: Int64
    peer_base5: Int64
    peer_base6: Int64
    peer_base7: Int64


@dataclass(frozen=True)
class DenseFfnSmem(rt.SmemLayout):
    """The CTA's shared memory as 32-bit addresses (ring slots 1 KiB aligned)."""

    base: Int32
    RING_SLOTS = 4
    SLOT_BYTES = 34816
    X_ROW = HIDDEN + 8
    X_ROW_BYTES = 2 * X_ROW
    X_ROW_WORDS = X_ROW_BYTES // 4
    NORM_VEC = 8
    NORM_CHUNK_BYTES = COMPUTE_THREADS * NORM_VEC * 2
    NORM_CHUNKS = HIDDEN * 2 // NORM_CHUNK_BYTES
    UP_PARTIAL_STRIDE = TOKENS * 2 * UpWeights.ROWS + 1
    DOWN_PARTIAL_STRIDE = TOKENS * DownWeights.ROWS + 1
    SCRATCH_WORDS = 2048

    a = rt.SmemAlloc(SMEM_BYTES)
    ring_full = a.mbarriers(RING_SLOTS, arrivals=1)
    ring_empty = a.mbarriers(RING_SLOTS, arrivals=COMPUTE_THREADS)
    x_full = a.mbarriers(1, arrivals=1)
    scratch = a.array(Float32, SCRATCH_WORDS, align=1024)
    epoch = a.array(Uint32, 1)
    down_scales = a.array(Float32, (2, INTERMEDIATE // BLOCK), align=16)
    local_o = a.array(BFloat16, (TOKENS, DownWeights.ROWS), align=512, also=(Uint32,))
    a.pad(PEERS * TOKENS * DownWeights.ROWS * 2)  # unused; keeps the later offsets
    act = a.array(BFloat16, (Activations.ROWS, Activations.ROW_BYTES // 2))
    a.pad(64)
    xn = a.array(BFloat16, (TOKENS, X_ROW))
    ring = a.array(Uint8, (RING_SLOTS, SLOT_BYTES), align=1024)
    assert (epoch.offset, act.offset, xn.offset, ring.offset) == (
        9216,
        12800,
        25536,
        74752,
    )
    assert (
        COMPUTE_WARPS * DOWN_PARTIAL_STRIDE <= SCRATCH_WORDS
        and UpWeights.PAGE_BYTES <= SLOT_BYTES
    )

    def pages(self):
        """The weight ring's full/empty mbarrier pairs."""
        return rt.Pipeline.Ring(self.ring_full, self.ring_empty, self.RING_SLOTS)

    @staticmethod
    def slot_base(ring, slot):
        """Byte offset of ring slot ``slot`` from the ring, aligned up to 1024."""
        return slot * DenseFfnSmem.SLOT_BYTES + (
            (1024 - ((ring + slot * DenseFfnSmem.SLOT_BYTES) & 1023)) & 1023
        )

    @staticmethod
    def x_word(token, chunk):
        """u32 word (from ``xn``) of 16-K chunk ``chunk`` of X row ``token``."""
        return token * DenseFfnSmem.X_ROW_WORDS + chunk * (CHUNK // 2)

    @staticmethod
    def up_partial_word(warp, token, proj, row):
        """Scratch word of a gate/up partial: [warp][TOKENS][proj][16 rows], padded."""
        return (
            warp * DenseFfnSmem.UP_PARTIAL_STRIDE
            + token * (2 * UpWeights.ROWS)
            + proj * UpWeights.ROWS
            + row
        )


class DenseFfn(rt.UpDownMapKernel):
    """128 CTAs x 384 threads: compute warps 0-7, loader warps 8-11."""

    Params = DenseFfnParams
    GRID = GRID
    BLOCK = THREADS
    SMEM_BYTES = SMEM_BYTES
    MAX_THREADS = True
    SLICES = rt.SliceExchange(TOKENS)
    ROLES = (
        rt.Role("compute", 0, COMPUTE_WARPS - 1, regs=216, increase=True),
        rt.Role("load", COMPUTE_WARPS, THREADS // 32 - 1, regs=72, increase=False),
    )

    def smem(self, base):
        """The shared-memory record."""
        return DenseFfnSmem(base)

    def tensor_maps(self, p) -> tuple:
        """Host wrapper: the Up pages' (gate and up rows) and Down records'
        descriptors.
        """
        up_box, down_box = (
            [BLOCK, UpWeights.ROWS, UpWeights.PAGE_SLICES],
            [BLOCK, DownWeights.ROWS, 1],
        )
        down = p.DenseDown.toint() + DownWeights.WEIGHTS
        return (
            rt.byte_map(
                p.DenseUp.toint(),
                [BLOCK, 2 * INTERMEDIATE, HIDDEN // BLOCK],
                [HIDDEN, BLOCK],
                up_box,
                True,
                True,
            ),
            rt.byte_map(
                down,
                [INTERMEDIATE, HIDDEN, 1],
                [INTERMEDIATE, HIDDEN * INTERMEDIATE],
                down_box,
                True,
                True,
            ),
        )

    def mbarriers(self) -> tuple:
        return DenseFfnSmem.barriers()

    @cute.jit
    def compute(self, p, s, ids):
        """Warps 0-7: RMSNorm, gate/up (CTAs < 96), down projection, exchange."""
        tid, bid, warp, lane = ids.tid, ids.bid, ids.warp, ids.lane
        self.set_regs("compute")
        views, p = p.views(), p.addresses()
        pages = s.pages()
        gamma = [
            rt.bf16x8_to_f32(
                rt.Memory.ldg_nc_v4(p.Gamma + s.NORM_CHUNK_BYTES * j + Int64(tid) * 16)
            )
            for j in range(s.NORM_CHUNKS)
        ]
        cute.arch.griddepcontrol_wait()
        if bid < UpWeights.CTAS:
            rt.mbar_wait(s.x_full, 0)
            self.rmsnorm(tid, warp, lane, s, gamma)
        rt.bar_sync(*BAR_ALL)
        epoch = rt.Memory.lds(s.epoch, Uint32)
        first_down = Int32(0)
        if bid < UpWeights.CTAS:
            self.gate_up(tid, warp, lane, bid, p, s, pages)
            first_down = Int32(UpWeights.PAGES)
        total = self.down(warp, lane, bid, s, pages, first_down)
        self.exchange(tid, warp, lane, bid, views, s, epoch, total)

    @cute.jit
    def rmsnorm(self, tid, warp, lane, s, gamma):
        """Normalize the four X rows in place (warps 0-7): ``x * rsqrt(mean(x^2) + 1e-5)
        * gamma``.
        """
        for token in cutlass.range_constexpr(TOKENS):
            row = s.xn + s.X_ROW_BYTES * token + tid * 16
            values = []
            for j in cutlass.range_constexpr(s.NORM_CHUNKS):
                values += rt.bf16x8_to_f32(
                    rt.Memory.lds_v4(row + s.NORM_CHUNK_BYTES * j)
                )
            square = rt.Reduce.warp_sum_down(rt.Reduce.sum_squares(values))
            if lane == 0:
                rt.Memory.sts(s.scratch + 4 * warp, square)
            rt.bar_sync(*BAR_COMPUTE)
            inverse = rt.Reduce.rms_inverse(
                rt.Reduce.ordered_sum(
                    lambda w: rt.Memory.lds(s.scratch + 4 * w), COMPUTE_WARPS
                )
            )
            for j in cutlass.range_constexpr(s.NORM_CHUNKS):
                rt.Memory.sts_v4(
                    row + s.NORM_CHUNK_BYTES * j,
                    rt.Reduce.norm_bf16x8(
                        values[s.NORM_VEC * j : s.NORM_VEC * j + s.NORM_VEC],
                        inverse,
                        gamma[j],
                    ),
                )
            rt.bar_sync(*BAR_COMPUTE)

    @cute.jit
    def gate_up(self, tid, warp, lane, bid, p, s, pages):
        """Gate/up over six ring pages, SiLU(gate) * up, publish the CTA's
        activations.
        """
        ring = s.ring
        accum = [Float32(0.0)] * 8
        q = lane & 3
        chunks = UpWeights.chunk_pair(q)
        group = lane >> 2
        x_words = [s.x_word(lane >> 3, c) for c in chunks]
        for page in cutlass.range_constexpr(UpWeights.PAGES):
            slot = page % s.RING_SLOTS
            pages.wait_full(page)
            base = s.slot_base(ring, slot)
            lane_chunks = [UpWeights.chunk(base, warp, group, c) for c in chunks]
            a_words = [None] * 32
            for proj in cutlass.range_constexpr(2):
                for hi in cutlass.range_constexpr(2):
                    for c in cutlass.range_constexpr(2):
                        k = 16 * proj + 8 * hi + 4 * c
                        a_words[k : k + 4] = rt.Memory.lds_b128(
                            ring
                            + lane_chunks[c]
                            + 8 * SWIZZLE_ROW * hi
                            + UpWeights.BOX_BYTES * proj
                        )
            slot_scales = ring + base + UpWeights.SLOT_SCALES
            scales = [
                rt.Memory.lds(slot_scales + UpWeights.SCALE_BYTES * proj + 4 * warp)
                for proj in range(2)
            ]
            k_words = page * (UpWeights.PAGE_K // 2) + warp * (UpWeights.SLICE_K // 2)
            b_words = lambda k, k_words=k_words: rt.Memory.lds_weak_b64(
                s.xn + 4 * (x_words[k // 4] + k_words + 2 * (k % 4))
            )
            rt.Mma.gate_up_steps(accum, a_words, b_words, *scales)
            pages.release(slot)
        store = s.scratch + 4 * s.up_partial_word(warp, q, 0, group)
        for proj in cutlass.range_constexpr(2):
            for hi in cutlass.range_constexpr(2):
                word = s.up_partial_word(0, 0, proj, 8 * hi)
                rt.Memory.sts(store + 4 * word, accum[4 * proj + 2 * hi])
        rt.bar_sync(*BAR_COMPUTE)
        if tid < TOKENS * UpWeights.ROWS:
            token, n = tid >> UpWeights.ROW_SHIFT, tid & (UpWeights.ROWS - 1)
            gate, up = [
                rt.Convert.bf16_to_f32(
                    rt.Convert.f32_to_bf16(
                        rt.Reduce.ordered_sum(
                            lambda w, proj=proj: rt.Memory.lds(
                                s.scratch + 4 * s.up_partial_word(w, token, proj, n)
                            ),
                            COMPUTE_WARPS,
                        )
                    )
                )
                for proj in (0, 1)
            ]
            e = cute.math.exp2(rt.mul(gate, NEG_LOG2E), fastmath=True)
            rcp = rt.FastMath.rcp_rn(rt.add(1.0, e))
            col = Activations.col(
                (bid & (Activations.GROUP_CTAS - 1)) * UpWeights.ROWS + n
            )
            index = Activations.index(bid >> Activations.GROUP_SHIFT, token, col)
            dst = p.Activation + Int64(index) * 2
            rt.Memory.stg_b16(
                dst, rt.Convert.f32_to_bf16(rt.mul(rt.mul(gate, rcp), up))
            )
        rt.bar_sync(*BAR_COMPUTE)
        rt.bar_sync(*BAR_COMPUTE)
        with rt.warp_leader(warp, 0):
            rt.Memory.fence_release_gpu()
            rt.Memory.red_add_u32(p.Flags + Int64(bid) * 4, 1)

    @cute.jit
    def down(self, warp, lane, bid, s, pages, first_down):
        """Down projection over three ring pages; returns the 12 FP32 partials."""
        ring = s.ring
        total = [Float32(0.0)] * (4 * DownWeights.TILES)
        lane_word = DownWeights.lane_word(warp, lane)
        act_word = Activations.b_word(warp, lane)
        rt.bar_sync(*BAR_ALL)
        for page in cutlass.range_constexpr(DownWeights.PAGES):
            sequence = first_down + page
            slot = pages.slot(sequence)
            pages.wait_full(sequence, slot)
            base = s.slot_base(ring, slot)
            rows = [
                rt.Memory.lds(
                    ring + 8 * DownWeights.word(base, half, lane_word, i), Uint64
                )
                for half in range(2)
                for i in range(DownWeights.ROW_GROUPS)
            ]
            scales = []
            for half in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(DownWeights.TILES):
                    word = DownWeights.scale_word(bid, j, page, half, warp >> 2)
                    scales.append(rt.Memory.lds(s.down_scales + 4 * word))
            partial = [[None] * 12, [None] * 12]
            for kh in cutlass.range_constexpr(2):
                for half in cutlass.range_constexpr(2):
                    group = 2 * page + half
                    b = rt.Memory.lds_weak_b64(
                        s.act
                        + 4 * (Activations.GROUP_WORDS * group + act_word + 4 * kh)
                    )
                    acc = partial[half]
                    for j in cutlass.range_constexpr(DownWeights.TILES):
                        quad, row = (
                            slice(4 * j, 4 * j + 4),
                            DownWeights.ROW_GROUPS * half + 2 * j,
                        )
                        acc[quad] = rt.Mma.step_u64(
                            acc[quad], rows[row], rows[row + 1], kh, b, kh == 0
                        )
            pages.release(slot)
            for half in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(DownWeights.TILES):
                    quad, scale = (
                        slice(4 * j, 4 * j + 4),
                        scales[DownWeights.TILES * half + j],
                    )
                    total[quad] = rt.Mma.scale_accumulate(
                        total[quad], partial[half][quad], scale
                    )
        return total

    @cute.jit
    def exchange(self, tid, warp, lane, bid, p, s, epoch, total):
        """Reduce the warps' partials, write LocalO, exchange tagged packets, write
        ``Final``.
        """
        v = rt.Memory.smem_views(s.base, DenseFfnSmem.views())
        words = v["local_o_words"]
        self.SLICES.reduce_rows(
            v["scratch"], v["local_o"], p.LocalO, total, tid, warp, lane, bid
        )
        self.SLICES.push_peers(p, words, epoch, bid, tid, warp, lane)
        if warp == 0:
            self.SLICES.combine(p, p.X, lane, bid, epoch, words)

    @cute.jit
    def load(self, p, s, ids):
        """Warps 8-11: launch ticket, weight pages, X rows, then the activation
        gather.
        """
        tid, bid, warp = ids.tid, ids.bid, ids.warp
        self.set_regs("load")
        p = p.addresses()
        pages = s.pages()
        with rt.warp_leader(warp, TICKET_WARP):
            ticket = rt.Memory.atom_add_u32(p.FullEpoch, 1)
            rt.Memory.sts(s.epoch, ticket >> GRID_SHIFT, Uint32)
        if bid >= UpWeights.CTAS:  # noqa: SIM102
            if warp == PRODUCER_WARP:
                for page in cutlass.range_constexpr(DownWeights.PAGES):
                    self.load_down(bid, p, s, pages, page, page)
        cute.arch.griddepcontrol_wait()
        if bid < UpWeights.CTAS:  # noqa: SIM102
            if warp == PRODUCER_WARP:
                if rt.elect_sync() != 0:
                    rt.mbar_expect_tx(s.x_full, TOKENS * HIDDEN * 2)
                    for token in cutlass.range_constexpr(TOKENS):
                        row = s.xn + s.X_ROW_BYTES * token
                        rt.bulk_g2s(row, p.X + 2 * HIDDEN * token, HIDDEN * 2, s.x_full)
                for page in cutlass.range_constexpr(s.RING_SLOTS):
                    self.load_up(bid, p, s, pages, page)
        rt.bar_sync(*BAR_ALL)
        epoch = rt.Memory.lds(s.epoch, Uint32)
        if bid < UpWeights.CTAS:  # noqa: SIM102
            if warp == PRODUCER_WARP:
                for page in cutlass.range_constexpr(s.RING_SLOTS, UpWeights.PAGES):
                    self.load_up(bid, p, s, pages, page)
                for page in cutlass.range_constexpr(DownWeights.PAGES):
                    self.load_down(bid, p, s, pages, page, UpWeights.PAGES + page)
        loader_lane = tid - COMPUTE_THREADS
        flag = p.Flags + Int64(cutlass.min(loader_lane, UpWeights.CTAS - 1)) * 4
        ready = Int32(0)
        while ready == 0:
            satisfied = Int32(1)
            if loader_lane < UpWeights.CTAS:
                observed = rt.Memory.ld_relaxed_gpu(flag)
                satisfied = Int32(observed >= epoch + 1)
            ready = Int32(rt.Sync.vote_all_asm(satisfied))
        rt.Memory.fence_acquire_gpu()
        rt.bar_sync(*BAR_LOADER)
        for k in cutlass.range_constexpr(Activations.CHUNKS_PER_LANE):
            chunk = loader_lane + LOADER_THREADS * k
            rt.AsyncCopy.cp_async_cg16_clobber(
                Activations.smem_addr(s.act, chunk), p.Activation + Int64(chunk) * CHUNK
            )
        rt.AsyncCopy.cp_async_commit_clobber()
        rt.AsyncCopy.cp_async_wait_clobber(0)
        rt.bar_sync(*BAR_ALL)

    @cute.jit
    def load_up(self, bid, p, s, pages, page: cutlass.Constexpr):
        """Producer: gate/up page ``page`` (two TMA boxes and their scales)."""
        ring, full = s.ring, pages.full(page % s.RING_SLOTS)
        if rt.elect_sync() != 0:
            pages.acquire(page, UpWeights.PAGE_BYTES)
            base = ring + s.slot_base(ring, page % s.RING_SLOTS)
            block = (
                bid >> UpWeights.CTA_BLOCK_SHIFT
            ) * UpWeights.SCALE_COLS + page * UpWeights.PAGE_SLICES
            for proj in cutlass.range_constexpr(2):
                box = base + UpWeights.BOX_BYTES * proj
                row, col = (
                    bid * UpWeights.ROWS + INTERMEDIATE * proj,
                    page * UpWeights.PAGE_SLICES,
                )
                rt.AsyncCopy.tma_3d(box, self.up_map, 0, row, col, full, TMA_HINT)
                scale_slot = base + UpWeights.SLOT_SCALES + UpWeights.SCALE_BYTES * proj
                scale_src = p.DenseUp + Int64(UpWeights.scale_offset(block, proj))
                rt.bulk_g2s(scale_slot, scale_src, UpWeights.SCALE_BYTES, full)

    @cute.jit
    def load_down(
        self, bid, p, s, pages, page: cutlass.Constexpr, sequence: cutlass.Constexpr
    ):
        """Producer: down page ``page`` as ring ``sequence`` (page 0 adds the
        scales).
        """
        ring, full = s.ring, pages.full(sequence % s.RING_SLOTS)
        if rt.elect_sync() != 0:
            pages.acquire(
                sequence,
                DownWeights.PAGE_BYTES + (DownWeights.SCALE_BYTES if page == 0 else 0),
            )
            base = ring + s.slot_base(ring, sequence % s.RING_SLOTS)
            for box in cutlass.range_constexpr(DownWeights.BOXES):
                dst, k = (
                    base + DownWeights.BOX_BYTES * box,
                    DownWeights.PAGE_K * page + BLOCK * box,
                )
                rt.AsyncCopy.tma_3d(
                    dst, self.down_map, k, bid * DownWeights.ROWS, 0, full, TMA_HINT
                )
            if cutlass.const_expr(page == 0):
                scale_src = p.DenseDown + Int64(DownWeights.scale_offset(bid))
                rt.bulk_g2s(s.down_scales, scale_src, DownWeights.SCALE_BYTES, full)
