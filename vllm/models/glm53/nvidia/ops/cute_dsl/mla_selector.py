# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 MLA selector (DSA indexer): q4 query/key projection, key scoring and top-2048
selection.
"""

import dataclasses
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass.base_dsl.typing import GridConstant
from cutlass.experimental.cuda.tensor_map import TensorMap

from . import _rt as rt
from .mla_worker import INPUT_PAGES, L2_HINT, MlaWorker

cdiv, cmod = rt.cdiv, rt.cmod
ga, ballot, popc = rt.ga, rt.Sync.ballot, rt.Sync.popc
ld_cg_b32, ld_cg_v2 = rt.Memory.ld_cg_b32, rt.Memory.ld_cg_v2
st_volatile_u64, ld_volatile_u64 = rt.Memory.st_volatile_u64, rt.Memory.ld_volatile_u64
bits_f32, ffs, as_u32 = rt.Convert.bits_f32, rt.Convert.ffs, rt.Convert.as_u32
fmul, fadd, ffma = rt.mul, rt.add, rt.fma
rmem_fill, mbar_wait, ld_cg_v4 = (rt.Memory.rmem_fill, rt.mbar_wait, rt.Memory.ld_cg_v4)
redux_add_u32, redux_max_u32 = rt.Sync.redux_add_u32, rt.Sync.redux_max_u32
elect_sync = rt.elect_sync
SMEM_BYTES = 215040
CTAS = 148
QA_SCALES = 2624 * 6144
WQB_SCALES = 4096 * 2048


class SelectorSmem(rt.SmemLayout):
    """The selector's dynamic shared memory in address order; views run to the end of
    smem.
    """

    a = rt.SmemAlloc(SMEM_BYTES)
    ring_full = a.mbarriers(4, arrivals=1)
    ring_empty = a.mbarriers(4, arrivals=256)
    x_full = a.mbarriers(1, arrivals=1)
    mma_done = a.mbarriers(1, arrivals=1)
    spare_bar = a.mbarriers(1, arrivals=1)  # initialized, never used
    ws_full = a.mbarriers(4, arrivals=32)
    ws_empty = a.mbarriers(4, arrivals=1)
    ws_done = a.mbarriers(4, arrivals=1)
    ws_free = a.mbarriers(4, arrivals=256)
    ws_query = a.mbarriers(1, arrivals=1)
    ws_scored = a.mbarriers(1, arrivals=1)
    TMEM_HOLD = a.alloc(4, align=8)  # the TMEM base word, addressed at its uses
    with a.union(align=1024):
        scratch = a.array(Float32, 1536)
        with a.arm():
            near_meta = a.array(Int32, 64)
            keep_meta = a.array(Int32, 448)
            unit_max = a.array(Uint32, 1024)
    q_scales = a.array(Float32, 1152)
    residual = a.array(BFloat16, 256)
    local_o = a.array(BFloat16, 256, also=(Uint32,))
    received_words = a.array(Uint32, 1088)
    epoch_slot = a.array(Uint32, 64)
    with a.union():
        ring = a.array(Uint8, 4 * 36864, also=(Uint64, Uint32, Float32))
        with a.arm():
            with a.union():
                hist = a.array(Uint32, 32)
                coarse = a.array(Uint32, 32)
            batch_bounds = a.array(Int32, 32)
            batch_counts = a.array(Int32, 192)
            warp_eq = a.array(Int32, 8)
            warp_sel = a.array(Int32, 8)
            control = a.array(Int32, 8)
            with a.union():
                refine_in = a.array(Uint32, 8)
                shard_in = a.array(Uint32, 8)
            tie_bits = a.array(Uint32, 224)
            refine_words = a.array(Uint32, 464)
            batch_cands = a.array(Int32, 256)
            batch_first = a.array(Int32, 256)
            near_counts = a.array(Int32, 912)
            near_offsets = a.array(Int32, 912)
            near_prefix = a.array(Int32, 912)
            round_first = a.array(Int32, 1920)
            rescore_keys8 = a.array(Uint8, 40960)
            coarse_scan = a.array(Int32, 272)
            emission = a.array(Uint32, 752)
            with a.union():
                stash = a.array(Float32, 4096)
                umma_keys8_next = a.array(Uint8, 16384)
            ws_tile1 = a.array(Uint8, 16384)
            with a.union():
                keep_words = a.array(Uint32, 4096)
                umma_keys = a.array(BFloat16, 8192)
            ws_tile3 = a.array(Uint8, 16384)
            near_words = a.array(Uint32, 128)
            loader_bounds = a.array(Int32, 896)
            group_counts = a.array(Uint32, 512)
            key_scales = a.array(Float32, 1536)
    with a.union():
        act = a.array(BFloat16, 8672, also=(Uint32,))
        query_words = a.array(Uint8, 17344)
    tie_blocks = a.array(Uint32, 4368)
    head_weight = a.array(Float32, 3840)

    ORDER = """scratch residual local_o received_words epoch_slot ring act
    head_weight hist warp_eq warp_sel control refine_words refine_in tie_bits
    tie_blocks coarse coarse_scan group_counts key_scales emission stash
    batch_bounds near_words near_meta q_scales unit_max keep_words keep_meta
    loader_bounds shard_in round_first near_counts near_offsets near_prefix
    batch_counts batch_cands batch_first umma_keys""".split()  # noqa: SIM905


#: The region offsets of the served layout.
_OFFSETS = """0 32 64 72 80 88 120 152 184 216 224 1024 1024 1280 3072 7168 11776
12288 12800 17152 17408 17408 17408 17536 17664 18432 18464 18496 18528 18528 18560
19456 21312 22336 23360 27008 30656 34304 41984 82944 84032 87040 87040 103424
119808 119808 136192 152576 153088 156672 158720 164864 164864 182208 199680 215040"""
_REGIONS = SelectorSmem.REGIONS
assert [r.offset for r in _REGIONS] + [_REGIONS[-1].offset + _REGIONS[-1].nbytes] == [
    int(x) for x in _OFFSETS.split()
]


@dataclass(frozen=True)
class SelectorParams(rt.Params):
    """The selector parameters in order; the paged entries append theirs."""

    X: cute.Pointer = rt.ptr("bfloat16")
    GammaX: cute.Pointer = rt.ptr("float32")
    QakiWeight: cute.Pointer = rt.ptr("float8_e4m3")
    WisWeight: cute.Pointer = rt.ptr("float8_e4m3")
    KeyGamma: cute.Pointer = rt.ptr("float32")
    KeyBias: cute.Pointer = rt.ptr("float32")
    GammaQ: cute.Pointer = rt.ptr("float32")
    WqbWeight: cute.Pointer = rt.ptr("float8_e4m3")
    Freq: cute.Pointer = rt.ptr("float32")
    CurPos: cute.Pointer = rt.ptr("int32")
    IndexKeyCache: cute.Pointer = rt.ptr("bfloat16")
    QA: cute.Pointer = rt.ptr("bfloat16")
    RawKey: cute.Pointer = rt.ptr("bfloat16")
    HeadWeights: cute.Pointer = rt.ptr("bfloat16")
    RawQuery: cute.Pointer = rt.ptr("bfloat16")
    IndexQuery: cute.Pointer = rt.ptr("bfloat16")
    Scores: cute.Pointer = rt.ptr("float32")
    SelectedIds: cute.Pointer = rt.ptr("int32")
    Final: cute.Pointer = rt.ptr("bfloat16")
    FullEpoch: cute.Pointer = rt.ptr("uint32")
    Flags: cute.Pointer = rt.ptr("uint32")
    SelectState: cute.Pointer = rt.ptr("uint32")
    score_stride: Int32
    peer_base0: Int64
    peer_base1: Int64
    peer_base2: Int64
    peer_base3: Int64
    peer_base4: Int64
    peer_base5: Int64
    peer_base6: Int64
    peer_base7: Int64
    selection_base1: Int64
    selection_base2: Int64
    selection_base3: Int64
    selection_base4: Int64
    selection_base5: Int64
    selection_base6: Int64
    selection_base7: Int64

    BlockTable = None
    UNSIGNED = dict.fromkeys(("QakiWeight", "WisWeight", "WqbWeight"), Uint8)
    UNSIGNED |= dict.fromkeys(("FullEpoch", "Flags", "SelectState"), Uint32)

    @property
    def selection_bases(self) -> list:
        """``selection_base1..7`` (index 0 is None: rank-local selection words live at
        peer_base0).
        """
        return [None] + [getattr(self, f"selection_base{i}") for i in range(1, 8)]

    def shard_base(self, shard: int):
        """Trace time: the mailbox base of key shard ``shard``; shard 0's is reached
        through peer 0.
        """
        if shard == 0:
            return self.peer_base0 + (self.selection_base1 - self.peer_base1)
        return self.selection_base1 if shard == 1 else self.selection_base2

    @property
    def shard_bases(self) -> tuple:
        """The three shard mailboxes."""
        return tuple(self.shard_base(i) for i in range(3))

    @property
    def near_bases(self) -> tuple:
        """The five replica mailboxes ``selection_base3..7``."""
        return tuple(self.selection_bases[3:])


@dataclass(frozen=True)
class PagedParams(SelectorParams):
    """``mla_selector_q4_shard3r{0,1,2}_paged_vlin``: + the paged index-key cache's
    block table.
    """

    BlockTable: cute.Pointer = rt.ptr("int32")


class StateWords:
    """SelectState, the rank's global u32 selection state, as word indices."""

    BUCKET = 32768
    EXTREMES = 32784
    REFINE = 32800
    REFINE_TOKEN = 8208
    HEADER = 160
    RECORDS = 65632
    RECORD_ITEMS = 59
    GROUPS = 263520
    GROUP_COUNT = 128
    SEEDS = 264544
    STAGED_RUNS = 264548
    FULL_SCANS = 264549
    PREVIOUS_SEED = 264564
    BINS = 264566
    BIN_COUNT = 4096

    @staticmethod
    def refine(token):
        """First word of ``token``'s refine block."""
        return StateWords.REFINE + token * StateWords.REFINE_TOKEN

    @staticmethod
    def refine_word(token, word):
        """Word ``word`` of ``token``'s refine block (a CTA's count, or HEADER +
        parity*8 + i).
        """
        return StateWords.refine(token) + word

    @staticmethod
    def cta_record(token, cta, item):
        """Word index of record ``item`` of CTA ``cta`` for ``token``."""
        return (
            StateWords.RECORDS
            + ((token * 144 + cta) * StateWords.RECORD_ITEMS + item) * 2
        )

    @staticmethod
    def bucket(parity, token):
        """Word of (parity, token)'s keys-above count (+1: candidate count)."""
        return StateWords.BUCKET + (parity * 4 + token) * 2

    @staticmethod
    def group(parity, token, group):
        """Word of coarse group ``group`` (0..127) of (parity, token)."""
        return StateWords.GROUPS + (parity * 4 + token) * StateWords.GROUP_COUNT + group

    @staticmethod
    def bin(parity, token, *terms):
        """Word of fine bin ``sum(terms)`` (summed left to right after the row base;
        keep this order: PTX identity).
        """
        word = (parity * 4 + token) * StateWords.BIN_COUNT
        for term in terms:
            word = word + term
        return StateWords.BINS + word * 2


class SelectionMailbox:
    """The selection mailboxes (u64 records at mailbox base + 131072): the selected ids
    ([2 parities][4 tokens][2048]) and, from byte 2078720, the shard exchange words
    ([2 parities][3 source shards][4 tokens][4224]: bins 0..4095, then the 128 group
    counts).
    """

    @staticmethod
    def word_addr(base, parity, shard, token, idx) -> Int64:
        """Address of exchange word ``idx`` of ``shard``/``token`` in ``parity`` (int
        math, then widened).
        """
        return base + Int64(
            131072 + (2078720 + (((parity * 3 + shard) * 4 + token) * 4224 + idx) * 8)
        )


class PartialMailbox:
    """The reduce's partial mailboxes (u32 words at a peer base): [4 rows: 57344][7
    source slots: 8192] [2 parities: 4096][CTA: 32][8 chunks: 4] 16-byte tagged
    packets (three payload words + tag).
    """

    @staticmethod
    def recv_addr(base, item, parity, bid):
        """Address of incoming packet ``item`` (row item/56, source slot item/8 % 7,
        chunk item % 8) for CTA ``bid``.
        """
        word = (
            cdiv(item, 56) * 57344
            + cmod(cdiv(item, 8), 7) * 8192
            + parity * 4096
            + bid * 32
            + cmod(item, 8) * 4
        )
        return base + Int64(word) * 4

    @staticmethod
    def store_received(received_words, item, packet):
        """The three payload words of one tagged packet into received_words (24 words
        per source slot).
        """
        payload = rt.TaggedMailbox.unpack24(packet)
        for i in range(3):
            received_words[cdiv(item, 8) * 24 + cmod(item, 8) * 3 + i] = payload[i]


SEL_TOKENS = 4
KEY_WARPS = 4
KEY_BLOCK = 128
SCORE_CTAS = 144
ROUND_KEYS = SCORE_CTAS * KEY_BLOCK
MAX_ROUNDS = 57
WINDOW_ROUNDS = 16
TOPK = 2048
KEEP_CAP = 256
CAND_CAP = 16240
CAND_OVERFLOW = CAND_CAP + 1
STREAM_KEYS = 4 * ROUND_KEYS
DIRECT_KEYS = 65536


class Keys:
    """Index-key ids and the shard striping."""

    @staticmethod
    def shard_live(p, shard):
        """Keys of a 3-way 128-row striped shard that lie below position p (plain
        integer ops).
        """
        return cdiv(cdiv(p, 128) - shard + 3 - 1, 3) * 128 + Int32(
            cmod(cdiv(p, 128), 3) == shard
        ) * (cmod(p, 128))

    @staticmethod
    @cute.jit
    def rounds_below(rows, bid):
        """18432-row rounds CTA ``bid`` owns below shard row ``rows``: 0 unless bid*128
        < rows.
        """
        rounds = Int32(0)
        if bid * 128 < rows:
            rounds = cdiv(rows - 1 - bid * 128, 18432) + 1
        return rounds

    @staticmethod
    def shard_rounds(live, bid, SHARD: int):
        """18432-row rounds this CTA owns below ``live`` (live keys of the 3-way striped
        shard).
        """
        return Keys.rounds_below(Keys.shard_live(live, SHARD), bid)

    @staticmethod
    def shard_id(index, SHARD: int):
        """Global key id of this shard's local ``index``: (index/128*3 + SHARD)*128 +
        index%128.
        """
        return (cdiv(index, 128) * 3 + SHARD) * 128 + cmod(index, 128)

    @staticmethod
    def exact_take(key, threshold, limit, index, SHARD: int):
        """Key > threshold, or key == threshold with limit >= id (the emission tie-
        break).
        """
        return Int32(key > threshold) | Int32(
            (key == threshold) & (limit >= Uint32(Keys.shard_id(index, SHARD)))
        )

    @staticmethod
    def takes(w0, w1, threshold, limit) -> Int32:
        """(key > threshold) | (key == threshold && id <= limit) on a cached (id, key)
        pair.
        """
        return Int32(w1 > threshold) | (Int32(w1 == threshold) & Int32(w0 <= limit))


class BucketMailbox:
    """A shard's selection mailbox (u64 words from ``base + 131072``): bucket exchange
    and candidates.
    """

    BASE = 131072
    WORDS = 2078720
    TOKEN_WORDS = 4224
    GROUP_WORDS = 4096

    @staticmethod
    def candidate(parity, token, slot):
        """Byte offset (from BASE) of candidate ``slot`` of (parity, token)."""
        return ((parity * 4 + token) * CAND_CAP + slot) * 16

    @staticmethod
    def push_words(bases, off, words):
        """Trace-time: one st.volatile.global.v4.b32 of ``words`` at base + 131072 + off
        per base.
        """
        for b in bases:
            rt.Memory.st_volatile_v4(b + BucketMailbox.BASE + off, words)

    @staticmethod
    @cute.jit
    def push_candidate(near_bases, near_flag, shard_bases, off, sid, key, launch_epoch):
        """One bucket candidate as words (id, gen, key, gen) at base + 131072 + off: the
        near replicas when ``near_flag`` (base kernel only; pull/r0 pass no near
        bases), then the three shard mailboxes.
        """
        words = (Uint32(sid), launch_epoch + 1, key, launch_epoch + 1)
        if cutlass.const_expr(len(near_bases) > 0):  # noqa: SIM102
            if near_flag != 0:
                BucketMailbox.push_words(near_bases, off, words)
        BucketMailbox.push_words(shard_bases, off, words)


class IdMailbox:
    """Selected-id records: u64 ``(epoch + 1) << 32 | id`` at ``base + (((epoch+1)%2*4 +
    token)*2048 + slot)*8``.
    """

    @staticmethod
    def row(epoch, token):
        """First record of (next parity, token): ((epoch + 1) % 2 * 4 + token) *
        2048.
        """
        return Int32((cmod(epoch + 1, 2) * 4 + Uint32(token)) * TOPK)

    @staticmethod
    def word(epoch, sid):
        """(epoch + 1) << 32 | id."""
        return (Uint64(epoch + 1) << 32) | Uint64(sid)

    @staticmethod
    def publish(bases, slot, record):
        """Trace-time: one generic st.volatile.u64 of ``record`` at base + slot*8 per
        base, in order.
        """
        for b in bases:
            st_volatile_u64(b + Uint64(slot) * 8, record)


class KeepList:
    """keep_words (smem u32): [8 warps][2 sides][256 entries][id, key]; side t holds
    token 2t + warp/4.
    """

    @staticmethod
    def word(row, entry):
        """First word (id) of entry ``entry`` of list ``row`` = warp*2 + side."""
        return (row * KEEP_CAP + entry) * 2

    @staticmethod
    @cute.jit
    def entry(keep_words, row, length, base, lane):
        """One keep-list entry per lane: (present, index, key) from keep_words row
        ``row``.
        """
        present = Int32(length > base + lane)
        index = Int32(0)
        key = Uint32(0)
        if present != 0:
            index = Int32(keep_words[(row * 256 + (base + lane)) * 2] & 2147483647)
            key = keep_words[(row * 256 + (base + lane)) * 2 + 1]
        return present, index, key


class Rounds:
    """Per-(token, round, key warp) tables in smem (unit_max, batch_counts, near_counts,
    ...): [4][57][4].
    """

    @staticmethod
    def row(token, rnd):
        """Index of (token, round, key warp 0); callers add the key warp after it (keep
        this order: PTX identity).
        """
        return (token * MAX_ROUNDS + rnd) * KEY_WARPS

    @staticmethod
    def cell(token, rnd, key_warp):
        """Index of (token, round, key warp)."""
        return (token * MAX_ROUNDS + rnd) * KEY_WARPS + key_warp

    @staticmethod
    def window(token, rnd, key_warp):
        """Index in a [4 tokens][16 window rounds][4 key warps] table (batch_cands,
        batch_first).
        """
        return (token * WINDOW_ROUNDS + rnd) * KEY_WARPS + key_warp


class Stash:
    """The stash (smem f32): [4 tokens][16 window rounds][128 keys] of this CTA's
    scores.
    """

    @staticmethod
    def index(token, rnd, key):
        """Index of key row ``key`` of window round ``rnd`` for ``token``."""
        return (token * WINDOW_ROUNDS + rnd) * KEY_BLOCK + key

    @staticmethod
    def row(token, rnd, first=None):
        """Index of key row 0 of round ``rnd`` (relative to the window's ``first``
        round) for ``token``.
        """
        return (
            token * WINDOW_ROUNDS + (rnd if first is None else rnd - first)
        ) * KEY_BLOCK


class KeepMeta:
    """keep_meta (smem i32): list lengths [8 warps][2 sides], the listing flags, per-
    token words.
    """

    USABLE = 16
    COUNTING = 17
    PRELIST_OK = 18
    EMITTED = 19
    INSIDE = 20
    FIRST = 36
    FLOOR_PREFIX = 52
    FLOOR_SHIFT = 56
    FLOOR_FOUND = 60


class NearMeta:
    """near_meta (smem i32): direct-mode bookkeeping."""

    ABOVE = 4
    INSIDE = 20
    RESERVED = 36
    LOWER = 40
    FIRST_ABOVE = 44
    WARM = 60
    ALL_DIRECT = 61


class Coarse:
    """coarse_scan (smem i32): [4] first coarse group of the token's window, [4] its
    shift, word 13: "a window needs the stash refilled".
    """

    SHIFT = 4
    STASH_NEEDED = 13

    @staticmethod
    def fine(shift, group) -> Int32:
        """1 when a crossing group needs its fine bins: an unseeded window (shift 20) or
        the band 56..79.
        """
        return Int32(shift == 20) | Int32((group >= 56) & (group < 80))

    @staticmethod
    def group_floor(previous) -> Int32:
        """min(max((int)(previous >> 15) - 2048) / 32 * 32, 0), 126976) (C '/' on a
        value clamped right after).
        """
        return cutlass.min(
            cutlass.max(cdiv(Int32(previous >> 15) - 2048, 32) * 32, 0), 126976
        )


class Refine:
    """refine_words (smem u32) on the final CTAs: [144] per-CTA taken counts, then the
    staged candidate cache of (id, key) pairs from word 144, the 1024-bin histogram
    at 32624 and the staged own ids at 33648.
    """

    CACHE = 144
    HIST = 32624
    STAGED = 33648


class Projection:
    """The X / q_b projections of the compute role: RMSNorm rows, FP8 ring-page MMAs,
    warp partials.
    """

    @staticmethod
    @cute.jit
    def rms_rows(
        values,
        gamma,
        scratch,
        act_addr,
        tid,
        warp,
        lane,
        NVEC: cutlass.Constexpr,
        ROW: cutlass.Constexpr,
    ):
        """RMSNorm of 4 token rows held as NVEC 8-wide chunks per thread
        (``values[q*8*NVEC + j*8 + i]``), then bf16 stores to act (row stride ROW).
        """
        squares = cute.make_rmem_tensor(4, Float32)
        inverse = cute.make_rmem_tensor(4, Float32)
        for q in cutlass.range_constexpr(4):
            squares[q] = Float32(0.0)
        for j in cutlass.range_constexpr(NVEC):
            for pair in cutlass.range_constexpr(4):
                for q in cutlass.range_constexpr(4):
                    v0 = values[q * 8 * NVEC + j * 8 + pair * 2]
                    v1 = values[q * 8 * NVEC + j * 8 + pair * 2 + 1]
                    squares[q] = fadd(squares[q], ffma(v0, v0, fmul(v1, v1)))
        for step in cutlass.range_constexpr(5):
            for q in cutlass.range_constexpr(4):
                squares[q] = fadd(
                    squares[q], cute.arch.shuffle_sync_down(squares[q], 16 >> step)
                )
        if lane == 0:
            for q in cutlass.range_constexpr(4):
                scratch[q * 8 + warp] = squares[q]
        rt.bar_sync(8, 256)
        if cutlass.const_expr(NVEC == 3):
            if tid < 4:
                scratch[32 + tid] = rt.FastMath.rsqrt(
                    fadd(
                        rt.FastMath.div_rn(
                            Projection.col_sum(scratch, tid * 8, 1), Float32(6144.0)
                        ),
                        Float32(1e-05),
                    )
                )
            rt.bar_sync(8, 256)
            for q in cutlass.range_constexpr(4):
                inverse[q] = scratch[32 + q]
        else:
            for q in cutlass.range_constexpr(4):
                inverse[q] = rt.FastMath.rsqrt(
                    ffma(Projection.col_sum(scratch, q * 8, 1), 0.00048828125, 1e-05)
                )
        for j in cutlass.range_constexpr(NVEC):
            normalized = cute.make_rmem_tensor(32, Float32)
            for item in cutlass.range_constexpr(8):
                for q in cutlass.range_constexpr(4):
                    normalized[q * 8 + item] = fmul(
                        values[q * 8 * NVEC + j * 8 + item],
                        fmul(gamma[j * 8 + item], inverse[q]),
                    )
            for q in cutlass.range_constexpr(4):
                rt.Memory.sts_v4(
                    act_addr + (q * ROW + j * 2048 + tid * 8) * 2,
                    [
                        rt.f32x2_to_bf16x2(
                            normalized[q * 8 + 2 * lp], normalized[q * 8 + 2 * lp + 1]
                        )
                        for lp in range(4)
                    ],
                )

    @staticmethod
    def col_sum(scratch, slot, stride):
        """scratch[slot] + scratch[stride + slot] + ..."""
        value = scratch[slot]
        for w in range(1, 8):
            value = fadd(value, scratch[w * stride + slot])
        return value

    @staticmethod
    @cute.jit
    def vin_proj_pages(
        accum,
        first,
        npages,
        ring_addr,
        ring_f32,
        act_addr,
        warp,
        lane,
        pages,
        AROW: cutlass.Constexpr,
        SCALES: cutlass.Constexpr = -1,
    ):
        """FP8 pages of vLLM's row-major weight (TMA, 128B swizzle: 8 K planes of 32
        rows x 128 B) times the staged bf16 activations, in the workers' vLLM-input K
        order (each lane takes 32 K of its warp's plane); two m16 tiles, then
        ``accum[i] = fma(page[i], scale, accum[i])`` with the page's eight K-block
        scales at f32 8192 of its slot (``SCALES`` >= 0: every page's at f32 ``SCALES
        + 8 page`` of the ring).
        """
        g = cdiv(lane, 4)
        q = cmod(lane, 4)
        for page in range(npages):
            sequence = first + page
            pages.wait_full(sequence)
            cute.arch.fence_proxy("async.shared", space="cta")
            region = cmod(sequence, 4) * 36864
            plane = cdiv(region + warp * 4096 + g * 128, 4)
            rows = [None] * 16
            for j in cutlass.range_constexpr(4):
                for h in cutlass.range_constexpr(2):
                    w = rt.Memory.lds_b128(
                        ring_addr + (plane + ((2 * q + h) ^ g) * 4 + 256 * j) * 4
                    )
                    rows[4 * j + 2 * h] = Uint64(w[0]) | Uint64(w[1]) << 32
                    rows[4 * j + 2 * h + 1] = Uint64(w[2]) | Uint64(w[3]) << 32
            k = cmod(g, 4) * AROW + q * 16 + cdiv(page * 1024 + warp * 128, 2)
            page_acc = [None] * 8
            for j in cutlass.range_constexpr(4):
                a = [None] * 16
                for h in cutlass.range_constexpr(2):
                    even = rt.Convert.unpack_u16x4(rows[j + 8 * h])
                    odd = rt.Convert.unpack_u16x4(rows[j + 4 + 8 * h])
                    for i in cutlass.range_constexpr(4):
                        a[8 * h + 2 * i] = rt.Convert.e4m3x2_to_bf16x2(even[i])
                        a[8 * h + 2 * i + 1] = rt.Convert.e4m3x2_to_bf16x2(odd[i])
                for t in cutlass.range_constexpr(2):
                    b = rt.Memory.lds_weak_b64(act_addr + (k + 4 * j + 2 * t) * 4)
                    for h in cutlass.range_constexpr(2):
                        if cutlass.const_expr(j == 0 and t == 0):
                            page_acc[4 * h : 4 * h + 4] = rt.TensorCore.mma(
                                None, tuple(a[8 * h : 8 * h + 4]), tuple(b), True
                            )
                        else:
                            rt.TensorCore.mma_acc_at(
                                page_acc,
                                4 * h,
                                tuple(a[8 * h + 4 * t : 8 * h + 4 * t + 4]),
                                tuple(b),
                            )
            scale_word = cdiv(region, 4) + 8192 + warp
            if cutlass.const_expr(SCALES >= 0):
                scale_word = SCALES + page * 8 + warp
            scale = ring_f32[scale_word]
            for i in cutlass.range_constexpr(8):
                accum[i] = ffma(page_acc[i], scale, accum[i])
            pages.release(pages.slot(sequence))

    @staticmethod
    @cute.jit
    def bf16_plane_mma(
        accum,
        line0,
        ring_words,
        act_words,
        lane,
        k0,
        M_TILES: cutlass.Constexpr,
        AROW: cutlass.Constexpr,
    ):
        """One 64-column K plane of a BF16 row-major page (TMA, 128B swizzle; rows 128 B
        apart from byte ``line0``) for ``M_TILES`` m16 tiles: 4 k16 steps of mma.sync
        into ``accum[4m..4m+3]``, in K order.
        """
        g = cdiv(lane, 4)
        q = cmod(lane, 4)
        bb = cute.make_rmem_tensor(2, Uint32)
        for kk in cutlass.range_constexpr(4):
            bb[0] = Uint32(0)
            bb[1] = Uint32(0)
            if g < 4:
                word = g * AROW + cdiv(k0 + 16 * kk + 2 * q, 2)
                bb[0] = act_words[word]
                bb[1] = act_words[word + 4]
            lo = ((2 * kk) ^ g) * 16 + 4 * q
            hi = ((2 * kk + 1) ^ g) * 16 + 4 * q
            for m in cutlass.range_constexpr(M_TILES):
                row = line0 + (16 * m + g) * 128
                a = [
                    ring_words[cdiv(row + lo, 4)],
                    ring_words[cdiv(row + 1024 + lo, 4)],
                    ring_words[cdiv(row + hi, 4)],
                    ring_words[cdiv(row + 1024 + hi, 4)],
                ]
                rt.TensorCore.mma_acc_at(accum, 4 * m, a, (bb[0], bb[1]))

    @staticmethod
    @cute.jit
    def mma_tiles_to_scratch(
        scratch, accum, warp, lane, TILES: cutlass.Constexpr, STRIDE: cutlass.Constexpr
    ):
        """Each warp's m16n8 accumulators (TILES m-tiles of 16 rows, 4 valid columns)
        into scratch[warp*STRIDE + ..].
        """
        group = cdiv(lane, 4)
        lane_pair = cmod(lane, 4)
        if lane_pair * 2 < 4:
            for m in cutlass.range_constexpr(TILES):
                base = warp * STRIDE + lane_pair * 2 * 16 * TILES + m * 16 + group
                scratch[base] = accum[m * 4]
                scratch[base + 8] = accum[m * 4 + 2]
                if lane_pair * 2 + 1 < 4:
                    scratch[base + 16 * TILES] = accum[m * 4 + 1]
                    scratch[base + 16 * TILES + 8] = accum[m * 4 + 3]


class Bucket:
    """The 2048-crossing bucket of a token from the three shards' group counts and fine
    bins.
    """

    @staticmethod
    @cute.jit
    def group_ranks(sums, first_group, step: cutlass.Constexpr, group):
        """Per shard s: warp totals of its group counts above / at ``group`` (lane's
        groups first_group + step*i in sums[4s + i]).
        """
        rank_above = cute.make_rmem_tensor(3, Uint32)
        rank_count = cute.make_rmem_tensor(3, Uint32)
        for s in cutlass.range_constexpr(3):
            part_above = Uint32(0)
            part_at = Uint32(0)
            for i in cutlass.range_constexpr(4):
                item_group = first_group + step * i
                part_above = part_above + sums[4 * s + i] * Uint32(item_group > group)
                part_at = part_at + sums[4 * s + i] * Uint32(item_group == group)
            rank_above[s] = redux_add_u32(part_above)
            rank_count[s] = redux_add_u32(part_at)
        return rank_above, rank_count

    @staticmethod
    @cute.jit
    def group_crossing(counts, lane):
        """Lane holds groups 127 - 4*lane - i in counts[3 - i]: the ballot of the lane
        holding the group where the descending total reaches 2048, and that lane's
        (group, keys above it, its count); callers broadcast them inside their own
        ``vote`` branch (a separate branch costs registers).
        """
        own = Int32(0)
        for item in cutlass.range_constexpr(4):
            own = own + Int32(counts[item])
        before = rt.Scan.up(own, lane) - own
        group = Int32(0)
        above = Int32(0)
        count = Int32(0)
        if before < 2048 and before + own >= 2048:
            running = before
            found = Int32(0)
            for item in cutlass.range_constexpr(4):
                item_count = Int32(counts[3 - item])
                running = running + item_count
                if found == 0 and running >= 2048:
                    group = 127 - lane * 4 - item
                    above = running - item_count
                    count = item_count
                    found = Int32(1)
        vote = ballot(before < 2048 and before + own >= 2048)
        return vote, group, above, count

    @staticmethod
    @cute.jit
    def fine_bin(bin_values, window, lane_bin, lane, ranks):
        """Fine-bin refinement of the 2048 crossing: lane holds bin ``bin`` (3 shard
        counts); returns (bin lane, prefix, above, end, shift, size) of the crossing
        bin and moves the per-shard ranks to it.
        """
        group_above, window_first, window_shift = window
        rank_above, rank_count = ranks
        count = Int32(0)
        for s in cutlass.range_constexpr(3):
            count = count + Int32(bin_values[s])
        before = group_above + rt.Scan.up(count, lane) - count
        bin_lane = ffs(ballot((before < 2048) & (before + count >= 2048))) - 1
        prefix = window_first + cute.arch.shuffle_sync(lane_bin, bin_lane)
        end = Int32(
            (window_shift != 20) & (cute.arch.shuffle_sync(lane_bin, bin_lane) == 4095)
        )
        for s in cutlass.range_constexpr(3):
            rank_above[s] = rank_above[s] + redux_add_u32(
                bin_values[s] * Uint32(bin_lane > lane)
            )
            rank_count[s] = cute.arch.shuffle_sync(bin_values[s], bin_lane)
        above = cute.arch.shuffle_sync(before, bin_lane)
        return (
            bin_lane,
            prefix,
            above,
            end,
            window_shift,
            cute.arch.shuffle_sync(count, bin_lane),
        )

    @staticmethod
    @cute.jit
    def write_bounds(dst, token, bucket, ranks, lane, SHARD: cutlass.Constexpr):
        """Lane 0: dst[token*8 ..] = (prefix, above, end, shift, size, first slot, own
        count, lower above).
        """
        rank_above, rank_count = ranks
        first_slot = Uint32(0)
        lower_above = Uint32(0)
        for s in cutlass.range_constexpr(SHARD):
            first_slot = first_slot + rank_count[s]
            lower_above = lower_above + rank_above[s]
        if lane == 0:
            words = (
                *bucket,
                Int32(first_slot),
                Int32(rank_count[SHARD]),
                Int32(lower_above),
            )
            for i in cutlass.range_constexpr(8):
                dst[token * 8 + i] = words[i]


class IndexScale:
    """The scale word of an index-key cache row (the key's FP8 bytes come first, then
    one 4-byte scale per key).
    """

    @staticmethod
    def word(exponent, packed: bool):
        """The cache word of the scale with biased exponent ``exponent``."""
        if cutlass.const_expr(packed):
            return exponent * Uint32(16843009)
        return exponent << 23

    @staticmethod
    def fp32(word, packed: bool):
        """The fp32 scale of cache word ``word``."""
        if cutlass.const_expr(packed):
            return bits_f32((as_u32(word) & Uint32(255)) << 23)
        return word


class KeyScore:
    """Index scores and their order-preserving u32 keys."""

    @staticmethod
    @cute.jit
    def key(bits):
        """Float bits -> order-preserving u32 key, -0 folded to +0 (keep this if-chain:
        PTX identity).
        """
        if bits == Uint32(2147483648):
            bits = Uint32(0)
        key = bits | Uint32(2147483648)
        if (bits & Uint32(2147483648)) != 0:
            key = bits ^ Uint32(4294967295)
        return key

    @staticmethod
    def weighted(logits, slot, head_weight, token, key_scale):
        """Index score of one key row: sum over 32 heads of relu(logit)*2 * weight (4
        interleaved fma chains), * 0.5 * scale.
        """
        chain = [Float32(0.0)] * 4
        for group in range(8):
            for lane4 in range(4):
                logit = logits[slot * 32 + group * 4 + lane4]
                chain[lane4] = ffma(
                    fadd(logit, rt.FastMath.abs_ftz(logit)),
                    head_weight[token * 32 + group * 4 + lane4],
                    chain[lane4],
                )
        return fmul(
            fmul(fadd(fadd(chain[0], chain[2]), fadd(chain[1], chain[3])), 0.5),
            key_scale,
        )


class QueryTile:
    """The swizzled FP8 indexer-query tile (UMMA B operand: [128 rows][128 B], 16-byte
    chunk c of row r at c ^ (r % 8)), its per-row scales and the head weights.
    """

    @staticmethod
    def pack_bf16x8(values):
        """Four bf16x2 words (cvt.rn) of eight floats."""
        return [rt.f32x2_to_bf16x2(values[2 * i], values[2 * i + 1]) for i in range(4)]

    @staticmethod
    def fp8_exponent(amax):
        """Biased exponent of the power-of-two FP8 scale covering ``amax`` (floored at
        1e-10) / 448, rounded up.
        """
        bits = as_u32(fmul(rt.max_noftz(amax, 1e-10), 0.002232142857142857))
        return (bits >> 23) + Uint32((bits & 8388607) != 0)

    @staticmethod
    def store_fp8(query_words_addr, row, chunk, values, exponent):
        """values[0:8] scaled by 2^(127 - exponent) as 8 E4M3 into the swizzled query
        tile (row, 8-byte chunk).
        """
        unit = bits_f32(254 - exponent << 23)
        scaled = [fmul(values[i], unit) for i in range(8)]
        packed = [rt.Convert.e4m3x4(*scaled[4 * h : 4 * h + 4]) for h in range(2)]
        swizzle = (cdiv(chunk, 2) ^ cmod(row, 8)) * 16 + cmod(chunk, 2) * 8
        rt.Memory.st_shared_v2(
            query_words_addr + row * 128 + swizzle, packed[0], packed[1]
        )

    @staticmethod
    @cute.jit
    def store_head_weight(head_weight, hoist_quad, tid):
        """Threads 224..255: the four bf16 head weights of quad tid-224 (prefetched in
        hoist_quad) to head_weight.
        """
        if tid - 224 >= 0:
            weights = rt.bf16x8_to_f32([hoist_quad[0], hoist_quad[1]])
            for i in cutlass.range_constexpr(4):
                head_weight[(tid - 224) * 4 + i] = weights[i]


@dataclass(frozen=True)
class Round:
    """A bucket round's constants for its window passes and keep-list stage."""

    launch: (
        tuple  # (position, epoch, direct_now, parity, (sb, shard_bases, near_bases))
    )
    lanes: tuple  # (half, key_warp, key_lane, lanes_below)
    bounds: tuple  # (bound_long, bound_digit, bound_shift) register tensors
    kept: cute.Tensor


class MlaSelector(rt.RoleKernel):
    """``glm53_mla_selector<SHARD>`` and its variants: 148 CTAs x 384 threads, PDL."""

    Params = PagedParams
    GRID = CTAS
    SMEM_BYTES = SMEM_BYTES
    ROLES = (rt.Role("compute", 0, 7, 208, True), rt.Role("load", 8, 11, 88, False))
    SHARD = 0
    #: The index-key cache holds packed UE8M0 scales (``IndexScale``).
    PACKED_SCALE = False
    #: CTAs below this run the X projections (vLLM's BF16 wk rows on 70-77).
    FRONT = 78
    REDUCE_CTAS = 0
    REDUCE_ROWS = 4

    def rows(self, p, n, token):
        """Trace time: ``n`` rows of ``token``."""
        return n

    def emit_id(self, p, bases, epoch, token, mailbox, slot, index):
        """Trace time: the id of shard key ``index`` to SelectedIds[token][slot] and as
        a tagged record to ``mailbox + slot`` of every base.
        """
        p.SelectedIds[token * TOPK + slot] = Keys.shard_id(index, self.SHARD)
        IdMailbox.publish(
            bases,
            mailbox + slot,
            IdMailbox.word(epoch, Keys.shard_id(index, self.SHARD)),
        )

    def smem(self, base):
        """The :class:`SelectorSmem` record over ``base`` (``taddr`` is set by the
        prologue).
        """
        return SelectorSmem.record(
            base,
            ring="pages",
            order=SelectorSmem.ORDER,
            to_end=True,
            typed=True,
            taddr=None,
        )

    def mbarriers(self) -> tuple:
        """Every mbarrier region of :class:`SelectorSmem`, in address order."""
        return SelectorSmem.barriers()

    @cute.jit
    def prologue(self, p):
        """Thread ids, smem record, mbarrier init, then the 512-column TMEM allocation
        (warp 0).
        """
        tid, bid, warp, lane = rt.Prologue.thread_ids()
        _, base = rt.Prologue.smem()
        s = self.smem(base)
        with rt.warp_leader(warp, 0):
            rt.Prologue.init_mbarriers(base, self.mbarriers())
        cute.arch.sync_warp()
        tmem_hold = rt.Memory.smem_ptr(base, SelectorSmem.TMEM_HOLD, Int32)
        if warp == 0:
            cute.arch.alloc_tmem(512, tmem_hold)
            cute.arch.sync_warp()
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        rt.TensorCore.fence_after_thread_sync()
        taddr = cute.make_tensor(tmem_hold, cute.make_layout(1))[0]
        return rt.ThreadIds(tid, bid, warp, lane), dataclasses.replace(s, taddr=taddr)

    @cute.jit
    def run(self, p, s, ids: rt.ThreadIds) -> None:
        """Dispatch the roles with the params record's flat views; the load test reads
        the compute role's warp.
        """
        views = p.views()
        warp = ids.warp
        if warp <= self.ROLES[0].last:
            warp = self.compute(views, s, ids)
        if (warp >= self.ROLES[1].first) & (warp <= self.ROLES[1].last):
            self.load(views, s, ids)

    @cute.jit
    def compute(self, p, s, ids: rt.ThreadIds):
        """Compute warps 0-7: the X RMSNorm and projections, q_b / key producers,
        scoring, selection, reduce.
        """
        tid, bid = ids.tid, ids.bid
        warp = ids.warp
        with rt.warp_leader(warp, 0):
            cute.arch.griddepcontrol_wait()
            if bid < self.FRONT:
                rt.mbar_expect_tx(s.x_full, 49152)
                for r3 in cutlass.range_constexpr(4):
                    rt.bulk_g2s(
                        s.act_addr + 12304 * r3, ga(p.X, 6144 * r3), 12288, s.x_full
                    )
        self.set_regs("compute")
        warp = rt.Sync.warp_uniform(cdiv(tid, 32))
        cute.arch.griddepcontrol_wait()
        position = p.CurPos[0]
        row_id = bid * 256 + tid
        if (
            (row_id >= 0)
            & (row_id < 1024)
            & (row_id < cdiv((position + 4 + 64 - 1 >> 6) + 32 - 1, 32))
        ):
            address = ga(p.BlockTable, row_id * 32)
            l2_line = cute.make_ptr(
                Uint8, Int64(address), cute.AddressSpace.gmem
            ).to_llvm_ptr()
            cute.arch.prefetch(l2_line, cache_level="L2")
        long_context = Int32(position + 4 > 2048)
        direct_now = Int32(position + 4 > 65536) & Int32(position + 4 <= 884736)
        seed_words = cute.make_rmem_tensor(4, Uint32)
        sw = ld_cg_v4(ga(p.SelectState, StateWords.SEEDS))
        for i in cutlass.range_constexpr(4):
            seed_words[i] = sw[i]
        epoch = self.x_projections(p, s, ids, warp)
        if bid < 128:
            self.query_projection(p, s, ids, warp, long_context)
        if (bid >= 128) & (bid < 132):
            self.key_producer(p, s, ids, warp)
        if bid < 128:
            with rt.warp_leader(warp, 0):
                rt.Memory.atomic_add_u32(p.Flags, 592 + bid, 1)
        self.score_stage(
            p, s, ids, warp, position, long_context, epoch, direct_now, seed_words
        )
        self.final_stage(p, s, ids, warp, position, epoch, direct_now)
        return warp

    @cute.jit
    def x_projections(self, p, s, ids: rt.ThreadIds, warp):
        """Bid < FRONT: RMSNorm of the four X rows into ``act``, then the QA (bid < 64,
        FP8 pages), HeadWeights (bid 68, 69) or RawKey (bid 70-77) projection from
        BF16 rows and its Flags count; returns the launch epoch.
        """
        tid, bid, lane = ids.tid, ids.bid, ids.lane
        epoch = Uint32(0)
        if bid < self.FRONT:
            gamma = [
                x
                for j in range(3)
                for x in rt.Memory.ld_global_v8_f32(ga(p.GammaX, 2048 * j + tid * 8))
            ]
            mbar_wait(s.x_full, 0)
            cute.arch.fence_proxy("async.shared", space="cta")
            values = cute.make_rmem_tensor(96, Float32)
            for r in cutlass.range_constexpr(12):
                for w in cutlass.range_constexpr(4):
                    values[r * 8 + 2 * w], values[r * 8 + 2 * w + 1] = (
                        rt.Convert.bf16x2_to_f32x2(
                            s.act_words[
                                cdiv((r // 3) * 6152 + (r % 3) * 2048 + tid * 8, 2) + w
                            ]
                        )
                    )
            Projection.rms_rows(
                values, gamma, s.scratch, s.act_addr, tid, warp, lane, 3, 6152
            )
            rt.bar_sync(8, 256)
            rt.bar_sync(11, 384)
            epoch = Uint32(s.epoch_slot[0])
            if bid < 64:
                accum = rmem_fill(8, Float32, 0)
                Projection.vin_proj_pages(
                    accum,
                    0,
                    6,
                    s.ring_addr,
                    s.ring_f32,
                    s.act_addr,
                    warp,
                    lane,
                    s.pages,
                    3076,
                    8192,
                )
                Projection.mma_tiles_to_scratch(s.scratch, accum, warp, lane, 2, 129)
                rt.bar_sync(8, 256)
                if tid < 128:
                    value = Projection.col_sum(s.scratch, tid, 129)
                    if bid < 64:
                        qa = p.QA
                        qa[cdiv(tid, 32) * 2048 + bid * 32 + cmod(tid, 32)] = value.to(
                            BFloat16
                        )
                    else:
                        raw_key = p.RawKey
                        raw_key[
                            cdiv(tid, 32) * 128 + (bid - 64) * 32 + cmod(tid, 32)
                        ] = value.to(BFloat16)
                rt.bar_sync(8, 256)
                rt.bar_sync(8, 256)
                with rt.warp_leader(warp, 0):
                    rt.Memory.fence_release_gpu()
                    rt.Memory.atomic_add_u32(p.Flags, bid, 1)
            elif bid >= 68:
                self.vlin_bf16_rows(p, s, tid, bid, warp, lane)
        else:
            rt.bar_sync(11, 384)
            epoch = Uint32(s.epoch_slot[0])
        return epoch

    @cute.jit
    def vlin_bf16_rows(self, p, s, tid, bid, warp, lane):
        """Bid 68-77: 16 rows of vLLM's BF16 ``wk_weights_proj`` (68, 69: HeadWeights;
        70-77: RawKey), six K1024 pages (warp w: K planes 2w, 2w + 1, in K order),
        then the Flags release.
        """
        accum = rmem_fill(4, Float32, 0)
        for page in range(6):
            s.pages.wait_full(page)
            cute.arch.fence_proxy("async.shared", space="cta")
            for half in cutlass.range_constexpr(2):
                Projection.bf16_plane_mma(
                    accum,
                    cmod(page, 4) * 36864 + (2 * warp + half) * 2048,
                    s.ring_words,
                    s.act_words,
                    lane,
                    page * 1024 + warp * 128 + half * 64,
                    1,
                    3076,
                )
            s.pages.release(cmod(page, 4))
        Projection.mma_tiles_to_scratch(s.scratch, accum, warp, lane, 1, 65)
        rt.bar_sync(8, 256)
        if tid < 64:
            value = Projection.col_sum(s.scratch, tid, 65).to(BFloat16)
            if bid < 70:
                head_weights = p.HeadWeights
                head_weights[cdiv(tid, 16) * 32 + (bid - 68) * 16 + cmod(tid, 16)] = (
                    value
                )
            else:
                raw_key = p.RawKey
                raw_key[cdiv(tid, 16) * 128 + (bid - 70) * 16 + cmod(tid, 16)] = value
        rt.bar_sync(8, 256)
        rt.bar_sync(8, 256)
        with rt.warp_leader(warp, 0):
            if bid < 70:
                rt.Memory.red_async_release_gpu_add(ga(p.Flags, 148 + bid), 1)
            else:
                rt.Memory.fence_release_gpu()
                rt.Memory.atomic_add_u32(p.Flags, bid, 1)

    @cute.jit
    def query_projection(self, p, s, ids: rt.ThreadIds, warp, long_context):
        """Bid < 128: (long context) RMSNorm of the four QA rows and the q_b projection
        (RoPE + per-quarter amax in the full kernel), then the Flags[444 + bid]
        release.
        """
        tid, bid, lane = ids.tid, ids.bid, ids.lane
        if long_context != 0:
            rt.bar_sync(11, 384)
            wqi_first = self.front_pages(bid)
            raw = [
                rt.Memory.ld_global_v4(ga(p.QA, 2048 * q + tid * 8)) for q in range(4)
            ]
            gamma_q = [
                bits_f32(x) for x in rt.Memory.ld_global_v8(ga(p.GammaQ, tid * 8))
            ]
            q_values = cute.make_rmem_tensor(32, Float32)
            for rk in cutlass.range_constexpr(4):
                for _pair in cutlass.range_constexpr(4):
                    q_values[8 * rk + 2 * _pair], q_values[8 * rk + 2 * _pair + 1] = (
                        rt.Convert.bf16x2_to_f32x2(raw[rk][_pair])
                    )
            Projection.rms_rows(
                q_values, gamma_q, s.scratch, s.act_addr, tid, warp, lane, 1, 2056
            )
            rt.bar_sync(8, 256)
            q_accum = cute.make_rmem_tensor(8, Float32)
            producer_dim = cmod(cmod(bid, 4) * 32 + cmod(tid, 32), 64)
            producer_freq = [
                bits_f32(x)
                for x in ld_cg_v2(
                    ga(
                        p.Freq,
                        cdiv(cmod(tid, 128), 32) * 64 + cdiv(producer_dim, 2) * 2,
                    )
                )
            ]
            for qi in cutlass.range_constexpr(8):
                q_accum[qi] = Float32(0.0)
            Projection.vin_proj_pages(
                q_accum,
                wqi_first,
                2,
                s.ring_addr,
                s.ring_f32,
                s.act_addr,
                warp,
                lane,
                s.pages,
                1028,
            )
            Projection.mma_tiles_to_scratch(s.scratch, q_accum, warp, lane, 2, 129)
            rt.bar_sync(8, 256)
            slot = cmod(tid, 128)
            q_value = Projection.col_sum(s.scratch, slot, 129)
            token = cdiv(slot, 32)
            row = cmod(slot, 32)
            own, _ = rt.Convert.bf16x2_to_f32x2(rt.f32x2_to_bf16x2(q_value, 0.0))
            out = own
            partner = cute.arch.shuffle_sync_bfly(own, 1)
            if cmod(bid, 4) < 2:
                pair_even = own
                pair_odd = partner
                if cmod(row, 2) != 0:
                    pair_even = partner
                    pair_odd = own
                rot0 = ffma(
                    pair_even, producer_freq[0], -(fmul(pair_odd, producer_freq[1]))
                )
                rot1 = ffma(
                    pair_even, producer_freq[1], fmul(pair_odd, producer_freq[0])
                )
                rot_lo, rot_hi = rt.Convert.bf16x2_to_f32x2(
                    rt.f32x2_to_bf16x2(rot0, rot1)
                )
                out = rot_lo
                if cmod(row, 2) != 0:
                    out = rot_hi
            if tid < 128:
                raw_query = p.RawQuery
                raw_query[token * 4096 + bid * 32 + row] = rt.Convert.to_bf16(out)
            quarter_bits = redux_max_u32(as_u32(rt.FastMath.abs_ftz(out)))
            if tid < 128:  # noqa: SIM102
                if row == 0:
                    scores = p.Scores
                    scores[(cmod(self.SHARD + 1, 3) + 3 * token) * 128 + bid] = (
                        bits_f32(quarter_bits)
                    )
            rt.bar_sync(8, 256)
        rt.bar_sync(8, 256)
        with rt.warp_leader(warp, 0):
            rt.Memory.red_async_release_gpu_add(ga(p.Flags, 444 + bid), 1)

    @cute.jit
    def key_producer(self, p, s, ids: rt.ThreadIds, warp):
        """Bid 128..131 (one per new token), warp 0: LayerNorm/affine, RoPE and E4M3 of
        the new index key row into the paged key cache, then the Flags[296 + bid]
        release.
        """
        bid, lane = ids.bid, ids.lane
        rt.bar_sync(11, 384)
        if warp == 0:
            lane_0 = cmod(lane, 16)
            kvalue = rmem_fill(8, Float32, 0)
            if lane < 16:
                kw = rt.Memory.ld_global_v4(
                    ga(p.RawKey, (bid - 128) * 128 + lane_0 * 8)
                )
                for kp in cutlass.range_constexpr(4):
                    kvalue[2 * kp], kvalue[2 * kp + 1] = rt.Convert.bf16x2_to_f32x2(
                        kw[kp]
                    )
            ktotal = Float32(0.0)
            for kitem in cutlass.range_constexpr(8):
                ktotal = fadd(ktotal, kvalue[kitem])
            for step in cutlass.range_constexpr(4):
                ktotal = fadd(ktotal, cute.arch.shuffle_sync_bfly(ktotal, 1 << step))
            centered = [ffma(ktotal, -0.0078125, kvalue[kitem]) for kitem in range(8)]
            ksq = Float32(0.0)
            for kitem in cutlass.range_constexpr(8):
                ksq = ffma(centered[kitem], centered[kitem], ksq)
            for step in cutlass.range_constexpr(4):
                ksq = fadd(ksq, cute.arch.shuffle_sync_bfly(ksq, 1 << step))
            kinv = rt.FastMath.rsqrt(ffma(ksq, 0.0078125, 1e-06))
            key_gamma = [
                bits_f32(x) for x in rt.Memory.ld_global_v8(ga(p.KeyGamma, lane_0 * 8))
            ]
            key_bias = [
                bits_f32(x) for x in rt.Memory.ld_global_v8(ga(p.KeyBias, lane_0 * 8))
            ]
            affine = cute.make_rmem_tensor(8, Float32)
            for kitem in cutlass.range_constexpr(8):
                affine[kitem] = ffma(
                    key_gamma[kitem], fmul(centered[kitem], kinv), key_bias[kitem]
                )
            for kitem in cutlass.range_constexpr(8):
                affine[kitem] = rt.Convert.bf16_round(affine[kitem])
            lane_1 = cmod(lane, 16)
            if lane < 8:
                for pair in cutlass.range_constexpr(4):
                    angle = (bid - 128) * 64 + (lane_1 * 4 + pair) * 2
                    cosine = p.Freq[angle]
                    sine = p.Freq[angle + 1]
                    even = affine[2 * pair]
                    odd = affine[2 * pair + 1]
                    affine[2 * pair] = ffma(even, cosine, -(fmul(odd, sine)))
                    affine[2 * pair + 1] = ffma(even, sine, fmul(odd, cosine))
            for kitem in cutlass.range_constexpr(8):
                affine[kitem] = rt.Convert.bf16_round(affine[kitem])
            amax = Float32(0.0)
            for kitem in cutlass.range_constexpr(8):
                amax = rt.max_noftz(amax, rt.FastMath.abs_ftz(affine[kitem]))
            for step in cutlass.range_constexpr(4):
                amax = rt.max_noftz(amax, cute.arch.shuffle_sync_bfly(amax, 1 << step))
            bits = as_u32(rt.FastMath.div_rn(rt.max_noftz(amax, 0.0001), 448.0))
            exponent = (bits >> 23) + Uint32((bits & 8388607) != 0)
            unit = bits_f32(Uint32(254) - exponent << 23)
            scaled = [fmul(affine[kitem], unit) for kitem in range(8)]
            key_fp8 = [rt.Convert.e4m3x4(*scaled[4 * h : 4 * h + 4]) for h in range(2)]
            krow = p.CurPos[0] + (bid - 128)
            page_rows = Int64(p.BlockTable[krow >> 6]) * 4224
            if lane < 16:
                dst = ga(
                    p.IndexKeyCache, page_rows + Int64(cmod(krow, 64) * 64 + lane_0 * 4)
                )
                rt.Memory.st_global_b32(dst, key_fp8[0])
                rt.Memory.st_global_b32(dst + 4, key_fp8[1])
            if lane == 0:
                rt.Memory.st_global_b32(
                    ga(p.IndexKeyCache, page_rows + Int64(cmod(krow, 64) * 2 + 4096)),
                    IndexScale.word(exponent, self.PACKED_SCALE),
                )
        rt.bar_sync(8, 256)
        with rt.warp_leader(warp, 0):
            rt.Memory.red_async_release_gpu_add(ga(p.Flags, 296 + bid), 1)

    @cute.jit
    def load(self, p, s, ids: rt.ThreadIds):
        """Load warps 8-11: weight ring pages, the epoch ticket, the window-score key
        tiles and their tcgen05 MMAs, the shard-mailbox polls and bucket bounds, the
        emission records and the reduce receive.
        """
        tid, bid = ids.tid, ids.bid
        self.set_regs("load")
        warp = rt.Sync.warp_uniform(cdiv(tid, 32))
        with rt.warp_leader(warp, 8):
            for name in ("InputTmap", "WisTmap", "WqbTmap"):
                rt.AsyncCopy.prefetch_tensormap(self.tmaps[name])
        cute.arch.griddepcontrol_wait()
        qaki_policy = rt.AsyncCopy.createpolicy_evict_first(
            ga(p.QakiWeight, 0), 13421568, 13421568
        )
        wis_policy = rt.AsyncCopy.createpolicy_evict_first(
            ga(p.WisWeight, 0), 393216, 393216
        )
        wqb_policy = rt.AsyncCopy.createpolicy_evict_first(
            ga(p.WqbWeight, 0), 8421376, 8421376
        )
        front = self.front_pages(bid)
        if front != 0:
            for page in range(0, 4):
                self.ring_page(p, s, page, bid, warp, qaki_policy, wis_policy)
        position = p.CurPos[0]
        long_context = Int32(position + 4 > 2048)
        direct_now = Int32(position + 4 > 65536) & Int32(position + 4 <= 884736)
        self.epoch_ticket(warp, p.FullEpoch.iterator, s.epoch_slot)
        rt.bar_sync(11, 384)
        epoch = s.epoch_slot[0]
        self.prefetch_residual(p, s, ids)
        if front != 0:
            for page in range(4, 6):
                self.ring_page(p, s, page, bid, warp, qaki_policy, wis_policy)
        if (long_context != 0) & (bid < 128):
            wqi_first = front
            for page in range(0, 2):
                slot = cmod(wqi_first + page, 4)
                with rt.warp_leader(warp, 8):
                    s.pages.acquire(wqi_first + page, 32800, slot)
                    rt.AsyncCopy.tma_3d(
                        s.ring_addr + slot * 36864,
                        self.tmaps["WqbTmap"],
                        Uint32(0),
                        Uint32(bid * 32),
                        Uint32(page * 8),
                        s.pages.full(slot),
                        Uint64(L2_HINT),
                    )
                    rt.bulk_g2s_hint(
                        s.ring_addr + slot * 36864 + 32768,
                        ga(
                            p.WqbWeight,
                            WQB_SCALES + (cdiv(bid, 4) * 16 + page * 8) * 4,
                        ),
                        32,
                        s.pages.full(slot),
                        wqb_policy,
                    )
            self.poll_flags(p, tid - 256, 64, 0, epoch)
            rt.bar_sync(11, 384)
        if (bid >= 128) & (bid < 132):
            self.poll_flags(p, tid - 256, 8, 70, epoch)
            rt.bar_sync(11, 384)
        if (long_context != 0) & (bid < 144):
            self.load_selection(
                p, s, ids, warp, position, epoch, long_context, direct_now
            )
        if (bid >= 144) & (bid < 148):  # noqa: SIM102
            if self.rows(p, position + (bid - 144) + 1, bid - 144) > 2048:
                self.stage_refine_words(p, s, ids, warp, position, epoch)
        self.receive_partials(p, s, ids, epoch)

    @cute.jit
    def load_selection(
        self, p, s, ids: rt.ThreadIds, warp, position, epoch, long_context, direct_now
    ):
        """Long context, bid < 144: the producer flags, the window-score key tiles, the
        direct-mode bucket bounds and the emission records.
        """
        tid, bid = ids.tid, ids.bid
        self.poll_flags(p, tid - 256, 2, 216, epoch)
        self.poll_flags(p, tid - 256, 4, 424, epoch)
        rt.bar_sync(11, 384)
        self.poll_flags(p, tid - 256, 128, 444, epoch)
        rt.bar_sync(11, 384)
        self.window_key_tiles(p, s, ids, warp, position)
        mbar_wait(s.ws_scored, 0)
        self.last_cta_push(p, s, ids, warp, epoch, direct_now)
        self.poll_ctas(p, tid - 256, 740, epoch)
        if (long_context == 0) | (direct_now == 0):
            rt.bar_sync(11, 384)
        if (long_context != 0) & (direct_now != 0):
            if warp < 12:
                self.direct_bounds(p, s, ids, warp, epoch)
            rt.bar_sync(11, 384)
        emit_gate = Int32(1)
        rt.bar_sync(11, 384)
        emit_gate = Int32(s.near_meta[61] == 0)
        if emit_gate != 0:
            for item_index in range(tid - 256, 236, 128):
                token = cdiv(item_index, 59)
                item = cmod(item_index, 59)
                live = self.rows(p, position + token + 1, token)
                rounds = Keys.shard_rounds(live, bid, self.SHARD)
                if (live > 2048) & (rounds > 0) & (item < 3):
                    record = StateWords.cta_record(token, bid, item)
                    cta_word = rt.Poll.word(ga(p.SelectState, record), epoch + 1, False)
                    emission = s.emission
                    emission[item_index] = Uint32(cta_word)
            rt.bar_sync(11, 384)

    @cute.jit
    def window_key_tiles(self, p, s, ids: rt.ThreadIds, warp, position):
        """Warps 8-10 stage this CTA's 128-key FP8 tiles round by round (4 slots, 2 when
        not streaming); warp 11 issues one tcgen05 MMA per tile against the query
        tile into TMEM columns 128 x slot.
        """
        bid, lane = ids.bid, ids.lane
        ws_live = Keys.shard_live(position + 4, self.SHARD)
        ws_count = Keys.rounds_below(ws_live, bid)
        ws_streaming = 1 - Int32(ws_live <= 73728)
        if warp < 11:
            for ws_round in range(warp - 7, ws_count, 3):
                if ws_round >= 2:
                    mbar_wait(s.ws_query, 0)
                produce_slot = (ws_round & 1 + 2 * ws_streaming) + 2 * (
                    1 - ws_streaming
                )
                x = bid * 128 + ws_round * 18432
                rj = (cdiv(x, 128) * 3 + self.SHARD) * 128 + cmod(x, 128)
                tile_rows0 = p.BlockTable[rj >> 6] * 66 - (rj >> 6 << 6)
                tile_rows1 = p.BlockTable[(rj >> 6) + 1] * 66 - ((rj >> 6) + 1 << 6)
                if ws_round >= 2 + 2 * ws_streaming:
                    use = ws_round >> 1 + ws_streaming
                    mbar_wait(s.ws_free + produce_slot * 8, use - 1 & 1)
                produce_block = bid * 128 + ws_round * 18432
                for slot in cutlass.range_constexpr(4):
                    if produce_slot == slot:
                        self.produce_key_tile(
                            p,
                            s,
                            slot,
                            produce_block,
                            ws_live,
                            tile_rows0,
                            tile_rows1,
                            lane,
                        )
        with rt.warp_leader(warp, 11):
            mbar_wait(s.ws_query, 0)
            for mma_round in range(0, ws_count):
                issue_slot = (mma_round & 1 + 2 * ws_streaming) + 2 * (1 - ws_streaming)
                issue_use = mma_round >> 1 + ws_streaming
                mbar_wait(s.ws_full + issue_slot * 8, issue_use & 1)
                if mma_round >= 2 + 2 * ws_streaming:
                    mbar_wait(s.ws_free + issue_slot * 8, issue_use - 1 & 1)
                cute.arch.fence_proxy("async.shared", space="cta")
                rt.TensorCore.fence_after_thread_sync()
                b_lo = (s.query_words_addr >> 4) & 0x3FFF
                tiles = (
                    s.umma_keys8_next_addr,
                    s.ws_tile1_addr,
                    s.umma_keys_addr,
                    s.ws_tile3_addr,
                )
                for slot in cutlass.range_constexpr(4):
                    if issue_slot == slot:
                        rt.TensorCore.umma_f8f6f4_k4(
                            (tiles[slot] >> 4) & 0x3FFF,
                            b_lo,
                            s.taddr + 128 * slot,
                            0,
                        )
                rt.TensorCore.tcgen05_commit(s.ws_done + issue_slot * 8)
                rt.TensorCore.tcgen05_commit(s.ws_empty + issue_slot * 8)

    @cute.jit
    def produce_key_tile(
        self, p, s, slot: cutlass.Constexpr, produce_block, ws_live, rows0, rows1, lane
    ):
        """Window-score key tile for ``slot``: 128 scale words + 32x16-B swizzled key
        rows (predicated cp.async).
        """
        ws_base = s.umma_keys_addr + (slot - 2) * 16384
        first = (cdiv(produce_block, 128) * 3 + self.SHARD) * 128 + cmod(
            produce_block, 128
        )
        row4 = first + lane * 4
        rows = rows0
        if (row4 & 64) != 0:
            rows = rows1
        rt.AsyncCopy.cp_async_cg16_pred(
            Int32(ws_live > produce_block + lane * 4),
            s.key_scales_addr + (slot * 128 + lane * 4) * 4,
            ga(
                p.IndexKeyCache,
                Int64(rows + row4 - (row4 & 63)) * 64 + (4096 + (row4 & 63) * 2),
            ),
        )
        for part in cutlass.range(0, 32):
            chunk = lane + part * 32
            row = cdiv(chunk, 8)
            q = first + row
            rows_q = rows0
            if (q & 64) != 0:
                rows_q = rows1
            rt.AsyncCopy.cp_async_cg16_pred_hint(
                Int32(ws_live > produce_block + row),
                ws_base + row * 128 + (cmod(chunk, 8) ^ cmod(row, 8)) * 16,
                ga(
                    p.IndexKeyCache,
                    Int64(cmod(chunk, 8) * 8) + Int64(Uint32(rows_q + q)) * 64,
                ),
                Uint64(0x12F0000000000000),
            )
        rt.AsyncCopy.cp_async_mbarrier_arrive_noinc(s.ws_full + slot * 8)

    @cute.jit
    def front_pages(self, bid):
        """The CTA's X-projection pages (six, or none), which is also the ring sequence
        of its first q_b page.
        """
        pages = Int32(0)
        if (bid < 64) | ((bid >= 68) & (bid < self.FRONT)):
            pages = Int32(6)
        return pages

    @cute.jit
    def ring_page(self, p, s, page, bid, warp, qaki_policy, wis_policy):
        """One q_a (bid < 64) or ``wk_weights_proj`` (bid 68-77) weight page into ring
        slot page % 4 (warp 8, elected).
        """
        slot = cmod(page, 4)
        with rt.warp_leader(warp, 8):
            if bid < 64:
                s.pages.acquire(
                    page, Uint32(32768) + Uint32(192) * Uint32(page == 0), slot
                )
                self.tma_page(s, slot, "InputTmap", bid * 32, page * 8)
                if page == 0:
                    rt.bulk_g2s_hint(
                        s.ring_addr + 32768,
                        ga(p.QakiWeight, QA_SCALES + cdiv(bid, 4) * 48 * 4),
                        192,
                        s.pages.full(slot),
                        qaki_policy,
                    )
            else:
                row = (bid - 70) * 16
                if bid < 70:
                    row = 128 + (bid - 68) * 16
                s.pages.acquire(page, 32768, slot)
                self.tma_page(s, slot, "WisTmap", row, page * 16)

    def tma_page(self, s, slot, tmap: str, row, plane):
        """One page box of a vLLM-weight descriptor (``row``, K ``plane``) into ring
        slot ``slot``.
        """
        rt.AsyncCopy.tma_3d(
            s.ring_addr + slot * 36864,
            self.tmaps[tmap],
            Uint32(0),
            Uint32(row),
            Uint32(plane),
            s.pages.full(slot),
            Uint64(L2_HINT),
        )

    @staticmethod
    @cute.jit
    def poll_flags(p, lane, n: cutlass.Constexpr, base, epoch):
        """Lanes < n spin (ld.relaxed.gpu) until Flags[base + lane] >= epoch + 1; then
        acquire + bar 9.
        """
        owned = Int32(lane < n)
        index = base + cutlass.min(lane, n - 1)
        done = Int32(0)
        while done == 0:
            satisfied = Int32(1)
            if owned != 0:
                observed = rt.Memory.ld_relaxed_gpu(ga(p.Flags, index))
                satisfied = Int32(observed >= epoch + 1)
            done = Int32(cute.arch.vote_all_sync(satisfied != 0))
        rt.Memory.fence_acquire_gpu()
        rt.bar_sync(9, 128)

    @staticmethod
    @cute.jit
    def poll_ctas(p, lane, base, epoch):
        """The 144-CTA arrival poll on Flags[base ..] (two 128-lane parts), then
        fence.acquire / bar 9 / fence.acquire.
        """
        done = Int32(0)
        while done == 0:
            satisfied = Int32(1)
            for part in cutlass.range_constexpr(2):
                if lane + part * 128 < 144:
                    observed = rt.Memory.ld_relaxed_gpu(
                        ga(p.Flags, base + lane + part * 128)
                    )
                    satisfied = satisfied & Int32(observed >= epoch + 1)
            done = Int32(cute.arch.vote_all_sync(satisfied != 0))
        rt.Memory.fence_acquire_gpu()
        rt.bar_sync(9, 128)
        rt.Memory.fence_acquire_gpu()

    @cute.jit
    def last_cta_push(self, p, s, ids: rt.ThreadIds, warp, epoch, direct_now):
        """Direct mode, last CTA: push the coarse group sums and the predicted crossing
        bin to the partner shards.
        """
        lane = ids.lane
        if direct_now != 0:
            rt.bar_sync(11, 384)
            with rt.warp_leader(warp, 8):
                rt.Memory.fence_release_gpu()
                last_old = rt.Memory.atom_add_acq_rel_gpu(ga(p.Flags, 884), 1)
                near_words = s.near_words
                near_words[124] = Uint32(last_old == 144 * epoch + 144 - 1)
            rt.bar_sync(9, 128)
            if s.near_words[124] != 0:  # noqa: SIM102
                if warp < 12:
                    token = warp - 8
                    generation = s.epoch_slot[0] + 1
                    parity = Int32(cmod(epoch, 2))
                    sums = cute.make_rmem_tensor(4, Uint32)
                    self.group_sums(p, sums, 0, parity, token, lane)
                    self.push_group_sums(p, sums, 0, parity, token, lane, generation)
                    word = self.predicted_bin(sums, 0, lane)
                    if word >= 0:
                        pair = ld_cg_v2(
                            ga(p.SelectState, StateWords.bin(parity, token, word))
                        )
                        self.sel_push(
                            p, parity, token, word, rt.Poll.bin_record(generation, pair)
                        )

    def group_sums(self, p, sums, base: int, parity, token, lane):
        """Trace time: this shard's coarse counts of groups 127 - lane - 32*i (i < 4)
        into sums[base + i].
        """
        for item in range(4):
            group = StateWords.group(parity, token, 127 - lane - 32 * item)
            sums[base + item] = ld_cg_b32(ga(p.SelectState, group))

    def push_group_sums(self, p, sums, base: int, parity, token, lane, generation):
        """Trace time: :meth:`group_sums` as tagged records to the partner shards' group
        words.
        """
        for item in range(4):
            record = Uint64(generation) << 32 | Uint64(sums[base + item])
            self.sel_push(
                p,
                parity,
                token,
                BucketMailbox.GROUP_WORDS + 127 - lane - 32 * item,
                record,
            )

    @cute.jit
    def predicted_bin(self, sums, base: cutlass.Constexpr, lane):
        """The lane's fine bin of the group where sums[base..base+4) cross 683 (a third
        of 2048), or -1.
        """
        before, found, item, value = self.sel_band(sums, base, 683)
        inclusive = rt.Scan.up(value, lane)
        crossing = (before + inclusive - value < 683) & (before + inclusive >= 683)
        vote = ballot((found != 0) & crossing)
        word = Int32(-1)
        if vote != 0:
            word = (
                cute.arch.shuffle_sync(127 - lane - 32 * item, ffs(vote) - 1) * 32
                + 32
                - 1
                - lane
            )
        return word

    def shard_bins(self, p, base, parity, token, n, generation, values, uppers):
        """Trace time: fine bin ``n`` of ``token`` per shard into ``values`` /
        ``uppers`` (count, upper half): this shard's from SelectState, the two
        others' polled from mailbox ``base``.
        """
        own = ld_cg_v2(ga(p.SelectState, StateWords.bin(parity, token, n)))
        values[self.SHARD], uppers[self.SHARD] = own[0], own[1]
        others = [o for o in range(3) if o != self.SHARD]
        addrs = [SelectionMailbox.word_addr(base, parity, o, token, n) for o in others]
        words = rt.Poll.wait(addrs, generation, tag40=True)
        for i, o in enumerate(others):
            values[o] = Uint32(words[i]) & 1048575
            uppers[o] = Uint32(words[i] >> 20) & 1048575

    @cute.jit
    def direct_bounds(self, p, s, ids: rt.ThreadIds, warp, epoch):
        """Direct mode, warp 8 + token: this token's 2048 crossing from the three
        shards' group counts (and, in the fine window, their fine bins), written as
        bucket bounds to ``loader_bounds[token]``.
        """
        bid, lane = ids.bid, ids.lane
        token = warp - 8
        if lane == 0:
            loader_bounds = s.loader_bounds
            loader_bounds[token * 8] = s.coarse_scan[token]
            loader_bounds[token * 8 + 1] = 0
            loader_bounds[token * 8 + 2] = 1
            loader_bounds[token * 8 + 3] = s.coarse_scan[4 + token]
            for loader_word in cutlass.range_constexpr(4, 8):
                loader_bounds[token * 8 + loader_word] = 0
        generation = s.epoch_slot[0] + 1
        parity = Int32(cmod(epoch, 2))
        first_group = 127 - lane
        own = 4 * self.SHARD
        sums = cute.make_rmem_tensor(12, Uint32)
        self.group_sums(p, sums, own, parity, token, lane)
        if bid < 128:
            cert = ld_cg_v2(
                ga(p.SelectState, StateWords.bin(parity, token, bid * 32 + lane))
            )
            self.sel_push(
                p, parity, token, bid * 32 + lane, rt.Poll.bin_record(generation, cert)
            )
        if bid == 0:
            self.push_group_sums(p, sums, own, parity, token, lane, generation)
        pred_word = self.predicted_bin(sums, own, lane)
        pred_bins = rmem_fill(3, Uint32, 0)
        pred_uppers = rmem_fill(3, Uint32, 0)
        if pred_word >= 0:
            pair = ld_cg_v2(ga(p.SelectState, StateWords.bin(parity, token, pred_word)))
            pred_bins[self.SHARD] = pair[0]
            pred_uppers[self.SHARD] = pair[1]
        others = [o for o in range(3) if o != self.SHARD]
        src = p.shard_base(self.SHARD)
        words = rt.Poll.wait(
            [
                SelectionMailbox.word_addr(
                    src, parity, o, token, 4096 + first_group - 32 * item
                )
                for o in others
                for item in range(4)
            ]
            + [
                SelectionMailbox.word_addr(src, parity, o, token, pred_word)
                for o in others
            ],
            generation,
            live=(None,) * 8 + (pred_word >= 0,) * 2,
            tag40=(False,) * 8 + (True,) * 2,
        )
        for o in cutlass.range_constexpr(2):
            for item in cutlass.range_constexpr(4):
                sums[4 * others[o] + item] = Uint32(words[4 * o + item])
            pred_bins[others[o]] = Uint32(words[8 + o]) & 1048575
            pred_uppers[others[o]] = Uint32(words[8 + o] >> 20) & 1048575
        total_sums = cute.make_rmem_tensor(4, Uint32)
        for item in cutlass.range_constexpr(4):
            total_sums[item] = sums[item]
            total_sums[item] = total_sums[item] + sums[4 + item]
            total_sums[item] = total_sums[item] + sums[8 + item]
        band_before, band_found, band_item, band_value = self.sel_band(
            total_sums, 0, 2048
        )
        band_inclusive = rt.Scan.up(band_value, lane)
        shard_group = first_group - 32 * band_item
        shard_group_above = band_before + band_inclusive - band_value
        band_vote = ballot(
            (band_found != 0)
            & (shard_group_above < 2048)
            & (band_before + band_inclusive >= 2048)
        )
        window_first = s.coarse_scan[token]
        window_shift = s.coarse_scan[4 + token]
        if band_vote != 0:
            source_lane = ffs(band_vote) - 1
            group = cute.arch.shuffle_sync(shard_group, source_lane)
            group_above = cute.arch.shuffle_sync(shard_group_above, source_lane)
            group_count = cute.arch.shuffle_sync(band_value, source_lane)
            rank_above, rank_count = Bucket.group_ranks(sums, first_group, -32, group)
            bucket_prefix = cdiv(window_first, 32) + group
            bucket_above = group_above
            bucket_end = Int32(group == 127)
            bucket_shift = Int32(20)
            bucket_size = group_count
            if Coarse.fine(window_shift, group) != 0:
                shard_bin = group * 32 + 32 - 1 - lane
                bin_values = cute.make_rmem_tensor(3, Uint32)
                bin_uppers = cute.make_rmem_tensor(3, Uint32)
                for i in cutlass.range_constexpr(3):
                    bin_values[i] = pred_bins[i]
                    bin_uppers[i] = pred_uppers[i]
                if shard_bin != pred_word:
                    self.shard_bins(
                        p,
                        src,
                        parity,
                        token,
                        shard_bin,
                        generation,
                        bin_values,
                        bin_uppers,
                    )
                fine_window = (group_above, window_first, window_shift)
                (
                    bin_lane,
                    bucket_prefix,
                    bucket_above,
                    bucket_end,
                    bucket_shift,
                    bucket_size,
                ) = Bucket.fine_bin(
                    bin_values, fine_window, shard_bin, lane, (rank_above, rank_count)
                )
                fine_upper_total = Int32(0)
                fine_rank_upper = cute.make_rmem_tensor(3, Uint32)
                for sh in cutlass.range_constexpr(3):
                    fine_rank_upper[sh] = cute.arch.shuffle_sync(
                        bin_uppers[sh], bin_lane
                    )
                    fine_upper_total = fine_upper_total + Int32(fine_rank_upper[sh])
                fine_take_upper = Int32(bucket_above + fine_upper_total >= 2048)
                bucket_shift = window_shift - 1
                bucket_prefix = 2 * bucket_prefix + fine_take_upper
                bucket_end = bucket_end & fine_take_upper
                if fine_take_upper != 0:
                    bucket_size = fine_upper_total
                    for sh in cutlass.range_constexpr(3):
                        rank_count[sh] = fine_rank_upper[sh]
                if fine_take_upper == 0:
                    bucket_above = bucket_above + fine_upper_total
                    bucket_size = bucket_size - fine_upper_total
                    for sh in cutlass.range_constexpr(3):
                        rank_above[sh] = rank_above[sh] + fine_rank_upper[sh]
                        rank_count[sh] = rank_count[sh] - fine_rank_upper[sh]
            bucket = (
                bucket_prefix,
                bucket_above,
                bucket_end,
                bucket_shift,
                bucket_size,
            )
            ranks = (rank_above, rank_count)
            Bucket.write_bounds(s.loader_bounds, token, bucket, ranks, lane, self.SHARD)

    @cute.jit
    def stage_refine_words(self, p, s, ids: rt.ThreadIds, warp, position, epoch):
        """Bid 144..147 (one per token): after all 144 CTAs published, copy the token's
        refine words to smem.
        """
        tid, bid = ids.tid, ids.bid
        self.poll_ctas(p, tid - 256, 888, epoch)
        parity = Int32(cmod(epoch, 2))
        lane_1 = tid - 256
        if lane_1 == 6:
            shard_in = s.shard_in
            shard_in[6] = ld_cg_b32(
                ga(p.SelectState, StateWords.refine(bid - 144) + 160 + parity * 8 + 4)
            )
        rt.bar_sync(9, 128)
        blocks = cdiv(position + (bid - 144) + 1 + 128 - 1, 128)
        cache_first = cdiv(blocks + 3, 4) * 4
        cache_first = Int32(144)
        candidate_words = Int32(0)
        with rt.warp_leader(warp, 8):
            cute.arch.fence_proxy("async")
            rt.mbar_expect_tx(s.x_full, (cache_first + candidate_words) * 4)
            rt.bulk_g2s(
                s.refine_words_addr,
                ga(p.SelectState, StateWords.refine(bid - 144)),
                cache_first * 4,
                s.x_full,
            )
            if candidate_words > 0:
                rt.bulk_g2s(
                    s.refine_words_addr + cache_first * 4,
                    ga(p.SelectState, 133600 + (bid - 144) * 16240 * 2),
                    candidate_words * 4,
                    s.x_full,
                )
        rt.bar_sync(11, 384)

    @cute.jit
    def prefetch_residual(self, p, s, ids: rt.ThreadIds):
        """CTAs of the reduce: cp.async the REDUCE_ROWS x 48 residual columns of X into
        ``residual``.
        """
        tid, bid = ids.tid, ids.bid
        if bid < self.REDUCE_CTAS:
            if tid < 256 + 6 * self.REDUCE_ROWS:
                chunk = tid - 256
                rt.AsyncCopy.cp_async_cg16(
                    s.residual_addr + chunk * 16,
                    ga(p.X, cdiv(chunk, 6) * 6144 + bid * 48 + cmod(chunk, 6) * 8),
                )
            cute.arch.cp_async_commit_group()

    @cute.jit
    def receive_partials(self, p, s, ids: rt.ThreadIds, epoch):
        """The reduce receive: each load lane polls two of the 224 incoming packets (4
        rows x 56) into received_words.
        """
        tid, bid = ids.tid, ids.bid
        if bid < self.REDUCE_CTAS:
            cute.arch.cp_async_wait_group(0)
            lane_0 = tid - 256
            parity = Int32(cmod(epoch + 1, 2))
            base = p.peer_bases[self.SHARD]
            item_1 = lane_0 + 128
            if item_1 >= 224:
                item_1 = lane_0
            rt.bar_sync(11, 384)
            packets = cute.make_rmem_tensor(8, Uint32)
            rt.TaggedMailbox.poll(
                packets,
                [
                    PartialMailbox.recv_addr(base, item, parity, bid)
                    for item in (lane_0, item_1)
                ],
                rt.TaggedMailbox.tag8(epoch),
            )
            for r2 in cutlass.range_constexpr(2):
                if lane_0 + 128 * r2 < 224:
                    PartialMailbox.store_received(
                        s.received_words,
                        lane_0 + 128 * r2,
                        [packets[4 * r2 + i] for i in range(4)],
                    )
            rt.bar_sync(11, 384)

    @cute.jit
    def score_stage(
        self, p, s, ids, warp, position, long_context, epoch, direct_now, seed_words
    ):
        """Compute warps, bid < 144: key tiles, query staging, tcgen05 scoring, coarse
        histograms, the prelist/listing, the shard bucket exchange and the near/far
        id emission.
        """
        batch_bounds, keep_meta = s.batch_bounds, s.keep_meta
        role_ids = rt.ThreadIds(ids.tid, ids.bid, warp, ids.lane)
        tid, bid = role_ids.tid, role_ids.bid
        SHARD = self.SHARD
        if bid < SCORE_CTAS:
            parity = Int32(cmod(epoch, 2))
            self.score_keys(p, s, role_ids, long_context, position, seed_words, parity)
            rt.bar_sync(8, 256)
            with rt.warp_leader(warp, 0):
                rt.Memory.red_async_release_gpu_add(ga(p.Flags, 740 + bid), 1)
            if long_context != 0 and direct_now != 0:
                rt.bar_sync(11, 384)
            else:
                with rt.warp_leader(warp, 0):
                    cute.arch.atomic_add(rt.Memory.at(p.Flags, 884), Uint32(1))
            shard_keys = Keys.shard_live(position + 4, SHARD)
            if (
                long_context != 0
                and direct_now != 0
                and Int32(shard_keys > STREAM_KEYS) != 0
            ):
                self.stash_scores(
                    p, s, role_ids, 0, Keys.rounds_below(shard_keys, bid), False
                )
            if long_context != 0:
                if direct_now == 0:
                    rt.bar_sync(11, 384)
                if direct_now == 0:
                    self.publish_counts(p, role_ids, epoch, parity)
                if tid < SEL_TOKENS:
                    window_first = s.coarse_scan[tid]
                    window_shift = s.coarse_scan[Coarse.SHIFT + tid]
                    bounds = (window_first, 0, 1, window_shift, 0, 0, 0, 0)
                    for k in cutlass.range_constexpr(8):
                        batch_bounds[tid * 8 + k] = bounds[k]
                if tid == 0:
                    keep_meta[KeepMeta.USABLE] = Int32(position + 4 > DIRECT_KEYS)
                    keep_meta[KeepMeta.EMITTED] = 0
                prelisted = direct_now
                prelist_near = Int32(0)
                prelist_far = Int32(0)
                if tid == 0:
                    keep_meta[KeepMeta.COUNTING] = 0
                    keep_meta[KeepMeta.PRELIST_OK] = 1
                rt.bar_sync(8, 256)
                prelist_near, prelist_far = self.prelist_stage(
                    p,
                    s,
                    role_ids,
                    prelisted,
                    parity,
                    position,
                    prelist_near,
                    prelist_far,
                )
                sb = p.shard_base(SHARD)
                shard_bases, near_bases = p.shard_bases, p.near_bases
                warm_first = 1 - prelisted
                launch = (
                    position,
                    epoch,
                    direct_now,
                    parity,
                    (sb, shard_bases, near_bases),
                )
                for warm_round in range(warm_first, 2):
                    self.bucket_round(
                        p,
                        s,
                        role_ids,
                        launch,
                        prelisted,
                        prelist_near,
                        prelist_far,
                        warm_round,
                    )
                rt.bar_sync(8, 256)
                cute.arch.griddepcontrol_launch_dependents()
                self.publish_listing(p, s, role_ids, position, parity)
                shard_full = Int32(0)
                for t in cutlass.range_constexpr(SEL_TOKENS):
                    pushable_end = s.batch_bounds[t * 8 + 2]
                    pushable_size = s.batch_bounds[t * 8 + 4]
                    shard_full = shard_full | Int32(
                        self.rows(p, position + t + 1, t) > TOPK
                    ) & 1 - Int32(pushable_end == 0 and pushable_size <= CAND_CAP)
                self.rescore_foreign(p, s, role_ids, shard_full, position)
            self.emit_stage(
                p, s, role_ids, position, long_context, epoch, direct_now, parity
            )

    @cute.jit
    def publish_counts(self, p, ids, epoch, parity):
        """Staged mode, bid < 128, tid < 132: publish this CTA's 32 fine bins and its
        coarse group count per token to the two other shards' mailboxes.
        """
        tid, bid = ids.tid, ids.bid
        SHARD = self.SHARD
        if bid < 128 and tid < 132:
            token = cdiv(tid, 33)
            item = cmod(tid, 33)
            state_index = (
                StateWords.BINS
                + ((parity * 4 + token) * StateWords.BIN_COUNT + bid * 32 + item) * 2
            )
            word = bid * 32 + item
            if item == 32:
                state_index = StateWords.group(parity, token, bid)
                word = BucketMailbox.GROUP_WORDS + bid
            value = ld_cg_b32(ga(p.SelectState, state_index))
            record = Uint64(epoch + 1) << 32 | Uint64(value)
            if item < 32:
                word_1 = ld_cg_b32(ga(p.SelectState, state_index + 1))
                record = (
                    Uint64(epoch + 1 & 16777215) << 40
                    | Uint64(word_1) << 20
                    | Uint64(value)
                )
            dst0, dst1 = (b for i, b in enumerate(p.shard_bases) if i != SHARD)
            st_volatile_u64(
                SelectionMailbox.word_addr(dst0, parity, SHARD, token, word), record
            )
            st_volatile_u64(
                SelectionMailbox.word_addr(dst1, parity, SHARD, token, word), record
            )

    @cute.jit
    def bucket_round(
        self, p, s, ids, launch, prelisted, prelist_near, prelist_far, warm_round
    ):
        """One bucket round (base: the warm round 0 on the prelist bounds, then round 1
        on the exchanged bounds): fix the bucket bounds, publish the early headers,
        then list / count / push the window candidates.
        """
        position, epoch, direct_now, parity, mailboxes = launch
        tid, bid, warp, lane = ids.tid, ids.bid, ids.warp, ids.lane
        SHARD = self.SHARD
        sb, shard_bases, near_bases = mailboxes
        batch_bounds, near_meta, keep_meta = s.batch_bounds, s.near_meta, s.keep_meta
        if tid == 0:
            near_meta[NearMeta.WARM] = Int32(warm_round == 0)
        self.shard_bucket(p, s, ids, prelisted, parity, sb)
        if prelisted != 0:
            if warm_round != 0:
                rt.bar_sync(11, 384)
                if tid < 32:
                    batch_bounds[tid] = s.loader_bounds[tid]
            if warm_round == 0:  # noqa: SIM102
                if tid < 32:
                    warm_token = cdiv(tid, 8)
                    warm_word = cmod(tid, 8)
                    warm_value = Int32(0)
                    if warm_word == 0:
                        warm_value = s.keep_meta[KeepMeta.FLOOR_PREFIX + warm_token]
                    if warm_word == 3:
                        warm_value = s.keep_meta[KeepMeta.FLOOR_SHIFT + warm_token]
                    if warm_word == 4:
                        warm_value = Int32(1)
                    batch_bounds[tid] = warm_value
        rt.bar_sync(8, 256)
        if tid < SEL_TOKENS:
            pushable_end = s.batch_bounds[tid * 8 + 2]
            pushable_size = s.batch_bounds[tid * 8 + 4]
            near_rows = self.rows(p, position + tid + 1, tid)
            near_meta[tid] = (
                direct_now
                & Int32(near_rows > TOPK)
                & Int32(pushable_end == 0 and pushable_size <= CAND_CAP)
            )
        early_go = Int32(warm_round != 0)
        if bid == 0 and tid < SEL_TOKENS and early_go != 0:
            self.publish_early(p, s, ids, parity, sb)
        token_half = cdiv(warp, 4)
        key_warp = cmod(warp, 4)
        key_lane = cmod(tid, 128)
        lanes_below = (Uint32(1) << Uint32(lane)) - 1
        live_end = Keys.shard_live(position + 4, SHARD)
        list_rounds = Keys.rounds_below(live_end, bid)
        rounds_0 = list_rounds
        slow_rounds = rounds_0
        bound_digit = cute.make_rmem_tensor(2, Int32)
        bound_shift = cute.make_rmem_tensor(2, Uint32)
        bound_long = cute.make_rmem_tensor(2, Int32)
        for t in cutlass.range_constexpr(2):
            token = 2 * t + token_half
            bound_digit[t] = s.batch_bounds[token * 8]
            bound_shift[t] = Uint32(s.batch_bounds[token * 8 + 3])
            bound_long[t] = Int32(token < SEL_TOKENS) & Int32(
                self.rows(p, position + token + 1, token) > TOPK
            )
        use_lists = Int32(position + 4 > DIRECT_KEYS)
        kept = cute.make_rmem_tensor(2, Int32)
        kept[0] = 0
        kept[1] = 0
        if prelisted != 0:
            if tid < SEL_TOKENS:  # noqa: SIM102
                if self.rows(p, position + tid + 1, tid) > TOPK:  # noqa: SIM102
                    if (
                        s.keep_meta[KeepMeta.FLOOR_FOUND + tid] == 0
                        or s.batch_bounds[tid * 8 + 4] == 0
                    ):
                        keep_meta[KeepMeta.PRELIST_OK] = 0
            rt.bar_sync(8, 256)
            prelist_ok = s.keep_meta[KeepMeta.PRELIST_OK]
            if tid == 0:
                keep_meta[KeepMeta.COUNTING] = prelist_ok
            rt.bar_sync(8, 256)
            if prelist_ok != 0:
                kept[0] = prelist_near
                kept[1] = prelist_far
                slow_rounds = Int32(0)
        slow_rounds = slow_rounds * Int32(warm_round != 0)
        bounds = (bound_long, bound_digit, bound_shift)
        rd = Round(launch, (token_half, key_warp, key_lane, lanes_below), bounds, kept)
        attempt_live = Int32(1)
        for _attempt in range(2):
            if attempt_live != 0:
                for window in range(0, slow_rounds, WINDOW_ROUNDS):
                    if live_end > STREAM_KEYS:
                        self.stash_scores(p, s, ids, window, rounds_0)
                    window_end = cutlass.min(window + WINDOW_ROUNDS, rounds_0)
                    largest = rmem_fill(2, Uint32, 0)
                    inverted = rmem_fill(2, Uint32, 0)
                    self.window_scan(
                        p, s, ids, rd, window, window_end, use_lists, largest, inverted
                    )
                    for t in cutlass.range_constexpr(2):
                        lg = redux_max_u32(largest[t])
                        iv = redux_max_u32(inverted[t])
                        if lane == 0 and (lg | iv) != 0:
                            extremes = (
                                StateWords.EXTREMES
                                + (parity * 4 + 2 * t + token_half) * 2
                            )
                            rt.Memory.atomic_max_u32(p.SelectState, extremes, lg)
                            rt.Memory.atomic_max_u32(p.SelectState, extremes + 1, iv)
                    rt.bar_sync(8, 256)
                    self.window_reserve(p, s, ids, rd, use_lists, window, window_end)
                if use_lists == 0:
                    attempt_live = Int32(0)
                else:
                    if lane == 0:
                        keep_meta[warp * 2] = kept[0]
                        keep_meta[warp * 2 + 1] = kept[1]
                        if kept[0] > KEEP_CAP or kept[1] > KEEP_CAP:
                            keep_meta[KeepMeta.USABLE] = 0
                    if tid < SEL_TOKENS:  # noqa: SIM102
                        if self.rows(p, position + tid + 1, tid) > TOPK:  # noqa: SIM102
                            if s.batch_bounds[tid * 8 + 4] == 0:
                                keep_meta[KeepMeta.USABLE] = 0
                    rt.bar_sync(8, 256)
                    usable = s.keep_meta[KeepMeta.USABLE]
                    attempt_live, use_lists, slow_rounds = self.keep_list_stage(
                        p,
                        s,
                        ids,
                        rd,
                        usable,
                        rounds_0,
                        warm_round,
                        attempt_live,
                        use_lists,
                        slow_rounds,
                    )

    @cute.jit
    def publish_early(self, p, s, ids, parity, sb):
        """Bid 0, tid < 4 (token = tid): the bucket header words to SelectState and the
        early bucket / slot records to this shard's mailbox (base shard 0: also the
        replica refine headers).
        """
        SelectState = p.SelectState
        tid = ids.tid
        SHARD = self.SHARD
        shard_word = StateWords.refine_word(tid, StateWords.HEADER) + parity * 8
        shard_end = s.batch_bounds[tid * 8 + 2]
        shard_above = s.batch_bounds[tid * 8 + 1]
        shard_size = s.batch_bounds[tid * 8 + 4]
        shard_first_slot = s.batch_bounds[tid * 8 + 5]
        shard_count = s.batch_bounds[tid * 8 + 6]
        shard_lower = s.batch_bounds[tid * 8 + 7]
        SelectState[shard_word] = Uint32(shard_above)
        SelectState[shard_word + 1] = Uint32(
            cutlass.max(shard_size, shard_end * CAND_OVERFLOW)
        )
        SelectState[shard_word + 2] = Uint32(shard_first_slot)
        SelectState[shard_word + 3] = Uint32(shard_count)
        SelectState[shard_word + 4] = Uint32(shard_lower)
        launch_epoch = s.epoch_slot[0]
        early_tag = Uint64(launch_epoch + 1) << 32
        early_size = cutlass.min(
            cutlass.max(shard_size, shard_end * CAND_OVERFLOW), CAND_OVERFLOW
        )
        early_bucket = early_tag | Uint64(
            Uint32(shard_above) | Uint32(early_size) << 16
        )
        early_slots = early_tag | Uint64(
            Uint32(shard_first_slot) & 65535 | (Uint32(shard_count) & 65535) << 16
        )
        own = parity * 3 + SHARD
        st_volatile_u64(
            sb
            + BucketMailbox.BASE
            + (BucketMailbox.WORDS + (own * 4 + tid) * BucketMailbox.TOKEN_WORDS * 8),
            early_bucket,
        )
        st_volatile_u64(
            sb
            + BucketMailbox.BASE
            + (
                BucketMailbox.WORDS
                + ((own * 4 + tid) * BucketMailbox.TOKEN_WORDS + 1) * 8
            ),
            early_slots,
        )
        if cutlass.const_expr(SHARD == 0):
            refined_size = Int32(CAND_OVERFLOW)
            if s.near_meta[tid] != 0:
                refined_size = early_size
            refine_header = early_tag | Uint64(
                Uint32(shard_above) | Uint32(refined_size) << 16
            )
            for list_base in p.near_bases:
                st_volatile_u64(
                    list_base
                    + BucketMailbox.BASE
                    + (BucketMailbox.WORDS + (parity * 4 + tid) * 8),
                    refine_header,
                )

    @cute.jit
    def publish_listing(self, p, s, ids, position, parity):
        """After the bucket rounds: warps 0-3 publish this CTA's listed-above total per
        token; bid 0 the bucket's above count (and the overflow candidate count).
        """
        SelectState = p.SelectState
        tid, bid, warp, lane = ids.tid, ids.bid, ids.warp, ids.lane
        SHARD = self.SHARD
        if warp < 4:
            lean_live = self.rows(p, position + warp + 1, warp)
            if lean_live > TOPK:
                lean_rows = Keys.shard_live(lean_live, SHARD)
                lean_rounds = Keys.rounds_below(lean_rows, bid)
                lean_total = Uint32(0)
                for lean_half in cutlass.range_constexpr(2):
                    lean_rnd = lane + lean_half * 32
                    if lean_rnd < lean_rounds and lean_rnd < MAX_ROUNDS:
                        for warp_0 in cutlass.range_constexpr(4):
                            lean_total = lean_total + Uint32(
                                s.batch_counts[Rounds.cell(warp, lean_rnd, warp_0)]
                            )
                lean_total = redux_add_u32(lean_total)
                if lane == 0:
                    SelectState[StateWords.refine_word(warp, bid)] = lean_total
        if bid == 0 and tid < SEL_TOKENS:  # noqa: SIM102
            if self.rows(p, position + tid + 1, tid) > TOPK:
                SelectState[StateWords.bucket(parity, tid)] = Uint32(
                    s.batch_bounds[tid * 8 + 1]
                )
                if s.batch_bounds[tid * 8 + 2] != 0:
                    rt.Memory.atomic_add_u32(
                        p.SelectState, StateWords.bucket(parity, tid) + 1, CAND_OVERFLOW
                    )

    @cute.jit
    def score_keys(self, p, s, ids, long_context, position, seed_words, parity):
        """Long context: stage the key tile, quantize the query, run the tcgen05 score
        rounds and publish the coarse histograms.
        """
        coarse, group_counts, head_weight = s.coarse, s.group_counts, s.head_weight
        tid, bid, warp, lane = ids.tid, ids.bid, ids.warp, ids.lane
        SHARD = self.SHARD
        if long_context != 0:
            rt.bar_sync(11, 384)
            first_live = Keys.shard_live(position + 4, SHARD)
            kb = (cdiv(bid * 128, 128) * 3 + SHARD) * 128 + cmod(bid * 128, 128)
            if first_live > STREAM_KEYS:
                self.stage_key_tile(
                    p, ids, kb, first_live, s.umma_keys_addr - 32768, s.key_scales_addr
                )
            else:
                self.stage_key_tile(
                    p, ids, kb, first_live, s.umma_keys_addr, s.key_scales_addr + 1024
                )
            self.coarse_seed(p, s, ids, seed_words)
            quad = tid - 224
            hoist_quad = cute.make_rmem_tensor(2, Uint32)
            hoist_quad[0] = Uint32(0)
            hoist_quad[1] = Uint32(0)
            if quad >= 0:
                hoist_quad[0], hoist_quad[1] = ld_cg_v2(ga(p.HeadWeights, quad * 4))
            for part in cutlass.range_constexpr(64):
                coarse[part * 256 + tid] = Uint32(0)
            for part in cutlass.range_constexpr(2):
                group_counts[part * 256 + tid] = Uint32(0)
            hoist_freq = [p.Freq[cmod(tid, 8) * 8 + i] for i in range(8)]
            self.stage_query(p, s, ids, hoist_quad, hoist_freq)
            rt.bar_sync(8, 256)
            if tid < 128:
                head_weight[tid] = fmul(
                    fmul(
                        fmul(s.head_weight[tid], s.q_scales[tid]), 0.08838834764831845
                    ),
                    0.1767766952966369,
                )
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            rt.bar_sync(8, 256)
            position_0 = p.CurPos[0]
            live_max = Keys.shard_live(position_0 + 4, SHARD)
            resident = Int32(live_max <= STREAM_KEYS)
            first_key = bid * KEY_BLOCK
            rounds = Keys.rounds_below(live_max, bid)
            row = cmod(warp, 4) * 32 + lane
            token_pair = cdiv(warp, 4)
            first_near = s.coarse_scan[token_pair * 2]
            first_far = s.coarse_scan[token_pair * 2 + 1]
            shift_near = s.coarse_scan[Coarse.SHIFT + token_pair * 2]
            shift_far = s.coarse_scan[Coarse.SHIFT + token_pair * 2 + 1]
            streaming = 1 - resident
            if warp == 0:
                rt.mbar_arrive(
                    s.ws_full + ((0 & 1 + 2 * streaming) + 2 * (1 - streaming)) * 8
                )
            with rt.warp_leader(warp, 0):
                rt.mbar_arrive(s.ws_query)
            geom = (
                rounds,
                first_key,
                streaming,
                live_max,
                row,
                token_pair,
                position_0,
                resident,
            )
            self.score_rounds(
                p, s, ids, geom, (first_near, shift_near, first_far, shift_far)
            )
            rt.bar_sync(8, 256)
            with rt.warp_leader(warp, 0):
                rt.mbar_arrive(s.ws_scored)
            self.publish_histograms(p, s, ids, parity)

    @cute.jit
    def publish_histograms(self, p, s, ids, parity):
        """Add this CTA's coarse group counts to SelectState and walk the non-empty
        groups to add their fine bins (published groups only: the band 56..79, or
        every group when the window is unseeded).
        """
        tid, warp, lane = ids.tid, ids.warp, ids.lane
        for half in cutlass.range_constexpr(2):
            cell = half * 256 + tid
            cell_count = s.group_counts[cell]
            if cell_count != 0:
                cute.arch.atomic_add(
                    rt.Memory.at(
                        p.SelectState,
                        StateWords.GROUPS + parity * 4 * StateWords.GROUP_COUNT + cell,
                    ),
                    cell_count,
                )
            pending = ballot(cell_count != 0)
            while pending != 0:
                walk_cell = half * 256 + warp * 32 + ffs(pending) - 1
                walk_token = cdiv(walk_cell, 128)
                walk_group = cmod(walk_cell, 128)
                walk_bin = walk_group * 32 + lane
                walk_count = s.coarse[walk_token * StateWords.BIN_COUNT + walk_bin]
                walk_banded = Int32(walk_group >= 56 and walk_group < 80)
                walk_publish = (
                    Int32(s.coarse_scan[Coarse.SHIFT + walk_token] == 20) | walk_banded
                )
                if walk_count != 0 and walk_publish != 0:
                    walk_upper = walk_count >> 16
                    walk_pair = Uint64(walk_upper) << 32 | Uint64(
                        (walk_count & 65535) + walk_upper
                    )
                    cute.arch.atomic_add(
                        cute.recast_ptr(
                            rt.Memory.at(
                                p.SelectState,
                                StateWords.bin(parity, walk_token, walk_bin),
                            ),
                            dtype=Uint64,
                        ),
                        walk_pair,
                    )
                pending = pending & pending - 1

    @cute.jit
    def stage_key_tile(self, p, ids, kb, live, key_addr, scale_addr):
        """cp.async this CTA's 128-key FP8 tile (swizzled 16 B chunks) from key ``kb``
        to ``key_addr`` and its 128 scales to ``scale_addr``.
        """
        tid, bid = ids.tid, ids.bid
        BlockTable, IndexKeyCache = p.BlockTable, p.IndexKeyCache
        page_rows0 = BlockTable[kb >> 6] * 66 - (kb >> 6 << 6)
        page_rows1 = BlockTable[(kb >> 6) + 1] * 66 - ((kb >> 6) + 1 << 6)
        for part in cutlass.range_constexpr(4):
            chunk = tid + part * 256
            row = cdiv(chunk, 8)
            col = cmod(chunk, 8)
            rows = page_rows0 if (kb + row & 64) == 0 else page_rows1
            src = ga(
                IndexKeyCache, Int64(col * 8) + Int64(Uint32(rows + (kb + row))) * 64
            )
            dst = key_addr + row * 128 + (col ^ cmod(row, 8)) * 16
            rt.AsyncCopy.cp_async_cg16_pred_hint(
                1 if live > bid * 128 + row else 0, dst, src, 0x12F0000000000000
            )
        if tid < 32:
            r = kb + tid * 4
            rows = page_rows0 if (r & 64) == 0 else page_rows1
            src = ga(
                IndexKeyCache,
                Int64(rows + r - (r & 63)) * 64 + Int64(4096 + (r & 63) * 2),
            )
            rt.AsyncCopy.cp_async_cg16_pred(
                1 if live > bid * 128 + tid * 4 else 0, scale_addr + tid * 16, src
            )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def coarse_seed(self, p, s, ids, seed_words):
        """Tid < 4: this step's coarse window (first group, shift) per token from the
        previous steps' selection seeds.
        """
        SelectState, coarse_scan = p.SelectState, s.coarse_scan
        tid, bid = ids.tid, ids.bid
        if tid < SEL_TOKENS:
            values = cute.make_rmem_tensor(4, Uint32)
            want = Uint32(0)
            for key_row in cutlass.range_constexpr(4):
                word = seed_words[key_row]
                want = want | word & 1
                values[key_row] = word & Uint32(4294967294)
            for outer in cutlass.range_constexpr(4):
                for inner in cutlass.range_constexpr(3 - outer):
                    high = cutlass.max(values[inner], values[inner + 1])
                    low = cutlass.min(values[inner], values[inner + 1])
                    values[inner] = high
                    values[inner + 1] = low
            known = Int32(0)
            for key_row in cutlass.range_constexpr(4):
                known = known + Int32(values[key_row] != 0)
            seed = values[0]
            for count in cutlass.range_constexpr(2, 5):
                if known == count:
                    seed = (values[cdiv(count - 1, 2)] >> 1) + (
                        values[cdiv(count, 2)] >> 1
                    )
            if seed > 4194304:
                seed = seed - 4194304
            if want == 0:
                seed = Uint32(0)
            seed_0 = seed
            if bid == 0 and tid == 0:
                SelectState[StateWords.PREVIOUS_SEED] = seed_0
            first = Int32(0)
            shift = Int32(20)
            if seed_0 != 0:
                shift = Int32(15)
                centred = Int32(seed_0 >> 15) - 2048
                first = cutlass.min(cutlass.max(cdiv(centred, 32) * 32, 0), 126976)
            coarse_scan[tid] = first
            coarse_scan[Coarse.SHIFT + tid] = shift

    @cute.jit
    def stage_query(self, p, s, ids, hoist_quad, hoist_freq):
        """Quantize the 128 indexer query rows to FP8 (per-row scales) into the UMMA
        query tile, scaled by the Scores maxima.
        """
        q_scales = s.q_scales
        tid, bid = ids.tid, ids.bid
        rt.bar_sync(11, 384)
        group_head = cdiv(tid, 8)
        group_column = cmod(tid, 8)
        q_rows = [32 * g + group_head for g in range(4)]
        q_raw = [
            ld_cg_v4(ga(p.RawQuery, q_rows[g] * 128 + half * 64 + group_column * 8))
            for g in range(4)
            for half in range(2)
        ]
        maxima_base = 128 * ((self.SHARD + 1) % 3) + group_head * 4
        q_maxima = [ld_cg_v4(ga(p.Scores, 384 * g + maxima_base)) for g in range(4)]
        QueryTile.store_head_weight(s.head_weight, hoist_quad, tid)
        for g in cutlass.range_constexpr(4):
            q_lo = rt.bf16x8_to_f32(list(q_raw[2 * g]))
            q_hi = rt.bf16x8_to_f32(list(q_raw[2 * g + 1]))
            if q_rows[g] == bid:
                rt.Memory.stg_u32x4(
                    p.IndexQuery,
                    q_rows[g] * 128 + group_column * 8,
                    QueryTile.pack_bf16x8(q_lo),
                )
                rt.Memory.stg_u32x4(
                    p.IndexQuery,
                    q_rows[g] * 128 + 64 + group_column * 8,
                    QueryTile.pack_bf16x8(q_hi),
                )
            group_amax = Float32(0.0)
            for quarter in cutlass.range_constexpr(4):
                group_amax = rt.max_noftz(group_amax, bits_f32(q_maxima[g][quarter]))
            q_exponent = QueryTile.fp8_exponent(group_amax)
            QueryTile.store_fp8(
                s.query_words_addr, q_rows[g], group_column, q_lo, q_exponent
            )
            QueryTile.store_fp8(
                s.query_words_addr, q_rows[g], 8 + group_column, q_hi, q_exponent
            )
            if group_column == 0:
                q_scales[q_rows[g]] = bits_f32(q_exponent << 23)

    @cute.jit
    def score_rounds(self, p, s, ids, geom, windows):
        """Per 18432-key round: wait for the tcgen05 scores, apply head weights and key
        scales, write Scores/stash and the coarse histograms.
        """
        rounds, first_key, streaming, live_max, row, pair, position_0, resident = geom
        Scores, stash, unit_max = p.Scores, s.stash, s.unit_max
        warp, lane = ids.warp, ids.lane
        SHARD = self.SHARD
        for rnd in range(rounds):
            block = first_key + rnd * ROUND_KEYS
            score_slot = (rnd & 1 + 2 * streaming) + 2 * (1 - streaming)
            score_use = rnd >> 1 + streaming
            mbar_wait(s.ws_full + score_slot * 8, score_use & 1)
            key_scale = Float32(0.0)
            if live_max > block + row:
                key_scale = IndexScale.fp32(
                    s.key_scales[score_slot * 128 + row], self.PACKED_SCALE
                )
            mbar_wait(s.ws_done + score_slot * 8, score_use & 1)
            rt.TensorCore.fence_after_thread_sync()
            tmem_load = rt.TensorCore.tmem_ld_32x32b_x64(
                s.taddr + (cmod(warp, 4) * 32 << 16) + score_slot * 128 + pair * 64
            )
            cute.arch.fence_view_async_tmem_load()
            rt.TensorCore.fence_before_thread_sync()
            rt.mbar_arrive(s.ws_free + score_slot * 8)
            key = block + row
            for slot_0 in cutlass.range_constexpr(2):
                tok = pair * 2 + slot_0
                live = Keys.shard_live(self.rows(p, position_0 + tok + 1, tok), SHARD)
                if tok >= SEL_TOKENS:
                    live = Int32(0)
                score = KeyScore.weighted(
                    tmem_load, slot_0, s.head_weight, tok, key_scale
                )
                key_0 = KeyScore.key(as_u32(score))
                unit_hi = redux_max_u32(key_0 * Uint32(key < live))
                if lane == 0 and tok < SEL_TOKENS:
                    unit_max[Rounds.row(tok, rnd) + cmod(warp, 4)] = unit_hi
                if key < live:
                    Scores[tok * p.score_stride + Keys.shard_id(key, SHARD)] = score
                    if resident != 0:
                        stash[Stash.index(tok, rnd, row)] = score
                    window_first, window_shift = (
                        windows[2 * slot_0],
                        windows[2 * slot_0 + 1],
                    )
                    digit = Int32(key_0 >> Uint32(window_shift)) - window_first
                    if digit >= 0:
                        fine_half = cutlass.min(
                            Int32(key_0 >> Uint32(window_shift - 1)) - 2 * window_first,
                            8191,
                        )
                        cute.arch.atomic_add(
                            rt.Memory.at(
                                s.coarse, tok * StateWords.BIN_COUNT + (fine_half >> 1)
                            ),
                            Uint32(1) << Uint32((fine_half & 1) * 16),
                        )
                        cute.arch.atomic_add(
                            rt.Memory.at(
                                s.group_counts,
                                tok * 128 + cdiv(cutlass.min(digit, 4095), 32),
                            ),
                            Uint32(1),
                        )

    @cute.jit
    def prelist_stage(
        self, p, s, ids, prelisted, parity, position, prelist_near, prelist_far
    ):
        """Direct mode: the floor group per token (warps 4-7), then list every warp's
        keys at or above it into keep_words.
        """
        warp, tid, lane = ids.warp, ids.tid, ids.lane
        if prelisted != 0:
            if warp >= 4 and warp < 8:
                self.prelist_floor(p, s, ids, parity, warp - 4)
            rt.bar_sync(8, 256)
            half, key_warp = cdiv(warp, 4), cmod(warp, 4)
            listing = (half, key_warp, cmod(tid, 128), (Uint32(1) << Uint32(lane)) - 1)
            shard_keys = Keys.shard_live(position + 4, self.SHARD)
            listing_rounds = Keys.rounds_below(shard_keys, ids.bid)
            if shard_keys > STREAM_KEYS:
                cute.arch.cp_async_wait_group(0)
                rt.bar_sync(8, 256)
            listed_near = self.prelist_side(
                p, s, ids, position, listing_rounds, listing, 0
            )
            listed_far = self.prelist_side(
                p, s, ids, position, listing_rounds, listing, 1
            )
            keep_meta = s.keep_meta
            if ids.lane == 0:
                keep_meta[warp * 2] = listed_near
                keep_meta[warp * 2 + 1] = listed_far
                if listed_near > KEEP_CAP or listed_far > KEEP_CAP:
                    keep_meta[KeepMeta.PRELIST_OK] = 0
            rt.bar_sync(8, 256)
            prelist_near = listed_near
            prelist_far = listed_far
        return prelist_near, prelist_far

    @cute.jit
    def prelist_floor(self, p, s, ids, parity, floor_token):
        """Warp 4 + token: the floor bucket (prefix, shift, found) of the token's top
        2048 from the coarse group counts and fine bins, into keep_meta.
        """
        lane = ids.lane
        floor_sums = ld_cg_v4(
            ga(p.SelectState, StateWords.group(parity, floor_token, 124 - lane * 4))
        )
        vote, floor_group, floor_group_above, _ = Bucket.group_crossing(
            floor_sums, lane
        )
        floor_first = s.coarse_scan[floor_token]
        floor_window_shift = s.coarse_scan[Coarse.SHIFT + floor_token]
        floor_prefix = Int32(0)
        floor_shift = Int32(20)
        floor_found = Int32(0)
        if vote != 0:
            floor_source = ffs(vote) - 1
            floor_group = cute.arch.shuffle_sync(floor_group, floor_source)
            floor_group_above = cute.arch.shuffle_sync(floor_group_above, floor_source)
            floor_fine = Coarse.fine(floor_window_shift, floor_group)
            if floor_fine == 0:
                floor_prefix = cdiv(floor_first, 32) + floor_group
                floor_found = Int32(1)
            if floor_fine != 0:
                floor_bin = floor_group * 32 + 32 - 1 - lane
                floor_bin_count = Int32(
                    ld_cg_b32(
                        ga(
                            p.SelectState,
                            StateWords.bin(parity, floor_token, floor_bin),
                        )
                    )
                )
                floor_bin_inclusive = rt.Scan.up(floor_bin_count, lane)
                floor_bin_before = (
                    floor_group_above + floor_bin_inclusive - floor_bin_count
                )
                vote_1 = ballot(
                    floor_bin_before < 2048
                    and floor_bin_before + floor_bin_count >= 2048
                )
                if vote_1 != 0:
                    floor_prefix = floor_first + cute.arch.shuffle_sync(
                        floor_bin, ffs(vote_1) - 1
                    )
                    floor_shift = floor_window_shift
                    floor_found = Int32(1)
                if vote_1 == 0:
                    floor_prefix = floor_first + floor_group * 32
                    floor_shift = floor_window_shift
                    floor_found = Int32(1)
        keep_meta = s.keep_meta
        if lane == 0:
            keep_meta[KeepMeta.FLOOR_PREFIX + floor_token] = floor_prefix
            keep_meta[KeepMeta.FLOOR_SHIFT + floor_token] = floor_shift
            keep_meta[KeepMeta.FLOOR_FOUND + floor_token] = floor_found

    @cute.jit
    def prelist_side(
        self, p, s, ids, position, listing_rounds, listing, side: cutlass.Constexpr
    ):
        """List this warp's keys at or above the floor of token ``side * 2 + warp / 4``
        (near side 0, far side 1) into keep list ``warp * 2 + side``; return how many
        were found (the list keeps the first 256).
        """
        bid, warp, lane = ids.bid, ids.warp, ids.lane
        SHARD = self.SHARD
        batch_counts, keep_words = s.batch_counts, s.keep_words
        listing_half, listing_key_warp, listing_lane, listing_below = listing
        listing_token = side * 2 + listing_half
        listing_prefix = s.keep_meta[KeepMeta.FLOOR_PREFIX + listing_token]
        listing_shift = Uint32(s.keep_meta[KeepMeta.FLOOR_SHIFT + listing_token])
        listing_rows = self.rows(p, position + listing_token + 1, listing_token)
        listing_live = Keys.shard_live(listing_rows, SHARD)
        listing_long = Int32(listing_rows > TOPK) & Int32(
            s.keep_meta[KeepMeta.FLOOR_FOUND + listing_token] != 0
        )
        listing_hi = Uint32(0)
        if listing_rounds > lane:
            batch_counts[Rounds.cell(listing_token, lane, listing_key_warp)] = 0
            listing_hi = s.unit_max[Rounds.cell(listing_token, lane, listing_key_warp)]
        listing_hot = (
            listing_long
            & Int32(listing_rounds > lane)
            & Int32(listing_prefix <= Int32(listing_hi >> listing_shift))
        )
        listing_pending = ballot(listing_hot != 0)
        listing_kept = Int32(0)
        while listing_pending != 0:
            member_round = [ffs(listing_pending) - 1]
            member_valid = [1]
            listing_pending = listing_pending & listing_pending - 1
            for m in cutlass.range_constexpr(3):
                member_valid.append(Int32(listing_pending != 0))
                member_round.append(
                    cutlass.max(ffs(listing_pending) - 1, member_round[0])
                )
                listing_pending = listing_pending & listing_pending - 1
            unit_index = []
            unit_key = []
            unit_keep = []
            for m in cutlass.range_constexpr(4):
                unit_index.append(
                    bid * 128 + listing_lane + member_round[m] * ROUND_KEYS
                )
                unit_key.append(
                    KeyScore.key(
                        as_u32(
                            s.stash[
                                Stash.index(
                                    listing_token, member_round[m], listing_lane
                                )
                            ]
                        )
                    )
                )
                unit_keep.append(
                    member_valid[m]
                    & Int32(unit_index[m] < listing_live)
                    & Int32(listing_prefix <= Int32(unit_key[m] >> listing_shift))
                )
            for m in cutlass.range_constexpr(4):
                vote = ballot(unit_keep[m] != 0)
                if unit_keep[m] != 0:
                    unit_entry = listing_kept + popc(vote & listing_below)
                    if unit_entry < KEEP_CAP:
                        unit_word = KeepList.word(warp * 2 + side, unit_entry)
                        keep_words[unit_word] = Uint32(unit_index[m])
                        keep_words[unit_word + 1] = unit_key[m]
                listing_kept = listing_kept + popc(vote)
        return listing_kept

    @cute.jit
    def shard_bucket(self, p, s, ids, prelisted, parity, sb):
        """Warps 0-3 (token = warp): sum the three shards' coarse group counts and bins
        from the mailboxes and fix this token's bucket bounds.
        """
        warp, lane = ids.warp, ids.lane
        SHARD = self.SHARD
        if prelisted == 0:  # noqa: SIM102
            if warp < 4:
                token = warp
                generation = s.epoch_slot[0] + 1
                first_group = 124 - lane * 4
                sums = cute.make_rmem_tensor(12, Uint32)
                w = ld_cg_v4(
                    ga(p.SelectState, StateWords.group(parity, token, first_group))
                )
                for i in cutlass.range_constexpr(4):
                    sums[4 * SHARD + i] = w[i]
                others = [o for o in range(3) if o != SHARD]
                for o in cutlass.range_constexpr(2):
                    ready = Int32(0)
                    while ready == 0:
                        hist_ready = Int32(1)
                        for item in cutlass.range_constexpr(4):
                            hist_word = ld_volatile_u64(
                                SelectionMailbox.word_addr(
                                    sb,
                                    parity,
                                    others[o],
                                    token,
                                    BucketMailbox.GROUP_WORDS + first_group + item,
                                )
                            )
                            sums[4 * others[o] + item] = Uint32(hist_word)
                            hist_ready = hist_ready & rt.Poll.tagged(
                                hist_word, generation, False
                            )
                        ready = Int32(cute.arch.vote_all_sync(hist_ready != 0))
                total_sums = cute.make_rmem_tensor(4, Uint32)
                for item in cutlass.range_constexpr(4):
                    total_sums[item] = sums[item]
                    total_sums[item] = total_sums[item] + sums[4 + item]
                    total_sums[item] = total_sums[item] + sums[8 + item]
                vote, group, group_above, group_count = Bucket.group_crossing(
                    total_sums, lane
                )
                window_first = s.coarse_scan[token]
                window_shift = s.coarse_scan[Coarse.SHIFT + token]
                if vote != 0:
                    source = ffs(vote) - 1
                    group = cute.arch.shuffle_sync(group, source)
                    group_above = cute.arch.shuffle_sync(group_above, source)
                    group_count = cute.arch.shuffle_sync(group_count, source)
                    rank_above, rank_count = Bucket.group_ranks(
                        sums, first_group, 1, group
                    )
                    bucket_prefix = cdiv(window_first, 32) + group
                    bucket_above = group_above
                    bucket_end = Int32(group == 127)
                    bucket_shift = Int32(20)
                    bucket_size = group_count
                    if Coarse.fine(window_shift, group) != 0:
                        shard_bin = group * 32 + 32 - 1 - lane
                        bin_values = cute.make_rmem_tensor(3, Uint32)
                        bin_uppers = cute.make_rmem_tensor(3, Uint32)
                        self.shard_bins(
                            p,
                            sb,
                            parity,
                            token,
                            shard_bin,
                            generation,
                            bin_values,
                            bin_uppers,
                        )
                        fine_window = (group_above, window_first, window_shift)
                        (
                            _,
                            bucket_prefix,
                            bucket_above,
                            bucket_end,
                            bucket_shift,
                            bucket_size,
                        ) = Bucket.fine_bin(
                            bin_values,
                            fine_window,
                            shard_bin,
                            lane,
                            (rank_above, rank_count),
                        )
                    bucket = (
                        bucket_prefix,
                        bucket_above,
                        bucket_end,
                        bucket_shift,
                        bucket_size,
                    )
                    ranks = (rank_above, rank_count)
                    Bucket.write_bounds(
                        s.batch_bounds, token, bucket, ranks, lane, SHARD
                    )

    @cute.jit
    def window_scan(
        self, p, s, ids, rd, window, window_end, use_lists, largest, inverted
    ):
        """One stash window: per (token, round) above/inside counts of the hot rounds,
        list or count the inside keys, track the inside key range.
        """
        (position, _, _, _, _), (half, key_warp, key_lane, lanes_below) = (
            rd.launch,
            rd.lanes,
        )
        (bound_long, bound_digit, bound_shift), kept = rd.bounds, rd.kept
        bid, warp, lane = ids.bid, ids.warp, ids.lane
        batch_counts, batch_cands, keep_words = (
            s.batch_counts,
            s.batch_cands,
            s.keep_words,
        )
        for t in cutlass.range_constexpr(2):
            token_0 = 2 * t + half
            probe_round = window + lane
            probe_hi = Uint32(0)
            if probe_round < window_end:
                probe_hi = s.unit_max[Rounds.cell(token_0, probe_round, key_warp)]
            hot = (
                Int32(bound_long[t] != 0)
                & Int32(probe_round < window_end)
                & Int32(bound_digit[t] <= Int32(probe_hi >> bound_shift[t]))
            )
            if hot == 0 and probe_round < window_end:
                batch_counts[Rounds.cell(token_0, probe_round, key_warp)] = 0
                batch_cands[Rounds.window(token_0, lane, key_warp)] = 0
            pending = ballot(hot != 0)
            while pending != 0:
                rnd = window + ffs(pending) - 1
                pending = pending & pending - 1
                index = bid * 128 + key_lane + rnd * ROUND_KEYS
                valid = bound_long[t] & Int32(
                    index
                    < Keys.shard_live(
                        self.rows(p, position + token_0 + 1, token_0), self.SHARD
                    )
                )
                key_0 = KeyScore.key(
                    as_u32(s.stash[Stash.row(token_0, rnd, window) + key_lane])
                )
                bucket = Int32(key_0 >> bound_shift[t])
                above = valid & Int32(bucket > bound_digit[t])
                inside = valid & Int32(bucket == bound_digit[t])
                keep_it = above | inside
                count_above = Int32(0)
                count_inside = Int32(0)
                if cute.arch.vote_any_sync(keep_it != 0):
                    _vote_2 = ballot(above != 0)
                    _vote_3 = ballot(inside != 0)
                    count_above = popc(_vote_2)
                    count_inside = popc(_vote_3)
                    if use_lists != 0:
                        kept_count = kept[t]
                        if keep_it != 0:
                            entry_index = kept_count + popc(
                                (_vote_2 | _vote_3) & lanes_below
                            )
                            if entry_index < KEEP_CAP:
                                entry_word = KeepList.word(warp * 2 + t, entry_index)
                                keep_words[entry_word] = Uint32(index)
                                keep_words[entry_word + 1] = key_0
                        kept[t] = kept[t] + popc(_vote_2 | _vote_3)
                    kept_u = Uint32(inside)
                    largest[t] = cutlass.max(largest[t], key_0 * kept_u)
                    inverted[t] = cutlass.max(
                        inverted[t], (key_0 ^ Uint32(4294967295)) * kept_u
                    )
                if lane == 0:
                    batch_counts[Rounds.cell(token_0, rnd, key_warp)] = count_above
                    batch_cands[
                        (token_0 * WINDOW_ROUNDS + rnd - window) * KEY_WARPS + key_warp
                    ] = count_inside

    @cute.jit
    def window_reserve(self, p, s, ids, rd, use_lists, window, window_end):
        """Window pass without lists: reserve each (token, round) block's candidate
        slots in the shard's candidate area and push the window's bucket candidates
        there.
        """
        position, _, _, parity, (_, shard_bases, near_bases) = rd.launch
        half, key_warp, key_lane, lanes_below = rd.lanes
        bid, warp, lane = ids.bid, ids.warp, ids.lane
        SHARD = self.SHARD
        batch_first = s.batch_first
        if use_lists == 0:
            if warp < 4:
                row = warp * WINDOW_ROUNDS * KEY_WARPS
                items = (window_end - window) * 4
                item = lane * 2
                count_low = Int32(0)
                count_high = Int32(0)
                if item < items:
                    count_low = s.batch_cands[row + item]
                if items > item + 1:
                    count_high = s.batch_cands[row + item + 1]
                pair_count = count_low + count_high
                reserved = rt.Scan.up(pair_count, lane)
                total = cute.arch.shuffle_sync(reserved, 31)
                base = Uint32(0)
                if lane == 0 and total != 0:
                    base = rt.Memory.atomic_add_u32(
                        p.SelectState, StateWords.bucket(parity, warp) + 1, total
                    )
                start = cute.arch.shuffle_sync(Int32(base), 0) + reserved - pair_count
                if item < items:
                    batch_first[row + item] = start
                if items > item + 1:
                    batch_first[row + item + 1] = start + count_low
            rt.bar_sync(8, 256)
            for rnd in range(window, window_end):
                index = bid * 128 + key_lane + rnd * ROUND_KEYS
                for t in cutlass.range_constexpr(2):
                    token_3_0 = 2 * t + half
                    live = self.rows(p, position + token_3_0 + 1, token_3_0)
                    valid = (
                        Int32(token_3_0 < SEL_TOKENS)
                        & Int32(live > TOPK)
                        & Int32(index < Keys.shard_live(live, SHARD))
                    )
                    key_4 = KeyScore.key(
                        as_u32(s.stash[Stash.row(token_3_0, rnd, window) + key_lane])
                    )
                    bucket = Int32(key_4 >> Uint32(s.batch_bounds[token_3_0 * 8 + 3]))
                    digit = s.batch_bounds[token_3_0 * 8]
                    inside = valid & Int32(bucket == digit)
                    _vote_2 = ballot(inside != 0)
                    slot = s.batch_first[
                        (token_3_0 * WINDOW_ROUNDS + rnd - window) * KEY_WARPS
                        + key_warp
                    ] + popc(_vote_2 & lanes_below)
                    if inside != 0:  # noqa: SIM102
                        if (
                            s.batch_bounds[token_3_0 * 8 + 2] == 0
                            and s.batch_bounds[token_3_0 * 8 + 4] <= CAND_CAP
                        ):
                            offset = (
                                (parity * 4 + token_3_0) * CAND_CAP
                                + (s.batch_bounds[token_3_0 * 8 + 5] + slot)
                            ) * 16
                            BucketMailbox.push_candidate(
                                near_bases,
                                s.near_meta[token_3_0],
                                shard_bases,
                                offset,
                                Keys.shard_id(index, SHARD),
                                key_4,
                                s.epoch_slot[0],
                            )

    @cute.jit
    def keep_list_stage(
        self,
        p,
        s,
        ids,
        rd,
        usable,
        rounds_0,
        warm_round,
        attempt_live,
        use_lists,
        slow_rounds,
    ):
        """Lists usable: count bucket candidates per warp, reserve mailbox slots, push
        them, emit keys above the bucket (direct mode), count the prelist.
        """
        position, epoch, direct_now, parity, (sb, shard_bases, near_bases) = rd.launch
        tid, warp, lane = ids.tid, ids.warp, ids.lane
        SHARD = self.SHARD
        keep_meta, near_meta = s.keep_meta, s.near_meta
        if usable != 0:
            FULL = True
            key_warp_0 = cmod(warp, 4)
            half_1 = cdiv(warp, 4)
            lanes_below_0 = (Uint32(1) << Uint32(lane)) - 1
            round_live = Int32(1)
            if cutlass.const_expr(FULL):
                round_live = Int32(s.near_meta[NearMeta.WARM] == 0)
            prelist_counting = s.keep_meta[KeepMeta.COUNTING]
            regs_index = rmem_fill(2, Int32, 0)
            regs_key = rmem_fill(2, Uint32, 0)
            regs_inside = rmem_fill(2, Int32, 0)
            regs_ballot = rmem_fill(2, Uint32, 0)
            for t in cutlass.range_constexpr(2):
                token_3 = 2 * t + half_1
                inside_total = Int32(0)
                above_total = Int32(0)
                if token_3 < SEL_TOKENS:
                    length = cutlass.min(s.keep_meta[warp * 2 + t], KEEP_CAP)
                    prefix = s.batch_bounds[token_3 * 8]
                    shift = s.batch_bounds[token_3 * 8 + 3]
                    for base in cutlass.range(0, length, 32, unroll=1):
                        present, index, key_word = KeepList.entry(
                            s.keep_words, warp * 2 + t, length, base, lane
                        )
                        inside = present & Int32(
                            Int32(key_word >> Uint32(shift)) == prefix
                        )
                        _vote_0 = ballot(inside != 0)
                        inside_total = inside_total + popc(_vote_0)
                        regs_index[t] = index
                        regs_key[t] = key_word
                        regs_inside[t] = inside
                        regs_ballot[t] = _vote_0
                        if cutlass.const_expr(FULL):
                            fast_above = present & Int32(
                                prefix < Int32(key_word >> Uint32(shift))
                            )
                            above_total = above_total + popc(ballot(fast_above != 0))
                    if lane == 0:
                        keep_meta[KeepMeta.INSIDE + token_3 * 4 + key_warp_0] = (
                            inside_total
                        )
                        if cutlass.const_expr(FULL):
                            near_meta[NearMeta.ABOVE + token_3 * 4 + key_warp_0] = (
                                above_total
                            )
                            near_meta[NearMeta.INSIDE + token_3 * 4 + key_warp_0] = (
                                inside_total
                            )
            rt.bar_sync(8, 256)
            fast_first = Uint32(0)
            if cutlass.const_expr(FULL):  # noqa: SIM102
                if tid < SEL_TOKENS:  # noqa: SIM102
                    if s.near_meta[tid] != 0:
                        fast_reserve = Int32(0)
                        for warp_0 in cutlass.range_constexpr(4):
                            fast_reserve = (
                                fast_reserve
                                + s.near_meta[NearMeta.ABOVE + tid * 4 + warp_0]
                            )
                        if fast_reserve != 0 and round_live != 0:
                            fast_first = rt.Memory.atomic_add_u32(
                                p.SelectState,
                                StateWords.refine_word(tid, StateWords.HEADER)
                                + parity * 8
                                + 5,
                                fast_reserve,
                            )
            if tid < SEL_TOKENS:  # noqa: SIM102
                if self.rows(p, position + tid + 1, tid) > TOPK:
                    total = Int32(0)
                    for warp_0 in cutlass.range_constexpr(4):
                        total = total + s.keep_meta[KeepMeta.INSIDE + tid * 4 + warp_0]
                    base_slot = Uint32(0)
                    if total != 0 and round_live != 0:
                        base_slot = rt.Memory.atomic_add_u32(
                            p.SelectState, StateWords.bucket(parity, tid) + 1, total
                        )
                    running = Int32(base_slot)
                    for warp_0 in cutlass.range_constexpr(4):
                        keep_meta[KeepMeta.FIRST + tid * 4 + warp_0] = running
                        running = (
                            running + s.keep_meta[KeepMeta.INSIDE + tid * 4 + warp_0]
                        )
            if cutlass.const_expr(FULL):  # noqa: SIM102
                if tid < SEL_TOKENS:
                    near_meta[NearMeta.RESERVED + tid] = Int32(fast_first)
            rt.bar_sync(8, 256)
            if round_live != 0:
                for t in cutlass.range_constexpr(2):
                    token_0 = 2 * t + half_1
                    if token_0 < SEL_TOKENS:
                        length = cutlass.min(s.keep_meta[warp * 2 + t], KEEP_CAP)
                        prefix = s.batch_bounds[token_0 * 8]
                        shift = s.batch_bounds[token_0 * 8 + 3]
                        stored = s.keep_meta[KeepMeta.FIRST + token_0 * 4 + key_warp_0]
                        for base in cutlass.range(0, length, 32, unroll=1):
                            regs_entry_index = regs_index[t]
                            regs_entry_key = regs_key[t]
                            regs_entry_inside = regs_inside[t]
                            regs_entry_ballot = regs_ballot[t]
                            if length > 32:
                                present, index, key_word = KeepList.entry(
                                    s.keep_words, warp * 2 + t, length, base, lane
                                )
                                regs_entry_index = index
                                regs_entry_key = key_word
                                regs_entry_inside = present & Int32(
                                    Int32(regs_entry_key >> Uint32(shift)) == prefix
                                )
                                regs_entry_ballot = ballot(regs_entry_inside != 0)
                            if regs_entry_inside != 0:
                                regs_slot = stored + popc(
                                    regs_entry_ballot & lanes_below_0
                                )
                                if (
                                    s.batch_bounds[token_0 * 8 + 2] == 0
                                    and s.batch_bounds[token_0 * 8 + 4] <= CAND_CAP
                                ):
                                    offset = (
                                        (parity * 4 + token_0) * CAND_CAP
                                        + (s.batch_bounds[token_0 * 8 + 5] + regs_slot)
                                    ) * 16
                                    BucketMailbox.push_candidate(
                                        near_bases if FULL else (),
                                        s.near_meta[token_0],
                                        shard_bases,
                                        offset,
                                        Keys.shard_id(regs_entry_index, SHARD),
                                        regs_entry_key,
                                        s.epoch_slot[0],
                                    )
                            stored = stored + popc(regs_entry_ballot)
            if round_live != 0:
                if cutlass.const_expr(FULL):  # noqa: SIM102
                    if direct_now != 0:
                        self.first_above_slots(s, ids)
                        rt.bar_sync(8, 256)
                        others = tuple(
                            b for i, b in enumerate(shard_bases) if i != SHARD
                        )
                        self.emit_above(p, s, ids, epoch, ((sb,) + near_bases, others))
                        if tid == 0:
                            keep_meta[KeepMeta.EMITTED] = 1
                if prelist_counting != 0:
                    self.count_prelist(s, ids)
            attempt_live = Int32(0)
        else:
            use_lists = Int32(0)
            slow_rounds = rounds_0 * Int32(warm_round != 0)
        return attempt_live, use_lists, slow_rounds

    @cute.jit
    def count_prelist(self, s, ids):
        """Count each listed key above the bucket into its (token, round, key warp) cell
        of batch_counts.
        """
        warp, lane = ids.warp, ids.lane
        key_warp_0 = cmod(warp, 4)
        half_1 = cdiv(warp, 4)
        for t in cutlass.range_constexpr(2):
            count_token = 2 * t + half_1
            if count_token < SEL_TOKENS:
                count_length = cutlass.min(s.keep_meta[warp * 2 + t], KEEP_CAP)
                count_prefix = s.batch_bounds[count_token * 8]
                count_shift = s.batch_bounds[count_token * 8 + 3]
                for base in cutlass.range(0, count_length, 32, unroll=1):
                    present, index, key_word = KeepList.entry(
                        s.keep_words, warp * 2 + t, count_length, base, lane
                    )
                    if present != 0 and count_prefix < Int32(
                        key_word >> Uint32(count_shift)
                    ):
                        cute.arch.atomic_add(
                            s.batch_counts.iterator
                            + (count_token * MAX_ROUNDS + cdiv(index, ROUND_KEYS))
                            * KEY_WARPS
                            + key_warp_0,
                            Int32(1),
                        )

    @cute.jit
    def first_above_slots(self, s, ids):
        """Tid < 16 (token tid/4, key warp tid%4): the first id slot of the key warp's
        keys above the bucket.
        """
        tid = ids.tid
        near_meta = s.near_meta
        if tid < 16:
            prefix_token = cdiv(tid, 4)
            prefix_warp = cmod(tid, 4)
            first_above = (
                s.batch_bounds[prefix_token * 8 + 7]
                + s.near_meta[NearMeta.RESERVED + prefix_token]
            )
            for warp_0 in cutlass.range_constexpr(3):
                if prefix_warp > warp_0:
                    first_above = (
                        first_above
                        + s.near_meta[NearMeta.ABOVE + prefix_token * 4 + warp_0]
                    )
            near_meta[NearMeta.FIRST_ABOVE + tid] = first_above

    @cute.jit
    def emit_above(self, p, s, ids, epoch, groups):
        """Emit this warp's listed keys strictly above the bucket prefix (tokens with
        NearMeta DIRECT set): ids to SelectedIds from slot FIRST_ABOVE on, id records
        to every base of each group in turn.
        """
        warp, lane = ids.warp, ids.lane
        below = (Uint32(1) << Uint32(lane)) - 1
        for g in cutlass.range_constexpr(len(groups)):
            for t in cutlass.range_constexpr(2):
                token = 2 * t + cdiv(warp, 4)
                if token < SEL_TOKENS:  # noqa: SIM102
                    if s.near_meta[token] != 0:
                        length = cutlass.min(s.keep_meta[warp * 2 + t], KEEP_CAP)
                        prefix = s.batch_bounds[token * 8]
                        shift = Uint32(s.batch_bounds[token * 8 + 3])
                        mailbox = IdMailbox.row(epoch, token)
                        slot = s.near_meta[
                            NearMeta.FIRST_ABOVE + token * 4 + cmod(warp, 4)
                        ]
                        for base in cutlass.range(0, length, 32, unroll=1):
                            present, index, key = KeepList.entry(
                                s.keep_words, warp * 2 + t, length, base, lane
                            )
                            above = present & Int32(prefix < Int32(key >> shift))
                            vote = ballot(above != 0)
                            if above != 0:
                                dst = slot + popc(vote & below)
                                self.emit_id(
                                    p, groups[g], epoch, token, mailbox, dst, index
                                )
                            slot = slot + popc(vote)

    @cute.jit
    def rescore_foreign(self, p, s, ids, shard_full, position):
        """Shards that miss the 2048 bound: re-score the other shards' key blocks
        against this query (tcgen05) into Scores.
        """
        tid, bid, warp, lane = ids.tid, ids.bid, ids.warp, ids.lane
        SHARD = self.SHARD
        Scores = p.Scores
        if shard_full != 0:
            live_max = position + 4
            blocks = cdiv(live_max + 128 - 1, 128)
            foreign = blocks - cdiv(blocks - SHARD + 3 - 1, 3)
            used_phases = Int32(0)
            row = cmod(warp, 4) * 32 + lane
            pair = cdiv(warp, 4)
            for foreign_block in range(bid, foreign, SCORE_CTAS):
                other = cmod(foreign_block, 2)
                owner = other + Int32(other >= SHARD)
                block = (cdiv(foreign_block, 2) * 3 + owner) * 128
                for part in cutlass.range_constexpr(4):
                    chunk = tid + part * 256
                    tile_row = cdiv(chunk, 8)
                    rt.AsyncCopy.cp_async_cg16_pred(
                        Int32(live_max > block + tile_row),
                        s.rescore_keys8_addr
                        + tile_row * 128
                        + (cmod(chunk, 8) ^ cmod(tile_row, 8)) * 16,
                        ga(
                            p.IndexKeyCache,
                            Int64(p.BlockTable[(block + tile_row) >> 6]) * 4224
                            + cmod(block + tile_row, 64) * 64
                            + cmod(chunk, 8) * 8,
                        ),
                    )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                rt.bar_sync(8, 256)
                with rt.warp_leader(warp, 0):
                    rt.TensorCore.fence_after_thread_sync()
                    rt.TensorCore.umma_f8f6f4_k4(
                        (s.rescore_keys8_addr >> 4) & 0x3FFF,
                        (s.query_words_addr >> 4) & 0x3FFF,
                        s.taddr,
                        0,
                    )
                    rt.TensorCore.tcgen05_commit(s.mma_done)
                rescore_scale = Float32(0.0)
                if live_max > block + row:
                    rescore_scale = IndexScale.fp32(
                        cute.recast_tensor(p.IndexKeyCache, Float32)[
                            cdiv(
                                Int64(p.BlockTable[(block + row) >> 6]) * 4224
                                + cmod(block + row, 64) * 2
                                + 4096,
                                2,
                            )
                        ],
                        self.PACKED_SCALE,
                    )
                mbar_wait(
                    s.mma_done,
                    cmod(used_phases + cdiv(foreign_block - bid, SCORE_CTAS), 2),
                )
                rt.TensorCore.fence_after_thread_sync()
                _tmem_load_0 = rt.TensorCore.tmem_ld_32x32b_x64(
                    s.taddr + (cmod(warp, 4) * 32 << 16) + pair * 64
                )
                cute.arch.fence_view_async_tmem_load()
                rt.TensorCore.fence_before_thread_sync()
                for slot in cutlass.range_constexpr(2):
                    token = pair * 2 + slot
                    rescore_score = KeyScore.weighted(
                        _tmem_load_0, slot, s.head_weight, token, rescore_scale
                    )
                    if token < SEL_TOKENS and block + row < self.rows(
                        p, position + token + 1, token
                    ):
                        Scores[token * p.score_stride + block + row] = rescore_score
                rt.bar_sync(8, 256)

    @cute.jit
    def stash_scores(
        self, p, s, ids, rnd0, round_limit, SYNC: cutlass.Constexpr = True
    ):
        """bar; 8 predicated 16 B cp.async.cg of Scores rows (rnd0..rnd0+15) into the
        stash; commit; wait 0; bar.
        """
        tid, bid = ids.tid, ids.bid
        if cutlass.const_expr(SYNC):
            rt.bar_sync(8, 256)
        for part in cutlass.range_constexpr(8):
            chunk = tid + part * 256
            token = cdiv(chunk, 512)
            rnd = rnd0 + cmod(cdiv(chunk, 32), 16)
            column = cmod(chunk, 32) * 4
            r = rnd * ROUND_KEYS + bid * 128
            rt.AsyncCopy.cp_async_cg16_pred(
                Int32((token < SEL_TOKENS) & (rnd < round_limit)),
                s.stash_addr + chunk * 16,
                ga(
                    p.Scores,
                    token * p.score_stride + Keys.shard_id(r, self.SHARD) + column,
                ),
            )
        cute.arch.cp_async_commit_group()
        if cutlass.const_expr(SYNC):
            cute.arch.cp_async_wait_group(0)
            rt.bar_sync(8, 256)

    @cute.jit
    def emit_stage(self, p, s, ids, position, long_context, epoch, direct_now, parity):
        """Compute warps, bid < 144: recycle the other parity's state, then emit this
        shard's selected ids (direct from the keep lists, or the slow path from the
        stash windows).
        """
        tid, bid, warp = ids.tid, ids.bid, ids.warp
        other = 1 - parity
        shard3, near5 = p.shard_bases, p.near_bases
        own_bases = (shard3[self.SHARD],) + near5
        other_bases = tuple(b for i, b in enumerate(shard3) if i != self.SHARD)
        all_bases = shard3 + near5
        self.sel_recycle(p, other, bid, tid, warp)
        if (long_context != 0) & (direct_now != 0):
            lists_used = s.keep_meta[KeepMeta.USABLE]
            direct_gate = Int32(1)
            direct_gate = Int32(s.keep_meta[KeepMeta.EMITTED] == 0)
            if direct_gate != 0:
                bases = (own_bases, other_bases, all_bases)
                self.emit_direct(p, s, ids, position, epoch, parity, lists_used, bases)
        if long_context != 0:
            slow_gate = Int32(1)
            near_meta = s.near_meta
            if tid == 0:
                all_direct = Int32(1)
                for exit_token in cutlass.range_constexpr(SEL_TOKENS):
                    if position + exit_token + 1 > TOPK:
                        all_direct = all_direct & s.near_meta[exit_token]
                near_meta[NearMeta.ALL_DIRECT] = all_direct
            rt.bar_sync(11, 384)
            slow_gate = Int32(s.near_meta[NearMeta.ALL_DIRECT] == 0)
            if slow_gate != 0:
                self.emit_slow(p, s, ids, position, epoch, all_bases)

    @cute.jit
    def emit_direct(self, p, s, ids, position, epoch, parity, lists_used, bases):
        """Direct mode (keep_meta EMITTED unset): per (token, key warp) above/inside
        counts, the reserved first slots, then the ids above the bucket from the keep
        lists or from the stash rounds.
        """
        tid, bid = ids.tid, ids.bid
        near_meta = s.near_meta
        if lists_used != 0:
            if tid < 16:
                list_token = cdiv(tid, 4)
                list_warp = cmod(tid, 4) + 4 * (cmod(list_token, 2))
                listed = cutlass.min(
                    s.keep_meta[list_warp * 2 + cdiv(list_token, 2)], KEEP_CAP
                )
                listed_inside = s.keep_meta[KeepMeta.INSIDE + tid]
                near_meta[NearMeta.ABOVE + tid] = listed - listed_inside
                near_meta[NearMeta.INSIDE + tid] = listed_inside
                if s.keep_meta[KeepMeta.COUNTING] != 0:
                    listed_above = Int32(0)
                    rounds = Keys.shard_rounds(position + 4, bid, self.SHARD)
                    for listed_round in cutlass.range(0, rounds, unroll=1):
                        listed_above = (
                            listed_above
                            + s.batch_counts[
                                Rounds.row(list_token, listed_round) + cmod(tid, 4)
                            ]
                        )
                    near_meta[NearMeta.ABOVE + tid] = listed_above
            rt.bar_sync(8, 256)
        if lists_used == 0:
            direct_rounds = Keys.shard_rounds(position + 4, bid, self.SHARD)
            if tid < 16:
                table_token = cdiv(tid, 4)
                table_warp = cmod(tid, 4)
                table_above = Int32(0)
                table_inside = Int32(0)
                for table_round in cutlass.range(0, direct_rounds, unroll=1):
                    table_above = (
                        table_above
                        + s.batch_counts[
                            Rounds.cell(table_token, table_round, table_warp)
                        ]
                    )
                for table_round in cutlass.range(0, direct_rounds, unroll=1):
                    table_inside = (
                        table_inside
                        + s.batch_cands[
                            Rounds.window(table_token, table_round, table_warp)
                        ]
                    )
                near_meta[NearMeta.ABOVE + tid] = table_above
                near_meta[NearMeta.INSIDE + tid] = table_inside
            rt.bar_sync(8, 256)
        if tid < SEL_TOKENS:
            reserved_first = Uint32(0)
            if s.near_meta[tid] != 0:
                reserve_above = Int32(0)
                for warp_0 in cutlass.range_constexpr(4):
                    reserve_above = (
                        reserve_above + s.near_meta[NearMeta.ABOVE + tid * 4 + warp_0]
                    )
                if reserve_above != 0:
                    reserve_word = (
                        StateWords.refine_word(tid, StateWords.HEADER) + parity * 8 + 5
                    )
                    reserved_first = cute.arch.atomic_add(
                        p.SelectState.iterator + reserve_word, Uint32(reserve_above)
                    )
            near_meta[NearMeta.RESERVED + tid] = Int32(reserved_first)
        rt.bar_sync(8, 256)
        self.first_above_slots(s, ids)
        rt.bar_sync(8, 256)
        if lists_used != 0:
            self.emit_above(p, s, ids, epoch, (bases[0], bases[1]))
        if lists_used == 0:
            self.emit_stash_above(p, s, ids, position, epoch, bases[2])

    @cute.jit
    def emit_stash_above(self, p, s, ids, position, epoch, all_bases):
        """Direct mode without keep lists: emit the stash keys above the bucket prefix,
        round by round.
        """
        tid, bid, warp, lane = ids.tid, ids.bid, ids.warp, ids.lane
        stash_rounds = Keys.shard_rounds(position + 4, bid, self.SHARD)
        if Keys.shard_live(position + 4, self.SHARD) > STREAM_KEYS:
            self.stash_scores(p, s, ids, 0, stash_rounds)
        stash_lane = cmod(tid, 128)
        stash_below = (Uint32(1) << Uint32(lane)) - 1
        for t in cutlass.range_constexpr(2):
            stash_token = 2 * t + cdiv(warp, 4)
            if stash_token < SEL_TOKENS:  # noqa: SIM102
                if s.near_meta[stash_token] != 0:
                    stash_live = Keys.shard_live(
                        self.rows(p, position + stash_token + 1, stash_token),
                        self.SHARD,
                    )
                    stash_prefix = s.batch_bounds[stash_token * 8]
                    stash_shift = Uint32(s.batch_bounds[stash_token * 8 + 3])
                    stash_mailbox = IdMailbox.row(epoch, stash_token)
                    stash_slot = s.near_meta[
                        NearMeta.FIRST_ABOVE + stash_token * 4 + cmod(warp, 4)
                    ]
                    for rnd in cutlass.range(0, stash_rounds, unroll=1):
                        index = bid * 128 + stash_lane + rnd * ROUND_KEYS
                        key_0 = KeyScore.key(
                            as_u32(s.stash[Stash.index(stash_token, rnd, stash_lane)])
                        )
                        stash_above = Int32(index < stash_live) & Int32(
                            stash_prefix < Int32(key_0 >> stash_shift)
                        )
                        _vote_0 = ballot(stash_above != 0)
                        if stash_above != 0:
                            dst = stash_slot + popc(_vote_0 & stash_below)
                            self.emit_id(
                                p,
                                all_bases,
                                epoch,
                                stash_token,
                                stash_mailbox,
                                dst,
                                index,
                            )
                        stash_slot = stash_slot + popc(_vote_0)

    @cute.jit
    def emit_slow(self, p, s, ids, position, epoch, all_bases):
        """Staged mode (rows not emitted directly): once the final CTAs published
        (threshold, limit) and this CTA's first slot per token (``emission``), emit
        the taken ids from the keep lists or from the stash windows.
        """
        tid, bid, warp, lane = ids.tid, ids.bid, ids.warp, ids.lane
        near_counts, near_offsets = s.near_counts, s.near_offsets
        rt.bar_sync(11, 384)
        lanes_below = (Uint32(1) << Uint32(lane)) - 1
        rounds_0 = Keys.shard_rounds(position + 4, bid, self.SHARD)
        if tid < SEL_TOKENS:
            near_offsets[tid] = Int32(
                s.emission[tid * StateWords.RECORD_ITEMS + 2] & 65535
            )
        rt.bar_sync(8, 256)
        slow_rounds = rounds_0
        if s.keep_meta[KeepMeta.USABLE] != 0:
            rounds_2 = Keys.shard_rounds(position + 4, bid, self.SHARD)
            for item in cutlass.range(tid, 912, 256, unroll=1):
                if (1 & Int32(s.near_meta[cdiv(item, 228)] == 0)) != 0:
                    near_counts[item] = 0
            rt.bar_sync(8, 256)
            self.keep_list_takes(p, s, ids, position, epoch, all_bases, False)
            rt.bar_sync(8, 256)
            if warp < 4:
                self.list_round_slots(s, ids, rounds_2)
            rt.bar_sync(8, 256)
            self.keep_list_takes(p, s, ids, position, epoch, all_bases, True)
            slow_rounds = 0
        for window in range(0, slow_rounds, WINDOW_ROUNDS):
            self.emit_window(
                p, s, ids, position, epoch, window, rounds_0, lanes_below, all_bases
            )

    @cute.jit
    def list_round_slots(self, s, ids, rounds_2):
        """Warp ``token`` (< 4): turn the per-(round, key warp) take counts into each
        round's first slot (round_first) and the per-key-warp offsets inside a round
        (near_offsets, near_prefix).
        """
        warp, lane = ids.warp, ids.lane
        near_offsets, round_first, near_prefix = (
            s.near_offsets,
            s.round_first,
            s.near_prefix,
        )
        lean_token = warp
        if (1 & Int32(s.near_meta[lean_token] == 0)) != 0:
            lean_low = 2 * lane
            lean_counts = cute.make_rmem_tensor(8, Int32)
            lean_totals = cute.make_rmem_tensor(2, Int32)
            for lean_half in cutlass.range_constexpr(2):
                lean_round = lean_low + lean_half
                lean_in_block = Int32(0)
                for warp_0 in cutlass.range_constexpr(4):
                    lean_counts[lean_half * 4 + warp_0] = 0
                    if lean_round < rounds_2:
                        lean_counts[lean_half * 4 + warp_0] = s.near_counts[
                            Rounds.cell(lean_token, lean_round, warp_0)
                        ]
                        near_offsets[Rounds.cell(lean_token, lean_round, warp_0)] = (
                            lean_in_block
                        )
                    lean_in_block = lean_in_block + lean_counts[lean_half * 4 + warp_0]
                lean_totals[lean_half] = lean_in_block
            lean_pair = lean_totals[0] + lean_totals[1]
            lean_inclusive = rt.Scan.up(lean_pair, lane)
            lean_first = (
                Int32(s.emission[lean_token * StateWords.RECORD_ITEMS + 2] & 65535)
                + lean_inclusive
                - lean_pair
            )
            if lean_low < rounds_2:
                round_first[lean_token * MAX_ROUNDS + lean_low] = lean_first
            if rounds_2 > lean_low + 1:
                round_first[lean_token * MAX_ROUNDS + lean_low + 1] = (
                    lean_first + lean_totals[0]
                )
            for warp_0 in cutlass.range_constexpr(4):
                lean_warp_pair = lean_counts[warp_0] + lean_counts[4 + warp_0]
                lean_earlier = rt.Scan.up(lean_warp_pair, lane) - lean_warp_pair
                if lean_low < rounds_2:
                    near_prefix[Rounds.cell(lean_token, lean_low, warp_0)] = (
                        lean_earlier
                    )
                if rounds_2 > lean_low + 1:
                    near_prefix[
                        (lean_token * MAX_ROUNDS + lean_low + 1) * KEY_WARPS + warp_0
                    ] = lean_earlier + lean_counts[warp_0]

    @cute.jit
    def emit_window(
        self, p, s, ids, position, epoch, window, rounds_0, lanes_below, all_bases
    ):
        """One 16-round stash window of the staged emission: refill the stash if a token
        needs it, count the taken keys per (token, round), carry the round slots,
        then emit the ids (two units per round, half the parts each).
        """
        tid, bid, warp, lane = ids.tid, ids.bid, ids.warp, ids.lane
        SHARD = self.SHARD
        coarse_scan, near_counts = s.coarse_scan, s.near_counts
        near_offsets, round_first = s.near_offsets, s.round_first
        if Keys.shard_live(position + 4, SHARD) > STREAM_KEYS:
            if tid == 0:
                coarse_scan[Coarse.STASH_NEEDED] = 0
            rt.bar_sync(8, 256)
            if tid < 64:
                check_token = cdiv(tid, 16)
                check_round = window + cmod(tid, 16)
                check_live = self.rows(p, position + check_token + 1, check_token)
                check_rounds = Keys.shard_rounds(check_live, bid, SHARD)
                if (  # noqa: SIM102
                    (check_live > TOPK)
                    & (check_round < check_rounds)
                    & ((1 & Int32(s.near_meta[check_token] == 0)) != 0)
                ):
                    if s.emission[check_token * StateWords.RECORD_ITEMS + 2] >> 16 != 0:
                        coarse_scan[Coarse.STASH_NEEDED] = 1
            rt.bar_sync(8, 256)
            if s.coarse_scan[Coarse.STASH_NEEDED] != 0:
                self.stash_scores(p, s, ids, window, rounds_0)
        span = cutlass.min(rounds_0 - window, WINDOW_ROUNDS)
        for count_unit in range(warp, 4 * span, 8):
            count_token = cdiv(count_unit, span)
            count_rnd = window + cmod(count_unit, span)
            count_live = self.rows(p, position + count_token + 1, count_token)
            round_count = Int32(0)
            count_record = s.emission[count_token * StateWords.RECORD_ITEMS + 2]
            count_rounds = Keys.shard_rounds(count_live, bid, SHARD)
            if (
                (count_live > TOPK)
                & (count_rnd < count_rounds)
                & ((1 & Int32(s.near_meta[count_token] == 0)) != 0)
                & (count_record >> 16 != 0)
            ):
                count_threshold = s.emission[count_token * StateWords.RECORD_ITEMS]
                count_limit = s.emission[count_token * StateWords.RECORD_ITEMS + 1]
                count_rows = Keys.shard_live(count_live, SHARD)
                for count_part in cutlass.range_constexpr(4):
                    count_index = (
                        bid * 128 + count_rnd * ROUND_KEYS + count_part * 32 + lane
                    )
                    count_key = KeyScore.key(
                        as_u32(
                            s.stash[
                                Stash.row(count_token, count_rnd, window)
                                + (count_part * 32 + lane)
                            ]
                        )
                    )
                    take = Keys.exact_take(
                        count_key, count_threshold, count_limit, count_index, SHARD
                    )
                    count_take = Int32(count_index < count_rows) & take
                    round_count = round_count + popc(ballot(count_take != 0))
            if lane == 0:
                near_counts[Rounds.cell(count_token, count_rnd, 0)] = round_count
        rt.bar_sync(8, 256)
        if tid < SEL_TOKENS:
            carry = s.near_offsets[tid]
            for carry_rnd in range(window, window + span):
                round_first[tid * MAX_ROUNDS + carry_rnd] = carry
                carry = carry + s.near_counts[Rounds.cell(tid, carry_rnd, 0)]
            near_offsets[tid] = carry
        rt.bar_sync(8, 256)
        for unit in range(warp, 4 * span * 2, 8):
            tok = cdiv(unit, span * 2)
            emit_rnd = window + cmod(cdiv(unit, 2), span)
            unit_half = cmod(unit, 2)
            live = self.rows(p, position + tok + 1, tok)
            block_record = (
                Uint32(s.round_first[tok * MAX_ROUNDS + emit_rnd])
                | Uint32(s.near_counts[Rounds.cell(tok, emit_rnd, 0)]) << 16
            )
            unit_rounds = Keys.shard_rounds(live, bid, SHARD)
            if (
                (live > TOPK)
                & (emit_rnd < unit_rounds)
                & (block_record >> 16 != 0)
                & ((1 & Int32(s.near_meta[tok] == 0)) != 0)
            ):
                threshold = s.emission[tok * StateWords.RECORD_ITEMS]
                limit = s.emission[tok * StateWords.RECORD_ITEMS + 1]
                win_mailbox = IdMailbox.row(epoch, tok)
                out_slot = Int32(block_record & 65535)
                live_limit = Keys.shard_live(live, SHARD)
                for part in cutlass.range_constexpr(4):
                    key_index = bid * 128 + emit_rnd * ROUND_KEYS + part * 32 + lane
                    win_key = KeyScore.key(
                        as_u32(
                            s.stash[
                                Stash.row(tok, emit_rnd, window) + (part * 32 + lane)
                            ]
                        )
                    )
                    take_1 = Int32(key_index < live_limit) & Keys.exact_take(
                        win_key, threshold, limit, key_index, SHARD
                    )
                    take_vote = ballot(take_1 != 0)
                    if (take_1 != 0) & (unit_half == cdiv(part, 2)):
                        own_slot = out_slot + popc(take_vote & lanes_below)
                        self.emit_id(
                            p, all_bases, epoch, tok, win_mailbox, own_slot, key_index
                        )
                    out_slot = out_slot + popc(take_vote)

    @cute.jit
    def keep_list_takes(
        self, p, s, ids, position, epoch, all_bases, EMIT: cutlass.Constexpr
    ):
        """Walk this warp's two keep lists (tokens warp/4 and warp/4 + 2, long rows not
        served directly) for keys the final threshold/limit takes.
        """
        warp, lane = ids.warp, ids.lane
        key_warp = cmod(warp, 4)
        below = (Uint32(1) << Uint32(lane)) - 1
        for pair in cutlass.range(0, 2, unroll=1):
            token = pair * 2 + cdiv(warp, 4)
            if token < SEL_TOKENS:  # noqa: SIM102
                if (self.rows(p, position + token + 1, token) > TOPK) & (
                    (1 & Int32(s.near_meta[token] == 0)) != 0
                ):
                    threshold = s.emission[token * StateWords.RECORD_ITEMS]
                    limit = s.emission[token * StateWords.RECORD_ITEMS + 1]
                    length = cutlass.min(s.keep_meta[warp * 2 + pair], KEEP_CAP)
                    seen = Int32(0)
                    for base in cutlass.range(0, length, 32, unroll=1):
                        present, index, key = KeepList.entry(
                            s.keep_words, warp * 2 + pair, length, base, lane
                        )
                        take = present & Keys.exact_take(
                            key, threshold, limit, index, self.SHARD
                        )
                        table = (
                            token * MAX_ROUNDS + cdiv(index, ROUND_KEYS)
                        ) * KEY_WARPS + key_warp
                        if cutlass.const_expr(not EMIT):
                            if take != 0:
                                cute.arch.atomic_add(
                                    s.near_counts.iterator + table, Int32(1)
                                )
                        else:
                            vote = ballot(take != 0)
                            if take != 0:
                                own_slot = (
                                    s.round_first[
                                        token * MAX_ROUNDS + cdiv(index, ROUND_KEYS)
                                    ]
                                    + s.near_offsets[table]
                                    + seen
                                    + popc(vote & below)
                                    - s.near_prefix[table]
                                )
                                mailbox = IdMailbox.row(epoch, token)
                                self.emit_id(
                                    p, all_bases, epoch, token, mailbox, own_slot, index
                                )
                            seen = seen + popc(vote)

    @cute.jit
    def final_stage(self, p, s, ids, warp, position, epoch, direct_now):
        """Compute warps, last stage: final top-2048 selection on bid 144..147, the
        reduce (pull/r0), the residual copy and the TMEM release.
        """
        role_ids = rt.ThreadIds(ids.tid, ids.bid, warp, ids.lane)
        tid, bid = role_ids.tid, role_ids.bid
        if (bid >= SCORE_CTAS) & (bid < CTAS):
            cute.arch.griddepcontrol_launch_dependents()
            row_token = bid - SCORE_CTAS
            sb = p.shard_base(self.SHARD)
            if self.rows(p, position + row_token + 1, row_token) > TOPK:
                early_generation = s.epoch_slot[0] + 1
                early_parity = Int32(cmod(early_generation - 1, 2))
                if warp == 0:
                    self.poll_early_bucket(
                        s, role_ids, sb, row_token, early_generation, early_parity
                    )
                rt.bar_sync(8, 256)
                staged_above = Int32(s.refine_in[0])
                staged_candidates = Int32(s.refine_in[1])
                full_scan = Int32(staged_candidates > CAND_CAP)
                if tid == 0:
                    cute.arch.atomic_add(
                        p.SelectState.iterator + StateWords.STAGED_RUNS, Uint32(1)
                    )
                    if full_scan != 0:
                        cute.arch.atomic_add(
                            p.SelectState.iterator + StateWords.FULL_SCANS, Uint32(1)
                        )
                if full_scan == 0:
                    staged = (row_token, staged_above, staged_candidates)
                    self.select_staged(
                        p, s, role_ids, (position, epoch, direct_now, sb), staged
                    )
                else:
                    self.radix_select(p, s, role_ids, row_token, sb, epoch)
            self.sel_copy_ids(p, sb, epoch, row_token, tid)
        if bid < 0:
            self.sel_reduce(p, s, role_ids, warp, epoch)
        Final = p.Final
        if bid * 256 < 24576:
            Final[bid * 256 + tid] = p.X[bid * 256 + tid]
        if warp == 0:
            rt.TensorCore.fence_after_thread_sync()
            rt.TensorCore.tmem_dealloc(
                rt.Memory.ld_volatile_shared_u32(s.base + SelectorSmem.TMEM_HOLD), 512
            )

    @cute.jit
    def poll_early_bucket(self, s, ids, sb, row_token, early_generation, early_parity):
        """Warp 0: poll this shard's early bucket / slot headers (words 0, 1 of the row)
        and stage them in shard_in (0: above, 1: size, 4: first slot, 5: own count;
        2, 3, 7 cleared).
        """
        lane = ids.lane
        shard_in = s.shard_in
        early_addr = SelectionMailbox.word_addr(
            sb, early_parity, self.SHARD, row_token, lane
        )
        early_word = rt.Poll.wait(
            [early_addr], early_generation, [lane < 2], backoff=True
        )[0]
        early_low = Uint32(early_word)
        if lane < 2:
            shard_in[4 * lane] = early_low & 65535
            shard_in[4 * lane + 1] = early_low >> 16
        if (lane == 2) | (lane == 3) | (lane == 7):
            shard_in[lane] = Uint32(0)

    @cute.jit
    def select_staged(self, p, s, ids, launch, staged):
        """The staged candidates fit the cache: gather them from the shard mailbox into
        refine_words, find the threshold key (two 10-bit histogram passes) and the
        tie limit, stage and publish the own ids, count the taken keys per CTA and
        publish (threshold, limit) and the next step's seed.
        """
        (
            (position, epoch, direct_now, sb),
            (row_token, staged_above, staged_candidates),
        ) = (
            launch,
            staged,
        )
        tid, warp, lane = ids.tid, ids.warp, ids.lane
        shard_in, refine_words = s.shard_in, s.refine_words
        _min_0 = cutlass.min(staged_candidates, CAND_CAP)
        key_blocks = cdiv(position + row_token + 1 + 128 - 1, 128)
        cache_first = Refine.CACHE
        bucket_max = s.refine_in[2]
        smallest = s.refine_in[3] ^ Uint32(4294967295)
        for part in cutlass.range_constexpr(4):
            refine_words[Refine.HIST + part * 256 + tid] = Uint32(0)
        rt.bar_sync(8, 256)
        candidate_generation = s.epoch_slot[0] + 1
        candidate_parity = Int32(cmod(candidate_generation - 1, 2))
        largest_key = Uint32(0)
        inverted_key = Uint32(0)
        for cbase in range(0, _min_0, 2048):
            slots = [cbase + item * 256 + tid for item in range(8)]
            records = [
                BucketMailbox.candidate(candidate_parity, row_token, slot) + half * 8
                for slot in slots
                for half in range(2)
            ]
            live = [slot < _min_0 for slot in slots for _ in range(2)]
            addrs = [sb + Int64(BucketMailbox.BASE + record) for record in records]
            words = rt.Poll.wait(addrs, candidate_generation, live, backoff=True)
            for item in cutlass.range_constexpr(8):
                store_slot = cbase + item * 256 + tid
                if store_slot < _min_0:
                    candidate_key = Uint32(words[2 * item + 1])
                    rt.Memory.st_shared_v2(
                        s.refine_words_addr + (cache_first + store_slot * 2) * 4,
                        Uint32(words[2 * item]),
                        candidate_key,
                    )
                    cute.arch.atomic_add(
                        refine_words.iterator
                        + (Refine.HIST + Int32(candidate_key >> 10 & 1023)),
                        Uint32(1),
                    )
                    largest_key = cutlass.max(largest_key, candidate_key)
                    inverted_key = cutlass.max(
                        inverted_key, candidate_key ^ Uint32(4294967295)
                    )
        largest_key = redux_max_u32(largest_key)
        inverted_key = redux_max_u32(inverted_key)
        if lane == 0:
            cute.arch.atomic_max(shard_in.iterator + 2, largest_key)
            cute.arch.atomic_max(shard_in.iterator + 3, inverted_key)
        rt.bar_sync(8, 256)
        bucket_max = s.shard_in[2]
        smallest = s.shard_in[3] ^ Uint32(4294967295)
        threshold = bucket_max
        remaining_need = TOPK - staged_above
        ties = _min_0
        cache = rt.Memory.smem_view_n(
            s.refine_words_addr, cache_first * 4, Uint32, 1 << 16
        )
        hist = rt.Memory.smem_view_n(s.refine_words_addr, Refine.HIST * 4, Uint32, 1024)
        if bucket_max != smallest:
            threshold, remaining_need, ties = rt.Selection.threshold(
                cache,
                hist,
                None,
                s.control,
                s.warp_eq,
                _min_0,
                bucket_max,
                remaining_need,
                ties,
                tid,
                warp,
                lane,
            )
        limit = Uint32(4294967295)
        if remaining_need < ties:
            limit = rt.Selection.tie_limit(
                cache,
                s.tie_blocks,
                None,
                s.tie_bits,
                None,
                s.control,
                s.warp_eq,
                _min_0,
                key_blocks,
                threshold,
                remaining_need,
                tid,
                warp,
                lane,
            )
        self.stage_own_ids(
            p, s, ids, direct_now, cache_first, threshold, limit, epoch, row_token
        )
        rt.bar_sync(11, 384)
        mbar_wait(s.x_full, 0)
        own_first = Int32(s.shard_in[4])
        own_end = own_first + Int32(s.shard_in[5])
        lower_taken = Uint32(0)
        for index in range(tid, _min_0, 256):
            own_id = s.refine_words[cache_first + index * 2]
            take = Keys.takes(
                own_id, s.refine_words[cache_first + index * 2 + 1], threshold, limit
            )
            if (take != 0) & (own_first <= index) & (own_end > index):
                cute.arch.atomic_add(
                    refine_words.iterator
                    + Int32(cmod(cdiv(cdiv(own_id, 128), 3), SCORE_CTAS)),
                    Uint32(1),
                )
            lower_taken = lower_taken + Uint32(
                Int32(take != 0) & Int32(own_first > index)
            )
        lower_taken = redux_add_u32(lower_taken)
        if (lane == 0) & (lower_taken != 0):
            cute.arch.atomic_add(shard_in.iterator + 7, lower_taken)
        rt.bar_sync(8, 256)
        self.sel_publish(p, s, ids, warp, epoch, row_token, threshold, limit)
        if tid == 0:
            self.seed_update(
                p, s, sb, epoch, row_token, threshold, Int32(s.refine_in[1]), True
            )

    @cute.jit
    def stage_own_ids(
        self, p, s, ids, direct_now, cache_first, threshold, limit, epoch, row_token
    ):
        """Direct mode: stage this shard's taken candidate ids in order and publish them
        to every selection mailbox.
        """
        tid, warp, lane = ids.tid, ids.warp, ids.lane
        refine_words, warp_eq, SelectedIds = s.refine_words, s.warp_eq, p.SelectedIds
        if direct_now != 0:
            own_first = Int32(s.shard_in[4])
            own_count = Int32(s.shard_in[5])
            warp_span = cdiv(own_count + 256 - 1, 256) * 32
            warp_begin = own_first + cutlass.min(warp * warp_span, own_count)
            warp_end = own_first + cutlass.min(warp * warp_span + warp_span, own_count)
            chunk_below = (Uint32(1) << Uint32(lane)) - 1
            warp_chosen = Int32(0)
            for count_chunk in cutlass.range(warp_begin, warp_end, 32, unroll=4):
                count_index = count_chunk + lane
                ci = cutlass.min(count_index, warp_end - 1)
                count_take = Int32(count_index < warp_end) & Keys.takes(
                    s.refine_words[cache_first + ci * 2],
                    s.refine_words[cache_first + ci * 2 + 1],
                    threshold,
                    limit,
                )
                warp_chosen = warp_chosen + Int32(popc(ballot(count_take != 0)))
            lower_span = cdiv(own_first + 256 - 1, 256) * 32
            lower_begin = cutlass.min(warp * lower_span, own_first)
            lower_end = cutlass.min(warp * lower_span + lower_span, own_first)
            lower_chosen_warp = Int32(0)
            if cutlass.const_expr(self.SHARD == 1):
                for lower_chunk in cutlass.range(lower_begin, lower_end, 32, unroll=4):
                    lower_index = lower_chunk + lane
                    li = cutlass.min(lower_index, lower_end - 1)
                    lower_take = Int32(lower_index < lower_end) & Keys.takes(
                        s.refine_words[cache_first + li * 2],
                        s.refine_words[cache_first + li * 2 + 1],
                        threshold,
                        limit,
                    )
                    lower_chosen_warp = lower_chosen_warp + Int32(
                        popc(ballot(lower_take != 0))
                    )
            if lane == 0:
                if cutlass.const_expr(self.SHARD != 1):
                    warp_eq[warp] = warp_chosen
                else:
                    warp_eq[warp] = warp_chosen + lower_chosen_warp * 65536
            rt.bar_sync(8, 256)
            lower_chosen = Int32(0)
            if cutlass.const_expr(self.SHARD != 1):
                warps_before, own_chosen = rt.Scan.block_prefix(
                    s.warp_eq, warp, Int32(0)
                )
            else:
                warps_before = Int32(0)
                own_chosen = Int32(0)
                for warp_0 in cutlass.range_constexpr(8):
                    warp_word = s.warp_eq[warp_0]
                    warp_count = warp_word & 65535
                    if warp > warp_0:
                        warps_before = warps_before + warp_count
                    own_chosen = own_chosen + warp_count
                    lower_chosen = lower_chosen + (warp_word >> 16)
            rt.bar_sync(8, 256)
            if cutlass.const_expr(self.SHARD == 0):
                emit_slot = warps_before + Int32(s.shard_in[0])
            elif cutlass.const_expr(self.SHARD == 1):
                emit_slot = warps_before + Int32(s.shard_in[0]) + lower_chosen
            else:
                emit_slot = warps_before + (TOPK - own_chosen)
            emit_mailbox = IdMailbox.row(epoch, row_token)
            stage_slot = warps_before
            for stage_chunk in cutlass.range(warp_begin, warp_end, 32, unroll=2):
                stage_index = stage_chunk + lane
                si = cutlass.min(stage_index, warp_end - 1)
                stage_id = s.refine_words[cache_first + si * 2]
                stage_take = Int32(stage_index < warp_end) & Keys.takes(
                    stage_id, s.refine_words[cache_first + si * 2 + 1], threshold, limit
                )
                stage_vote = ballot(stage_take != 0)
                if stage_take != 0:
                    refine_words[
                        Refine.STAGED
                        + stage_slot
                        + Int32(popc(stage_vote & chunk_below))
                    ] = stage_id
                stage_slot = stage_slot + Int32(popc(stage_vote))
            rt.bar_sync(8, 256)
            staged_first = emit_slot - warps_before
            for staged_index in cutlass.range(tid, own_chosen, 256, unroll=1):
                staged_id = s.refine_words[Refine.STAGED + staged_index]
                SelectedIds[row_token * TOPK + (staged_first + staged_index)] = Int32(
                    staged_id
                )
                id_record = Uint64(epoch + 1) << 32 | Uint64(staged_id)
                record_off = Int64(emit_mailbox + (staged_first + staged_index)) * 8
                for dst in cutlass.range_constexpr(3):
                    st_volatile_u64(p.shard_base(dst) + record_off, id_record)

    @cute.jit
    def radix_select(self, p, s, ids, row_token, sb, epoch):
        """Too many candidates for the stage cache: 4-pass 8-bit radix select straight
        from Scores, then publish.
        """
        tid, warp, lane = ids.tid, ids.warp, ids.lane
        hist, control, refine_words = s.hist, s.control, s.refine_words
        shard_in, warp_eq, warp_sel = s.shard_in, s.warp_eq, s.warp_sel
        rt.bar_sync(11, 384)
        mbar_wait(s.x_full, 0)
        key_count = p.CurPos[0] + row_token + 1
        score_row = row_token * p.score_stride
        prefix = Uint32(0)
        need = Int32(TOPK)
        coarse_bucket = Int32(0)
        for radix_pass in cutlass.range(0, 4, unroll=1):
            hist[tid] = Uint32(0)
            rt.bar_sync(8, 256)
            for index in range(tid, key_count, 256):
                key_0 = KeyScore.key(ld_cg_b32(ga(p.Scores, score_row + index)))
                if radix_pass == 0:
                    cute.arch.atomic_add(
                        hist.iterator
                        + Int32(key_0 >> Uint32(24 - 8 * radix_pass) & 255),
                        Uint32(1),
                    )
                else:
                    if key_0 >> Uint32(24 - 8 * radix_pass + 8) == prefix:
                        cute.arch.atomic_add(
                            hist.iterator
                            + Int32(key_0 >> Uint32(24 - 8 * radix_pass) & 255),
                            Uint32(1),
                        )
            rt.bar_sync(8, 256)
            if warp == 0:
                top_bin = 255 - lane * 8
                counts = cute.make_rmem_tensor(8, Int32)
                own = Int32(0)
                for item in cutlass.range_constexpr(8):
                    counts[item] = Int32(s.hist[top_bin - item])
                    own = own + counts[item]
                radix_inclusive = rt.Scan.up(own, lane)
                if lane == ffs(ballot(radix_inclusive >= need)) - 1:
                    running = radix_inclusive - own
                    radix_found = Int32(0)
                    for item in cutlass.range_constexpr(8):
                        running = running + counts[item]
                        if (radix_found == 0) & (running >= need):
                            control[0] = top_bin - item
                            control[1] = running - counts[item]
                            radix_found = Int32(1)
            rt.bar_sync(8, 256)
            digit = s.control[0]
            above = s.control[1]
            prefix = prefix << 8 | Uint32(digit)
            need = need - above
            if radix_pass == 1:
                if tid == 0:
                    for nibble in cutlass.range_constexpr(16):
                        coarse_bucket = coarse_bucket + Int32(
                            s.hist[(digit & 240) + nibble]
                        )
                rt.bar_sync(8, 256)
        if tid == 0:
            self.seed_update(p, s, sb, epoch, row_token, prefix, coarse_bucket, False)
        lanes_below = (Uint32(1) << Uint32(lane)) - 1
        ties_seen = Int32(0)
        written = Int32(0)
        if tid < SCORE_CTAS:
            refine_words[tid] = Uint32(0)
        if tid < 2:
            shard_in[6 + tid] = Uint32(0)
        rt.bar_sync(8, 256)
        scan_base = Int32(0)
        scanning = Int32(1)
        while (scanning != 0) & (scan_base < key_count):
            index = scan_base + tid
            key = Uint32(0)
            if index < key_count:
                key = KeyScore.key(ld_cg_b32(ga(p.Scores, score_row + index)))
            valid = Int32(index < key_count)
            tie = valid & Int32(key == prefix)
            better = valid & Int32(key > prefix)
            tie_vote = ballot(tie != 0)
            if lane == 0:
                warp_eq[warp] = Int32(popc(tie_vote))
            rt.bar_sync(8, 256)
            ties_before, chunk_ties = rt.Scan.block_prefix(s.warp_eq, warp, ties_seen)
            tie_rank = ties_before + Int32(popc(tie_vote & lanes_below))
            take = better | tie & Int32(tie_rank < need)
            if (tie != 0) & (tie_rank == need - 1):
                control[3] = index
            take_vote = ballot(take != 0)
            if lane == 0:
                warp_sel[warp] = Int32(popc(take_vote))
            rt.bar_sync(8, 256)
            slot, chunk_taken = rt.Scan.block_prefix(s.warp_sel, warp, written)
            scan_owner = cmod(cdiv(index, 128), 3)
            if (take != 0) & (scan_owner == self.SHARD):
                cute.arch.atomic_add(
                    refine_words.iterator + cmod(cdiv(cdiv(index, 128), 3), SCORE_CTAS),
                    Uint32(1),
                )
            lower_vote = ballot(
                (Int32(take != 0) & Int32(scan_owner < self.SHARD)) != 0
            )
            if (lane == 0) & (lower_vote != 0):
                cute.arch.atomic_add(shard_in.iterator + 7, Uint32(popc(lower_vote)))
            ties_seen = ties_seen + chunk_ties
            written = written + chunk_taken
            if written >= TOPK:
                scanning = Int32(0)
            scan_base = scan_base + 256
        rt.bar_sync(8, 256)
        self.sel_publish(
            p, s, ids, warp, epoch, row_token, prefix, Uint32(s.control[3])
        )

    @cute.jit
    def shard_sum(self, p, s, at, n, tag40: cutlass.Constexpr):
        """Own partial plus the two other shards' words of bin ``n`` (``tag40``) or
        group ``n``.
        """
        base, parity, token = at
        lo, hi = (1, 0, 0)[self.SHARD], (2, 2, 1)[self.SHARD]
        if cutlass.const_expr(tag40):
            own_off, idx = StateWords.bin(parity, token, n), n
        else:
            own_off, idx = (
                StateWords.group(parity, token, n),
                BucketMailbox.GROUP_WORDS + n,
            )
        total = ld_cg_b32(ga(p.SelectState, own_off))
        for o in cutlass.range_constexpr(2):
            want = s.epoch_slot[0] + 1
            polled = rt.Poll.word(
                SelectionMailbox.word_addr(base, parity, (lo, hi)[o], token, idx),
                want,
                tag40,
            )
            if cutlass.const_expr(tag40):
                total = total + (Uint32(polled) & Uint32(1048575))
            else:
                total = total + Uint32(polled)
        return Int32(total)

    @cute.jit
    def seed_update(
        self, p, s, base, epoch, token, key, coarse_in, refresh: cutlass.Constexpr
    ):
        """Tid 0: next-step selection seed (StateWords.SEEDS + token) from this step's
        threshold ``key``.
        """
        SelectState = p.SelectState
        previous = ld_cg_b32(ga(p.SelectState, StateWords.PREVIOUS_SEED))
        parity = Int32(cmod(epoch + 1 - 1, 2))
        at = (base, parity, token)
        coarse_bucket = coarse_in
        if cutlass.const_expr(refresh):
            if (previous != 0) & (coarse_in <= 8192):
                group = cutlass.min(
                    cutlass.max(
                        cdiv(Int32(key >> 15) - Coarse.group_floor(previous), 32), 0
                    ),
                    127,
                )
                coarse_bucket = self.shard_sum(p, s, at, group, False)
            if (previous == 0) & (coarse_in <= 8192):
                digit = Int32(key >> 20)
                coarse_bucket = self.shard_sum(p, s, at, digit, True)
        next_seed = key & Uint32(4294967294)
        if coarse_bucket > 8192:
            next_seed = next_seed | Uint32(1)
        if ((next_seed & Uint32(1)) == 0) & (key >= Uint32(3254779904)):
            below = Int32(0)
            if previous == 0:
                digit_below = Int32(key >> 20) - 1
                if digit_below >= 0:
                    below = self.shard_sum(p, s, at, digit_below, True)
            else:
                group_below = (
                    cdiv(Int32(key >> 15) - Coarse.group_floor(previous), 32) - 1
                )
                if (group_below >= 0) & (group_below < 127):
                    below = self.shard_sum(p, s, at, group_below, False)
            if below > 8192:
                next_seed = next_seed | Uint32(1)
        SelectState[StateWords.SEEDS + token] = next_seed

    @cute.jit
    def sel_recycle(self, p, other, bid, tid, warp):
        """Clear the other parity's bin/group/limit words, then count this CTA done in
        Flags[888 + bid].
        """
        state = p.SelectState
        state_words = 114
        if (tid < 2 * state_words) & (bid * 2 * state_words + tid < 32768):
            state[
                StateWords.BINS + 2 * other * 4 * 4096 + bid * 2 * state_words + tid
            ] = 0
        if (bid == 0) & (tid < 8):
            state[32768 + other * 4 * 2 + tid] = 0
            state[32784 + other * 4 * 2 + tid] = 0
        group_words = 4
        if (group_words > tid) & (bid * group_words + tid < 512):
            state[StateWords.GROUPS + other * 4 * 128 + bid * group_words + tid] = 0
        if (bid == 0) & (tid < 4):
            state[StateWords.refine(tid) + 160 + other * 8 + 5] = 0
        rt.bar_sync(8, 256)
        with rt.warp_leader(warp, 0):
            rt.Memory.fence_release_gpu()
            cute.arch.atomic_add(p.Flags.iterator + (888 + bid), Uint32(1))

    @cute.jit
    def sel_copy_ids(self, p, sb, epoch, token, tid):
        """Full kernel, long rows: copy the token's 2048 selected ids from the selection
        mailbox.
        """
        token_parity = Int32(cmod(epoch + 1, 2))
        if p.CurPos[0] + token + 1 > 2048:
            selected = p.SelectedIds
            for token_slot in range(tid, 2048, 256):
                copy_addr = sb + Int64(
                    ((token_parity * 4 + token) * 2048 + token_slot) * 8
                )
                token_copied = rt.Poll.wait([copy_addr], epoch + 1)[0]
                selected[token * 2048 + token_slot] = Int32(Uint32(token_copied))

    @cute.jit
    def sel_publish(self, p, s, ids: rt.ThreadIds, warp, epoch, token, rec0, rec1):
        """Publish (threshold, limit) and this CTA's per-row selected counts to
        SelectState (volatile u64 records).
        """
        tid, lane = ids.tid, ids.lane
        tag = Uint64(epoch + 1) << 32
        if tid < 144:
            st_volatile_u64(
                ga(p.SelectState, StateWords.cta_record(token, tid, 0)),
                tag | Uint64(rec0),
            )
            st_volatile_u64(
                ga(p.SelectState, StateWords.cta_record(token, tid, 1)),
                tag | Uint64(rec1),
            )
        cta_count = Int32(0)
        if tid < 144:
            cta_count = Int32(s.refine_words[tid])
        inclusive, before, total = rt.Scan.cta(cta_count, s.warp_eq, warp, lane)
        lower_above_word = s.shard_in[6]
        lower_taken_word = s.shard_in[7]
        if tid < 144:
            packed = Uint32(
                before
                + inclusive
                - cta_count
                + Int32(lower_above_word + lower_taken_word)
            ) | (Uint32(cta_count) << 16)
            st_volatile_u64(
                ga(p.SelectState, StateWords.cta_record(token, tid, 2)),
                tag | Uint64(packed),
            )

    @cute.jit
    def sel_reduce(self, p, s, ids: rt.ThreadIds, warp, epoch):
        """Shard reduce of the reduce CTAs: stamp this CTA's tag into every peer's
        mailbox slot, copy the SelectedIds (not the shared kernel), sum the eight
        shard partials of REDUCE_ROWS rows and write ``Final = reduced + residual``.
        """
        tid, bid, lane = ids.tid, ids.bid, ids.lane
        local_o_words = s.local_o_words
        if tid < 96:
            local_o_words[tid] = Uint32(0)
        rt.bar_sync(8, 256)
        tag_high = (128 | epoch + 1 & 127) << 24
        parity = Int32(cmod(epoch + 1, 2))
        if tid < 56 * self.REDUCE_ROWS:
            destination = cmod(cdiv(tid, 8), 7) + Int32(
                cmod(cdiv(tid, 8), 7) >= self.SHARD
            )
            source_slot = self.SHARD - Int32(destination < self.SHARD)
            first = (
                cdiv(tid, 56) * 57344
                + source_slot * 8192
                + parity * 4096
                + bid * 32
                + cmod(tid, 8) * 4
            )
            peer = p.peer_base0
            for rank in cutlass.range_constexpr(1, 8):
                if destination == rank:
                    peer = p.peer_bases[rank]
            rt.Memory.st_volatile_v4(peer + Int64(first * 4), (tag_high,) * 4)
        rt.bar_sync(11, 384)
        query = cdiv(warp, 2)
        copy_item = bid * 256 + tid
        copy_token = cdiv(copy_item, 2048)
        if (copy_token < 4) & (
            self.rows(p, p.CurPos[0] + copy_token + 1, copy_token) > 2048
        ):
            copied = self.poll_word_all(
                p.shard_base(self.SHARD)
                + Int64((Int32(cmod(epoch + 1, 2)) * 4 * 2048 + copy_item) * 8),
                epoch + 1,
            )
            selected = p.SelectedIds
            selected[copy_item] = Int32(Uint32(copied))
        rt.bar_sync(11, 384)
        if (cmod(warp, 2) == 0) & (lane < 12) & (query < self.REDUCE_ROWS):
            self.sum_partials(s, query, lane)
        rt.bar_sync(8, 256)
        if tid < 48:
            final = p.Final
            for row in cutlass.range_constexpr(self.REDUCE_ROWS):
                final[row * 6144 + bid * 48 + tid] = (
                    fadd(
                        s.local_o[row * 48 + tid].to(Float32),
                        s.residual[row * 48 + tid].to(Float32),
                    )
                ).to(BFloat16)

    @cute.jit
    def sum_partials(self, s, query, lane):
        """Row ``query``, lane's 4 columns: received partials R0..R6 with this shard's
        own partial inserted at position SHARD (trace-time arms), summed from zero,
        stored back to local_o.
        """
        sums = rmem_fill(4, Float32, 0)
        sources = [
            (s.received_words, query * 168 + 24 * k + lane * 2)
            for k in (0, 1, 2, 3, 4, 5, 6)
        ]
        sources.insert(self.SHARD, (s.local_o_words, query * 24 + lane * 2))
        for k in cutlass.range_constexpr(8):
            words, off = sources[k]
            for w in cutlass.range_constexpr(2):
                f_lo, f_hi = rt.Convert.bf16x2_to_f32x2(words[off + w])
                sums[2 * w] = fadd(sums[2 * w], f_lo)
                sums[2 * w + 1] = fadd(sums[2 * w + 1], f_hi)
        rt.Memory.st_shared_v2(
            s.local_o_words_addr + (query * 24 + lane * 2) * 4,
            rt.f32x2_to_bf16x2(sums[0], sums[1]),
            rt.f32x2_to_bf16x2(sums[2], sums[3]),
        )

    @staticmethod
    @cute.jit
    def poll_word_all(addr, want):
        """Warp-wide spin: every lane reads its u64 word until all tags (hi 32 bits)
        equal ``want``.
        """
        polled = Uint64(0)
        waiting = Int32(1)
        while waiting != 0:
            polled = ld_volatile_u64(addr)
            if cute.arch.vote_all_sync(Uint32(polled >> 32) == want):
                waiting = Int32(0)
        return polled

    def sel_push(self, p, parity, token, idx, record):
        """Trace time: ``record`` to this shard's exchange word ``idx`` of ``token`` in
        both partners' mailboxes.
        """
        for shard in range(3):
            if shard != self.SHARD:
                addr = SelectionMailbox.word_addr(
                    p.shard_base(shard), parity, self.SHARD, token, idx
                )
                st_volatile_u64(addr, record)

    @staticmethod
    @cute.jit
    def sel_band(sums, base: cutlass.Constexpr, threshold: cutlass.Constexpr):
        """Warp band search over 4 groups: (before, found, item, value) of the first
        group reaching ``threshold``.
        """
        before = Int32(0)
        found = Int32(0)
        band_item = Int32(0)
        value = Int32(0)
        for item in cutlass.range_constexpr(4):
            total = Int32(redux_add_u32(sums[base + item]))
            reaches = Int32((found == 0) & (before + total >= threshold))
            if reaches != 0:
                found = Int32(1)
                band_item = Int32(item)
                value = Int32(sums[base + item])
            if (found == 0) & (reaches == 0):
                before = before + total
        return before, found, band_item, value


WQB_PAGES = (4096, 2048, 128, 32, 1024, True)
WIS_PAGES = (160, 12288, 128, 16, 2048, True)


class SelectorFull(MlaSelector):
    """``mla_selector_q4_shard3r{SHARD}_paged_vlin``: the full selector of key shard
    ``SHARD`` (0..2).
    """

    @cute.kernel
    def kernel(
        self,
        p,
        InputTmap: GridConstant[TensorMap],
        WqbTmap: GridConstant[TensorMap],
        WisTmap: GridConstant[TensorMap],
    ):
        """The params record, then the three page descriptors (kept by address for the
        loader).
        """
        maps = {"InputTmap": InputTmap, "WqbTmap": WqbTmap, "WisTmap": WisTmap}
        self.tmaps = {name: m.get_ptr().toint() for name, m in maps.items()}
        ids, s = self.prologue(p)
        self.run(p, s, ids)

    def tensor_maps(self, p) -> tuple:
        """Host wrapper: the page descriptors of q_a (the worker's input pages),
        ``wq_b`` and ``wk_weights_proj``.
        """
        weights = (
            (p.QakiWeight, INPUT_PAGES),
            (p.WqbWeight, WQB_PAGES),
            (p.WisWeight, WIS_PAGES),
        )
        return tuple(MlaWorker.page_map(weight, *pages) for weight, pages in weights)
