# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative verification against vocab-sharded top-64 candidates, in CuTe.

From the candidate packets every TP rank's `ShardTopK` posts to the
`CandidateMailbox`, three kernels verify the drafts without the full-vocab
logits:

- `_DraftStatsKernel` (grid: 8192-column blocks x rows) writes the max and the
  sum of exponentials of each draft row's blocks of logits over the
  temperature, as the stock `_compute_local_logits_stats_kernel` does. It
  reads only the draft logits cache, so it runs beside the select.
- `_SelectKernel` (one CTA of two warps per logits row) waits for the ranks'
  packets of its row in the mailbox and masks their candidates as the stock
  sampler masks logits, in its order: temperature,
  min-p, top-k (at most 64, ties to the smaller column), top-p. It keeps the
  survivors in column order. A row's top k (k <= 64) over the temperature are
  among the candidates: the scaling is monotonic.
- `_VerifyKernel` (one CTA per request, a warp per logits row) verifies the
  request as `rejection_sample` does (greedy, standard, synthetic or block
  verification) over the kept columns, samples the recovered or bonus token
  by Gumbel-max, and writes `num_sampled` and `num_rejected` as
  `get_num_sampled_and_rejected` does. It reads the draft tokens and positions
  as `gather_draft_sampled` does.

Greedy rows are identical to stock. Sampling follows the same distribution but
not the same random draws as the stock sampler, so a seeded request can differ
between the two paths. The draft probabilities are those of the draft logits
cache over the temperature, as the draft sampled them up to rounding.
"""

import functools
import struct

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.utils.math_utils import cdiv

from ..common.host import assert_not_capturing, pdl_supported
from ..common.mailbox import SPIN_TIMEOUT_NS, spin_deadline
from ..common.primitives import (
    _pure,
    above_u32,
    global_ptr,
    load_global_bf16x8_as_f32_address,
    load_global_u32_address,
    order_key,
    store_global_u32_address,
    store_shared_u32x4_pred,
)
from ..common.sequenced import generation, load_relaxed_sys_b64x2, next_sequence
from .shard_top_k import (
    COLUMN_BITS,
    PACKET_WORDS,
    TAG_BITS,
    TAG_MASK,
    CandidateMailbox,
    K,
    ballot_count,
    count_words,
    place,
    untag_candidate,
    warp_merge,
    warp_scan_inclusive,
    warp_sort_desc,
)

DRAFT_BLOCK = 8192
DRAFT_THREADS = 256
SELECT_THREADS = 64
# The candidates of a row: 8 keys per select thread.
MAX_CANDIDATES = SELECT_THREADS * 8
STANDARD, SYNTHETIC, BLOCK = 0, 1, 2
_NEG_INF = float("-inf")
# Two BF16 -inf, as one 32-bit word.
_NEG_INF_BF16X2 = -8323200
# The smallest uniform of `uniform` and its scale: the smallest positive value
# of `tl.rand` (`_TL_RAND_MIN` of `sample/gumbel.py`).
_RAND_MIN = struct.unpack("f", struct.pack("I", 0x2FFFFFFF))[0]


@dsl_user_op
def key_value(key: Uint32, *, loc=None, ip=None) -> Float32:
    """The FP32 value of an `order_key`."""
    return Float32(
        _pure(
            "{\n\t.reg .pred p;\n\t.reg .b32 a, b;\n\t"
            "setp.ge.u32 p, $1, -2147483648;\n\txor.b32 a, $1, -2147483648;\n\t"
            "not.b32 b, $1;\n\tselp.b32 a, a, b, p;\n\tmov.b32 $0, a;\n\t}",
            "=f,r",
            (Uint32(key),),
            loc,
            ip,
            T.f32(),
        )
    )


def gmem(dtype, address):
    """A flat view of global memory at `address`."""
    return cute.make_tensor(global_ptr(dtype, address), cute.make_layout(1 << 30))


def exp(x):
    return cute.math.exp(x, fastmath=True)


def log(x):
    return cute.math.log(x, fastmath=True)


def _mulhi(a: Uint32, b: int) -> Uint32:
    return Uint32((Uint64(a) * Uint64(b)) >> 32)


def uniform(seed: Int64, offset: Int64, stream: Uint32 | int = 0) -> Float32:
    """A uniform in (0, 1): Philox 4x32-10 keyed by `seed`, counter (`offset`,
    `stream`, 0), scaled as `tl.rand` and clamped to at least its smallest
    positive value. Stream 0 is `tl_rand32(seed, offset, includes_zero=False)`
    (`sample/gumbel.py`)."""
    c0, c1 = Uint32(offset), Uint32(offset >> 32)
    c2, c3 = Uint32(stream), Uint32(0)
    k0, k1 = Uint32(seed), Uint32(seed >> 32)
    for _ in range(10):
        a, b = c0, c2
        c0 = _mulhi(b, 0xCD9E8D57) ^ c1 ^ k0
        c2 = _mulhi(a, 0xD2511F53) ^ c3 ^ k1
        c1 = b * Uint32(0xCD9E8D57)
        c3 = a * Uint32(0xD2511F53)
        k0 = k0 + Uint32(0x9E3779B9)
        k1 = k1 + Uint32(0xBB67AE85)
    x = Int32(c0)
    x = x ^ (x >> 31)
    return cute.arch.fmax(Float32(x) * Float32(_RAND_MIN), Float32(_RAND_MIN))


def waiting(a: Uint32, b: Uint32, c: Uint32, d: Uint32, tag: Uint32, valid: Int32):
    """Whether a lane of the warp has a valid quad of packet words with a
    word not tagged `tag` (uniform over the warp, which then stays
    converged)."""
    stale = ((a ^ tag) | (b ^ tag) | (c ^ tag) | (d ^ tag)) & Uint32(TAG_MASK)
    return cute.arch.vote_ballot_sync((stale != Uint32(0)) & (valid != 0)) != 0


def warp_argmax(value, column):
    """The largest `value` of the warp and its column, the smallest column
    among equal values."""
    for m in (16, 8, 4, 2, 1):
        other = cute.arch.shuffle_sync_bfly(value, offset=m)
        other_column = cute.arch.shuffle_sync_bfly(column, offset=m)
        take = (other > value) | ((other == value) & (other_column < column))
        value = cutlass.select_(take, other, value)
        column = cutlass.select_(take, other_column, column)
    return value, column


# --- kernels -------------------------------------------------------------------


class _DraftStatsKernel:
    """Grid (blocks, rows), 256 threads of 32 columns: the max and sum of
    exponentials of a draft row's 8192-column block of BF16 logits over the
    request's temperature, `[rows, blocks, 2]`. Bonus rows (no draft) are
    skipped, and greedy rows' stats go unread."""

    def __init__(self, vocab: int, enable_pdl: bool) -> None:
        self.vocab, self.enable_pdl = vocab, enable_pdl
        self.blocks = cdiv(vocab, DRAFT_BLOCK)

    @cute.jit
    def __call__(
        self,
        stats: Int64,
        draft: Int64,
        draft_stride_0: Int64,
        draft_stride_1: Int64,
        expanded_idx_mapping: Int64,
        expanded_local_pos: Int64,
        temperature: Int64,
        rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            stats,
            draft,
            draft_stride_0,
            draft_stride_1,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            rows,
        ).launch(
            grid=(self.blocks, rows, 1),
            block=(DRAFT_THREADS, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        stats: Int64,
        draft: Int64,
        draft_stride_0: Int64,
        draft_stride_1: Int64,
        expanded_idx_mapping: Int64,
        expanded_local_pos: Int64,
        temperature: Int64,
        rows: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block, row, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        smem = cutlass.memory.SmemAllocator()
        partial = smem.allocate_tensor(Float32, cute.make_layout(8), 16)
        local_pos = gmem(Int32, expanded_local_pos)
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        next_pos = Int32(0)
        if row + 1 < rows:
            next_pos = local_pos[row + 1]
        if next_pos != 0:
            req_state = gmem(Int64, expanded_idx_mapping)[row]
            temp = gmem(Float32, temperature)[req_state]
            # Greedy rows' stats go unread.
            inv_temp = Float32(1.0) / cutlass.select_(
                temp == Float32(0.0), Float32(1.0), temp
            )
            base = draft + (
                req_state * draft_stride_0
                + Int64(local_pos[row]) * draft_stride_1
                + Int64(block * DRAFT_BLOCK)
            ) * Int64(2)
            values = []
            for i in cutlass.range_constexpr(DRAFT_BLOCK // (8 * DRAFT_THREADS)):
                offset = i * 8 * DRAFT_THREADS + tidx * 8
                values.extend(
                    load_global_bf16x8_as_f32_address(
                        base + Int64(offset) * 2,
                        Int32(block * DRAFT_BLOCK + offset < self.vocab),
                        fill_bits=_NEG_INF_BF16X2,
                    )
                )
            values = [v * inv_temp for v in values]
            top = values[0]
            for v in values[1:]:
                top = cute.arch.fmax(top, v)
            top = cute.arch.warp_reduction(top, cute.arch.fmax)
            if lane == 0:
                partial[warp] = top
            cute.arch.sync_threads()
            top = partial[0]
            for w in cutlass.range_constexpr(1, DRAFT_THREADS // 32):
                top = cute.arch.fmax(top, partial[w])
            cute.arch.sync_threads()
            total = Float32(0.0)
            for v in values:
                total = total + exp(v - top)
            total = cute.arch.warp_reduction_sum(total)
            if lane == 0:
                partial[warp] = total
            cute.arch.sync_threads()
            if tidx == 0:
                total = partial[0]
                for w in cutlass.range_constexpr(1, DRAFT_THREADS // 32):
                    total = total + partial[w]
                out = gmem(Float32, stats)
                at = (row * self.blocks + block) * 2
                out[at] = top
                out[at + 1] = cutlass.select_(
                    top > Float32(_NEG_INF), total, Float32(0.0)
                )


class _SelectKernel:
    """Grid (rows), 64 threads: the row's logits that survive the stock
    sampler's masks among the `tp` ranks' candidate packets in this rank's
    `CandidateMailbox` copy (`mailbox`), in column order, then -inf at column
    0, as `[rows, 2, K]` (values, columns).

    A sampling row's candidates are divided by its temperature; it keeps those
    at or above its largest plus log(min-p), of those its top k (k its top-k,
    at most K), and of those the fewest largest whose probabilities sum to its
    top-p. Each mask keeps a prefix of the row's order (value descending,
    column ascending), so the row keeps its first `count`. A greedy row keeps
    its K largest.

    Thread t polls the words of the candidates `8t .. 8t+7` of the row's
    packets (rank `t // 8`, in column order) with relaxed system-scope loads
    of a candidate's 64 bits each, the size it was stored with, until each
    word carries the row's sequence number, then reads them in place;
    thread 0 advances the sequence number once every thread has read it.
    Each warp sorts its 256 keys, warp 0 merges the two warps' 64 largest and
    finds `count` and the key at it, and a scan over the threads takes, in
    column order, the keys above that key and the first ties.

    A CTA waits for the slowest rank's `ShardTopK` of its row, the wait the
    exchange always had; its 64 threads leave the SMs to the draft stats
    beside it.
    """

    def __init__(self, tp: int, enable_pdl: bool) -> None:
        self.tp, self.enable_pdl = tp, enable_pdl

    @cute.jit
    def __call__(
        self,
        mailbox: Int64,
        seqs: Int64,
        generation_bytes: Int64,
        rank_bytes: Int64,
        lanes: Int64,
        expanded_idx_mapping: Int64,
        temperature: Int64,
        top_k: Int64,
        top_p: Int64,
        min_p: Int64,
        rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            mailbox,
            seqs,
            generation_bytes,
            rank_bytes,
            lanes,
            expanded_idx_mapping,
            temperature,
            top_k,
            top_p,
            min_p,
        ).launch(
            grid=(rows, 1, 1),
            block=(SELECT_THREADS, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mailbox: Int64,
        seqs: Int64,
        generation_bytes: Int64,
        rank_bytes: Int64,
        lanes: Int64,
        expanded_idx_mapping: Int64,
        temperature: Int64,
        top_k: Int64,
        top_p: Int64,
        min_p: Int64,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        smem = cutlass.memory.SmemAllocator()
        slots = smem.allocate_tensor(Uint32, cute.make_layout(2 * K), 16)
        # kth, need, count, warp 0's packed total.
        shared = smem.allocate_tensor(Int32, cute.make_layout(4), 16)
        base = Int32(slots.iterator.toint())

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        req_state = gmem(Int64, expanded_idx_mapping)[row]
        temp = gmem(Float32, temperature)[req_state]
        sampling = temp != Float32(0.0)
        rank = tidx // 8
        valid = Int32(rank < self.tp)
        seq = load_global_u32_address(seqs + Int64(row) * 4, ordered=True)
        source = (
            mailbox
            + Int64(generation(seq)) * generation_bytes
            + Int64(rank) * rank_bytes
            + Int64(row * PACKET_WORDS + (tidx % 8) * 16) * 4
        )
        # Every load is issued before the first wait.
        quads = [
            load_relaxed_sys_b64x2(source + Int64(i * 16), valid) for i in range(4)
        ]
        deadline = Int64(0)
        for i in cutlass.range_constexpr(4):
            a, b, c, d = quads[i]
            while waiting(a, b, c, d, seq, valid):
                deadline = spin_deadline(deadline, SPIN_TIMEOUT_NS)
                a, b, c, d = load_relaxed_sys_b64x2(source + Int64(i * 16), valid)
            quads[i] = (a, b, c, d)
        raw = [word for quad in quads for word in quad]
        candidates = [untag_candidate(raw[2 * e], raw[2 * e + 1]) for e in range(8)]
        inv_temp = Float32(1.0) / cutlass.select_(sampling, temp, Float32(1.0))
        keys = [order_key(key_value(key) * inv_temp, valid) for key, _ in candidates]
        columns = [column for _, column in candidates]

        top = warp_sort_desc(list(keys), lane)
        for e in cutlass.range_constexpr(2):
            store_shared_u32x4_pred(
                base + (warp * K + lane * 8 + 4 * e) * 4,
                top[4 * e : 4 * e + 4],
                Int32(lane < K // 8),
            )
        cute.arch.sync_threads()
        final = warp_merge(base, 0, 1, lane)
        if warp == 0:
            # At least one: a NaN largest logit masks every candidate, and a
            # count of 0 would place them at negative lanes.
            count = cutlass.max(
                self.count(final, lane, req_state, sampling, top_k, top_p, min_p),
                Int32(1),
            )
            # The key at `count` (entry 2 * lane + e of `final`).
            at = count - 1
            pick = cutlass.select_((at & 1) != 0, final[1], final[0])
            kth = cute.arch.shuffle_sync(pick, at >> 1)
            above = ballot_count(above_u32(final[0], kth) != 0) + ballot_count(
                above_u32(final[1], kth) != 0
            )
            if lane == 0:
                shared[0] = Int32(kth)
                shared[1] = count - above
                shared[2] = count
        cute.arch.sync_threads()
        kth = Uint32(shared[0])
        need = shared[1]
        count = shared[2]
        above, ties, packed = count_words(keys, [valid] * 8, kth)
        inclusive = warp_scan_inclusive(packed)
        if warp == 0 and lane == 31:
            shared[3] = inclusive
        cute.arch.sync_threads()
        prefix = inclusive - packed + shared[3] * Int32(warp > 0)
        values = gmem(Float32, lanes + Int64(row) * Int64(K * 8))
        cols = gmem(Uint32, lanes + Int64(row) * Int64(K * 8) + Int64(K * 4))
        for e, (selected, position) in enumerate(place(above, ties, prefix, need)):
            if selected != 0:
                values[position] = key_value(keys[e])
                cols[position] = columns[e]
        if tidx >= count:
            values[tidx] = Float32(_NEG_INF)
            cols[tidx] = Uint32(0)
        # Every thread read the sequence number before the barriers above.
        if tidx == 0:
            store_global_u32_address(
                seqs + Int64(row) * 4, next_sequence(seq, TAG_BITS)
            )

    @cute.jit
    def count(self, final, lane, req_state, sampling, top_k, top_p, min_p):
        """How many of the row's largest it keeps, from warp 0's merged top
        64 (`final`, descending; entry `2 * lane + e`)."""
        count = Int32(K)
        if sampling:
            k = cutlass.min(gmem(Int32, top_k)[req_state], Int32(K))
            p = gmem(Float32, top_p)[req_state]
            values = [key_value(final[0]), key_value(final[1])]
            largest = cute.arch.shuffle_sync(values[0], 0)
            cut = largest + log(gmem(Float32, min_p)[req_state])
            index = [2 * lane, 2 * lane + 1]
            kept = [(index[e] < k) & (values[e] >= cut) for e in range(2)]
            count = ballot_count(kept[0]) + ballot_count(kept[1])
            if p < Float32(1.0):
                probs = [
                    cutlass.select_(kept[e], exp(values[e] - largest), Float32(0.0))
                    for e in range(2)
                ]
                pair = probs[0] + probs[1]
                inclusive = pair
                for i in cutlass.range_constexpr(5):
                    up = cute.arch.shuffle_sync_up(inclusive, 1 << i)
                    inclusive = inclusive + cutlass.select_(
                        lane >= (1 << i), up, Float32(0.0)
                    )
                bound = p * cute.arch.shuffle_sync(inclusive, 31)
                before = inclusive - pair
                count = ballot_count(kept[0] & (before < bound)) + ballot_count(
                    kept[1] & (before + probs[0] < bound)
                )
        return count


class _VerifyKernel:
    """Grid (requests), a warp per logits row of the request (`warps` = the
    speculative steps + 1); see the module docstring."""

    def __init__(self, vocab: int, warps: int, mode: int, enable_pdl: bool) -> None:
        self.vocab, self.warps, self.mode = vocab, warps, mode
        self.blocks = cdiv(vocab, DRAFT_BLOCK)
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        lanes: Int64,
        stats: Int64,
        draft: Int64,
        draft_stride_0: Int64,
        draft_stride_1: Int64,
        cu_num_logits: Int64,
        idx_mapping: Int64,
        logits_indices: Int64,
        positions: Int64,
        input_ids: Int64,
        expanded_local_pos: Int64,
        prefill_len: Int64,
        seq_lens: Int64,
        temperature: Int64,
        seeds: Int64,
        rates: Int64,
        sampled: Int64,
        num_sampled: Int64,
        num_rejected: Int64,
        num_reqs: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            lanes,
            stats,
            draft,
            draft_stride_0,
            draft_stride_1,
            cu_num_logits,
            idx_mapping,
            logits_indices,
            positions,
            input_ids,
            expanded_local_pos,
            prefill_len,
            seq_lens,
            temperature,
            seeds,
            rates,
            sampled,
            num_sampled,
            num_rejected,
        ).launch(
            grid=(num_reqs, 1, 1),
            block=(32 * self.warps, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        lanes: Int64,
        stats: Int64,
        draft: Int64,
        draft_stride_0: Int64,
        draft_stride_1: Int64,
        cu_num_logits: Int64,
        idx_mapping: Int64,
        logits_indices: Int64,
        positions: Int64,
        input_ids: Int64,
        expanded_local_pos: Int64,
        prefill_len: Int64,
        seq_lens: Int64,
        temperature: Int64,
        seeds: Int64,
        rates: Int64,
        sampled: Int64,
        num_sampled: Int64,
        num_rejected: Int64,
    ) -> None:
        warps, mode = self.warps, self.mode
        tidx, _, _ = cute.arch.thread_idx()
        req, _, _ = cute.arch.block_idx()
        w = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        # Per row: its target and draft logsumexps, the draft token's target
        # and draft logprobs, its uniform, log p after it (block) and its
        # residual mass; its draft token (-1: none), accept flag, target
        # argmax and position.
        smem = cutlass.memory.SmemAllocator()
        s_tlse, s_dlse, s_tlp, s_dlp, s_u, s_cum, s_mass = (
            smem.allocate_tensor(Float32, cute.make_layout(warps), 16) for _ in range(7)
        )
        s_tok, s_accept, s_argmax = (
            smem.allocate_tensor(Int32, cute.make_layout(warps), 16) for _ in range(3)
        )
        s_pos = smem.allocate_tensor(Int64, cute.make_layout(warps), 16)
        s_j = smem.allocate_tensor(Int32, cute.make_layout(1), 16)
        cu = gmem(Int32, cu_num_logits)
        indices = gmem(Int64, logits_indices)
        local_pos = gmem(Int32, expanded_local_pos)
        pos_of = gmem(Int64, positions)

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        start = cu[req]
        n = cu[req + 1] - start
        nd = n - 1
        req_state = gmem(Int64, idx_mapping)[req]
        seed = gmem(Int64, seeds)[req_state]
        temp = gmem(Float32, temperature)[req_state]
        greedy = temp == Float32(0.0)
        # The draft logits' scale (unread if greedy).
        inv_temp = Float32(1.0) / cutlass.select_(greedy, Float32(1.0), temp)
        prefill = Int64(gmem(Int32, prefill_len)[req_state])
        draft_req = draft + req_state * draft_stride_0 * 2

        if w < n:
            row = start + w
            pos = pos_of[indices[row]]
            tlse, argmax = self.row_stats(lanes, row, lane)
            tok = Int32(-1)
            tlp = Float32(_NEG_INF)
            dlp = Float32(_NEG_INF)
            dlse = Float32(0.0)
            u = Float32(1.0)
            accept = cutlass.Boolean(False)
            if w < nd:
                index = indices[row + 1]
                next_local = Int64(local_pos[row + 1])
                placeholder = (next_local > Int64(0)) & (
                    pos_of[index] - next_local < prefill
                )
                if not placeholder:
                    tok = gmem(Int32, input_ids)[index]
                step_seed = seed
                if cutlass.const_expr(mode == BLOCK):
                    step_seed = seed ^ (Int64(w) << 32)
                u = uniform(step_seed, pos)
                if cutlass.const_expr(mode == SYNTHETIC):
                    accept = u < gmem(Float32, rates)[w]
                if tok >= 0:
                    if greedy:
                        if cutlass.const_expr(mode != SYNTHETIC):
                            accept = argmax == tok
                    else:
                        tlp = self.row_value(lanes, row, tok, lane) - tlse
                        dlse = self.draft_lse(stats, row, lane)
                        drafts = gmem(
                            cutlass.BFloat16, draft_req + Int64(w) * draft_stride_1 * 2
                        )
                        dlp = Float32(drafts[tok]) * inv_temp - dlse
                        if cutlass.const_expr(mode != SYNTHETIC):
                            accept = tlp > log(u) + dlp
                accept = accept & (tok >= 0)
            if lane == 0:
                s_tlse[w] = tlse
                s_dlse[w] = dlse
                s_tlp[w] = tlp
                s_dlp[w] = dlp
                s_u[w] = u
                s_tok[w] = tok
                s_accept[w] = Int32(accept)
                s_argmax[w] = argmax
                s_pos[w] = pos
        cute.arch.sync_threads()

        if tidx == 0:
            s_j[0] = self.prefix_length(nd, s_accept)
        if cutlass.const_expr(mode == BLOCK):
            self.block_length(
                lanes,
                draft_req,
                draft_stride_1,
                start,
                nd,
                w,
                greedy,
                inv_temp,
                (s_tlse, s_dlse, s_tlp, s_dlp, s_u, s_cum, s_mass, s_tok, s_j),
            )
        cute.arch.sync_threads()

        j = s_j[0]
        if w == 0:
            token = s_argmax[j]
            if not greedy:
                plain = (j == nd) | (s_tok[j] < 0)
                log_tau = Float32(0.0)
                if cutlass.const_expr(mode == BLOCK):
                    # Log p after the accepted drafts.
                    last = s_cum[cutlass.max(j, 1) - 1]
                    log_tau = cutlass.select_(j > 0, last, Float32(0.0))
                token = self.resample(
                    lanes,
                    start + j,
                    draft_req + Int64(j) * draft_stride_1 * 2,
                    plain,
                    s_tlse[j],
                    s_dlse[j],
                    log_tau,
                    inv_temp,
                    seed,
                    s_pos[j],
                    lane,
                )
            out = gmem(Int64, sampled + Int64(req) * Int64(warps * 8))
            if lane < j:
                out[lane] = Int64(s_tok[lane])
            if lane == j:
                out[lane] = Int64(token)
            if lane == 0:
                producing = Int32(Int64(gmem(Int32, seq_lens)[req]) >= prefill)
                gmem(Int32, num_sampled)[req] = (j + 1) * producing
                gmem(Int32, num_rejected)[req] = (n - j - 1) * producing

    @cute.jit
    def prefix_length(self, nd, s_accept) -> Int32:
        """The accepted prefix of the rows' accept flags."""
        length = Int32(0)
        verifying = Int32(1)
        for s in cutlass.range(nd):
            verifying = verifying & s_accept[s]
            length = length + verifying
        return length

    @cute.jit
    def block_length(
        self, lanes, draft_req, draft_stride_1, start, nd, w, greedy, inv_temp, s
    ):
        """Unless `greedy`, block verification's accepted length (Sun et al.,
        2024) into `s_j`, as `_compute_cumulative_log_p_kernel`,
        `_compute_local_residual_mass_kernel` and `_rejection_kernel` compute
        it: `s_cum[s]` is log p after draft s, `s_mass[s]` the residual mass of
        row s, which weighs its target probabilities by exp(s_cum[s - 1])."""
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        s_tlse, s_dlse, s_tlp, s_dlp, s_u, s_cum, s_mass, s_tok, s_j = s
        if not greedy:
            if tidx == 0:
                log_p = Float32(0.0)
                for i in cutlass.range(nd):
                    if s_tok[i] >= 0:
                        step = s_tlp[i] - s_dlp[i]
                        log_p = cutlass.min(log_p + step, Float32(0.0))
                    s_cum[i] = log_p
            cute.arch.sync_threads()
            # Rows 1 .. nd - 1 with a valid draft.
            if (w > 0) & (w < nd) & (s_tok[w] >= 0):
                mass = self.residual_mass(
                    lanes,
                    start + w,
                    draft_req + Int64(w) * draft_stride_1 * 2,
                    exp(s_cum[w - 1]),
                    s_tlse[w],
                    s_dlse[w],
                    inv_temp,
                    lane,
                )
                if lane == 0:
                    s_mass[w] = mass
            cute.arch.sync_threads()
            if tidx == 0:
                length = Int32(0)
                verifying = cutlass.Boolean(True)
                for i in cutlass.range(nd):
                    verifying = verifying & (s_tok[i] >= 0)
                    if verifying:
                        prefix = exp(s_cum[i])
                        h = prefix
                        # The next draft is valid (its slot past nd is unread).
                        if (i + 1 < nd) & (s_tok[i + 1] >= 0):
                            mass = s_mass[i + 1]
                            denom = mass + Float32(1.0) - prefix
                            h = Float32(1.0)
                            if denom > Float32(0.0):
                                h = mass / denom
                        if s_u[i] <= h:
                            length = i + 1
                s_j[0] = length

    # A row is its 64 kept (value, column) lanes, two per lane in column order.

    @cute.jit
    def lane_pair(self, lanes: Int64, row: Int32, lane):
        at = lanes + Int64(row) * Int64(K * 8)
        values = gmem(Float32, at)
        columns = gmem(Int32, at + Int64(K * 4))
        return (
            values[2 * lane],
            columns[2 * lane],
            values[2 * lane + 1],
            columns[2 * lane + 1],
        )

    @cute.jit
    def row_stats(self, lanes, row, lane):
        """(logsumexp, first argmax column) of the row's kept logits."""
        v0, c0, v1, c1 = self.lane_pair(lanes, row, lane)
        take = v1 > v0
        best, column = warp_argmax(
            cutlass.select_(take, v1, v0), cutlass.select_(take, c1, c0)
        )
        total = cute.arch.warp_reduction_sum(exp(v0 - best) + exp(v1 - best))
        return best + log(total), column

    @cute.jit
    def row_value(self, lanes, row, column, lane):
        """The row's kept logit at `column` (-inf if masked)."""
        v0, c0, v1, c1 = self.lane_pair(lanes, row, lane)
        found = cutlass.select_(c0 == column, v0, Float32(_NEG_INF))
        found = cutlass.select_(c1 == column, cute.arch.fmax(found, v1), found)
        return cute.arch.warp_reduction(found, cute.arch.fmax)

    @cute.jit
    def draft_lse(self, stats, row, lane):
        """`_compute_global_logsumexp` of the row's draft block stats."""
        top = Float32(_NEG_INF)
        total = Float32(0.0)
        if lane < self.blocks:
            at = gmem(Float32, stats + Int64(row * self.blocks + lane) * 8)
            top = at[0]
            total = at[1]
        best = cute.arch.warp_reduction(top, cute.arch.fmax)
        return best + log(cute.arch.warp_reduction_sum(total * exp(top - best)))

    @cute.jit
    def residual_mass(self, lanes, row, draft_row, p, tlse, dlse, inv_temp, lane):
        """`sum(max(p * target_probs - draft_probs, 0))` over the kept columns."""
        v0, c0, v1, c1 = self.lane_pair(lanes, row, lane)
        drafts = gmem(cutlass.BFloat16, draft_row)
        mass = Float32(0.0)
        for v, col in ((v0, c0), (v1, c1)):
            q = exp(Float32(drafts[col]) * inv_temp - dlse)
            mass = mass + cute.arch.fmax(p * exp(v - tlse) - q, Float32(0.0))
        return cute.arch.warp_reduction_sum(mass)

    @cute.jit
    def resample(
        self,
        lanes,
        row,
        draft_row,
        plain,
        tlse,
        dlse,
        log_tau,
        inv_temp,
        seed,
        pos,
        lane,
    ):
        """The Gumbel-max sample of the row's target (if `plain`) or residual
        distribution `max(p_tau * p - q, 0)`, as `_resample_kernel` defines
        it."""
        v0, c0, v1, c1 = self.lane_pair(lanes, row, lane)
        drafts = gmem(cutlass.BFloat16, draft_row)
        best = Float32(_NEG_INF)
        best_col = Int32(0)
        for v, col in ((v0, c0), (v1, c1)):
            r = v
            if not plain:
                tlp = v - tlse + log_tau
                ratio = exp(Float32(drafts[col]) * inv_temp - dlse - tlp)
                r = Float32(_NEG_INF)
                if ratio < Float32(1.0):
                    r = tlp + cute.math.log1p(-ratio)
            noisy = r - log(-log(uniform(seed, pos, Uint32(col) + Uint32(1))))
            take = noisy > best
            best = cutlass.select_(take, noisy, best)
            best_col = cutlass.select_(take, col, best_col)
        best, best_col = warp_argmax(best, best_col)
        return cutlass.select_(best > Float32(_NEG_INF), best_col, Int32(0))


@functools.cache
def compile_draft_stats(vocab: int, enable_pdl: bool):
    assert_not_capturing("the draft logits stats")
    return cute.compile(
        _DraftStatsKernel(vocab, enable_pdl),
        *([Int64(0)] * 7),
        Int32(1),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


@functools.cache
def compile_select(tp: int, enable_pdl: bool):
    assert_not_capturing("the vocab-sharded top-k select")
    return cute.compile(
        _SelectKernel(tp, enable_pdl),
        *([Int64(0)] * 10),
        Int32(1),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


@functools.cache
def compile_verify(vocab: int, warps: int, mode: int, enable_pdl: bool):
    assert_not_capturing("the vocab-sharded verify")
    return cute.compile(
        _VerifyKernel(vocab, warps, mode, enable_pdl),
        *([Int64(0)] * 19),
        Int32(1),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _address(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()


class ShardedVerify:
    """The compiled kernels and buffers of the verification of at most
    `max_rows` logits rows over a `tp`-way vocab sharding."""

    @staticmethod
    def supports(tp: int, vocab: int, num_speculative_steps: int) -> bool:
        return (
            tp * K <= MAX_CANDIDATES
            and vocab % 8 == 0
            and vocab <= 1 << COLUMN_BITS
            and cdiv(vocab, DRAFT_BLOCK) <= 32
            and num_speculative_steps + 1 <= 32
        )

    def __init__(
        self,
        tp: int,
        vocab: int,
        max_rows: int,
        num_speculative_steps: int,
        mode: int,
        device: torch.device,
    ) -> None:
        assert self.supports(tp, vocab, num_speculative_steps)
        self.tp, self.vocab = tp, vocab
        self.warps = num_speculative_steps + 1
        blocks = cdiv(vocab, DRAFT_BLOCK)
        pdl = pdl_supported()
        self.stats_kernel = compile_draft_stats(vocab, pdl)
        self.select_kernel = compile_select(tp, pdl)
        self.verify_kernel = compile_verify(vocab, self.warps, mode, pdl)
        self.stats = torch.empty(max_rows, blocks, 2, device=device)
        self.lanes = torch.empty(max_rows, 2, K, dtype=torch.int32, device=device)
        self._stats, self._lanes = self.stats.data_ptr(), self.lanes.data_ptr()

    def draft_stats(
        self,
        draft: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        temperature: torch.Tensor,
        stream: int | None = None,
    ) -> None:
        """The block stats of the BF16 draft logits `[reqs, steps, vocab]` over
        the `temperature` (per request state) of each logits row that has a
        draft, on the raw `stream` (default: the current one)."""
        self.stats_kernel(
            self._stats,
            _address(draft),
            draft.stride(0),
            draft.stride(1),
            _address(expanded_idx_mapping),
            _address(expanded_local_pos),
            _address(temperature),
            expanded_idx_mapping.shape[0],
            torch.cuda.current_stream().cuda_stream if stream is None else stream,
        )

    def select(
        self,
        mailbox: CandidateMailbox,
        rows: int,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        min_p: torch.Tensor,
        stream: int | None = None,
    ) -> None:
        """Merge the ranks' packets of the first `rows` rows, as `mailbox`
        receives them, into the rows' kept lanes, masked by the
        `temperature`, `top_k` (int32), `top_p` and `min_p` of each row's
        request state; advances the rows' sequence numbers."""
        self.select_kernel(
            mailbox.local,
            mailbox.seqs.data_ptr(),
            mailbox.generation_bytes,
            mailbox.rank_bytes,
            self._lanes,
            _address(expanded_idx_mapping),
            _address(temperature),
            _address(top_k),
            _address(top_p),
            _address(min_p),
            rows,
            torch.cuda.current_stream().cuda_stream if stream is None else stream,
        )

    def verify(
        self,
        draft: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        logits_indices: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        prefill_len: torch.Tensor,
        seq_lens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        rates: torch.Tensor | None,
        stream: int | None = None,
        out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(sampled, num_sampled, num_rejected)` as the stock rejection
        sampler and `get_num_sampled_and_rejected` return them, in `out`
        (contiguous `[reqs, steps + 1]` int64, `[reqs]` and `[reqs]` int32)
        if given."""
        num_reqs = idx_mapping.shape[0]
        if out is None:
            device = idx_mapping.device
            sampled = torch.empty(
                num_reqs, self.warps, dtype=torch.int64, device=device
            )
            num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=device)
            num_rejected = torch.empty_like(num_sampled)
        else:
            sampled, num_sampled, num_rejected = out
        self.verify_kernel(
            self._lanes,
            self._stats,
            _address(draft),
            draft.stride(0),
            draft.stride(1),
            _address(cu_num_logits),
            _address(idx_mapping),
            _address(logits_indices),
            _address(positions),
            _address(input_ids),
            _address(expanded_local_pos),
            _address(prefill_len),
            _address(seq_lens),
            _address(temperature),
            _address(seeds),
            _address(rates),
            _address(sampled),
            _address(num_sampled),
            _address(num_rejected),
            num_reqs,
            torch.cuda.current_stream().cuda_stream if stream is None else stream,
        )
        return sampled, num_sampled, num_rejected
