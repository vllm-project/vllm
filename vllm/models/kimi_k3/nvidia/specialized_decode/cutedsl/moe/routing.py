# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top16 routing (threshold, regional candidates and exact composite ranking) and the
helpers the TP8 and TP16 small-batch fronts share."""

from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, Uint64

from ..common.primitives import (
    bits_float,
    copy_evict,
    float_bits,
    gate_bf16_bits,
    mailbox_load,
    mailbox_store,
    no_neg_zero,
    published,
    redux,
    store_shared_words,
    view,
    zeros,
)


@cute.jit
def bf16_word(lo, hi):
    return no_neg_zero(Uint32(gate_bf16_bits(lo)) | (Uint32(gate_bf16_bits(hi)) << 16))


@cute.jit
def poll_words(words, pointer, valid):
    """Copy four mailbox words into ``words``; clear ``valid`` if any is unpublished."""
    values = mailbox_load(pointer, count=4)
    for j in cutlass.range_constexpr(4):
        words[j] = values[j]
        valid = valid & (values[j] != Uint32(0x80000000))
    return valid


@cute.jit
def gather_x(xmoe, par, cta, dst, first_chunk, active, tokens=1, h2=None):
    """Poll this CTA's x_moe rows (and the TP16 H2 slices if ``h2 = (warp, lane,
    pairs)``) until published, then stage the row chunks at ``dst``."""
    words = zeros((tokens, 2, 4), Uint32)
    h2_words: Any = None
    if cutlass.const_expr(h2 is not None):
        h2_words = zeros((tokens, 4, 4), Uint32)
    ready = False
    while not ready:
        valid = True
        for token in cutlass.range_constexpr(tokens):
            rows = (par * xmoe.shape[1] + cta % xmoe.shape[1]) * xmoe.shape[2]
            base = xmoe.iterator + rows * 7168 + token * 7168
            if active:
                for chunk in cutlass.range_constexpr(2):
                    pointer = base + (first_chunk + 28 * chunk) * 8
                    valid = poll_words(words[token, chunk, None], pointer, valid)
            if cutlass.const_expr(h2 is not None):
                warp, lane, pairs = h2
                if (warp < 7) & (pairs > 0):
                    for chunk in cutlass.range_constexpr(4):
                        pointer = base + (32 * warp + lane + 224 * chunk) * 8
                        valid = poll_words(h2_words[token, chunk, None], pointer, valid)
        ready = cute.arch.vote_all_sync(valid)
    if active:
        for token in cutlass.range_constexpr(tokens):
            for chunk in cutlass.range_constexpr(2):
                pointer = dst + token * 14336 + (first_chunk + 28 * chunk) * 16
                values = tuple(words[token, chunk, j] for j in range(4))
                store_shared_words(pointer, values)
    return words, h2_words


@cute.jit
def score_record(score, biased):
    """Published (score, score+bias) bits; NaN or 0x80000000 bits become (0, -inf)."""
    score_bits, biased_bits = Uint32(float_bits(score)), Uint32(float_bits(biased))
    sentinel = (score_bits == Uint32(0x80000000)) | (biased_bits == Uint32(0x80000000))
    if sentinel | (score != score) | (biased != biased):
        score_bits, biased_bits = Uint32(0), Uint32(0xFF800000)
    return score_bits, biased_bits


@cute.jit
def stage_rows(slots, pitch, slot, weights, first, rows, barrier, skip=0):
    """Bulk-copy weight rows ``first + skip + i`` into slots ``slot + i``."""
    cute.arch.mbarrier_arrive_and_expect_tx(barrier, rows * 14336)
    for row in range(rows):
        dst = slots + (slot + row) * pitch
        src = weights.iterator + (first + skip + row) * 7168
        copy_evict(dst, src, size=14336, barrier=barrier)


class Top16Fast:
    """Eight-warp top16 of 896 experts, lower ids winning ties. ``scores`` holds FP32
    (score, score+bias) bits per [parity, token, expert]; ``run`` polls ``token``'s
    parity-``par`` records until published and writes ``ids[16*token:]``. Runs on the
    first 256 threads of the CTA with named barrier 1 and 13,200 bytes of SMEM."""

    def __init__(self, batch):
        self.batch = batch

    @cute.jit
    def arrays(self, storage):
        km = view(storage, 0, Uint32, (8, 32), (32, 1))
        gc = view(storage, 1024, Uint32, 32)
        wc = view(storage, 1152, Uint32, 8)
        threshold = view(storage, 1184, Uint32, 1)
        cc = view(storage, 1200, Uint64, 904)
        cs = view(storage, 8432, Uint32, 904)
        cc2 = view(storage, 12048, Uint64, 64)
        selected_scores = view(storage, 12560, Float32, 80)
        selected_ids = view(storage, 12880, Int32, 80)
        return km, gc, wc, threshold, cc, cs, cc2, selected_scores, selected_ids

    @cute.jit
    def rank_small(self, arrays, counts, n, lane, ids, groups: cutlass.Constexpr[int]):
        cc, cs, cc2, selected_scores, selected_ids = arrays[4:]
        mine = cute.make_rmem_tensor(groups, Uint64)
        score = cute.make_rmem_tensor(groups, Uint32)
        for half in cutlass.range_constexpr(groups):
            index = Uint32(lane + 32 * half)
            total = Uint32(0)
            slot = index
            for warp in cutlass.range_constexpr(7):
                total += counts[warp]
                past = index >= total
                gap = (128 if warp < 4 else 96) - counts[warp]
                slot += gap if past else Uint32(0)
            slot = min(slot, Uint32(895))
            mine[half] = cc[slot]
            score[half] = cs[slot]
            if index >= n:
                mine[half] = Uint64(0)
            cc2[index] = mine[half]
        cute.arch.sync_warp()
        for half in cutlass.range_constexpr(groups):
            rank = zeros(4, Uint32)
            for q in cutlass.range_constexpr(32 * groups):
                rank[q % 4] += Uint32(cc2[q] > mine[half])
            r = (rank[0] + rank[1]) + (rank[2] + rank[3])
            index = Uint32(lane + 32 * half)
            selected = (index < n) & (r < 16)
            slot = r if selected else 16 + index
            expert = Int32(Uint32(0xFFFFFFFF) - Uint32(mine[half]))
            selected_ids[slot] = expert
            selected_scores[slot] = bits_float(score[half])
            if selected:
                mailbox_store(ids + r, (Uint32(expert),))

    @cute.jit
    def run(self, scores, ids, storage, par, token):
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        records = zeros((4, 2), Uint32)
        pending = Uint32(0)
        for i in cutlass.range_constexpr(4):
            if warp + 8 * i < 28:
                pending |= Uint32(1 << i)
        while pending != 0:
            for i in cutlass.range_constexpr(4):
                if (pending & Uint32(1 << i)) != 0:
                    row = (par * self.batch + token) * 896 + lane
                    pointer = scores.iterator + (row + 32 * (warp + 8 * i)) * 2
                    words = mailbox_load(pointer, count=2)
                    records[i, 0], records[i, 1] = words[0], words[1]
            for i in cutlass.range_constexpr(4):
                done = (pending & Uint32(1 << i)) != 0
                if published((records[i, j] for j in range(2)), done):
                    pending &= Uint32(0xFFFFFFFF ^ (1 << i))
        cute.arch.barrier(barrier_id=1, number_of_threads=256)
        ids = ids.iterator + token * 16
        arrays = self.arrays(storage)
        km, gc, wc, threshold, cc, cs, _, selected_scores, selected_ids = arrays
        key = cute.make_rmem_tensor(4, Uint32)
        maximum = Uint32(0)
        for i in cutlass.range_constexpr(4):
            biased = bits_float(records[i, 1])
            bits = records[i, 1]
            negative = (bits & Uint32(0x80000000)) != 0
            ordered = ~bits if negative else bits | Uint32(0x80000000)
            key[i] = ordered if (warp + 8 * i < 28) & (biased == biased) else Uint32(0)
            maximum = max(maximum, key[i])
        km[warp, lane] = maximum
        cute.arch.barrier(barrier_id=1, number_of_threads=256)
        if warp == 0:
            group_max = km[0, lane]
            for w in cutlass.range_constexpr(1, 8):
                group_max = max(group_max, km[w, lane])
            composite = (group_max & Uint32(0xFFFFFFE0)) | Uint32(31 - lane)
            gc[lane] = composite
            cute.arch.sync_warp()
            ranks = zeros(4, Int32)
            for q in cutlass.range_constexpr(32):
                ranks[q % 4] += Int32(gc[q] > composite)
            r = (ranks[0] + ranks[1]) + (ranks[2] + ranks[3])
            least = redux(composite if r < 16 else Uint32(0xFFFFFFFF))
            threshold[0] = least & Uint32(0xFFFFFFE0)
        cute.arch.barrier(barrier_id=1, number_of_threads=256)
        cutoff = threshold[0]
        lane_mask = (Uint32(1) << lane) - Uint32(1)
        region = Uint32(128 * warp if warp < 4 else 512 + 96 * (warp - 4))
        offset = Uint32(0)
        for i in cutlass.range_constexpr(4):
            live = warp + 8 * i < 28
            ballot = cute.arch.vote_ballot_sync(live & (key[i] >= cutoff))
            candidate = ((ballot >> lane) & 1) != 0
            compact = region + offset + cute.arch.popc(ballot & lane_mask)
            position = compact if candidate else Uint32(896 + warp)
            expert = Uint32(lane + 32 * (warp + 8 * i))
            cc[position] = (Uint64(key[i]) << 32) | Uint64(Uint32(0xFFFFFFFF) - expert)
            cs[position] = records[i, 0]
            offset += cute.arch.popc(ballot)
        wc[warp] = offset
        cute.arch.barrier(barrier_id=1, number_of_threads=256)
        if warp == 0:
            counts = cute.make_rmem_tensor(8, Uint32)
            for w in cutlass.range_constexpr(8):
                counts[w] = wc[w]
            n = ((counts[0] + counts[1]) + (counts[2] + counts[3])) + (
                (counts[4] + counts[5]) + (counts[6] + counts[7])
            )
            if n <= 32:
                self.rank_small(arrays, counts, n, lane, ids, 1)
            elif n <= 64:
                self.rank_small(arrays, counts, n, lane, ids, 2)
            else:
                prefix = cute.make_rmem_tensor(8, Uint32)
                total = Uint32(0)
                for w in cutlass.range_constexpr(8):
                    prefix[w] = total
                    total += counts[w]
                for index in range(lane, n, 32):
                    slot = Uint32(index)
                    for w in cutlass.range_constexpr(7):
                        past = index >= prefix[w + 1]
                        gap = (128 if w < 4 else 96) - counts[w]
                        slot += gap if past else Uint32(0)
                    mine = cc[slot]
                    rank = Uint32(0)
                    for w in cutlass.range_constexpr(8):
                        base = 128 * w if w < 4 else 512 + 96 * (w - 4)
                        for q in range(counts[w]):
                            rank += Uint32(cc[base + q] > mine)
                    if rank < 16:
                        selected_expert = Int32(Uint32(0xFFFFFFFF) - Uint32(mine))
                        selected_ids[rank] = selected_expert
                        selected_scores[rank] = bits_float(cs[slot])
                        mailbox_store(ids + rank, (Uint32(selected_expert),))
            cute.arch.sync_warp()
        return selected_scores
