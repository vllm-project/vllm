# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The QSA sparse attention of a decode step as one CuTe DSL kernel per
(row, KV head).

What it computes. A QSA layer attends each query row to the KV positions its
indexer selected: up to `topk` (2,051 for
Qwen3.8-Flash-Next: 512 blocks of four tokens and a tail of up to three) token
positions per row, in no particular order. The packed selection is an int32
`[rows, topk + 1]` tensor whose last column holds the row's count; the
other columns hold logical token positions, padded with -1. The positions
live in vLLM's paged KV cache (pages of `page` tokens; a row's request maps
logical pages to physical blocks through the block table). For each row and
query head `h` of KV head `kv`::

    out[row, h] = BF16(BF16(softmax(q[row, h] . K / sqrt(256)) V)
                       * sigmoid(gate[row, h]))

over the row's valid positions (an empty row gives zeros), exactly the
operator of the stock Triton kernels in `ops/qsa.py`.

Why a second kernel. The stock path splits every (row, KV head) over up to
eight CTAs of one warp and merges the splits in a second kernel through FP32
partials. At 128 rows (32 requests with three draft tokens each, the TP1
decode step at concurrency 32) that is 2,048 one-warp CTAs and 25 MB of
partials a call. `QsaRowAttention` gives each (row, KV head) one
128-thread CTA that streams all of the row's positions through shared memory
and writes the gated BF16 output itself: 256 CTAs, two per SM, one wave.

Work split. A *stage* is 32 consecutive selection columns. Warp `w` of the
four owns columns `32 s + 8 w .. 32 s + 8 w + 7` of every stage `s`: it
gathers their K and V rows (512 bytes each) into its own slice of three
shared-memory stage buffers with `cp.async` (one 16-byte chunk per lane and
row), computes their scores against the KV head's query heads (padded to 16,
the MMA's M) with `mma.sync.m16n8k16`, keeps its own online softmax, and
accumulates its own 16 x 256 FP32 output with `mma.sync.m16n8k8` (the
probabilities as the A fragment, V through `ldmatrix.trans`). No barrier
across warps runs inside the stage loop. At the end the four warps' (max,
sum, output) merge through shared memory in warp order, and each thread
finishes two dims of every query head.

Sequence of a CTA:

1. `griddepcontrol.wait` (every input is written by earlier kernels of the
   step), then the dependents' trigger;
2. the row's count and request, the query fragments (registers, 16-byte
   loads straight from global memory), and the selection columns of the
   warp's positions; then their block-table entries. Each position becomes a
   *slot*, its token index in the cache (`block * block_tokens + offset`),
   or -1 when stock's guards reject it: request outside `[0, requests)`,
   column at or past `topk`, token -1, logical page at or past the table
   width, physical block outside `[0, blocks)`. The warp's slots go to its
   own shared array;
3. stages 0 and 1 are issued; in loop iteration `c` the warp waits for
   stage `c`'s copies, issues stage `c + 2` into the buffer stage
   `c - 1` used, and computes stage `c`. Rejected slots copy nothing and
   zero-fill their rows, and their scores are masked;
4. the merge and the gated output.

Softmax. Scores are scaled into the log2 domain by stock's FP32 constant
`256^-0.5 * log2(e)`. Each warp keeps per query head the running maximum of
its positions' scores; a stage that raises it for any head of the warp
rescales the warp's sums and outputs (a warp vote skips the rescale in the
other stages). The probabilities `2^(x - m)` (`ex2.approx.ftz`, at most 1,
so each warp's largest score has the exact probability 1) are rounded to BF16
for the P V product, and the sums add the FP32 values, as stock does. The
merge weighs warp `w` by `2^(m_w - max_w m_w)`, divides by the merged sum
and rounds to BF16, then multiplies by `tl.sigmoid(gate)` (Triton's
instruction sequence) and rounds again, the stock rounding points. The result
is not bitwise equal to stock (which sums in other orders); its error against
an FP32 reference is stock's (GB300, real selections at 128 rows: relative
Frobenius error 0.0025 against stock's 0.0024, the same largest absolute
error). A maximum moved only by rises of more than 8 (probabilities up to 256)
would be 2 us a call faster at 128 rows but 12% less accurate: the largest
probabilities would no longer be exact.

Programmatic dependent launch. The kernel launches with PDL and waits before
its first global read. It triggers its dependents right after the wait: a
dependent reads this kernel's output only after its own wait, which returns
when this grid has completed.

Scope. BF16 K/V caches with 256-dim heads, at most 16 query heads per KV
head, a cache whose token stride divides its block stride and whose slots fit
in int32, a BF16 output with the queries' shape and contiguous BF16 gate
logits; `row_attention_unsupported` names what a call misses.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..boundary._ptx import pack_bf16x2, store_u32
from ..gdn._ptx import load_s32
from ..hc._ptx import (
    load_shared_f32,
    load_shared_x4,
    load_x4,
    mma_chunk,
    store_shared_f32,
)
from ..logits._head_ptx import (
    copy_async_commit,
    copy_async_wait_group,
    load_u32,
    round_bf16,
    store_shared_s32,
    widen_bf16x2,
)
from ._ptx import (
    copy_async_16_fill,
    exp2_ftz,
    load_matrix_x4_trans,
    load_s32_if,
    load_shared_f32x2,
    load_shared_s32x2,
    load_shared_s32x4,
    mma_bf16_1688,
    sigmoid,
    store_shared_f32x2,
    vote_any,
)

WARP = 32
WARPS = 4
THREADS = WARP * WARPS
HEAD_DIM = 256
MAX_GROUP = 16  # query heads of a KV head at most (the MMA's M)
STAGE = 32  # selection columns of a stage
WARP_POSITIONS = STAGE // WARPS  # columns of a warp in a stage (the MMA's N)
STAGES = 3  # stage buffers
ROW_BYTES = HEAD_DIM * 2  # one position's K (or V) row
WARP_STAGE_BYTES = WARP_POSITIONS * ROW_BYTES
STAGE_BYTES = STAGE * ROW_BYTES
PART_STRIDE = HEAD_DIM + 8  # FP32 words of a merge partial row
LOG2E = 1.4426950408889634
SLOT_LIMIT = 2**31  # slots are int32

# Shared memory, bytes: the K and V stage buffers (the merge partials reuse
# them), the warps' slot arrays, the warps' maxima and sums.
OFF_K = 0
OFF_V = OFF_K + STAGES * STAGE_BYTES
OFF_SLOTS = OFF_V + STAGES * STAGE_BYTES
assert WARPS * MAX_GROUP * PART_STRIDE * 4 <= OFF_SLOTS


def max_stages(topk: int) -> int:
    """Stages of a row with `topk` selection columns."""
    return -(-topk // STAGE)


def shared_bytes(topk: int) -> int:
    """Shared memory of a CTA for `topk` selection columns."""
    return OFF_SLOTS + WARPS * max_stages(topk) * WARP_POSITIONS * 4 + 2 * WARPS * 64


@cache
def score_scale(head_dim: int) -> float:
    """The FP32 product `head_dim^-0.5 * log2(e)` (stock's score scale)."""
    scale = torch.tensor(head_dim**-0.5, dtype=torch.float32)
    return float(scale * torch.tensor(LOG2E, dtype=torch.float32))


class QsaRowAttention:
    """Sparse paged attention of one (row, KV head) per CTA (module docstring).

    Args:
        group: Query heads of a KV head (1 to 16).
        topk: Selection columns of a row (the packed width minus the count).

    """

    def __init__(self, group: int, topk: int) -> None:
        assert 1 <= group <= MAX_GROUP and topk >= 1
        self.group = group
        self.topk = topk
        self.stages = max_stages(topk)
        self.slot_bytes = self.stages * WARP_POSITIONS * 4  # one warp's slots
        self.off_m = OFF_SLOTS + WARPS * self.slot_bytes
        self.off_l = self.off_m + WARPS * 64
        self.shared_bytes = shared_bytes(topk)
        assert self.off_l + WARPS * 64 == self.shared_bytes
        # Rounds of 32 slots a lane computes (the warp's positions over lanes).
        self.slot_rounds = -(-self.stages * WARP_POSITIONS // WARP)

    @cute.jit
    def __call__(
        self,
        q: Int64,
        q_row: Int32,
        q_head: Int32,
        k: Int64,
        v: Int64,
        kv_token: Int32,
        kv_head: Int32,
        indices: Int64,
        indices_row: Int32,
        table: Int64,
        table_row: Int32,
        page: Int32,
        width: Int32,
        token_to_req: Int64,
        gate: Int64,
        gate_row: Int32,
        gate_head: Int32,
        out: Int64,
        out_row: Int32,
        out_head: Int32,
        requests: Int32,
        blocks: Int32,
        block_tokens: Int32,
        scale: Float32,
        rows: Int32,
        kv_heads: Int32,
        stream: cuda.CUstream,
    ):
        """Launch one CTA per (row, KV head) on `stream` (addresses of BF16
        or int32 data, strides in elements)."""
        self.kernel(
            q,
            q_row,
            q_head,
            k,
            v,
            kv_token,
            kv_head,
            indices,
            indices_row,
            table,
            table_row,
            page,
            width,
            token_to_req,
            gate,
            gate_row,
            gate_head,
            out,
            out_row,
            out_head,
            requests,
            blocks,
            block_tokens,
            scale,
        ).launch(
            grid=(rows, kv_heads, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=2,
        )

    @cute.jit
    def issue(self, stage: Int32, ctx):
        """Issue the copies of the warp's K and V rows of `stage`: lane l
        copies chunk l of each row; row j of K keeps chunk x at
        `x ^ 4 (j & 1)`, row j of V at `x ^ j`, so the stage's fragment
        loads are free of bank conflicts. A rejected slot (-1) copies nothing
        and zero-fills its chunks."""
        slots, k_smem, v_smem, k_at, v_at, token_bytes, lane = ctx
        buffer = (stage % STAGES) * STAGE_BYTES
        first = load_shared_s32x4(slots + stage * (WARP_POSITIONS * 4))
        second = load_shared_s32x4(slots + stage * (WARP_POSITIONS * 4) + 16)
        positions = first + second
        for j in cutlass.range_constexpr(WARP_POSITIONS):
            position = positions[j]
            size = cutlass.select_(position >= 0, Int32(16), Int32(0))
            source = Int64(cute.max(position, Int32(0))) * token_bytes
            copy_async_16_fill(
                k_smem + buffer + j * ROW_BYTES + ((lane ^ (4 * (j & 1))) * 16),
                k_at + source,
                size,
            )
            copy_async_16_fill(
                v_smem + buffer + j * ROW_BYTES + ((lane ^ j) * 16),
                v_at + source,
                size,
            )

    @cute.kernel
    def kernel(
        self,
        q: Int64,
        q_row: Int32,
        q_head: Int32,
        k: Int64,
        v: Int64,
        kv_token: Int32,
        kv_head: Int32,
        indices: Int64,
        indices_row: Int32,
        table: Int64,
        table_row: Int32,
        page: Int32,
        width: Int32,
        token_to_req: Int64,
        gate: Int64,
        gate_row: Int32,
        gate_head: Int32,
        out: Int64,
        out_row: Int32,
        out_head: Int32,
        requests: Int32,
        blocks: Int32,
        block_tokens: Int32,
        scale: Float32,
    ):
        """One CTA: every selected position of one (row, KV head)."""
        G = self.group
        TOPK = self.topk
        tid, _, _ = cute.arch.thread_idx()
        row, kv, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // WARP)
        lane = tid % WARP
        g = lane // 4
        t = lane % 4
        shared = (
            cutlass.memory.SmemAllocator()
            .allocate_array(cutlass.Uint8, self.shared_bytes, byte_alignment=128)
            .toint()
        )
        k_smem = shared + OFF_K + warp * WARP_STAGE_BYTES
        v_smem = shared + OFF_V + warp * WARP_STAGE_BYTES
        slots = shared + OFF_SLOTS + warp * self.slot_bytes
        neg_inf = Float32(float("-inf"))

        # 1. Every input is written by earlier kernels of the step.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()

        # 2. The row, the query fragments and the warp's slots. The loads of
        # one round are independent, so their latencies overlap.
        selection = indices + Int64(row) * Int64(indices_row) * 4
        count = load_s32(selection + TOPK * 4)
        request = load_s32(token_to_req + Int64(row) * 4)
        # Lane (g, t): query heads g and g + 8 of the KV head (zero past the
        # group), dims 32 c + 8 t .. 32 c + 8 t + 7 of chunk c (hc._ptx's
        # mma_chunk layout).
        head_lo = kv * G + cute.min(g, Int32(G - 1))
        head_hi = kv * G + cute.min(g + 8, Int32(G - 1))
        q_lo_at = q + (Int64(row) * Int64(q_row) + Int64(head_lo) * Int64(q_head)) * 2
        q_hi_at = q + (Int64(row) * Int64(q_row) + Int64(head_hi) * Int64(q_head)) * 2
        lo_valid = g < G
        hi_valid = g + 8 < G
        q_lo = []
        q_hi = []
        for qc in cutlass.range_constexpr(HEAD_DIM // 32):
            lo_words = load_x4(q_lo_at + Int64(32 * qc + 8 * t) * 2)
            hi_words = load_x4(q_hi_at + Int64(32 * qc + 8 * t) * 2)
            q_lo.append(
                tuple(
                    cutlass.select_(lo_valid, lo_words[qe], Uint32(0))
                    for qe in range(4)
                )
            )
            q_hi.append(
                tuple(
                    cutlass.select_(hi_valid, hi_words[qe], Uint32(0))
                    for qe in range(4)
                )
            )
        # Local slot i = 8 s + j of the warp is column 32 s + 8 warp + j; lane
        # l computes slots l, l + 32, ...
        tokens = []
        for r in cutlass.range_constexpr(self.slot_rounds):
            i = lane + WARP * r
            column = (i // WARP_POSITIONS) * STAGE + warp * WARP_POSITIONS
            column = column + i % WARP_POSITIONS
            tokens.append(
                load_s32_if(selection + Int64(column) * 4, Int32(column < TOPK))
            )
        count = cute.min(cute.max(count, Int32(0)), Int32(TOPK))
        stages = (count + STAGE - 1) // STAGE
        request_ok = (request >= 0) & (request < requests)
        safe_request = cute.min(cute.max(request, Int32(0)), requests - 1)
        table_at = table + Int64(safe_request) * Int64(table_row) * 4
        keep = []
        pages = []
        physical = []
        for r in cutlass.range_constexpr(self.slot_rounds):
            i = lane + WARP * r
            token = tokens[r]
            logical = cute.max(token, Int32(0)) // page
            ok = (i < stages * WARP_POSITIONS) & request_ok & (token >= 0)
            ok = ok & (logical < width)
            keep.append(ok)
            pages.append(logical)
            physical.append(
                load_s32_if(
                    table_at + Int64(cute.min(logical, width - 1)) * 4,
                    Int32(ok),
                )
            )
        for r in cutlass.range_constexpr(self.slot_rounds):
            i = lane + WARP * r
            block = physical[r]
            ok = keep[r] & (block >= 0) & (block < blocks)
            slot = cutlass.select_(
                ok, block * block_tokens + tokens[r] - pages[r] * page, Int32(-1)
            )
            if i < stages * WARP_POSITIONS:
                store_shared_s32(slots + i * 4, slot)
        cute.arch.sync_warp()

        # 3. The stage pipeline. Lane l copies chunk l of each of the warp's
        # eight K and V rows of a stage; row j of K keeps chunk x at
        # x ^ 4 (j & 1), row j of V at x ^ j (conflict-free fragment loads).
        k_at = k + Int64(kv) * Int64(kv_head) * 2 + Int64(lane * 16)
        v_at = v + Int64(kv) * Int64(kv_head) * 2 + Int64(lane * 16)
        token_bytes = Int64(kv_token) * 2

        ctx = (slots, k_smem, v_smem, k_at, v_at, token_bytes, lane)
        for s in cutlass.range_constexpr(STAGES - 1):
            if s < stages:
                self.issue(Int32(s), ctx)
            copy_async_commit()

        # Lane (g, t)'s K fragment: row g, chunk 4 c + t (swizzled); its
        # ldmatrix row: lane & 7 of matrix lane >> 3.
        k_read = g * ROW_BYTES + t * 16
        k_flip = (g & 1) * 64
        v_row = lane % 8
        v_matrix = lane // 8
        o = cute.make_rmem_tensor((4 * HEAD_DIM // 8,), Float32)
        o.fill(0.0)
        # Running maxima and sums of heads g and g + 8.
        state = cute.make_rmem_tensor((4,), Float32)
        state[0] = neg_inf
        state[1] = neg_inf
        state[2] = Float32(0.0)
        state[3] = Float32(0.0)
        current = Int32(0)
        while current < stages:
            copy_async_wait_group(STAGES - 2)
            # Every lane's copies of this stage landed; every lane is done
            # with the buffer the next issue overwrites.
            cute.arch.sync_warp()
            if current + STAGES - 1 < stages:
                self.issue(current + STAGES - 1, ctx)
            copy_async_commit()
            buffer = (current % STAGES) * STAGE_BYTES
            k_stage = k_smem + buffer
            v_stage = v_smem + buffer

            # S = Q K^T for the warp's 8 positions: four interleaved chains.
            partial = []
            for _sa in cutlass.range_constexpr(4):
                partial.append((Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)))
            for sc in cutlass.range_constexpr(HEAD_DIM // 32):
                x = load_shared_x4(k_stage + k_read + ((64 * sc) ^ k_flip))
                partial[sc % 4] = mma_chunk(partial[sc % 4], q_lo[sc], q_hi[sc], x)
            # Lane (g, t) holds the scores of heads g, g + 8 at positions
            # 2 t, 2 t + 1.
            seen = load_shared_s32x2(slots + current * (WARP_POSITIONS * 4) + t * 8)
            scores = []
            for se in cutlass.range_constexpr(4):
                total = (partial[0][se] + partial[1][se]) + (
                    partial[2][se] + partial[3][se]
                )
                scores.append(
                    cutlass.select_(seen[se % 2] >= 0, total * scale, neg_inf)
                )
            top_lo = cute.arch.fmax(scores[0], scores[1])
            top_hi = cute.arch.fmax(scores[2], scores[3])
            for so in (1, 2):
                top_lo = cute.arch.fmax(
                    top_lo, cute.arch.shuffle_sync_bfly(top_lo, offset=so)
                )
                top_hi = cute.arch.fmax(
                    top_hi, cute.arch.shuffle_sync_bfly(top_hi, offset=so)
                )
            ref_lo = state[0]
            ref_hi = state[1]
            grow_lo = top_lo > ref_lo
            grow_hi = top_hi > ref_hi
            if vote_any(Int32(grow_lo) | Int32(grow_hi)) != 0:
                alpha_lo = cutlass.select_(
                    grow_lo, exp2_ftz(ref_lo - top_lo), Float32(1.0)
                )
                alpha_hi = cutlass.select_(
                    grow_hi, exp2_ftz(ref_hi - top_hi), Float32(1.0)
                )
                state[0] = cutlass.select_(grow_lo, top_lo, ref_lo)
                state[1] = cutlass.select_(grow_hi, top_hi, ref_hi)
                state[2] = state[2] * alpha_lo
                state[3] = state[3] * alpha_hi
                for rj in cutlass.range_constexpr(HEAD_DIM // 8):
                    o[4 * rj] = o[4 * rj] * alpha_lo
                    o[4 * rj + 1] = o[4 * rj + 1] * alpha_lo
                    o[4 * rj + 2] = o[4 * rj + 2] * alpha_hi
                    o[4 * rj + 3] = o[4 * rj + 3] * alpha_hi
            m_lo = state[0]
            m_hi = state[1]
            probability = []
            for pe in cutlass.range_constexpr(4):
                m_e = m_lo if pe < 2 else m_hi
                probability.append(
                    cutlass.select_(
                        seen[pe % 2] >= 0, exp2_ftz(scores[pe] - m_e), Float32(0.0)
                    )
                )
            # P as BF16 (the A fragment); the sums add the FP32 values.
            p_lo = pack_bf16x2(probability[0], probability[1])
            p_hi = pack_bf16x2(probability[2], probability[3])
            state[2] = state[2] + (probability[0] + probability[1])
            state[3] = state[3] + (probability[2] + probability[3])

            # O += P V over all 256 dims: dims 8 j .. 8 j + 7 are block j.
            for vm in cutlass.range_constexpr(HEAD_DIM // 32):
                chunk = (v_matrix + 4 * vm) ^ v_row
                vt = load_matrix_x4_trans(v_stage + v_row * ROW_BYTES + chunk * 16)
                for ve in cutlass.range_constexpr(4):
                    vj = 4 * vm + ve
                    acc = mma_bf16_1688(
                        (o[4 * vj], o[4 * vj + 1], o[4 * vj + 2], o[4 * vj + 3]),
                        (p_lo, p_hi),
                        vt[ve],
                    )
                    for vf in cutlass.range_constexpr(4):
                        o[4 * vj + vf] = acc[vf]
            current = current + 1

        # 4. The merge. Every warp is done with the stage buffers once every
        # copy landed and every thread passed the barrier.
        copy_async_wait_group(0)
        cute.arch.sync_threads()
        sum_lo = state[2]
        sum_hi = state[3]
        for mo in (1, 2):
            sum_lo = sum_lo + cute.arch.shuffle_sync_bfly(sum_lo, offset=mo)
            sum_hi = sum_hi + cute.arch.shuffle_sync_bfly(sum_hi, offset=mo)
        part = shared + OFF_K + warp * (MAX_GROUP * PART_STRIDE * 4)
        for mj in cutlass.range_constexpr(HEAD_DIM // 8):
            dims = 8 * mj + 2 * t
            store_shared_f32x2(
                part + (g * PART_STRIDE + dims) * 4, (o[4 * mj], o[4 * mj + 1])
            )
            store_shared_f32x2(
                part + ((g + 8) * PART_STRIDE + dims) * 4,
                (o[4 * mj + 2], o[4 * mj + 3]),
            )
        if t == 0:
            store_shared_f32(shared + self.off_m + (warp * 16 + g) * 4, state[0])
            store_shared_f32(shared + self.off_m + (warp * 16 + g + 8) * 4, state[1])
            store_shared_f32(shared + self.off_l + (warp * 16 + g) * 4, sum_lo)
            store_shared_f32(shared + self.off_l + (warp * 16 + g + 8) * 4, sum_hi)
        cute.arch.sync_threads()

        # Thread i finishes dims 2 i, 2 i + 1 of every query head.
        dim = 2 * tid
        for h in cutlass.range_constexpr(G):
            maxima = []
            for w in cutlass.range_constexpr(WARPS):
                maxima.append(load_shared_f32(shared + self.off_m + (w * 16 + h) * 4))
            top = maxima[0]
            for w in cutlass.range_constexpr(1, WARPS):
                top = cute.arch.fmax(top, maxima[w])
            num_lo = Float32(0.0)
            num_hi = Float32(0.0)
            den = Float32(0.0)
            for w in cutlass.range_constexpr(WARPS):
                weight = cutlass.select_(
                    maxima[w] != neg_inf, exp2_ftz(maxima[w] - top), Float32(0.0)
                )
                values = load_shared_f32x2(
                    shared + OFF_K + ((w * MAX_GROUP + h) * PART_STRIDE) * 4 + dim * 4
                )
                num_lo = num_lo + values[0] * weight
                num_hi = num_hi + values[1] * weight
                den = (
                    den
                    + load_shared_f32(shared + self.off_l + (w * 16 + h) * 4) * weight
                )
            inverse = Float32(1.0) / cute.arch.fmax(den, Float32(1.0e-20))
            y_lo = round_bf16(
                cutlass.select_(den > 0.0, num_lo * inverse, Float32(0.0))
            )
            y_hi = round_bf16(
                cutlass.select_(den > 0.0, num_hi * inverse, Float32(0.0))
            )
            head = kv * G + h
            gate_word = load_u32(
                gate
                + (
                    Int64(row) * Int64(gate_row)
                    + Int64(head) * Int64(gate_head)
                    + Int64(dim)
                )
                * 2
            )
            gate_lo, gate_hi = widen_bf16x2(gate_word)
            store_u32(
                out
                + (
                    Int64(row) * Int64(out_row)
                    + Int64(head) * Int64(out_head)
                    + Int64(dim)
                )
                * 2,
                pack_bf16x2(y_lo * sigmoid(gate_lo), y_hi * sigmoid(gate_hi)),
            )


@cache
def _compiled(device_index: int, group: int, topk: int):
    """`QsaRowAttention` compiled for a device and shape."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            QsaRowAttention(group, topk),
            Int64(0),
            Int32(8),
            Int32(8),
            Int64(0),
            Int64(0),
            Int32(8),
            Int32(8),
            Int64(0),
            Int32(1),
            Int64(0),
            Int32(1),
            Int32(1),
            Int32(1),
            Int64(0),
            Int64(0),
            Int32(8),
            Int32(8),
            Int64(0),
            Int32(8),
            Int32(8),
            Int32(1),
            Int32(1),
            Int32(1),
            Float32(1.0),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


def row_attention_unsupported(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
    output_gate: torch.Tensor,
) -> str:
    """What `qsa_row_attention` needs and these operands miss, or `""`.

    Args:
        q: The queries `[rows, heads, 256]`.
        k_cache: The key cache view `[blocks, page, kv_heads, 256]`.
        v_cache: The value cache view, with the key cache's shape and strides.
        indices: The packed selection `[rows, topk + 1]`.
        block_table: The block table `[requests, width]`.
        out: The output, with the queries' shape.
        output_gate: The gate logits, one per output value.

    """
    missing = []
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        missing.append("BF16 queries and a BF16 K/V cache")
    if v_cache.dtype != k_cache.dtype or v_cache.shape != k_cache.shape:
        missing.append("a value cache with the key cache's dtype and shape")
    if v_cache.stride() != k_cache.stride():
        missing.append("a value cache with the key cache's strides")
    if q.ndim != 3 or k_cache.ndim != 4 or q.shape[2] != HEAD_DIM:
        missing.append(f"{HEAD_DIM}-dim heads")
    elif k_cache.shape[3] != HEAD_DIM or q.shape[1] % k_cache.shape[2]:
        missing.append("grouped query heads over 256-dim KV heads")
    elif q.shape[1] // k_cache.shape[2] > MAX_GROUP:
        missing.append(f"at most {MAX_GROUP} query heads a KV head")
    if k_cache.ndim == 4:
        blocks, page, _, _ = k_cache.shape
        token = k_cache.stride(1)
        if k_cache.stride(3) != 1 or token % 8 or k_cache.stride(2) % 8:
            missing.append("unit dim stride and 16-byte aligned cache rows")
        elif k_cache.stride(0) % token:
            missing.append("a block stride that is a multiple of the token stride")
        elif blocks * (k_cache.stride(0) // token) >= SLOT_LIMIT:
            missing.append("cache slots that fit in int32")
        if k_cache.data_ptr() % 16 or v_cache.data_ptr() % 16:
            missing.append("16-byte aligned caches")
    if q.ndim == 3 and (
        q.stride(2) != 1 or q.stride(1) % 8 or q.stride(0) % 8 or q.data_ptr() % 16
    ):
        missing.append("16-byte aligned query rows")
    if indices.dtype != torch.int32 or block_table.dtype != torch.int32:
        missing.append("int32 selections and block table")
    if indices.ndim != 2 or indices.stride(1) != 1 or indices.shape[1] < 2:
        missing.append("a packed selection with unit column stride")
    if block_table.ndim != 2 or block_table.stride(1) != 1:
        missing.append("a block table with unit column stride")
    # The epilogue loads and stores the two dims of a thread as one 32-bit word.
    if (
        out.dtype != torch.bfloat16
        or out.ndim != 3
        or out.shape != q.shape
        or out.stride(2) != 1
        or out.stride(1) % 2
        or out.stride(0) % 2
        or out.data_ptr() % 4
    ):
        missing.append("a BF16 output with the queries' shape and 4-byte aligned pairs")
    if (
        output_gate.dtype != torch.bfloat16
        or not output_gate.is_contiguous()
        or output_gate.numel() != q.numel()
        or output_gate.data_ptr() % 4
    ):
        missing.append("contiguous BF16 gate logits, one per output value")
    return "; ".join(missing)


def qsa_row_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
    *,
    output_gate: torch.Tensor,
) -> torch.Tensor:
    """Gated sparse paged attention of every row (module docstring), with the
    operands of the stock `qsa_sparse_paged_attention` for a BF16 cache.

    Args:
        q: BF16 queries `[rows, heads, 256]`.
        k_cache: The key cache view `[blocks, page, kv_heads, 256]`.
        v_cache: The value cache view (the key cache's shape and strides).
        indices: The packed selection `[rows, topk + 1]` (int32; the last
            column holds each row's count).
        block_table: The block table `[requests, width]` (int32).
        token_to_req: Each row's request `[rows]` (int32).
        out: The BF16 output `[rows, heads, 256]` (written in full).
        output_gate: The BF16 gate logits, contiguous, `rows * heads * 256`
            values.

    Returns:
        `out`.

    """
    missing = row_attention_unsupported(
        q, k_cache, v_cache, indices, block_table, out, output_gate
    )
    assert not missing, missing
    rows, heads, _ = q.shape
    assert indices.shape[0] == rows and token_to_req.shape == (rows,)
    assert token_to_req.dtype == torch.int32 and token_to_req.stride(0) == 1
    gate = output_gate.view_as(q)
    if rows == 0:
        return out
    blocks, page, kv_heads, _ = k_cache.shape
    _compiled(q.device.index, heads // kv_heads, indices.shape[1] - 1)(
        Int64(q.data_ptr()),
        Int32(q.stride(0)),
        Int32(q.stride(1)),
        Int64(k_cache.data_ptr()),
        Int64(v_cache.data_ptr()),
        Int32(k_cache.stride(1)),
        Int32(k_cache.stride(2)),
        Int64(indices.data_ptr()),
        Int32(indices.stride(0)),
        Int64(block_table.data_ptr()),
        Int32(block_table.stride(0)),
        Int32(page),
        Int32(block_table.shape[1]),
        Int64(token_to_req.data_ptr()),
        Int64(gate.data_ptr()),
        Int32(gate.stride(0)),
        Int32(gate.stride(1)),
        Int64(out.data_ptr()),
        Int32(out.stride(0)),
        Int32(out.stride(1)),
        Int32(block_table.shape[0]),
        Int32(blocks),
        Int32(k_cache.stride(0) // k_cache.stride(1)),
        Float32(score_scale(HEAD_DIM)),
        Int32(rows),
        Int32(kv_heads),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return out
