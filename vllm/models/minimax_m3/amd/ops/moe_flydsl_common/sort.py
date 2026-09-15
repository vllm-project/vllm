# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill routing sort, aiter's 3-stage ``moe_3stage_sort.cuh`` contract in two
launches: ``sort_count`` (each CTA counts the experts of its share of the (token,
slot) pairs) and ``sort_place_pad`` (each CTA: expert totals padded to block_m,
start rows by scan, placement through LDS cursors, padding; CTA 0 writes
``sorted_expert_ids`` per block, ``num_valid_ids[0]`` = padded total and
``num_valid_ids[1]`` = the last expert's first row).

Outputs: ``sorted_ids[row] = token | slot << 24`` (padding rows: ``n_tokens``),
``sorted_weights`` (padding 0), ``sorted_expert_ids``, ``num_valid_ids``. Rows inside
an expert come out in atomic arrival order (as in aiter); consumers do not rely on it.
"""

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.utils import _lds_atomic_add_i32

SORT_CTAS = 128  # histogram / placement blocks
THREADS = 1024  # >= E and >= block_m


def max_sorted_rows(n_tokens: int, E: int, topk: int, block_m: int) -> int:
    """aiter's bound: every active expert pads by < block_m rows."""
    active = min(E, n_tokens * topk)
    cumsum_max = n_tokens * topk + active * (block_m - 1)
    return ((cumsum_max + block_m - 1) // block_m) * block_m


@dataclass
class SortBuffers:
    sorted_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: (
        torch.Tensor
    )  # [2] i32: padded row count, the last expert's first row
    sorted_weights: torch.Tensor
    block_offsets: torch.Tensor  # workspace [E * SORT_CTAS]: per-CTA counts
    max_sorted: int
    block_m: int

    @staticmethod
    def allocate(n_tokens, E, topk, block_m, device):
        ms = max_sorted_rows(n_tokens, E, topk, block_m)
        i32 = torch.int32
        return SortBuffers(
            sorted_ids=torch.empty(ms, dtype=i32, device=device),
            sorted_expert_ids=torch.empty(ms // block_m, dtype=i32, device=device),
            num_valid_ids=torch.empty(
                2, dtype=i32, device=device
            ),  # written by sort_place_pad
            sorted_weights=torch.empty(ms, dtype=torch.float32, device=device),
            block_offsets=torch.empty(E * SORT_CTAS, dtype=i32, device=device),
            max_sorted=ms,
            block_m=block_m,
        )

    def launch_args(self, topk_ids, topk_weights, n_tokens):
        return (
            topk_ids.contiguous().int().view(-1),
            topk_weights.contiguous().float().view(-1),
            self.sorted_ids,
            self.sorted_expert_ids,
            self.num_valid_ids,
            self.sorted_weights,
            self.block_offsets,
            int(n_tokens),
            torch.cuda.current_stream(),
        )


@functools.cache
def compile_moe_sort(*, E: int, topk: int, block_m: int):
    assert (block_m & (block_m - 1)) == 0 and max(E, block_m) <= THREADS
    experts_per_cta = (E + SORT_CTAS - 1) // SORT_CTAS
    bm_shift = block_m.bit_length() - 1
    tag = f"E{E}_K{topk}_BM{block_m}"

    SCAN_W = THREADS  # the padded-count scan runs on the whole block (no branch)
    assert E <= SCAN_W and (SCAN_W & (SCAN_W - 1)) == 0
    PART = 4  # threads per expert summing the per-CTA counts in place_pad
    assert PART * E <= THREADS and SORT_CTAS % PART == 0 and PART * E <= 2 * SCAN_W
    scan_rounds = SCAN_W.bit_length() - 1

    @fx.struct
    class Shared:
        count: fx.Array[fx.Int32, E]  # per-CTA count / cursor
        total: fx.Array[fx.Int32, E]  # expert total
        padded: fx.Array[fx.Int32, E]
        prefix: fx.Array[
            fx.Int32, E
        ]  # rows of expert e placed by the CTAs before this one
        starts: fx.Array[fx.Int32, E + 1]  # expert start rows, [E] = padded total
        scan: fx.Array[
            fx.Int32, 2 * SCAN_W
        ]  # ping-pong scan arrays; first the PART sums

    def pair_range(n_tok, bx):
        """this CTA's [start, end) of the n_tok * topk routing pairs"""
        total = n_tok * topk
        per_cta = (total + (SORT_CTAS - 1)) // SORT_CTAS
        start = bx * per_cta
        end = start + per_cta
        return start, (end < total).select(end, total)

    @flyc.kernel(name=f"m3_sort_count_{tag}", known_block_size=[THREADS, 1, 1])
    def sort_count(topk_ids: fx.Tensor, block_offsets: fx.Tensor, n_tok: fx.Int32):
        count = fx.SharedAllocator().allocate(Shared).peek().count.ptr
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        start, end = pair_range(n_tok, bx)
        if tx < E:
            count[tx] = 0
        fx.gpu.barrier()
        for i in range(start + tx, end, THREADS):
            _lds_atomic_add_i32(count + fx.Int32(topk_ids[fx.Int32(i)]), 1)
        fx.gpu.barrier()
        if tx < E:
            block_offsets[tx * SORT_CTAS + bx] = fx.Int32(count[tx])

    @flyc.kernel(name=f"m3_sort_place_pad_{tag}", known_block_size=[THREADS, 1, 1])
    def sort_place_pad(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        block_offsets: fx.Tensor,
        num_valid: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_w: fx.Tensor,
        n_tok: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        cursor, total_c, padded_c = smem.count.ptr, smem.total.ptr, smem.padded.ptr
        prefix_c, starts = smem.prefix.ptr, smem.starts.ptr
        scan = [smem.scan.ptr, smem.scan.ptr + SCAN_W]
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        # 1. per expert: the total over the SORT_CTAS counts and the rows the CTAs
        #    before this one place; PART threads per expert, SORT_CTAS / PART
        #    counts each (independent loads), partial sums through LDS
        e, k = tx // PART, tx % PART
        if tx < PART * E:
            tot = fx.Int32(0)
            pre = fx.Int32(0)
            for i in range_constexpr(SORT_CTAS // PART):
                c = k * (SORT_CTAS // PART) + i
                v = fx.Int32(block_offsets[e * SORT_CTAS + c])
                tot = tot + v
                pre = pre + (c < bx).select(v, 0)
            scan[0][tx] = tot
            scan[1][tx] = pre
        fx.gpu.barrier()
        if tx < E:
            tot = fx.Int32(0)
            pre = fx.Int32(0)
            for i in range_constexpr(PART):
                tot = tot + fx.Int32(scan[0][tx * PART + i])
                pre = pre + fx.Int32(scan[1][tx * PART + i])
            total_c[tx] = tot
            padded_c[tx] = (tot + (block_m - 1)) & ~(block_m - 1)
            prefix_c[tx] = pre
        fx.gpu.barrier()
        # 2. inclusive Hillis-Steele scan of the padded counts (0 beyond E) over the
        #    block's THREADS lanes -> expert start rows (every CTA computes them)
        in_e = tx < E
        scan[0][tx] = in_e.select(fx.Int32(padded_c[in_e.select(tx, 0)]), 0)
        fx.gpu.barrier()
        for r in range_constexpr(scan_rounds):
            d, src, dst = 1 << r, r % 2, 1 - r % 2
            mine = fx.Int32(scan[src][tx])
            has = tx >= d
            other = fx.Int32(scan[src][has.select(tx - d, tx)])
            scan[dst][tx] = has.select(mine + other, mine)
            fx.gpu.barrier()
        incl = scan[scan_rounds % 2]
        if tx < E:
            start = fx.Int32(incl[tx]) - fx.Int32(padded_c[tx])
            starts[tx] = start
            cursor[tx] = start + fx.Int32(prefix_c[tx])
        if tx == 0:
            starts[E] = fx.Int32(incl[E - 1])
        fx.gpu.barrier()
        if bx == 0:
            if tx == 0:
                num_valid[0] = fx.Int32(starts[E])
                num_valid[1] = fx.Int32(starts[E - 1])  # the last expert's first row
            if tx < E:
                for b in range(
                    fx.Int32(starts[tx]) >> bm_shift,
                    fx.Int32(starts[tx + 1]) >> bm_shift,
                ):
                    sorted_expert_ids[fx.Int32(b)] = tx
        # 3. place this CTA's pairs
        start, end = pair_range(n_tok, bx)
        for i in range(start + tx, end, THREADS):
            p = fx.Int32(i)
            row = _lds_atomic_add_i32(cursor + fx.Int32(topk_ids[p]), 1)
            sorted_ids[row] = ((p // topk) & 0x00FFFFFF) | ((p % topk) << 24)
            sorted_w[row] = fx.Float32(topk_w[p])
        fx.gpu.barrier()
        # 4. padding: every expert pads by < block_m rows, lanes [0, block_m) do it
        if tx < block_m:
            for ee in range_constexpr(experts_per_cta):
                e = bx * experts_per_cta + ee
                if e < E:
                    lo = fx.Int32(starts[e]) + fx.Int32(total_c[e]) + tx
                    for r in range(lo, fx.Int32(starts[e + 1]), THREADS):
                        sorted_ids[fx.Int32(r)] = n_tok  # token n_tokens, slot 0
                        sorted_w[fx.Int32(r)] = fx.Float32(0.0)

    @flyc.jit
    def launch_sort(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid: fx.Tensor,
        sorted_w: fx.Tensor,
        block_offsets: fx.Tensor,
        n_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        sort_count(topk_ids, block_offsets, n_tokens).launch(
            grid=(SORT_CTAS, 1, 1), block=(THREADS, 1, 1), stream=stream
        )
        sort_place_pad(
            topk_ids,
            topk_w,
            block_offsets,
            num_valid,
            sorted_expert_ids,
            sorted_ids,
            sorted_w,
            n_tokens,
        ).launch(grid=(SORT_CTAS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch_sort
