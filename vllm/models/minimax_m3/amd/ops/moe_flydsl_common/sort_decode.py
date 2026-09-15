# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode routing sort (M <= max_tokens), one launch: block 0 sorts the (token, slot)
pairs by expert (LDS histogram, Hillis-Steele scan of the padded counts, placement
through LDS cursors, padding); the other blocks zero the output gemm2 accumulates
into. aiter ``moe_sorting`` contract:

  sorted_ids[row]        = token | slot << 24   (padding rows: token = n_tokens)
  sorted_weights[row]    = routing weight       (padding rows: 0)
  sorted_expert_ids[b]   = expert of block b (block_m rows)
  num_valid_ids[0]       = padded sorted row count
  out[n_tokens, H]       = 0 (bf16)

``wide_first=(expert, wide_bm)`` puts that expert's rows first in wide_bm-row blocks.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.sort import max_sorted_rows
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.utils import _lds_atomic_add_i32


def wide_layout_rows(n_tokens: int, E: int, topk: int, block_m: int, wide_bm: int):
    """Row bounds of the wide-first layout: (rows of the shared expert = its
    ceil(n_tokens/wide_bm) wide blocks, max rows of the routed experts in
    block_m blocks). The shared expert owns one of the ``topk`` slots of every
    token."""
    shared = ((n_tokens + wide_bm - 1) // wide_bm) * wide_bm
    routed_pairs = n_tokens * (topk - 1)
    active = min(E - 1, routed_pairs)
    routed = (
        (routed_pairs + active * (block_m - 1) + block_m - 1) // block_m
    ) * block_m
    return shared, routed


threads = 256  # sorter block (one thread per expert, power of two for the scan)
zero_ctas = 127  # blocks that zero the output


@functools.cache
def compile_decode_sort(
    *, E: int, topk: int, block_m: int, H: int, max_tokens: int, wide_first=None
):
    """``wide_first = (expert, wide_bm)`` puts that expert's rows first, padded
    to ``wide_bm``-row blocks (the shared expert, which every token routes to,
    so the gemms can run it in wide blocks); the other experts follow in
    ``block_m`` blocks. Block ids: the wide blocks are 0 .. S-1, S =
    padded_rows / wide_bm, then one per block_m block."""
    assert (block_m & (block_m - 1)) == 0 and threads >= E and (H * 2) % 16 == 0
    bm_shift = block_m.bit_length() - 1
    PPT = (max_tokens * topk + threads - 1) // threads  # pairs per thread
    scan_rounds = threads.bit_length() - 1
    tag = f"E{E}_K{topk}_BM{block_m}_H{H}_M{max_tokens}_Z{zero_ctas}_T{threads}"
    if wide_first is not None:
        shared_e, wide_bm = wide_first
        assert (wide_bm & (wide_bm - 1)) == 0 and 0 <= shared_e < E
        wide_shift = wide_bm.bit_length() - 1
        tag += f"_wide{wide_bm}s{shared_e}"

    def slot_of(e):  # scan slot of expert e: the wide expert first, then id order
        if wide_first is None:
            return e
        return (e + fx.Int32(E - shared_e)) % fx.Int32(E)

    @fx.struct
    class Shared:
        count: fx.Array[fx.Int32, E]  # rows per expert
        cursor: fx.Array[fx.Int32, E]  # next free sorted row per expert
        scan: fx.Array[fx.Int32, 2 * threads]  # ping-pong scan arrays

    @flyc.kernel(name=f"m3_decode_sort_zero_{tag}", known_block_size=[threads, 1, 1])
    def sort_zero(
        topk_ids: fx.Tensor,  # [n_tokens * topk] i32
        topk_w: fx.Tensor,  # [n_tokens * topk] f32
        sorted_ids: fx.Tensor,  # [max_sorted] i32
        sorted_w: fx.Tensor,  # [max_sorted] f32
        sorted_eids: fx.Tensor,  # [max_sorted / block_m] i32
        num_valid: fx.Tensor,  # [2] i32
        out_i32: fx.Tensor,  # out[n_tokens, H] bf16 viewed as i32 [n_tokens * H / 2]
        n_tok: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        if bx == 0:
            count, cursor = smem.count.ptr, smem.cursor.ptr
            scan = [smem.scan.ptr, smem.scan.ptr + threads]
            n_pairs = n_tok * topk
            last_pair = n_pairs - 1
            t_lt_E = tx < E

            # 0. this thread's pairs: one batch of loads (clamped index, validity)
            pairs = []
            for k in range_constexpr(PPT):
                p = tx + k * threads
                valid = p < n_pairs
                pc = valid.select(p, last_pair)
                pairs.append((p, valid, fx.Int32(topk_ids[pc]), fx.Float32(topk_w[pc])))

            # 1. zero the counters
            if t_lt_E:
                count[tx] = 0
            fx.gpu.barrier()

            # 2. histogram (by scan slot)
            for p, valid, e, _w in pairs:
                if valid:
                    _lds_atomic_add_i32(count + slot_of(e), 1)
            fx.gpu.barrier()

            # 3. inclusive Hillis-Steele scan of the padded counts (0 beyond E)
            cnt = t_lt_E.select(
                fx.Int32(count[t_lt_E.select(tx, fx.Int32(0))]), fx.Int32(0)
            )
            padded = (cnt + (block_m - 1)) & ~(block_m - 1)
            if const_expr(wide_first is not None):
                padded = (tx == 0).select(
                    (cnt + (wide_bm - 1)) & ~(wide_bm - 1), padded
                )
            scan[0][tx] = padded
            fx.gpu.barrier()
            src, dst = 0, 1
            for r in range_constexpr(scan_rounds):
                d = 1 << r
                mine = fx.Int32(scan[src][tx])
                has = tx >= d
                other = fx.Int32(scan[src][has.select(tx - d, tx)])
                scan[dst][tx] = has.select(mine + other, mine)
                fx.gpu.barrier()
                src, dst = dst, src
            end = fx.Int32(scan[src][tx])
            start = end - padded

            # 4. thread t (slot t): cursor, expert ids of its blocks, padding rows,
            # total
            if const_expr(wide_first is None):
                e_t = tx
                b_lo, b_hi = start >> bm_shift, end >> bm_shift
            else:
                e_t = (tx + shared_e) % E
                p0 = fx.Int32(scan[src][0])  # rows of the wide expert (S blocks)
                n_wide = p0 >> wide_shift
                is0 = tx == 0
                b_lo = is0.select(fx.Int32(0), n_wide + ((start - p0) >> bm_shift))
                b_hi = is0.select(n_wide, n_wide + ((end - p0) >> bm_shift))
            if t_lt_E:
                cursor[tx] = start
                if tx == E - 1:
                    num_valid[0] = end
                for b in range(b_lo, b_hi):
                    sorted_eids[fx.Int32(b)] = e_t
                for r in range(start + cnt, end):
                    sorted_ids[fx.Int32(r)] = n_tok  # padding: token n_tokens, slot 0
                    sorted_w[fx.Int32(r)] = fx.Float32(0.0)
            fx.gpu.barrier()

            # 5. place the pairs
            for p, valid, e, w in pairs:
                if valid:
                    row = _lds_atomic_add_i32(cursor + slot_of(e), 1)
                    sorted_ids[row] = ((p // topk) & 0x00FFFFFF) | ((p % topk) << 24)
                    sorted_w[row] = w
        else:
            # blocks 1..zero_ctas: out[n_tokens, H] = 0, 16 B per store
            # 16 B per store through a (4, n/4) view (no vector-element pointer store
            # in this flydsl)
            out16 = fx.logical_divide(out_i32, fx.make_layout(4, 1))
            atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
            zero_r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            zero_r.store(fx.Vector.filled(4, 0, fx.Int32))
            n_chunks = n_tok * (H * 2 // 16)
            for i in range((bx - 1) * threads + tx, n_chunks, zero_ctas * threads):
                fx.copy(atom, zero_r, fx.slice(out16, (None, fx.Int32(i))))

    @flyc.jit
    def launch(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_w: fx.Tensor,
        sorted_eids: fx.Tensor,
        num_valid: fx.Tensor,
        out_i32: fx.Tensor,
        n_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        sort_zero(
            topk_ids,
            topk_w,
            sorted_ids,
            sorted_w,
            sorted_eids,
            num_valid,
            out_i32,
            n_tokens,
        ).launch(grid=(1 + zero_ctas, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


def moe_sort_decode(topk_ids, topk_weights, E, H, block_m, out, wide_first=None):
    """Drop-in for aiter ``moe_sorting`` -> (sorted_ids, sorted_weights,
    sorted_expert_ids, num_valid_ids); ``out`` (``[n_tokens, H]`` bf16, contiguous)
    is zeroed in place. With ``wide_first`` ``sorted_expert_ids`` has one entry
    per wide block followed by one per block_m block.
    """
    n_tokens, topk = topk_ids.shape
    dev = topk_ids.device
    if wide_first is None:
        ms = max_sorted_rows(n_tokens, E, topk, block_m)
        n_blocks = ms // block_m
    else:
        shared_rows, routed_rows = wide_layout_rows(
            n_tokens, E, topk, block_m, wide_first[1]
        )
        ms = shared_rows + routed_rows
        n_blocks = shared_rows // wide_first[1] + routed_rows // block_m
    sorted_ids = torch.empty(ms, dtype=torch.int32, device=dev)
    sorted_w = torch.empty(ms, dtype=torch.float32, device=dev)
    sorted_eids = torch.empty(n_blocks, dtype=torch.int32, device=dev)
    num_valid = torch.empty(2, dtype=torch.int32, device=dev)
    assert n_tokens <= 256
    launch = compile_decode_sort(
        E=E,
        topk=topk,
        block_m=block_m,
        H=H,
        max_tokens=64 if n_tokens <= 64 else 256,
        wide_first=wide_first,
    )
    launch(
        topk_ids.contiguous().int().view(-1),
        topk_weights.contiguous().float().view(-1),
        sorted_ids,
        sorted_w,
        sorted_eids,
        num_valid,
        out.view(torch.int32).view(-1),
        int(n_tokens),
        torch.cuda.current_stream(),
    )
    return sorted_ids, sorted_w, sorted_eids, num_valid
