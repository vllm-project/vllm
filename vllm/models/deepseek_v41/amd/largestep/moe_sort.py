# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decode step's picks sorted into groups of 16 picks of one expert, on
the GPU, bit for bit the table of `pick_groups` in the tests'
`dsv41_largestep_reference`, so the routed MoE needs no host step inside a CUDA
graph. Table int32 [G, 4 + 16]: the
expert, three zeros, then 16 picks (-1 past the last). `count` gets G.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.deepseek_v41.amd.mono.attention.device import kernel_symbol
from vllm.models.deepseek_v41.amd.mono.common.ops import rsrc

from .shapes import EXPERTS, GROUP, GROUP_WORDS, TOPK

THREADS = 512
MAX_ROWS = 256
MAX_PICKS = MAX_ROWS * TOPK
_STREAM = fx.Stream(None)


# One launcher a shape for the process: the 40 layers share it, and it stays
# alive as long as the CUDA graphs that launch it (FlyDSL unloads a kernel's
# code object when its launcher is collected).
@functools.cache
def build_moe_sort(experts: int = EXPERTS):
    assert experts <= THREADS
    name = kernel_symbol("dsv41_ls_moe_sort", e=experts)

    @fx.struct
    class _Lds:
        ids: fx.Array[fx.Int32, MAX_PICKS, 16]
        cnt: fx.Array[fx.Int32, THREADS, 16]
        scan: fx.Array[fx.Int32, 2 * THREADS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(topk_ids: Int64, picks: Int32, table: Int64, count: Int64):
        lds = fx.SharedAllocator().allocate(_Lds).peek()
        ids, cnt, scan = lds.ids.ptr, lds.cnt.ptr, lds.scan.ptr
        tid = fx.thread_idx.x
        fx.ptr_store(fx.Int32(0), cnt + tid)
        gpu.barrier()
        # 1. count. The router gives CUDA graph padding rows expert -1
        # (VLLM_MOE_SKIP_PADDING). Such a pick is not counted: -1 as an LDS
        # index would add to the word before `cnt`. Step 3 never matches it,
        # so it is left out of every group.
        for i in range(tid, picks, THREADS):
            e = fx.Int32(bo.buffer_load(rsrc(topk_ids), i, vec_width=1, dtype=T.i32))
            fx.ptr_store(e, ids + i)
            if (e >= 0) & (e < experts):
                fx.llvm.atomic_add(cnt + e, fx.Int32(1), syncscope="workgroup")
        gpu.barrier()
        # 2. groups an expert, then an inclusive prefix sum (two LDS buffers,
        # each step reads one and writes the other)
        n = fx.ptr_load(cnt + tid)
        ng = (tid < experts).select((n + GROUP - 1) // GROUP, fx.Int32(0))
        fx.ptr_store(ng, scan + tid)
        gpu.barrier()
        src = 0
        off = 1
        while off < THREADS:
            v = fx.ptr_load(scan + src * THREADS + tid)
            if tid >= off:
                v = v + fx.ptr_load(scan + src * THREADS + tid - off)
            fx.ptr_store(v, scan + (1 - src) * THREADS + tid)
            gpu.barrier()
            src = 1 - src
            off = off * 2
        incl = fx.ptr_load(scan + src * THREADS + tid)
        first = incl - ng
        if tid == THREADS - 1:
            bo.buffer_store(incl, rsrc(count), 0)
        # 3. place: an expert's picks in pick order
        if (tid < experts) & (ng > 0):
            for gi in range(0, ng, 1):
                row = (first + gi) * GROUP_WORDS
                bo.buffer_store(tid, rsrc(table), row)
                for z in range(1, 4, 1):
                    bo.buffer_store(fx.Int32(0), rsrc(table), row + z)
                for z in range(0, GROUP, 1):
                    bo.buffer_store(fx.Int32(-1), rsrc(table), row + 4 + z)
            k = fx.Int32(0)
            for p in range(0, picks, 1):
                if fx.ptr_load(ids + p) == tid:
                    at = (first + k // GROUP) * GROUP_WORDS + 4 + k % GROUP
                    bo.buffer_store(p, rsrc(table), at)
                    k = k + 1
        return

    @flyc.jit
    def launch(
        topk_ids: Int64,
        picks: Int32,
        table: Int64,
        count: Int64,
        stream: fx.Stream = _STREAM,
    ):
        kernel(topk_ids, picks, table, count).launch(
            grid=(1,), block=(THREADS,), stream=stream
        )

    return launch
