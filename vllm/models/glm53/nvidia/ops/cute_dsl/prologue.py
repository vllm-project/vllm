# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode prologue: gather each query token's embedding row and RoPE row."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from . import _rt as rt
from .row_table import RowTableLayout

const_expr = cutlass.const_expr

HIDDEN = rt.Core.HIDDEN
ROPE_COLS = 64


@rt.record
class PrologueQ4Params:
    """Kernel parameters of ``prologue_q4``."""

    TokenIds: cute.Pointer = rt.ptr("int32", align=4)
    Position: cute.Pointer = rt.ptr("int32", align=4)
    Embedding: cute.Pointer = rt.ptr("bfloat16")
    RopeTable: cute.Pointer = rt.ptr("float32")
    Hidden: cute.Pointer = rt.ptr("bfloat16")
    Freq: cute.Pointer = rt.ptr("float32")
    rope_capacity: Int32
    RowTable: cute.Pointer = rt.ptr("int32", align=4)


@rt.record
class PrologueVllmParams:
    """Kernel parameters of ``prologue_vllm{,_q4}``."""

    TokenIds: cute.Pointer = rt.ptr("int32", align=4)
    Position: cute.Pointer = rt.ptr("int32", align=4)
    RopeTable: cute.Pointer = rt.ptr("float32")
    Hidden: cute.Pointer = rt.ptr("bfloat16")
    Freq: cute.Pointer = rt.ptr("float32")
    rope_capacity: Int32
    RowTable: cute.Pointer = rt.ptr("int32", align=4)
    EmbeddingShard0: Int64
    EmbeddingShard1: Int64
    EmbeddingShard2: Int64
    EmbeddingShard3: Int64
    EmbeddingShard4: Int64
    EmbeddingShard5: Int64
    EmbeddingShard6: Int64
    EmbeddingShard7: Int64

    @property
    def shards(self) -> list:
        return [getattr(self, f"EmbeddingShard{i}") for i in range(8)]


class PrologueKernel(rt.DecodeKernel):
    """One 256-thread CTA per row of the build's capacity ``ROWS``; the row table
    (``row_table.py``) says which rows are live.
    """

    BLOCK = 256
    MIN_BLOCKS_PER_MP = None
    #: Row capacity (Rmax) and request capacity (Bmax) of the build.
    ROWS = 4
    REQUESTS = 1
    PACKED = False

    @property
    def GRID(self) -> int:  # type: ignore[override]
        return self.ROWS

    @property
    def layout(self) -> RowTableLayout:
        return RowTableLayout(self.ROWS, self.REQUESTS)

    @cute.jit
    def prologue(self, p):
        tid, _, _ = cute.arch.thread_idx()
        query, _, _ = cute.arch.block_idx()
        rt.Sync.pdl_prologue()
        return rt.ThreadIds(tid, query, None, None), None

    @cute.jit
    def run(self, p, s, ids: rt.ThreadIds):
        tid, query = ids.tid, ids.bid
        layout = self.layout
        table = cute.make_tensor(
            p.RowTable,
            cute.make_layout(layout.table_words if self.PACKED else layout.words),
        )
        if query < table[0]:
            request = table[layout.row_request_word(query)]
            first = table[layout.first_row_word(request)]
            token = cute.make_tensor(p.TokenIds, cute.make_layout(self.ROWS))[query]
            words = cute.make_tensor(p.Position, cute.make_layout(2 * self.ROWS))
            start = words[2 * first]
            position = start + query - first
            if (
                (token < 0)
                | (token >= rt.Core.VOCAB)
                | (start < 0)
                | (position >= p.rope_capacity)
            ):
                rt.Sync.trap()
            self.copy_row(p, tid, query, token, position)
            if (tid == 0) & (query == first):
                table[layout.context_word(request)] = start + 1
            if const_expr(self.PACKED):  # noqa: SIM102
                if (tid == 0) & (query == 0):
                    table[layout.step_word] = table[layout.step_word] + 1

    @cute.jit
    def copy_row(self, p, tid, query, token, position):
        embedding_row = self.embedding_row(p, token)  # type: ignore[attr-defined]
        hidden = cute.make_tensor(
            p.Hidden,
            cute.make_layout((self.ROWS, 3, 256, 8), stride=(HIDDEN, 2048, 8, 1)),
        )
        vals = [rt.Convert.ld_bf16x8_f32(embedding_row[j, tid, None]) for j in range(3)]
        for j in cutlass.range_constexpr(3):
            rt.Convert.st_f32_bf16x8(vals[j], hidden[query, j, tid, None])
        if tid < 16:
            rope = cute.make_tensor(
                p.RopeTable,
                cute.make_layout((1 << 24, 16, 4), stride=(ROPE_COLS, 4, 1)),
            )
            freq = cute.make_tensor(
                p.Freq, cute.make_layout((self.ROWS, 16, 4), stride=(ROPE_COLS, 4, 1))
            )
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            )
            v = cute.make_rmem_tensor(4, Float32)
            cute.copy(atom, rope[position, tid, None], v)
            cute.copy(atom, v, freq[query, tid, None])


class PrologueQ4(PrologueKernel):
    """One contiguous embedding table."""

    Params = PrologueQ4Params

    def embedding_row(self, p: PrologueQ4Params, token):
        table = cute.make_tensor(
            p.Embedding,
            cute.make_layout((rt.Core.VOCAB, 3, 256, 8), stride=(HIDDEN, 2048, 8, 1)),
        )
        return table[token, None, None, None]


class PrologueVllm(PrologueKernel):
    """The vLLM vocab-parallel table: eight shards at eight peer base addresses."""

    Params = PrologueVllmParams

    def embedding_row(self, p: PrologueVllmParams, token):
        owner = token // rt.Core.SHARD_ROWS
        local = token - owner * rt.Core.SHARD_ROWS
        base = rt.Memory.select_peer(owner, p.shards)
        row_ptr = rt.Memory.gmem_ptr(BFloat16, base + Int64(local) * (HIDDEN * 2))
        return cute.make_tensor(
            row_ptr, cute.make_layout((3, 256, 8), stride=(2048, 8, 1))
        )
