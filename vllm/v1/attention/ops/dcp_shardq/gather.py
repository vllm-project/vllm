# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push a DCP rank's context rows into every rank's gathered workspace.

Under DCP (interleave 1) rank ``r`` of ``W`` stores global token ``t`` iff
``t % W == r``, at local position ``t // W`` of its paged caches: a DSA layer's
MLA cache (one row per token) and its indexer cache (64-token pages of fp8
keys, then their scales). This kernel copies the rank's rows, byte for byte, to
their global positions in every rank's workspace (peer addresses, e.g.
symmetric memory), which holds each request's tokens in order in dense pages of
the same two formats. The work comes from the step's device metadata: CTA
row ``r`` of the grid serves request ``r``, and each CTA moves eight rows of one
of its local pages (CTAs past the request's rows exit).
"""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor

from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)
from vllm.v1.attention.ops.dcp_shardq import PAGE_TOKENS


@dsl_user_op
def _ld_v4(address: Int64, *, loc=None, ip=None) -> tuple[Int32, ...]:
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::no_allocate.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def _st_v4(
    address: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            d.ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_u32(address: Int64, *, loc=None, ip=None) -> Int32:
    out = llvm.inline_asm(
        T.i32(),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.global.nc.u32 $0, [$1];",
        "=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(out)


@dsl_user_op
def _st_u32(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class DcpContextPushKernel(VllmCuTeDSLJitKernel["DcpContextPushKernel.CompileKey"]):
    num_threads = 128
    rows_per_cta = 8

    @dataclass(frozen=True)
    class CompileKey:
        world: int
        mla_row_bytes: int
        index_row_bytes: int
        index_scale_bytes: int

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        num_threads = DcpContextPushKernel.num_threads
        rows_per_cta = DcpContextPushKernel.rows_per_cta
        ctas_per_page = PAGE_TOKENS // rows_per_cta
        world = compile_key.world
        mla_row = compile_key.mla_row_bytes
        index_row = compile_key.index_row_bytes
        scale_bytes = compile_key.index_scale_bytes
        assert mla_row % 16 == 0 and index_row % 16 == 0 and scale_bytes == 4
        mla_page = PAGE_TOKENS * mla_row
        index_page = PAGE_TOKENS * (index_row + scale_bytes)
        # An indexer page holds its tokens' values, then their scales.
        scale_offset = PAGE_TOKENS * index_row

        @cute.jit
        def copy_rows(
            src: Int64,
            dst: list,
            dst_page: Int64,
            rank: Int32,
            row0: Int32,
            num_rows: Int32,
            page_bytes: int,
            section: int,
            row_bytes: int,
        ):
            # Vector v of the slice's rows: row v // vecs, 16 B column v % vecs.
            # Local row i is global token world * i + rank of the page's span.
            vecs = row_bytes // 16
            v = Int32(cute.arch.thread_idx()[0])
            while v < num_rows * vecs:
                i = row0 + v // vecs
                col = Int64((v % vecs) * 16)
                token = i * world + rank
                w0, w1, w2, w3 = _ld_v4(src + section + Int64(i) * row_bytes + col)
                dst_offset = (
                    (dst_page + Int64(token // PAGE_TOKENS)) * page_bytes
                    + section
                    + Int64(token % PAGE_TOKENS) * row_bytes
                    + col
                )
                for d in cutlass.range_constexpr(world):
                    _st_v4(dst[d] + dst_offset, w0, w1, w2, w3)
                v += num_threads

        @cute.jit
        def page_address(
            base: Int64,
            block_stride: Int64,
            pages_per_block: Int32,
            block_table: cute.Tensor,
            req: Int32,
            page: Int32,
            page_bytes: int,
        ) -> Int64:
            block = Int64(block_table[req, page // pages_per_block])
            in_block = Int64((page % pages_per_block) * page_bytes)
            return base + block * block_stride + in_block

        @cute.kernel
        def device_kernel(
            mla_base: Int64,
            mla_block_stride: Int64,
            mla_pages_per_block: Int32,
            mla_block_table: cute.Tensor,
            index_base: Int64,
            index_block_stride: Int64,
            index_block_table: cute.Tensor,
            seq_lens: cute.Tensor,
            query_start_loc: cute.Tensor,
            dst_block_table: cute.Tensor,
            dst_addrs: cute.Tensor,
            dst_index_offset: Int64,
            rank: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, req, _ = cute.arch.block_idx()
            page = bidx // ctas_per_page
            row0 = (bidx % ctas_per_page) * rows_per_cta
            # The request's local rows: tokens t < n with t % world == rank.
            # Its MLA rows stop at the prior context, since every rank writes
            # the new tokens' MLA rows itself.
            seq_len = seq_lens[req]
            context_len = seq_len - (query_start_loc[req + 1] - query_start_loc[req])
            index_rows = cutlass.max(seq_len - rank + world - 1, 0) // world
            mla_rows = cutlass.max(context_len - rank + world - 1, 0) // world
            first_row = page * PAGE_TOKENS + row0
            num_index = cutlass.max(
                cutlass.min(index_rows - first_row, rows_per_cta), 0
            )
            num_mla = cutlass.max(cutlass.min(mla_rows - first_row, rows_per_cta), 0)
            if num_index > 0:
                # Local page j holds global tokens [W * 64 * j, W * 64 * (j + 1)).
                dst_page = Int64(dst_block_table[req, 0]) + Int64(world * page)
                mla_dst = [Int64(dst_addrs[d]) for d in range(world)]
                index_dst = [
                    Int64(dst_addrs[d]) + dst_index_offset for d in range(world)
                ]

                mla_src = page_address(
                    mla_base,
                    mla_block_stride,
                    mla_pages_per_block,
                    mla_block_table,
                    req,
                    page,
                    mla_page,
                )
                copy_rows(
                    mla_src,
                    mla_dst,
                    dst_page,
                    rank,
                    row0,
                    num_mla,
                    mla_page,
                    0,
                    mla_row,
                )
                index_src = (
                    index_base
                    + Int64(index_block_table[req, page]) * index_block_stride
                )
                copy_rows(
                    index_src,
                    index_dst,
                    dst_page,
                    rank,
                    row0,
                    num_index,
                    index_page,
                    0,
                    index_row,
                )
                if tidx < num_index:
                    i = row0 + tidx
                    token = i * world + rank
                    scale = _ld_u32(index_src + scale_offset + Int64(i) * scale_bytes)
                    dst_offset = (
                        (dst_page + Int64(token // PAGE_TOKENS)) * index_page
                        + scale_offset
                        + Int64(token % PAGE_TOKENS) * scale_bytes
                    )
                    for d in cutlass.range_constexpr(world):
                        _st_u32(index_dst[d] + dst_offset, scale)

        @cute.jit
        def host_entrypoint(
            mla_base: Int64,
            mla_block_stride: Int64,
            mla_pages_per_block: Int32,
            mla_block_table: cute.Tensor,
            index_base: Int64,
            index_block_stride: Int64,
            index_block_table: cute.Tensor,
            seq_lens: cute.Tensor,
            query_start_loc: cute.Tensor,
            dst_block_table: cute.Tensor,
            dst_addrs: cute.Tensor,
            dst_index_offset: Int64,
            rank: Int32,
            max_local_pages: Int32,
            stream: CUstream,
        ):
            device_kernel(
                mla_base,
                mla_block_stride,
                mla_pages_per_block,
                mla_block_table,
                index_base,
                index_block_stride,
                index_block_table,
                seq_lens,
                query_start_loc,
                dst_block_table,
                dst_addrs,
                dst_index_offset,
                rank,
            ).launch(
                grid=(max_local_pages * ctas_per_page, seq_lens.shape[0], 1),
                block=(num_threads, 1, 1),
                stream=stream,
            )

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self,
        *,
        world: int,
        mla_row_bytes: int,
        index_row_bytes: int,
        index_scale_bytes: int,
    ) -> CompileKey:
        return self.CompileKey(
            world=world,
            mla_row_bytes=mla_row_bytes,
            index_row_bytes=index_row_bytes,
            index_scale_bytes=index_scale_bytes,
        )

    # The JIT warmup registry passes a registration's keywords through as is
    # (vllm_config only to a registration without arguments).
    def get_warmup_keys(  # type: ignore[override]
        self,
        *,
        world: int,
        mla_row_bytes: int,
        index_row_bytes: int,
        index_scale_bytes: int,
    ) -> list[CompileKey]:
        return [
            self.dispatch(
                world=world,
                mla_row_bytes=mla_row_bytes,
                index_row_bytes=index_row_bytes,
                index_scale_bytes=index_scale_bytes,
            )
        ]

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        def table():
            return cute.runtime.make_fake_tensor(
                Int32,
                (cute.sym_int(), cute.sym_int()),
                stride=(cute.sym_int(), 1),
                assumed_align=4,
            )

        def vector():
            return make_fake_tensor(Int32, (cute.sym_int(),))

        dst_addrs = make_fake_tensor(Int64, (compile_key.world,))
        return (
            Int64(0),
            Int64(0),
            Int32(1),
            table(),
            Int64(0),
            Int64(0),
            table(),
            vector(),
            vector(),
            table(),
            dst_addrs,
            Int64(0),
            Int32(0),
            Int32(1),
        )

    @kernel_launcher
    def __call__(
        self,
        *,
        mla_cache: torch.Tensor,
        mla_block_table: torch.Tensor,
        index_cache: torch.Tensor,
        index_block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        dst_block_table: torch.Tensor,
        max_local_pages: int,
        dst_addrs: torch.Tensor,
        dst_index_offset: int,
        rank: int,
        index_row_bytes: int,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        """Push this rank's rows of the first ``seq_lens`` tokens of each
        request (MLA rows of its context only; ``query_start_loc`` bounds its
        new tokens) to the workspace pages from ``dst_block_table[:, 0]`` on.

        ``mla_cache`` is a [blocks, block_size, row] byte view of the MLA cache
        (blocks of whole pages, dense rows) and ``index_cache`` one of the
        indexer cache in pages; ``max_local_pages`` bounds any request's local
        pages. ``dst_addrs`` holds every rank's workspace, MLA pages first,
        indexer pages ``dst_index_offset`` bytes in."""
        assert mla_cache.shape[1] % PAGE_TOKENS == 0 and index_cache.shape[1] == (
            PAGE_TOKENS
        )
        for cache in (mla_cache, index_cache):
            assert cache.element_size() == 1 and cache.stride(1) == cache.shape[2]
        index_scale_bytes = index_cache.shape[-1] - index_row_bytes
        compile_key = self.dispatch(
            world=dst_addrs.shape[0],
            mla_row_bytes=mla_cache.shape[-1],
            index_row_bytes=index_row_bytes,
            index_scale_bytes=index_scale_bytes,
        )
        return compile_key, (
            mla_cache.data_ptr(),
            mla_cache.stride(0),
            mla_cache.shape[1] // PAGE_TOKENS,
            mla_block_table,
            index_cache.data_ptr(),
            index_cache.stride(0),
            index_block_table,
            seq_lens,
            query_start_loc,
            dst_block_table,
            dst_addrs,
            dst_index_offset,
            rank,
            max_local_pages,
        )


DCP_CONTEXT_PUSH_KERNEL = DcpContextPushKernel()
