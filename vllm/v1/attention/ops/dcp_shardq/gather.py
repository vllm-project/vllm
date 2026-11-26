# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push a DCP rank's KV shard into every rank's gathered context workspace.

Under DCP (interleave 1) rank `r` of `W` stores global token `t` iff
`t % W == r`, at local position `t // W` of its MLA and indexer caches. This
kernel copies the rank's tokens byte for byte to their global positions in
every rank's workspace, which holds each request's tokens in order in dense
64-token pages of the same formats. CTA (x, y) moves eight local tokens of
request y.
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

from vllm.cute_utils.ptx_mem import ld_global_v4, st_global_v4
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)
from vllm.v1.attention.ops.dcp_shardq import PAGE_TOKENS


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
    tokens_per_cta = 8

    @dataclass(frozen=True)
    class CompileKey:
        world: int
        mla_token_bytes: int
        # Indexer cache entry: values, then a 4-byte scale.
        index_token_bytes: int

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        num_threads = DcpContextPushKernel.num_threads
        tokens_per_cta = DcpContextPushKernel.tokens_per_cta
        ctas_per_page = PAGE_TOKENS // tokens_per_cta
        world = compile_key.world
        mla_bytes = compile_key.mla_token_bytes
        index_bytes = compile_key.index_token_bytes - 4
        assert mla_bytes % 16 == 0 and index_bytes % 16 == 0
        mla_page = PAGE_TOKENS * mla_bytes
        index_page = PAGE_TOKENS * compile_key.index_token_bytes
        # An indexer page holds its tokens' values, then their scales.
        scale_offset = PAGE_TOKENS * index_bytes

        @cute.jit
        def copy_tokens(
            src: Int64,
            dst: list,
            dst_page: Int64,
            rank: Int32,
            local0: Int32,
            num_tokens: Int32,
            page_bytes: int,
            section: int,
            token_bytes: int,
        ):
            # Vector v: local token local0 + v // vecs, chunk v % vecs of 16 B
            # (4 B for the scales). Local token i is global token world * i + rank.
            width = 4 if token_bytes == 4 else 16
            vecs = token_bytes // width
            v = Int32(cute.arch.thread_idx()[0])
            while v < num_tokens * vecs:
                i = local0 + v // vecs
                col = Int64((v % vecs) * width)
                token = i * world + rank
                src_addr = src + section + Int64(i) * token_bytes + col
                dst_offset = (
                    (dst_page + Int64(token // PAGE_TOKENS)) * page_bytes
                    + section
                    + Int64(token % PAGE_TOKENS) * token_bytes
                    + col
                )
                if cutlass.const_expr(width == 4):
                    word = _ld_u32(src_addr)
                    for d in cutlass.range_constexpr(world):
                        _st_u32(dst[d] + dst_offset, word)
                else:
                    w0, w1, w2, w3 = ld_global_v4(src_addr, nc=True)
                    for d in cutlass.range_constexpr(world):
                        st_global_v4(dst[d] + dst_offset, w0, w1, w2, w3)
                v += num_threads

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
            bidx, req, _ = cute.arch.block_idx()
            page = bidx // ctas_per_page
            local0 = (bidx % ctas_per_page) * tokens_per_cta
            # The request's local tokens: t < n with t % world == rank. Its MLA
            # entries stop at the prior context, since every rank writes the
            # new tokens' MLA entries itself.
            seq_len = seq_lens[req]
            context_len = seq_len - (query_start_loc[req + 1] - query_start_loc[req])
            local_seq_len = cutlass.max(seq_len - rank + world - 1, 0) // world
            local_context_len = cutlass.max(context_len - rank + world - 1, 0) // world
            first_local = page * PAGE_TOKENS + local0
            num_index = cutlass.max(
                cutlass.min(local_seq_len - first_local, tokens_per_cta), 0
            )
            num_mla = cutlass.max(
                cutlass.min(local_context_len - first_local, tokens_per_cta), 0
            )
            if num_index > 0:
                # Local page j holds global tokens [W * 64 * j, W * 64 * (j + 1)).
                dst_page = Int64(dst_block_table[req, 0]) + Int64(world * page)
                mla_dst = [Int64(dst_addrs[d]) for d in range(world)]
                index_dst = [
                    Int64(dst_addrs[d]) + dst_index_offset for d in range(world)
                ]

                # MLA blocks may span several pages.
                mla_src = (
                    mla_base
                    + Int64(mla_block_table[req, page // mla_pages_per_block])
                    * mla_block_stride
                    + Int64((page % mla_pages_per_block) * mla_page)
                )
                copy_tokens(
                    mla_src,
                    mla_dst,
                    dst_page,
                    rank,
                    local0,
                    num_mla,
                    mla_page,
                    0,
                    mla_bytes,
                )
                index_src = (
                    index_base
                    + Int64(index_block_table[req, page]) * index_block_stride
                )
                # Values, then scales (a static loop: not over a range).
                for section, token_bytes in ((0, index_bytes), (scale_offset, 4)):
                    copy_tokens(
                        index_src,
                        index_dst,
                        dst_page,
                        rank,
                        local0,
                        num_index,
                        index_page,
                        section,
                        token_bytes,
                    )

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

    def get_warmup_keys(self, **key: int) -> list[CompileKey]:  # type: ignore[override]
        return [self.CompileKey(**key)]

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
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        """Push this rank's KV of each request's `seq_lens` tokens (MLA entries
        of its context only) to the workspace pages from `dst_block_table[:, 0]`
        on. The caches are [blocks, block_size, token_bytes] byte views (MLA
        blocks of whole pages, indexer blocks of one page). `dst_addrs` holds
        every rank's workspace: MLA pages, then indexer pages at
        `dst_index_offset`."""
        assert mla_cache.shape[1] % PAGE_TOKENS == 0 and index_cache.shape[1] == (
            PAGE_TOKENS
        )
        for cache in (mla_cache, index_cache):
            assert cache.element_size() == 1 and cache.stride(1) == cache.shape[2]
        compile_key = self.CompileKey(
            world=dst_addrs.shape[0],
            mla_token_bytes=mla_cache.shape[-1],
            index_token_bytes=index_cache.shape[-1],
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
