# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 indexer scores over DSV4.1's candidate blocks."""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float8E4M3FN, Float32, Int32, Int64
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, warpgroup
from cutlass.cute.runtime import make_fake_tensor
from cutlass.utils import hopper_helpers

from vllm.cute_utils import mbarrier, recast_val, simple_tma_copy
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)


class CandidatePagePlanKernel(VllmCuTeDSLJitKernel[tuple[int, int]]):
    """Cache ``(physical_page << 32) | token_offset`` for all consumer layers."""

    @staticmethod
    def kernel(key: tuple[int, int]) -> Any:
        page_size, sparse_block_size = key

        @cute.kernel
        def device_kernel(
            block_table: cute.Tensor,
            sparse_indices: cute.Tensor,
            valid_end: cute.Tensor,
            row_indices: cute.Tensor,
            physical_blocks: cute.Tensor,
        ):
            tile, row, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            block = tile * 256 + tid
            if block < sparse_indices.shape[1]:
                address = Int64(0)
                if block * sparse_block_size < valid_end[row]:
                    pos = sparse_indices[row, block] * sparse_block_size
                    page = block_table[row_indices[row], pos // page_size]
                    address = (Int64(page) << 32) | Int64(pos % page_size)
                physical_blocks[row, block] = address

        @cute.jit
        def entry(
            block_table: cute.Tensor,
            sparse_indices: cute.Tensor,
            valid_end: cute.Tensor,
            row_indices: cute.Tensor,
            physical_blocks: cute.Tensor,
            stream: CUstream,
        ):
            device_kernel(
                block_table, sparse_indices, valid_end, row_indices, physical_blocks
            ).launch(
                grid=(
                    cute.ceil_div(sparse_indices.shape[1], 256),
                    sparse_indices.shape[0],
                    1,
                ),
                block=(256, 1, 1),
                stream=stream,
            )

        return entry

    def get_warmup_keys(
        self, *, page_size: int, sparse_block_size: int
    ) -> list[tuple[int, int]]:
        return [(page_size, sparse_block_size)]

    def warmup_inputs(self, key: tuple[int, int]) -> tuple[Any, ...]:
        def matrix(dtype):
            return make_fake_tensor(
                dtype,
                (cute.sym_int(), cute.sym_int()),
                (cute.sym_int(), 1),
                assumed_align=dtype.width // 8,
            )

        def vector():
            return make_fake_tensor(Int32, (cute.sym_int(),), (1,), assumed_align=4)

        return matrix(Int32), matrix(Int32), vector(), vector(), matrix(Int64)

    @kernel_launcher
    def __call__(
        self,
        block_table: torch.Tensor,
        sparse_indices: torch.Tensor,
        valid_end: torch.Tensor,
        row_indices: torch.Tensor,
        physical_blocks: torch.Tensor,
        *,
        page_size: int,
        sparse_block_size: int,
    ) -> CuTeDSLLaunchSpec[tuple[int, int]]:
        return (page_size, sparse_block_size), (
            block_table,
            sparse_indices,
            valid_end,
            row_indices,
            physical_blocks,
        )


class CandidatePagedLogitsKernel(
    VllmCuTeDSLJitKernel["CandidatePagedLogitsKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        page_size: int
        sparse_block_size: int
        aligned_cache: bool = True
        use_tma: bool = True
        tile_tokens: int = 64

    @staticmethod
    def kernel(key: CompileKey) -> Any:
        page_size = key.page_size
        sparse_block_size = key.sparse_block_size
        tile_tokens = key.tile_tokens

        @cute.kernel
        def device_kernel(
            q: cute.Tensor,
            cache: cute.Tensor,
            weights: cute.Tensor,
            physical_blocks: cute.Tensor,
            valid_end: cute.Tensor,
            logits: cute.Tensor,
            page_stride: Int64,
            tiled_mma: cute.TiledMma,
            tma: cpasync.TmaInfo | None,
        ):
            tid, _, _ = cute.arch.thread_idx()
            row, tile, _ = cute.arch.block_idx()
            start = tile * tile_tokens
            end = valid_end[row]
            if start < end:
                smem = utils.SmemAllocator()
                k_layout = cute.slice_(
                    hopper_helpers.make_smem_layout_a(
                        utils.LayoutEnum.ROW_MAJOR,
                        (tile_tokens, 32, 128),
                        Float8E4M3FN,
                        1,
                    ),
                    (None, None, 0),
                )
                q_layout = cute.slice_(
                    hopper_helpers.make_smem_layout_b(
                        utils.LayoutEnum.ROW_MAJOR,
                        (tile_tokens, 32, 128),
                        Float8E4M3FN,
                        1,
                    ),
                    (None, None, 0),
                )
                sk = smem.allocate_tensor(
                    Float8E4M3FN,
                    k_layout.outer,
                    byte_alignment=1024,
                    swizzle=k_layout.inner,
                )
                sq = smem.allocate_tensor(
                    Float8E4M3FN,
                    q_layout.outer,
                    byte_alignment=1024,
                    swizzle=q_layout.inner,
                )
                scales = smem.allocate_tensor(
                    Float32, cute.make_layout(tile_tokens), byte_alignment=16
                )
                if cutlass.const_expr(key.use_tma):
                    assert tma is not None
                    loaded = smem.allocate_array(Int64, 1)
                    if tid == 0:
                        cute.arch.mbarrier_init(loaded, 1)
                        cute.arch.mbarrier_init_fence()
                    cute.arch.sync_threads()
                    if tid // 32 == 0:
                        with cute.arch.elect_one():
                            mbarrier.arrive_expect_tx(loaded, tile_tokens * 128)
                        src_tiles = cute.zipped_divide(
                            tma.tma_tensor, (sparse_block_size, 128)
                        )
                        dst_tiles = cute.zipped_divide(sk, (sparse_block_size, 128))
                        block_address = Int64(0)
                        if (
                            tid < tile_tokens // sparse_block_size
                            and start + tid * sparse_block_size < end
                        ):
                            block_address = physical_blocks[
                                row, tile * (tile_tokens // sparse_block_size) + tid
                            ]
                        for block in cutlass.range_constexpr(
                            tile_tokens // sparse_block_size
                        ):
                            address = cute.arch.shuffle_sync(block_address, block)
                            page = Int32(address >> 32)
                            offset = Int32(address & 0xFFFFFFFF)
                            simple_tma_copy(
                                tma.atom,
                                src_tiles[None, (offset // sparse_block_size, 0, page)],
                                dst_tiles[None, (block, 0)],
                                loaded,
                            )
                # recast_tensor drops the pointer swizzle in CuTe DSL 4.8.
                sk_words = cute.make_tensor(
                    cute.recast_ptr(sk.iterator, k_layout.inner, Int32),
                    cute.recast_layout(32, 8, sk.layout),
                )
                sq_words = cute.make_tensor(
                    cute.recast_ptr(sq.iterator, q_layout.inner, Int32),
                    cute.recast_layout(32, 8, sq.layout),
                )
                copy = cute.make_copy_atom(
                    cpasync.CopyG2SOp(), Int32, num_bits_per_copy=128
                )
                q_vectors = cute.logical_divide(q, (None, None, 4))
                sq_vectors = cute.logical_divide(sq_words, (None, 4))
                sk_vectors = cute.logical_divide(sk_words, (None, 4))
                for i in cutlass.range_constexpr(2):
                    head = tid // 8 + i * 16
                    cute.copy(
                        copy,
                        q_vectors[row, head, (None, tid % 8)],
                        sq_vectors[head, (None, tid % 8)],
                    )
                if cutlass.const_expr(key.use_tma):
                    for i in cutlass.range_constexpr(cute.ceil_div(tile_tokens, 128)):
                        token = tid + i * 128
                        if token < tile_tokens:
                            col = start + token
                            scale = Float32(0)
                            if col < end:
                                address = physical_blocks[row, col // sparse_block_size]
                                page = Int32(address >> 32)
                                base = Int64(page) * page_stride
                                offset = (
                                    Int32(address & 0xFFFFFFFF)
                                    + col % sparse_block_size
                                )
                                scale = recast_val(
                                    cache[base + page_size * 32 + offset], Float32
                                )
                            scales[token] = scale
                else:
                    for i in cutlass.range_constexpr(tile_tokens // 16):
                        token = tid // 8 + i * 16
                        col = start + token
                        scale = Float32(0)
                        base = Int64(0)
                        offset = Int32(0)
                        if col < end:
                            address = physical_blocks[row, col // sparse_block_size]
                            page = Int32(address >> 32)
                            # Interleaved KV pools need 64-bit address arithmetic.
                            base = Int64(page) * page_stride
                            offset = (
                                Int32(address & 0xFFFFFFFF) + col % sparse_block_size
                            )
                            scale = recast_val(
                                cache[base + page_size * 32 + offset], Float32
                            )
                        if cutlass.const_expr(key.aligned_cache):
                            src_ptr = cache.iterator + base + offset * 32 + tid % 8 * 4
                            src = cute.make_tensor(
                                cute.make_ptr(
                                    Int32,
                                    src_ptr.toint(),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout(4),
                            )
                            pred = cute.make_rmem_tensor((4,), cutlass.Boolean)
                            pred.fill(col < end)
                            cute.copy(
                                copy, src, sk_vectors[token, (None, tid % 8)], pred=pred
                            )
                        else:
                            for j in cutlass.range_constexpr(4):
                                value = Int32(0)
                                if col < end:
                                    value = cache[base + offset * 32 + tid % 8 * 4 + j]
                                sk_words[token, tid % 8 * 4 + j] = value
                        if tid % 8 == 0:
                            scales[token] = scale
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                if cutlass.const_expr(key.use_tma):
                    cute.arch.mbarrier_wait(loaded, 0)
                cute.arch.sync_threads()
                cute.arch.fence_proxy("async.shared", space="cta")

                thread_mma = tiled_mma.get_slice(tid)
                a = tiled_mma.make_fragment_A(tiled_mma.get_slice(0).partition_A(sk))
                b = tiled_mma.make_fragment_B(tiled_mma.get_slice(0).partition_B(sq))
                coords = thread_mma.partition_C(
                    cute.make_identity_tensor((tile_tokens, 32))
                )
                acc = cute.make_rmem_tensor(coords.shape, Float32)
                acc.fill(0)
                warpgroup.fence()
                for k in cutlass.range_constexpr(cute.size(a, mode=[2])):
                    tiled_mma.set(warpgroup.Field.ACCUMULATE, k != 0)
                    cute.gemm(tiled_mma, acc, a[None, None, k], b[None, None, k], acc)
                warpgroup.commit_group()
                warpgroup.wait_group(0)

                # Match DeepGEMM's FP32 reduction order, including scale placement.
                # Each group of four lanes owns two token rows and all 32 heads.
                for m in cutlass.range_constexpr(tile_tokens // 64):
                    partial = cute.make_rmem_tensor((4,), Float32)
                    for j in cutlass.range_constexpr(4):
                        i = m * 16 + j
                        partial[j] = (
                            cute.arch.fmax(acc[i], Float32(0))
                            * weights[row, coords[i][1]]
                        )
                    for h in cutlass.range_constexpr(1, 4):
                        for j in cutlass.range_constexpr(4):
                            i = m * 16 + h * 4 + j
                            partial[j] += (
                                cute.arch.fmax(acc[i], Float32(0))
                                * weights[row, coords[i][1]]
                            )
                    token0 = coords[m * 16][0]
                    token1 = coords[m * 16 + 2][0]
                    sum0 = (partial[0] + partial[1]) * scales[token0]
                    sum1 = (partial[2] + partial[3]) * scales[token1]
                    for delta in cutlass.range_constexpr(2):
                        sum0 += cute.arch.shuffle_sync_bfly(sum0, offset=1 << delta)
                        sum1 += cute.arch.shuffle_sync_bfly(sum1, offset=1 << delta)
                    if tid % 4 == 0:
                        value0 = sum0
                        value1 = sum1
                        if start + token0 >= end:
                            value0 = Float32(-float("inf"))
                        if start + token1 >= end:
                            value1 = Float32(-float("inf"))
                        logits[row, start + token0] = value0
                        logits[row, start + token1] = value1
            else:
                for i in cutlass.range_constexpr(cute.ceil_div(tile_tokens, 128)):
                    if tid + i * 128 < tile_tokens:
                        logits[row, start + tid + i * 128] = Float32(-float("inf"))

        @cute.jit
        def entry(
            q: cute.Tensor,
            cache: cute.Tensor,
            weights: cute.Tensor,
            physical_blocks: cute.Tensor,
            valid_end: cute.Tensor,
            logits: cute.Tensor,
            page_stride: Int64,
            page_count: Int32,
            stream: CUstream,
        ):
            tiled_mma = cute.make_tiled_mma(
                warpgroup.MmaF8Op(
                    Float8E4M3FN,
                    Float8E4M3FN,
                    Float32,
                    (64, 32, 32),
                    warpgroup.OperandSource.SMEM,
                    OperandMajorMode.K,
                    OperandMajorMode.K,
                )
            )
            tma = None
            if cutlass.const_expr(key.use_tma):
                # Each page stores all FP8 key bytes before its FP32 scales.
                paged = cute.make_tensor(
                    cute.recast_ptr(cache.iterator, dtype=Float8E4M3FN),
                    cute.make_layout(
                        (page_size, 128, page_count),
                        stride=(128, 1, Int32(page_stride * 4)),
                    ),
                )
                tma_layout = cute.make_composed_layout(
                    cute.make_swizzle(3, 4, 3),
                    0,
                    cute.make_layout((sparse_block_size, 128), stride=(128, 1)),
                )
                tma = cpasync.make_tiled_tma_atom(
                    cpasync.CopyBulkTensorTileG2SOp(),
                    paged,
                    tma_layout,
                    (sparse_block_size, 128),
                )
            device_kernel(
                q,
                cache,
                weights,
                physical_blocks,
                valid_end,
                logits,
                page_stride,
                tiled_mma,
                tma,
            ).launch(
                grid=(q.shape[0], cute.ceil_div(logits.shape[1], tile_tokens), 1),
                block=(128, 1, 1),
                stream=stream,
            )

        return entry

    def get_warmup_keys(
        self, *, page_size: int, sparse_block_size: int
    ) -> list[CompileKey]:
        return [
            self.CompileKey(
                page_size,
                sparse_block_size,
                aligned_cache=aligned,
                use_tma=use_tma,
                tile_tokens=tile,
            )
            for aligned, use_tma, tile in (
                (False, False, 64),
                (True, False, 64),
                (True, True, 64),
                (True, True, 128),
            )
        ]

    def warmup_inputs(self, key: CompileKey) -> tuple[Any, ...]:
        rows, width = cute.sym_int(), cute.sym_int(divisibility=64)
        return (
            make_fake_tensor(
                Int32,
                (rows, 32, 32),
                (cute.sym_int(divisibility=4), 32, 1),
                assumed_align=16,
            ),
            make_fake_tensor(
                Int32,
                (cute.sym_int64(),),
                (1,),
                assumed_align=16 if key.aligned_cache else 4,
            ),
            make_fake_tensor(Float32, (rows, 32), (cute.sym_int(), 1), assumed_align=4),
            make_fake_tensor(
                Int64, (rows, cute.sym_int()), (cute.sym_int(), 1), assumed_align=8
            ),
            make_fake_tensor(Int32, (rows,), (cute.sym_int(),), assumed_align=4),
            make_fake_tensor(
                Float32, (rows, width), (cute.sym_int(), 1), assumed_align=4
            ),
            Int64(0),
            Int32(0),
        )

    @kernel_launcher
    def __call__(
        self,
        q: torch.Tensor,
        cache: torch.Tensor,
        weights: torch.Tensor,
        physical_blocks: torch.Tensor,
        valid_end: torch.Tensor,
        logits: torch.Tensor,
        *,
        sparse_block_size: int,
        tile_tokens: int | None = None,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        page_size = cache.shape[1]
        assert q.dtype == torch.float8_e4m3fn and q.shape[1:] == (32, 128)
        assert q.stride(1) == 128 and q.stride(2) == 1
        assert q.stride(0) % 16 == 0 and q.storage_offset() % 16 == 0
        assert cache.dtype == torch.uint8 and cache.shape[2] == 132
        assert cache.stride(1) == 132 and cache.stride(2) == 1
        assert cache.stride(0) % 4 == 0 and cache.storage_offset() % 4 == 0
        assert page_size in (16, 32, 64) and sparse_block_size in (8, 16)
        assert logits.dtype == weights.dtype == torch.float32
        assert weights.shape == q.shape[:2] and weights.stride(1) == 1
        assert physical_blocks.dtype == torch.int64 and physical_blocks.stride(1) == 1
        assert valid_end.dtype == torch.int32 and valid_end.shape == (q.shape[0],)
        assert logits.shape == (
            q.shape[0],
            physical_blocks.shape[1] * sparse_block_size,
        )
        assert logits.stride(1) == 1
        aligned = cache.stride(0) % 16 == 0 and cache.storage_offset() % 16 == 0
        # The TMA descriptor uses a 32-bit page stride; retain 64-bit cp.async
        # addressing for unusually wide interleaved-cache layouts.
        use_tma = aligned and cache.stride(0) < 2**31
        if tile_tokens is None:
            # These dimensions stay fixed during CUDA Graph replay.
            tile_tokens = (
                128
                if use_tma and q.shape[0] >= 24 and logits.shape[1] % 128 == 0
                else 64
            )
        assert tile_tokens in (64, 128, 256) and logits.shape[1] % tile_tokens == 0
        cache_bytes = (cache.shape[0] - 1) * cache.stride(0) + page_size * 132
        cache_words = cache.as_strided((cache_bytes,), (1,)).view(torch.int32)
        return (
            self.CompileKey(
                page_size,
                sparse_block_size,
                aligned_cache=aligned,
                use_tma=use_tma,
                tile_tokens=tile_tokens,
            ),
            (
                q.view(torch.int32),
                cache_words,
                weights,
                physical_blocks,
                valid_end,
                logits,
                cache.stride(0) // 4,
                cache.shape[0],
            ),
        )


prepare_candidate_pages = CandidatePagePlanKernel()
candidate_paged_logits = CandidatePagedLogitsKernel()
