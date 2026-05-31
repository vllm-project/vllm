# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr


@dataclass(frozen=True)
class PagedKVManager:
    mPageTable: cute.Tensor
    page_size: cutlass.Constexpr[int]
    n_block_size: cutlass.Constexpr[int]
    segment_rows: cutlass.Constexpr[int]
    segments_per_block: cutlass.Constexpr[int]
    blocks_per_page: cutlass.Constexpr[int]

    @staticmethod
    def create(
        mPageTable: cute.Tensor,
        *,
        page_size: int,
        n_block_size: int,
    ):
        if page_size < 8:
            raise ValueError(
                f"page_size must be >= 8 for TMA segmented load, got {page_size}"
            )
        if page_size % n_block_size == 0:
            segment_rows = n_block_size
            segments_per_block = 1
            blocks_per_page = page_size // n_block_size
        elif n_block_size % page_size == 0:
            segment_rows = page_size
            segments_per_block = n_block_size // page_size
            blocks_per_page = 1
        else:
            raise ValueError(
                f"page_size ({page_size}) must divide blk_kv ({n_block_size}) "
                f"or be divisible by it"
            )
        return PagedKVManager(
            mPageTable,
            page_size=page_size,
            n_block_size=n_block_size,
            segment_rows=segment_rows,
            segments_per_block=segments_per_block,
            blocks_per_page=blocks_per_page,
        )

    @cute.jit
    def logical_length(
        self,
        batch_idx: Int32,
        num_kv_blocks: Int32,
        mSeqUsedK=None,
    ) -> Int32:
        if const_expr(mSeqUsedK is not None):
            return mSeqUsedK[batch_idx]
        return num_kv_blocks * Int32(self.n_block_size)

    @cute.jit
    def valid_cols_in_block(
        self,
        batch_idx: Int32,
        kv_block_idx: Int32,
        num_kv_blocks: Int32,
        mSeqUsedK=None,
    ) -> Int32:
        seqlen_k = self.logical_length(batch_idx, num_kv_blocks, mSeqUsedK)
        block_start = kv_block_idx * Int32(self.n_block_size)
        remaining = seqlen_k - block_start
        remaining = cutlass.max(remaining, Int32(0))
        return cutlass.min(remaining, Int32(self.n_block_size))

    @cute.jit
    def physical_page_and_tile(
        self,
        batch_idx: Int32,
        kv_block_idx: Int32,
        seg_idx: Int32,
    ) -> tuple[Int32, Int32]:
        if const_expr(self.page_size >= self.n_block_size):
            logical_page = kv_block_idx // Int32(self.blocks_per_page)
            page_tile_idx = kv_block_idx % Int32(self.blocks_per_page)
            physical_page = self.mPageTable[batch_idx, logical_page]
            return physical_page, page_tile_idx
        logical_page = kv_block_idx * Int32(self.segments_per_block) + seg_idx
        physical_page = self.mPageTable[batch_idx, logical_page]
        return physical_page, Int32(0)

    @cute.jit
    def physical_block_index(
        self,
        batch_idx: Int32,
        kv_block_idx: Int32,
    ) -> Int32:
        physical_page, page_tile_idx = self.physical_page_and_tile(
            batch_idx, kv_block_idx, Int32(0)
        )
        return physical_page * Int32(self.blocks_per_page) + page_tile_idx

    @cute.jit
    def physical_page_index(
        self,
        batch_idx: Int32,
        kv_block_idx: Int32,
        seg_idx: Int32,
    ) -> Int32:
        physical_page, _ = self.physical_page_and_tile(batch_idx, kv_block_idx, seg_idx)
        return physical_page

    @cute.jit
    def physical_row_start(
        self,
        batch_idx: Int32,
        kv_block_idx: Int32,
        seg_idx: Int32,
    ) -> Int32:
        physical_page, page_tile_idx = self.physical_page_and_tile(
            batch_idx, kv_block_idx, seg_idx
        )
        return physical_page * Int32(self.page_size) + page_tile_idx * Int32(
            self.n_block_size
        )


__all__ = ["PagedKVManager"]
