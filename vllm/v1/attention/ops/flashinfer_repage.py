# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only kernel pages over a packed BF16/FP16 HND cache."""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@dataclass(frozen=True)
class PackedKVPageGeometry:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    page_size: int
    pages_per_block: int
    block_stride_pages: int
    num_pages: int

    @classmethod
    def from_cache(cls, cache: torch.Tensor, page_size: int) -> "PackedKVPageGeometry":
        """Derive kernel pages from an allocator-owned [B,H,M,2D] cache."""
        num_blocks, _, block_size, _ = cache.shape
        block_stride, head_stride, token_stride, _ = cache.stride()
        if block_size % page_size:
            raise ValueError("Kernel page size must divide the manager block size.")
        page_stride = page_size * token_stride
        if block_stride % page_stride:
            raise ValueError("Packed KV block stride must contain whole kernel pages.")
        pages_per_block = block_size // page_size
        block_stride_pages = block_stride // page_stride
        num_pages = (num_blocks - 1) * block_stride_pages + pages_per_block
        if max(num_pages, head_stride, page_stride) > 2**31 - 1:
            raise ValueError("Packed KV geometry exceeds FlashInfer's int32 limits.")
        return cls(
            shape=tuple(cache.shape),
            strides=cache.stride(),
            page_size=page_size,
            pages_per_block=pages_per_block,
            block_stride_pages=block_stride_pages,
            num_pages=num_pages,
        )

    def read_view(self, cache: torch.Tensor) -> torch.Tensor:
        _, heads, _, channels = self.shape
        _, head_stride, token_stride, channel_stride = self.strides
        # Only referenced pages are valid. Never register this view for writes,
        # block copies, zeroing or transfer: its virtual holes alias other layers.
        return cache.as_strided(
            (self.num_pages, heads, self.page_size, channels),
            (self.page_size * token_stride, head_stride, token_stride, channel_stride),
        )


@triton.jit
def remap_page_ids(
    ids, PAGES_PER_BLOCK: tl.constexpr, BLOCK_STRIDE_PAGES: tl.constexpr
):
    physical = (
        ids.to(tl.int64) // PAGES_PER_BLOCK * BLOCK_STRIDE_PAGES
        + ids.to(tl.int64) % PAGES_PER_BLOCK
    )
    return tl.where(ids >= 0, physical, ids).to(tl.int32)


@triton.jit(do_not_specialize=["num_cols"])
def _repage_block_table_kernel(
    block_table,
    seq_lens,
    out,
    num_cols,
    block_table_stride,
    out_stride,
    PAGE_SIZE: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    BLOCK_STRIDE_PAGES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tiles = tl.cdiv(num_cols, BLOCK_SIZE)
    row = tl.program_id(0) // tiles
    cols = (tl.program_id(0) % tiles) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    length = tl.load(seq_lens + row)
    active = cols < tl.cdiv(length, PAGE_SIZE)
    ids = tl.load(
        block_table + row.to(tl.int64) * block_table_stride + cols,
        (cols < num_cols) & active,
        other=0,
    )
    ids = remap_page_ids(ids, PAGES_PER_BLOCK, BLOCK_STRIDE_PAGES)
    tl.store(
        out + row.to(tl.int64) * out_stride + cols,
        tl.where(active, ids, 0),
        cols < num_cols,
    )


def repage_block_table(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    geometry: PackedKVPageGeometry,
) -> torch.Tensor:
    """Remap into the builder's separate, preallocated int32 scratch buffer."""
    rows, cols = block_table.shape
    if rows == 0 or cols == 0:
        return out[:rows, :cols]
    _repage_block_table_kernel[(rows * triton.cdiv(cols, 256),)](
        block_table,
        seq_lens,
        out,
        cols,
        block_table.stride(0),
        out.stride(0),
        geometry.page_size,
        geometry.pages_per_block,
        geometry.block_stride_pages,
        BLOCK_SIZE=256,
    )
    return out[:rows, :cols]
