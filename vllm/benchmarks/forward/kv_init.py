# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV blocks of a benchmark case: initialization and the blocks a step writes.

Only the case's blocks are touched, group by group: groups can alias one
allocation with different layouts, and the batch builder keeps their block IDs
disjoint. Zero KV times within a few percent of random KV, even for sparse
attention (DeepSeek-V4.1, ~100K context), so zeros are the only init for now.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from vllm.benchmarks.forward.batch import BenchBatch, kv_block_size
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.v1.worker.utils import AttentionGroup


def _tensors(kv_cache) -> Iterator[torch.Tensor]:
    if isinstance(kv_cache, torch.Tensor):
        if kv_cache.numel():
            yield kv_cache
    elif isinstance(kv_cache, (list, tuple)):
        for item in kv_cache:
            yield from _tensors(item)


def _kernel_blocks_per_block(group: "AttentionGroup") -> int | None:
    """The group's kernel blocks per allocator block, when the layout fixes it.

    Layer views hold one row per allocator block unless a backend splits them
    into kernel blocks, and packed splits are mapped back to manager blocks by
    `_group_tensors`. A spec that keeps the raw buffer and lays it out itself
    returns None and keeps the shape heuristic.
    """
    spec = group.kv_cache_spec
    if not spec.has_layer_views:
        return None
    if group.kernel_block_stride:
        return 1
    if group.kernel_block_size is None:
        return 1
    return spec.block_size // group.kernel_block_size


def _block_dim(
    tensor: torch.Tensor, num_blocks: int, kernel_blocks_per_block: int | None = None
) -> tuple[int, int]:
    """Return (block dim, kernel blocks per allocator block).

    A non-None `kernel_blocks_per_block` marks a layout the group fixes, with
    the block dim on dim 0; None keeps the shape heuristic for storage a spec
    lays out itself.
    """
    if kernel_blocks_per_block is not None:
        # The extensible KV cache commits a prefix of its planned capacity:
        # layer views span the whole reservation while kv_cache_config
        # .num_blocks holds the committed count, so the row count need not
        # equal or divide num_blocks. Allocator blocks stay a linear prefix
        # of the rows, and the uncommitted tail never holds a block the
        # scheduler hands out. The rows must still be whole allocator blocks
        # covering the committed ones, so a planned capacity that happens to
        # be an integer multiple of the committed count is not mistaken for
        # a kernel split.
        rows = tensor.shape[0] if tensor.dim() else 0
        if (
            rows % kernel_blocks_per_block == 0
            and rows // kernel_blocks_per_block >= num_blocks
        ):
            return 0, kernel_blocks_per_block
    else:
        for dim in range(min(2, tensor.dim())):
            if tensor.shape[dim] >= num_blocks and tensor.shape[dim] % num_blocks == 0:
                return dim, tensor.shape[dim] // num_blocks
    raise ValueError(f"No block dim of size {num_blocks} in {tuple(tensor.shape)}")


def _group_tensors(
    kv_caches: dict[str, Any], group: "AttentionGroup", num_blocks: int
) -> Iterator[torch.Tensor]:
    for name in group.layer_names:
        cache = kv_caches.get(name)
        if group.kernel_block_stride:
            assert isinstance(cache, torch.Tensor)
            cache = group.map_kv_cache_to_manager_blocks(cache, num_blocks)
        yield from _tensors(cache)


@torch.inference_mode()
def zero_kv_blocks(
    kv_caches: dict[str, Any],
    attn_groups: list["AttentionGroup"],
    batch: BenchBatch,
    num_blocks: int,
) -> None:
    for group in attn_groups:
        ranges = batch.group_block_ranges[group.kv_cache_group_id]
        kbp = _kernel_blocks_per_block(group)
        for tensor in _group_tensors(kv_caches, group, num_blocks):
            dim, factor = _block_dim(tensor, num_blocks, kbp)
            for begin, end in ranges:
                tensor.narrow(dim, begin * factor, (end - begin) * factor).zero_()


def touched_kv_views(
    kv_caches: dict[str, Any],
    attn_groups: list["AttentionGroup"],
    batch: BenchBatch,
    num_blocks: int,
    dcp_size: int,
    lookahead: int = 0,
) -> list[torch.Tensor]:
    """Views a step may write: attention tails and all recurrent state slots."""
    scheduled_tokens = [
        batch.expected[req_id][0] + lookahead for req_id in batch.req_ids
    ]
    views = []
    for group in attn_groups:
        ranges = batch.group_block_ranges[group.kv_cache_group_id]
        block_size = kv_block_size(group.kv_cache_spec, dcp_size)
        kbp = _kernel_blocks_per_block(group)
        for tensor in _group_tensors(kv_caches, group, num_blocks):
            dim, factor = _block_dim(tensor, num_blocks, kbp)
            for (begin, end), tokens in zip(ranges, scheduled_tokens):
                first = (
                    begin
                    if isinstance(group.kv_cache_spec, MambaSpec)
                    else max(begin, end - (-(-tokens // block_size) + 2))
                )
                views.append(tensor.narrow(dim, first * factor, (end - first) * factor))
    return views
