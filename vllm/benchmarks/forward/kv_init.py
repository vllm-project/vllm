# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV blocks of a benchmark case: initialization and the blocks a step writes.

Only the case's blocks are touched, group by group: groups can alias one
allocation with different layouts, and the batch builder keeps their block IDs
disjoint. Zero KV times within a few percent of random KV, even for sparse
attention (DeepSeek-V4.1, ~100K context), so zeros are the only init for now.
"""

from collections.abc import Iterator

import torch

from vllm.benchmarks.forward.batch import BenchBatch
from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_block_size


def _tensors(kv_cache) -> Iterator[torch.Tensor]:
    if isinstance(kv_cache, torch.Tensor):
        if kv_cache.numel():
            yield kv_cache
    elif isinstance(kv_cache, (list, tuple)):
        for item in kv_cache:
            yield from _tensors(item)


def _block_dim(tensor: torch.Tensor, num_blocks: int) -> tuple[int, int]:
    """Return (block dim, kernel blocks per allocator block)."""
    for dim in range(min(2, tensor.dim())):
        if tensor.shape[dim] >= num_blocks and tensor.shape[dim] % num_blocks == 0:
            return dim, tensor.shape[dim] // num_blocks
    raise ValueError(f"No block dim of size {num_blocks} in {tuple(tensor.shape)}")


@torch.inference_mode()
def zero_kv_blocks(
    static_forward_context: dict, kv_cache_groups, batch: BenchBatch, num_blocks: int
) -> None:
    for group, ranges in zip(kv_cache_groups, batch.group_block_ranges):
        for name in group.layer_names:
            layer = static_forward_context[name]
            for tensor in _tensors(getattr(layer, "kv_cache", None)):
                dim, factor = _block_dim(tensor, num_blocks)
                for begin, end in ranges:
                    tensor.narrow(dim, begin * factor, (end - begin) * factor).zero_()


def touched_kv_views(
    static_forward_context: dict,
    kv_cache_groups,
    batch: BenchBatch,
    num_blocks: int,
    dcp_size: int,
) -> list[torch.Tensor]:
    """Views of the KV blocks a step may write: each request's last
    ceil(scheduled / block_size) + 2 blocks, in every group."""
    scheduled_tokens = [batch.expected[req_id][0] for req_id in batch.req_ids]
    views = []
    for group, ranges in zip(kv_cache_groups, batch.group_block_ranges):
        block_size = resolve_dcp_kv_block_size(group, dcp_size)
        for name in group.layer_names:
            layer = static_forward_context[name]
            for tensor in _tensors(getattr(layer, "kv_cache", None)):
                dim, factor = _block_dim(tensor, num_blocks)
                for (begin, end), tokens in zip(ranges, scheduled_tokens):
                    first = max(begin, end - (-(-tokens // block_size) + 2))
                    views.append(
                        tensor.narrow(dim, first * factor, (end - first) * factor)
                    )
    return views
