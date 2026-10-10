# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.utils.cache import (
    VISION_ROPE_SHAPE_CACHE_BYTES,
    LRUCache,
    tensors_nbytes,
)
from vllm.utils.mem_constants import MiB_bytes


def test_tensors_nbytes_sums_nested_tensors():
    a = torch.empty(100, dtype=torch.uint8)
    b = torch.empty(50, dtype=torch.uint8)
    assert tensors_nbytes(a) == 100
    assert tensors_nbytes((a, b, "skip")) == 150
    assert tensors_nbytes("not-a-tensor") == 0


def test_vision_rope_shape_cache_evicts_when_byte_budget_exceeded():
    """Many unique shapes must not pin more than the byte budget."""
    cache: LRUCache[int, tuple[torch.Tensor, ...]] = LRUCache(
        capacity=VISION_ROPE_SHAPE_CACHE_BYTES,
        getsizeof=tensors_nbytes,
    )
    entry_bytes = 8 * MiB_bytes
    n_entries = int(VISION_ROPE_SHAPE_CACHE_BYTES // entry_bytes) + 32
    half = entry_bytes // 2

    for i in range(n_entries):
        cos = torch.empty(half, dtype=torch.uint8)
        sin = torch.empty(half, dtype=torch.uint8)
        empty = torch.empty(0, dtype=torch.int32)
        assert cache.put_if_fits(i, (cos, sin, empty, empty, empty))

    assert cache.currsize <= VISION_ROPE_SHAPE_CACHE_BYTES
    assert len(cache) <= VISION_ROPE_SHAPE_CACHE_BYTES // entry_bytes
    assert len(cache) < n_entries


def test_vision_rope_shape_cache_skips_entry_larger_than_budget():
    cache: LRUCache[str, torch.Tensor] = LRUCache(
        capacity=VISION_ROPE_SHAPE_CACHE_BYTES,
        getsizeof=tensors_nbytes,
    )
    oversized = torch.empty(
        VISION_ROPE_SHAPE_CACHE_BYTES + MiB_bytes, dtype=torch.uint8
    )
    assert cache.put_if_fits("big", oversized) is False
    assert "big" not in cache
