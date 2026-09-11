# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.models.deepseek_v4_1.common.pipeline import SharingDependency
from vllm.models.deepseek_v4_1.common.pipeline_transfer import (
    get_sharing_routes,
    restore_cache_blocks,
    snapshot_cache_blocks,
)


def test_sharing_routes_relay_across_idle_stages_without_duplicate_sends():
    routes = get_sharing_routes(
        (
            SharingDependency("kv", 2, 8, 0, 2),
            SharingDependency("kv", 2, 9, 0, 2),
            SharingDependency("kv", 2, 10, 0, 3),
            SharingDependency("index", 2, 3, 0, 0),
        )
    )
    assert [(r.kind, r.source_layer, r.sender, r.receiver) for r in routes] == [
        ("kv", 2, 0, 1),
        ("kv", 2, 1, 2),
        ("kv", 2, 2, 3),
    ]
    assert len({r.payload_keys for r in routes}) == 1


@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16])
def test_snapshot_refreshes_prefix_blocks_and_keeps_packed_layout(dtype):
    # Interleaved cache allocations have gaps between physical blocks.
    source = torch.arange(6 * 2 * 3 * 4).reshape(6, 2, 3, 4).to(dtype)[:, 0]
    replica = torch.full((6, 2, 3, 4), 17, dtype=dtype)[:, 0]
    ids, blocks = snapshot_cache_blocks(
        source, [torch.tensor([[4, 1, -1], [1, 4, -1]])], max_bytes=1024
    )
    assert ids.tolist() == [1, 4]
    old = source.clone()
    source.zero_()
    restore_cache_blocks(replica, ids, blocks)
    torch.testing.assert_close(replica[ids], old[ids], rtol=0, atol=0)
    assert torch.all(replica[[0, 2, 3, 5]] == 17)


def test_empty_snapshot_preserves_replica():
    cache = torch.ones(2, 3, 4)
    ids, blocks = snapshot_cache_blocks(cache, [torch.tensor([[-1]])], max_bytes=0)
    restore_cache_blocks(cache, ids, blocks)
    assert torch.all(cache == 1)


def test_snapshot_enforces_memory_budget():
    with pytest.raises(ValueError, match="byte budget"):
        snapshot_cache_blocks(torch.ones(2, 3, 4), [torch.tensor([[0, 1]])], 1)


def test_snapshot_rejects_invalid_physical_blocks():
    with pytest.raises(ValueError, match="unallocated cache block"):
        snapshot_cache_blocks(torch.ones(2, 3, 4), [torch.tensor([[2]])], 1024)


@pytest.mark.parametrize("blocks", [torch.zeros(1, 4, 3), torch.zeros(1, 3, 4).int()])
def test_restore_rejects_different_cache_layouts(blocks):
    with pytest.raises(ValueError, match="different block layout"):
        restore_cache_blocks(torch.ones(2, 3, 4), torch.tensor([0]), blocks)
