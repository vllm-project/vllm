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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16])
@pytest.mark.parametrize("empty", [False, True])
def test_host_block_plan_restores_cuda_prefix_without_device_readback(dtype, empty):
    """Keep packed prefix bytes exact while host IDs avoid CUDA scalar reads."""
    source = (
        torch.arange(6 * 2 * 3 * 4, device="cuda").reshape(6, 2, 3, 4).to(dtype)[:, 0]
    )
    replica = torch.full((6, 2, 3, 4), 17, device="cuda", dtype=dtype)[:, 0]
    table = (
        torch.tensor([[-1, -1]]) if empty else torch.tensor([[4, 1, -1], [1, 4, -1]])
    )
    expected = source.clone()
    # Initialize lazy CUDA facilities before enabling the synchronization guard.
    warm_ids, warm_blocks = snapshot_cache_blocks(source, [table], max_bytes=1024)
    restore_cache_blocks(replica, warm_ids, warm_blocks)
    replica.fill_(17)
    torch.accelerator.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        ids, blocks = snapshot_cache_blocks(source, [table], max_bytes=1024)
        restore_cache_blocks(replica, ids, blocks)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert ids.device.type == "cpu"
    selected = [] if empty else [1, 4]
    assert ids.tolist() == selected
    torch.testing.assert_close(replica[selected], expected[selected], rtol=0, atol=0)
    untouched = [i for i in range(6) if i not in selected]
    assert torch.all(replica[untouched] == 17)


def test_unpadding_preserves_the_matching_host_block_table():
    from vllm.v1.attention.backend import CommonAttentionMetadata

    table = torch.tensor([[3, 7], [4, 9]], dtype=torch.int32)
    metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2]),
        query_start_loc_cpu=torch.tensor([0, 1, 2]),
        seq_lens=torch.tensor([1, 1]),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=1,
        block_table_tensor=table,
        slot_mapping=torch.tensor([0, 1]),
        block_table_cpu=table.clone(),
    )
    actual = metadata.unpadded(1, 1)
    torch.testing.assert_close(actual.block_table_cpu, table[:1])
    assert actual.block_table_cpu.shape == actual.block_table_tensor.shape
