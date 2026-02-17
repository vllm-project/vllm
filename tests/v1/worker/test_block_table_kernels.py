"""Tests for fused block table kernels."""

import pytest
import torch

from vllm.v1.worker.gpu.block_table import BlockTables


@pytest.fixture
def device():
    return torch.device("cuda:0")


def _setup_block_tables(
    device,
    num_kv_cache_groups=1,
    block_size=16,
    max_num_reqs=32,
    max_model_len=512,
    max_num_batched_tokens=1024,
):
    block_sizes = [block_size] * num_kv_cache_groups
    bt = BlockTables(
        block_sizes=block_sizes,
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        device=device,
    )
    return bt


def _populate_block_tables(bt, num_reqs, block_size, max_model_len):
    max_blocks = max_model_len // block_size
    for req_idx in range(num_reqs):
        num_blocks = min(req_idx + 1, max_blocks)
        block_ids = tuple(
            list(range(req_idx * 100, req_idx * 100 + num_blocks))
            for _ in range(bt.num_kv_cache_groups)
        )
        bt.append_block_ids(req_idx, block_ids, overwrite=True)
    bt.apply_staged_writes()


@pytest.mark.parametrize("num_kv_cache_groups", [1, 2, 4])
@pytest.mark.parametrize("num_reqs", [1, 16, 64])
def test_fused_gather_and_slots_matches_separate(device, num_kv_cache_groups, num_reqs):
    """Verify fused kernel produces identical results to separate kernels."""
    block_size = 16
    max_model_len = 512
    max_num_batched_tokens = 2048

    bt = _setup_block_tables(
        device,
        num_kv_cache_groups,
        block_size,
        max_num_reqs=64,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    _populate_block_tables(bt, num_reqs, block_size, max_model_len)

    # Create test inputs
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    tokens_per_req = max(1, max_num_batched_tokens // max(num_reqs, 1))
    tokens_per_req = min(tokens_per_req, 32)  # Cap for test
    num_tokens = num_reqs * tokens_per_req

    query_start_loc = torch.arange(
        0, num_tokens + 1, tokens_per_req, dtype=torch.int32, device=device
    )[: num_reqs + 1]
    query_start_loc[-1] = num_tokens

    positions = torch.zeros(num_tokens, dtype=torch.long, device=device)
    for i in range(num_reqs):
        start = query_start_loc[i].item()
        end = query_start_loc[i + 1].item()
        positions[start:end] = torch.arange(
            end - start, dtype=torch.long, device=device
        )

    # Run SEPARATE kernels (reference)
    ref_block_tables = bt.gather_block_tables(idx_mapping)
    ref_slot_mappings = bt.compute_slot_mappings(
        idx_mapping, query_start_loc, positions
    )

    # Save reference results
    ref_bt_list = [t.clone() for t in ref_block_tables]
    ref_sm = ref_slot_mappings.clone()

    # Reset input_block_tables to zeros
    for ibt in bt.input_block_tables:
        ibt.zero_()
    bt.slot_mappings.zero_()

    # Run FUSED kernel
    fused_block_tables, fused_slot_mappings = bt.gather_and_compute_slot_mappings(
        idx_mapping, query_start_loc, positions
    )

    # Compare block tables
    for i, (ref, fused) in enumerate(zip(ref_bt_list, fused_block_tables)):
        assert torch.equal(ref, fused), (
            f"Block table mismatch for group {i}:\n"
            f"  ref:   {ref[:3, :10]}\n"
            f"  fused: {fused[:3, :10]}"
        )

    # Compare slot mappings
    assert torch.equal(ref_sm, fused_slot_mappings), (
        f"Slot mapping mismatch:\n"
        f"  ref:   {ref_sm[0, :20]}\n"
        f"  fused: {fused_slot_mappings[0, :20]}"
    )

    print(f"PASSED: groups={num_kv_cache_groups}, reqs={num_reqs}")


@pytest.mark.parametrize("num_kv_cache_groups", [1, 2])
def test_fused_with_shuffled_idx_mapping(device, num_kv_cache_groups):
    """Test with non-identity idx_mapping (requests not in order)."""
    num_reqs = 16
    block_size = 16
    max_model_len = 256

    bt = _setup_block_tables(
        device,
        num_kv_cache_groups,
        block_size,
        max_num_reqs=32,
        max_model_len=max_model_len,
        max_num_batched_tokens=1024,
    )
    _populate_block_tables(bt, 32, block_size, max_model_len)

    # Shuffled mapping: batch_idx 0 -> req 5, batch_idx 1 -> req 2, etc.
    perm = torch.randperm(32, device=device)[:num_reqs].to(torch.int32)
    idx_mapping = perm

    tokens_per_req = 8
    num_tokens = num_reqs * tokens_per_req
    query_start_loc = torch.arange(
        0, num_tokens + 1, tokens_per_req, dtype=torch.int32, device=device
    )[: num_reqs + 1]

    positions = torch.zeros(num_tokens, dtype=torch.long, device=device)
    for i in range(num_reqs):
        start = query_start_loc[i].item()
        end = query_start_loc[i + 1].item()
        positions[start:end] = torch.arange(
            end - start, dtype=torch.long, device=device
        )

    # Reference
    ref_block_tables = bt.gather_block_tables(idx_mapping)
    ref_slot_mappings = bt.compute_slot_mappings(
        idx_mapping, query_start_loc, positions
    )
    ref_bt_list = [t.clone() for t in ref_block_tables]
    ref_sm = ref_slot_mappings.clone()

    # Reset and run fused
    for ibt in bt.input_block_tables:
        ibt.zero_()
    bt.slot_mappings.zero_()

    fused_block_tables, fused_slot_mappings = bt.gather_and_compute_slot_mappings(
        idx_mapping, query_start_loc, positions
    )

    for i, (ref, fused) in enumerate(zip(ref_bt_list, fused_block_tables)):
        assert torch.equal(ref, fused), (
            f"Block table mismatch for group {i} with shuffled mapping"
        )

    assert torch.equal(ref_sm, fused_slot_mappings), (
        "Slot mapping mismatch with shuffled mapping"
    )
    print(f"PASSED: shuffled idx_mapping, groups={num_kv_cache_groups}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
