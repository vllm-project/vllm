# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.block_table import BlockTables

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="requires CUDA",
)


def test_block_tables_apply_staged_writes_fuses_kv_groups(monkeypatch):
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16, 32, 8],
        max_num_reqs=4,
        max_num_batched_tokens=64,
        max_num_blocks_per_group=[8, 8, 8],
        device=device,
        kernel_block_sizes=[16, 16, 8],
    )

    def fail_if_apply_write_called():
        pytest.fail("multi-group writes should use the fused apply kernel")

    for block_table in block_tables.block_tables:
        monkeypatch.setattr(block_table, "apply_write", fail_if_apply_write_called)

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([1, 2], [10, 11], []),
        overwrite=True,
    )
    block_tables.append_block_ids(
        req_index=1,
        new_block_ids=([3], [12], [5, 6]),
        overwrite=True,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :2],
        torch.tensor([1, 2], dtype=torch.int32, device=device),
    )
    # Group 1 has blocks_per_kv_block == 2, so each KV block expands to two
    # kernel block IDs.
    assert torch.equal(
        block_tables.block_tables[1].gpu[0, :4],
        torch.tensor([20, 21, 22, 23], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[0].gpu[1, :1],
        torch.tensor([3], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[1].gpu[1, :2],
        torch.tensor([24, 25], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[2].gpu[1, :2],
        torch.tensor([5, 6], dtype=torch.int32, device=device),
    )
    assert block_tables.num_blocks.np[0, 0] == 2
    assert block_tables.num_blocks.np[1, 0] == 4
    assert block_tables.num_blocks.np[2, 0] == 0
    assert block_tables.num_blocks.np[0, 1] == 1
    assert block_tables.num_blocks.np[1, 1] == 2
    assert block_tables.num_blocks.np[2, 1] == 2
    assert torch.equal(
        block_tables.num_blocks.gpu[:, :2],
        torch.tensor([[2, 1], [4, 2], [0, 2]], dtype=torch.int32, device=device),
    )

    for block_table in block_tables.block_tables:
        assert not block_table._staged_write_indices
        assert not block_table._staged_write_starts
        assert not block_table._staged_write_contents
        assert not block_table._staged_write_cu_lens

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([7], [13], [8]),
        overwrite=False,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :3],
        torch.tensor([1, 2, 7], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[1].gpu[0, :6],
        torch.tensor([20, 21, 22, 23, 26, 27], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[2].gpu[0, :1],
        torch.tensor([8], dtype=torch.int32, device=device),
    )
    assert block_tables.num_blocks.np[0, 0] == 3
    assert block_tables.num_blocks.np[1, 0] == 6
    assert block_tables.num_blocks.np[2, 0] == 1


def test_block_tables_apply_staged_writes_single_group():
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16],
        max_num_reqs=2,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[4],
        device=device,
        kernel_block_sizes=[16],
    )

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([1, 2],),
        overwrite=True,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :2],
        torch.tensor([1, 2], dtype=torch.int32, device=device),
    )


def _compact_rank_local_slots(rank_local_slots: torch.Tensor) -> torch.Tensor:
    """Reference the removed DCP all-gather and compaction."""
    # [dcp_rank, group, token] -> [group, token, dcp_rank]
    owner_slots = rank_local_slots.permute(1, 2, 0).contiguous()
    valid = owner_slots >= 0
    owner_rank = valid.to(torch.int64).argmax(dim=-1)
    owner_local_slot = owner_slots.gather(-1, owner_rank.unsqueeze(-1)).squeeze(-1)
    is_valid = valid.sum(dim=-1) == 1
    owner_rank = torch.where(is_valid, owner_rank, torch.full_like(owner_rank, -1))
    owner_local_slot = torch.where(
        is_valid, owner_local_slot, torch.full_like(owner_local_slot, -1)
    )
    return torch.stack((owner_rank, owner_local_slot), dim=-1)


def test_compute_slot_mappings_emits_direct_owner_slots() -> None:
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[64, 64],
        max_num_reqs=2,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[4, 4],
        device=device,
        kernel_block_sizes=[64, 64],
        cp_size=4,
        cp_rank=0,
        cp_interleave=64,
    )
    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([5, 7], [50, 70]),
        overwrite=True,
    )
    block_tables.append_block_ids(
        req_index=1,
        new_block_ids=([11, 13], [110, 130]),
        overwrite=True,
    )
    block_tables.apply_staged_writes()

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 7, 12], dtype=torch.int32, device=device)
    positions = torch.tensor(
        [0, 63, 64, 127, 255, 256, 319, 32, 95, 191, 256, 511],
        dtype=torch.int64,
        device=device,
    )
    owner_slots_by_rank = []
    rank_local_slots = []
    for rank in range(4):
        block_tables.cp_rank = rank
        local_out = torch.empty((2, 16), dtype=torch.int64, device=device)
        owner_out = torch.empty((2, 16, 2), dtype=torch.int64, device=device)
        local = block_tables.compute_slot_mappings(
            idx_mapping,
            query_start_loc,
            positions,
            num_tokens_padded=16,
            out=local_out,
            owner_slot_mappings_out=owner_out,
        )
        rank_local_slots.append(local.clone())
        owner_slots_by_rank.append(owner_out.clone())

    torch.accelerator.synchronize()
    old_compacted = _compact_rank_local_slots(torch.stack(rank_local_slots))
    for owner_slots in owner_slots_by_rank:
        torch.testing.assert_close(owner_slots, old_compacted)

    noncontiguous_pairs = torch.empty(
        (2, 2, 16), dtype=torch.int64, device=device
    ).permute(0, 2, 1)
    with pytest.raises(ValueError, match="contiguous in their last dimension"):
        block_tables.compute_slot_mappings(
            idx_mapping,
            query_start_loc,
            positions,
            num_tokens_padded=16,
            out=torch.empty((2, 16), dtype=torch.int64, device=device),
            owner_slot_mappings_out=noncontiguous_pairs,
        )

    # The kernel pads both ordinary and direct-owner metadata to the persistent
    # buffer boundary, including both fields in the owner pair.
    assert torch.all(old_compacted[:, 12:] == -1)
    assert torch.all(torch.stack(rank_local_slots)[:, :, 12:] == -1)
