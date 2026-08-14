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


def test_v1_block_table_move_row_clears_vacated_row():
    """condense() moves the last row into a freed slot; the vacated row must
    not keep stale block ids. Padded dummy-run batches dereference stale rows
    as mamba state slots (bypassing the NULL_BLOCK_ID fill of real decode
    padding) and write state in place there — corrupting the blocks' new
    owner once they are reallocated, e.g. to an in-flight NIXL load."""
    from vllm.v1.worker.block_table import BlockTable

    block_table = BlockTable(
        block_size=16,
        max_num_reqs=4,
        max_num_blocks_per_req=8,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cuda"),
        kernel_block_size=16,
        cp_kv_cache_interleave_size=1,
    )
    block_table.add_row([7, 8, 9], row_idx=0)
    block_table.add_row([4, 5], row_idx=1)

    block_table.move_row(1, 0)

    assert block_table.block_table.np[0, :2].tolist() == [4, 5]
    assert block_table.num_blocks_per_row[0] == 2
    # The vacated source row routes to the reserved null block.
    assert block_table.num_blocks_per_row[1] == 0
    assert (block_table.block_table.np[1] == 0).all()


def test_get_dummy_block_tables_returns_zeroed_rows():
    """Dummy runs bypass the gather, so the persistent input_block_tables
    hold the previous real step's rows. Mamba/GDN metadata routes in-place
    state writes through block_table[:, 0] (dummy slot mappings are
    PAD-filled, state indices are not), so stale rows would direct dummy
    state writes at freed — possibly reallocated — blocks.
    get_dummy_block_tables must hand out zeroed (null block) rows while
    preserving the persistent storage address for CUDA graphs."""
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16],
        max_num_reqs=4,
        max_num_batched_tokens=64,
        max_num_blocks_per_group=[8],
        device=device,
        kernel_block_sizes=[16],
    )
    # Simulate a real step: stage a request's blocks and gather them into
    # the persistent input block tables.
    block_tables.append_block_ids(req_index=0, new_block_ids=([1, 2],), overwrite=True)
    block_tables.apply_staged_writes()
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=device)
    block_tables.gather_block_tables(idx_mapping, num_reqs_padded=1)
    torch.accelerator.synchronize()
    assert block_tables.input_block_tables[0][0, 0].item() == 1

    dummy = block_tables.get_dummy_block_tables(num_reqs=1)
    torch.accelerator.synchronize()
    assert (dummy[0] == 0).all()
    # CUDA graph invariant: same persistent tensor, not a fresh allocation.
    assert dummy[0].data_ptr() == block_tables.input_block_tables[0].data_ptr()
