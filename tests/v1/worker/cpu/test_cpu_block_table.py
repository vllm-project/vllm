# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.cpu.block_table import CPUBlockTables

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)


def test_append_block_ids_updates_cpu_tensors_directly() -> None:
    block_tables = CPUBlockTables(
        block_sizes=[8, 16],
        max_num_reqs=2,
        max_num_batched_tokens=8,
        max_num_blocks_per_group=[4, 4],
        device=torch.device("cpu"),
    )

    block_tables.append_block_ids(1, ([1, 2], [10]), overwrite=True)
    block_tables.append_block_ids(1, ([3], [11, 12]), overwrite=False)

    assert block_tables.block_tables[0][1].tolist() == [1, 2, 3, 0]
    assert block_tables.block_tables[1][1].tolist() == [10, 11, 12, 0]
    assert block_tables.num_blocks[:, 1].tolist() == [3, 3]


def test_append_block_ids_rejects_write_past_row_capacity() -> None:
    block_tables = CPUBlockTables(
        block_sizes=[8],
        max_num_reqs=2,
        max_num_batched_tokens=8,
        max_num_blocks_per_group=[4],
        device=torch.device("cpu"),
    )
    block_tables.append_block_ids(1, ([1, 2, 3],), overwrite=True)

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(1, ([4, 5],), overwrite=False)

    assert block_tables.block_tables[0][1].tolist() == [1, 2, 3, 0]
    assert block_tables.num_blocks[0, 1].item() == 3


def test_gather_block_tables_cpu() -> None:
    block_tables = CPUBlockTables(
        block_sizes=[8],
        max_num_reqs=4,
        max_num_batched_tokens=8,
        max_num_blocks_per_group=[6],
        device=torch.device("cpu"),
    )
    source_storage = torch.arange(4 * 8, dtype=torch.int32).view(4, 8)
    block_tables.block_tables[0].copy_(source_storage[:, 1:7])
    block_tables.num_blocks.copy_(torch.tensor([[3, 1, 4, 2]], dtype=torch.int32))
    output_storage = torch.full((5, 8), -1, dtype=torch.int32)
    output = output_storage[:, 1:7]

    gathered = block_tables.gather_block_tables(
        idx_mapping=torch.tensor([2, 0], dtype=torch.int64),
        num_reqs_padded=3,
        out=(output,),
    )

    assert gathered[0].data_ptr() == output.data_ptr()
    assert gathered[0].tolist() == [
        [17, 18, 19, 20, 0, 0],
        [1, 2, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    assert output[3:].eq(-1).all()


def test_compute_slot_mappings_cpu() -> None:
    block_tables = CPUBlockTables(
        block_sizes=[8, 8],
        max_num_reqs=2,
        max_num_batched_tokens=8,
        max_num_blocks_per_group=[3, 3],
        device=torch.device("cpu"),
        slot_mapping_enabled=[True, False],
    )
    block_tables.block_tables[0].copy_(
        torch.tensor([[2, 3, 4], [10, 11, 12]], dtype=torch.int32)
    )
    output_storage = torch.full((2, 10), 99, dtype=torch.int64)
    output = output_storage[:, 1:9]
    positions_storage = torch.tensor(
        [-99, 0, 4, 100, 101, 3, 8, -99], dtype=torch.int64
    )

    actual = block_tables.compute_slot_mappings(
        idx_mapping=torch.tensor([1, -1, 0], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        positions=positions_storage[1:7],
        num_tokens_padded=7,
        out=output,
    )

    assert actual.data_ptr() == output.data_ptr()
    assert actual.tolist() == [
        [80, 84, -1, -1, 19, 24, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ]
    assert output[:, 7:].eq(-1).all()


def test_dummy_block_tables_clear_previous_request_rows() -> None:
    block_tables = CPUBlockTables(
        block_sizes=[16],
        max_num_reqs=4,
        max_num_batched_tokens=64,
        max_num_blocks_per_group=[8],
        device=torch.device("cpu"),
    )
    block_tables.append_block_ids(0, ([1, 2],), overwrite=True)
    block_tables.gather_block_tables(torch.tensor([0]), num_reqs_padded=1)
    assert block_tables.input_block_tables[0][0, 0].item() == 1

    dummy = block_tables.get_dummy_block_tables(num_reqs=1)

    assert torch.count_nonzero(dummy[0]).item() == 0
    assert dummy[0].data_ptr() == block_tables.input_block_tables[0].data_ptr()


def test_dummy_request_slot_mapping_is_pad() -> None:
    block_tables = CPUBlockTables(
        block_sizes=[4],
        max_num_reqs=2,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[4],
        device=torch.device("cpu"),
    )
    block_tables.append_block_ids(0, ([7, 8],), overwrite=True)
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    positions = torch.tensor([1, 2, 3], dtype=torch.int64)

    real = block_tables.compute_slot_mappings(
        torch.tensor([0]), query_start_loc, positions, num_tokens_padded=3
    )
    assert real[0].tolist() == [7 * 4 + 1, 7 * 4 + 2, 7 * 4 + 3]

    dummy = block_tables.compute_slot_mappings(
        torch.tensor([-1]), query_start_loc, positions, num_tokens_padded=3
    )
    assert dummy[0].tolist() == [PAD_SLOT_ID] * 3


def test_gather_nulls_blocks_below_sliding_window() -> None:
    block_tables = CPUBlockTables(
        block_sizes=[16, 16],
        max_num_reqs=4,
        max_num_batched_tokens=64,
        max_num_blocks_per_group=[16, 16],
        device=torch.device("cpu"),
        sliding_window_reach=[0, 31],
    )
    full_ids = [list(range(1, 13)), list(range(41, 53))]
    swa_ids = [list(range(21, 33)), list(range(61, 73))]
    req_slots = [0, 2]
    for req, slot in enumerate(req_slots):
        block_tables.append_block_ids(
            slot, (full_ids[req], swa_ids[req]), overwrite=True
        )

    idx_mapping = torch.tensor(req_slots)
    num_computed_tokens = torch.tensor([145, 0, 120, 0], dtype=torch.int32)
    full, swa = block_tables.gather_block_tables(
        idx_mapping,
        num_reqs_padded=3,
        num_computed_tokens=num_computed_tokens,
    )

    assert full[0, :12].tolist() == full_ids[0]
    assert full[1, :12].tolist() == full_ids[1]
    assert swa[0, :12].tolist() == [0] * 7 + swa_ids[0][7:]
    assert swa[1, :12].tolist() == [0] * 5 + swa_ids[1][5:]
    assert torch.count_nonzero(swa[2]).item() == 0
    assert block_tables.block_tables[1][0, :12].tolist() == swa_ids[0]

    _, swa = block_tables.gather_block_tables(idx_mapping, num_reqs_padded=2)
    assert swa[0, :12].tolist() == swa_ids[0]


def test_shared_block_tables_clear_rows_from_larger_batch() -> None:
    tables = CPUBlockTables([16], 8, 64, [3], torch.device("cpu"))
    for req_index in range(8):
        tables.append_block_ids(req_index, ([req_index + 1] * 3,), overwrite=True)
    tables.gather_block_tables(torch.arange(8), num_reqs_padded=8)
    ptr = tables.input_block_tables[0].data_ptr()

    idx = torch.tensor([7, 2, 0, 5, 1])
    for num_reqs in (3, 5, 0):
        num_rows = num_reqs + 1
        out = tables.gather_block_tables(idx[:num_reqs], num_rows)[0]
        assert out.shape == (num_rows, 3)
        assert tables.input_block_tables[0].data_ptr() == ptr
        assert torch.count_nonzero(tables.input_block_tables[0][num_reqs:]).item() == 0
        expected = (idx[:num_reqs, None] + 1).expand(-1, 3)
        torch.testing.assert_close(out[:num_reqs], expected.to(torch.int32))


def test_external_block_table_output_preserves_shared_rows() -> None:
    tables = CPUBlockTables([16], 4, 64, [3], torch.device("cpu"))
    for req_index in range(4):
        tables.append_block_ids(req_index, ([req_index + 1] * 3,), overwrite=True)
    tables.gather_block_tables(torch.arange(4), num_reqs_padded=4)
    previous = tables.input_block_tables[0].clone()
    external = torch.full_like(previous[:3], -1)

    result = tables.gather_block_tables(torch.tensor([3, 1]), 3, out=(external,))[0]

    torch.testing.assert_close(tables.input_block_tables[0], previous)
    assert result[:, 0].tolist() == [4, 2, 0]
    tables.gather_block_tables(torch.tensor([3]), num_reqs_padded=1)
    assert torch.count_nonzero(tables.input_block_tables[0][1:]).item() == 0
