# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.cpu.block_table import CPUBlockTables
from vllm.v1.worker.gpu.block_table import BlockTables

pytestmark = pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")


def _make_cpu_block_tables(**kwargs: Any) -> CPUBlockTables:
    block_tables = CPUBlockTables.__new__(CPUBlockTables)
    vars(block_tables).update(kwargs)
    return block_tables


def test_gather_block_tables_cpu() -> None:
    source_storage = torch.arange(4 * 8, dtype=torch.int32).view(4, 8)
    source = source_storage[:, 1:7]
    output_storage = torch.full((5, 8), -1, dtype=torch.int32)
    output = output_storage[:, 1:7]
    block_tables = _make_cpu_block_tables(
        num_kv_cache_groups=1,
        block_tables=[SimpleNamespace(gpu=source)],
        input_block_tables=[output],
        num_blocks=SimpleNamespace(gpu=torch.tensor([[3, 1, 4, 2]], dtype=torch.int32)),
    )

    gathered = block_tables.gather_block_tables(
        idx_mapping=torch.tensor([2, 0], dtype=torch.int64),
        num_reqs_padded=3,
    )

    assert gathered[0].data_ptr() == output.data_ptr()
    assert gathered[0].tolist() == [
        [17, 18, 19, 20, -1, -1],
        [1, 2, 3, -1, -1, -1],
        [0, 0, 0, 0, 0, 0],
    ]
    assert output[3:].eq(-1).all()


def test_compute_slot_mappings_cpu() -> None:
    block_table = torch.tensor(
        [[2, 3, 4], [10, 11, 12]],
        dtype=torch.int32,
    )
    output_storage = torch.full((2, 10), 99, dtype=torch.int64)
    output = output_storage[:, 1:9]
    block_tables = _make_cpu_block_tables(
        num_kv_cache_groups=2,
        block_tables=[
            SimpleNamespace(gpu=block_table),
            SimpleNamespace(gpu=torch.zeros_like(block_table)),
        ],
        slot_mappings=output,
        block_sizes=[8, 8],
        _slot_mapping_enabled=[True, False],
        dcp_sharded=torch.tensor([True, True]),
        cp_size=1,
        cp_rank=0,
        cp_interleave=1,
    )
    positions_storage = torch.tensor(
        [-99, 0, 4, 100, 101, 3, 8, -99], dtype=torch.int64
    )

    actual = block_tables.compute_slot_mappings(
        idx_mapping=torch.tensor([1, -1, 0], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        positions=positions_storage[1:7],
        num_tokens_padded=7,
    )

    assert actual.data_ptr() == output.data_ptr()
    assert actual.tolist() == [
        [80, 84, -1, -1, 19, 24, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ]
    assert output[:, 7:].eq(-1).all()


def test_compute_slot_mappings_cpu_dcp() -> None:
    positions = torch.arange(16, dtype=torch.int64)
    for cp_rank in range(2):
        output = torch.empty((2, 16), dtype=torch.int64)
        block_tables = _make_cpu_block_tables(
            num_kv_cache_groups=2,
            block_tables=[
                SimpleNamespace(gpu=torch.tensor([[10, 11]], dtype=torch.int32)),
                SimpleNamespace(
                    gpu=torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
                ),
            ],
            slot_mappings=output,
            block_sizes=[8, 8],
            _slot_mapping_enabled=[True, True],
            dcp_sharded=torch.tensor([True, False]),
            cp_size=2,
            cp_rank=cp_rank,
            cp_interleave=2,
        )

        actual = block_tables.compute_slot_mappings(
            idx_mapping=torch.tensor([0], dtype=torch.int64),
            query_start_loc=torch.tensor([0, 16], dtype=torch.int32),
            positions=positions,
            num_tokens_padded=16,
        )

        expected_sharded = torch.full((16,), -1, dtype=torch.int64)
        local_mask = positions.div(2, rounding_mode="floor").remainder(2) == cp_rank
        local_positions = positions.div(4, rounding_mode="floor") * 2
        local_positions += positions.remainder(2)
        expected_sharded[local_mask] = 80 + local_positions[local_mask]
        assert torch.equal(actual[0], expected_sharded)
        assert torch.equal(actual[1], 160 + positions)


def test_cpu_runner_selects_block_tables_only_without_triton() -> None:
    from vllm.v1.worker.cpu import model_runner as cpu_model_runner

    triton_runner = object.__new__(cpu_model_runner.CPUModelRunner)
    no_triton_runner = object.__new__(cpu_model_runner.NoTritonCPUModelRunner)

    assert triton_runner.block_tables_cls is BlockTables
    assert no_triton_runner.block_tables_cls is CPUBlockTables
