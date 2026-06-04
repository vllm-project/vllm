# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def test_block_table_warmup_compute_slot_mapping(monkeypatch):
    from vllm.v1.worker import block_table as block_table_module
    from vllm.v1.worker.block_table import BlockTable

    calls = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(
        block_table_module,
        "_compute_slot_mapping_kernel",
        FakeKernel(),
    )

    block_table = BlockTable(
        block_size=16,
        max_num_reqs=4,
        max_num_blocks_per_req=8,
        max_num_batched_tokens=128,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=16,
        cp_kv_cache_interleave_size=1,
    )

    block_table.warmup_compute_slot_mapping()

    assert len(calls) == 1
    grid, args, kwargs = calls[0]
    assert grid == (2,)
    assert args[0] == 1
    assert args[1] == 128
    assert args[2].tolist() == [0, 1]
    assert args[3].tolist() == [0]
    assert args[6] == 16
    assert kwargs["BLOCK_SIZE"] == 1024


def test_multi_group_block_table_warmup_compute_slot_mapping(monkeypatch):
    from vllm.v1.worker import block_table as block_table_module
    from vllm.v1.worker.block_table import MultiGroupBlockTable

    calls = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(
        block_table_module,
        "_compute_slot_mapping_kernel",
        FakeKernel(),
    )

    block_tables = MultiGroupBlockTable(
        max_num_reqs=4,
        max_model_len=128,
        max_num_batched_tokens=128,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16, 32],
        kernel_block_sizes=[16, 16],
        max_num_blocks=[8, 8],
    )

    block_tables.warmup_compute_slot_mapping()

    assert len(calls) == 2
    assert [call[0] for call in calls] == [(2,), (2,)]
    assert [call[1][6] for call in calls] == [16, 16]
