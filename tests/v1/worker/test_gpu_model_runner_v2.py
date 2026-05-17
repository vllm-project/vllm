# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


def test_v2_block_tables_kernel_block_expansion():
    from vllm.v1.worker.gpu.block_table import BlockTables

    block_tables = BlockTables(
        block_sizes=[128],
        kernel_block_sizes=[64],
        max_num_reqs=4,
        max_num_batched_tokens=256,
        max_num_blocks_per_group=[10],
        device=torch.device(DEVICE_TYPE),
    )

    block_tables.append_block_ids(0, ([0, 1, 2],), overwrite=True)
    block_tables.apply_staged_writes()

    assert block_tables.blocks_per_kv_block == [2]
    assert block_tables.block_tables[0].gpu[0, :6].cpu().tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
