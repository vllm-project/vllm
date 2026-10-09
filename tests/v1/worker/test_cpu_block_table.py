# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.v1.worker.cpu.block_table import gather_block_tables


def test_gather_block_tables_cpu() -> None:
    source_storage = torch.arange(4 * 8, dtype=torch.int32).view(4, 8)
    source = source_storage[:, 1:7]
    output_storage = torch.full((5, 8), -1, dtype=torch.int32)
    output = output_storage[:, 1:7]
    block_tables = SimpleNamespace(
        num_kv_cache_groups=1,
        block_tables=[SimpleNamespace(gpu=source)],
        input_block_tables=[output],
        num_blocks=SimpleNamespace(gpu=torch.tensor([[3, 1, 4, 2]], dtype=torch.int32)),
    )

    gathered = gather_block_tables(
        block_tables,
        idx_mapping=torch.tensor([2, 0], dtype=torch.int32),
        num_reqs_padded=3,
    )

    assert gathered[0].data_ptr() == output.data_ptr()
    assert gathered[0].tolist() == [
        [17, 18, 19, 20, -1, -1],
        [1, 2, 3, -1, -1, -1],
        [0, 0, 0, 0, 0, 0],
    ]
    assert output[3:].eq(-1).all()
