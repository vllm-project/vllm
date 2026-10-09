# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm import _custom_ops as ops

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.block_table import BlockTables


def gather_block_tables(
    self: "BlockTables",
    idx_mapping: torch.Tensor,
    num_reqs_padded: int,
    out: tuple[torch.Tensor, ...] | None = None,
    out_ptrs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if self.num_kv_cache_groups == 0:
        return ()
    if out is None:
        out = tuple(self.input_block_tables)
    else:
        assert out_ptrs is not None
        assert len(out) == self.num_kv_cache_groups

    for group_id, output in enumerate(out):
        ops.gather_block_table_cpu(
            self.block_tables[group_id].gpu,
            output,
            idx_mapping,
            self.num_blocks.gpu[group_id],
            num_reqs_padded,
        )
    return tuple(block_table[:num_reqs_padded] for block_table in out)
