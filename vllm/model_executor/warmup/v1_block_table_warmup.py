# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up v1 block-table Triton kernels."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

_SLOT_MAPPING_WARMUP_TOKENS = 8


def warm_v1_block_table_kernels(runner: "GPUModelRunner") -> None:
    """JIT-compile ``_compute_slot_mapping_kernel`` for the real block tables."""
    device = runner.device
    block_table = runner.input_batch.block_table
    num_tokens = min(
        _SLOT_MAPPING_WARMUP_TOKENS,
        runner.scheduler_config.max_num_batched_tokens,
    )
    if num_tokens <= 0:
        return

    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    block_table.compute_slot_mapping(1, query_start_loc, positions)
