# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runs the specialized decode's collective checks on TP4 or TP8 ranks, one
spawned process a GPU, with vLLM's distributed state initialized."""

import tempfile

import pytest
import torch

TP_SIZES = [
    pytest.param(tp, marks=pytest.mark.distributed(num_gpus=tp), id=f"tp{tp}")
    for tp in (4, 8)
]


def _rank(rank, store, check, tp):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    torch.accelerator.set_device_index(rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=tp,
            rank=rank,
            distributed_init_method=f"file://{store}",
            local_rank=rank,
        )
        ensure_model_parallel_initialized(tp, 1)
    try:
        check(rank, torch.device("cuda", rank))
    finally:
        cleanup_dist_env_and_memory()


def run_tp(check, tp):
    """Runs `check(rank, device)` on every rank; needs TP GPUs with NVLink."""
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"Requires {tp} GPUs with NVLink multicast")
    # A file store needs no free port and leaves nothing behind.
    with tempfile.TemporaryDirectory() as tmp:
        torch.multiprocessing.spawn(_rank, args=(f"{tmp}/store", check, tp), nprocs=tp)
