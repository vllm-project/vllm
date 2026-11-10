# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runs the specialized decode's collective checks on four TP ranks, one
spawned process a GPU, with vLLM's distributed state initialized."""

import tempfile

import pytest
import torch

TP = 4


def _rank(rank, store, check):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    torch.accelerator.set_device_index(rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=TP,
            rank=rank,
            distributed_init_method=f"file://{store}",
            local_rank=rank,
        )
        ensure_model_parallel_initialized(TP, 1)
    try:
        check(rank, torch.device("cuda", rank))
    finally:
        cleanup_dist_env_and_memory()


def run_tp4(check):
    """Runs `check(rank, device)` on every rank; skips without four GPUs."""
    if torch.accelerator.device_count() < TP:
        pytest.skip("Requires four GPUs with NVLink multicast")
    # A file store needs no free port and leaves nothing behind.
    with tempfile.TemporaryDirectory() as tmp:
        torch.multiprocessing.spawn(_rank, args=(f"{tmp}/store", check), nprocs=TP)
