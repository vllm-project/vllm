# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import os
import random

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    init_distributed_environment,
)
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)


def _distributed_worker_wrapper(fn, env, world_size, args, rank, skip_queue):
    try:
        fn(env, world_size, *args)
    except BaseException as exc:
        if isinstance(exc, pytest.skip.Exception):
            skip_queue.put((rank, str(exc)))
            return
        raise


def distributed_run(fn, world_size, *args):
    number_of_processes = world_size
    processes: list[mp.Process] = []
    skip_queue: mp.SimpleQueue = mp.SimpleQueue()
    for i in range(number_of_processes):
        env: dict[str, str] = {}
        env["RANK"] = str(i)
        env["LOCAL_RANK"] = str(i)
        env["WORLD_SIZE"] = str(number_of_processes)
        env["LOCAL_WORLD_SIZE"] = str(number_of_processes)
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "12345"
        p = mp.Process(
            target=_distributed_worker_wrapper,
            args=(fn, env, world_size, args, i, skip_queue),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    skipped: list[tuple[int, str]] = []
    while not skip_queue.empty():
        rank, reason = skip_queue.get()
        skipped.append((rank, reason))

    if len(skipped) == number_of_processes:
        reason = skipped[0][1]
        pytest.skip(reason)
    if 0 < len(skipped) < number_of_processes:
        skipped_ranks = sorted(rank for rank, _ in skipped)
        raise AssertionError(
            "Distributed test had partial skips; expected either all ranks "
            f"to skip or none. Skipped ranks: {skipped_ranks}, "
            f"total ranks: {number_of_processes}"
        )

    for p in processes:
        assert p.exitcode == 0


def set_env_vars_and_device(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = os.environ["LOCAL_RANK"]
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)

    # Create a minimal vllm config for init_distributed_environment
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        init_distributed_environment()
    atexit.register(_destroy_process_group_if_initialized)
    # Ensure each worker process has the same random seed
    random.seed(42)
    torch.manual_seed(42)


def _destroy_process_group_if_initialized() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
