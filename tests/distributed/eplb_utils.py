# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import random

import torch
import torch.multiprocessing as mp

from vllm.distributed.parallel_state import (
    init_distributed_environment,
)
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)


def distributed_run(fn, world_size, *args, eplb_communicator: str = "torch"):
    number_of_processes = world_size
    processes: list[mp.Process] = []
    for i in range(number_of_processes):
        env: dict[str, str] = {}
        env["RANK"] = str(i)
        env["LOCAL_RANK"] = str(i)
        env["WORLD_SIZE"] = str(number_of_processes)
        env["LOCAL_WORLD_SIZE"] = str(number_of_processes)
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "12345"
        env["VLLM_EPLB_COMMUNICATOR"] = eplb_communicator
        p = mp.Process(target=fn, args=(env, world_size, *args))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def set_env_vars_and_device(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = os.environ["LOCAL_RANK"]
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    init_distributed_environment()

    # Ensure each worker process has the same random seed
    random.seed(42)
    torch.manual_seed(42)
