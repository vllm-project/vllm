# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import torch
import torch.multiprocessing as mp


def _run_worker(
    rank: int,
    worker_fn,
    world_size: int,
    tp_size: int,
    dp_size: int,
    init_ports: list[int],
    dp_port: int,
    params,
) -> None:
    worker_fn(
        rank,
        world_size,
        tp_size,
        dp_size,
        init_ports[rank],
        dp_port,
        params,
    )


def spawn_cpu_workers(
    worker_fn,
    world_size: int,
    tp_size: int,
    dp_size: int,
    params,
    *,
    distributed_init_ports: list[int] | None = None,
    dp_port: int | None = None,
    timeout: float = 300,
) -> None:
    """Run CPU distributed workers with spawn and bound hung test cleanup."""
    from vllm.utils.network_utils import get_open_port

    if distributed_init_ports is None:
        init_ports = [get_open_port()] * world_size
    elif len(distributed_init_ports) != world_size:
        raise ValueError("distributed_init_ports must provide one port per rank")
    else:
        init_ports = distributed_init_ports

    if dp_port is None:
        dp_port = get_open_port()

    context = mp.spawn(
        _run_worker,
        args=(
            worker_fn,
            world_size,
            tp_size,
            dp_size,
            init_ports,
            dp_port,
            params,
        ),
        nprocs=world_size,
        join=False,
    )

    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if context.join(timeout=min(1, remaining)):
            return

    for process in context.processes:
        if process.is_alive():
            process.terminate()
    for process in context.processes:
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)

    raise TimeoutError(f"CPU distributed workers exceeded {timeout:.0f}s timeout")


def init_cpu_distributed_environment(
    rank, tp_size, dp_size, port, dp_port, *, enable_expert_parallel=False
):
    """Init vLLM distributed env for TP=tp_size, DP=dp_size."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    dp_rank = rank // tp_size
    tp_rank = rank % tp_size

    vllm_config = VllmConfig()
    vllm_config.parallel_config = ParallelConfig(
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        enable_expert_parallel=enable_expert_parallel,
        _data_parallel_master_port_list=[int(dp_port)],
    )
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=tp_size,
            rank=tp_rank,
            distributed_init_method=f"tcp://localhost:{port}",
            local_rank=rank,
            backend="gloo",
        )
        ensure_model_parallel_initialized(tp_size, 1, backend="gloo")


def make_cpu_forward_context(
    dp_rank,
    dp_size,
    num_tokens_per_dp_rank,
    num_tokens_across_dp=None,
):
    """Forward context with DP metadata for EP group dispatch."""
    from vllm.config.parallel import ParallelConfig
    from vllm.config.vllm import VllmConfig
    from vllm.forward_context import set_forward_context

    class _AttnMeta:
        dp_metadata = None

    vllm_config = VllmConfig()
    vllm_config.parallel_config = ParallelConfig(
        data_parallel_size=dp_size,
        is_moe_model=True,
        data_parallel_rank=dp_rank,
    )
    return set_forward_context(
        _AttnMeta(),
        vllm_config,
        num_tokens=num_tokens_per_dp_rank,
        num_tokens_across_dp=torch.tensor(
            num_tokens_across_dp
            if num_tokens_across_dp is not None
            else [num_tokens_per_dp_rank] * dp_size,
            dtype=torch.int,
        ),
    )
