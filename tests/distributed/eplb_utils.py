# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import os
import random
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.eplb import async_worker
from vllm.distributed.eplb import eplb_state as eplb_state_module
from vllm.distributed.parallel_state import (
    get_ep_group,
    init_distributed_environment,
)
from vllm.platforms import current_platform
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
    accelerator_type = torch.accelerator.current_accelerator().type
    device = torch.device(f"{accelerator_type}:{local_rank}")
    torch.accelerator.set_device_index(device)

    backend = "xccl" if current_platform.is_xpu() else "nccl"

    # Create a minimal vllm config for init_distributed_environment
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(backend=backend)
    atexit.register(_destroy_process_group_if_initialized)
    # Ensure each worker process has the same random seed
    random.seed(42)
    torch.manual_seed(42)


def _destroy_process_group_if_initialized() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# Only turns a deadlock into a failure; a healthy job finishes in milliseconds.
ASYNC_JOB_TIMEOUT_S = 60


class _RollPolicy:
    """Moves every physical expert one slot to the right."""

    calls = 0

    @classmethod
    def rebalance_experts(cls, load, replicas, groups, nodes, gpus, old_map):
        cls.calls += 1
        return torch.roll(old_map, 1, dims=1)


def make_async_eplb_state(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
    num_models: int = 1,
    num_layers: int = 2,
    num_experts: int = 4,
) -> SimpleNamespace:
    """Build async EPLB over the current EP group with fake models whose layer
    transfers and buffer copies are recorded instead of run.

    `run.fail` holds layers whose next transfer raises, and `run.gates` maps a
    layer to a `threading.Event` its transfer waits for.
    """
    run = SimpleNamespace(
        transfers=[], applied=[], fail=set(), gates={}, policy=_RollPolicy
    )
    layer_of: dict[int, tuple[int, int]] = {}

    def transfer_layer(*, expert_weights, **kwargs):
        layer = layer_of[id(expert_weights)]
        gate = run.gates.get(layer)
        if gate is not None:
            assert gate.wait(ASYNC_JOB_TIMEOUT_S)
        if layer in run.fail:
            run.fail.discard(layer)
            raise RuntimeError(f"injected transfer failure at {layer}")
        run.transfers.append(layer)
        return ("metadata", layer, len(run.transfers))

    monkeypatch.setattr(async_worker, "transfer_layer", transfer_layer)
    monkeypatch.setattr(async_worker, "get_eplb_group", get_ep_group)
    monkeypatch.setattr(
        eplb_state_module,
        "move_from_buffer",
        lambda *, transfer_metadata, **kwargs: run.applied.append(transfer_metadata),
    )

    window = 2
    model_states = {}
    for m in range(num_models):
        expert_weights = []
        for layer in range(num_layers):
            weights = [torch.zeros(num_experts, 1, device=device)]
            layer_of[id(weights)] = (m, layer)
            expert_weights.append(weights)
        phy2log = torch.arange(num_experts, device=device).repeat(num_layers, 1)
        model_state = MagicMock()
        model_state.model_name = f"model{m}"
        model_state.model.num_moe_layers = num_layers
        model_state.model.num_logical_experts = num_experts
        model_state.model.expert_weights = expert_weights
        model_state.physical_to_logical_map = phy2log
        model_state.logical_to_physical_map = phy2log.unsqueeze(-1).clone()
        model_state.logical_replica_count = torch.ones_like(phy2log)
        model_state.expert_load_pass = torch.zeros(
            num_layers, num_experts, dtype=torch.int32, device=device
        )
        model_state.expert_load_window = torch.zeros(
            window, num_layers, num_experts, dtype=torch.int32, device=device
        )
        model_states[f"model{m}"] = model_state

    parallel_config = ParallelConfig()
    parallel_config.eplb_config.use_async = True
    state = eplb_state_module.EplbState(parallel_config, device)
    if state.device_index is None:
        pytest.skip("Async EPLB needs a CUDA or XPU device")
    state.is_async = True
    state.policy = _RollPolicy
    state.expert_load_window_size = window
    # Large enough that finishing a cycle never triggers another one; tests
    # start each rearrangement with start_async_rearrangement().
    state.expert_rearrangement_step_interval = 1000
    state.model_states = model_states
    state.start_async_loop()
    run.state = state
    run.model_states = list(model_states.values())
    _RollPolicy.calls = 0
    return run


def start_async_rearrangement(state) -> None:
    """Run the step that reaches the rearrangement interval."""
    state.expert_rearrangement_step = state.expert_rearrangement_step_interval - 1
    state.step()


def async_cycle_running(state) -> bool:
    return state.async_runner.cycle_running


def finish_async_cycle(state) -> None:
    """Step until the running async cycle has applied every layer."""
    deadline = time.monotonic() + ASYNC_JOB_TIMEOUT_S
    while async_cycle_running(state):
        assert time.monotonic() < deadline, "async EPLB job hung"
        state.step()
        time.sleep(0.001)
