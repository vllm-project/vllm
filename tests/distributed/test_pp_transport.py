# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP transports carry every step's tensors intact through a PP chain, with
varying sizes and stages that take turns lagging behind."""

from collections.abc import Callable
from typing import Any

import pytest
import ray
import torch

from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.distributed.pp_transport import PPTransport
from vllm.distributed.pp_transport.nccl import NcclTransport

from ..utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)

MAX_ROWS, WIDTH, STEPS = 4096, 1024, 40
KEYS = ("hidden_states", "residual")


def _rows(step: int) -> int:
    return [MAX_ROWS, 1, 1337, 512][step % 4]


def _run_chain(
    make: Callable[[Any, dict | None, dict | None], PPTransport],
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(1, pp_size, rank, distributed_init_port)
    pp_group = get_pp_group()
    first, last = pp_group.is_first_rank, pp_group.is_last_rank

    def buffers() -> dict[str, torch.Tensor]:
        return {
            k: torch.zeros(MAX_ROWS, WIDTH, dtype=torch.bfloat16, device="cuda")
            for k in KEYS
        }

    persistent = None if first else buffers()
    profile_send = None if last else buffers()
    transport = make(pp_group, persistent, profile_send)

    lag = torch.randn(4096, 4096, device="cuda")
    for step in range(STEPS):
        transport.before_forward()
        # Senders often lag (a receiver reading too early sees stale values),
        # the last stage sometimes (a sender overwriting too early corrupts
        # values it has not read yet).
        if (step % 2 == 0) if not last else (step % 7 == 3):
            for _ in range(20):
                lag @ lag
        if first:
            h = torch.full(
                (_rows(step), WIDTH), step % 100, dtype=torch.bfloat16, device="cuda"
            )
            transport.send({"hidden_states": h, "residual": -h}, pp_group)
            continue
        received = transport.recv(pp_group).tensors
        if not last:
            transport.send({k: v + 1 for k, v in received.items()}, pp_group)
            continue
        want = step % 100 + pp_size - 2
        assert received["hidden_states"].shape == (_rows(step), WIDTH)
        assert (received["hidden_states"] == want).all(), step
        assert (received["residual"] == want - 2 * (step % 100)).all(), step
    torch.accelerator.synchronize()
    # A stage's last metadata send completes only once the next stage
    # receives it, so no stage may exit before the chain is done.
    torch.distributed.barrier(group=pp_group.cpu_group)


@ray.remote(num_gpus=1, max_calls=1)
def nccl_chain_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _run_chain(
        lambda pp_group, persistent, profile_send: NcclTransport(
            get_tp_group(), profile_send
        ),
        pp_size,
        rank,
        distributed_init_port,
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("test_target", [nccl_chain_worker])
def test_pp_transport_pp2(monkeypatch: pytest.MonkeyPatch, test_target):
    multi_process_parallel(monkeypatch, 1, 2, test_target)


@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("test_target", [nccl_chain_worker])
def test_pp_transport_pp4(monkeypatch: pytest.MonkeyPatch, test_target):
    multi_process_parallel(monkeypatch, 1, 4, test_target)
