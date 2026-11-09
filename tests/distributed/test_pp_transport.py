# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The copy-engine PP transport carries every step's tensors intact through a
PP chain, with varying sizes and stages that take turns lagging behind, and
every rank makes the same transport choice when one rank cannot set it up."""

from types import SimpleNamespace

import pytest
import ray
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.pp_transport import create_pp_transport
from vllm.distributed.pp_transport.copy_engine import (
    CopyEngineTransport,
    CopyEngineUnavailable,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from ..utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)

MAX_ROWS, WIDTH, STEPS = 4096, 1024, 40
KEYS = ("hidden_states", "residual")


def _rows(step: int) -> int:
    return [MAX_ROWS, 1, 1337, 512][step % 4]


@ray.remote(num_gpus=1, max_calls=1)
def chain_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(1, pp_size, rank, distributed_init_port)
    pp_group = get_pp_group()
    first, last = pp_group.is_first_rank, pp_group.is_last_rank

    persistent = None
    if not first:
        persistent = {
            k: torch.zeros(MAX_ROWS, WIDTH, dtype=torch.bfloat16, device="cuda")
            for k in KEYS
        }
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, prefill_context_parallel_size=1
        ),
    )
    transport = CopyEngineTransport(config, pp_group, persistent)

    lag = torch.randn(4096, 4096, device="cuda")
    for step in range(STEPS):
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
def one_rank_fails_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(1, pp_size, rank, distributed_init_port)
    # Imported here: Ray cannot pickle the module as a global of this function.
    from vllm import _custom_ops as ops

    pp_group = get_pp_group()
    if pp_group.rank_in_group == 1:

        def fail(handle):
            raise RuntimeError("injected open failure")

        monkeypatch.setattr(ops, "open_mem_handle", fail)
    persistent = None
    if not pp_group.is_first_rank:
        persistent = IntermediateTensors(
            {
                k: torch.zeros(MAX_ROWS, WIDTH, dtype=torch.bfloat16, device="cuda")
                for k in KEYS
            }
        )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, prefill_context_parallel_size=1
        ),
    )
    torch.accelerator.synchronize()
    free_before = torch.accelerator.get_memory_info()[0]

    monkeypatch.setenv("VLLM_PP_TRANSPORT", "copy_engine")
    with pytest.raises(CopyEngineUnavailable, match="injected open failure"):
        create_pp_transport(config, persistent)
    monkeypatch.setenv("VLLM_PP_TRANSPORT", "auto")
    assert create_pp_transport(config, persistent) is None

    # Every rank's buffers and peer mappings were released.
    torch.distributed.barrier(group=pp_group.cpu_group)
    torch.accelerator.synchronize()
    assert torch.accelerator.get_memory_info()[0] >= free_before - (2 << 20)
    torch.distributed.barrier(group=pp_group.cpu_group)


def _run(monkeypatch: pytest.MonkeyPatch, pp_size: int, worker=chain_worker) -> None:
    # NVML, not CUDA: the test process must not initialize CUDA.
    if not (
        current_platform.is_cuda()
        and current_platform.is_fully_connected(list(range(pp_size)))
    ):
        pytest.skip("copy engines need NVLink-connected CUDA GPUs")
    multi_process_parallel(monkeypatch, 1, pp_size, worker)


@multi_gpu_test(num_gpus=2)
def test_copy_engine_pp2(monkeypatch: pytest.MonkeyPatch):
    _run(monkeypatch, 2)


@multi_gpu_test(num_gpus=4)
def test_copy_engine_pp4(monkeypatch: pytest.MonkeyPatch):
    _run(monkeypatch, 4)


@multi_gpu_test(num_gpus=4)
def test_one_rank_failing_setup_gives_the_same_choice_everywhere(
    monkeypatch: pytest.MonkeyPatch,
):
    """Rank 1 (a middle stage) cannot map its peers' buffers: under
    "copy_engine" every rank raises, under "auto" every rank falls back to
    NCCL, and nothing stays allocated."""
    _run(monkeypatch, 4, one_rank_fails_worker)
