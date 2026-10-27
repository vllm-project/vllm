# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP send buffers carry every step's tensors intact through a PP chain while
senders overwrite their outputs every step (as CUDA graph outputs are reused),
middle stages relay views of their receive buffer (as MRV2 relays aux hidden
states), sizes vary, and stages take turns lagging behind."""

import pytest
import ray
import torch

from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.v1.worker.pp_send_buffer import PPSendBuffer, packed_nbytes

from ..utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)

MAX_ROWS, WIDTH, STEPS = 4096, 1024, 40


def _rows(step: int) -> int:
    return [MAX_ROWS, 1, 1337, 512][step % 4]


def _fill(step: int, rank: int, key: int) -> int:
    return (step * 7 + rank * 3 + key) % 120


@ray.remote(num_gpus=1, max_calls=1)
def send_buffer_chain_worker(
    monkeypatch, tp_size, pp_size, rank, distributed_init_port
):
    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(1, pp_size, rank, distributed_init_port)
    pp_group = get_pp_group()
    first, last = pp_group.is_first_rank, pp_group.is_last_rank

    def zeros() -> torch.Tensor:
        return torch.zeros(MAX_ROWS, WIDTH, dtype=torch.bfloat16, device="cuda")

    # Every step's outputs land in the same storage.
    outputs = {"hidden_states": zeros(), "residual": zeros()}
    # Persistent receive buffer, relayed onward as views.
    relayed = zeros()
    payload = dict(outputs) if first else dict(outputs, relay=relayed)
    send_buffer = None if last else PPSendBuffer(packed_nbytes(payload), "cuda")

    lag = torch.randn(4096, 4096, device="cuda")
    for step in range(STEPS):
        n = _rows(step)
        if not first:
            received, handles, postprocess = pp_group.irecv_tensor_dict(
                all_gather_group=get_tp_group()
            )
            for handle in handles:
                handle.wait()
            for fn in postprocess:
                fn()
            expected = {"hidden_states": 0, "residual": 1}
            for key, offset in expected.items():
                want = _fill(step, rank - 1, offset)
                assert torch.equal(
                    received[key], torch.full_like(received[key], want)
                ), (step, key)
            if rank > 1:
                want = _fill(step, rank - 2, 0)
                assert torch.equal(
                    received["relay"], torch.full_like(received["relay"], want)
                ), (step, "relay")
            relayed[:n].copy_(received["hidden_states"])
        if last:
            continue
        # Stages lag in turns: a slow sender lets receivers run ahead, a slow
        # receiver makes the next snapshot wait for its send.
        if (step + rank) % 3 == 0:
            for _ in range(20):
                lag @ lag
        for offset, key in enumerate(outputs):
            outputs[key][:n].fill_(_fill(step, rank, offset))
        tensors = {key: value[:n] for key, value in outputs.items()}
        if not first:
            tensors["relay"] = relayed[:n]
        assert send_buffer is not None
        send_buffer.send(tensors, pp_group, get_tp_group())
    torch.accelerator.synchronize()
    # A stage's last metadata send completes only once the next stage
    # receives it, so no stage may exit before the chain is done.
    torch.distributed.barrier(group=pp_group.cpu_group)


@multi_gpu_test(num_gpus=2)
def test_pp_send_buffer_pp2(monkeypatch: pytest.MonkeyPatch):
    multi_process_parallel(monkeypatch, 1, 2, send_buffer_chain_worker)


@multi_gpu_test(num_gpus=4)
def test_pp_send_buffer_pp4(monkeypatch: pytest.MonkeyPatch):
    multi_process_parallel(monkeypatch, 1, 4, send_buffer_chain_worker)
