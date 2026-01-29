# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import socket

import pytest
import ray
import torch

import vllm.envs as envs
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import cuda_device_count_stateless

from ..utils import multi_gpu_test


@ray.remote
class _CUDADeviceCountStatelessTestActor:
    def get_count(self):
        return cuda_device_count_stateless()

    def set_cuda_visible_devices(self, cuda_visible_devices: str):
        update_environment_variables({"CUDA_VISIBLE_DEVICES": cuda_visible_devices})

    def get_cuda_visible_devices(self):
        return envs.CUDA_VISIBLE_DEVICES


def test_cuda_device_count_stateless():
    """Test that cuda_device_count_stateless changes return value if
    CUDA_VISIBLE_DEVICES is changed."""
    if current_platform.is_rocm():
        pytest.skip(
            "Skip for ROCm because Ray uses HIP_VISIBLE_DEVICES."
        )
    actor = _CUDADeviceCountStatelessTestActor.options(  # type: ignore
        num_gpus=2
    ).remote()
    assert len(sorted(ray.get(actor.get_cuda_visible_devices.remote()).split(","))) == 2
    assert ray.get(actor.get_count.remote()) == 2
    ray.get(actor.set_cuda_visible_devices.remote("0"))
    assert ray.get(actor.get_count.remote()) == 1
    ray.get(actor.set_cuda_visible_devices.remote(""))
    assert ray.get(actor.get_count.remote()) == 0


def cpu_worker(rank, WORLD_SIZE, port1, port2):
    pg1 = StatelessProcessGroup.create(
        host="127.0.0.1", port=port1, rank=rank, world_size=WORLD_SIZE
    )
    if rank <= 2:
        pg2 = StatelessProcessGroup.create(
            host="127.0.0.1", port=port2, rank=rank, world_size=3
        )
    data = torch.tensor([rank])
    data = pg1.broadcast_obj(data, src=2)
    assert data.item() == 2
    if rank <= 2:
        data = torch.tensor([rank + 1])
        data = pg2.broadcast_obj(data, src=2)
        assert data.item() == 3
        pg2.barrier()
    pg1.barrier()


def gpu_worker(rank, WORLD_SIZE, port1, port2):
    torch.cuda.set_device(rank)
    pg1 = StatelessProcessGroup.create(
        host="127.0.0.1", port=port1, rank=rank, world_size=WORLD_SIZE
    )
    pynccl1 = PyNcclCommunicator(pg1, device=rank)
    if rank <= 2:
        pg2 = StatelessProcessGroup.create(
            host="127.0.0.1", port=port2, rank=rank, world_size=3
        )
        pynccl2 = PyNcclCommunicator(pg2, device=rank)
    data = torch.tensor([rank]).cuda()
    pynccl1.all_reduce(data)
    pg1.barrier()
    torch.cuda.synchronize()
    if rank <= 2:
        pynccl2.all_reduce(data)
        pg2.barrier()
        torch.cuda.synchronize()
    item = data[0].item()
    print(f"rank: {rank}, item: {item}")
    if rank == 3:
        assert item == 6
    else:
        assert item == 18


def broadcast_worker(rank, WORLD_SIZE, port1, port2):
    pg1 = StatelessProcessGroup.create(
        host="127.0.0.1", port=port1, rank=rank, world_size=WORLD_SIZE
    )
    if rank == 2:
        pg1.broadcast_obj("secret", src=2)
    else:
        obj = pg1.broadcast_obj(None, src=2)
        assert obj == "secret"
    pg1.barrier()


def allgather_worker(rank, WORLD_SIZE, port1, port2):
    pg1 = StatelessProcessGroup.create(
        host="127.0.0.1", port=port1, rank=rank, world_size=WORLD_SIZE
    )
    data = pg1.all_gather_obj(rank)
    assert data == list(range(WORLD_SIZE))
    pg1.barrier()


@pytest.mark.skip(reason="This test is flaky and prone to hang.")
@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize(
    "worker", [cpu_worker, gpu_worker, broadcast_worker, allgather_worker]
)
def test_stateless_process_group(worker):
    port1 = get_open_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", port1))
        port2 = get_open_port()
    WORLD_SIZE = 4
    from multiprocessing import get_context

    ctx = get_context("fork")
    processes = []
    for i in range(WORLD_SIZE):
        rank = i
        processes.append(
            ctx.Process(target=worker, args=(rank, WORLD_SIZE, port1, port2))
        )
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    for p in processes:
        assert not p.exitcode
    print("All processes finished.")
