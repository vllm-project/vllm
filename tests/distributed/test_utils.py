import pytest
import ray
import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.distributed.utils import stateless_init_process_group
from vllm.utils import (cuda_device_count_stateless,
                        update_environment_variables)

from ..utils import multi_gpu_test


@ray.remote
class _CUDADeviceCountStatelessTestActor:

    def get_count(self):
        return cuda_device_count_stateless()

    def set_cuda_visible_devices(self, cuda_visible_devices: str):
        update_environment_variables(
            {"CUDA_VISIBLE_DEVICES": cuda_visible_devices})

    def get_cuda_visible_devices(self):
        return envs.CUDA_VISIBLE_DEVICES


def test_cuda_device_count_stateless():
    """Test that cuda_device_count_stateless changes return value if
    CUDA_VISIBLE_DEVICES is changed."""
    actor = _CUDADeviceCountStatelessTestActor.options(  # type: ignore
        num_gpus=2).remote()
    assert len(
        sorted(ray.get(
            actor.get_cuda_visible_devices.remote()).split(","))) == 2
    assert ray.get(actor.get_count.remote()) == 2
    ray.get(actor.set_cuda_visible_devices.remote("0"))
    assert ray.get(actor.get_count.remote()) == 1
    ray.get(actor.set_cuda_visible_devices.remote(""))
    assert ray.get(actor.get_count.remote()) == 0


def cpu_worker(rank, WORLD_SIZE):
    pg1 = stateless_init_process_group(init_method="tcp://127.0.0.1:29500",
                                       rank=rank,
                                       world_size=WORLD_SIZE,
                                       backend="gloo")
    if rank <= 2:
        pg2 = stateless_init_process_group(init_method="tcp://127.0.0.1:29501",
                                           rank=rank,
                                           world_size=3,
                                           backend="gloo")
    data = torch.tensor([rank])
    dist.all_reduce(data, op=dist.ReduceOp.SUM, group=pg1)
    if rank <= 2:
        dist.all_reduce(data, op=dist.ReduceOp.SUM, group=pg2)
    item = data[0].item()
    print(f"rank: {rank}, item: {item}")
    if rank == 3:
        assert item == 6
    else:
        assert item == 18


def gpu_worker(rank, WORLD_SIZE):
    pg1 = stateless_init_process_group(init_method="tcp://127.0.0.1:29502",
                                       rank=rank,
                                       world_size=WORLD_SIZE,
                                       backend="nccl")
    if rank <= 2:
        pg2 = stateless_init_process_group(init_method="tcp://127.0.0.1:29503",
                                           rank=rank,
                                           world_size=3,
                                           backend="nccl")
    torch.cuda.set_device(rank)
    data = torch.tensor([rank]).cuda()
    dist.all_reduce(data, op=dist.ReduceOp.SUM, group=pg1)
    if rank <= 2:
        dist.all_reduce(data, op=dist.ReduceOp.SUM, group=pg2)
    item = data[0].item()
    print(f"rank: {rank}, item: {item}")
    if rank == 3:
        assert item == 6
    else:
        assert item == 18


@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("worker", [cpu_worker, gpu_worker])
def test_stateless_init_process_group(worker):
    WORLD_SIZE = 4
    from multiprocessing import get_context
    ctx = get_context("fork")
    processes = []
    for i in range(WORLD_SIZE):
        rank = i
        processes.append(ctx.Process(target=worker, args=(rank, WORLD_SIZE)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    for p in processes:
        assert not p.exitcode
    print("All processes finished.")
