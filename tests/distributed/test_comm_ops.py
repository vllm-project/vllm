"""Test the communication operators.

Run `pytest tests/distributed/test_comm_ops.py --forked`.
"""
from multiprocessing import Process

import pytest
import torch

from vllm.config import ParallelConfig
from vllm.engine.ray_utils import get_open_port
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
)
from vllm.worker.worker import _init_distributed_environment


def init_test_distributed_environment(pipeline_parallel_size: int,
                                      tensor_parallel_size: int, rank: int,
                                      distributed_init_port: str):
    parallel_config = ParallelConfig(pipeline_parallel_size,
                                     tensor_parallel_size,
                                     worker_use_ray=True)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    torch.cuda.set_device(rank)
    _init_distributed_environment(parallel_config, rank,
                                  distributed_init_method)


def all_reduce_test_worker(tensor_parallel_size: int, rank: int,
                           distributed_init_port: str):
    init_test_distributed_environment(1, tensor_parallel_size, rank,
                                      distributed_init_port)
    num_elements = 8
    all_tensors = [
        torch.arange(num_elements, dtype=torch.float32, device="cuda") *
        (r + 1) for r in range(tensor_parallel_size)
    ]
    expected = torch.sum(torch.stack(all_tensors, dim=0), dim=0)
    t = all_tensors[rank]
    t = tensor_model_parallel_all_reduce(t)
    assert torch.allclose(t, expected)


def all_gather_test_worker(tensor_parallel_size: int, rank: int,
                           distributed_init_port: str):
    init_test_distributed_environment(1, tensor_parallel_size, rank,
                                      distributed_init_port)
    num_dimensions = 3
    tensor_size = list(range(2, num_dimensions + 2))
    total_size = 1
    for s in tensor_size:
        total_size *= s
    for all_gather_dimension in range(num_dimensions):
        all_tensors = [
            torch.arange(total_size, dtype=torch.float32,
                         device="cuda").reshape(tensor_size) * (r + 1)
            for r in range(tensor_parallel_size)
        ]
        expected = torch.cat(all_tensors, dim=all_gather_dimension)
        t = all_tensors[rank]
        t = tensor_model_parallel_all_gather(t, all_gather_dimension)
        assert torch.allclose(t, expected)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("test_target",
                         [all_reduce_test_worker, all_gather_test_worker])
def test_multi_process_tensor_parallel(tensor_parallel_size, test_target):
    distributed_init_port = get_open_port()
    processes = []
    for rank in range(tensor_parallel_size):
        p = Process(target=test_target,
                    args=(tensor_parallel_size, rank, distributed_init_port))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    assert all(p.exitcode == 0 for p in processes)
