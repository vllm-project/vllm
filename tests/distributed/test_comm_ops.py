"""Test the communication operators.

Run `pytest tests/distributed/test_comm_ops.py --forked`.
"""
import pytest
import torch
import ray

from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
    broadcast_tensor_dict,
)
from vllm.test_utils import (init_test_distributed_environment,
                             multi_process_tensor_parallel)


@ray.remote(num_gpus=1, max_calls=1)
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


@ray.remote(num_gpus=1, max_calls=1)
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


@ray.remote(num_gpus=1, max_calls=1)
def broadcast_tensor_dict_test_worker(tensor_parallel_size: int, rank: int,
                                      distributed_init_port: str):
    init_test_distributed_environment(1, tensor_parallel_size, rank,
                                      distributed_init_port)
    test_dict = {
        "a": torch.arange(8, dtype=torch.float32, device="cuda"),
        "b": torch.arange(16, dtype=torch.int8, device="cuda"),
        "c": "test",
        "d": [1, 2, 3],
        "e": {
            "a": 1,
            "b": 2
        },
    }

    if rank == 0:
        broadcast_tensor_dict(test_dict, src=0)
    else:
        recv_dict = broadcast_tensor_dict(src=0)
        assert len(recv_dict) == len(test_dict)
        assert torch.allclose(recv_dict["a"], test_dict["a"])
        assert torch.allclose(recv_dict["b"], test_dict["b"])
        assert recv_dict["c"] == test_dict["c"]
        assert recv_dict["d"] == test_dict["d"]
        assert recv_dict["e"] == test_dict["e"]


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("test_target", [
    all_reduce_test_worker, all_gather_test_worker,
    broadcast_tensor_dict_test_worker
])
def test_multi_process_tensor_parallel(tensor_parallel_size, test_target):
    multi_process_tensor_parallel(tensor_parallel_size, test_target)
