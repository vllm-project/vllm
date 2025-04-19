# SPDX-License-Identifier: Apache-2.0
"""Test the communication operators.

Run `pytest tests/distributed/test_comm_ops.py`.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import ray
import torch

from vllm.distributed import (broadcast_tensor_dict, get_pp_group,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)

from ..utils import init_test_distributed_environment, multi_process_parallel


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(tp_size, pp_size, rank,
                                      distributed_init_port)
    num_elements = 8
    all_tensors = [
        torch.arange(num_elements, dtype=torch.float32, device="cuda") *
        (r + 1) for r in range(tp_size)
    ]
    expected = torch.sum(torch.stack(all_tensors, dim=0), dim=0)
    t = all_tensors[rank % tp_size]
    t = tensor_model_parallel_all_reduce(t)
    torch.testing.assert_close(t, expected)


@ray.remote(num_gpus=1, max_calls=1)
def reduce_scatter_test_worker(monkeypatch: pytest.MonkeyPatch, tp_size: int,
                               pp_size: int, rank: int,
                               distributed_init_port: str):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(tp_size, pp_size, rank,
                                      distributed_init_port)

    num_elements = 8
    all_tensors = [
        torch.arange(num_elements, dtype=torch.float32, device="cuda") *
        (r + 1) for r in range(tp_size)
    ]

    index = rank % tp_size
    partition_size = num_elements // tp_size
    all_reduce = torch.sum(torch.stack(all_tensors, dim=0), dim=0)
    expected = all_reduce[index * partition_size:(index + 1) * partition_size]
    t = all_tensors[index]
    t = tensor_model_parallel_reduce_scatter(t, 0)
    torch.testing.assert_close(t, expected)


@ray.remote(num_gpus=1, max_calls=1)
def all_gather_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(tp_size, pp_size, rank,
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
            for r in range(tp_size)
        ]
        expected = torch.cat(all_tensors, dim=all_gather_dimension)
        t = all_tensors[rank % tp_size]
        t = tensor_model_parallel_all_gather(t, all_gather_dimension)
        torch.testing.assert_close(t, expected)


@ray.remote(num_gpus=1, max_calls=1)
def broadcast_tensor_dict_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(tp_size, pp_size, rank,
                                      distributed_init_port)
    test_dict = {
        # device tensor
        "a": torch.arange(8, dtype=torch.float32, device="cuda"),
        # CPU tensor
        "b": torch.arange(16, dtype=torch.int8, device="cpu"),
        "c": "test",
        "d": [1, 2, 3],
        "e": {
            "a": 1,
            "b": 2
        },
        # empty tensor
        "f": torch.tensor([], dtype=torch.float32, device="cuda"),
    }

    if (rank % tp_size) == 0:
        broadcast_tensor_dict(test_dict, src=0)
    else:
        recv_dict = broadcast_tensor_dict(src=0)
        assert len(recv_dict) == len(test_dict)
        torch.testing.assert_close(recv_dict["a"], test_dict["a"])
        torch.testing.assert_close(recv_dict["b"], test_dict["b"])
        assert recv_dict["c"] == test_dict["c"]
        assert recv_dict["d"] == test_dict["d"]
        assert recv_dict["e"] == test_dict["e"]
        torch.testing.assert_close(recv_dict["f"], test_dict["f"])


@ray.remote(num_gpus=1, max_calls=1)
def send_recv_tensor_dict_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(tp_size, pp_size, rank,
                                      distributed_init_port)

    test_dict = {
        # device tensor
        "a": torch.arange(8, dtype=torch.float32, device="cuda"),
        # CPU tensor
        "b": torch.arange(16, dtype=torch.int8, device="cpu"),
        "c": "test",
        "d": [1, 2, 3],
        "e": {
            "a": 1,
            "b": 2
        },
        # empty tensor
        "f": torch.tensor([], dtype=torch.float32, device="cuda"),
    }

    if not get_pp_group().is_first_rank:
        recv_dict = get_pp_group().recv_tensor_dict()

    if not get_pp_group().is_last_rank:
        get_pp_group().send_tensor_dict(test_dict)

    if not get_pp_group().is_first_rank:
        assert len(recv_dict) == len(test_dict)
        torch.testing.assert_close(recv_dict["a"], test_dict["a"])
        torch.testing.assert_close(recv_dict["b"], test_dict["b"])
        assert recv_dict["c"] == test_dict["c"]
        assert recv_dict["d"] == test_dict["d"]
        assert recv_dict["e"] == test_dict["e"]
        torch.testing.assert_close(recv_dict["f"], test_dict["f"])


@ray.remote(num_gpus=1, max_calls=1)
def send_recv_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(tp_size, pp_size, rank,
                                      distributed_init_port)

    size = 64
    test_tensor = torch.arange(64, dtype=torch.float32, device="cuda")

    if not get_pp_group().is_first_rank:
        recv_tensor = get_pp_group().recv(size, dtype=torch.float32)

    if not get_pp_group().is_last_rank:
        get_pp_group().send(test_tensor)

    if not get_pp_group().is_first_rank:
        torch.testing.assert_close(test_tensor, recv_tensor)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("test_target", [
    all_reduce_test_worker, all_gather_test_worker,
    broadcast_tensor_dict_test_worker
])
def test_multi_process_tensor_parallel(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    test_target: Callable[..., Any],
):
    multi_process_parallel(monkeypatch, tp_size, 1, test_target)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("pp_size", [2])
@pytest.mark.parametrize(
    "test_target", [send_recv_test_worker, send_recv_tensor_dict_test_worker])
def test_multi_process_pipeline_parallel(
    monkeypatch: pytest.MonkeyPatch,
    pp_size: int,
    test_target: Callable[..., Any],
):
    multi_process_parallel(monkeypatch, 1, pp_size, test_target)


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pp_size", [2])
@pytest.mark.parametrize("test_target", [
    send_recv_test_worker, send_recv_tensor_dict_test_worker,
    all_reduce_test_worker, all_gather_test_worker,
    broadcast_tensor_dict_test_worker
])
def test_multi_process_tensor_parallel_pipeline_parallel(
    tp_size: int,
    pp_size: int,
    test_target: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
):
    multi_process_parallel(monkeypatch, tp_size, pp_size, test_target)
