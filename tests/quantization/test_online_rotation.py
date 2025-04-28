# SPDX-License-Identifier: Apache-2.0
"""Test the Hadamard Transform used in online rotations for outlier reduction.

Run `pytest tests/quantization/test_online_rotation.py`.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import ray
import torch

import vllm.model_executor.layers.quantization.quark.schemes as schemes

from ..utils import init_test_distributed_environment, multi_process_parallel

hadamard_transform = schemes.hadamard_transform
hadamard_sizes = hadamard_transform.hadamard_sizes.hadamard_sizes
hadamard_transform_registry = hadamard_transform.hadamard_transform_registry


@ray.remote(num_gpus=1, max_calls=1)
def test_quarot_r4(
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

    hadamard_transform = hadamard_transform_registry['quarot_r4']
    from types import SimpleNamespace
    input_size = 14336
    dummylayer = SimpleNamespace(
        weight=torch.Tensor([0]).to(device).bfloat16(),
        input_size_per_partition=input_size // tp_size,
        input_size=input_size)
    rotation_function = hadamard_transform(layer=dummylayer)
    """TP: Perform rotation function on activation shard - bfloat16"""
    activation = torch.ones(2, input_size, device=device).bfloat16()
    isye = input_size // tp_size
    activation_shard = activation[:, rank * (isye):(rank + 1) * (isye)]
    activation_shard = activation_shard.contiguous().to(device)
    rotated_activation_shard = rotation_function(activation_shard)
    """Uniproc: Perform rotation function on entire activation - bfloat16"""
    activation = torch.ones(2, input_size, device=device).bfloat16()
    og_shape = activation.shape
    X = activation.view(-1, rotation_function.chunk_size)
    X = rotation_function.FHT(X, rotation_function.scale)
    print(X)
    X = X.view(-1, rotation_function.hadamard_k.input_size,
               rotation_function.chunk_size)
    X = hadamard_sizes[rotation_function.k].to(device).bfloat16() @ X
    X = X.view(og_shape)
    X_shard = X[:, rank * (input_size // tp_size):(rank + 1) *
                (input_size // tp_size)]
    """Validate the Tensor Parallel output against the Uniprocessor output"""
    #rotated_activation_shard=rotated_activation_shard[0:64,:]
    #X_shard=X_shard[0:64,:]
    print("Comparing:")
    if rank == 0:
        print("Tensor Comparison")
        print(rotated_activation_shard)
        print(rotated_activation_shard.shape)
        print(X_shard)
        print(X_shard.shape)
    torch.testing.assert_close(rotated_activation_shard, X_shard)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("test_target", [test_quarot_r4])
def test_hadamard_transform_tensor_parallel(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    test_target: Callable[..., Any],
):
    multi_process_parallel(monkeypatch, tp_size, 1, test_target)