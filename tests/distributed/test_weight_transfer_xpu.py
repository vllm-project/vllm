# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XCCL weight transfer tests between Ray tasks on XPU."""

import pytest
import ray
import torch

from tests.distributed.test_weight_transfer import (
    TestDeferredProcessingContract as _TestDeferredProcessingContract,
)
from tests.distributed.test_weight_transfer import (
    TestModuleSource as _TestModuleSource,
)
from tests.distributed.test_weight_transfer import (
    TestTrainerClients as _TestTrainerClients,
)
from tests.distributed.test_weight_transfer import (
    TestTrainerEngineBase as _TestTrainerEngineBase,
)
from tests.distributed.test_weight_transfer import (
    TestTrainerFactory as _TestTrainerFactory,
)
from tests.distributed.test_weight_transfer import (
    TestWeightSourceGroupContract as _TestWeightSourceGroupContract,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


class TestXPUTrainerClients(_TestTrainerClients):
    """XPU-suite counterpart for platform-independent client tests."""


class TestXPUModuleSource(_TestModuleSource):
    """XPU-suite counterpart for platform-independent source tests."""


class TestXPUWeightSourceGroupContract(_TestWeightSourceGroupContract):
    """XPU-suite counterpart for platform-independent source-group tests."""


class TestXPUDeferredProcessingContract(_TestDeferredProcessingContract):
    """XPU-suite counterpart for platform-independent engine contracts."""


class TestXPUTrainerFactory(_TestTrainerFactory):
    """XPU-suite counterpart for platform-independent trainer factory tests."""


class TestXPUTrainerEngineBase(_TestTrainerEngineBase):
    """XPU-suite counterpart for platform-independent trainer base tests."""


def _init_ray_for_weight_transfer() -> None:
    if ray.is_initialized():
        return
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
            }
        },
    )


def _set_ray_assigned_xpu_device() -> torch.device:
    # Ray assigns a physical GPU ID, but ZE_AFFINITY_MASK makes that device
    # process-local index 0 for torch.xpu.
    assert ray.get_gpu_ids()
    device = torch.device("xpu:0")
    current_platform.set_device(device)
    return device


def _init_xccl_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
) -> None:
    torch.distributed.init_process_group(
        backend="xccl",
        init_method=f"tcp://{master_address}:{master_port}",
        rank=rank,
        world_size=world_size,
    )


@ray.remote(num_gpus=1)
def trainer_broadcast_tensor(
    master_address: str,
    master_port: int,
    world_size: int,
    tensor_shape: list[int],
    tensor_dtype: str,
) -> bool:
    """Trainer task that broadcasts a tensor via XCCL."""
    device = _set_ray_assigned_xpu_device()
    _init_xccl_process_group(master_address, master_port, rank=0, world_size=world_size)

    try:
        dtype = getattr(torch, tensor_dtype)
        tensor_to_send = torch.ones(tensor_shape, dtype=dtype, device=device)
        torch.distributed.broadcast(tensor_to_send, src=0)
        torch.accelerator.synchronize()
    finally:
        torch.distributed.destroy_process_group()

    return True


@ray.remote(num_gpus=1)
def inference_receive_tensor(
    master_address: str,
    master_port: int,
    world_size: int,
    tensor_shape: list[int],
    tensor_dtype: str,
) -> dict:
    """Inference task that receives a tensor via XCCL."""
    device = _set_ray_assigned_xpu_device()
    _init_xccl_process_group(master_address, master_port, rank=1, world_size=world_size)

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received = []

        def load_weights(self, weights):
            for name, tensor in weights:
                self.received.append((name, tensor.clone()))

    recorder = Recorder()
    try:
        dtype = getattr(torch, tensor_dtype)
        received_tensor = torch.empty(tensor_shape, dtype=dtype, device=device)
        torch.distributed.broadcast(received_tensor, src=0)
        recorder.load_weights([("test.weight", received_tensor)])
        torch.accelerator.synchronize()

        success = False
        received_shape = None
        received_sum = None
        if len(recorder.received) == 1:
            _, tensor = recorder.received[0]
            received_shape = list(tensor.shape)
            received_sum = tensor.sum().item()
            if received_shape == tensor_shape:
                expected_sum = torch.tensor(tensor_shape).prod().item()
                if abs(received_sum - expected_sum) < 0.01:
                    success = True
    finally:
        torch.distributed.destroy_process_group()

    return {
        "success": success,
        "received_shape": received_shape,
        "received_sum": received_sum,
    }


@pytest.mark.skipif(not current_platform.is_xpu(), reason="requires XPU")
@pytest.mark.skipif(
    not torch.distributed.is_xccl_available(), reason="requires XCCL support"
)
@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 XPUs to run XCCL weight transfer test.",
)
def test_xccl_weight_transfer_between_processes():
    """Test XCCL weight transfer from trainer to inference process using Ray."""
    _init_ray_for_weight_transfer()

    master_address = "127.0.0.1"
    master_port = get_open_port()
    world_size = 2

    tensor_shape = [100, 100]
    tensor_dtype = "float32"

    inference_future = inference_receive_tensor.remote(
        master_address, master_port, world_size, tensor_shape, tensor_dtype
    )
    trainer_future = trainer_broadcast_tensor.remote(
        master_address, master_port, world_size, tensor_shape, tensor_dtype
    )

    trainer_result, result = ray.get([trainer_future, inference_future])

    assert trainer_result, "Trainer should complete successfully"
    assert result["success"], (
        f"Weight transfer failed. "
        f"Received shape: {result['received_shape']}, "
        f"Received sum: {result['received_sum']}"
    )
