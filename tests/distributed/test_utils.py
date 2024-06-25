import os

import pytest
import ray
import torch
from vllm.utils import is_hip
from vllm.utils import cuda_device_count_stateless


@ray.remote
class _CUDADeviceCountStatelessTestActor:

    def get_count(self):
        return cuda_device_count_stateless()

    def set_cuda_visible_devices(self, cuda_visible_devices: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    def get_cuda_visible_devices(self):
        return os.environ["CUDA_VISIBLE_DEVICES"]


def test_cuda_device_count_stateless():
    """Test that cuda_device_count_stateless changes return value if
    CUDA_VISIBLE_DEVICES is changed."""

    # skip this test if rocm does not support this yet
    if is_hip() and (not hasattr(torch.cuda, "_device_count_amdsmi")
                     or not torch.cuda._HAS_PYNVML):
        pytest.skip("no support to _device_count_amdsmi yet")

    actor = _CUDADeviceCountStatelessTestActor.options(  # type: ignore
        num_gpus=2).remote()
    assert sorted(ray.get(
        actor.get_cuda_visible_devices.remote()).split(",")) == ["0", "1"]
    assert ray.get(actor.get_count.remote()) == 2
    ray.get(actor.set_cuda_visible_devices.remote("0"))
    assert ray.get(actor.get_count.remote()) == 1
    ray.get(actor.set_cuda_visible_devices.remote(""))
    assert ray.get(actor.get_count.remote()) == 0
