import os

import pytest
import ray

from tests.nm_utils.utils_skip import should_skip_test_group
from vllm.utils import cuda_device_count_stateless

if should_skip_test_group(group_name="TEST_DISTRIBUTED"):
    pytest.skip("TEST_DISTRIBUTED=DISABLE, skipping distributed test group",
                allow_module_level=True)


@ray.remote
class _CUDADeviceCountStatelessTestActor():

    def get_count(self):
        return cuda_device_count_stateless()

    def set_cuda_visible_devices(self, cuda_visible_devices: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    def get_cuda_visible_devices(self):
        return os.environ["CUDA_VISIBLE_DEVICES"]


def test_cuda_device_count_stateless():
    """Test that cuda_device_count_stateless changes return value if
    CUDA_VISIBLE_DEVICES is changed."""

    actor = _CUDADeviceCountStatelessTestActor.options(num_gpus=2).remote()
    assert ray.get(actor.get_cuda_visible_devices.remote()) == "0,1"
    assert ray.get(actor.get_count.remote()) == 2
    ray.get(actor.set_cuda_visible_devices.remote("0"))
    assert ray.get(actor.get_count.remote()) == 1
    ray.get(actor.set_cuda_visible_devices.remote(""))
    assert ray.get(actor.get_count.remote()) == 0
