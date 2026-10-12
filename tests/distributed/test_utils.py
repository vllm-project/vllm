# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import ray

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables


@ray.remote
class _CUDADeviceCountStatelessTestActor:
    def get_count(self):
        return current_platform.device_count()

    def set_cuda_visible_devices(self, cuda_visible_devices: str):
        update_environment_variables({"CUDA_VISIBLE_DEVICES": cuda_visible_devices})

    def get_cuda_visible_devices(self):
        return envs.CUDA_VISIBLE_DEVICES


def test_cuda_device_count_stateless():
    """Test that cuda_device_count_stateless changes return value if
    CUDA_VISIBLE_DEVICES is changed."""
    if current_platform.is_rocm():
        pytest.skip("Skip for ROCm because Ray uses HIP_VISIBLE_DEVICES.")
    actor = _CUDADeviceCountStatelessTestActor.options(  # type: ignore
        num_gpus=2
    ).remote()
    assert len(sorted(ray.get(actor.get_cuda_visible_devices.remote()).split(","))) == 2
    assert ray.get(actor.get_count.remote()) == 2
    ray.get(actor.set_cuda_visible_devices.remote("0"))
    assert ray.get(actor.get_count.remote()) == 1
    ray.get(actor.set_cuda_visible_devices.remote(""))
    assert ray.get(actor.get_count.remote()) == 0
