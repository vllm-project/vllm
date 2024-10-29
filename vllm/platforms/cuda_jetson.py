"""For use on Jetson devices which support CUDA but not NVML.
`is_full_nvlink` is not implemented, as there are currently no Jetson
devices with NVLink.
Code inside this file can safely assume cuda platform.
Due to lack of NVML, we must initialize cuda context via torch
to fetch device information.
"""

from functools import lru_cache

import torch

from .interface import DeviceCapability, Platform, PlatformEnum


class CudaJetsonPlatform(Platform):
    _enum = PlatformEnum.CUDA_JETSON

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory
