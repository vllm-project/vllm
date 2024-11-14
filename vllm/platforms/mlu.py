from functools import lru_cache

import torch

from .interface import DeviceCapability, Platform, PlatformEnum


class MluPlatform(Platform):
    _enum = PlatformEnum.MLU

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.mlu.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.mlu.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.mlu.get_device_properties(device_id)
        return device_props.total_memory