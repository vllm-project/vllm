from typing import Tuple

import torch

from .interface import DeviceCapability, Platform, PlatformEnum


class XpuPlatform(Platform):
    _enum = PlatformEnum.XPU

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = (100, 9)
        return DeviceCapability(major=major, minor=minor)

    @staticmethod
    def inference_mode():
        return torch.no_grad()
