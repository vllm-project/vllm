from typing import Tuple

import torch

from .interface import Platform, PlatformEnum


class XPUPlatform(Platform):
    _enum = PlatformEnum.XPU

    @staticmethod
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        return torch.xpu.get_device_capability(device_id)

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        return torch.xpu.get_device_name(device_id)
