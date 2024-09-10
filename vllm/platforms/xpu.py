from typing import Tuple

import torch

from .interface import Platform, PlatformEnum


class XpuPlatform(Platform):
    _enum = PlatformEnum.XPU

    @staticmethod
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        return (100, 9)

    @staticmethod
    def inference_mode():
        return torch.no_grad()
