from typing import Tuple

import torch

from .interface import Platform, PlatformEnum


class HpuPlatform(Platform):
    _enum = PlatformEnum.HPU

    @staticmethod
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        raise RuntimeError("HPU does not have device capability.")

    @staticmethod
    def inference_mode():
        return torch.no_grad()
