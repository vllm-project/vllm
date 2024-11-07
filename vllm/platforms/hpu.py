import torch

from .interface import Platform, PlatformEnum


class HpuPlatform(Platform):
    _enum = PlatformEnum.HPU

    @staticmethod
    def inference_mode():
        return torch.no_grad()
