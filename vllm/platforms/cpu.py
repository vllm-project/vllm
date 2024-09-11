import torch

from .interface import Platform, PlatformEnum


class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU

    @staticmethod
    def inference_mode():
        return torch.no_grad()
