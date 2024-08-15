import torch

from .interface import Platform, PlatformEnum


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU

    @staticmethod
    def inference_mode():
        return torch.no_grad()
