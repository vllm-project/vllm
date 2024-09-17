import torch

from .interface import Platform, PlatformEnum


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()
