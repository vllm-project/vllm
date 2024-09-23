import torch

from .interface import Platform, PlatformEnum


class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "cpu"

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()
