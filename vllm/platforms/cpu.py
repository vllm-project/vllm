import torch

from .interface import Platform, PlatformEnum


class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        return "cpu"

    @staticmethod
    def inference_mode():
        return torch.no_grad()
