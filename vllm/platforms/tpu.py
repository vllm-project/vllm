import torch

from .interface import Platform, PlatformEnum


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()
