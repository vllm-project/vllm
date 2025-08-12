import torch

from .interface import Platform, PlatformEnum, _Backend


class HpuPlatform(Platform):
    _enum = PlatformEnum.HPU
    device_type: str = "hpu"

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend) -> _Backend:
        return _Backend.HPU_ATTN

    @staticmethod
    def inference_mode():
        return torch.no_grad()
