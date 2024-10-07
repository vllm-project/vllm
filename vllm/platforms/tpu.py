import os

import torch

from vllm.plugins import set_torch_compile_backend

from .interface import Platform, PlatformEnum

if "VLLM_TORCH_COMPILE_LEVEL" not in os.environ:
    os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "2"

set_torch_compile_backend("openxla")


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
