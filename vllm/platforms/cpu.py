import psutil
import torch

from vllm.utils import print_warning_once

from .interface import Platform, PlatformEnum


class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "cpu"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return psutil.virtual_memory().total

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @staticmethod
    def is_pin_memory_available() -> bool:
        print_warning_once("Pin memory is not supported on CPU.")
        return False
