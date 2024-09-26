import torch

from vllm.utils import print_warning_once

from .interface import Platform, PlatformEnum


class OpenVinoPlatform(Platform):
    _enum = PlatformEnum.OPENVINO

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        return "openvino"

    @staticmethod
    def inference_mode():
        return torch.inference_mode()

    @staticmethod
    def is_pin_memory_available() -> bool:
        print_warning_once("Pin memory is not supported on OpenViNO.")
        return False
