import torch

from vllm.utils import print_warning_once

from .interface import Platform, PlatformEnum


class NeuronPlatform(Platform):
    _enum = PlatformEnum.NEURON

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        return "neuron"

    @staticmethod
    def inference_mode():
        return torch.inference_mode()

    @staticmethod
    def is_pin_memory_available() -> bool:
        print_warning_once("Pin memory is not supported on Neuron.")
        return False
