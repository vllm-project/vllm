from typing import Tuple
import os

import torch

from .interface import Platform, PlatformEnum


def device_id_to_physical_device_id(device_id: int) -> int:
    if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
        if device_ids == [""]:
            raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES is set to empty string,"
                               " which means Ascend NPU support is disabled.")
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


class AscendPlatform(Platform):
    _enum = PlatformEnum.ASCEND

    @staticmethod
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        raise RuntimeError("Ascend NPU does not have device capability.")

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        return torch.npu.get_device_name(physical_device_id)

    @staticmethod
    def inference_mode():
        return torch.inference_mode()

    @staticmethod
    def set_device(device: torch.device) -> torch.device:
        torch.npu.set_device(device)

    @staticmethod
    def empty_cache():
        torch.npu.empty_cache()

    @staticmethod
    def synchronize():
        torch.npu.synchronize()

    @staticmethod
    def mem_get_info() -> Tuple[int, int]:
        return torch.npu.mem_get_info()

    @staticmethod
    def current_memory_usage(device: torch.device) -> float:
        torch.npu.reset_peak_memory_stats(device)  # type: ignore
        mem = torch.npu.max_memory_allocated(device)  # type: ignore
        return mem
