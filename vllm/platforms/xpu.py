import torch

from .interface import DeviceCapability, Platform, PlatformEnum


class XPUPlatform(Platform):
    _enum = PlatformEnum.XPU

    @staticmethod
    def get_device_capability(device_id: int = 0) -> DeviceCapability:
        major, minor, *_ = torch.xpu.get_device_capability(
            device_id)['version'].split('.')
        return DeviceCapability(major=int(major), minor=int(minor))

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        return torch.xpu.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.xpu.get_device_properties(device_id)
        return device_props.total_memory
