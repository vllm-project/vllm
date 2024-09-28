import torch

from .interface import DeviceCapability, Platform, PlatformEnum


class XPUPlatform(Platform):
    _enum = PlatformEnum.XPU

    @staticmethod
    def get_device_capability(device_id: int = 0) -> DeviceCapability:
        return DeviceCapability(major=int(
            torch.xpu.get_device_capability(device_id)['version'].split('.')
            [0]),
                                minor=int(
                                    torch.xpu.get_device_capability(device_id)
                                    ['version'].split('.')[1]))

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        return torch.xpu.get_device_name(device_id)
