import torch

from vllm.logger import init_logger

from .interface import DeviceCapability, Platform, PlatformEnum, _Backend

logger = init_logger(__name__)


class XPUPlatform(Platform):
    _enum = PlatformEnum.XPU

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend) -> _Backend:
        if selected_backend != _Backend.IPEX:
            logger.info("Cannot use %s backend on XPU.", selected_backend)
        return _Backend.IPEX

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

    @staticmethod
    def inference_mode():
        return torch.no_grad()
