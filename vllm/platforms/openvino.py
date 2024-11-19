import torch

import vllm.envs as envs
from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

logger = init_logger(__name__)


class OpenVinoPlatform(Platform):
    _enum = PlatformEnum.OPENVINO

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend) -> _Backend:
        if selected_backend != _Backend.OPENVINO:
            logger.info("Cannot use %s backend on OpenVINO.", selected_backend)
        return _Backend.OPENVINO

    @classmethod
    def get_device_name(self, device_id: int = 0) -> str:
        return "openvino"

    @classmethod
    def inference_mode(self):
        return torch.inference_mode(mode=True)

    @classmethod
    def is_openvino_cpu(self) -> bool:
        return "CPU" in envs.VLLM_OPENVINO_DEVICE

    @classmethod
    def is_openvino_gpu(self) -> bool:
        return "GPU" in envs.VLLM_OPENVINO_DEVICE

    @classmethod
    def is_pin_memory_available(self) -> bool:
        logger.warning("Pin memory is not supported on OpenViNO.")
        return False
