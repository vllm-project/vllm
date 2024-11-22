from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class OpenVinoPlatform(Platform):
    _enum = PlatformEnum.OPENVINO
    device_type: str = "openvino"

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

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        assert (
            parallel_config.world_size == 1
        ), "OpenVINOExecutor only supports single CPU socket currently."

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = \
                "vllm.worker.openvino_worker.OpenVINOWorker"
