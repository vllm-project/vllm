import os
from typing import TYPE_CHECKING

import torch

from .interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


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

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm.config import CompilationLevel
        compilation_config = vllm_config.compilation_config
        if "VLLM_TORCH_COMPILE_LEVEL" not in os.environ:
            compilation_config.level = CompilationLevel.DYNAMO_ONCE
        assert compilation_config.level < CompilationLevel.PIECEWISE,\
            "TPU does not support Inductor."

        if compilation_config.backend == "":
            compilation_config.backend = "openxla"
