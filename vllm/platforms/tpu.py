import os

import torch

import vllm.envs as envs
from vllm.compilation.levels import CompilationLevel
from vllm.plugins import set_torch_compile_backend

from .interface import Platform, PlatformEnum

if "VLLM_TORCH_COMPILE_LEVEL" not in os.environ:
    os.environ["VLLM_TORCH_COMPILE_LEVEL"] = str(CompilationLevel.DYNAMO_ONCE)

assert envs.VLLM_TORCH_COMPILE_LEVEL < CompilationLevel.INDUCTOR,\
     "TPU does not support Inductor."

set_torch_compile_backend("openxla")


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
