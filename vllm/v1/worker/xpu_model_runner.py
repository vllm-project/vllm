# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import supports_xpu_graph
from vllm.v1.worker.gpu.model_runner import (
    GPUModelRunner as GPUModelRunnerV2,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        # FIXME: To be verified.
        self.cascade_attn_enabled = False


class XPUModelRunnerV2(GPUModelRunnerV2):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)


@contextmanager
def _torch_cuda_wrapper():
    try:
        # replace cuda APIs with xpu APIs, this should work by default
        torch.Stream = torch.xpu.Stream
        torch.cuda.default_stream = torch.xpu.current_stream
        torch.cuda.current_stream = torch.xpu.current_stream
        torch.cuda.stream = torch.xpu.stream
        torch.cuda.mem_get_info = torch.xpu.mem_get_info
        torch.cuda.Event = torch.Event
        torch.cuda.set_stream = torch.xpu.set_stream
        if supports_xpu_graph():
            torch.cuda.graph = torch.xpu.graph
            torch.cuda.CUDAGraph = torch.xpu.XPUGraph
            torch.cuda.graph_pool_handle = torch.xpu.graph_pool_handle
        yield
    finally:
        pass
