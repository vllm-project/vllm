# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU 模型运行器模块。

本模块定义 XPU（Intel GPU）设备专用的模型运行器，负责：
- 继承 GPU 模型运行器并适配 XPU 后端
- 将 CUDA API 调用替换为 XPU API 调用
- 支持 XPU Graph（类似 CUDA Graph）

主要类：
- XPUModelRunner: XPU 模型运行器（旧版本）
- XPUModelRunnerV2: XPU 模型运行器（新版本）
"""
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
    """XPU 模型运行器类。

    继承自 GPUModelRunner，通过替换 CUDA API 为 XPU API 来支持 Intel GPU 设备。
    """

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
    """XPU 模型运行器类（新版本）。

    继承自 GPUModelRunnerV2，通过替换 CUDA API 为 XPU API 来支持 Intel GPU 设备。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """初始化 XPU 模型运行器。

        Args:
            vllm_config: vLLM 配置
            device: 计算设备
        """
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)


@contextmanager
def _torch_cuda_wrapper():
    """上下文管理器：将 CUDA API 替换为 XPU API。

    通过动态替换 torch.cuda 命名空间中的 API 为 torch.xpu 对应 API，
    使得原本为 CUDA 设计的代码可以在 Intel XPU 设备上运行。

    替换的 API 包括：
    - Stream: 流管理
    - default_stream/current_stream: 默认/当前流
    - mem_get_info: 内存信息查询
    - Event: 事件同步
    - graph: 图形捕获（类似 CUDA Graph）
    """
    try:
        # replace cuda APIs with xpu APIs, this should work by default
        torch.cuda.Stream = torch.xpu.Stream
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
