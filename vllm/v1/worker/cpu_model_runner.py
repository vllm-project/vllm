# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU 模型运行器模块。

本模块定义 CPU 设备专用的模型运行器，负责：
- 继承 GPU 模型运行器并适配 CPU 后端
- 将 GPU 张量替换为 CPU 张量
- 禁用 CUDA Graph 和 Cascade Attention
- 支持 MKLDNN 和 CPPGEMM 后端编译优化

主要类：
- CPUModelRunner: CPU 模型运行器
"""
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.tracing import instrument
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    """CPU 模型运行器类。

    继承自 GPUModelRunner，适配 CPU 后端运行环境。
    主要特点：
    - 不支持 spec decode
    - 禁用 CUDA Graph
    - 禁用 Cascade Attention
    - 使用 CPU 张量替代 GPU 张量
    """
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        """初始化 CPU 模型运行器。

        Args:
            vllm_config: vLLM 配置
            device: 计算设备（必须为 cpu）
        """
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        assert device == torch.device("cpu")
        assert self.speculative_config is None, "spec decode is not supported."

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        self._postprocess_tensors()

    def _postprocess_tensors(self) -> None:
        """后处理张量：将 GPU 张量替换为 CPU 张量。

        遍历所有 CpuGpuBuffer 和 input_batch 中的张量，
        将原本分配给 GPU 的张量替换为 CPU 张量。
        """
        def replace_tensor(obj: Any, cpu_attr_name: str, device_attr_name) -> None:
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            device_tensor = getattr(obj, device_attr_name, None)
            if isinstance(cpu_tensor, torch.Tensor) and isinstance(
                device_tensor, torch.Tensor
            ):
                setattr(obj, device_attr_name, cpu_tensor)

        for v in vars(self).values():
            if isinstance(v, CpuGpuBuffer):
                v.gpu = v.cpu

        for k, v in vars(self.input_batch).items():
            if k.endswith("_cpu_tensor") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch, k, k[:-11])

        for block_table in self.input_batch.block_table.block_tables:
            for v in vars(block_table).values():
                if isinstance(v, CpuGpuBuffer):
                    v.gpu = v.cpu

    @instrument(span_name="Loading (CPU)")
    def load_model(self, load_dummy_weights: bool = False) -> None:
        """加载模型。

        Args:
            load_dummy_weights: 是否加载虚拟权重（CPU 不支持）

        Raises:
            ValueError: 如果尝试加载虚拟权重（弹性 EP 扩展不支持）
        """
        if load_dummy_weights:
            raise ValueError(
                "Loading dummy weights (needed for elastic EP scale-up) "
                "Is not supported by the CPU Model Runner."
            )
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

    def get_model(self) -> nn.Module:
        """获取模型。

        Returns:
            加载的模型模块
        """
        return self.model

    @instrument(span_name="Warmup (CPU)")
    def warming_up_model(self) -> None:
        """预热模型用于编译。

        执行虚拟运行以触发模型编译，仅针对通用形状生成。
        """
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings(self.vllm_config):
            self._dummy_run(
                min(
                    max(16, self.max_num_reqs),
                    self.scheduler_config.max_num_batched_tokens,
                )
            )

        logger.info("Warming up done.")

    def _init_device_properties(self) -> None:
        """初始化设备属性（CPU 不需要）。"""
        pass

    def _sync_device(self) -> None:
        """同步设备（CPU 不需要）。"""
        pass

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        """获取 DP padding。

        Args:
            num_tokens: token 数量

        Returns:
            (padding 数量，padding 张量) 元组。CPU 后端不需要 DP padding。
        """
        # Note: For CPU backend, dp padding is not required for now.
        return 0, None


@contextmanager
def _torch_cuda_wrapper():
    """上下文管理器：模拟 CUDA API 用于 CPU 后端。

    提供虚拟的 Event 和 Stream 类，使得原本依赖 CUDA API 的代码
    可以在 CPU 上运行而不出错。
    """
    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            pass

    cuda_event = torch.Event
    cuda_stream = torch.cuda.Stream
    try:
        torch.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        yield
    finally:
        torch.Event = cuda_event
        torch.cuda.Stream = cuda_stream


@contextmanager
def _set_global_compilation_settings(config: VllmConfig):
    """设置全局编译参数的上下文管理器。

    MKLDNN 和 CPPGEMM 后端需要冻结参数，此上下文管理器
    在 max_autotune 启用时设置 freezing=True。

    Args:
        config: vLLM 配置
    """
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    # Note: The MKLDNN and CPPGEMM backend requires freezing parameters.
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value
