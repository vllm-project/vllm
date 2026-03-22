# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""异步输出工具函数模块。

本模块提供异步输出相关的辅助函数和类，负责：
- 异步模型输出的封装和管理
- 跨 CUDA 流的异步数据复制
- 采样输出和池化输出的异步处理

主要类：
- AsyncOutput: 异步模型输出类（用于生成模型）
- AsyncPoolingOutput: 异步池化输出类（用于池化模型）
"""
import contextlib

import numpy as np
import torch

from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsTensors, ModelRunnerOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class AsyncOutput(AsyncModelRunnerOutput):
    """异步模型输出类（用于生成模型）。

    在独立的 CUDA 流上异步执行从 GPU 到 CPU 的数据复制，
    允许模型在输出尚未完全生成时继续调度新的步骤。

    Attributes:
        model_runner_output: 模型运行器输出
        sampler_output: 采样器输出
        num_sampled_tokens: 采样 token 数量
        copy_event: CUDA 复制事件
        sampled_token_ids: 采样的 token ID（numpy 数组）
        logprobs_tensors: logprobs 张量
        num_nans: NaN 计数
        num_sampled_tokens_np: 采样 token 数量（numpy 数组）
        prompt_logprobs_dict: prompt logprobs 字典
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        copy_event: torch.cuda.Event,
    ):
        """初始化异步输出。

        Args:
            model_runner_output: 模型运行器输出
            sampler_output: 采样器输出
            num_sampled_tokens: 采样 token 数量张量
            main_stream: 主 CUDA 流
            copy_stream: 复制 CUDA 流
            copy_event: CUDA 复制事件
        """
        # NOTE(woosuk): 我们必须保留对 GPU 张量的引用，
        # 因为复制操作在与张量创建时不同的 CUDA 流上执行。
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.copy_event = copy_event

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)

            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        """获取同步后的模型运行器输出。

        等待异步复制完成，然后将数据转换为所需的格式。

        Returns:
            同步后的 ModelRunnerOutput
        """
        self.copy_event.synchronize()

        # NOTE(woosuk): 以下代码是为了与现有模型运行器兼容。
        # 未来，我们应该将这些数据结构保留为 NumPy 数组
        # 而不是 Python 列表。
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        return self.model_runner_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    """异步池化输出类（用于池化模型）。

    在独立的 CUDA 流上异步执行池化输出的复制。

    Attributes:
        model_runner_output: 模型运行器输出
        pooler_output: 池化器输出
        is_valid: 有效性标记
        copy_event: CUDA 复制事件
        pooler_output_cpu: CPU 上的池化输出
        is_valid_cpu: CPU 上的有效性标记
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: torch.Tensor,
        is_valid: torch.Tensor | None,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        copy_event: torch.cuda.Event,
    ):
        """初始化异步池化输出。

        Args:
            model_runner_output: 模型运行器输出
            pooler_output: 池化器输出张量
            is_valid: 有效性标记张量
            main_stream: 主 CUDA 流
            copy_stream: 复制 CUDA 流
            copy_event: CUDA 复制事件
        """
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        self.is_valid = is_valid
        self.copy_event = copy_event

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            self.pooler_output_cpu = self.pooler_output.to("cpu", non_blocking=True)
            if self.is_valid is not None:
                self.is_valid_cpu = self.is_valid.to("cpu", non_blocking=True)
            else:
                self.is_valid_cpu = None
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        """获取同步后的模型运行器输出。

        Returns:
            同步后的 ModelRunnerOutput
        """
        pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        self.copy_event.synchronize()
        if self.is_valid_cpu is not None:
            is_valid_cpu = self.is_valid_cpu.tolist()
            for i, is_valid in enumerate(is_valid_cpu):
                if not is_valid:
                    pooler_output[i] = None
        self.model_runner_output.pooler_output = pooler_output
        return self.model_runner_output


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    """异步复制张量到 numpy 数组。

    Args:
        x: 要复制的张量

    Returns:
        CPU 上的 numpy 数组
    """
    return x.to("cpu", non_blocking=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """轻量级的 torch.cuda.stream() 上下文管理器。

    避免 current_stream 和 device 查找的开销。

    Args:
        to_stream: 要切换到的 CUDA 流
        from_stream: 要恢复的 CUDA 流
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)
