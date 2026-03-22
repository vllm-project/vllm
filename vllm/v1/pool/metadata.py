# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pooling 模块。

本模块提供了 vLLM V1 的池化（Pooling）相关功能，包括：
- 池化元数据管理
- 池化状态跟踪
- 池化游标（Cursor）用于定位 token

主要类：
- PoolingCursor: 池化操作的游标，定位首批和末尾 token
- PoolingStates: 池化状态容器，缓存隐藏状态
- PoolingMetadata: 池化元数据容器
"""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.utils.platform_utils import is_pin_memory_available

# 检查是否支持固定内存（pin memory）
pin_memory = is_pin_memory_available()


@dataclass
class PoolingCursor:
    """池化操作游标。

    用于在池化操作中快速定位每个序列的首个和末尾 token，
    以及存储序列长度相关信息。

    Attributes:
        first_token_indices_gpu: GPU 上的首个 token 索引 [batch_size]
        last_token_indices_gpu: GPU 上的末尾 token 索引 [batch_size]
        prompt_lens_cpu: CPU 上的 prompt 长度 [batch_size]
        seq_lens_cpu: CPU 上的序列长度 [batch_size]
        num_scheduled_tokens_cpu: CPU 上的调度 token 数量 [batch_size]
    """
    first_token_indices_gpu: torch.Tensor
    last_token_indices_gpu: torch.Tensor
    prompt_lens_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    num_scheduled_tokens_cpu: torch.Tensor

    def __getitem__(self, indices: slice):
        """切片操作，返回部分游标数据。

        Args:
            indices: 切片索引

        Returns:
            新的 PoolingCursor 实例，包含切片后的数据
        """
        return PoolingCursor(
            first_token_indices_gpu=self.first_token_indices_gpu[indices],
            last_token_indices_gpu=self.last_token_indices_gpu[indices],
            prompt_lens_cpu=self.prompt_lens_cpu[indices],
            seq_lens_cpu=self.seq_lens_cpu[indices],
            num_scheduled_tokens_cpu=self.num_scheduled_tokens_cpu[indices],
        )

    def is_partial_prefill(self):
        """检查是否是部分预填充。

        Returns:
            如果 prompt 长度不等于调度 token 数量，返回 True
        """
        return not torch.all(self.prompt_lens_cpu == self.num_scheduled_tokens_cpu)

    def is_finished(self):
        """检查序列是否完成。

        Returns:
            如果 prompt 长度等于序列长度，返回 True
        """
        return self.prompt_lens_cpu == self.seq_lens_cpu


class PoolingStates:
    """池化状态容器。

    用于在分块预填充（chunked prefill）期间缓存隐藏状态。
    支持所有池化操作类型。

    Attributes:
        hidden_states_cache: 隐藏状态缓存列表
    """

    def __init__(self):
        """初始化池化状态容器。

        用于分块预填充的 ALL 池化操作。
        """
        # 用于分块预填充的 ALL 池化
        self.hidden_states_cache: list[torch.Tensor] = []

    def clean(self):
        """清空隐藏状态缓存。"""
        self.hidden_states_cache.clear()


@dataclass
class PoolingMetadata:
    """池化元数据容器。

    存储池化操作所需的所有元数据信息，包括 prompt 长度、
    token ID、池化参数、状态和游标。

    Attributes:
        prompt_lens: prompt 长度张量（CPU）
        prompt_token_ids: prompt token ID 张量（可选）
        pooling_params: 池化参数列表
        pooling_states: 池化状态列表
        pooling_cursor: 池化游标（可选）
        tasks: 从参数中提取的池化任务列表（post_init 后）
    """
    prompt_lens: torch.Tensor  # CPU Tensor
    prompt_token_ids: torch.Tensor | None
    pooling_params: list[PoolingParams]
    pooling_states: list[PoolingStates]
    pooling_cursor: PoolingCursor | None = None

    def __post_init__(self) -> None:
        """后处理初始化，提取池化任务列表。

        从 pooling_params 中提取 task 字段，验证所有参数都有任务。
        """
        pooling_params = self.pooling_params

        # 提取所有池化任务
        tasks: list[PoolingTask] = [
            task
            for pooling_param in pooling_params
            if (task := pooling_param.task) is not None
        ]
        assert len(pooling_params) == len(tasks)

        self.tasks = tasks

    def __getitem__(self, indices: slice):
        """切片操作，返回部分元数据。

        Args:
            indices: 切片索引

        Returns:
            新的 PoolingMetadata 实例，包含切片后的数据
        """
        return PoolingMetadata(
            prompt_lens=self.prompt_lens[indices],
            prompt_token_ids=None
            if self.prompt_token_ids is None
            else self.prompt_token_ids[indices],
            pooling_params=self.pooling_params[indices],
            pooling_states=self.pooling_states[indices],
            pooling_cursor=None
            if self.pooling_cursor is None
            else self.pooling_cursor[indices],
        )

    def get_prompt_token_ids(self) -> list[torch.Tensor]:
        """获取每个 prompt 的 token ID 列表。

        Returns:
            prompt token ID 列表，每个元素是一个 1D 张量

        Raises:
            AssertionError: 如果 prompt_token_ids 为 None
        """
        prompt_token_ids = self.prompt_token_ids
        assert prompt_token_ids is not None, (
            "Please set `requires_token_ids=True` in `get_pooling_updates`"
        )

        return [prompt_token_ids[i, :num] for i, num in enumerate(self.prompt_lens)]

    def get_pooling_cursor(self) -> PoolingCursor:
        """获取池化游标。

        Returns:
            池化游标

        Raises:
            AssertionError: 如果 pooling_cursor 为 None
        """
        pooling_cursor = self.pooling_cursor
        assert pooling_cursor is not None, "Should call `build_pooling_cursor` first"

        return pooling_cursor

    def build_pooling_cursor(
        self,
        num_scheduled_tokens_np: np.ndarray,
        seq_lens_cpu: torch.Tensor,
        device: torch.device,
        query_start_loc_gpu: torch.Tensor | None = None,
    ):
        """构建池化游标。

        根据调度 token 数量和序列长度构建游标，用于定位每个序列的
        首个和末尾 token 位置。

        Args:
            num_scheduled_tokens_np: 调度 token 数量的 numpy 数组
            seq_lens_cpu: CPU 上的序列长度张量
            device: 设备类型
            query_start_loc_gpu: 可选的 GPU 查询起始位置

        Raises:
            ValueError: 如果 query_start_loc_gpu 长度不匹配或设备不匹配
        """
        n_seq = len(num_scheduled_tokens_np)
        prompt_lens = self.prompt_lens

        assert len(prompt_lens) == n_seq

        num_scheduled_tokens_cpu = torch.from_numpy(num_scheduled_tokens_np)
        if query_start_loc_gpu is None:
            # 从 num_scheduled_tokens 计算累积和
            cumsum = torch.zeros(
                n_seq + 1, dtype=torch.int64, pin_memory=pin_memory, device="cpu"
            )
            torch.cumsum(num_scheduled_tokens_cpu, dim=0, out=cumsum[1:])
            cumsum = cumsum.to(device, non_blocking=True)
        else:
            # 验证 query_start_loc_gpu 的长度和设备
            if query_start_loc_gpu.shape[0] != n_seq + 1:
                raise ValueError(
                    "query_start_loc_gpu length does not match "
                    f"the number of sequences: {query_start_loc_gpu.shape[0]} "
                    f"!= {n_seq + 1}."
                )
            if query_start_loc_gpu.device != device:
                raise ValueError(
                    "query_start_loc_gpu must be on the same device as the "
                    f"hidden states: {query_start_loc_gpu.device} != {device}."
                )
            cumsum = query_start_loc_gpu

        # 构建游标
        self.pooling_cursor = PoolingCursor(
            first_token_indices_gpu=cumsum[:n_seq],
            last_token_indices_gpu=cumsum[1:] - 1,
            prompt_lens_cpu=prompt_lens,
            seq_lens_cpu=seq_lens_cpu,
            num_scheduled_tokens_cpu=num_scheduled_tokens_cpu,
        )
