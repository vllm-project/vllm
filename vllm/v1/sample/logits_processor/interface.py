# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logits 处理器接口模块。

本模块实现了 logits 处理器的抽象接口，负责：
- 定义 logits 处理器的基类和接口
- 管理批次更新的状态
- 支持日志概率模式配置
- 处理批次内请求的添加、移除和移动

主要类和接口：
- MoveDirectionality: 请求移动方向枚举
- BatchUpdate: 批次更新数据类
- LogitsProcessor: logits 处理器抽象基类
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class MoveDirectionality(Enum):
    """请求移动方向枚举。

    用于描述批次内请求的移动方式。

    Attributes:
        UNIDIRECTIONAL: 单向移动，从 i1 到 i2
        SWAP: 双向交换，i1 和 i2 互换位置
    """
    # 批次内请求的单向移动 i1->i2
    UNIDIRECTIONAL = auto()
    # 批次内请求的双向交换 i1<->i2
    SWAP = auto()


# 任何被移除请求的批次索引
RemovedRequest = int

# 新添加到批次的请求的 (索引，采样参数，prompt_token_ids, output_token_ids) 元组
AddedRequest = tuple[int, SamplingParams, list[int] | None, list[int]]

# 表示批次内请求移动的 (索引 1, 索引 2, 方向性) 元组
MovedRequest = tuple[int, int, MoveDirectionality]


@dataclass(frozen=True)
class BatchUpdate:
    """Logits 处理器的持久化批次状态变更信息。

    用于通知 logits 处理器批次中请求的变化情况，
    包括添加、移除和移动的请求。

    Attributes:
        batch_size: 当前批次中的请求数量

        removed: 从批次中移除的请求索引列表
        added: 新添加到批次的请求信息列表
              每个元组包含 (索引，采样参数，prompt_token_ids, output_token_ids)
        moved: 在批次内移动的请求信息列表
              每个元组包含 (索引 1, 索引 2, 方向性)

    注意：
        * `added` 中的 `output_token_ids` 列表是对请求运行输出 token 列表的引用；
          通过此引用，logits 处理器总是能看到最新的生成输出 token 列表。
        * 添加或移动的请求可能会替换具有相同索引的现有请求。
        * 操作应按以下顺序处理：
          - removed（移除）, added（添加）, moved（移动）
    """

    batch_size: int  # 当前批次中的请求数量
    """当前批次中的请求数量"""

    # 批次元数据，用于从持久化批次中移除、添加和移动的请求。
    #
    # 关键假设：`added` 中的每个元组的 `output_token_ids` 列表
    # 是对请求运行输出 token 列表的引用；通过此引用，
    # logits 处理器总是能看到最新的生成输出 token 列表。
    #
    # 注意：
    # * 添加或移动的请求可能会替换具有相同索引的现有请求。
    # * 操作应按以下顺序处理：
    #   - removed, added, moved
    removed: Sequence[RemovedRequest]
    """从批次中移除的请求索引列表"""
    added: Sequence[AddedRequest]
    """新添加到批次的请求信息列表"""
    moved: Sequence[MovedRequest]
    """在批次内移动的请求信息列表"""


class LogitsProcessor(ABC):
    """Logits 处理器抽象基类。

    定义了 logits 处理器的通用接口，所有 logits 处理器
    都必须实现此接口定义的方法。

    Logits 处理器用于在采样前修改模型输出的 logits，
    实现各种约束和惩罚功能。
    """

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        """验证采样参数对此 logits 处理器是否有效。

        如果参数无效则抛出 ValueError。

        Args:
            sampling_params: 要验证的采样参数
        """
        return None

    @abstractmethod
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        """初始化 logits 处理器。

        Args:
            vllm_config: vLLM 配置
            device: 要使用的设备
            is_pin_memory: 是否使用锁页内存
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """将 logits 处理器应用到批次 logits 张量。

        必须返回更新后的张量，但可以就地修改。

        Args:
            logits: 输入的 logits 张量

        Returns:
            处理后的 logits 张量
        """
        raise NotImplementedError

    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        """判断此 logits 处理器是否对贪婪采样中的 argmax 计算没有影响。

        注意：对于给定的 LogitsProcessor 子类的所有实例，
        可能具有相同或不同的值，具体取决于子类的实现。

        Returns:
            如果对 argmax 计算没有影响则返回 True
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(
        self,
        batch_update: "BatchUpdate | None",
    ) -> None:
        """在每次前向传递之前，当有新输出 token 时调用。

        Args:
            batch_update: 如果不为 None，则表示批次构成发生了变化
        """
        raise NotImplementedError
