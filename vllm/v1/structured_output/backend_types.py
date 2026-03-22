# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""结构化输出后端类型模块。

本模块定义了结构化输出后端的抽象类型，负责：
- 定义结构化输出的选项类型
- 定义文法对象的抽象接口
- 定义后端的抽象接口

主要类：
- StructuredOutputOptions: 结构化输出选项枚举
- StructuredOutputGrammar: 文法抽象基类
- StructuredOutputBackend: 后端抽象基类
"""

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.tokenizers import TokenizerLike
else:
    VllmConfig = object
    TokenizerLike = object


class StructuredOutputOptions(enum.Enum):
    """结构化输出选项枚举。

    定义支持的结构化输出类型：
    - JSON: JSON Schema 约束
    - JSON_OBJECT: 任意 JSON 对象
    - REGEX: 正则表达式约束
    - GRAMMAR: EBNF 文法约束
    - CHOICE: 选择列表约束
    - STRUCTURAL_TAG: 结构标签约束
    """

    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()
    STRUCTURAL_TAG = enum.auto()


# 结构化输出键类型：(选项类型，规范字符串)
StructuredOutputKey = tuple[StructuredOutputOptions, str]


class StructuredOutputGrammar(ABC):
    """请求级结构化输出文法抽象基类。

    定义文法对象的接口，用于处理 token 的验证和位掩码生成。
    每个具体实现需要实现以下方法：
    - accept_tokens: 接受 token 列表并推进解析器
    - validate_tokens: 验证 token 列表（不推进解析器）
    - rollback: 回退指定数量的 token
    - fill_bitmask: 填充位掩码
    - is_terminated: 检查是否已终止
    - reset: 重置状态
    """

    @abstractmethod
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """确定是否接受提供的 token 列表。

        Args:
            request_id: 请求的唯一标识符
            tokens: 要评估的 token ID 列表

        Returns:
            如果 token 被接受则返回 True，否则返回 False
        """
        pass

    @abstractmethod
    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """针对文法验证提供的 token 列表。

        不会推进有限状态机（FSM）。

        Args:
            tokens: 要验证的 token ID 列表

        Returns:
            被接受的 token ID 列表。将是输入 token 的前缀，
            如果没有被接受的则返回空列表
        """
        pass

    @abstractmethod
    def rollback(self, num_tokens: int) -> None:
        """回退文法状态指定数量的 token。

        同时会回退已处理 token 的计数器。

        Args:
            num_tokens: 要回退的 token 数量
        """
        pass

    @abstractmethod
    def fill_bitmask(self, bitmask: "torch.Tensor", batch_index: int) -> None:
        """填充特定 batch 索引的位掩码。

        Args:
            bitmask: 要填充的位掩码 tensor
            batch_index: 要填充的位掩码索引
        """
        pass

    @abstractmethod
    def is_terminated(self) -> bool:
        """检查结构化输出过程是否已终止。

        Returns:
            如果过程已终止则返回 True，否则返回 False
        """
        pass

    @abstractmethod
    def reset(self):
        """重置结构化输出文法的状态。"""
        pass


@dataclass
class StructuredOutputBackend(ABC):
    """引擎级结构化输出后端抽象基类。

    定义后端的接口，用于编译文法和分配资源。
    每个具体实现需要实现以下方法：
    - compile_grammar: 编译文法规范
    - allocate_token_bitmask: 分配 token 位掩码
    - destroy: 清理后端资源

    Attributes:
        vllm_config: vLLM 配置
        tokenizer: 分词器
        vocab_size: 词表大小
    """

    vllm_config: VllmConfig
    tokenizer: TokenizerLike
    vocab_size: int

    @abstractmethod
    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        """将文法规范编译为结构化输出文法。

        Args:
            request_type: 结构化输出请求类型
            grammar_spec: 要编译的文法规范

        Returns:
            编译后的结构化输出文法
        """
        pass

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int) -> "torch.Tensor":
        """为指定的最大序列数分配 token 位掩码。

        Args:
            max_num_seqs: 要分配位掩码的最大序列数
        """
        pass

    @abstractmethod
    def destroy(self):
        """后端特定的清理操作。"""
        pass
