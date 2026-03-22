# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""结构化输出请求模块。

本模块定义了结构化输出请求的数据类，负责：
- 管理请求的结构化输出参数
- 提供文法（grammar）的异步编译支持
- 跟踪 reasoning 状态

主要类：
- StructuredOutputRequest: 结构化输出请求数据类
"""

import dataclasses
import functools
import json
from concurrent.futures import Future
from concurrent.futures._base import TimeoutError
from typing import cast

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_types import (
    StructuredOutputGrammar,
    StructuredOutputKey,
    StructuredOutputOptions,
)


@dataclasses.dataclass
class StructuredOutputRequest:
    """结构化输出请求数据类。

    封装请求的结构化输出参数和文法状态。

    Attributes:
        params: 结构化输出参数
        _grammar: 文法对象或编译文法的 Future
        reasoning_ended: reasoning 阶段是否已结束
        _grammar: 文法对象或 Future（内部属性）
        status: 请求状态（内部属性）
    """

    params: StructuredOutputsParams
    _grammar: Future[StructuredOutputGrammar] | StructuredOutputGrammar | None = None
    reasoning_ended: bool | None = None

    @staticmethod
    def from_sampling_params(
        sampling_params: SamplingParams | None,
    ) -> "StructuredOutputRequest | None":
        """从采样参数创建结构化输出请求。

        Args:
            sampling_params: 采样参数

        Returns:
            结构化输出请求，如果不满足条件则返回 None
        """
        if sampling_params is None:
            return None
        params = sampling_params.structured_outputs
        if not params or params.all_constraints_none():
            return None
        return StructuredOutputRequest(params=params)

    def _check_grammar_completion(self) -> bool:
        """检查文法编译是否完成。

        如果是 Future，尝试获取结果（超时 100 微秒）。
        如果完成，将 Future 替换为实际的文法对象。

        Returns:
            如果文法已准备好则返回 True
        """
        # NOTE: 必须延迟导入以避免循环导入
        from vllm.v1.request import RequestStatus

        if isinstance(self._grammar, Future):
            try:
                # 在 100 微秒内检查 Future 是否就绪
                self._grammar = self._grammar.result(timeout=0.0001)
                self.status = RequestStatus.WAITING
            except TimeoutError:
                return False
        return True

    @property
    def is_grammar_ready(self) -> bool:
        """返回文法是否已准备好。

        Returns:
            如果文法编译完成则返回 True
        """
        return self._check_grammar_completion()

    @property
    def grammar(self) -> StructuredOutputGrammar | None:
        """获取文法对象。

        Returns:
            文法对象，如果尚未准备好则返回 None
        """
        completed = self._check_grammar_completion()
        return (
            cast(StructuredOutputGrammar | None, self._grammar) if completed else None
        )

    @grammar.setter
    def grammar(
        self, grammar: StructuredOutputGrammar | Future[StructuredOutputGrammar]
    ) -> None:
        """设置文法对象。

        Args:
            grammar: 文法对象或编译文法的 Future
        """
        self._grammar = grammar

    @functools.cached_property
    def structured_output_key(self) -> StructuredOutputKey:
        """获取结构化输出的缓存键。

        Returns:
            用于缓存的文法键
        """
        return get_structured_output_key(self.params)


def get_structured_output_key(params: StructuredOutputsParams) -> StructuredOutputKey:
    """从结构化输出参数生成缓存键。

    根据参数类型返回相应的键值对：
    - JSON: (JSON, JSON 字符串)
    - JSON_OBJECT: (JSON_OBJECT, "")
    - REGEX: (REGEX, 正则表达式字符串)
    - CHOICE: (CHOICE, JSON 编码的选择列表)
    - GRAMMAR: (GRAMMAR, 文法字符串)
    - STRUCTURAL_TAG: (STRUCTURAL_TAG, 结构标签)

    Args:
        params: 结构化输出参数

    Returns:
        (选项类型，规范字符串) 元组

    Raises:
        ValueError: 如果没有找到有效的结构化输出参数
    """
    if params.json is not None:
        if not isinstance(params.json, str):
            json_str = json.dumps(params.json)
        else:
            json_str = params.json
        return StructuredOutputOptions.JSON, json_str
    if params.json_object:
        return StructuredOutputOptions.JSON_OBJECT, ""
    if params.regex is not None:
        return StructuredOutputOptions.REGEX, params.regex
    if params.choice is not None:
        if not isinstance(params.choice, str):
            json_str = json.dumps(params.choice)
        else:
            json_str = params.choice
        return StructuredOutputOptions.CHOICE, json_str
    if params.grammar is not None:
        return StructuredOutputOptions.GRAMMAR, params.grammar
    if params.structural_tag is not None:
        return StructuredOutputOptions.STRUCTURAL_TAG, params.structural_tag
    raise ValueError("No valid structured output parameter found")
