# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guidance 后端模块。

本模块实现了基于 Guidance/llguidance 的结构化输出后端，负责：
- 编译 Guidance 文法
- 处理 JSON Schema 约束
- 管理 token 位掩码
- 支持结构标签（structural tags）

主要类：
- GuidanceBackend: Guidance 后端实现
- GuidanceGrammar: Guidance 文法实现

主要函数：
- serialize_guidance_grammar: 序列化文法为 Guidance 格式
- validate_guidance_grammar: 验证 Guidance 文法
"""

import copy
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.request import get_structured_output_key

if TYPE_CHECKING:
    import llguidance
    import llguidance.hf as llguidance_hf
    import llguidance.torch as llguidance_torch
else:
    llguidance = LazyLoader("llguidance", globals(), "llguidance")
    llguidance_hf = LazyLoader("llguidance.hf", globals(), "llguidance.hf")
    llguidance_torch = LazyLoader("llguidance.torch", globals(), "llguidance.torch")

logger = init_logger(__name__)


def _walk_json_for_additional_properties(data: object):
    """遍历 JSON 对象，添加 additionalProperties: False。

    递归遍历 JSON 结构，为所有包含 properties 或 patternProperties
    但没有 additionalProperties 的对象添加 additionalProperties: False。

    Args:
        data: 要遍历的 JSON 数据
    """
    if isinstance(data, dict):
        for value in data.values():
            _walk_json_for_additional_properties(value)
        if "additionalProperties" not in data and (
            "properties" in data or "patternProperties" in data
        ):
            data["additionalProperties"] = False
    elif isinstance(data, list):
        for item in data:
            _walk_json_for_additional_properties(item)


def has_guidance_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """检查 JSON Schema 是否包含 guidance/llguidance 不支持的功能。

    目前不支持的功能：
    - patternProperties：llguidance 不支持的模式属性

    Args:
        schema: JSON Schema 对象

    Returns:
        如果包含不支持的功能则返回 True，否则返回 False
    """

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        # llguidance 不支持 patternProperties
        if "patternProperties" in obj:
            return True

        # 递归检查所有嵌套对象和数组
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


def process_for_additional_properties(
    guide_json: str | dict[str, Any],
) -> dict[str, Any]:
    """处理 JSON，为所有对象添加 additionalProperties: False。

    Args:
        guide_json: JSON 字符串或对象

    Returns:
        处理后的 JSON 对象
    """
    if isinstance(guide_json, str):
        guide_json_obj = json.loads(guide_json)
    else:
        # 复制以便修改
        guide_json_obj = copy.deepcopy(guide_json)
    _walk_json_for_additional_properties(guide_json_obj)
    return guide_json_obj


@dataclass
class GuidanceBackend(StructuredOutputBackend):
    """Guidance 后端实现。

    使用 llguidance 库实现结构化输出约束。
    支持 JSON Schema、正则表达式、EBNF 文法等多种格式。

    Attributes:
        vllm_config: vLLM 配置
        tokenizer: 分词器
        vocab_size: 词表大小
        disable_any_whitespace: 是否禁用任意空白
        disable_additional_properties: 是否禁用额外属性
        ll_tokenizer: llguidance 分词器
        serialized_grammar: 序列化的文法
    """

    def __post_init__(self):
        """初始化后端配置。

        从配置中读取选项并创建 llguidance 分词器。
        """
        self.disable_any_whitespace = (
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        )
        self.disable_additional_properties = (
            self.vllm_config.structured_outputs_config.disable_additional_properties
        )

        self.ll_tokenizer = llguidance_hf.from_tokenizer(
            self.tokenizer, max(self.vocab_size, len(self.tokenizer))
        )

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        """编译文法规范为 Guidance 文法。

        Args:
            request_type: 结构化输出请求类型
            grammar_spec: 文法规范

        Returns:
            编译后的 Guidance Grammar
        """
        self.serialized_grammar = serialize_guidance_grammar(
            request_type,
            grammar_spec,
            self.disable_any_whitespace,
            self.disable_additional_properties,
        )

        ll_matcher = llguidance.LLMatcher(
            self.ll_tokenizer,
            self.serialized_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )

        r = GuidanceGrammar(
            ll_matcher=ll_matcher,
            ll_tokenizer=self.ll_tokenizer,
            vocab_size=self.vocab_size,
        )

        r.check_error()
        return r

    def allocate_token_bitmask(self, max_num_seqs: int):
        """分配 token 位掩码。

        Args:
            max_num_seqs: 最大序列数

        Returns:
            分配的位掩码 tensor
        """
        return llguidance_torch.allocate_token_bitmask(
            max_num_seqs, self.ll_tokenizer.vocab_size
        )

    def destroy(self):
        """清理后端资源。

        Guidance 后端无需特殊清理。
        """
        pass


@dataclass
class GuidanceGrammar(StructuredOutputGrammar):
    """Guidance 文法实现。

    封装 llguidance.LLMatcher，提供文法约束功能。

    Attributes:
        ll_matcher: llguidance 匹配器
        ll_tokenizer: llguidance 分词器
        vocab_size: 词表大小
        printed_error: 是否已打印错误
        terminated: 是否已终止
        rollback_lag: 回退滞后计数
    """

    ll_matcher: llguidance.LLMatcher
    ll_tokenizer: llguidance.LLTokenizer
    vocab_size: int
    printed_error: bool = False
    terminated: bool = False
    rollback_lag: int = 0

    def check_error(self):
        """检查并记录匹配器错误。

        如果检测到错误且尚未打印，则记录警告日志。
        """
        if not self.printed_error:
            err = self.ll_matcher.get_error()
            if err:
                self.printed_error = True
                logger.warning("LLMatcher error: %s", err)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """接受 token 列表并推进解析器。

        Args:
            request_id: 请求 ID（未使用）
            tokens: token ID 列表

        Returns:
            如果解析器成功推进则返回 True，否则返回 False
        """
        if self.ll_tokenizer.eos_token in tokens:
            if self.ll_matcher.is_stopped() and not self.terminated:
                self.rollback_lag = 1
            self.terminated = True

        if self.ll_matcher.is_stopped():
            return True

        # TODO - 未来支持 jump decoding：
        # self.ll_matcher.compute_ff_bytes() - 应该总是有效
        # self.ll_matcher.compute_ff_tokens() - 仅适用于"canonical"tokenizer
        # 两者之间的转换参考：
        # https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md

        r = self.ll_matcher.consume_tokens(tokens)

        self.check_error()

        return r

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """按顺序检查 token 列表是否被解析器接受。

        不会推进解析器。

        Args:
            tokens: token ID 列表

        Returns:
            被解析器接受的前缀 token 列表
        """
        if len(tokens) == 0:
            return []
        if self.ll_matcher.is_stopped():
            return []

        num_tokens = self.ll_matcher.validate_tokens(tokens)

        self.check_error()

        return tokens[:num_tokens]

    def rollback(self, num_tokens: int) -> None:
        """回退文法状态指定数量的 token。

        Args:
            num_tokens: 要回退的 token 数量
        """
        if num_tokens > 0:
            self.ll_matcher.rollback(num_tokens - self.rollback_lag)
            self.terminated = False
            self.rollback_lag = 0
            self.check_error()

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        """填充下一个 token 的位掩码。

        如果匹配器已停止或处于错误状态，会自动返回 [EOS] 掩码。

        Args:
            bitmask: 要填充的位掩码 tensor
            idx: 位掩码索引
        """
        # 匹配器停止或处于错误状态时会自动返回 [EOS] 掩码
        llguidance_torch.fill_next_token_bitmask(self.ll_matcher, bitmask, idx)
        self.check_error()

    def is_terminated(self) -> bool:
        """检查文法是否已终止。

        Returns:
            如果已终止则返回 True
        """
        return self.terminated

    def reset(self):
        """重置文法状态。

        此方法可能已不再需要？TODO
        """
        self.ll_matcher.reset()


def serialize_guidance_grammar(
    request_type: StructuredOutputOptions,
    grammar_spec: str | dict[str, Any],
    disable_any_whitespace: bool = False,
    disable_additional_properties: bool = False,
) -> str:
    """将文法规范序列化为 Guidance 格式。

    根据请求类型将不同的文法格式转换为 llguidance 可识别的格式。

    Args:
        request_type: 结构化输出类型
        grammar_spec: 文法规范（字符串或 JSON 对象）
        disable_any_whitespace: 是否禁用任意空白
        disable_additional_properties: 是否禁用额外属性

    Returns:
        序列化的 Guidance 文法字符串
    """
    def _process_schema(
        grammar_spec: str | dict[str, Any],
    ) -> str:
        """处理 JSON Schema。

        Args:
            grammar_spec: JSON Schema

        Returns:
            序列化的文法字符串
        """
        if disable_additional_properties:
            grammar_spec = process_for_additional_properties(grammar_spec)
        return llguidance.LLMatcher.grammar_from_json_schema(
            grammar_spec,
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            },
        )

    if request_type == StructuredOutputOptions.JSON:
        return _process_schema(grammar_spec)
    elif request_type == StructuredOutputOptions.JSON_OBJECT:
        return llguidance.LLMatcher.grammar_from_json_schema(
            '{"type": "object"}',
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            },
        )
    else:
        if request_type == StructuredOutputOptions.REGEX:
            tp = "regex"
        elif request_type == StructuredOutputOptions.GRAMMAR:
            tp = "grammar"
        elif request_type == StructuredOutputOptions.CHOICE:
            tp = "choice"
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            if isinstance(grammar_spec, str):
                s_tag = json.loads(grammar_spec)
            else:
                s_tag = grammar_spec
            triggers: list[str] = s_tag["triggers"]
            tags: list[llguidance.StructTag] = []
            for s in s_tag["structures"]:
                begin: str = s["begin"]
                trig = next((t for t in triggers if begin.startswith(t)), None)
                if trig is None:
                    raise ValueError(
                        f"Trigger {begin} not found in triggers {triggers}"
                    )
                tags.append(
                    llguidance.StructTag(
                        trigger=trig,
                        begin=s["begin"],
                        grammar=_process_schema(s["schema"]),
                        end=s["end"],
                    )
                )
            if not tags:
                raise ValueError("No structural tags found in the grammar spec.")
            return llguidance.StructTag.to_grammar(tags)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})"
            )
        return llguidance.grammar_from(tp, grammar_spec)


def validate_guidance_grammar(
    sampling_params: SamplingParams, tokenizer: llguidance.LLTokenizer | None = None
) -> None:
    """验证 Guidance 文法。

    Args:
        sampling_params: 采样参数
        tokenizer: llguidance 分词器（可选）

    Raises:
        ValueError: 如果文法无效
    """
    # 如果未启用结构化输出，则无需验证
    if sampling_params.structured_outputs is None:
        return
    tp, grm = get_structured_output_key(sampling_params.structured_outputs)
    guidance_grm = serialize_guidance_grammar(tp, grm)
    err = llguidance.LLMatcher.validate_grammar(guidance_grm, tokenizer)
    if err:
        raise ValueError(f"Grammar error: {err}")
