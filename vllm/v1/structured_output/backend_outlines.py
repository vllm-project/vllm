# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Outlines 后端模块。

本模块实现了基于 Outlines/outlines_core 的结构化输出后端，负责：
- 编译正则表达式为 DFA 索引
- 缓存已编译的索引
- 管理 token 位掩码

主要类：
- OutlinesBackend: Outlines 后端实现
- OutlinesGrammar: Outlines 文法实现

主要函数：
- validate_structured_output_request_outlines: 验证请求
- validate_regex_is_buildable: 验证正则表达式是否可编译
"""

from __future__ import annotations

import ast
import importlib
import json
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from regex import escape as regex_escape

from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.utils import (
    OutlinesVocabulary,
    get_outlines_cache,
    get_outlines_vocabulary,
)

if TYPE_CHECKING:
    import outlines_core as oc
    import outlines_core.json_schema as json_schema
else:
    oc = LazyLoader("oc", globals(), "outlines_core")
    json_schema = LazyLoader("json_schema", globals(), "outlines_core.json_schema")

# Python 3.11+ sre_parse 和 sre_constants 已弃用
# 必须从 re 模块导入
if sys.version_info >= (3, 11):
    # Hack：绕过 pre-commit 的 regex 模块规则
    # 因为从 re 导入是 Python 3.11+ 获取 sre_parse 和 sre_constants 的唯一方式
    _re = importlib.import_module("re")
    sre_parse = _re._parser
    sre_constants = _re._constants
else:
    import sre_constants
    import sre_parse


@dataclass
class OutlinesBackend(StructuredOutputBackend):
    """Outlines 后端实现。

    使用 outlines_core 库实现结构化输出约束。
    支持 JSON Schema、正则表达式和选择列表。

    Attributes:
        vllm_config: vLLM 配置
        tokenizer: 分词器
        vocab_size: 词表大小
        vocabulary: Outlines 词表
        cache: 索引缓存
    """

    def __post_init__(self):
        """初始化后端。

        获取词表和缓存实例。
        """
        self.vocabulary = get_outlines_vocabulary(self.tokenizer)
        self.cache = get_outlines_cache()

    def _compile_index(
        self, regex_string: str, vocabulary: OutlinesVocabulary
    ) -> oc.Index:
        """编译正则表达式为索引。

        使用缓存避免重复编译相同的正则表达式。

        Args:
            regex_string: 正则表达式字符串
            vocabulary: 词表

        Returns:
            编译后的索引
        """
        cache_key = f"{vocabulary._hash}_{regex_string}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        index = oc.Index(regex_string, vocabulary.inner)
        self.cache[cache_key] = index

        return index

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        """编译文法规范。

        Args:
            request_type: 结构化输出请求类型
            grammar_spec: 文法规范

        Returns:
            编译后的 OutlinesGrammar

        Raises:
            ValueError: 如果不支持请求类型
        """
        if request_type == StructuredOutputOptions.JSON:
            regex = json_schema.build_regex_from_schema(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            regex = grammar_spec
        elif request_type == StructuredOutputOptions.CHOICE:
            choices = ast.literal_eval(grammar_spec)
            choices = [regex_escape(c) for c in choices]
            regex = "(" + "|".join(choices) + ")"
        else:
            raise ValueError(
                f"Invalid request type for Outlines backend ({request_type!s})"
            )
        index = self._compile_index(regex, self.vocabulary)
        max_rollback_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )
        return OutlinesGrammar(
            vocab_size=self.vocab_size,
            guide=oc.Guide(index, max_rollback=max_rollback_tokens),
        )

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        """分配 token 位掩码。

        Args:
            max_num_seqs: 最大序列数

        Returns:
            分配的位掩码 tensor
        """
        return torch.full(
            (max_num_seqs, (self.vocab_size + 31) // 32),
            -1,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )

    def destroy(self):
        """清理后端资源。

        Outlines 后端无需特殊清理。
        """
        pass


@dataclass
class OutlinesGrammar(StructuredOutputGrammar):
    """Outlines 文法实现。

    封装 outlines_core.Guide，提供文法约束功能。

    Attributes:
        vocab_size: 词表大小
        guide: outlines_core Guide 实例
        num_processed_tokens: 已处理的 token 数量
        _prev_finished: 上一次是否已完成（用于延迟终止标志）
    """

    vocab_size: int
    guide: oc.Guide = field(hash=False)
    num_processed_tokens: int = field(
        default_factory=lambda: 0, repr=False, hash=False, init=False
    )

    # outlines_core 在 DFA 接受时标记完成；vLLM 期望在 EOS 后标记完成
    # 我们将完成标志延迟一步，以便仍能发出 EOS token
    _prev_finished: bool = field(default=False, init=False, repr=False, hash=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """接受 token 列表并推进 FSM。

        Args:
            request_id: 请求 ID（未使用）
            tokens: token ID 列表

        Returns:
            如果 FSM 成功推进则返回 True，否则返回 False
        """
        if self.guide.accepts_tokens(tokens):
            # 当使用当前 token 推进后到达死状态时，advance 会失败
            # 这是因为 Guide.accepts_tokens() 只检查当前 token 是否可接受
            # 而 guide.advance() 还会检查接受所有 token 后的下一个状态
            # 需要注意 FSM 必须在不包含死状态的情况下准备
            for t in tokens:
                self.guide.advance(t)
                self.num_processed_tokens += 1
            return True
        return False

    def rollback(self, num_tokens: int) -> None:
        """回退指定数量的 token。

        Args:
            num_tokens: 要回退的 token 数量
        """
        self.guide.rollback_state(num_tokens)
        self.num_processed_tokens -= num_tokens

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """验证 token 列表是否被接受（不推进 FSM）。

        Args:
            tokens: token ID 列表

        Returns:
            被接受的前缀 token 列表
        """
        accepted: list[int] = []
        for tok in tokens:
            accepted.append(tok)
            if not self.guide.accepts_tokens(accepted):
                accepted.pop()
                break
        return accepted

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        """填充位掩码。

        Args:
            bitmask: 位掩码 tensor
            idx: batch 索引
        """
        mask = bitmask[idx]
        self.guide.write_mask_into(mask.data_ptr(), mask.numel(), mask.element_size())

    def is_terminated(self) -> bool:
        """检查是否已终止。

        延迟一步返回完成标志，以便 EOS token 能被正确发出。

        Returns:
            如果已终止则返回 True
        """
        curr = self.guide.is_finished()
        prev = self._prev_finished
        self._prev_finished = curr
        return prev

    def reset(self):
        """重置状态。"""
        self.num_processed_tokens = 0
        self._prev_finished = False
        self.guide.reset()


def validate_structured_output_request_outlines(params: SamplingParams):
    """验证 Outlines 的结构化输出请求。

    Args:
        params: 采样参数

    Raises:
        ValueError: 如果请求无效或不支持
    """
    if params.structured_outputs is None:
        return

    so_params = params.structured_outputs

    if so_params.regex:
        validate_regex_is_buildable(so_params.regex)
    elif so_params.json:
        if isinstance(so_params.json, str):
            try:
                # 确保 schema 是有效的 JSON
                json.loads(so_params.json)
                schema = so_params.json
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            try:
                schema = json.dumps(so_params.json)
            except Exception as e:
                raise ValueError(
                    f"Error serializing structured outputs jsonschema: {e}"
                ) from e
        pattern = json_schema.build_regex_from_schema(schema)
        validate_regex_is_buildable(pattern)
    elif so_params.choice:
        choices = [regex_escape(str(choice)) for choice in so_params.choice]
        regex = "(" + "|".join(choices) + ")"
        validate_regex_is_buildable(regex)
    elif so_params.grammar:
        raise ValueError(
            "Outlines structured outputs backend "
            "does not support grammar specifications"
        )


def _prefix_needs_context(parsed) -> bool:
    """检查正则表达式前缀是否需要上下文（锚点或环视）。

    如果在任何消费字符之前存在环视或锚点，则返回 True。

    Args:
        parsed: 已解析的正则表达式

    Returns:
        如果前缀需要上下文则返回 True
    """

    def subpattern_consumes(parsed) -> bool:
        """检查子模式是否能消费至少一个字符。

        Args:
            parsed: 已解析的子模式

        Returns:
            如果能消费字符则返回 True
        """
        tokens = parsed.data if hasattr(parsed, "data") else parsed
        for ttype, tval in tokens:
            # 字面量、字符类或点号总是消费
            if ttype in (sre_parse.LITERAL, sre_parse.IN, sre_parse.ANY):
                return True
            # 量化的子模式：检查内部模式
            elif ttype == sre_parse.MAX_REPEAT:
                _, mx, sub = tval
                if mx != 0 and subpattern_consumes(sub):
                    return True
            # 交替：如果任何分支消费，则整体消费
            elif ttype == sre_parse.BRANCH:
                _, branches = tval
                if any(subpattern_consumes(br) for br in branches):
                    return True
            # 分组子模式：递归检查内容
            elif ttype == sre_parse.SUBPATTERN and subpattern_consumes(tval[3]):
                return True
        # 没有消费者，返回 False
        return False

    tokens = parsed.data if hasattr(parsed, "data") else parsed
    for ttype, tval in tokens:
        # 直接锚点或环视
        if ttype == sre_parse.AT or ttype in (
            sre_constants.ASSERT,
            sre_constants.ASSERT_NOT,
        ):
            return True

        # 嵌套子模式：递归检查
        if ttype == sre_parse.SUBPATTERN:
            # tval: (group, add_flags, del_flags, subpattern)
            if _prefix_needs_context(tval[3]):
                return True
            if subpattern_consumes(tval[3]):
                return False

        # 交替分支：如果任何分支有前缀锚点 => True
        # 否则如果至少一个分支消费 => 前缀结束 => False
        elif ttype == sre_parse.BRANCH:
            saw_consumer = False
            for br in tval[1]:
                if _prefix_needs_context(br):
                    return True
                if subpattern_consumes(br):
                    saw_consumer = True
            if saw_consumer:
                return False

        # 直接消费 token
        elif ttype in (sre_parse.LITERAL, sre_parse.IN, sre_parse.ANY):
            return False

        # 如果子模式有锚点 => True，如果能消费 => 停止
        elif ttype == sre_parse.MAX_REPEAT:
            if _prefix_needs_context(tval[2]):
                return True
            if subpattern_consumes(tval[2]):
                return False

    return False


def _check_unsupported(parsed) -> None:
    """检查正则表达式是否使用了 regex-automata 不支持的功能。

    不支持的功能：
    - 后向引用（backreferences）
    - 环视断言（look-around assertions）
    - Unicode 单词边界

    Args:
        parsed: 已解析的正则表达式

    Raises:
        ValueError: 如果使用了不支持的功能
    """
    tokens = parsed.data if hasattr(parsed, "data") else parsed
    for ttype, tval in tokens:
        # 后向引用
        if ttype in (sre_parse.GROUPREF, sre_parse.GROUPREF_EXISTS):
            raise ValueError("Backreferences are unsupported.")

        # 环视断言
        elif ttype in (sre_constants.ASSERT, sre_constants.ASSERT_NOT):
            raise ValueError("Look-Around assertion are unsupported.")

        # Unicode 单词边界
        elif ttype == sre_parse.AT:
            if tval in (sre_constants.AT_BOUNDARY, sre_constants.AT_NON_BOUNDARY):
                raise ValueError("Unicode word boundaries are unsupported.")

        elif ttype == sre_parse.BRANCH:
            # tval 是 (None, branches)
            for branch in tval[1]:
                _check_unsupported(branch)

        # tval 是 (min, max, subpattern)
        elif ttype == sre_parse.MAX_REPEAT:
            _check_unsupported(tval[2])


def validate_regex_is_buildable(pattern: str) -> None:
    """验证正则表达式是否可编译为 DFA。

    检查：
    1. 正则表达式语法是否有效
    2. 是否使用了不支持的功能
    3. 是否有通用的起始状态（universal start state）

    Args:
        pattern: 正则表达式模式

    Raises:
        ValueError: 如果正则表达式无效或不可编译
    """
    try:
        parsed = sre_parse.parse(pattern)

    except sre_constants.error as e:
        raise ValueError(f"Error parsing regex: {e}") from e

    try:
        _check_unsupported(parsed)
    except ValueError as e:
        raise ValueError(
            f"Regex uses unsupported feature for structured outputs: {e}. "
            "Only basic matching constructs are supported—lookarounds, "
            "backreferences, and unicode boundaries are not."
        ) from e

    if _prefix_needs_context(parsed):
        raise ValueError(
            "Regex does not have a anchored universal start state. "
            "This means that the Regex uses anchors (^) or look-arounds "
            "in a way which requires context before any token is matched. "
            "Structured outputs needs regexes that can match without needing "
            "that context. Try rewriting the pattern without using these "
            f"constructs. Pattern:\n{pattern}"
        )
