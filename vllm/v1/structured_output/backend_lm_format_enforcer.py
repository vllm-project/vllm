# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LM Format Enforcer 后端模块。

本模块实现了基于 LM Format Enforcer 的结构化输出后端，负责：
- 编译字符级解析器为 TokenEnforcer
- 处理 JSON Schema、正则表达式、选择列表约束
- 管理 token 位掩码

主要类：
- LMFormatEnforcerBackend: LM Format Enforcer 后端实现
- LMFormatEnforcerGrammar: LM Format Enforcer 文法实现

主要函数：
- validate_structured_output_request_lm_format_enforcer: 验证请求
"""

import ast
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from transformers import PreTrainedTokenizerBase

from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)

if TYPE_CHECKING:
    import lmformatenforcer
    import lmformatenforcer.integrations.vllm as lmfe_vllm
else:
    lmformatenforcer = LazyLoader("lmformatenforcer", globals(), "lmformatenforcer")
    lmfe_vllm = LazyLoader(
        "lmformatenforcer.integrations.vllm",
        globals(),
        "lmformatenforcer.integrations.vllm",
    )


@lru_cache
def _cached_build_vllm_token_enforcer_tokenizer_data(
    tokenizer: PreTrainedTokenizerBase, vocab_size: int
) -> "lmfe_vllm.TokenEnforcerTokenizerData":
    """构建并缓存 tokenizer 数据。

    使用 LRU 缓存避免重复构建相同的 tokenizer 数据。

    Args:
        tokenizer: 分词器
        vocab_size: 词表大小

    Returns:
        TokenEnforcerTokenizerData 对象
    """
    return lmfe_vllm.build_vllm_token_enforcer_tokenizer_data(
        tokenizer, use_bitmask=True, vocab_size=vocab_size
    )


@dataclass
class LMFormatEnforcerGrammar(StructuredOutputGrammar):
    """LM Format Enforcer 文法实现。

    封装 TokenEnforcer，提供文法约束功能。

    Attributes:
        token_enforcer: TokenEnforcer 实例
        current_tokens_prefix: 当前已接受的 token 前缀
    """

    token_enforcer: lmformatenforcer.TokenEnforcer
    current_tokens_prefix: list[int] = field(default_factory=list)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """接受 token 列表并推进解析器。

        Args:
            request_id: 请求 ID（未使用）
            tokens: token ID 列表

        Returns:
            如果所有 token 都被接受则返回 True，否则返回 False 并回滚
        """
        original_len = len(self.current_tokens_prefix)
        for token in tokens:
            if not self.token_enforcer.get_allowed_tokens(
                self.current_tokens_prefix
            ).is_token_allowed(token):
                # 回滚部分更新以确保原子性
                del self.current_tokens_prefix[original_len:]
                return False
            self.current_tokens_prefix.append(token)
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """验证 token 列表是否被接受（不推进解析器）。

        Args:
            tokens: token ID 列表

        Returns:
            被接受的前缀 token 列表
        """
        for prefix_length in range(len(tokens)):
            prefix = tokens[:prefix_length]
            next_token = tokens[prefix_length]
            if not self.token_enforcer.get_allowed_tokens(
                self.current_tokens_prefix + prefix
            ).is_token_allowed(next_token):
                break
        else:
            return tokens

        return tokens[:prefix_length]

    def rollback(self, num_tokens: int) -> None:
        """回退指定数量的 token。

        Args:
            num_tokens: 要回退的 token 数量
        """
        self.current_tokens_prefix = self.current_tokens_prefix[:-num_tokens]

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        """填充位掩码。

        Args:
            bitmask: 位掩码 tensor
            batch_index: batch 索引
        """
        allowed_tokens = self.token_enforcer.get_allowed_tokens(
            self.current_tokens_prefix
        )
        bitmask[batch_index] = allowed_tokens.allowed_tokens

    def is_terminated(self) -> bool:
        """检查是否已终止。

        如果前缀以 eos_token_id 结尾则视为终止。

        Returns:
            如果已终止则返回 True
        """
        # 如果前缀以 eos_token_id 结尾则视为终止
        return_value = (
            len(self.current_tokens_prefix) > 0
            and self.current_tokens_prefix[-1] == self.token_enforcer.eos_token_id
        )
        return return_value

    def reset(self):
        """重置状态。"""
        self.current_tokens_prefix = []


@dataclass
class LMFormatEnforcerBackend(StructuredOutputBackend):
    """LM Format Enforcer 后端实现。

    使用 lmformatenforcer 库实现结构化输出约束。
    支持 JSON Schema 和正则表达式。

    Attributes:
        vllm_config: vLLM 配置
        tokenizer: 分词器
        vocab_size: 词表大小
        tokenizer_data: 缓存的 tokenizer 数据
    """

    def __post_init__(self):
        """初始化后端。

        构建并缓存 tokenizer 数据。
        """
        self.tokenizer_data = _cached_build_vllm_token_enforcer_tokenizer_data(
            self.tokenizer, self.vocab_size
        )

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        """编译文法规范。

        Args:
            request_type: 结构化输出请求类型
            grammar_spec: 文法规范

        Returns:
            编译后的 LMFormatEnforcerGrammar

        Raises:
            ValueError: 如果不支持请求类型或有推测 token
        """
        character_level_parser: lmformatenforcer.CharacterLevelParser
        if request_type == StructuredOutputOptions.JSON:
            spec_dict = json.loads(grammar_spec)
            character_level_parser = lmformatenforcer.JsonSchemaParser(spec_dict)
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            character_level_parser = lmformatenforcer.JsonSchemaParser(None)
        elif request_type == StructuredOutputOptions.REGEX:
            character_level_parser = lmformatenforcer.RegexParser(grammar_spec)
        elif request_type == StructuredOutputOptions.CHOICE:
            choices = ast.literal_eval(grammar_spec)
            character_level_parser = lmformatenforcer.UnionParser(
                [lmformatenforcer.StringParser(choice) for choice in choices]
            )
        else:
            raise ValueError(
                f"Invalid request type for LM Format Enforcer backend({request_type!s})"
            )
        max_rollback_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )

        if max_rollback_tokens > 0:
            raise ValueError(
                "LM Format Enforcer backend does not support speculative tokens"
            )

        token_enforcer = lmformatenforcer.TokenEnforcer(
            tokenizer_data=self.tokenizer_data,
            parser=character_level_parser,
        )
        return LMFormatEnforcerGrammar(token_enforcer)

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

        LMFormatEnforcer 后端无需特殊清理。
        """
        pass


def validate_structured_output_request_lm_format_enforcer(params: SamplingParams):
    """验证 LM Format Enforcer 的结构化输出请求。

    Args:
        params: 采样参数

    Raises:
        ValueError: 如果请求无效或不支持
    """
    if params.structured_outputs is None:
        return

    so_params = params.structured_outputs

    if so_params.regex:
        return
    elif so_params.json:
        if isinstance(so_params.json, str):
            try:
                # 确保 schema 是有效的 JSON
                json.loads(so_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            try:
                json.dumps(so_params.json)
            except Exception as e:
                raise ValueError(
                    f"Error serializing structured outputs jsonschema: {e}"
                ) from e
        return
    elif so_params.choice:
        return
    elif so_params.grammar:
        raise ValueError(
            "LM Format Enforcer structured outputs backend "
            "does not support grammar specifications"
        )
