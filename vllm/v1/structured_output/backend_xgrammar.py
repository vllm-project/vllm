# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Xgrammar 后端模块。

本模块实现了基于 Xgrammar 的结构化输出后端，负责：
- 编译 JSON Schema、正则表达式、EBNF 文法为 GrammarMatcher
- 处理结构标签（structural tags）
- 管理 token 位掩码
- 支持 Mistral tokenizer 的特殊处理

主要类：
- XgrammarBackend: Xgrammar 后端实现
- XgrammarGrammar: Xgrammar 文法实现

主要函数：
- has_xgrammar_unsupported_json_features: 检查不支持的 JSON 功能
- validate_xgrammar_grammar: 验证文法请求
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.utils import (
    choice_as_grammar,
    convert_lark_to_ebnf,
    grammar_is_likely_lark,
)

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


# Xgrammar 支持的字符串格式列表
STRING_SUPPORTED_FORMATS = {
    "email",
    "date",
    "time",
    "date-time",
    "duration",
    "ipv4",
    "ipv6",
    "hostname",
    "uuid",
    "uri",
    "uri-reference",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
}


@dataclass
class XgrammarBackend(StructuredOutputBackend):
    """Xgrammar 后端实现。

    使用 xgrammar 库实现结构化输出约束。
    支持 JSON Schema、正则表达式、EBNF 文法和结构标签。

    Attributes:
        vllm_config: vLLM 配置
        tokenizer: 分词器
        vocab_size: 词表大小
        disable_any_whitespace: 是否禁用任意空白
        compiler: GrammarCompiler 实例
        num_speculative_tokens: 推测 token 数量
    """

    def __post_init__(self):
        """初始化后端。

        根据 tokenizer 类型创建 TokenizerInfo，并初始化编译器。
        """
        self.disable_any_whitespace = (
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        )

        if is_mistral_tokenizer(self.tokenizer):
            # NOTE: 理想情况下，xgrammar 应该相应地处理这个
            # 参考：https://github.com/mlc-ai/xgrammar/blob/d77c0a0173ef14779c918e3be7966ba852f7910f/python/xgrammar/tokenizer_info.py#L98
            stop_token_ids = [self.tokenizer.eos_token_id]

            # 不使用 self.tokenizer.vocab_size，因为 self.tokenizer.vocab
            # 会将所有解码错误折叠为单个 token
            self.vocab_size = len(self.tokenizer.vocab)
            tokenizer_info = xgr.TokenizerInfo(  # type: ignore
                encoded_vocab=self.tokenizer.vocab,
                # NOTE: https://github.com/mlc-ai/xgrammar/blob/5e141f6ff1ca02bc31f9e512e68b61f2a8ae88e5/tests/python/test_tokenizer_info.py#L43
                vocab_type=xgr.VocabType.RAW
                if self.tokenizer.is_tekken
                else xgr.VocabType.BYTE_FALLBACK,
                vocab_size=self.vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=True,
            )
        else:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.vocab_size,
            )
        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
            cache_limit_bytes=vllm.envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024,
        )

        self.num_speculative_tokens = 0
        if self.vllm_config.speculative_config is not None:
            self.num_speculative_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        """编译文法规范。

        Args:
            request_type: 结构化输出请求类型
            grammar_spec: 文法规范

        Returns:
            编译后的 XgrammarGrammar

        Raises:
            ValueError: 如果不支持请求类型
        """
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(
                grammar_spec, any_whitespace=not self.disable_any_whitespace
            )
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_json_schema(
                '{"type": "object"}', any_whitespace=not self.disable_any_whitespace
            )
        elif request_type == StructuredOutputOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = self.compiler.compile_regex(grammar_spec)
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            s_tag = json.loads(grammar_spec)
            if "structures" in s_tag:
                # 回退到已弃用的结构标签编译方法
                tags = [
                    xgr.StructuralTagItem(
                        begin=s["begin"],
                        schema=json.dumps(s["schema"]),
                        end=s["end"],
                    )
                    for s in s_tag["structures"]
                ]
                ctx = self.compiler.compile_structural_tag(tags, s_tag["triggers"])
            else:
                ctx = self.compiler.compile_structural_tag(grammar_spec)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})"
            )

        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(
                ctx,
                max_rollback_tokens=self.num_speculative_tokens,
            ),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        """分配 token 位掩码。

        Args:
            max_num_seqs: 最大序列数

        Returns:
            分配的位掩码 tensor
        """
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)

    def destroy(self):
        """清理后端资源。

        删除编译器以释放内存。
        """
        del self.compiler


@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    """Xgrammar 文法实现。

    封装 xgrammar.GrammarMatcher，提供文法约束功能。
    这是一个通用的文法类，未来可支持不同的后端。

    Attributes:
        vocab_size: 词表大小
        matcher: GrammarMatcher 实例
        ctx: CompiledGrammar 实例
        num_processed_tokens: 已处理的 token 数量
        _is_terminated: 是否已终止
    """

    # NOTE: 这是一个通用类，未来可支持不同的后端
    # 目前仅支持 xgrammar
    #
    # 参考 jump-forward 解码：
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string

    vocab_size: int
    matcher: xgr.GrammarMatcher = field(hash=False)
    ctx: xgr.CompiledGrammar = field(hash=False)
    num_processed_tokens: int = field(
        default_factory=lambda: 0, repr=False, hash=False, init=False
    )
    _is_terminated: bool = field(default=False, repr=False, hash=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """接受 token 列表并推进 FSM。

        Args:
            request_id: 请求 ID
            tokens: token ID 列表

        Returns:
            如果 FSM 成功推进则返回 True，否则返回 False
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """按顺序检查 token 列表是否被 FSM 接受。

        不会推进 FSM。

        Args:
            tokens: token ID 列表

        Returns:
            被 FSM 接受的前缀 token 列表
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            # 回滚 FSM 到初始状态
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        """回退指定数量的 token。

        Args:
            num_tokens: 要回退的 token 数量
        """
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        """填充位掩码。

        Args:
            bitmask: 位掩码 tensor
            idx: batch 索引
        """
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        """检查是否已终止。

        Returns:
            如果已终止则返回 True
        """
        return self._is_terminated

    def reset(self):
        """重置状态。"""
        self.num_processed_tokens = 0
        self.matcher.reset()


def has_xgrammar_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """检查 JSON Schema 是否包含 xgrammar 不支持的功能。

    不支持的功能：
    - 数值类型：multipleOf
    - 数组类型：uniqueItems, contains, minContains, maxContains
    - 字符串类型：format 不在支持列表中
    - 对象类型：patternProperties, propertyNames

    Args:
        schema: JSON Schema 对象

    Returns:
        如果包含不支持的功能则返回 True，否则返回 False
    """

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        # 检查数值范围
        if obj.get("type") in ("integer", "number") and ("multipleOf" in obj):
            return True

        # 检查数组不支持的关键字
        if obj.get("type") == "array" and any(
            key in obj
            for key in ("uniqueItems", "contains", "minContains", "maxContains")
        ):
            return True

        # 字符串不支持的关键字
        if (
            obj.get("type") == "string"
            and "format" in obj
            and obj["format"] not in STRING_SUPPORTED_FORMATS
        ):
            return True

        # 对象不支持的关键字
        if obj.get("type") == "object" and any(
            key in obj for key in ("patternProperties", "propertyNames")
        ):
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


def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None:
    """验证结构化输出请求是否受支持。

    检查：
    - 正则表达式：是否可有效解析
    - 选择列表：是否可转换为 EBNF 文法
    - JSON Schema：是否包含不支持的功能
    - EBNF 文法：是否可有效解析（必要时从 Lark 转换）
    - 结构标签：是否有效

    Args:
        sampling_params: 采样参数

    Raises:
        ValueError: 如果请求不受支持
    """
    if sampling_params.structured_outputs is None:
        return

    so_params = sampling_params.structured_outputs

    if so_params.regex:
        try:
            xgr.Grammar.from_regex(so_params.regex)
        except Exception as err:
            raise ValueError(
                f"Failed to transform regex into a grammar: {err}"
            ) from err

    if so_params.choice:
        choice_grammar = choice_as_grammar(so_params.choice)
        try:
            xgr.Grammar.from_ebnf(choice_grammar)
        except Exception as err:
            raise ValueError(
                "Failed to transform choices into a grammar: {err}"
            ) from err
        so_params.choice = None
        so_params.grammar = choice_grammar
        return

    if so_params.json:
        if isinstance(so_params.json, str):
            try:
                schema = json.loads(so_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            schema = so_params.json

        if has_xgrammar_unsupported_json_features(schema):
            raise ValueError(
                "The provided JSON schema contains features not supported by xgrammar."
            )

        try:
            xgr.Grammar.from_json_schema(schema)
        except Exception as err:
            raise ValueError(
                f"Failed to transform json schema into a grammar: {err}"
            ) from err
        return

    if so_params.grammar:
        if grammar_is_likely_lark(so_params.grammar):
            # xgrammar 仅支持 EBNF 文法
            try:
                so_params.grammar = convert_lark_to_ebnf(so_params.grammar)
            except ValueError as e:
                raise ValueError(
                    "Failed to convert the grammar from Lark to EBNF. "
                ) from e

        # 解析 EBNF 文法（可能已从 Lark 转换）
        try:
            # 解析文法，但不编译
            xgr.Grammar.from_ebnf(so_params.grammar)
        except Exception as e:
            raise ValueError("Invalid grammar specification.") from e
        return

    if so_params.structural_tag:
        try:
            s_tag = json.loads(so_params.structural_tag)

            # 使用已弃用的结构标签编译方法
            if "structures" in s_tag:
                tags = [
                    xgr.StructuralTagItem(
                        begin=s["begin"],
                        schema=json.dumps(s["schema"]),
                        end=s["end"],
                    )
                    for s in s_tag["structures"]
                ]
                xgr.Grammar.from_structural_tag(tags, s_tag["triggers"])
            else:
                xgr.Grammar.from_structural_tag(so_params.structural_tag)
        except Exception as e:
            raise ValueError("Invalid structural tag specification.") from e
