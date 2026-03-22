# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""结构化输出工具函数模块。

本模块提供结构化输出的辅助函数，负责：
- 应用文法位掩码到输出 logits
- 管理 Outlines 词表缓存
- 处理 Lark 到 EBNF 文法转换
- 生成选择文法

主要函数：
- apply_grammar_bitmask: 应用文法位掩码到 logits
- get_outlines_vocabulary: 获取 Outlines 词表
- convert_lark_to_ebnf: 转换 Lark 文法到 EBNF 格式
- choice_as_grammar: 将选择列表转换为 EBNF 文法
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import regex as re
import torch
from cachetools import LRUCache
from diskcache import Cache

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.import_utils import LazyLoader
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

if TYPE_CHECKING:
    import outlines_core as oc
    import transformers.convert_slow_tokenizer as convert_slow_tokenizer
    import transformers.file_utils as file_utils
    import xgrammar as xgr

    from vllm.tokenizers import TokenizerLike
    from vllm.v1.worker.gpu_input_batch import InputBatch
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    oc = LazyLoader("oc", globals(), "outlines_core")
    file_utils = LazyLoader("file_utils", globals(), "transformers.file_utils")
    convert_slow_tokenizer = LazyLoader(
        "convert_slow_tokenizer", globals(), "transformers.convert_slow_tokenizer"
    )


logger = init_logger(__name__)

CACHE = None


def apply_grammar_bitmask(
    scheduler_output: SchedulerOutput,
    grammar_output: GrammarOutput,
    input_batch: InputBatch,
    logits: torch.Tensor,
) -> None:
    """应用文法位掩码到模型输出 logits。

    使用 xgrammar 函数将结构化输出的位掩码应用到 logits 上，
    限制模型只能生成符合文法的 token。

    处理流程：
    1. 从 scheduler 获取结构化输出的位掩码
    2. 根据 batch 中的请求顺序重新排序掩码
    3. 处理 speculative tokens 的偏移
    4. 调用 xgrammar 应用掩码到 logits

    Args:
        scheduler_output: 调度器输出
        grammar_output: 文法输出（包含位掩码）
        input_batch: 模型输入批次
        logits: 模型前向输出

    Note:
        接收的 bitmask 是 numpy 数组而非 tensor，因为序列化效率更高
    """
    # numpy 数组的序列化比 tensor 更高效，所以接收该格式
    grammar_bitmask = grammar_output.grammar_bitmask

    # 从调度器接收结构化输出位掩码，
    # 压缩为仅包含结构化输出请求的位掩码
    # 位掩码中请求的顺序不一定与 GPU runner batch 中的顺序相同
    # 需要排序位掩码以匹配此处使用的请求顺序

    # 获取 batch 中结构化输出请求的索引
    # 跟踪 batch 中每个请求的 speculative token 数量
    # logit 索引会因此偏移
    struct_out_req_batch_indices: dict[str, int] = {}
    cumulative_offset = 0
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    struct_out_req_ids = set(grammar_output.structured_output_request_ids)
    for batch_index, req_id in enumerate(input_batch.req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(spec_tokens.get(req_id, ()))
        if req_id in struct_out_req_ids:
            struct_out_req_batch_indices[req_id] = logit_index

    out_indices = []

    # 重新排序位掩码以匹配 batch 中请求的顺序
    sorted_bitmask = np.full(
        shape=(logits.shape[0], grammar_bitmask.shape[1]),
        fill_value=-1,
        dtype=grammar_bitmask.dtype,
    )
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(spec_tokens.get(req_id, ()))
        if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
            for i in range(1 + num_spec_tokens):
                bitmask_index = logit_idx + i
                sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index + i]
                out_indices.append(bitmask_index)
        cumulative_index += 1 + num_spec_tokens

    # 异步复制到设备 tensor
    grammar_bitmask = torch.from_numpy(sorted_bitmask).to(
        logits.device, non_blocking=True
    )

    # 如果 out_indices 长度与 logits 形状相同，
    # 则无需传递 indices，因为 bitmask 已与 logits 对齐
    skip_out_indices = len(out_indices) == logits.shape[0]

    index_tensor = None
    if not skip_out_indices:
        # xgrammar 期望 Python 列表，但使用 tensor 也能工作
        # 此处手动复制可非阻塞方式完成，xgrammar 内不会有 CPU 同步
        index_tensor = torch.tensor(
            out_indices, dtype=torch.int32, device="cpu", pin_memory=True
        )
        index_tensor = index_tensor.to(logits.device, non_blocking=True)

    # 处理 CPU 的 dtype 转换（旧版 xgrammar CPU kernel 需要 float32）
    # 参考：https://github.com/vllm-project/vllm/issues/31901
    if logits.device.type == "cpu" and logits.dtype != torch.float32:
        # 转换为 float32，应用位掩码，然后转换回来
        logits_float32 = logits.to(torch.float32)
        xgr.apply_token_bitmask_inplace(
            logits_float32, grammar_bitmask, indices=index_tensor
        )
        # 将修改后的值复制回原始 tensor
        logits.copy_(logits_float32.to(logits.dtype))
    else:
        xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)


class OutlinesVocabulary:
    """Outlines 词表的包装类。

    封装 outlines_core.Vocabulary，允许存储词表的哈希值用于缓存。

    Attributes:
        inner: 实际的词表对象
        _hash: 词表内容的 SHA256 哈希值（用于缓存键）
    """

    def __init__(self, vocabulary: oc.Vocabulary) -> None:
        # 实际词表对象
        self.inner = vocabulary
        # 必须使用 abs(hash()) 因为 Python 哈希可能为负数
        # 我们使用哈希作为缓存键
        hex_str = hashlib.sha256(vocabulary.__repr__().encode("utf-8")).hexdigest()
        hash_int = int(hex_str, 16)
        self._hash = hash_int


def get_outlines_cache_path() -> str:
    """获取 Outlines 缓存目录路径。

    按优先级顺序查找：
    1. OUTLINES_CACHE_DIR 环境变量
    2. XDG_CACHE_HOME/.cache/outlines
    3. ~/.cache/outlines（Unix 默认）
    4. /tmp/.cache/outlines（容器环境回退）

    Returns:
        缓存目录路径
    """
    outlines_cache_dir = os.getenv("OUTLINES_CACHE_DIR")
    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    home_dir = os.path.expanduser("~")

    if outlines_cache_dir:
        # OUTLINES_CACHE_DIR 优先级最高
        return outlines_cache_dir
    if xdg_cache_home:
        return os.path.join(xdg_cache_home, ".cache", "outlines")
    # 如果 home 目录是"/"，可能在容器内，写入根目录会有问题
    # 回退到使用临时目录
    # 验证路径存在，因为 os.path.expanduser 不保证存在
    if os.path.isdir(home_dir) and home_dir != "/":
        # 默认 Unix 回退：~/.cache/outlines
        return os.path.join(home_dir, ".cache", "outlines")

    # home_dir 在 Docker 容器内可能是 /（没有现有用户）
    tempdir = tempfile.gettempdir()
    return os.path.join(tempdir, ".cache", "outlines")


def get_outlines_cache():
    """获取用于索引缓存的 Cache 实例。

    如果启用了 VLLM_V1_USE_OUTLINES_CACHE 环境变量：
    - 创建磁盘上的无界缓存（警告：可能消耗大量磁盘空间）
    - 检查 outlines_core 版本，版本变化时清除缓存

    否则：
    - 返回最大 128 项的 LRUCache

    Returns:
        Cache 或 LRUCache 实例
    """
    cache_dir = get_outlines_cache_path()
    if envs.VLLM_V1_USE_OUTLINES_CACHE:
        logger.warning(
            "Enabling outlines cache. This is an unbounded on-disk "
            "cache. It may consume a lot of disk space and should "
            "not be used with untrusted clients."
        )
        cache = Cache(cache_dir, eviction_policy="none", cull_limit=0)
        outlines_version = importlib.metadata.version("outlines_core")

        cached_version = cache.get("__version__", None)
        if cached_version != outlines_version:
            cache.clear()
        cache.set("__version__", outlines_version)
        return cache

    return LRUCache(maxsize=128)


# 匹配 Llama 字节 token 的正则表达式，如 <0x41>
re_llama_byte_token = re.compile(r"^<0x[0-9A-F]{2}>$")
# 匹配包含 replacement character 的序列
re_replacement_seq = re.compile(r"^.{0,6}+.{0,6}$")


def _reduced_vocabulary(tokenizer: TokenizerLike) -> dict[bytes, list[int]]:
    """创建从词表 token 到等价 token ID 列表的映射。

    处理各种 tokenizer 的特殊情况：
    - 特殊 token 被跳过
    - BPE tokenizer 的字节 token
    - 无效 UTF-8 序列的处理
    - Llama tokenizers 的前导空格处理

    Args:
        tokenizer: 分词器

    Returns:
        Dict[token 字节串 -> 等价 token ID 列表]
    """
    eos_token_id = tokenizer.eos_token_id

    # 构建 unicode 到字节的映射
    unicode_to_bytes = {
        v: k for k, v in convert_slow_tokenizer.bytes_to_unicode().items()
    }

    def convert_token_to_string(token: str) -> str:
        """将 token 转换为字符串表示。

        Args:
            token: 输入 token

        Returns:
            转换后的字符串
        """
        string = tokenizer.convert_tokens_to_string([token])

        # Hack：处理 HF Llama tokenizers 缺失的空格
        if (
            type(token) is str
            and token.startswith(file_utils.SPIECE_UNDERLINE)
            or token == "<0x20>"
        ):
            return " " + string

        return string

    vocabulary: dict[bytes, list[int]] = {}
    empty_token_ids: list[int] = []
    for token, token_idx in tokenizer.get_vocab().items():
        if token in tokenizer.all_special_tokens:
            continue

        token_str = convert_token_to_string(token)
        if token_str:
            if isinstance(token, (bytes, bytearray)):
                # BPE tokenizer：token 以字节形式存储
                # 此时 token_str 也是字节类型，可以安全转换
                token_bytes = bytes(token_str)  # type: ignore[arg-type]

            elif (token_str == "\ufffd" and token != "\ufffd") or (
                "\ufffd" in token_str and not re_replacement_seq.match(token_str)
            ):
                # 处理无效 UTF-8 序列
                if re_llama_byte_token.match(token):
                    # Llama tokenizer：使用 <0xXX> 表示不完整序列
                    token_bytes = bytes([int(token[3:5], 16)])
                else:
                    # GPT2 tokenizer：使用 unicode_to_bytes 映射每个字符
                    byte_vals = [unicode_to_bytes.get(c) for c in token]
                    if None in byte_vals:
                        raise RuntimeError(
                            f"Cannot convert token `{token}`"
                            f" ({token_idx}) to bytes: {token_str}"
                        )
                    # 如果 byte_vals 中有 None，会抛出错误，此处可以安全转换
                    token_bytes = bytes(byte_vals)  # type: ignore[arg-type]
            else:
                token_bytes = token_str.encode("utf-8")

            if token_idx != eos_token_id:
                vocabulary.setdefault(token_bytes, []).append(token_idx)
        else:
            empty_token_ids.append(token_idx)

    return vocabulary


def get_outlines_vocabulary(tokenizer: TokenizerLike) -> oc.Vocabulary:
    """获取给定 tokenizer 的 Outlines Vocabulary 对象。

    使用缓存机制避免重复计算。

    Args:
        tokenizer: 分词器

    Returns:
        Outlines Vocabulary 对象
    """
    if hasattr(tokenizer, "_outlines_vocabulary"):
        return tokenizer._outlines_vocabulary  # type: ignore

    reduced_vocab = _reduced_vocabulary(tokenizer)
    vocabulary = OutlinesVocabulary(
        oc.Vocabulary(tokenizer.eos_token_id, reduced_vocab)
    )
    tokenizer._outlines_vocabulary = vocabulary  # type: ignore

    return vocabulary


def grammar_is_likely_lark(grammar_str: str) -> bool:
    """检查文法是否看起来是 Lark 语法。

    Lark 使用 `:` 作为规则定义符，EBNF 使用 `::=`。

    Args:
        grammar_str: 输入文法字符串

    Returns:
        如果文法看起来是 Lark 格式则返回 True，否则返回 False

    Examples:
        >>> grammar_is_likely_lark("rule: 'abc'")
        True
        >>> grammar_is_likely_lark("rule ::= 'abc'")
        False
    """
    if not grammar_str or not isinstance(grammar_str, str):
        return False

    for line in grammar_str.split("\n"):
        # 移除两种注释风格
        line = re.sub(r"(#|//).*$", "", line).strip()
        if not line:
            continue

        # 查找 EBNF 规则定义符
        if "::=" in line:
            return False

    return True


def convert_lark_to_ebnf(grammar_str: str) -> str:
    """将 Lark 文法字符串转换为 EBNF 格式。

    EBNF 参考：
    https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
    Lark 文法参考：
    https://lark-parser.readthedocs.io/en/latest/grammar.html

    转换规则：
    - 将 Lark 的 `:` 转换为 `::=`
    - 将单引号字符串转换为双引号
    - 将 `|` 行转换为 EBNF 的替代形式
    - 添加 root 规则指向第一个定义的规则（或 start 规则）

    Args:
        grammar_str: Lark 格式的输入文法

    Returns:
        EBNF 格式的文法字符串

    Raises:
        ValueError: 如果输入格式无效

    Examples:
        >>> print(convert_lark_to_ebnf("rule: 'hello'"))
        root ::= rule
        rule ::= "hello"
    """
    if not isinstance(grammar_str, str):
        raise ValueError(f"Grammar must be a string, got {type(grammar_str)}")
    if not grammar_str.strip():
        raise ValueError("Grammar string cannot be empty")

    defined_rules = set()
    referenced_rules = set()
    output_lines = []

    def clean_line(line: str) -> str:
        """移除行中的注释和空白。

        Args:
            line: 输入行

        Returns:
            清理后的行
        """
        return re.sub(r"(#|//).*$", "", line).strip()

    def check_quotes(text: str, rule_name: str, line_num: int) -> None:
        """验证引号匹配。

        Args:
            text: 要检查的文本
            rule_name: 规则名称
            line_num: 行号

        Raises:
            ValueError: 如果引号不匹配
        """
        if text.count("'") % 2 != 0 or text.count('"') % 2 != 0:
            raise ValueError(f"Mismatched quotes in {rule_name} on line {line_num}")

    def extract_references(text: str) -> set[str]:
        """从文本中提取规则引用。

        Args:
            text: 输入文本

        Returns:
            规则名称集合
        """
        # 移除引号字符串和特殊字符
        text = re.sub(r'"[^"]*"', "", text)
        text = re.sub(r"[+*?()|\[\]{}]", " ", text)
        return set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text))

    # 第一遍：查找根规则并验证规则定义
    lines = [clean_line(line) for line in grammar_str.split("\n")]
    first_rule = None

    for line_num, line in enumerate(lines, 1):
        if not line or line.startswith("|"):
            continue

        if ":" in line:
            try:
                name = line.split(":", 1)[0].strip().strip("?")
                defined_rules.add(name)
                if first_rule is None:
                    first_rule = name
                if name == "start":
                    first_rule = "start"
            except IndexError as e:
                raise ValueError(
                    f"Invalid rule format on line {line_num}. "
                    "Expected 'rule_name: definition'"
                ) from e

    if not defined_rules:
        raise ValueError("No valid rules found in grammar")

    # 添加根规则
    output_lines.append(f"root ::= {first_rule}")

    # 第二遍：处理规则定义和替代规则
    current_rule = None
    current_definition = []

    for line_num, line in enumerate(lines, 1):
        if not line:
            continue

        try:
            if ":" in line and not line.startswith("|"):
                # 保存之前的规则
                if current_rule:
                    output_lines.append(
                        f"{current_rule} ::= {' | '.join(current_definition)}"
                    )

                # 处理新规则
                name, definition = line.split(":", 1)
                current_rule = name.strip().strip("?")

                check_quotes(definition, f"rule '{current_rule}'", line_num)
                # Lark 单引号转 EBNF 双引号
                definition = re.sub(r"'([^']*)'", r'"\1"', definition)
                referenced_rules.update(extract_references(definition))
                current_definition = [definition.strip()]

            elif line.startswith("|"):
                if not current_rule:
                    raise ValueError(
                        f"Alternative '|' on line {line_num} "
                        "without a preceding rule definition"
                    )

                alt_def = line[1:].strip()
                check_quotes(
                    alt_def, f"alternative for rule '{current_rule}'", line_num
                )
                alt_def = re.sub(r"'([^']*)'", r'"\1"', alt_def)
                referenced_rules.update(extract_references(alt_def))
                current_definition.append(alt_def)

        except ValueError as e:
            raise ValueError(f"Error on line {line_num}: {str(e)}") from e

    # 添加最后的规则
    if current_rule:
        output_lines.append(f"{current_rule} ::= {' | '.join(current_definition)}")

    # 验证所有引用的规则都已定义
    undefined_rules = referenced_rules - defined_rules - {"root"}
    if undefined_rules:
        raise ValueError(
            f"Referenced rules are not defined: {', '.join(sorted(undefined_rules))}"
        )

    return "\n".join(output_lines)


def choice_as_grammar(choice: list[str]) -> str:
    """将选择列表转换为 EBNF 文法。

    例如：["hello", "world"] -> root ::= "hello" | "world"

    Args:
        choice: 字符串选择列表

    Returns:
        EBNF 文法字符串
    """
    def escape_ebnf_string(s: str) -> str:
        """转义 EBNF 字符串中的特殊字符。

        Args:
            s: 输入字符串

        Returns:
            转义后的字符串
        """
        # 转义双引号和反斜杠
        return re.sub(r'(["\\])', r"\\\1", s)

    escaped_choices = (escape_ebnf_string(c) for c in choice)
    grammar = "root ::= " + " | ".join(f'"{c}"' for c in escaped_choices)
    return grammar
