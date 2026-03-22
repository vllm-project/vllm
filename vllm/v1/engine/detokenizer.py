# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""增量反词元化器模块。

本模块实现了 vLLM V1 引擎的增量反词元化功能，负责：
- 将生成的 token IDs 转换为文本
- 支持流式输出和完整输出
- 处理停止字符串检测
- 支持快速（tokenizers 库）和慢速（Python 实现）两种实现

主要类：
- IncrementalDetokenizer: 基类，工厂方法创建适当的实现
- BaseIncrementalDetokenizer: 抽象基类，实现通用逻辑
- FastIncrementalDetokenizer: 使用 tokenizers 库的快速实现
- SlowIncrementalDetokenizer: 基于 Python 的慢速实现
"""
from abc import ABC, abstractmethod

import tokenizers
from packaging import version
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from transformers import PreTrainedTokenizerFast

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.detokenizer_utils import (
    convert_prompt_ids_to_tokens,
    detokenize_incrementally,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)

# 仅 tokenizers >= 0.22.0 支持 DecodeStream 与 native prefill（ids 参数）
# 用于 FastIncrementalDetokenizer
USE_FAST_DETOKENIZER = version.parse(tokenizers.__version__) >= version.parse("0.22.0")

# 来自 https://github.com/huggingface/tokenizers/blob/909fdde2a4ffedd9295206f705eb612be2a91b12/tokenizers/src/tokenizer/mod.rs#L1042
# 的错误字符串
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"


class IncrementalDetokenizer:
    """增量反词元化器类。

    用于将 token IDs 增量转换为文本输出。
    根据 tokenizer 类型自动选择快速或慢速实现。
    """

    def __init__(self):
        """初始化反词元化器。"""
        self.token_ids: list[int] = []

    @property
    def output_token_ids(self) -> list[int]:
        """返回输出 token IDs。"""
        return self.token_ids

    def num_output_tokens(self) -> int:
        """返回输出 token 数量。

        Returns:
            token 数量
        """
        return len(self.token_ids)

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """更新 token IDs。

        Args:
            new_token_ids: 新的 token IDs
            stop_terminated: 是否因停止而终止

        Returns:
            匹配的停止字符串或 None
        """
        self.token_ids.extend(new_token_ids)
        return None

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """获取下一个输出文本。

        Args:
            finished: 是否已完成
            delta: 是否只返回增量文本

        Returns:
            输出文本
        """
        return ""

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":
        """从新请求创建反词元化器。

        Args:
            tokenizer: 分词器
            request: 引擎核心请求

        Returns:
            反词元化器实例
        """
        assert request.sampling_params is not None

        if tokenizer is None:
            # 没有 tokenizer => 跳过反词元化
            return IncrementalDetokenizer()

        if USE_FAST_DETOKENIZER and isinstance(tokenizer, PreTrainedTokenizerFast):
            # Fast tokenizer => 使用 tokenizers 库 DecodeStream
            return FastIncrementalDetokenizer(tokenizer, request)

        # 回退到基于 Python 的慢速增量反词元化
        return SlowIncrementalDetokenizer(tokenizer, request)


class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):
    """增量反词元化器抽象基类。

    实现通用的增量反词元化逻辑，包括：
    - 停止字符串处理
    - 增量文本生成
    - min_tokens 支持

    子类需要实现 decode_next 方法。
    """

    def __init__(self, request: EngineCoreRequest):
        """初始化基类反词元化器。

        Args:
            request: 引擎核心请求
        """
        super().__init__()

        # 停止字符串
        params = request.sampling_params
        assert params is not None
        if params.stop is None:
            self.stop = []
        elif isinstance(params.stop, str):
            self.stop = [params.stop]
        else:
            self.stop = params.stop
        self.min_tokens = params.min_tokens
        self.include_stop_str_in_output = params.include_stop_str_in_output

        # 当停止字符串需要从流式输出中排除时要保留的字符数
        if self.stop and not self.include_stop_str_in_output:
            self.stop_buffer_length = max(len(s) for s in self.stop) - 1
        else:
            self.stop_buffer_length = 0
        self._last_output_text_offset: int = 0

        # 生成数据
        self.output_text = ""

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """更新请求状态。

        通过以下方式：
        1) 增量反词元化新的 token IDs
        2) 评估停止条件

        Args:
            new_token_ids: 新的 token IDs
            stop_terminated: 是否因停止而终止

        Returns:
            匹配的停止字符串或 None
        """
        if not new_token_ids:
            # 如果没有新的 token IDs 则跳过反词元化
            return None

        if stop_terminated and not self.include_stop_str_in_output:
            # 如果已因停止终止，根据 include_stop_str_in_output 参数
            # 从反词元化中排除最后一个 token
            skipped_stop_token_id = new_token_ids[-1]
            new_token_ids = new_token_ids[:-1]
        else:
            skipped_stop_token_id = None

        # 1. 增量反词元化新的 token IDs
        stop_check_offset = len(self.output_text)
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            self.output_text += self.decode_next(new_token_id)
            # 支持 min_tokens
            if self.min_tokens and self.num_output_tokens() <= self.min_tokens:
                stop_check_offset = len(self.output_text)

        if skipped_stop_token_id is not None:
            # 清理跳过反词元化后的状态
            self.token_ids.append(skipped_stop_token_id)

        # 2. 评估停止字符串
        stop_string = None
        if self.stop and self.num_output_tokens() > self.min_tokens:
            stop = check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(self.output_text) - stop_check_offset,
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if stop is not None:
                stop_string, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]

        return stop_string

    @abstractmethod
    def decode_next(self, next_token_id: int) -> str:
        """解码下一个 token。

        Args:
            next_token_id: 下一个 token ID

        Returns:
            解码后的字符串
        """
        raise NotImplementedError

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """获取下一个输出文本。

        Args:
            finished: 是否已完成
            delta: 如果为 True，只返回自上次调用以来的新文本

        Returns:
            输出文本
        """
        # 如果序列已完成则返回完整输出文本
        buffer_length = 0 if finished else self.stop_buffer_length
        if not delta:
            if not buffer_length:
                return self.output_text
            return self.output_text[:-buffer_length]

        length = len(self.output_text) - buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""


class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):
    """快速增量反词元化器。

    使用 tokenizers 库的 DecodeStream 实现快速反词元化。
    仅适用于 PreTrainedTokenizerFast 且 tokenizers 版本 >= 0.22.0。
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast, request: EngineCoreRequest):
        """初始化快速反词元化器。

        Args:
            tokenizer: 快速分词器
            request: 引擎核心请求
        """
        super().__init__(request)

        sampling_params = request.sampling_params
        assert sampling_params is not None

        self.request_id = request.request_id
        self.skip_special_tokens = sampling_params.skip_special_tokens

        self.tokenizer: Tokenizer = tokenizer._tokenizer

        # 使用 native prefill 用 prompt tokens 初始化 decode stream
        self.stream = DecodeStream(
            ids=request.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )

        self.spaces_between_special_tokens = (
            sampling_params.skip_special_tokens
            or sampling_params.spaces_between_special_tokens
        )

        if not self.spaces_between_special_tokens:
            # 存储已添加的 token IDs 字典以便抑制它们之间的空格
            added_token_ids = getattr(self.tokenizer, "added_token_ids", None)
            if added_token_ids is None:
                self.tokenizer.added_token_ids = added_token_ids = {
                    tid: tok.content
                    for tid, tok in self.tokenizer.get_added_tokens_decoder().items()
                }

            if added_token_ids:
                self.last_special = False
                self.added_token_ids = added_token_ids
            else:
                # 没有已添加的 tokens
                self.spaces_between_special_tokens = True

    def decode_next(self, next_token_id: int) -> str:
        """解码下一个 token。

        Args:
            next_token_id: 下一个 token ID

        Returns:
            解码后的字符串
        """
        token = self._protected_step(next_token_id)

        if not self.spaces_between_special_tokens:
            special_token = self.added_token_ids.get(next_token_id)
            is_special = special_token is not None
            if is_special and self.last_special:
                # 返回原始 token 字符串不带任何前缀空格
                token = special_token
            self.last_special = is_special

        return token or ""

    def _protected_step(self, next_token_id: int) -> str | None:
        """受保护的 step 调用，处理异常。

        Args:
            next_token_id: 下一个 token ID

        Returns:
            解码后的字符串或 None
        """
        try:
            token = self.stream.step(self.tokenizer, next_token_id)
        except (OverflowError, TypeError):
            # 处理罕见的溢出错误
            # 参见 https://github.com/vllm-project/vllm/issues/21951
            logger.exception("Encountered invalid token id: %r", next_token_id)
            token = None
        except Exception as e:
            if not str(e).startswith(INVALID_PREFIX_ERR_MSG):
                raise e
            # 从 tokenizer 产生非单调、无效 UTF-8 输出的边缘情况中恢复
            # 这会破坏 tokenizers 的 DecodeStream 的内部状态
            # 参见 https://github.com/vllm-project/vllm/issues/17448
            logger.warning(
                "Encountered invalid prefix detokenization error"
                " for request %s, resetting decode stream.",
                self.request_id,
            )
            self.stream = DecodeStream(skip_special_tokens=self.skip_special_tokens)
            token = self.stream.step(self.tokenizer, next_token_id)
        return token


class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    """慢速增量反词元化器。

    基于 Python 实现的增量反词元化器，作为 FastIncrementalDetokenizer
    的回退方案。适用于不支持快速实现的 tokenizer。
    """

    def __init__(self, tokenizer: TokenizerLike, request: EngineCoreRequest):
        """初始化慢速反词元化器。

        Args:
            tokenizer: 分词器
            request: 引擎核心请求
        """
        super().__init__(request)

        self.tokenizer = tokenizer
        params = request.sampling_params
        assert params is not None

        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )

        # Metadata for incremental detokenization.
        if request.prompt_token_ids is not None:
            self.tokens, self.prefix_offset, self.read_offset = (
                convert_prompt_ids_to_tokens(
                    tokenizer=tokenizer,
                    prompt_ids=request.prompt_token_ids,
                    skip_special_tokens=params.skip_special_tokens,
                )
            )
        else:
            # Prompt embedding requests cannot be detokenized, in general.
            self.tokens = [""] * self.prompt_len
            self.prefix_offset = 0
            self.read_offset = 0

        self.token_ids.extend(request.prompt_token_ids or [0] * self.prompt_len)

        self.skip_special_tokens = params.skip_special_tokens
        self.spaces_between_special_tokens = params.spaces_between_special_tokens

    @property
    def output_token_ids(self) -> list[int]:
        if self.prompt_len:
            return self.token_ids[self.prompt_len :]
        return self.token_ids

    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len

    def decode_next(self, next_token_id: int) -> str:
        new_tokens, decoded_text, prefix_offset, read_offset = detokenize_incrementally(
            tokenizer=self.tokenizer,
            all_input_ids=self.token_ids,
            prev_tokens=self.tokens,
            prefix_offset=self.prefix_offset,
            read_offset=self.read_offset,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )

        self.tokens.extend(new_tokens)
        self.prefix_offset = prefix_offset
        self.read_offset = read_offset

        return decoded_text


def check_stop_strings(
    output_text: str,
    new_char_count: int,
    stop: list[str],
    include_in_output: bool,
) -> tuple[str, int] | None:
    """Check if any stop strings are matched and truncate sequence
    output text accordingly.

    Returns tuple (stop_string, offset) if matched or else None.

    Where stop_string is the matched stop string and offset is the
    length to which output_text should be truncated, or -1 for no
    truncation.
    """
    if not new_char_count or not stop:
        return None

    for stop_str in stop:
        stop_string_len = len(stop_str)
        # Avoid searching already-searched text.
        stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)
        if stop_index == -1:
            continue

        if include_in_output:
            # Truncate to end of stop string.
            stop_index += stop_string_len
            if stop_index >= len(output_text):
                # No truncation required.
                return stop_str, -1

        # Truncate the output text to either the beginning
        # or end of the stop string.
        return stop_str, stop_index
    return None
