# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logprobs 处理器模块。

本模块实现了 vLLM V1 引擎的 logprobs 处理功能，负责：
- 处理采样 logprobs（生成 token 的对数概率）
- 处理提示词 logprobs（提示词 token 的对数概率）
- 累积 logprob 计算
- Token 解码和 UTF-8 校正
- 支持扁平 logprobs 格式

主要类：
- LogprobsProcessor: Logprobs 处理器
"""
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.logprobs import (
    PromptLogprobs,
    SampleLogprobs,
    append_logprobs_for_next_position,
    create_prompt_logprobs,
    create_sample_logprobs,
)
from vllm.tokenizers.detokenizer_utils import (
    TokenizerLike,
    convert_ids_list_to_tokens,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

logger = init_logger(__name__)

NONES = itertools.repeat(None)


@dataclass
class LogprobsProcessor:
    """Logprobs 处理器。

    负责处理请求的 logprobs，包括：
    - 采样 logprobs（生成 token 的对数概率）
    - 提示词 logprobs（提示词 token 的对数概率）
    - 累积 logprob 计算
    - Token 解码和 UTF-8 校正

    Attributes:
        tokenizer: 请求的分词器，如果禁用反词元化则为 None
        logprobs: 采样 logprobs 列表
        prompt_logprobs: 提示词 logprobs 列表
        cumulative_logprob: 累积 logprob
        num_logprobs: 采样 logprobs 数量
        num_prompt_logprobs: 提示词 logprobs 数量
    """

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        """从新请求创建 Logprobs 处理器。

        Args:
            tokenizer: 分词器
            request: 引擎核心请求

        Returns:
            LogprobsProcessor 实例
        """
        sampling_params = request.sampling_params
        assert sampling_params is not None
        num_logprobs = sampling_params.logprobs
        num_prompt_logprobs = sampling_params.prompt_logprobs
        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.0),
            logprobs=(
                None
                if num_logprobs is None
                else create_sample_logprobs(sampling_params.flat_logprobs)
            ),
            prompt_logprobs=(
                None
                if num_prompt_logprobs is None
                else create_prompt_logprobs(sampling_params.flat_logprobs)
            ),
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
        )

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        """使用来自 EngineCore 的采样 logprobs 更新。

        如果 EngineCore 在上一步中生成了多个 token（例如在 spec decoding 中），
        外部列表长度可能大于 1。

        Args:
            logprobs_lists: logprob token 列表、logprobs 和 ranks 的元组
        """

        assert self.num_logprobs is not None
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        token_ids_lst, logprobs_lst, ranks_lst, _ = logprobs_lists

        for rank_np, logprobs_np, token_ids_np in zip(
            ranks_lst, logprobs_lst, token_ids_lst
        ):
            rank = rank_np.tolist()
            logprobs = logprobs_np.tolist()
            token_ids = token_ids_np.tolist()
            # Detokenize (non-incrementally).
            decoded_tokens: list[str] | Iterable[None]
            if self.tokenizer is None:
                decoded_tokens = NONES
            else:
                decoded_tokens_list = convert_ids_list_to_tokens(
                    self.tokenizer, token_ids
                )
                decoded_tokens = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_list, tokens=token_ids
                )

            # Sampler puts the sampled logprob in first.
            sampled_token_logprob = logprobs[0]
            self.cumulative_logprob += sampled_token_logprob

            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(
                self.logprobs,
                token_ids,
                logprobs,
                decoded_tokens,
                rank,
                self.num_logprobs,
            )

    def _update_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
    ) -> None:
        """使用来自 EngineCore 的提示词 logprobs 更新。

        Args:
            prompt_logprobs_tensors: 包含提示词 logprobs 张量的元组
        """

        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None

        token_ids, logprobs, ranks, _ = prompt_logprobs_tensors

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        all_decoded_tokens: list[str] | None = (
            None
            if self.tokenizer is None
            else convert_ids_list_to_tokens(
                self.tokenizer, token_ids.flatten().tolist()
            )
        )

        # Pythonize the torch tensors.
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids_list = token_ids.tolist()

        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening and UTF-8 correction per position
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs

            decoded_tokens_for_pos: list[str] | Iterable[None]
            if all_decoded_tokens is None:
                decoded_tokens_for_pos = NONES
            else:
                # Extract decoded tokens for this position
                decoded_tokens_slice = all_decoded_tokens[offset:offset_end]
                # Apply UTF-8 correction within this position's token boundaries
                decoded_tokens_for_pos = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_slice, tokens=token_ids_list[pos]
                )

            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(
                self.prompt_logprobs,
                token_ids_list[pos],
                prompt_logprobs[pos],
                decoded_tokens_for_pos,
                prompt_token_ranks[pos],
                self.num_prompt_logprobs,
            )

    def pop_prompt_logprobs(self) -> PromptLogprobs | None:
        """弹出并返回所有请求的提示词 logprobs。

        Logprobs 处理器会在一个或多个 prefill 块上聚合提示词 logprobs。
        此方法一次性返回所有提示词 logprobs 然后清除它们。
        确保正确的 RequestOutputKind.DELTA 语义，
        即在 prefill 结束时一次性返回所有提示词 logprobs。

        Returns:
            如果提示词 logprobs 被禁用则返回 None，
            否则返回所有提示词 logprobs 列表
        """
        plp = self.prompt_logprobs
        if plp:
            self.prompt_logprobs = []
        return plp

    def _correct_decoded_token(self, idx: int, tokens: list[int]) -> str:
        """校正解码后的 token。

        处理 UTF-8 不完整字符的情况，尝试使用前一个 token 进行完整解码。

        Args:
            idx: 当前 token 索引
            tokens: token IDs 列表

        Returns:
            校正后的解码字符串，如果无法校正则返回空字符串
        """

        # try with prev token id in same list
        if idx > 0:
            possible_decoded_token = self.tokenizer.decode(tokens[idx - 1 : idx + 1])
            if not possible_decoded_token.endswith("�"):
                return possible_decoded_token
        # try with previous logprob token id
        if self.logprobs:
            latest_token_id = next(iter(self.logprobs[-1]))

            decode_ids = [latest_token_id]
            if idx > 0:
                decode_ids.extend(tokens[idx - 1 : idx + 1])
            else:
                decode_ids.extend(tokens[idx : idx + 1])

            possible_decoded_token = self.tokenizer.decode(decode_ids)
            if not possible_decoded_token.endswith("�"):
                return possible_decoded_token

        # by default return empty string
        return ""

    def _verify_tokens(
        self, decoded_tokens_list: list[str], tokens: list[int]
    ) -> list[str]:
        """验证并校正解码后的 token。

        检查 UTF-8 不完整字符并进行校正。

        Args:
            decoded_tokens_list: 解码后的 token 文本列表
            tokens: token IDs 列表

        Returns:
            校正后的解码 token 列表
        """
        corrected_decoded_token_map = dict()
        for idx, text in enumerate(decoded_tokens_list):
            if text.endswith("�"):
                # utf-8 字符结尾表示可能是来自字节回退词元化的未完成字节序列
                corrected_decoded_token_map[idx] = self._correct_decoded_token(
                    idx, tokens
                )

        for idx, text in corrected_decoded_token_map.items():
            decoded_tokens_list[idx] = text

        return decoded_tokens_list

    def update_from_output(self, output: EngineCoreOutput) -> None:
        """从引擎输出更新 logprobs。

        Args:
            output: 引擎核心输出
        """
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
