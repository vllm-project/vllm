# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""采样器模块。

本模块实现了从模型输出中采样下一个 token 的采样层，负责：
- 应用温度和各种约束到 logits
- 执行贪婪采样和随机采样
- 支持 top-k 和 top-p 滤波
- 计算和收集 logprobs
- 应用惩罚（频率、存在、重复）

主要类：
- Sampler: 采样层，继承自 nn.Module
"""

import torch
import torch.nn as nn

from vllm.config.model import LogprobsMode
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):
    """从模型输出中采样下一个 token 的层。

    采样过程按以下顺序执行：

    1. 如果需要 logprobs：
       a) 如果 `logprobs_mode` 是 `raw_logprobs`，计算 logprobs 作为最终要返回的 logprobs
       b) 如果 `logprobs_mode` 是 `raw_logits`，克隆 logits 作为最终要返回的 logprobs
    2. 将 logits 转换为 float32
    3. 应用允许的 token IDs 白名单
    4. 应用禁用词排除
    5. 应用非 argmax 不变的 logits 处理器，即可能影响贪婪采样的处理器
       a) Min tokens 处理器
       b) Logit bias 处理器
    6. 应用惩罚
       a) 重复惩罚
       b) 频率惩罚
       c) 存在惩罚
    7. 采样下一个 token。`sample` 方法执行以下步骤：
       a) 如果不是 `all_random`，执行贪婪采样。如果 `all_greedy`，
          返回贪婪采样的 token 和最终 logprobs（如果请求）
       b) 应用温度
       c) 应用 argmax 不变的 logits 处理器，默认是 min_p 处理器
       d) 应用 top_k 和/或 top_p
       e) 使用概率分布采样下一个 token
       f) 如果 `all_random` 或 temperature >= epsilon (1e-5)，返回随机采样的 token
          和最终 logprobs（如果请求），否则返回贪婪采样的 token 和 logprobs
    8. 收集前 `max_num_logprobs` 个和采样 token 的 logprobs（如果请求）。
       注意：如果采样 token 在前 `max_num_logprobs` 内，logprob 最终会在
       输出处理期间合并到 `LogprobsProcessor` 中。因此，最终输出可能包含
       `max_num_logprobs + 1` 或 `max_num_logprobs` 个 logprobs
    9. 返回最终的 `SamplerOutput`

    Attributes:
        topk_topp_sampler: TopKTopPSampler 实例
        pin_memory: 是否使用锁页内存
        logprobs_mode: logprobs 模式配置
    """

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        """初始化采样器。

        Args:
            logprobs_mode: logprobs 模式，默认为 "raw_logprobs"
        """
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler(logprobs_mode)
        self.pin_memory = is_pin_memory_available()
        self.logprobs_mode = logprobs_mode

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        """执行采样。

        Args:
            logits: 模型输出的 logits
            sampling_metadata: 采样元数据
            predict_bonus_token: 是否预测 bonus token（用于推测解码）
            logprobs_mode_override: 可选的 logprobs 模式覆盖

        Returns:
            SamplerOutput 包含采样的 token IDs 和 logprobs
        """
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        # 注意 (woosuk): 对于 top-k logprobs，使用原始 logits（在任何惩罚或
        # 温度缩放之前）。这与 V0 采样器不同，V0 使用用于采样的 logits
        # （在惩罚和温度缩放之后）。
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                if logits.dtype == torch.float32:
                    raw_logprobs = logits.clone()
                else:
                    raw_logprobs = logits.to(torch.float32)

        # 使用 float32 表示 logits。
        logits = logits.to(torch.float32)

        logits = self.apply_logits_processors(
            logits, sampling_metadata, predict_bonus_token
        )
        # 采样下一个 token。
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
        # 将采样 token IDs 转换为 int64 (long) 类型，以确保与后续可能使用这些
        # 值作为索引的操作兼容。此转换是必要的，因为 FlashInfer 采样操作返回
        # int32（而 PyTorch argmax 和 topk 返回 int64）。
        sampled = sampled.long()

        if num_logprobs is None:
            logprobs_tensors = None
        elif num_logprobs == -1:
            # 返回完整的未排序和未排名的 logprobs。
            logprobs_tensors = LogprobsTensors(
                torch.empty(0), raw_logprobs, torch.empty(0)
            )
        else:
            # 收集 topk 和采样 token 的 logprobs 和排名。
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )

        # 使用 int32 减少张量大小。
        sampled = sampled.to(torch.int32)

        # 这些是 GPU 张量。
        sampler_output = SamplerOutput(
            # 采样 token 被扩展为形状为 [num_requests, 1] 的 2D 张量，
            # 其中每行表示每个请求生成的一个 token。
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        """应用温度缩放。

        Args:
            logits: 输入的 logits
            temp: 温度张量
            all_random: 是否所有请求都是随机采样

        Returns:
            温度缩放后的 logits
        """
        # 使用原地除法以避免创建新张量。
        # 如果有贪婪请求，避免除以零。
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        """贪婪采样，返回 argmax。

        Args:
            logits: 输入的 logits

        Returns:
            argmax token IDs
        """
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """基于采样元数据采样 logits。

        此方法中调用的各种 logits 处理函数可能会就地更新 logits 张量。

        Args:
            logits: 输入的 logits
            sampling_metadata: 采样元数据
            logprobs_mode_override: 可选的 logprobs 模式覆盖

        Returns:
            (采样 token IDs, 处理后的 logprobs 或 None)
        """

        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None

        # 应用温度。
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

        # 应用仅适用于随机采样的 logits 处理器（argmax 不变）。
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # 应用 top_k 和/或 top_p。
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # 重用张量
        )
        return sampled, processed_logprobs

    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        """计算 logprobs。

        Args:
            logits: 输入的 logits

        Returns:
            logprobs 张量
        """
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """收集 topk 和采样/prompt token 的 logprobs。

        Args:
            logprobs: (num tokens) x (vocab) 张量
            num_logprobs: 每个 token 保留的最大 logprobs 数量
            token_ids: prompt tokens（如果是 prompt logprobs）
                      或采样 tokens（如果是采样 logprobs）；
                      形状为 (num tokens) 的 1D token ID 张量
                      必须是 int64 类型

        Returns:
            Top-k int 索引张量，形状为 (num tokens) x (num_logprobs + 1)
            Top-k float logprobs 张量，形状为 (num tokens) x (num_logprobs + 1)
            采样 token 排名张量，形状为 (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # 找到 topK 值。
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # 获取 prompt 或采样 token 的 logprob。
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # 计算实际 token 的排名。
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # 与 topk 连接。
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # 使用 int32 减少张量大小。
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        """将基础输出与推测 token 组合。

        Args:
            output_token_ids: 基础输出 token IDs
            spec_token_ids: 推测 token IDs（可选）

        Returns:
            组合后的 token IDs
        """
        if spec_token_ids is None:
            return output_token_ids

        return [
            [*out, *spec] if spec else out
            for out, spec in zip(output_token_ids, spec_token_ids)
        ]

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        """应用 logits 处理器。

        Args:
            logits: 输入的 logits
            sampling_metadata: 采样元数据
            predict_bonus_token: 是否预测 bonus token

        Returns:
            处理后的 logits
        """
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        any_penalties_or_bad_words = (
            bool(bad_words_token_ids) or not sampling_metadata.no_penalties
        )

        output_token_ids = sampling_metadata.output_token_ids
        if predict_bonus_token and any_penalties_or_bad_words:
            # 当启用推测解码时，将基础输出与推测 token 组合。
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # 应用允许的 token IDs。
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))

        # 应用禁用词排除。
        if bad_words_token_ids:
            apply_bad_words(logits, bad_words_token_ids, output_token_ids)

        # 应用可能影响贪婪采样的 logits 处理器。
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        # 应用惩罚（例如，频率惩罚）。
        logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        """应用惩罚到 logits。

        Args:
            logits: 输入的 logits
            sampling_metadata: 采样元数据
            output_token_ids: 输出 token IDs

        Returns:
            应用惩罚后的 logits
        """
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )
