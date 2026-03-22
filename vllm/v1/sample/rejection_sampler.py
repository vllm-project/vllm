# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""拒绝采样器模块。

本模块实现了推测解码中的拒绝采样功能，负责：
- 根据 Chen et al. (2022) 的推测采样算法进行 token 验证
- 处理贪婪采样和随机采样两种模式
- 支持 bonus token 预测和 logprobs 计算
- 应用采样约束（温度、top-k、top-p）到目标模型分布

主要类：
- RejectionSampler: 拒绝采样器，继承自 nn.Module

算法说明：
    拒绝采样算法基于论文：https://arxiv.org/abs/2211.17192
    核心思想：使用草稿模型生成候选 token，然后用目标模型验证
    - 如果目标概率/草稿概率 >= 均匀采样概率，接受 token
    - 否则拒绝，从调整后的分布中采样恢复 token
"""

from collections.abc import Sequence
from dataclasses import replace

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, SamplerOutput
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

logger = init_logger(__name__)

# 占位符 token ID，用于标记被拒绝的 token
PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
# 贪婪采样的温度值（0 表示贪婪）
GREEDY_TEMPERATURE: tl.constexpr = 0
# 每个请求在单步中允许的最大推测 draft token 数量
MAX_SPEC_LEN = 128


class RejectionSampler(nn.Module):
    """拒绝采样器实现。

    实现严格遵循 https://arxiv.org/abs/2211.17192 中描述的算法。

    术语说明：
    - accepted tokens（接受 token）：基于草稿和目标概率之间的关系被接受的 token
    - recovered tokens（恢复 token）：从调整后的概率分布中采样的 token，
      该分布由草稿和目标概率共同导出
    - bonus tokens（奖励 token）：如果所有推测 token 都被接受，在序列末尾
      添加的额外 token。仅从目标概率采样，在拒绝采样器外部传入以支持
      更多采样策略（如 top_p、top_k）
    - output tokens（输出 token）：最终生成的 token
      output tokens = accepted tokens + recovered tokens + bonus tokens

    Attributes:
        sampler: 基础采样器实例
        is_processed_logprobs_mode: logprobs 模式是否以 "processed" 开头
        is_logits_logprobs_mode: logprobs 模式是否以 "logits" 结尾
    """

    def __init__(self, sampler: Sampler):
        """初始化拒绝采样器。

        Args:
            sampler: 基础采样器实例
        """
        super().__init__()
        self.sampler = sampler
        logprobs_mode = self.sampler.logprobs_mode
        self.is_processed_logprobs_mode = logprobs_mode.startswith("processed")
        self.is_logits_logprobs_mode = logprobs_mode.endswith("logits")

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """执行拒绝采样。

        Args:
            metadata: 推测解码的元数据
            draft_probs: 草稿 token 的概率分布，形状为 [num_tokens, vocab_size]
                如果不提供概率（如 ngram 推测解码），可以为 None
            logits: 目标模型的 logits 概率分布，形状为 [num_tokens + batch_size, vocab_size]
                来自不同请求的概率被展平为单个张量
                注意：logits 可能会在原地更新以节省内存
            sampling_metadata: 采样元数据，包含温度、top-k/top-p 等参数

        Returns:
            SamplerOutput: 包含最终输出 token IDs 和 logprobs（如果请求）
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # 使用张量索引时，PyTorch 会创建新张量，与原始 logits 分离存储
        # 因此对 bonus_logits 的原地操作不会影响原始 logits
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,  # 返回完整 logprobs
            ),
            predict_bonus_token=True,
            # 覆盖 logprobs 模式以返回 logits，因为后面需要计算接受 token 的 logprobs
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # target_logits 同样是新张量，可以安全原地更新
        raw_target_logits = logits[target_logits_indices]
        # 使用 float32 表示 target_logits
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
        if not self.is_processed_logprobs_mode:
            # 在应用处理器之前克隆，保留原始 logits 用于 logprobs 计算
            # apply_logits_processors 会原地修改张量
            target_logits = target_logits.clone()
        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )
        # [num_tokens, vocab_size]
        # 注意：apply_sampling_constraints 可能会原地更新 target_logits
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        # 执行拒绝采样
        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )

        # 处理 logprobs
        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )

    def _get_logprobs_tensors(
        self,
        max_num_logprobs: int,
        metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """获取 logprobs 张量。

        Args:
            max_num_logprobs: 每个 token 的最大 logprobs 数量
            metadata: 推测解码元数据
            logits: 原始 logits
            target_logits: 目标 logits
            bonus_logits: bonus logits
            sampled_token_ids: 采样的 token IDs

        Returns:
            LogprobsTensors 包含索引、logprobs 和排名
        """
        cu_num_sampled_tokens = torch.zeros_like(metadata.cu_num_sampled_tokens)
        cu_num_sampled_tokens[1:] = metadata.cu_num_sampled_tokens[:-1]

        # 收集目标和 bonus logits
        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices
        final_logits = torch.zeros_like(logits, dtype=torch.float32)
        final_logits[target_logits_indices] = target_logits.to(torch.float32)
        final_logits[bonus_logits_indices] = bonus_logits.to(torch.float32)

        # 注意：为避免 CPU-GPU 同步，我们为所有 draft token 计算索引
        # 包括被拒绝的 token，稍后在 parse_output 中过滤
        logit_start_indices = cu_num_sampled_tokens
        offsets = torch.arange(
            sampled_token_ids.shape[-1],
            device=logit_start_indices.device,
            dtype=logit_start_indices.dtype,
        )
        accepted_logit_indices = (
            logit_start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        ).flatten()
        accepted_logit_indices.clamp_(max=final_logits.shape[0] - 1)
        accepted_tokens = sampled_token_ids.clone().flatten()
        # 将占位符 token IDs 替换为 0，避免 gather_logprobs 错误
        accepted_tokens[accepted_tokens == PLACEHOLDER_TOKEN_ID] = 0

        # 计算接受 token 的 logprobs
        accepted_logits = final_logits[accepted_logit_indices]
        accepted_logprobs = (
            accepted_logits
            if self.is_logits_logprobs_mode
            else self.sampler.compute_logprobs(accepted_logits)
        )
        return self.sampler.gather_logprobs(
            accepted_logprobs,
            max_num_logprobs,
            accepted_tokens.to(torch.int64),
        )

    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> tuple[list[list[int]], LogprobsLists | None]:
        """解析拒绝采样器的输出。

        Args:
            output_token_ids: 采样的 token IDs，形状为 [batch_size, max_spec_len + 1]
                被拒绝的 token 被替换为 PLACEHOLDER_TOKEN_ID
            vocab_size: 词表大小
            discard_req_indices: 可选的要丢弃的请求索引列表
            logprobs_tensors: 可选的要过滤的 logprobs 张量

        Returns:
            包含 token IDs 列表的列表，以及可选的 logprobs 列表
        """
        output_token_ids_np = output_token_ids.cpu().numpy()
        # 创建有效 token 的掩码
        valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size
        )
        output_logprobs = None
        if logprobs_tensors is not None:
            cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
            output_logprobs = filtered_tensors.tolists(cu_num_tokens)

        if len(discard_req_indices) > 0:
            valid_mask[discard_req_indices] = False
        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs, output_logprobs

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        """应用 logits 处理器。

        Args:
            logits: 输入的 logits
            sampling_metadata: 采样元数据
            metadata: 推测解码元数据

        Returns:
            处理后的 logits
        """
        has_penalties = not sampling_metadata.no_penalties
        any_penalties_or_bad_words = (
            sampling_metadata.bad_words_token_ids or has_penalties
        )

        output_token_ids = sampling_metadata.output_token_ids
        if any_penalties_or_bad_words:
            # 在推测解码场景中，将基础输出与推测 token 组合
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # 计算目标 logits 的索引
        if sampling_metadata.allowed_token_ids_mask is not None or has_penalties:
            num_requests = len(metadata.num_draft_tokens)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(
                device=logits.device, non_blocking=True
            )
            logits = self.apply_penalties(
                logits, sampling_metadata, metadata, repeat_indices, output_token_ids
            )

            # 应用允许的 token IDs
            if sampling_metadata.allowed_token_ids_mask is not None:
                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        # 应用禁用词排除
        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        # 应用非 argmax 不变的 logits 处理器
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            if isinstance(processor, MinTokensLogitsProcessor):
                # MinTokens 处理器需要特殊的推测解码版本
                logits = processor.apply_with_spec_decode(
                    logits, metadata.num_draft_tokens
                )

        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
        repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        """应用惩罚到 logits。

        Args:
            logits: 输入的 logits
            sampling_metadata: 采样元数据
            metadata: 推测解码元数据
            repeat_indices: 重复索引张量
            output_token_ids: 输出 token IDs

        Returns:
            应用惩罚后的 logits
        """
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None

        # 根据重复索引扩展惩罚参数
        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
        presence_penalties = sampling_metadata.presence_penalties[repeat_indices]
        frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]
        repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]

        logits = apply_all_penalties(
            logits,
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
        return logits

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

        result = []
        for out, spec in zip(output_token_ids, spec_token_ids):
            if len(spec) == 0:
                continue
            result.append(out)
            for i in range(len(spec) - 1):
                result.append([*result[-1], spec[i]])
        return result


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """执行拒绝采样。

    Args:
        draft_token_ids: 草稿 token IDs，形状为 [num_tokens]
        num_draft_tokens: 每个请求的 draft token 数量列表
        max_spec_len: 最大推测长度
        cu_num_draft_tokens: 累积的 draft token 数量，形状为 [batch_size]
        draft_probs: 草稿概率分布，形状为 [num_tokens, vocab_size]
        target_logits: 目标 logits，形状为 [num_tokens, vocab_size]
        bonus_token_ids: bonus token IDs，形状为 [batch_size, 1]
        sampling_metadata: 采样元数据

    Returns:
        输出 token IDs，形状为 [batch_size, max_spec_len + 1]
    """
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    # 创建输出缓冲区
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # 与 SamplerOutput.sampled_token_ids 保持一致
        device=device,
    )

    # 处理贪婪采样模式
    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # 贪婪采样请求的拒绝采样
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # 从目标 logits 计算概率分布
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    assert target_probs.is_contiguous()

    # 生成用于拒绝采样的均匀概率
    # [num_tokens]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # 为每个位置采样恢复 token
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # 随机采样请求的拒绝采样
    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return output_token_ids


def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """应用采样约束。

    此函数基于采样元数据处理 logits：
    - 应用温度缩放
    - 应用 top-k 和 top-p 约束
    - 对于贪婪解码，返回原始 logits

    Args:
        logits: 要处理的输入 logits 张量
        cu_num_draft_tokens: 累积的 draft token 数量
        sampling_metadata: 包含温度、是否贪婪采样等参数的元数据

    Returns:
        如果是非贪婪采样则返回处理后的 logits，否则返回原始 logits
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    # 扩展温度张量到 token 级别
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,  # 贪婪采样的温度替换为 1（避免除以 0）
    )
    # 原地除法以节省内存
    logits.div_(temperature.unsqueeze(-1))

    # 获取扩展后的 top_k 和 top_p 张量
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # 注意：apply_top_k_top_p 使用排序来计算掩码
    # 对于大词表可能较慢，可能导致性能问题
    return apply_top_k_top_p(logits, top_k, top_p)


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """基于 cu_num_tokens 将 [batch_size] 张量扩展到 [num_tokens]。

    示例：
        x = [a, b, c], cu_num_tokens = [2, 5, 6]
        num_tokens = 6
        expanded_x = [a, a, b, b, b, c]

    Args:
        x: 要扩展的 [batch_size] 张量
        cu_num_tokens: [batch_size] 张量，包含每个批次的累积 token 数量
            每个元素表示到该批次为止的总 token 数
        num_tokens: token 总数
        replace_from: x 中要被替换的值
        replace_to: 替换后的值

    Returns:
        expanded_x: [num_tokens] 张量
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_kernel[(batch_size,)](
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # 避免重新编译
    )
    return expanded_x


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """生成均匀分布的随机概率。

    此方法生成形状为 (num_tokens,) 的张量，填充 [0, 1) 范围内的均匀随机值。
    如果提供了 generators 字典，有自定义种子的请求会使用提供的 Generator
    以确保可重复性。其他请求则无种子生成。

    Args:
        num_tokens: token 总数
        num_draft_tokens: 每个请求的 draft token 数量列表
        generators: 将批次索引映射到 Generator 的字典
        device: 要分配张量的设备

    Returns:
        uniform_rand: 形状为 (num_tokens,) 的张量，包含 [0, 1) 范围内的均匀随机值

    注意：
        这里使用 float64 而不是 float32，因为使用 float32 时，
        uniform_prob 有不可忽视的概率被采样为精确的 0.0
        （参见 https://github.com/pytorch/pytorch/issues/16706）
        使用 float64 可以缓解这个问题。
    """
    # 使用 float64 避免浮点精度问题
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float64,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        # 不为没有 draft token 的请求生成随机数
        # 这对可重复性很重要
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    """采样恢复 token。

    恢复 token 用于当草稿 token 被拒绝时，从调整后的分布中采样。

    Args:
        max_spec_len: 最大推测长度
        num_draft_tokens: 每个请求的 draft token 数量列表
        cu_num_draft_tokens: 累积的 draft token 数量
        draft_token_ids: 草稿 token IDs
        draft_probs: 草稿概率（可选）
        target_probs: 目标概率
        sampling_metadata: 采样元数据
        device: 设备

    Returns:
        恢复的 token IDs，形状与 draft_token_ids 相同
    """
    # 注意：只为每个请求创建一个分布
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # 不为没有 draft token 的请求生成随机数
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    inv_q = q.reciprocal()

    recovered_token_ids = torch.empty_like(draft_token_ids)
    BLOCK_SIZE = 8192
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        vocab_size,
        BLOCK_SIZE,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids


# 注意：避免 specialization 以防止不必要的重新编译
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] 或 None
    max_spec_len,
):
    """贪婪采样的拒绝采样内核。

    对于贪婪采样，如果草稿 token 与目标 argmax 匹配则接受，
    否则拒绝并使用恢复 token。

    Args:
        output_token_ids_ptr: 输出 token IDs 指针
        cu_num_draft_tokens_ptr: 累积 draft token 数量指针
        draft_token_ids_ptr: 草稿 token IDs 指针
        target_argmax_ptr: 目标 argmax 指针
        bonus_token_ids_ptr: bonus token IDs 指针
        is_greedy_ptr: 是否贪婪采样的指针
        max_spec_len: 最大推测长度
    """
    req_idx = tl.program_id(0)
    # 注意：因为在 profiling 运行期间 is_greedy_ptr 不为 None，
    # 所以在运行时当 is_greedy_ptr 为 None 时可能会发生重新编译
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # 非贪婪采样请求提前退出
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                target_argmax_id,
            )
            if draft_token_id != target_argmax_id:
                # 拒绝
                rejected = True

    if not rejected:
        # 如果所有 token 都被接受，添加 bonus token
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# 注意：避免 specialization 以防止不必要的重新编译
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] 或 None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    """随机采样的拒绝采样内核。

    实现拒绝采样算法：
    - 计算比率 = target_prob / draft_prob
    - 如果比率 >= uniform_prob，接受草稿 token
    - 否则拒绝，使用恢复 token

    Args:
        output_token_ids_ptr: 输出 token IDs 指针
        cu_num_draft_tokens_ptr: 累积 draft token 数量指针
        draft_token_ids_ptr: 草稿 token IDs 指针
        draft_probs_ptr: 草稿概率指针（可选）
        target_probs_ptr: 目标概率指针
        bonus_token_ids_ptr: bonus token IDs 指针
        recovered_token_ids_ptr: 恢复 token IDs 指针
        uniform_probs_ptr: 均匀概率指针
        is_greedy_ptr: 是否贪婪采样的指针
        max_spec_len: 最大推测长度
        vocab_size: 词表大小
        NO_DRAFT_PROBS: 是否不提供草稿概率的编译时常量
    """
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # 贪婪采样请求提前退出
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            if NO_DRAFT_PROBS:
                # 没有草稿概率时使用 1（如 ngram 推测）
                draft_prob = 1
            else:
                draft_prob = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
            target_prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
            )
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            # 注意：虽然草稿概率理论上不应该为 0，但我们检查以避免 NaN
            # 如果确实为 0，则拒绝
            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                # 接受
                token_id = draft_token_id
            else:
                # 拒绝，使用恢复 token
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # 如果所有 token 都被接受，添加 bonus token
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# 注意：避免 specialization 以防止不必要的重新编译
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    """批次扩展内核。

    将 [batch_size] 张量扩展到 [num_tokens] 张量。

    Args:
        output_ptr: 输出指针
        input_ptr: 输入指针
        cu_num_tokens_ptr: 累积 token 数量指针
        replace_from: 要替换的值
        replace_to: 替换值
        MAX_NUM_TOKENS: 最大 token 数量（编译时常量）
    """
    req_idx = tl.program_id(0)
    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] 或 None
    target_probs_ptr,  # [num_tokens, vocab_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    """恢复 token 采样内核。

    使用 Gumbel-Max 技巧高效采样恢复 token：
    - 计算调整后概率：max(target_prob - draft_prob, 0)
    - 使用 inv_q (Gumbel 噪声的倒数) 进行加权采样
    - 通过分块归约找到 argmax

    Args:
        output_token_ids_ptr: 输出 token IDs 指针
        cu_num_draft_tokens_ptr: 累积 draft token 数量指针
        draft_token_ids_ptr: 草稿 token IDs 指针
        draft_probs_ptr: 草稿概率指针（可选）
        target_probs_ptr: 目标概率指针
        inv_q_ptr: inv_q 指针（Gumbel 噪声倒数）
        vocab_size: 词表大小
        BLOCK_SIZE: 分块大小
        NO_DRAFT_PROBS: 是否不提供草稿概率
    """
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # 超出范围的位置提前退出
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos

    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

    max_val = float("-inf")
    recovered_id = 0
    # 分块遍历词表
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size

        if NO_DRAFT_PROBS:
            # 没有草稿概率时，排除草稿 token 本身
            prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=(vocab_mask & (vocab_offset != draft_token_id)),
                other=0.0,
            )
        else:
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            # 调整后的概率分布
            prob = tl.maximum(target_prob - draft_prob, 0.0)
            # 注意：这里不需要归一化，因为 tl.argmax 会选择最大值

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        # 局部分块归约
        score = prob * inv_q
        local_max, local_id = tl.max(score, axis=0, return_indices=True)

        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id

    tl.store(output_token_ids_ptr + token_idx, recovered_id)
