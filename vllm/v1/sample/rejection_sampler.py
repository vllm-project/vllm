# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 投机解码的拒绝采样器（严格对应 arXiv:2211.17192 的算法）。
# [CN] 术语先把清楚：
# [CN]   accepted  —— 按 draft/target 概率关系判定被接受的草稿 token
# [CN]   recovered —— 按"调整后分布"max(p_target - p_draft, 0) 重新采出的补偿 token
# [CN]   bonus     —— 草稿全中时追加的那一个额外 token
# [CN]   output    —— 最终产出 = accepted + recovered + bonus
# [CN] 核心不变式：无论接受与否，输出分布都严格等于目标模型分布，
# [CN] 这正是拒绝采样能保证"投机解码不改变结果分布"的原因。

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config.model import PROCESSED_LOGPROBS_MODES
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
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig

logger = init_logger(__name__)

# [CN] 占位符：被拒绝的位置填 -1，由 parse_output 阶段过滤掉。
# [CN] 之所以不直接压缩，是因为内核是固定形状 (batch, max_spec_len+1) 的。

PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
# [CN] 温度==0 视为贪心。这样"是否贪心"就是一个可向量化判定。

GREEDY_TEMPERATURE: tl.constexpr = 0
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
# [CN] 单步草稿上限。Triton 内核的 MAX_NUM_TOKENS 用它做编译期常量，
# [CN] 避免每个不同的 max_spec_len 都触发一次重新编译。

MAX_SPEC_LEN = 128


# [CN] 继承 nn.Module 而非普通类：这样可以参与 torch.compile / 图捕获，
# [CN] 与模型主体一起被编译进同一张图。

class RejectionSampler(nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    # [CN] 构造期就把 logprobs 模式的分支判定算成布尔量缓存起来，
    # [CN] 避免每次 forward 都去做字符串集合查询。

    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig | None = None,
        device: torch.device | None = None,
    ):
        # [CN] 必须先于任何属性赋值调用：nn.Module 的内部状态要先行初始化。

        super().__init__()
        # [CN] 复用标准 Sampler：bonus token 的采样策略应与普通解码完全一致。

        self.sampler = sampler
        # [CN] 用 getattr 兜底：老版本的 Sampler 可能没有这个开关。
        # [CN] float64 Gumbel 能减少采样偏差，但代价是双倍带宽。

        self.use_fp64_gumbel = getattr(sampler, "use_fp64_gumbel", False)
        # [CN] logprobs 模式决定后面要不要保留未经修改的原始 logits，
        # [CN] 因此要在构造期就锁定。

        logprobs_mode = self.sampler.logprobs_mode
        self.is_processed_logprobs_mode = logprobs_mode in PROCESSED_LOGPROBS_MODES
        self.is_logits_logprobs_mode = logprobs_mode in (
            "raw_logits",
            "processed_logits",
        )

        # [CN] synthetic 模式：不看真实 draft 概率，而是按人为设定的接受率采样。
        # [CN] 用途是压测与 ablation —— 研究"接受率变化对吞吐的影响"。

        self.synthetic_conditional_rates: torch.Tensor | None = None
        if (
            spec_config is not None
            and spec_config.rejection_sample_method == "synthetic"
        ):
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        # [CN] 用"有无条件率表"来标记模式：省一个独立开关字段。

        self.synthetic_mode = self.synthetic_conditional_rates is not None

    # [CN] 主流程：先采 bonus token -> 处理 target logits -> 拒绝采样 -> 补 logprobs。
    # [CN] 注意 logits 会被就地修改（省显存），调用方不能假设它不变。

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        """
        # [CN] 超过上限会导致 Triton 内核的分块常量不够用，必须提前拦下。

        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        # [CN] 索引表把"含 bonus 的完整 logits"映射到"仅草稿位"的子集。

        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        # [CN] 张量索引会生成**新存储**的副本，所以后面就地改 bonus_logits
        # [CN] 不会污染原始 logits —— 这里的安全性依赖这个语义。

        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            # Override the logprobs mode to return logits because they are
            # needed later to compute the accepted token logprobs.
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        # [CN] bonus token 由外部 Sampler 采好后传入：这样就能对 bonus
        # [CN] 使用 top_p/top_k 等策略（拒绝采样本身不支持这些）。

        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        # [CN] 转 float32 再计算：softmax/除法在低精度下会引入可观的接受率偏差。

        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
        # [CN] 非 processed 模式才克隆：后续 processor 会就地改张量，
        # [CN] 而计算 logprobs 需要未改动的原始 logits。

        if not self.is_processed_logprobs_mode:
            # Clone raw_target_logits before applying processors to preserve
            # the original raw logits for logprobs computation, since
            # apply_logits_processors modifies the tensor in-place.
            target_logits = target_logits.clone()
        # [CN] 惩罚项/禁用词/logits processor 都要在**拒绝采样之前**施加，
        # [CN] 否则接受判定用的是未经约束的分布，会绕过用户的约束。

        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `apply_sampling_constraints` function.
        # [CN] 温度/top-k/top-p 也要先施加：草稿模型与目标模型必须处在
        # [CN] 同一个"采样约束空间"里，接受率才有意义。

        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        # [CN] 核心：Triton 内核完成接受/补偿/bonus 三步。

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            synthetic_mode=self.synthetic_mode,
            synthetic_conditional_rates=self.synthetic_conditional_rates,
            use_fp64_gumbel=self.use_fp64_gumbel,
        )

        # [CN] 没请求 logprobs 就完全跳过：这块会额外构造一整份 float32 logits。

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

        # [CN] 输出结构要与标准 SamplerOutput 完全一致，
        # [CN] 这样下游 output_processor 无需区分是否走了投机路径。

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )

    # [CN] 为被接受的 token 补 logprobs。被拒绝的位置的 logprobs
    # [CN] 在这里也会算，留到 parse_output 再过滤。

    def _get_logprobs_tensors(
        self,
        max_num_logprobs: int,
        metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        cu_num_sampled_tokens = torch.zeros_like(metadata.cu_num_sampled_tokens)
        # [CN] 整体后移一位，把"累计终点"变成"累计起点"。

        cu_num_sampled_tokens[1:] = metadata.cu_num_sampled_tokens[:-1]

        # Collect target and bonus logits.
        bonus_logits_indices = metadata.bonus_logits_indices
        # [CN] 这里反过来用：把处理过的 target logits 回填到完整形状的容器里。

        target_logits_indices = metadata.target_logits_indices
        # [CN] 用 target + bonus 回填一份完整 logits：这样索引时形状与原始一致，
        # [CN] 不必维护复杂的偏移表。

        final_logits = torch.zeros_like(logits, dtype=torch.float32)
        final_logits[target_logits_indices] = target_logits.to(torch.float32)
        final_logits[bonus_logits_indices] = bonus_logits.to(torch.float32)

        # NOTE: To avoid cpu-gpu synchronization, we now simply compute indices for
        # all draft tokens, including the rejected ones. The rejected tokens will
        # be filtered out in the `parse_output`.
        logit_start_indices = cu_num_sampled_tokens
        offsets = torch.arange(
            sampled_token_ids.shape[-1],
            device=logit_start_indices.device,
            dtype=logit_start_indices.dtype,
        )
        accepted_logit_indices = (
            logit_start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        ).flatten()
        # [CN] 被拒绝位置也会算出下标，可能越界；夹到最后一行即可
        # [CN] （这些行反正会被 parse_output 丢掉）。

        accepted_logit_indices.clamp_(max=final_logits.shape[0] - 1)
        accepted_tokens = sampled_token_ids.clone().flatten()
        # we replace rejected token ids with 0 to avoid gather_logprobs error
        # [CN] gather 时 -1 会被当成负索引（取最后一个 token），
        # [CN] 所以先替换成 0 避免语义错乱。

        accepted_tokens[accepted_tokens == PLACEHOLDER_TOKEN_ID] = 0

        # Compute logprobs for accepted tokens.
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
    # [CN] 把固定形状的张量还原成"每请求一个变长列表"。
    # [CN] 设备 -> 主机拷贝发生在这里，是整条链路上少有的同步点之一。

    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> tuple[list[list[int]], LogprobsLists | None]:
        """Parse the output of the rejection sampler.
        Args:
            output_token_ids: The sampled token IDs in shape
                [batch_size, max_spec_len + 1]. The rejected tokens are
                replaced with `PLACEHOLDER_TOKEN_ID` by the rejection sampler
                and will be filtered out in this function.
            vocab_size: The size of the vocabulary.
            discard_req_indices: Optional row indices to discard tokens in.
            logprobs_tensors: Optional logprobs tensors to filter.
        Returns:
            A list of lists of token IDs.
        """
        # [CN] 转 CPU + numpy：后续要用 Python 列表推导，留在 GPU 上无法做。

        output_token_ids_np = output_token_ids.cpu().numpy()
        # Create mask for valid tokens.
        # [CN] 两个条件都要：-1 是被拒占位，>=vocab 是非法 id
        # [CN] （某些极端情况下 recovered 会算出越界 id）。

        valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size
        )
        output_logprobs = None
        if logprobs_tensors is not None:
            cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            # [CN] 展平掩码后一次过滤：被拒位置的 logprobs 在这里才真正被丢掉。

            filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
            output_logprobs = filtered_tensors.tolists(cu_num_tokens)

        # [CN] 需要丢弃的请求行在这里整行置 False，比在 GPU 上做更便宜。

        if len(discard_req_indices) > 0:
            valid_mask[discard_req_indices] = False
        # [CN] 按行走 Python 循环：行数等于批大小，规模可控，
        # [CN] 且这一步本来就已经在 CPU 上了，无需再为它写内核。

        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs, output_logprobs

    # [CN] 难点：这些 processor 的输入是**按请求**的（如惩罚项统计），
    # [CN] 而 logits 是**按草稿 token** 排的，所以要先建重复索引。

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        # [CN] no_penalties 是"整批都没开惩罚"的快速关闭位，
        # [CN] 常驻全零张量换统一代码路径。

        has_penalties = not sampling_metadata.no_penalties
        any_penalties_or_bad_words = (
            sampling_metadata.bad_words_token_ids or has_penalties
        )
        output_token_ids = sampling_metadata.output_token_ids
        if any_penalties_or_bad_words:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Calculate indices of target logits.
        repeat_indices: torch.Tensor | None = None
        # [CN] 只有确实要用到按请求的参数时才建索引表：建它要一次 H2D 传输。

        need_repeat_indices = (
            sampling_metadata.allowed_token_ids_mask is not None or has_penalties
        )
        if need_repeat_indices:
            num_requests = len(metadata.num_draft_tokens)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            # [CN] 每条请求重复其草稿数量次，得到 token -> 请求 的映射。

            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(
                device=logits.device, non_blocking=True
            )
            logits = self.apply_penalties(
                logits, sampling_metadata, metadata, repeat_indices, output_token_ids
            )

            # Apply allowed token ids.
            if sampling_metadata.allowed_token_ids_mask is not None:
                # [CN] 掩码是"反"的：True 表示被禁用，所以用 masked_fill_(-inf)。

                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        # Apply bad words exclusion.
        # [CN] 海象运算符：取值与判空一步到位。禁用词需要专门处理草稿场景
        # [CN] （一串草稿里只要有一个命中就整段作废）。

        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        # [CN] 只处理"不影响 argmax 结果"之外的 processor：
        # [CN] min_tokens 这类要靠 N-gram 判断是否屏蔽 EOS。

        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            if isinstance(processor, MinTokensLogitsProcessor):
                logits = processor.apply_with_spec_decode(
                    logits, metadata.num_draft_tokens
                )
        # [CN] 思考预算控制（如强制/禁止某段时间输出特殊 token），最后统一施加。

        holder = sampling_metadata.thinking_budget_state_holder
        if holder is not None and holder.has_tracked_requests():
            logits = holder.apply_to_logits(
                logits,
                predict_bonus_token=False,
                spec_token_ids=sampling_metadata.spec_token_ids,
            )
        # [CN] 贪心批直接原样返回：温度缩放对 argmax 无影响，跳过即可。

        return logits

    @staticmethod
    # [CN] 先用 repeat_indices 把按请求的 penalty 参数展开到 token 维。

    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
        repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None

        # [CN] 惩罚统计要把 prompt 也算进去，所以要一并展开。

        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
        # [CN] 三种惩罚参数都要按 token 展开：它们都是按请求配的。

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
    # [CN] 把"历史输出"与"草稿 token"拼接成逐位置的前缀列表，
    # [CN] 供惩罚项按位置计算出现次数。

    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        if spec_token_ids is None:
            return output_token_ids

        result = []
        for out, spec in zip(output_token_ids, spec_token_ids):
            if len(spec) == 0:
                continue
            result.append(out)
            # [CN] 最后一个草稿位置不追加：它在下一轮才成为历史。

            for i in range(len(spec) - 1):
                result.append([*result[-1], spec[i]])
        return result


# [CN] 总调度函数：按批是否含贪心/随机请求，决定要不要跑哪个内核。
# [CN] 两条内核路径分开是为了避免分支过多拖慢内核。

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
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
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

    # Create output buffer.
    # [CN] 预分配并全部填占位符：被拒或被裁剪的位置自然留成 -1。

    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

    # Generate uniform probabilities before either kernel because synthetic
    # mode needs them in the greedy kernel too.  Skip only when all requests
    # are greedy *and* synthetic mode is off (the standard fast-path).
    # [num_tokens]
    # [CN] 随机数在两个内核启动前统一生成：
    # [CN] 一是保证可复现性，二是避免内核里各自 RNG 导致结果难调。

    uniform_probs: torch.Tensor | None = None
    if synthetic_mode or not sampling_metadata.all_greedy:
        uniform_probs = generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
        )

    # [CN] 只要批里有一条贪心请求就得跑贪心内核。

    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        # [CN] 贪心路径不需要完整分布，只要 argmax，省一次 softmax。

        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
            uniform_probs,
            synthetic_conditional_rates,
            SYNTHETIC_MODE=synthetic_mode,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Compute probability distribution from target logits.
    # [CN] 随机路径必须有完整概率分布，到这里才付 softmax 的代价。

    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    assert target_probs.is_contiguous()

    # Sample recovered tokens for each position.
    # [num_tokens]
    # [CN] 先为每个位置预先采好"被拒时的替补 token"：
    # [CN] 即便最终没用到，也比在内核里做 Gumbel 采样便宜。

    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
        use_fp64_gumbel,
    )

    # Rejection sampling for random sampling requests.
    assert uniform_probs is not None
    # [CN] 一维网格 = 每请求一个 program，请求内部串行循环（草稿长度有限）。

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
        synthetic_conditional_rates,
        NO_DRAFT_PROBS=draft_probs is None,
        SYNTHETIC_MODE=synthetic_mode,
    )
    return output_token_ids


# [CN] 温度 + top-k/top-p。贪心请求直接原样返回，完全不参与。

def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Process logits based on sampling metadata.

    This function applies temperature scaling to the logits,
    as well as top-k and top-p. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be processed.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Processed logits if non-greedy sampling is used,
        otherwise returns the original logits.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    # [CN] 把按请求的参数展开到 token 维；温度 0 会被替换成 1
    # [CN] （贪心请求走这条公共路径时不该除以 0）。

    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    # [CN] 就地除，避免再分配一张同尺寸张量（词表维通常十几万，很贵）。

    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    # [CN] 未配置就传 None，让下游跳过相应分支而不是填一个大数。

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

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    # [CN] 内部要做排序，大词表下较慢 —— 已知性能热点。

    return apply_top_k_top_p(logits, top_k, top_p)


# [CN] [batch] -> [num_tokens] 的展开，例：[a,b,c] + [2,5,6] -> [a,a,b,b,b,c]。

def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
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
        # [CN] 固定编译期常量：否则每换一个最大草稿长度就重编译一遍内核。

        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
    )
    return expanded_x


# [CN] 生成接受判定用的均匀随机数，支持按请求的独立种子。

def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """
    Generates a batch of uniform random samples, with optional seeding
    if available.

    This method creates a tensor of shape `(num_tokens, )` filled
    with uniform random values in the range [0, 1). If `generators` is provided,
    the requests with their own seeds will use the provided `torch.Generator`
    for reproducibility. The samples for the other requests will be generated
    without a seed.

    Args:
        num_tokens: int
            Total number of tokens.
        num_draft_tokens: List[List[int]]
            Number of draft tokens per request.
        generators: Optional[Dict[int, torch.Generator]]
            A dictionary mapping indices in the batch to
            `torch.Generator` objects.
        device: torch.device
            The device on which to allocate the tensor.
    Returns:
        uniform_rand: torch.Tensor
            A tensor of shape `(num_tokens, )` containing uniform
            random values in the range [0, 1).
    """
    # NOTE(woosuk): We deliberately use float64 instead of float32 here
    # because when using float32, there's a non-negligible chance that
    # uniform_prob is sampled to be exact 0.0 as reported in
    # https://github.com/pytorch/pytorch/issues/16706. Using float64
    # mitigates the issue.
    # [CN] 刻意用 float64：float32 下 rand 有不可忽略的概率抽到精确 0.0，
    # [CN] 会让"必然接受"的判定意外失败（PyTorch issue 16706）。

    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float64,
        device=device,
    )
    start_idx = 0
    # [CN] 逐个请求补种子：只有带了自定义 seed 的请求才重新生成随机数。

    for req_idx, n in enumerate(num_draft_tokens):
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        # [CN] 草稿数为 0 的请求不消耗随机数，保证同种子下结果可复现。

        if n == 0:
            continue
        end_idx = start_idx + n
        # [CN] 按批内下标取用户种子对应的生成器；没有就用全局 RNG。

        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        # [CN] 推进游标；被跳过的请求不占区间，后续下标自然错开。

        start_idx = end_idx
    return uniform_probs


# [CN] Gumbel-max 技巧：对每个请求采一次 q~Exp(1)，用 argmax(p / q)
# [CN] 等价于按分布采样，且能天然给出"连续位置一致的替补序列"。

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
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    # [CN] Gumbel 噪声的精度选择：float64 更准但占用双倍带宽。

    q_dtype = torch.float64 if use_fp64_gumbel else torch.float32
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=q_dtype,
        device=device,
    )
    # [CN] 每条请求**共享**一个 Gumbel 噪声向量（而非每个位置一份），
    # [CN] 这样同一请求各位置的替补 token 才具有相关性。

    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        # [CN] 同样对零草稿请求跳过：保持与上面 uniform_probs 一致的随机数消耗量。

        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    # [CN] 取倒数而不是除法：内核里乘法比除法便宜。

    inv_q = q.reciprocal()

    # [CN] 形状与草稿 token 一致：每个草稿位都要有一个替补候选。

    recovered_token_ids = torch.empty_like(draft_token_ids)
    # [CN] 词表分块遍历的 tile 大小，受共享内存/寄存器容量约束。

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
        USE_FP64_GUMBEL=use_fp64_gumbel,
    )
    return recovered_token_ids


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
# [CN] 贪心路径内核：草稿 == target argmax 即接受，否则立刻拒绝，
# [CN] 被拒位置取 target 的 argmax（贪心下"替补"就是 argmax 本身）。

def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    max_spec_len,
    uniform_probs_ptr,  # [num_tokens] or None (synthetic mode only)
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    SYNTHETIC_MODE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    # [CN] 指针为 None 表示整批都贪心，直接短路省一次访存。
    # [CN] FIXME：这会导致 profiling 时编译出的内核与运行期不同，触发重编译。

    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    # [CN] 累计偏移表中取当前请求的终点即本请求的 token 区间。

    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # [CN] 一旦被拒就锁存：后续草稿位置全部作废（投机解码的基本约束）。

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
            # [CN] synthetic 模式不比较真实分布，只看设定的接受率；
            # [CN] 同时要排除填充的 -1 草稿 id。

            if SYNTHETIC_MODE:
                uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                # -1 is used for padded draft token ids that should be rejected.
                accepted = (uniform_prob < rate) and draft_token_id >= 0
                token_id = draft_token_id if accepted else target_argmax_id
                rejected = not accepted
            else:
                token_id = target_argmax_id
                rejected = draft_token_id != target_argmax_id
            tl.store(
                # [CN] 二维 -> 一展平的手写索引：行 stride 是 max_spec_len+1
                # [CN] （最后一位留给 bonus token）。

                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
# [CN] 随机路径内核：接受概率 = min(1, p_target / p_draft)，
# [CN] 用 uniform_prob 做一次伯努利判定。

def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    NO_DRAFT_PROBS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            # [CN] -1 是填充的草稿位，一律判不接受。

            if draft_token_id < 0:
                # -1 is used for padded draft token ids that should be rejected.
                accepted = False
            elif SYNTHETIC_MODE:
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                accepted = uniform_prob < rate
            else:
                if NO_DRAFT_PROBS:
                    draft_prob = 1
                else:
                    draft_prob = tl.load(
                        draft_probs_ptr
                        + (start_idx + pos) * vocab_size
                        + draft_token_id
                    )
                target_prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
                # NOTE(woosuk): While the draft probability should never be 0,
                # we check it to avoid NaNs. If it happens to be 0, we reject.
                # [CN] 标准接受判定。draft_prob==0 会导致除零 NaN，故先排除。

                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
            if accepted:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
# [CN] 展开内核：一个请求一个 program，用掩码写入本请求的那几行。

def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:
        start_idx = tl.zeros([], dtype=cu_num_tokens_ptr.dtype.element_ty)
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    # [CN] 顺手做值替换（如温度 0 -> 1），省掉一次额外遍历。

    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    # [CN] 掩码存储：超出本请求草稿数的位置不写（保持未初始化/旧值）。

    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


@triton.jit
# [CN] 二维网格 (请求, 草稿位置)：每个 program 负责采一个替补 token。

def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    USE_FP64_GUMBEL: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    # [CN] 第二维 = 草稿位置，与请求维共同铺成 (batch, max_spec_len) 网格。

    pos = tl.program_id(1)
    # [CN] 网格按 max_spec_len 铺满，草稿不满的请求要提前退出。

    if pos >= num_draft_tokens:
        return

    # [CN] 由"请求内偏移"换算到"全局 token 下标"。

    token_idx = start_idx + pos

    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

    if USE_FP64_GUMBEL:
        max_val = tl.full((), float("-inf"), tl.float64)
    else:
        max_val = tl.full((), float("-inf"), tl.float32)
    recovered_id = 0
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size

        if NO_DRAFT_PROBS:
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
            # [CN] 拒绝采样的"残余分布"：p_recovered ∝ max(p_target - p_draft, 0)。
            # [CN] 这正是保证最终分布等于目标分布的关键。

            prob = tl.maximum(target_prob - draft_prob, 0.0)
            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        # Local tile reduction.
        # Mask out-of-vocabulary entries to -inf so they can never win
        # the argmax — prevents producing recovered_id >= vocab_size
        # when all valid entries in the last tile have zero probability.
        # [CN] Gumbel-max：argmax(p * inv_q) 等价于从 p 归一化分布采样。

        score = prob * inv_q
        score = tl.where(vocab_mask, score, float("-inf"))
        # [CN] 分块求局部最优，再跨块比较 跨块比较得出全局最优。

        local_max, local_id = tl.max(score, axis=0, return_indices=True)

        if local_max > max_val:
            max_val = local_max
            # [CN] 局部下标要加回该分块的词表起始偏移才是真实 token id。

            recovered_id = v + local_id

    # [CN] 最后兜底夹一次：全零概率的极端情况会选出越界 id。

    recovered_id = tl.minimum(recovered_id, vocab_size - 1)
    tl.store(output_token_ids_ptr + token_idx, recovered_id)
