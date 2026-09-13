# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that samples the next tokens from the model's outputs."""

# [CN] 文件总览：Sampler —— 从模型输出 logits 采样出下一个 token 的那一层。
# [CN] 链路位置：model forward 产出 logits → 本文件加工(logits processors/惩罚)并采样 →
# [CN] SamplerOutput 交给 gpu_model_runner / output_processor。
# [CN] 输入参数全部来自 v1/sample/metadata.py 的 SamplingMetadata（纯数据容器）。
# [CN] 类 docstring 里那份 1~9 步是**权威执行顺序**，改动代码前请先对照它。
# [CN] 五个最容易看错的点：
# [CN]   1. top-k logprobs 用的是**未经惩罚和温度缩放的原始 logits**（见 forward 里的
# [CN]      NOTE），这与 V0 sampler 用加工后 logits 的做法不同 —— 语义差异的根源。
# [CN]   2. 几乎所有加工都是 in-place（masked_fill_ / div_），logits 会被就地改写；
# [CN]      上层若要保留原始 logits 必须先 clone（logprobs_mode=raw_logits 就是这么做的）。
# [CN]   3. 混合批（all_greedy / all_random 都为 False）走「先 greedy 再 torch.where 按
# [CN]      温度阈值合并」的慢路径，并用 out=greedy_sampled 复用张量避免额外分配。
# [CN]   4. sampled 的类型要来回转换：FlashInfer 返回 int32 → 先 long() 才能当索引 →
# [CN]      用完后转回 int32 减小张体积（回传与落盘都按 int32）。
# [CN]   5. logprobs 有四种 mode：raw_logprobs / raw_logits / processed_logits /
# [CN]      processed_logprobs，取的是 pipeline 不同位置的量，不能混用。
import torch
import torch.nn as nn

from vllm.config.model import LogprobsMode
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

# [CN] 温度阈值：temperature < _SAMPLING_EPS 视为 greedy（温度≈0）。
# [CN] 它同时用于「避免除零」和「混合批里挑选 greedy 结果」两处判断。
_SAMPLING_EPS = 1e-5


# [CN] Sampler 本身是无参数 nn.Module（子模块 TopKTopPSampler 也无参数），
# [CN] 因此不参与权重加载；它只是把「采样算法」组织成可被 torch.compile 捕获的形式。
class Sampler(nn.Module):
    """
    A layer that samples the next tokens from the model's outputs
    with the following steps in order:

    1. If logprobs are requested:
        a) If `logprobs_mode` is `raw_logprobs`, compute logprobs
           as the final logprobs to return.
        b) If `logprobs_mode` is `raw_logits`, clone the logits
           as the final logprobs to return.
    2. Convert logits to float32.
    3. Apply allowed token ids whitelist.
    4. Apply bad words exclusion.
    5. Apply logit processors which are not argmax-invariant,
       i.e. that can impact greedy sampling.
        a) Min tokens processor
        b) Logit bias processor
    6. Apply penalties
        a) Repetition penalty
        b) Frequency penalty
        c) Presence penalty
    7. Sample the next tokens. `sample` method performs the following steps:
        a) If not `all_random`, perform greedy sampling. If `all_greedy`,
           return the greedily sampled tokens and final logprobs if requested.
        b) Apply temperature.
        c) Apply logit processors which are argmax-invariant, by default
           the min_p processor.
        d) Apply top_k and/or top_p.
        e) Sample the next tokens with the probability distribution.
        f) If `all_random` or temperature >= epsilon (1e-5), return the
           randomly sampled tokens and final logprobs if requested. Else,
           return the greedily sampled tokens and logprobs if requested.
    8. Gather the logprobs of the top `max_num_logprobs` and sampled token
       (if requested). Note that if the sampled token is within the top
       `max_num_logprobs`, the logprob will be eventually merged in
       `LogprobsProcessor` during output processing. Therefore, the
       final output may contain either `max_num_logprobs + 1` or
       `max_num_logprobs` logprobs.
    9. Return the final `SamplerOutput`.
    """

    # [CN] logprobs_mode 决定返回哪种 logprobs（见文件头第 5 点）；
    # [CN] use_fp64_gumbel 控制 gumbel-max 采样是否用 float64（影响随机质量与速度）。
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
    ):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler(logprobs_mode, use_fp64_gumbel)
        self.pin_memory = PIN_MEMORY
        self.logprobs_mode = logprobs_mode
        self.use_fp64_gumbel = use_fp64_gumbel

    # [CN] forward 主流程，严格对应类 docstring 的 1~9 步。
    # [CN] predict_bonus_token：投机解码下本步要额外产出 bonus token 时置位。
    # [CN] logprobs_mode_override：允许调用方单步覆盖默认 mode（如 spec decode 场景）。
    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        # [CN] 注意：raw_logprobs 取自**原始 logits**（惩罚/温度之前），与 V0 不同。
        # [CN] 只有在确实需要 logprobs 时才计算 —— log_softmax 是整词表扫描，代价很高。
        num_logprobs = sampling_metadata.max_num_logprobs
        raw_logprobs: torch.Tensor | None = None
        if num_logprobs is not None or sampling_metadata.logprob_token_ids:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                if logits.dtype == torch.float32:
                    raw_logprobs = logits.clone()
                else:
                    raw_logprobs = logits.to(torch.float32)

        # [CN] 统一转 float32：后续 topk / softmax / penalties 都在 fp32 下做，
        # [CN] 避免 bf16 精度不足导致采样分布失真（这里发生一次显存拷贝）。
        # Use float32 for the logits.
        logits = logits.to(torch.float32)

        # [CN] 第 3~6 步集中在这里：allowed token ids → bad words →
        # [CN] non_argmax_invariant 处理器 → penalties。全部 in-place。
        logits = self.apply_logits_processors(
            logits, sampling_metadata, predict_bonus_token
        )
        # [CN] 第 7 步。返回的第二个值可能是「加工后的 logprobs」（processed_* 模式）。
        # Sample the next token.
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        # [CN] 转 int64 才能当作 gather / index 的下标：FlashInfer 采样返回 int32，
        # [CN] 而 PyTorch 的 argmax / topk 返回 int64，这里统一到 int64。
        sampled = sampled.long()

        # Handle logprob_token_ids if specified (more efficient than full vocab)
        # This is used by generative_scoring API to get logprobs for specific tokens
        # [CN] 第 8 步分支 A：只 gather 指定 token 的 logprob（generative scoring API）。
        # [CN] 它比全词表 top-k 更省，因为只回传用户点名的那些 token。
        logprob_token_ids_tensors = None
        if sampling_metadata.logprob_token_ids:
            assert raw_logprobs is not None
            logprob_token_ids_tensors = self.gather_specific_token_logprobs(
                raw_logprobs, sampling_metadata.logprob_token_ids, sampled
            )

        # [CN] 三分支入口：None=不要 logprobs（除非有 logprob_token_ids）；-1=完整词表；
        # [CN] 其余=top-k + 被采样 token。
        if num_logprobs is None:
            logprobs_tensors = logprob_token_ids_tensors
        # [CN] -1 特例：返回完整词表、不排序、不取 top。索引与 rank 都塞 torch.empty(0)，
        # [CN] 由下游 LogprobsProcessor 自行处理。注意这里 raw_logprobs 可能为 None
        # [CN] （既没开 logprobs 又传了 -1 的非法组合由上游保证不出现）。
        elif num_logprobs == -1:
            # Return the full unsorted and unranked logprobs.
            logprobs_tensors = LogprobsTensors(
                torch.empty(0), raw_logprobs, torch.empty(0)
            )
        else:
            # Gather the logprobs and ranks of the topk and sampled token.
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )

        # [CN] 两者同时存在时以更具体的 logprob_token_ids 为准，覆盖上面的 top-k 结果。
        # If we have both num_logprobs and logprob_token_ids, prefer
        # logprob_token_ids as it's more specific
        if logprob_token_ids_tensors is not None and num_logprobs is not None:
            logprobs_tensors = logprob_token_ids_tensors

        # Use int32 to reduce the tensor size.
        # [CN] 回传前转回 int32：减小张体积、降低回传带宽（CPU 侧再做类型适配）。
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    # [CN] 为「异构长度」的 token 列表做 gather：各请求要的 token 数不同，
    # [CN] 做法是 pad 到 max 长度 + valid_mask，pad 位填 -inf。
    # [CN] 第 0 列固定放「被采样的 token」，因此矩阵宽度是 max_num_tokens + 1。
    def gather_specific_token_logprobs(
        self,
        logprobs: torch.Tensor,
        logprob_token_ids: dict[int, list[int]],
        sampled: torch.Tensor,
    ) -> LogprobsTensors | None:
        """Gather logprobs for specific token IDs requested per request.

        Used by the generative_scoring API to return logprobs for an explicit
        set of token ids rather than the top-k. Handles heterogeneous token
        id lists across requests by padding shorter lists to the max length.

        Args:
            logprobs: [batch_size, vocab_size] tensor of (raw) logprobs to
                gather from.
            logprob_token_ids: dict mapping req_index -> list of token IDs
            sampled: [batch_size] tensor of sampled token IDs

        Returns:
            LogprobsTensors with logprobs for the specified tokens, or None
            if no requests have logprob_token_ids.
        """
        if not logprob_token_ids:
            return None

        batch_size = logprobs.shape[0]
        device = logprobs.device

        # Find max number of tokens across all requests
        max_num_tokens = max(len(tids) for tids in logprob_token_ids.values())
        pin = self.pin_memory

        # Build the padded token_ids and valid_mask matrices on pinned CPU,
        # then upload non-blocking.
        token_ids_cpu = torch.zeros(
            batch_size, max_num_tokens + 1, dtype=torch.int64, pin_memory=pin
        )
        # Create mask for valid positions (True = valid, False = padded)
        valid_mask_cpu = torch.zeros(
            batch_size, max_num_tokens + 1, dtype=torch.bool, pin_memory=pin
        )
        valid_mask_cpu[:, 0] = True  # Sampled token is always valid
        for req_idx, token_ids in logprob_token_ids.items():
            num_tokens = len(token_ids)
            token_ids_cpu[req_idx, 1 : num_tokens + 1] = torch.as_tensor(
                token_ids, dtype=torch.int64
            )
            valid_mask_cpu[req_idx, 1 : num_tokens + 1] = True

        # [CN] 索引矩阵在 pinned CPU 内存上构建后异步上传；第 0 列再在 GPU 上直接用
        # [CN] sampled 填充，避免一次 D2H + 重新上传的来回。
        token_ids_tensor = token_ids_cpu.to(device, non_blocking=True)
        valid_mask = valid_mask_cpu.to(device, non_blocking=True)
        # Sampled token in column 0 — fill on-device from the sampled GPU
        # tensor so we don't need to D2H + re-upload.
        token_ids_tensor[:, 0] = sampled

        # Gather logprobs at the requested token ids.
        # [CN] 一次 gather 拿到所有请求点名 token 的 logprob，pad 位随后被置为 -inf。
        gathered_logprobs = logprobs.gather(-1, token_ids_tensor)

        # Mask invalid (padded) positions with -inf
        gathered_logprobs = gathered_logprobs.masked_fill(~valid_mask, float("-inf"))

        # Compute ranks for the sampled token. log_softmax is monotonic w.r.t.
        # the original logits, so ranks computed from logprobs are equivalent.
        sampled_logprobs = logprobs.gather(-1, sampled.unsqueeze(-1))
        # Avoid 0/1 specialization recompile on the batch dimension of the
        # compiled batched_count_greater_than. See gather_logprobs for context.
        torch._dynamo.decorators.mark_unbacked(logprobs, 0)
        torch._dynamo.decorators.mark_unbacked(sampled_logprobs, 0)
        token_ranks = batched_count_greater_than(logprobs, sampled_logprobs)

        return LogprobsTensors(
            logprob_token_ids=token_ids_tensor.to(torch.int32),
            logprobs=gathered_logprobs,
            selected_token_ranks=token_ranks,
        )

    @staticmethod
    # [CN] 就地除以温度（div_）。若批内存在 greedy 请求（all_random=False），
    # [CN] 先把 < eps 的温度钳成 1.0，既避免除零、也让 greedy 行数值不变。
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        # Avoid division by zero if there are greedy requests.
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    @staticmethod
    # [CN] argmax 后 view(-1)：把 [batch, 1] 压回 [batch]，保持返回形状稳定。
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    # [CN] 第 7 步的实现。三种情形：
    # [CN]   all_random → 只走随机路径；all_greedy → 算完 greedy 立即返回（最快）；
    # [CN]   两者皆 False → 混合批，两条路径都算完再用 torch.where 合并。
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        # [CN] 两个标志互斥：同时为 True 说明上层组装 metadata 时算错了批次性质。
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        # [CN] all_random=True 时完全跳过 greedy（省一次 argmax）；否则先算 greedy 备着，
        # [CN] 因为混合批最终要用 torch.where 在 greedy / random 之间逐行挑选。
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if (
                    sampling_metadata.max_num_logprobs is not None
                    or sampling_metadata.logprob_token_ids
                ):
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        # [CN] 只要不是纯 greedy，温度张量必须存在（否则无法区分该行走 greedy 还是随机）。
        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        # [CN] argmax-invariant 处理器（默认 min_p）只影响随机采样、不改变 argmax，
        # [CN] 因此放在温度之后、top-k/top-p 之前；与 non_argmax_invariant 的时机不同。
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Apply top_k and/or top_p.
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        # [CN] 纯随机批（all_random）直接返回；下面是混合批的合并逻辑。
        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        # [CN] 混合批合并：温度 < eps 的行取 greedy，其余取随机。
        # [CN] out=greedy_sampled 是刻意的张量复用 —— 否则每步都要多分配一份输出。
        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, processed_logprobs

    @staticmethod
    # [CN] log_softmax(dtype=float32)：即便输入是 bf16 也强制用 fp32 计算，
    # [CN] 保证 logprobs 数值精度（代价是一次词表级扫描）。
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    # [CN] 标准 top-k logprobs：topk → 拼上被采样 token → 计算 rank。
    # [CN] 输出宽度是 num_logprobs + 1（多出来的一列就是被采样 token 自己）。
    def gather_logprobs(
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: maximum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        # [CN] 词表级 topk，是整条 logprobs 路径里最贵的一步；所以上层尽量用
        # [CN] no_penalties / logprob_token_ids 等开关绕开全词表操作。
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        # Avoid 0/1 specialization recompile on the batch dimension
        # of the compiled batched_count_greater_than. mark_unbacked makes
        # the size fully symbolic so dynamo doesn't specialize when
        # batch_size transitions from 1 to >=2.
        # [CN] rank 计算需要一次词表级比较，这里显式允许「仅首次」的 GPU 同步；
        # [CN] mark_unbacked 把 batch 维标记为符号化，避免 batch_size 从 1 变到 >=2 时
        # [CN] dynamo 触发 0/1 特化而重新编译（典型冷启动抖动来源）。
        with gpu_sync_allowed(first_only=True):
            torch._dynamo.decorators.mark_unbacked(logprobs, 0)
            torch._dynamo.decorators.mark_unbacked(token_logprobs, 0)
            token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    @staticmethod
    # [CN] 把投机解码的草稿 token 追加到已输出序列尾部，供 penalties / bad words 使用。
    # [CN] 只在 predict_bonus_token 且有惩罚/禁词时才调用 —— 草稿尚未被验证，
    # [CN] 平时不能混进 output_token_ids，否则被拒的草稿会污染统计。
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        if spec_token_ids is None:
            return output_token_ids

        return [
            [*out, *spec] if spec else out
            for out, spec in zip(output_token_ids, spec_token_ids)
        ]

    # [CN] 第 3~6 步：allowed mask → bad words → non_argmax_invariant → penalties
    # [CN] → thinking budget。注意顺序不可换：allowed/bad words 用 -inf 屏蔽，
    # [CN] 必须在 log_softmax 之类会「抹平 -inf 影响」的操作之前完成。
    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        any_penalties_or_bad_words = (
            bool(bad_words_token_ids) or not sampling_metadata.no_penalties
        )
        output_token_ids = sampling_metadata.output_token_ids
        # [CN] 只有确实需要惩罚/禁词时才做这次 list 拼接（纯 CPU 侧、变长，有开销）。
        if predict_bonus_token and any_penalties_or_bad_words:
            # Combine base outputs with spec tokens when speculative decoding
            # is enabled.
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # [CN] mask 语义是 True=禁用 → masked_fill_(-inf)（反掩码，见 metadata.py 注释）。
        # Apply allowed token ids.
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids:
            apply_bad_words(logits, bad_words_token_ids, output_token_ids)

        # Apply logits processors which can impact greedy sampling.
        # [CN] 这些处理器会改变 argmax（如 min_tokens、logit bias），必须在 greedy 之前生效，
        # [CN] 否则 all_greedy 的快路径会拿到错误结果。
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        # Apply penalties (e.g., freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
        holder = sampling_metadata.thinking_budget_state_holder
        if holder is not None and holder.has_tracked_requests():
            # Committed outputs only; spec drafts live in ``spec_token_ids``.
            holder.update_state(
                sampling_metadata.output_token_ids,
                sampling_metadata.spec_token_ids,
                repeat_indices=None,
            )
            logits = holder.apply_to_logits(
                logits,
                predict_bonus_token,
                sampling_metadata.spec_token_ids,
            )
        return logits

    @staticmethod
    # [CN] no_penalties 为 True 时直接返回（省一次词表级扫描）。
    # [CN] 惩罚同时考虑 prompt 与 output 两侧的 token 计数，因此需要 prompt_token_ids。
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
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
