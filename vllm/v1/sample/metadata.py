# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：SamplingMetadata —— V1 采样链路的「参数总线」容器。
# [CN] 本文件只有数据结构、没有计算逻辑：它是 GPU Worker 上持久批次（InputBatch）与
# [CN] Sampler.forward 之间唯一的参数载体。
# [CN] 链路位置：scheduler 产出调度决策 → gpu_input_batch 组装 → 本文件打包 →
# [CN] sampler 消费 → output_processor 回传。
# [CN] 与 forward_context.py 的 ForwardContext 区分：ForwardContext 是「模型前向」的全局
# [CN] 上下文（attention metadata、slot_mapping、CUDA graph 模式…）；SamplingMetadata 是
# [CN] 「采样阶段」专用参数（温度、top_p、惩罚、logprobs、业务约束…）。
# [CN] 设计要点：多数字段是已在 GPU 上的 torch.Tensor 且每步被原地复用，因此不要对这些
# [CN] 字段做 .item() / .cpu() / Python 循环 —— 那会强制 GPU 同步，直接打穿吞吐。
# [CN] 另一类字段是 CPU 侧 Python 对象（list[list[int]]、dict[int, ...]），长度随 batch
# [CN] 变化、形状不固定，天然不适合 CUDA Graph 捕获 —— 这是本文件最本质的二分界线。
# [CN] 四个最容易看错的语义（详见各字段注释）：
# [CN]   1. all_greedy / all_random 都为 False = 混合批，sampler 必须走「先算 greedy 再
# [CN]      按温度阈值 torch.where 合并」的慢路径；上层只在能确定整批同性质时才置位。
# [CN]   2. max_num_logprobs 三态且含 -1 特例：None=不要；0=只要被采样 token 的；
# [CN]      -1=返回完整词表且不排序；n>0=top-n。三者在 sampler.forward 里是不同的分支。
# [CN]   3. allowed_token_ids_mask 语义是 True=禁用（会被 masked_fill_(-inf)），反掩码。
# [CN]   4. no_penalties=True 时整批跳过 apply_all_penalties（省一次词表级扫描）；
# [CN]      penalty 张量即使全零也常驻，靠「开关」而非「是否传参」省算子。
from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


# [CN] SamplingMetadata：一次采样步所需的全部参数，每步被 InputBatch 原地更新而非重建。
# [CN] 纯数据、无方法：所有业务逻辑都在 vllm/v1/sample/sampler.py。
# [CN] 字段按用途分三组：GPU 张量采样参数（temperature/top_p/top_k/…）、CPU 业务约束
# [CN] （list/dict）、布尔开关（all_greedy/all_random/no_penalties）；改动字段顺序前
# [CN] 必须确认与 input_batch 侧的组装顺序一致（dataclass 是位置参数敏感的）。
@dataclass
class SamplingMetadata:
    # [CN] temperature / top_p / top_k：形状 [batch, 1] 的 GPU 张量，逐请求取值。
    # [CN] None 表示整批都不启用该算子（sampler 会整段跳过）；非 None 时保留 [batch, 1]
    # [CN] 形状是为了能和 [batch, vocab] 的 logits 直接广播。
    temperature: torch.Tensor | None
    # [CN] all_greedy / all_random：批次性质标志，二者互斥，但可以同时为 False。
    # [CN] 都为 False = 混合批：sampler 走「先 greedy_sample，再按 temperature < _SAMPLING_EPS
    # [CN] 用 torch.where 合并」的慢路径；上层 scheduler 只有在能确定整批同性质时才置位，
    # [CN] 目的就是让高频路径避开这条慢路径。forward 开头有 assert 禁止二者同时为 True。
    all_greedy: bool
    all_random: bool

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None

    # [CN] req_index -> torch.Generator：每个请求独享 generator，保证「同 seed + 同请求」
    # [CN] 可复现。它是 CPU 侧 dict（generator 无法搬到 GPU），不要换成单一全局 RNG，
    # [CN] 否则请求之间会互相污染随机数序列，破坏 per-request seed 的可复现性。
    generators: dict[int, torch.Generator]

    # [CN] max_num_logprobs 三态 + -1 特例（本文件最容易踩的坑）：
    # [CN]   None = 不返回 logprobs；0 = 只要被采样 token 的 logprob；n>0 = 返回 top-n。
    # [CN]   -1 = 返回完整词表 logprobs 且不排序不取 top（见 sampler.py 中 num_logprobs == -1
    # [CN]   那个分支，构造 LogprobsTensors 时直接塞 raw_logprobs）。三者代码路径完全不同。
    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None

    # [CN] no_penalties：性能开关。True → 整批跳过 apply_all_penalties（省一次词表级扫描）。
    # [CN] 下面三个 penalty 张量即使全零也常驻，用「开关」而不是「是否传参」来省算子，
    # [CN] 避免因传参变化导致每步重建 tensor。
    no_penalties: bool
    prompt_token_ids: torch.Tensor | None
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    # [CN] output_token_ids：每个请求已生成 token 的列表，CPU 侧 list[list[int]]。
    # [CN] 长度随 batch 变化、形状不固定 → 不能被 CUDA Graph 捕获，penalties / bad words
    # [CN] 这些依赖「变长序列」的逻辑只能在 Python 侧处理。
    output_token_ids: list[list[int]]

    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    # [CN] allowed_token_ids_mask：形状 [max_batch, vocab] 的 bool 张量。
    # [CN] 注意语义是 True=禁用（会被 masked_fill_(-inf)）—— 这是「反掩码」，最容易被看反。
    # [CN] None 表示本批没有任何 allowed_token_ids 约束。
    allowed_token_ids_mask: torch.Tensor | None

    # req_index -> bad_words_token_ids
    # [CN] bad_words_token_ids：req_index -> list[token_ids 序列]。
    # [CN] 必须按「序列」匹配（多 token 禁词），因此需要 output_token_ids 参与扫描，
    # [CN] 不能只用词表级 mask 表达 —— 这也是它必须是 CPU 侧 Python 对象的原因。
    bad_words_token_ids: dict[int, list[list[int]]]

    # Loaded logits processors
    # [CN] logitsprocs：已按「是否 argmax-invariant」分好类的 logits 处理器集合。
    # [CN] 分类决定调用时机：non_argmax_invariant 必须在 greedy 之前（会改变 argmax 结果），
    # [CN] argmax_invariant 在采样之前（不改变 argmax，只缩放分布）。
    logitsprocs: LogitsProcessors

    # [CN] logprob_token_ids：只 gather 指定 token 的 logprob，比全词表 top-k 更省。
    # [CN] 由 generative scoring API 使用；命中时走 Sampler.gather_specific_token_logprobs。
    # [CN] 与 max_num_logprobs 同时给出时优先用本字段（sampler.py 有 prefer 分支）。
    # Specific token IDs to compute logprobs for (more efficient than full vocab)
    # When set, logprobs are computed only for these token IDs using gather
    # req_index -> list of token IDs to get logprobs for
    logprob_token_ids: dict[int, list[int]] | None = None

    # [CN] spec_token_ids：投机解码的草稿 token。它们不算「已提交输出」，
    # [CN] 因此与 output_token_ids 分开存；只有确实要参与 penalty/logprob 时才合并，
    # [CN] 不要主动 merge 这两个字段，否则草稿被拒时会污染惩罚项统计。
    # Speculative token ids
    spec_token_ids: list[list[int]] | None = None
    # When non-None, use ``holder.has_tracked_requests()`` to see if this batch applies
    # thinking-token-budget logits (holder may exist with an empty tracking set).
    # [CN] thinking_budget_state_holder：控制 thinking token 预算（reasoning 模型思考过长时
    # [CN] 截断/引导）。注意 holder 可能存在但追踪集为空 —— 必须用 has_tracked_requests()
    # [CN] 判断本批是否真的生效，不能直接用 is not None 代替。
    thinking_budget_state_holder: ThinkingBudgetStateHolder | None = None
