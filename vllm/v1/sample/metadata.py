# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""采样元数据模块。

本模块实现了采样相关的元数据结构，负责：
- 封装采样参数（温度、top_p、top_k 等）
- 管理采样所需的生成器状态
- 处理惩罚参数和约束条件
- 支持 logits 处理器加载

主要类：
- SamplingMetadata: 采样元数据封装
"""

from dataclasses import dataclass

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors


@dataclass
class SamplingMetadata:
    """采样元数据封装类。

    封装了采样所需的所有参数和状态信息，
    用于在采样过程中传递给采样器。

    Attributes:
        temperature: 温度张量，控制采样随机性
        all_greedy: 是否所有请求都是贪婪采样
        all_random: 是否所有请求都是随机采样
        top_p: top_p 采样参数张量
        top_k: top_k 采样参数张量
        generators: 随机数生成器字典（请求索引 -> 生成器）
        max_num_logprobs: 最大 logprobs 数量，None 表示不需要，
                         0 表示只需要采样 token 的 logprobs
        no_penalties: 是否没有惩罚
        prompt_token_ids: prompt token IDs 张量
        frequency_penalties: 频率惩罚张量
        presence_penalties: 存在惩罚张量
        repetition_penalties: 重复惩罚张量
        output_token_ids: 输出 token IDs 列表（每个请求一个列表）
        allowed_token_ids_mask: 允许的 token IDs 掩码，
                              形状为 (最大批次大小，词表大小) 的 2D bool 张量
        bad_words_token_ids: 禁用词 token IDs 字典
                           （请求索引 -> 禁用词 token IDs 列表）
        logitsprocs: 已加载的 logits 处理器
        spec_token_ids: 推测 token IDs 列表（可选）
    """

    temperature: torch.Tensor | None
    """温度张量，控制采样随机性"""

    all_greedy: bool
    """是否所有请求都是贪婪采样"""

    all_random: bool
    """是否所有请求都是随机采样"""

    top_p: torch.Tensor | None
    """top_p 采样参数张量"""

    top_k: torch.Tensor | None
    """top_k 采样参数张量"""

    generators: dict[int, torch.Generator]
    """随机数生成器字典（请求索引 -> 生成器）"""

    # None 表示不需要 logprobs，0 表示只需要采样 token 的 logprobs
    max_num_logprobs: int | None
    """最大 logprobs 数量"""

    no_penalties: bool
    """是否没有惩罚"""

    prompt_token_ids: torch.Tensor | None
    """prompt token IDs 张量"""

    frequency_penalties: torch.Tensor
    """频率惩罚张量"""

    presence_penalties: torch.Tensor
    """存在惩罚张量"""

    repetition_penalties: torch.Tensor
    """重复惩罚张量"""

    output_token_ids: list[list[int]]
    """输出 token IDs 列表（每个请求一个列表）"""

    # `allowed_token_ids_mask` 是形状为 (最大批次大小，词表大小) 的 2D bool 张量
    allowed_token_ids_mask: torch.Tensor | None
    """允许的 token IDs 掩码"""

    # 请求索引 -> 禁用词 token IDs
    bad_words_token_ids: dict[int, list[list[int]]]
    """禁用词 token IDs 字典"""

    # 已加载的 logits 处理器
    logitsprocs: LogitsProcessors
    """已加载的 logits 处理器"""

    # 推测 token IDs
    spec_token_ids: list[list[int]] | None = None
    """推测 token IDs 列表（可选）"""
