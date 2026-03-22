# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""禁用词处理工具函数模块。

本模块实现了禁用词（bad words）处理功能，负责：
- 在生成过程中屏蔽禁用词
- 基于已生成 token 的前缀匹配
- 支持推测解码场景的批量处理

主要函数：
- _apply_bad_words_single_batch: 单个批次的禁用词处理
- apply_bad_words: 批量禁用词处理
- apply_bad_words_with_drafts: 支持推测解码的禁用词处理
"""

import torch


def _apply_bad_words_single_batch(
    logits: torch.Tensor,
    bad_words_token_ids: list[list[int]],
    past_tokens_ids: list[int],
) -> None:
    """对单个批次应用禁用词约束。

    对于每个禁用词序列，检查已生成 token 的后缀是否匹配禁用词的前缀。
    如果匹配，则将禁用词的下一个 token 的 logit 设为负无穷。

    Args:
        logits: 单个样本的 logits 张量
        bad_words_token_ids: 禁用词 token IDs 列表
        past_tokens_ids: 已生成的 token IDs 列表
    """
    for bad_word_ids in bad_words_token_ids:
        if len(bad_word_ids) > len(past_tokens_ids) + 1:
            continue

        prefix_length = len(bad_word_ids) - 1
        last_token_id = bad_word_ids[-1]
        actual_prefix = past_tokens_ids[-prefix_length:] if prefix_length > 0 else []
        expected_prefix = bad_word_ids[:prefix_length]

        assert len(actual_prefix) == len(expected_prefix)

        if actual_prefix == expected_prefix:
            logits[last_token_id] = float("-inf")


def apply_bad_words(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
) -> None:
    """批量应用禁用词约束。

    Args:
        logits: logits 张量，形状为 (batch_size, vocab_size)
        bad_words_token_ids: 请求索引 -> 禁用词 token IDs 列表的字典
        past_tokens_ids: 每个请求的已生成 token IDs 列表
    """
    for i, bad_words_ids in bad_words_token_ids.items():
        _apply_bad_words_single_batch(logits[i], bad_words_ids, past_tokens_ids[i])


def apply_bad_words_with_drafts(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
    num_draft_tokens: list[int],
) -> None:
    """支持推测解码的禁用词处理。

    在推测解码场景中，为每个 draft token 位置应用禁用词约束。

    Args:
        logits: logits 张量
        bad_words_token_ids: 请求索引 -> 禁用词 token IDs 列表的字典
        past_tokens_ids: 每个请求的已生成 token IDs 列表
        num_draft_tokens: 每个请求的 draft token 数量列表
    """
    start_idx = 0
    remaining = len(bad_words_token_ids)
    for i, n in enumerate(num_draft_tokens):
        if (bad_words_ids := bad_words_token_ids.get(i)) is not None:
            for draft_idx in range(start_idx, start_idx + n):
                _apply_bad_words_single_batch(
                    logits[draft_idx],
                    bad_words_ids,
                    past_tokens_ids[draft_idx],
                )
            remaining -= 1
            if not remaining:
                break
        start_idx += n
