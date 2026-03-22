# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""惩罚应用工具函数模块。

本模块实现了采样惩罚相关工具函数，负责：
- 应用频率惩罚、存在惩罚和重复惩罚
- 将输出 token 列表转换为张量格式
- 支持异步调度场景的占位符处理

主要函数：
- apply_all_penalties: 应用所有类型的惩罚
- _convert_to_tensors: 转换输出 token 为张量
"""

import torch

from vllm.model_executor.layers.utils import apply_penalties
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """应用存在惩罚、频率惩罚和重复惩罚到 logits。

    这些惩罚用于控制生成文本的多样性：
    - 存在惩罚：对出现过的 token 施加惩罚
    - 频率惩罚：根据 token 出现频率施加惩罚
    - 重复惩罚：对重复 token 施加惩罚

    Args:
        logits: 输入的 logits 张量，形状为 (batch_size, vocab_size)
        prompt_token_ids: prompt token IDs 张量
        presence_penalties: 存在惩罚张量
        frequency_penalties: 频率惩罚张量
        repetition_penalties: 重复惩罚张量
        output_token_ids: 输出 token IDs 列表（每个请求一个列表）

    Returns:
        应用惩罚后的 logits 张量
    """
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)

    # 在异步调度情况下，不会应用惩罚的行可能包含 -1 占位符 token IDs。
    # 我们必须将这些替换为有效的 token IDs，以便 apply_penalties 中的
    # scatter 操作是有效的。
    # 注意 (nick): 惩罚实现目前效率较低，将来会重新设计。
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

    return apply_penalties(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )


def _convert_to_tensors(
    output_token_ids: list[list[int]], vocab_size: int, device: torch.device
) -> torch.Tensor:
    """将不同的列表数据结构转换为张量。

    使用 vocab_size 的值作为填充值，因为我们没有这个值的 token_id。

    Args:
        output_token_ids: 输出 token IDs 列表（每个请求一个列表）
        vocab_size: 词表大小
        device: 要放置张量的设备

    Returns:
        填充后的输出 token IDs 张量
    """
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        # 使用 vocab_size 的值作为填充，因为我们没有这个值的 token_id
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(),
    )
    return output_tokens_tensor.to(device, non_blocking=True)
