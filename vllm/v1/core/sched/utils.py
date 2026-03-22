# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""调度工具函数模块。

本模块提供调度相关的辅助函数，负责：
- 检测重复序列模式
- 检查停止条件
- 从列表中移除多个元素

主要函数：
- _has_repeating_pattern: 检查 token 序列是否有重复模式
- check_sequence_repetition: 检查序列重复
- remove_all: 从列表中移除多个元素
- check_stop: 检查请求是否应该停止
"""

import contextlib
from collections.abc import Sequence

from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.request import Request, RequestStatus


def _has_repeating_pattern(
    token_ids: Sequence[int],
    pattern_len: int,
    repetition_min_count: int,
) -> bool:
    """检查 token_ids 的尾部是否有重复模式。

    比较最后 pattern_len 个 token 与前面 (repetition_min_count - 1) 次
    重复的相同长度的 token。

    Args:
        token_ids: token ID 序列
        pattern_len: 模式长度
        repetition_min_count: 最小重复次数

    Returns:
        如果检测到重复模式则返回 True
    """
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
) -> bool:
    """检查 token ID 序列是否有重复模式。

    用于检测模型生成的文本是否存在重复循环，如 "hello hello hello..."。

    Args:
        token_ids: token ID 列表
        params: 重复检测参数

    Returns:
        如果检测到重复模式则返回 True
    """
    max_pattern_size = params.max_pattern_size
    min_pattern_size = params.min_pattern_size
    min_count = params.min_count

    if min_pattern_size <= 0:
        min_pattern_size = 1

    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False

    for pattern_len in range(
        min_pattern_size,
        max_pattern_size + 1,
    ):
        if pattern_len * min_count > len(token_ids):
            return False

        if _has_repeating_pattern(token_ids, pattern_len, min_count):
            return True

    return False


def remove_all(lst: list, items_to_remove: set) -> list:
    """从列表中移除所有在 items_to_remove 集合中的元素。

    此方法针对移除单个元素的常见情况进行了优化，
    对于多个元素的情况回退到列表推导。

    Args:
        lst: 要移除元素的列表
        items_to_remove: 要移除的元素集合

    Returns:
        修改后的原始列表（单个元素移除）或
        新列表（多个元素移除）。调用者应该使用返回值。

    Note:
        对于单个元素移除，此方法原地修改列表并返回它。
        对于多个元素，它创建并返回新列表。
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # 单个元素移除的快速路径（最常见的情况）
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # 对于多个元素，使用列表推导
    return [item for item in lst if item not in items_to_remove]


def check_stop(request: Request, max_model_len: int) -> bool:
    """检查请求是否应该停止生成。

    检查以下停止条件：
    1. 是否达到 min_tokens
    2. 是否生成 EOS token
    3. 是否生成 stop token
    4. 是否达到最大模型长度或最大输出 token 数
    5. 是否检测到重复模式

    Args:
        request: 要检查的请求
        max_model_len: 最大模型长度

    Returns:
        如果请求应该停止则返回 True
    """
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    # 检查是否达到最小生成 token 数
    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    # 检查是否生成 EOS token
    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    # 检查是否生成 stop token
    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True

    # 检查是否达到长度限制
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    # 检查是否检测到重复模式
    repetition_detection = sampling_params.repetition_detection
    if repetition_detection is not None and (
        check_sequence_repetition(
            request.output_token_ids,
            repetition_detection,
        )
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False
