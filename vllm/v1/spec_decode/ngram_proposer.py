# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""N-gram 推测解码 proposer 模块。

本模块实现了基于 N-gram 匹配的 CPU 推测解码 proposer，负责：
- 使用 N-gram 匹配在历史 token 中查找重复模式
- 找到匹配后提取后续 token 作为草稿 token
- 使用 Numba JIT 编译加速批量处理

主要类：
- NgramProposer: 基于 N-gram 匹配的 CPU proposer

主要函数：
- batch_propose_numba: Numba 并行批量 N-gram 提议
- _find_longest_matched_ngram_and_propose_tokens: 查找最长匹配 N-gram 并提取后续 token

算法说明：
N-gram 推测解码通过在序列的历史 token 中查找与当前后缀匹配的模式，
如果找到匹配，则使用匹配位置后的 token 作为草稿 token。
这种方法无需额外的模型，计算开销低，适合重复模式多的场景。
"""

import os

import numpy as np
import torch
from numba import get_num_threads, jit, njit, prange, set_num_threads

from vllm.config import VllmConfig


class NgramProposer:
    """基于 N-gram 匹配的 CPU 推测解码 proposer。

    N-gram 推测解码通过在序列的历史 token 中查找与当前后缀匹配的模式，
    如果找到匹配，则使用匹配位置后的 token 作为草稿 token。
    这种方法无需额外的模型，计算开销低，适合重复模式多的场景。

    使用 Numba JIT 编译和并行化加速批量处理。

    Attributes:
        min_n: N-gram 最小长度
        max_n: N-gram 最大长度
        k: 匹配后跟随的 token 数量（草稿 token 数量）
        max_model_len: 模型最大长度
        valid_ngram_draft: 预分配的 N-gram 草稿缓冲区 [max_num_seqs, k]
        valid_ngram_num_drafts: 预分配的有效草稿数量缓冲区 [max_num_seqs]
        num_tokens_threshold: 启用多线程的 token 数量阈值
        num_numba_thread_available: 可用的 Numba 线程数
    """

    def __init__(self, vllm_config: VllmConfig):
        """初始化 NgramProposer。

        Args:
            vllm_config: vLLM 配置

        Raises:
            AssertionError: 如果 speculative_config、prompt_lookup_min、
                prompt_lookup_max 未设置
        """
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # N-gram 匹配的最小长度
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        # N-gram 匹配的最大长度
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # 匹配后跟随的 token 数量（草稿 token 数量）
        # 如果匹配位置后的 token 少于 k 个，则返回直到末尾的所有 token
        self.k = vllm_config.speculative_config.num_speculative_tokens
        # 模型最大长度
        self.max_model_len = vllm_config.model_config.max_model_len

        # 为 numba 批量提议预分配缓冲区
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros((max_num_seqs), dtype=np.int32)

        # 在 numba 批量提议中启用多线程的总 token 数量阈值
        self.num_tokens_threshold = 8192
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        cpu_count = os.cpu_count()
        # Numba 并行处理的最大线程数
        if cpu_count:
            # 除以 2 使用物理核心而非逻辑核心（超线程）
            # 将线程数限制为 8，以避免使用过多线程
            # 因为其他组件如前端（包括 tokenization）和结构化输出也使用多线程
            # TODO(ekagra-ranjan): 当实现 ngram 的 TP 并行化时，将上限从 1 提高到 8
            self.num_numba_thread_available = min(1, (cpu_count // 2))
            # 除以 tp_size 确保每个张量并行 rank 都有一些线程
            # 因为所有 ranks 都会运行这个
            self.num_numba_thread_available //= tp_size
        else:
            self.num_numba_thread_available = 1

        # 触发 N-gram proposer 的 Numba JIT 编译
        # 这通常需要不到 1 秒
        self.propose(
            [[]] * 1024,
            np.zeros(1024, dtype=np.int32),
            np.zeros((1024, self.max_model_len), dtype=np.int32),
        )

    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        """使用 Numba 加速的 N-gram 提议批量版本。

        Args:
            num_requests: 请求总数
            valid_ngram_requests: 需要 N-gram 提议的请求索引列表
            num_tokens_no_spec: 形状为 (batch_size,) 的数组，表示每个请求
                不包含推测 token 的 token 数量
            token_ids_cpu: 形状为 (batch_size, max_model_len) 的数组，
                表示每个请求的 token ID

        Returns:
            list[list[int]]: 每个请求的草稿 token ID 列表
        """
        draft_token_ids: list[list[int]] = []

        # 只有在有需要 N-gram 提议的请求时才运行批量提议
        # 避免用空列表调用 numba 函数导致错误 ValueError
        if num_ngram_requests := len(valid_ngram_requests):
            original_num_numba_threads = get_num_threads()
            # 确保至少使用一个线程
            # 如果总 token 数量较少，使用多线程可能因开销而变慢
            total_tokens = np.sum(num_tokens_no_spec)
            if total_tokens >= self.num_tokens_threshold:
                final_num_threads = max(
                    1, min(self.num_numba_thread_available, num_ngram_requests)
                )
                set_num_threads(final_num_threads)
            else:
                set_num_threads(1)

            # 调用 Numba 并行 kernel
            batch_propose_numba(
                valid_ngram_requests,
                num_tokens_no_spec,
                token_ids_cpu,
                self.min_n,
                self.max_n,
                self.max_model_len,
                self.k,
                self.valid_ngram_draft,
                self.valid_ngram_num_drafts,
            )

            # 恢复原始线程数
            set_num_threads(original_num_numba_threads)

        # 收集结果
        for i in range(num_requests):
            if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(
                    self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
                )
            else:
                draft_token_ids.append([])

        return draft_token_ids

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        """为每个请求生成 N-gram 草稿 token。

        Args:
            sampled_token_ids: 采样的 token ID 列表
            num_tokens_no_spec: 每个请求不包含推测 token 的 token 数量
            token_ids_cpu: 每个请求的 token ID 数组
            slot_mappings: 槽映射（未使用）

        Returns:
            每个请求的草稿 token ID 列表
        """
        # 找出需要 N-gram 提议的请求
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # 跳过推测解码
                continue

            num_tokens = num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # 跳过已达到最大模型长度的请求
                continue

            valid_ngram_requests.append(i)

        # 批量提议
        draft_token_ids = self.batch_propose(
            len(sampled_token_ids),
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
        )

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        """加载模型（N-gram proposer 无需加载模型）。"""
        # 无需加载模型
        pass


@njit(parallel=True)
def batch_propose_numba(
    valid_ngram_requests: list,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    """Numba 并行批量 N-gram 提议 kernel。

    使用 prange 并行处理多个请求，每个线程独立处理一个请求。

    Args:
        valid_ngram_requests: 需要 N-gram 提议的请求索引列表
        num_tokens_no_spec: 每个请求不包含推测 token 的 token 数量
        token_ids_cpu: 每个请求的 token ID 数组
        min_n: N-gram 最小长度
        max_n: N-gram 最大长度
        max_model_len: 模型最大长度
        k: 最大草稿 token 数量
        valid_ngram_draft: 输出缓冲区，存储每个请求的草稿 token
        valid_ngram_num_drafts: 输出缓冲区，存储每个请求的有效草稿数量
    """
    for i in prange(len(valid_ngram_requests)):
        idx = valid_ngram_requests[i]
        num_tokens = num_tokens_no_spec[idx]
        # 获取上下文 token ID
        context_token_ids = token_ids_cpu[idx, :num_tokens]
        # 查找最长匹配的 N-gram 并提议后续 token
        drafter_output = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=context_token_ids,
            min_ngram=min_n,
            max_ngram=max_n,
            max_model_len=max_model_len,
            k=k,
        )

        # 存储结果到输出缓冲区
        valid_ngram_num_drafts[idx] = drafter_output.shape[0]
        if len(drafter_output):
            valid_ngram_draft[idx, : drafter_output.shape[0]] = drafter_output


@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """查找与给定 token 后缀匹配的最长 N-gram 并提议后续 token。

    查找与长度为 [min_ngram, max_ngram]（包含）的后缀匹配的 N-gram。
    如果找到，提取匹配 N-gram 后的 k 个 token。

    算法使用 KMP（Knuth-Morris-Pratt）算法的变体，通过构建
    最长前缀后缀（LPS）数组来高效查找匹配。

    Args:
        origin_tokens: 原始 token 序列
        min_ngram: N-gram 最小长度
        max_ngram: N-gram 最大长度
        max_model_len: 模型最大长度
        k: 最大草稿 token 数量

    Returns:
        提议的草稿 token 数组，如果没有匹配则返回空数组
    """
    # 如果上下文短于最小 N-gram，则不生成草稿 token
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # 不生成超出最大模型长度的草稿 token
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # 翻转 token，目标变为在右侧查找最长 N-gram
    # 使其与长度为 [min_n, max_n]（包含）的前缀匹配
    tokens = origin_tokens[::-1]

    # 最长前缀（不包括自身）是当前后缀的算法
    # lps[i] = max{v, tokens[0:v] == tokens[i+1-v:i+1]}
    # 由于 ngram 受 max_ngram 限制以节省内存，我们只需要
    # 存储前 max_ngram 个前缀的 lps
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    # lps[0] 始终为 0，从索引 1 开始
    prev_lps = 0
    i = 1
    while i < total_token:
        # tokens[:prev_lps] 是 tokens[:i] 后缀的最长前缀
        if tokens[prev_lps] == tokens[i]:
            # Token 匹配：tokens[:prev_lps+1] 是 tokens[:i+1] 后缀的最长前缀
            prev_lps += 1
            # 检查是否找到更长的有效 ngram
            # 当 longest_ngram 匹配 prev_lps 时更新 position，
            # 因为我们想要在原始 token 中最早的 position
            # （即在翻转 token 中最新的位置）
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                # 存储前 max_ngram 个前缀的 LPS
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                # 当 prev_lps 达到 max_ngram 时，更新 prev_lps 为 lps[max_ngram-1]
                # 以避免匹配超过 max_ngram 的 ngram
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            # Token 不匹配：尝试 tokens[:i] 所有后缀中的第二长前缀
            # 也就是 tokens[:prev_lps] 的最长前缀
            prev_lps = lps[prev_lps - 1]
        else:
            # Token 不匹配，且没有更多前缀（空字符串除外）
            # 作为 tokens[:i] 的后缀
            i += 1

    if longest_ngram < min_ngram:
        # 没有找到有效的 ngram
        return np.empty((0,), dtype=origin_tokens.dtype)

    # 翻转回原始位置，所以在 origin_tokens 中，
    # origin_tokens[total_token-1-position:total_token-1-position+longest_ngram]
    # 是匹配的 ngram，所以我们应该从
    # total_token-1-position+longest_ngram 开始起草 token
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]
