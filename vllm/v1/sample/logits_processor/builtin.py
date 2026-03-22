# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""内置 Logits 处理器模块。

本模块实现了 vLLM 内置的 logits 处理器，负责：
- MinP 采样：基于相对概率阈值的采样约束
- Logit Bias：对特定 token 添加偏置值
- Min Tokens：最小 token 数量约束，防止过早结束

主要类：
- MinPLogitsProcessor: MinP 采样处理器
- LogitBiasLogitsProcessor: Logit 偏置处理器
- MinTokensLogitsProcessor: 最小 token 处理器
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

T = TypeVar("T")


class MinPLogitsProcessor(LogitsProcessor):
    """MinP 采样 logits 处理器。

    MinP 采样基于相对概率阈值进行约束：
    - 计算每个序列的最大概率
    - 将 min_p 乘以最大概率得到动态阈值
    - 过滤掉概率低于阈值的 token

    这种方法的优点是阈值会随着分布的峰值变化而调整，
    在不同置信度下都能保持合理的过滤效果。
    """

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        """初始化 MinP 处理器。

        Args:
            vllm_config: vLLM 配置
            device: 要使用的设备
            is_pin_memory: 是否使用锁页内存
        """
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.min_p_count: int = 0
        """使用 MinP 的请求数量"""

        self.min_p_cpu_tensor = torch.zeros(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=is_pin_memory
        )
        """CPU 上的 min_p 张量"""
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()
        """CPU 上的 min_p numpy 数组"""

        self.use_double_tensor = torch.device(device).type != "cpu"
        """是否使用双张量（CPU 和设备各一个）"""

        if self.use_double_tensor:
            # 预分配的设备张量
            self.min_p_device: torch.Tensor = torch.empty(
                (max_num_reqs,), dtype=torch.float32, device=device
            )
        else:
            self.min_p_device = self.min_p_cpu_tensor
        # 设备张量的当前切片
        self.min_p: torch.Tensor = self.min_p_device[:0]
        """当前使用的 min_p 张量切片"""

    def is_argmax_invariant(self) -> bool:
        """MinP 从不影响贪婪采样。

        Returns:
            总是返回 True
        """
        return True

    def get_min_p_by_index(self, index: int) -> float:
        """根据索引获取 min_p 值。

        Args:
            index: 请求索引

        Returns:
            min_p 值
        """
        return float(self.min_p_cpu[index])

    def update_state(self, batch_update: BatchUpdate | None):
        """更新状态以反映批次变化。

        Args:
            batch_update: 批次更新信息
        """
        if not batch_update:
            return

        needs_update = False
        # 处理添加的请求。
        for index, params, _, _ in batch_update.added:
            min_p = params.min_p
            min_p_before = self.min_p_cpu[index]
            if min_p_before != min_p:
                needs_update = True
                self.min_p_cpu[index] = min_p
                if min_p and not min_p_before:
                    self.min_p_count += 1
                elif not min_p and min_p_before:
                    self.min_p_count -= 1

        if self.min_p_count:
            # 处理移除的请求。
            if batch_update.removed:
                needs_update = True
                for index in batch_update.removed:
                    if self.min_p_cpu[index]:
                        self.min_p_cpu[index] = 0
                        self.min_p_count -= 1

            # 处理移动的请求，单向 (a->b) 和交换 (a<->b)。
            for adx, bdx, direct in batch_update.moved:
                min_p_a, min_p_b = self.min_p_cpu[adx], self.min_p_cpu[bdx]
                if min_p_a != min_p_b:
                    needs_update = True
                    self.min_p_cpu[bdx] = min_p_a
                    if direct == MoveDirectionality.SWAP:
                        self.min_p_cpu[adx] = min_p_b
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    if min_p_a:
                        self.min_p_cpu[adx] = 0
                    if min_p_b:
                        self.min_p_count -= 1

        # 根据需要更新张量。
        size = batch_update.batch_size
        if self.min_p_count and (needs_update or self.min_p.shape[0] != size):
            self.min_p = self.min_p_device[:size]
            if self.use_double_tensor:
                self.min_p.copy_(self.min_p_cpu_tensor[:size], non_blocking=True)
            self.min_p.unsqueeze_(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """应用 MinP 约束到 logits。

        Args:
            logits: 输入的 logits 张量

        Returns:
            处理后的 logits 张量
        """
        if not self.min_p_count:
            return logits

        # 将 logits 转换为概率分布
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # 计算每个序列的最大概率
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        # 调整 min_p
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # 使用阈值比较识别有效 token
        invalid_token_mask = probability_values < adjusted_min_p
        # 应用掩码
        logits.masked_fill_(invalid_token_mask, -float("inf"))
        return logits


class LogitBiasLogitsProcessor(LogitsProcessor):
    """Logit 偏置处理器。

    允许对特定 token 添加偏置值，影响其被选中的概率。
    可以用于引导生成方向或限制某些 token 的使用。
    """

    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        """初始化 Logit 偏置处理器。

        Args:
            _: vLLM 配置（未使用）
            device: 要使用的设备
            is_pin_memory: 是否使用锁页内存
        """
        self.device = device
        self.pin_memory = is_pin_memory
        self.biases: dict[int, dict[int, float]] = {}
        """请求索引 -> (token 索引 -> 偏置值) 的字典"""

        self.bias_tensor: torch.Tensor = torch.tensor(())
        """偏置值张量"""
        self.logits_slice = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )
        """logits 切片索引（请求索引，token 索引）"""

    def is_argmax_invariant(self) -> bool:
        """Logit 偏置可以重新平衡 token 概率并改变贪婪采样中 argmax 的结果。

        Returns:
            返回 False（会影响 argmax）
        """
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        """更新状态以反映批次变化。

        Args:
            batch_update: 批次更新信息
        """
        needs_update = process_dict_updates(
            self.biases, batch_update, lambda params, _, __: params.logit_bias or None
        )

        # 根据需要更新张量。
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            biases: list[float] = []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = self._device_tensor(biases, torch.float32)
            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        """创建设备张量。

        Args:
            data: 数据列表
            dtype: 数据类型

        Returns:
            设备上的张量
        """
        return torch.tensor(
            data, device="cpu", dtype=dtype, pin_memory=self.pin_memory
        ).to(device=self.device, non_blocking=True)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """应用 logit 偏置。

        Args:
            logits: 输入的 logits 张量

        Returns:
            处理后的 logits 张量
        """
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


class MinTokensLogitsProcessor(LogitsProcessor):
    """最小 Token 数量处理器。

    确保生成了至少指定数量的 token 后才允许生成停止 token。
    用于防止过早结束生成。
    """

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        """初始化最小 token 处理器。

        Args:
            vllm_config: vLLM 配置
            device: 要使用的设备
            is_pin_memory: 是否使用锁页内存
        """
        # 索引 -> (min_toks, output_token_ids, stop_token_ids)
        self.device = device
        self.pin_memory = is_pin_memory
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}
        """请求索引 -> (最小 token 数，输出 token IDs, 停止 token IDs)"""

        # (req_idx_tensor, eos_tok_id_tensor)
        self.logits_slice: tuple[torch.Tensor, torch.Tensor] = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )
        """logits 切片索引"""

        self.neg_inf_tensor = torch.tensor(
            -float("inf"), dtype=torch.float32, device=self.device
        )
        """负无穷张量，用于屏蔽 token"""

    def is_argmax_invariant(self) -> bool:
        """通过审查停止 token，min-tokens 可以改变贪婪采样中 argmax 操作的结果。

        Returns:
            返回 False（会影响 argmax）
        """
        return False

    @staticmethod
    def add_request(
        params: SamplingParams, _: list[int] | None, output_tok_ids: list[int]
    ) -> tuple[int, Sequence[int], set[int]] | None:
        """添加请求到处理器。

        Args:
            params: 采样参数
            _: prompt token IDs（未使用）
            output_tok_ids: 输出 token IDs

        Returns:
            如果请求需要约束则返回 (min_tokens, output_token_ids, stop_token_ids)，
            否则返回 None
        """
        min_tokens = params.min_tokens
        if not min_tokens or len(output_tok_ids) >= min_tokens:
            return None
        return min_tokens, output_tok_ids, params.all_stop_token_ids

    def update_state(self, batch_update: BatchUpdate | None):
        """更新状态以反映批次变化。

        Args:
            batch_update: 批次更新信息
        """
        needs_update = process_dict_updates(
            self.min_toks, batch_update, self.add_request
        )
        if self.min_toks:
            # 检查是否有请求已达到最小 token 数量。
            to_remove = tuple(
                index
                for index, (min_toks, out_tok_ids, _) in self.min_toks.items()
                if len(out_tok_ids) >= min_toks
            )
            if to_remove:
                needs_update = True
                for index in to_remove:
                    del self.min_toks[index]

        # 根据需要更新张量。
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            for req, (_, _, stop_tok_ids) in self.min_toks.items():
                reqs.extend([req] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)

            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        """创建设备张量。

        Args:
            data: 数据列表
            dtype: 数据类型

        Returns:
            设备上的张量
        """
        return torch.tensor(
            data, device="cpu", dtype=dtype, pin_memory=self.pin_memory
        ).to(device=self.device, non_blocking=True)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """应用最小 token 约束。

        对于未达到最小长度的请求，屏蔽其停止 token。

        Args:
            logits: 输入的 logits 张量

        Returns:
            处理后的 logits 张量
        """
        if self.min_toks:
            # 对于未达到最小长度的请求，屏蔽 EOS token
            logits.index_put_(self.logits_slice, self.neg_inf_tensor)
        return logits

    def apply_with_spec_decode(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int],
    ) -> torch.Tensor:
        """Spec-decode 版本的 apply()。

        优先级：``min_tokens`` > ``stop_token_ids`` / EOS。

        示例：``num_draft_tokens = [2, 3, 1]``
          → ``logits`` 形状 ``[6, V]``, ``cumsum = [0, 2, 5, 6]``
          → 请求 0 拥有行 0-1, 请求 1 拥有行 2-4, 请求 2 拥有行 5

        Args:
            logits: 输入的 logits 张量
            num_draft_tokens: 每个请求的 draft token 数量列表

        Returns:
            处理后的 logits 张量
        """
        if not self.min_toks:
            return logits

        num_draft_arr = np.array(num_draft_tokens, dtype=np.int64)
        cumsum = np.concatenate([[0], np.cumsum(num_draft_arr)])

        entries = [
            (req_idx, min_tok, len(out_tok_ids), list(stop_tok_ids))
            for req_idx, (min_tok, out_tok_ids, stop_tok_ids) in self.min_toks.items()
            if stop_tok_ids
        ]

        if not entries:
            return logits

        all_rows: list[np.ndarray] = []  # 要屏蔽的行索引
        all_toks: list[np.ndarray] = []  # 在这些行的停止 token ids

        for req_idx, min_tok, current_len, stop_toks in entries:
            remaining = min_tok - current_len
            # 有多少前导 draft 位置仍需要停止 token 屏蔽
            n_mask = int(min(max(remaining, 0), num_draft_arr[req_idx]))

            if n_mask > 0:
                offset = cumsum[req_idx]
                row_indices = np.arange(offset, offset + n_mask, dtype=np.int64)
                n_stop = len(stop_toks)
                all_rows.append(np.repeat(row_indices, n_stop))
                all_toks.append(np.tile(stop_toks, n_mask))

        if all_rows:
            rows_arr = np.concatenate(all_rows)
            toks_arr = np.concatenate(all_toks)
            # (row_indices, token_indices) 用于 index_put_ 设置 -inf
            logits_slice = (
                torch.from_numpy(rows_arr).to(self.device, non_blocking=True),
                torch.from_numpy(toks_arr).to(self.device, non_blocking=True),
            )
            logits.index_put_(logits_slice, self.neg_inf_tensor)

        return logits


def process_dict_updates(
    req_entries: dict[int, T],
    batch_update: BatchUpdate | None,
    new_state: Callable[[SamplingParams, list[int] | None, list[int]], T | None],
) -> bool:
    """用于更新稀疏 LogitsProcessors 字典状态的实用函数。

    Args:
        req_entries: 请求条目的字典
        batch_update: 批次更新信息
        new_state: 从采样参数创建新状态的函数

    Returns:
        如果有更新则返回 True
    """

    if not batch_update:
        # 无需处理
        return False

    updated = False
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        if (state := new_state(params, prompt_tok_ids, output_tok_ids)) is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True

    if req_entries:
        # 处理移除的请求。
        for index in batch_update.removed:
            if req_entries.pop(index, None):
                updated = True

        # 处理移动的请求，单向 (a->b) 和交换 (a<->b)
        for a_index, b_index, direct in batch_update.moved:
            a_entry = req_entries.pop(a_index, None)
            b_entry = req_entries.pop(b_index, None)
            if a_entry is not None:
                req_entries[b_index] = a_entry
                updated = True
            if b_entry is not None:
                updated = True
                if direct == MoveDirectionality.SWAP:
                    req_entries[a_index] = b_entry

    return updated
