# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logits 处理器状态管理模块。

本模块实现了 logits 处理器的状态管理功能，负责：
- 构建和管理批次更新信息
- 维护移除请求的排序状态
- 封装已初始化的 logits 处理器对象
- 支持 logprobs 模式的批次变更跟踪

主要类：
- BatchUpdateBuilder: 批次更新构建器
- LogitsProcessors: logits 处理器封装类
"""

from collections.abc import Iterable, Iterator
from itertools import chain
from typing import TYPE_CHECKING

from vllm.v1.sample.logits_processor.interface import (
    AddedRequest,
    BatchUpdate,
    MovedRequest,
    RemovedRequest,
)

if TYPE_CHECKING:
    from vllm.v1.sample.logits_processor.interface import LogitsProcessor


class BatchUpdateBuilder:
    """帮助构建批次更新信息并构建 logits 处理器的批次更新数据结构。

    用于跟踪持久化批次状态变化，并为 logits 处理器提供更新信息。

    假设：
    * 关于从持久化批次中移除的请求的所有信息都通过调用
      self.removed_append() 在步骤开始时聚合到 self._removed 中。
      这必须发生在步骤中首次读取 self.removed、self.pop_removed()
      或 self.peek_removed() 之前。
    * 在步骤中首次调用 self.removed、self.pop_removed() 或
      self.peek_removed() 后，不再使用 self.removed_append() 注册新的移除。
    * self._removed 的元素永远不会被直接修改、添加或移除
      （即修改只能通过 self.removed_append() 和 self.pop_removed() 进行）。

    在上述假设下的保证：
    * self.removed 始终按降序排序
    * self.pop_removed() 和 self.peek_removed() 都返回当前步骤中
      最低的移除请求索引
    """

    _removed: list[RemovedRequest]
    """已移除请求索引列表（内部存储）"""
    _is_removed_sorted: bool
    """已移除请求是否已排序"""
    added: list[AddedRequest]
    """已添加请求列表"""
    moved: list[MovedRequest]
    """已移动请求列表"""

    def __init__(
        self,
        removed: list[RemovedRequest] | None = None,
        added: list[AddedRequest] | None = None,
        moved: list[MovedRequest] | None = None,
    ) -> None:
        """初始化批次更新构建器。

        Args:
            removed: 初始已移除请求列表
            added: 初始已添加请求列表
            moved: 初始已移动请求列表
        """
        self._removed = removed or []
        self.added = added or []
        self.moved = moved or []
        self._is_removed_sorted = False

        # 用于在 pooling 情况下跟踪变化，
        # 此时我们不填充 added 列表。
        self.batch_changed = False

    def _ensure_removed_sorted(self) -> None:
        """将已移除请求索引按降序排序。

        在给定步骤中首次调用后是幂等的，直到重置。
        """
        if not self._is_removed_sorted:
            self._removed.sort(reverse=True)
            self._is_removed_sorted = True

    @property
    def removed(self) -> list[RemovedRequest]:
        """已移除请求索引，按降序排序。

        Returns:
            降序排序的已移除请求索引列表
        """
        self._ensure_removed_sorted()
        return self._removed

    def removed_append(self, index: int) -> None:
        """注册从持久化批次中移除请求。

        在 self.removed、self.pop_removed() 或 self.peek_removed()
        首次被调用后，不得调用此方法。

        Args:
            index: 请求索引

        Raises:
            RuntimeError: 当在 self.removed 被读取后尝试注册新移除时
        """
        if self._is_removed_sorted:
            raise RuntimeError(
                "在 self.removed 被读取后无法注册新的移除请求。"
            )
        self._removed.append(index)
        self.batch_changed = True

    def has_removed(self) -> bool:
        """检查是否有已移除的请求。

        Returns:
            如果有已移除的请求则返回 True
        """
        return bool(self._removed)

    def peek_removed(self) -> int | None:
        """返回最低的移除请求索引（不弹出）。

        Returns:
            最低的移除请求索引，如果没有则返回 None
        """
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed[-1]
        return None

    def pop_removed(self) -> int | None:
        """弹出最低的移除请求索引。

        Returns:
            最低的移除请求索引，如果没有则返回 None
        """
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed.pop()
        return None

    def reset(self) -> bool:
        """重置内部状态。

        Returns:
            如果批次有任何变化则返回 True
        """
        self._is_removed_sorted = False
        self._removed.clear()
        self.added.clear()
        self.moved.clear()
        batch_changed = self.batch_changed
        self.batch_changed = False
        return batch_changed

    def get_and_reset(self, batch_size: int) -> BatchUpdate | None:
        """生成 logits 处理器的批次更新数据结构并重置内部状态。

        Args:
            batch_size: 当前持久化批次大小

        Returns:
            冻结的 logits 处理器批次更新实例；如果没有更新则返回 None
        """
        # 重置移除排序逻辑
        self._is_removed_sorted = False
        self.batch_changed = False
        if not any((self._removed, self.moved, self.added)):
            # 没有更新，快速返回
            return None
        # 构建批次状态更新
        batch_update = BatchUpdate(
            batch_size=batch_size,
            removed=self._removed,
            moved=self.moved,
            added=self.added,
        )
        self._removed = []
        self.moved = []
        self.added = []
        return batch_update


class LogitsProcessors:
    """封装已初始化的 logits 处理器对象。

    将 logits 处理器分为两类：
    - argmax_invariant: 对 argmax 计算没有影响的处理器
    - non_argmax_invariant: 可能影响 argmax 计算的处理器

    这种分类允许在采样时优化处理顺序。
    """

    def __init__(self, logitsprocs: Iterable["LogitsProcessor"] | None = None) -> None:
        """初始化 LogitsProcessors 封装类。

        Args:
            logitsprocs: logits 处理器可迭代对象
        """
        self.argmax_invariant: list[LogitsProcessor] = []
        """对 argmax 计算没有影响的处理器列表"""
        self.non_argmax_invariant: list[LogitsProcessor] = []
        """可能影响 argmax 计算的处理器列表"""
        if logitsprocs:
            for logitproc in logitsprocs:
                (
                    self.argmax_invariant
                    if logitproc.is_argmax_invariant()
                    else self.non_argmax_invariant
                ).append(logitproc)

    @property
    def all(self) -> Iterator["LogitsProcessor"]:
        """所有 logits 处理器的迭代器。

        Returns:
            所有 logits 处理器的链式迭代器
        """
        return chain(self.argmax_invariant, self.non_argmax_invariant)
