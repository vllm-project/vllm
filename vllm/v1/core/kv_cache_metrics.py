# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 缓存指标追踪模块。

本模块实现了 KV 缓存块的生命周期指标追踪功能，负责：
- 记录每个块的分配时间、最后访问时间和访问历史
- 计算块的生存时间、空闲时间和重用间隔
- 采样收集 KV 缓存驱逐事件指标

主要类：
- BlockMetricsState: 单个 KV 缓存块的指标状态
- KVCacheMetricsCollector: KV 缓存指标收集器（支持采样）
"""

import random
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

from vllm.v1.metrics.stats import KVCacheEvictionEvent


class BlockMetricsState:
    """追踪单个 KV 缓存块的生命周期指标。

    记录块的分配时间、最后访问时间以及访问历史，
    用于计算生存时间、空闲时间和重用间隔等指标。

    Attributes:
        birth_time_ns: 块的分配时间（纳秒）
        last_access_ns: 块的最后访问时间（纳秒）
        access_history: 访问历史队列（最多保留 4 条记录）
    """

    def __init__(self):
        """初始化块指标状态。

        记录当前时间为块的出生时间和最后访问时间。
        """
        now_ns = time.monotonic_ns()
        self.birth_time_ns = now_ns
        self.last_access_ns = now_ns
        # 限制队列长度以防止块被多次访问时无限制增长
        self.access_history: deque[int] = deque(maxlen=4)

    def record_access(self) -> None:
        """记录一次块访问。

        更新最后访问时间为当前时间，并添加到访问历史队列。
        """
        now_ns = time.monotonic_ns()
        self.last_access_ns = now_ns
        self.access_history.append(now_ns)

    def get_lifetime_seconds(self) -> float:
        """获取块的生存时间（秒）。

        Returns:
            从块分配至今的时间（秒）
        """
        now_ns = time.monotonic_ns()
        return (now_ns - self.birth_time_ns) / 1e9

    def get_idle_time_seconds(self) -> float:
        """获取块的空闲时间（秒）。

        Returns:
            从块最后访问至今的时间（秒）
        """
        now_ns = time.monotonic_ns()
        return (now_ns - self.last_access_ns) / 1e9

    def get_reuse_gaps_seconds(self) -> list[float]:
        """获取块的重用间隔列表（秒）。

        计算相邻两次访问之间的时间间隔。

        Returns:
            重用间隔列表（秒），如果访问历史少于 2 次则返回空列表
        """
        if len(self.access_history) < 2:
            return []
        history = list(self.access_history)
        return [(history[i] - history[i - 1]) / 1e9 for i in range(1, len(history))]


class KVCacheMetricsCollector:
    """KV 缓存指标收集器（支持采样）。

    负责收集 KV 缓存块的生命周期指标，通过采样方式跟踪块的分配、
    访问和驱逐事件，用于性能分析和优化。

    Attributes:
        sample_rate: 采样率（0 到 1 之间）
        block_metrics: 块指标字典（block_id -> BlockMetricsState）
        _eviction_events: 驱逐事件列表
    """

    def __init__(self, sample_rate: float = 0.01):
        """初始化 KV 缓存指标收集器。

        Args:
            sample_rate: 采样率，必须在 (0, 1.0] 范围内，默认 0.01
        """
        assert 0 < sample_rate <= 1.0, (
            f"sample_rate must be in (0, 1.0], got {sample_rate}"
        )
        self.sample_rate = sample_rate

        self.block_metrics: dict[int, BlockMetricsState] = {}

        self._eviction_events: list[KVCacheEvictionEvent] = []

    def should_sample_block(self) -> bool:
        """判断是否应该采样当前块。

        Returns:
            True 表示应该采样，False 表示跳过
        """
        return random.random() < self.sample_rate

    def on_block_allocated(self, block: "KVCacheBlock") -> None:
        """块分配时的回调。

        如果被采样，则为该块创建指标状态。

        Args:
            block: 被分配的 KV 缓存块
        """
        if self.should_sample_block():
            self.block_metrics[block.block_id] = BlockMetricsState()

    def on_block_accessed(self, block: "KVCacheBlock") -> None:
        """块访问时的回调。

        如果该块正在被跟踪，则更新其访问记录。

        Args:
            block: 被访问的 KV 缓存块
        """
        metrics = self.block_metrics.get(block.block_id)
        if metrics:
            metrics.record_access()

    def on_block_evicted(self, block: "KVCacheBlock") -> None:
        """块驱逐时的回调。

        记录块的驱逐事件指标，包括生存时间、空闲时间和重用间隔。

        Args:
            block: 被驱逐的 KV 缓存块
        """
        metrics = self.block_metrics.pop(block.block_id, None)
        if not metrics:
            return

        lifetime = metrics.get_lifetime_seconds()
        idle_time = metrics.get_idle_time_seconds()
        reuse_gaps = tuple(metrics.get_reuse_gaps_seconds())

        self._eviction_events.append(
            KVCacheEvictionEvent(
                lifetime_seconds=lifetime,
                idle_seconds=idle_time,
                reuse_gaps_seconds=reuse_gaps,
            )
        )

    def reset(self) -> None:
        """重置所有状态。

        在缓存重置时清除所有块指标和驱逐事件记录。
        """
        self.block_metrics.clear()
        self._eviction_events.clear()

    def drain_events(self) -> list[KVCacheEvictionEvent]:
        """排出并清除所有驱逐事件。

        Returns:
            驱逐事件列表，调用后内部队列为空
        """
        events = self._eviction_events
        self._eviction_events = []
        return events
