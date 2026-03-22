# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""推测解码指标模块。

本模块实现了推测解码的性能指标统计和日志记录，负责：
- 统计草稿 token 和接受 token 数量
- 计算接受率和吞吐量
- 记录日志和 Prometheus 指标

主要类：
- SpecDecodingStats: 单步迭代的解码统计
- SpecDecodingLogging: 日志记录器（聚合多步指标）
- SpecDecodingProm: Prometheus 指标记录器

指标说明：
- 接受率 = 接受 token 数 / 草稿 token 数
- 平均接受长度 = 1 + (接受 token 数 / 草稿数)，包含 bonus token
- 吞吐量 = token 数 / 时间
"""

import time
from dataclasses import dataclass, field

import numpy as np
import prometheus_client

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.v1.metrics.utils import create_metric_per_engine

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    """单步迭代的推测解码统计。

    每个调度步骤聚合所有请求的统计信息，并通过
    EngineCoreOutputs->SchedulerStats 返回给前端。

    Attributes:
        num_spec_tokens: 每个请求的最大草稿 token 数量
        num_drafts: 草稿请求总数（累计）
        num_draft_tokens: 草稿 token 总数（累计）
        num_accepted_tokens: 接受的 token 总数（累计）
        num_accepted_tokens_per_pos: 每个位置接受的 token 数列表
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        """创建新的统计实例。

        Args:
            num_spec_tokens: 每个请求的最大草稿 token 数量

        Returns:
            初始化的 SpecDecodingStats 实例
        """
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
        )

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        """观察一次草稿生成结果。

        更新累计统计信息，包括草稿数、token 数和每个位置的接受数。

        Args:
            num_draft_tokens: 本次草稿的 token 数量
            num_accepted_tokens: 本次接受的 token 数量
        """
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1


class SpecDecodingLogging:
    """聚合并记录推测解码指标到日志。

    使用 observe() 聚合指定时间间隔内的每步指标，
    然后使用 log() 记录并重置为零。

    Attributes:
        num_drafts: 每步的草稿数列表
        num_draft_tokens: 每步的草稿 token 数列表
        num_accepted_tokens: 每步的接受 token 数列表
        accepted_tokens_per_pos_lists: 每步的每位置接受数列表
        last_log_time: 上次日志记录时间
    """

    def __init__(self):
        """初始化日志记录器。"""
        self.reset()

    def reset(self):
        """重置所有统计为零。"""
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        self.last_log_time = time.monotonic()

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        """观察并聚合一步的统计信息。

        Args:
            spec_decoding_stats: 单步统计信息
        """
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(spec_decoding_stats.num_accepted_tokens)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos
        )

    def log(self, log_fn=logger.info):
        """记录聚合的指标到日志。

        Args:
            log_fn: 日志函数，默认为 logger.info
        """
        if not self.num_drafts:
            return
        num_drafts = np.sum(self.num_drafts)
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)
        draft_throughput = 0
        accepted_throughput = 0

        elapsed_time = time.monotonic() - self.last_log_time
        if elapsed_time > 0:
            draft_throughput = num_draft_tokens / elapsed_time
            accepted_throughput = num_accepted_tokens / elapsed_time

        draft_acceptance_rate = (
            num_accepted_tokens / num_draft_tokens * 100
            if num_draft_tokens > 0
            else float("nan")
        )

        # 按照惯例，平均接受长度包含 bonus token
        mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)

        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)

        log_fn(
            "SpecDecoding metrics: "
            "Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, "
            "Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Per-position acceptance rate: %s, "
            "Avg Draft acceptance rate: %.1f%%",
            mean_acceptance_length,
            accepted_throughput,
            draft_throughput,
            num_accepted_tokens,
            num_draft_tokens,
            rates_str,
            draft_acceptance_rate,
        )
        self.reset()


class SpecDecodingProm:
    """记录推测解码指标到 Prometheus。

    接受率可以使用 PromQL 查询计算：

      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_draft_tokens_total[$interval])

    平均接受长度（惯例包含 bonus token）可以使用：

      1 + (
      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_drafts[$interval]))

    每位置接受率向量可以使用：

      vllm:spec_decode_num_accepted_tokens_per_pos[$interval] /
      vllm:spec_decode_num_drafts[$interval]

    Attributes:
        spec_decoding_enabled: 是否启用了推测解码
        counter_spec_decode_num_drafts: 草稿数计数器
        counter_spec_decode_num_draft_tokens: 草稿 token 数计数器
        counter_spec_decode_num_accepted_tokens: 接受 token 数计数器
        counter_spec_decode_num_accepted_tokens_per_pos: 每位置接受数计数器
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        """初始化 Prometheus 指标。

        Args:
            speculative_config: 推测解码配置
            labelnames: Prometheus 标签名列表
            per_engine_labelvalues: 每个引擎的标签值映射
        """
        self.spec_decoding_enabled = speculative_config is not None
        if not self.spec_decoding_enabled:
            return

        counter_drafts = self._counter_cls(
            name="vllm:spec_decode_num_drafts",
            documentation="Number of spec decoding drafts.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_drafts = create_metric_per_engine(
            counter_drafts, per_engine_labelvalues
        )

        counter_draft_tokens = self._counter_cls(
            name="vllm:spec_decode_num_draft_tokens",
            documentation="Number of draft tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_draft_tokens = create_metric_per_engine(
            counter_draft_tokens, per_engine_labelvalues
        )

        counter_accepted_tokens = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens",
            documentation="Number of accepted tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens = create_metric_per_engine(
            counter_accepted_tokens, per_engine_labelvalues
        )

        assert speculative_config is not None
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if self.spec_decoding_enabled
            else 0
        )
        pos_labelnames = labelnames + ["position"]
        base_counter = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens_per_pos",
            documentation="Accepted tokens per draft position.",
            labelnames=pos_labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {
            idx: [base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)]
            for idx, lv in per_engine_labelvalues.items()
        }

    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):
        """观察并更新 Prometheus 指标。

        Args:
            spec_decoding_stats: 单步统计信息
            engine_idx: 引擎索引（默认为 0）
        """
        if not self.spec_decoding_enabled:
            return
        self.counter_spec_decode_num_drafts[engine_idx].inc(
            spec_decoding_stats.num_drafts
        )
        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(
            spec_decoding_stats.num_draft_tokens
        )
        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(
            spec_decoding_stats.num_accepted_tokens
        )
        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos[engine_idx]
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])
