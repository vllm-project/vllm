# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import prometheus_client

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in EngineCoreOutputs->SchedulerStats.
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        return cls(num_spec_tokens=num_spec_tokens,
                   num_accepted_tokens_per_pos=[0] * num_spec_tokens)

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1


class SpecDecodingLogging:
    """Aggregate and log spec decoding metrics.

    LoggingStatLogger aggregates per-iteration metrics over a set
    time interval using observe() and then logs them using log()
    before resetting to zero.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(
            spec_decoding_stats.num_accepted_tokens)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos)

    def log(self, log_fn=logger.info):
        if not self.num_drafts:
            return
        num_drafts = np.sum(self.num_drafts)
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)

        draft_acceptance_rate = (num_accepted_tokens / num_draft_tokens *
                                 100 if num_draft_tokens > 0 else float("nan"))

        # Conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)

        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)

        log_fn(
            "SpecDecoding metrics: "
            "Draft acceptance rate: %.1f%%, "
            "Mean acceptance length: %.2f, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Per-position acceptance rate: %s",
            draft_acceptance_rate,
            mean_acceptance_length,
            num_accepted_tokens,
            num_draft_tokens,
            rates_str,
        )
        self.reset()


class SpecDecodingProm:
    """Record spec decoding metrics in Prometheus.

    The acceptance rate can be calculated using a PromQL query:

      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_draft_tokens_total[$interval])

    The mean acceptance length (conventionally including bonus tokens)
    can be calculated using:

      1 + (
      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_drafts[$interval]))

    A per-position acceptance rate vector can be computed using

      vllm:spec_decode_num_accepted_tokens_per_pos[$interval] /
      vllm:spec_decode_num_drafts[$interval]
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        speculative_config: Optional[SpeculativeConfig],
        labelnames: list[str],
        labelvalues: list[str],
    ):
        self.spec_decoding_enabled = speculative_config is not None
        if not self.spec_decoding_enabled:
            return

        self.counter_spec_decode_num_drafts = \
            self._counter_cls(
                name="vllm:spec_decode_num_drafts",
                documentation="Number of spec decoding drafts.",
                labelnames=labelnames).labels(*labelvalues)
        self.counter_spec_decode_num_draft_tokens = \
            self._counter_cls(
                name="vllm:spec_decode_num_draft_tokens",
                documentation="Number of draft tokens.",
                labelnames=labelnames,).labels(*labelvalues)
        self.counter_spec_decode_num_accepted_tokens = \
            self._counter_cls(
                name="vllm:spec_decode_num_accepted_tokens",
                documentation="Number of accepted tokens.",
                labelnames=labelnames).labels(*labelvalues)

        assert speculative_config is not None
        num_spec_tokens = (speculative_config.num_speculative_tokens
                           if self.spec_decoding_enabled else 0)
        pos_labelnames = labelnames + ["position"]
        base_counter = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens_per_pos",
            documentation="Accepted tokens per draft position.",
            labelnames=pos_labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens_per_pos: list[
            prometheus_client.Counter] = []
        for pos in range(num_spec_tokens):
            pos_labelvalues = labelvalues + [str(pos)]
            self.counter_spec_decode_num_accepted_tokens_per_pos.append(
                base_counter.labels(*pos_labelvalues))

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        if not self.spec_decoding_enabled:
            return
        self.counter_spec_decode_num_drafts.inc(spec_decoding_stats.num_drafts)
        self.counter_spec_decode_num_draft_tokens.inc(
            spec_decoding_stats.num_draft_tokens)
        self.counter_spec_decode_num_accepted_tokens.inc(
            spec_decoding_stats.num_accepted_tokens)
        for pos, counter in enumerate(
                self.counter_spec_decode_num_accepted_tokens_per_pos):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])
