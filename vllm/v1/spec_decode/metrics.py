# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import numpy as np
import prometheus_client

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0

    def take(self):
        copied = SpecDecodingStats(self.num_draft_tokens,
                                   self.num_accepted_tokens)
        self.reset()
        return copied

    def reset(self):
        self.num_draft_tokens = 0
        self.num_accepted_tokens = 0

    def observe(self, num_draft_tokens: int, num_accepted_tokens: int):
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens


class SpecDecodingLogging:

    def __init__(self):
        self.reset()

    def reset(self):
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(
            spec_decoding_stats.num_accepted_tokens)

    def log(self, log_fn=logger.info):
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)

        draft_acceptance_rate = (num_accepted_tokens / num_draft_tokens *
                                 100 if num_draft_tokens > 0 else float("nan"))

        log_fn(
            "SpecDecoding metrics: "
            "Draft acceptance rate: %.1f%%, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens",
            draft_acceptance_rate,
            num_accepted_tokens,
            num_draft_tokens,
        )
        self.reset()


class SpecDecodingProm:

    def __init__(self, speculative_config: Optional[SpeculativeConfig],
                 labelnames: list[str], labelvalues: list[str]):
        self.spec_decoding_enabled = speculative_config is not None
        if not self.spec_decoding_enabled:
            return

        self.counter_spec_decode_num_draft_tokens = \
            prometheus_client.Counter(
                name="vllm:spec_decode_num_draft_tokens_total",
                documentation="Number of draft tokens.",
                labelnames=labelnames).labels(*labelvalues)
        self.counter_spec_decode_num_accepted_tokens = \
            prometheus_client.Counter(
                name="vllm:spec_decode_num_accepted_tokens_total",
                documentation="Number of accepted tokens.",
                labelnames=labelnames).labels(*labelvalues)

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        if not self.spec_decoding_enabled:
            return
        self.counter_spec_decode_num_draft_tokens.inc(
            spec_decoding_stats.num_draft_tokens)
        self.counter_spec_decode_num_accepted_tokens.inc(
            spec_decoding_stats.num_accepted_tokens)
