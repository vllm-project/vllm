# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

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


class SpecDecodingMetrics:

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
