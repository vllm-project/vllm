# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_emitted_tokens: int = 0

    def take(self):
        copied = SpecDecodingStats(self.num_draft_tokens,
                                   self.num_accepted_tokens,
                                   self.num_emitted_tokens)
        self.reset()
        return copied

    def reset(self):
        self.num_draft_tokens = 0
        self.num_accepted_tokens = 0
        self.num_emitted_tokens = 0

    def observe(self, num_draft_tokens: int, num_accepted_tokens: int,
                num_emitted_tokens: int):
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        self.num_emitted_tokens += num_emitted_tokens


class SpecDecodingMetrics:

    def __init__(self, speculative_config: SpeculativeConfig):
        self.num_spec_tokens = (speculative_config.num_speculative_tokens
                                if speculative_config is not None else 0)
        self.reset()

    def reset(self):
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.num_emitted_tokens: list[int] = []

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(
            spec_decoding_stats.num_accepted_tokens)
        self.num_emitted_tokens.append(spec_decoding_stats.num_emitted_tokens)

    def log(self):
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)
        num_emitted_tokens = np.sum(self.num_emitted_tokens)

        draft_acceptance_rate = (num_accepted_tokens / num_draft_tokens
                                 if num_draft_tokens > 0 else float("nan"))

        logger.info(
            "Speculative metrics: "
            "Draft acceptance rate: %.3f, "
            "Number of speculative tokens: %d, "
            "Number of accepted tokens: %d, "
            "Number of draft tokens: %d, "
            "Number of emitted tokens: %d.", draft_acceptance_rate,
            num_accepted_tokens, num_draft_tokens, num_emitted_tokens)
        self.reset()
