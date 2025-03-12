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


class SpecDecodingMetrics:

    def __init__(self, speculative_config: SpeculativeConfig):
        self.num_spec_tokens = speculative_config.num_speculative_tokens
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
        # FIXME: relies on num_draft_tokens % k == 0 assumption
        #max_num_emitted_tokens = get_max_num_emitted_tokens(
        #    draft_tokens=num_draft_tokens, k=self.num_spec_tokens)
        draft_acceptance_rate = (num_accepted_tokens / num_draft_tokens
                                 if num_draft_tokens > 0 else float("nan"))
        #system_efficiency = (num_emitted_tokens / max_num_emitted_tokens
        #                     if max_num_emitted_tokens > 0 else float("nan"))
        system_efficiency = float("nan")
        logger.info(
            "Speculative metrics: "
            "Draft acceptance rate: %.3f, "
            "System efficiency: %.3f, "
            "Number of speculative tokens: %d, "
            "Number of accepted tokens: %d, "
            "Number of draft tokens: %d, "
            "Number of emitted tokens: %d.", draft_acceptance_rate,
            system_efficiency, self.num_spec_tokens, num_accepted_tokens,
            num_draft_tokens, num_emitted_tokens)
        self.reset()


def get_max_num_emitted_tokens(draft_tokens: int, k: int) -> int:
    """Calculate the number of emitted tokens, assuming all tokens accepted.

    This is equal to the number of sequences that have been speculated on,
    times (speculation len + 1). The +1 comes from the bonus token.
    """
    # Determine the number of sequences that have been speculated on. Since
    # the batch size can be variable, we divide by k.
    print(f"DRAFT TOKENS {draft_tokens} K {k}")
    # Cannot assume this - ngram proposer says "If there are less than k
    # tokens follow the match, we will return the maximum amount of tokens
    # until the end."
    assert draft_tokens % k == 0
    total_num_spec_seqs = draft_tokens // k

    # A single sequence may emit k accepted tokens and one bonus token in
    # the best case.
    num_emitted_per_seq_if_all_accepted = k + 1

    # The max num of emitted tokens is the number of speculated sequences
    # times the max emitted per seq.
    return total_num_spec_seqs * num_emitted_per_seq_if_all_accepted
