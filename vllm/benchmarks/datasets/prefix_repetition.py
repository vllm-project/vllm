# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import random

import numpy as np
from transformers import PreTrainedTokenizerBase

from vllm.benchmarks.datasets.abstractions import BenchmarkDataset, SampleRequest
from vllm.benchmarks.datasets.utils import gen_prompt_decode_to_target_len

logger = logging.getLogger(__name__)


class PrefixRepetitionRandomDataset(BenchmarkDataset):
    # Default values copied from benchmark_serving.py for the repeated prefix
    # dataset.
    DEFAULT_PREFIX_LEN = 256
    DEFAULT_SUFFIX_LEN = 256
    DEFAULT_NUM_PREFIXES = 10
    DEFAULT_OUTPUT_LEN = 128

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        suffix_len: int = DEFAULT_SUFFIX_LEN,
        num_prefixes: int = DEFAULT_NUM_PREFIXES,
        output_len: int = DEFAULT_OUTPUT_LEN,
        **kwargs,
    ) -> list[SampleRequest]:
        vocab_size = tokenizer.vocab_size
        prompts_per_prefix = num_requests // num_prefixes
        if prompts_per_prefix == 0:
            raise ValueError(
                f"num_requests ({num_requests}) must be greater than or equal "
                f"to num_prefixes ({num_prefixes})"
            )

        def _generate_exact_length_tokens(target_length: int) -> tuple[list[int], int]:
            """Generate tokens that decode and re-encode to exactly
            target_length."""
            # Generate random tokens
            tokens = np.random.randint(0, vocab_size, size=target_length).tolist()

            _, adjusted_tokens, token_mismatch = gen_prompt_decode_to_target_len(  # noqa: E501
                tokenizer=tokenizer,
                token_sequence=tokens,
                target_token_len=target_length,
                add_special_tokens=False,
            )
            return adjusted_tokens, token_mismatch

        requests = []
        token_mismatch_total = 0
        for _ in range(num_prefixes):
            prefix_tokens, prefix_mismatch = _generate_exact_length_tokens(prefix_len)
            token_mismatch_total += prefix_mismatch

            for _ in range(prompts_per_prefix):
                suffix_tokens, suffix_mismatch = _generate_exact_length_tokens(
                    suffix_len
                )
                token_mismatch_total += suffix_mismatch
                combined_tokens = prefix_tokens + suffix_tokens
                prompt = tokenizer.decode(combined_tokens)
                prompt_len = len(combined_tokens)
                requests.append(
                    SampleRequest(
                        prompt=prompt,
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                    )
                )

        if token_mismatch_total != 0:
            sign = "more" if token_mismatch_total > 0 else "fewer"
            logger.warning(
                "Across all generated prompts, there were %d %s tokens "
                "than expected after decoding and re-encoding. This is "
                "expected due to the imperfect nature of the sampling "
                "procedure.",
                abs(token_mismatch_total),
                sign,
            )
        if not getattr(self, "disable_shuffle", False):
            random.shuffle(requests)
        return requests
