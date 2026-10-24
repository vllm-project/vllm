# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in suffix-only staging of ``RequestState.all_token_ids`` for new requests.

A new request normally stages its whole token history. With a prefix-cache hit,
input preparation, post-update and speculative drafting read only positions from
``num_computed_tokens`` on, plus the model state's ``token_lookback_depth``
tokens below it, and frequency / presence penalties read output tokens only.
Positions below the staging start keep stale data from the slot's previous
occupant, so requests and engine features that read older history are staged
fully.
"""

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import NewRequestData

logger = init_logger(__name__)


class SuffixTokenStaging:
    def __init__(self, margin: int, min_start: int):
        self.margin = margin
        self.min_start = min_start

    @classmethod
    def create(
        cls,
        vllm_config: VllmConfig,
        has_custom_logits_processors: bool,
        token_lookback_depth: int,
    ) -> "SuffixTokenStaging | None":
        """Return the staging policy, or None if it is off or unsupported.

        Watermarking, custom logits processors and the n-gram drafter read the
        whole history; KV connectors can rewind ``num_computed_tokens`` of a
        running request after a failed load. The margin is raised to the
        model state's lookback.
        """
        margin = envs.VLLM_SUFFIX_TOKEN_STAGING
        if margin is None:
            return None
        if margin < 0:
            raise ValueError(f"VLLM_SUFFIX_TOKEN_STAGING must be >= 0, got {margin}")
        margin = max(margin, token_lookback_depth)
        spec_config = vllm_config.speculative_config
        reason = None
        if vllm_config.watermark_config is not None:
            reason = "watermarking"
        elif has_custom_logits_processors:
            reason = "custom logits processors"
        elif spec_config is not None and spec_config.use_ngram():
            reason = "n-gram speculative decoding"
        elif vllm_config.kv_transfer_config is not None:
            reason = "KV connector"
        if reason is not None:
            logger.warning("Suffix-only token staging is disabled: %s.", reason)
            return None
        min_start = envs.VLLM_SUFFIX_TOKEN_STAGING_MIN
        logger.info(
            "Suffix-only token staging enabled (margin %d, min start %d).",
            margin,
            min_start,
        )
        return cls(margin, min_start)

    def get_start(self, new_req_data: NewRequestData) -> int:
        """Return the first ``all_token_ids`` position to stage (0: all)."""
        sp = new_req_data.sampling_params
        if (
            sp is None
            or new_req_data.pooling_params is not None
            or new_req_data.mm_features
            or sp.prompt_logprobs is not None
            # The prompt bin mask read for repetition penalty covers the prompt.
            or sp.repetition_penalty != 1.0
            or sp.bad_words
            or sp.bad_words_token_ids
            or sp.thinking_token_budget is not None
        ):
            return 0
        start = (
            min(new_req_data.num_computed_tokens, new_req_data.prompt_len) - self.margin
        )
        return start if start >= self.min_start else 0
