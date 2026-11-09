# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig

logger = init_logger(__name__)


class SuffixSpeculatorConfig(SpeculatorConfig):
    base_method = "suffix"
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.suffix.speculator.SuffixSpeculator"
    )
    prefer_dense_token_history = True
    # Confidences come from acceptance rates learned online, not draft logits.
    adaptive_verification_uses_draft_logits = False

    @classmethod
    def configure(cls, spec: "SpeculativeConfig") -> None:
        if spec.num_speculative_tokens is None:
            # Every slot is verified (unless adaptive verification trims it),
            # so a deep default costs throughput at high concurrency.
            spec.num_speculative_tokens = min(8, spec.suffix_decoding_max_tree_depth)
            logger.warning(
                "Defaulted num_speculative_tokens to %s for suffix decoding.",
                spec.num_speculative_tokens,
            )
        # Validate values
        if spec.suffix_decoding_max_tree_depth < 1:
            raise ValueError(
                f"suffix_decoding_max_tree_depth="
                f"{spec.suffix_decoding_max_tree_depth} must be >= 1"
            )
        if not 0 <= spec.suffix_decoding_corpus_tokens < (1 << 31):
            raise ValueError(
                f"suffix_decoding_corpus_tokens="
                f"{spec.suffix_decoding_corpus_tokens} must be in [0, 2**31)"
            )
        if spec.suffix_decoding_max_cached_requests < 0:
            raise ValueError(
                f"suffix_decoding_max_cached_requests="
                f"{spec.suffix_decoding_max_cached_requests} must be >= 0"
            )
        if spec.suffix_decoding_max_spec_factor < 0:
            raise ValueError(
                f"suffix_decoding_max_spec_factor="
                f"{spec.suffix_decoding_max_spec_factor} must be >= 0"
            )
        if not 0 <= spec.suffix_decoding_min_token_prob <= 1:
            raise ValueError(
                f"suffix_decoding_min_token_prob="
                f"{spec.suffix_decoding_min_token_prob} must be in [0, 1]"
            )
