# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

WatermarkingAlgorithm = Literal["gumbel"]
WatermarkPRFName = Literal["philox"]
WatermarkDeduplicationScope = Literal["none", "single_turn", "all"]

_SPECULATIVE_DECODING_SUPPORT: dict[WatermarkingAlgorithm, bool] = {
    "gumbel": False,
}
_MIN_RECOMMENDED_DEDUP_HISTORY = 1024


@config
class WatermarkConfig:
    """Configuration for text watermark generation."""

    key: int = Field(ge=0, repr=False, exclude=True)
    """Secret key used to watermark generated text."""
    algorithm: WatermarkingAlgorithm = "gumbel"
    """Algorithm used to watermark generated text."""
    context_width: int = Field(default=4, ge=1)
    """Number of prior tokens used by the watermark PRF."""
    deduplicate_contexts: WatermarkDeduplicationScope = "single_turn"
    """Which history is searched for a repeated context before a token is
    sampled; a repeated context is sampled without the watermark. `none`
    disables the search. `single_turn` (default) searches this request's
    generated tokens. `all` also searches the prompt and samples the first
    `context_width` generated tokens without watermarking."""
    deduplicate_contexts_max_history: int | None = Field(default=8192, ge=1)
    """Number of most recent history positions searched (default 8192), or
    `None` for the whole scope. Each position is compared over the
    `context_width` tokens before it. Ignored when `deduplicate_contexts` is
    `none`."""
    prf: WatermarkPRFName = "philox"
    """Pseudorandom function used by the watermarking algorithm."""

    @model_validator(mode="after")
    def validate_watermark_settings(self) -> Self:
        if self.key > 2**64 - 1:
            raise ValueError("philox keys must fit in 64 bits")
        history_is_too_short = (
            self.deduplicate_contexts_max_history is not None
            and self.deduplicate_contexts_max_history < _MIN_RECOMMENDED_DEDUP_HISTORY
        )
        if self.algorithm == "gumbel" and (
            self.deduplicate_contexts == "none" or history_is_too_short
        ):
            logger.warning_once(
                "Single-key Gumbel-max watermarking with context deduplication "
                "disabled or limited to fewer than "
                f"{_MIN_RECOMMENDED_DEDUP_HISTORY} positions may increase the "
                "frequency of degenerate generations, including repetition loops. "
                "Use deduplicate_contexts='single_turn' or 'all' with "
                "deduplicate_contexts_max_history at least "
                f"{_MIN_RECOMMENDED_DEDUP_HISTORY} or null to mitigate this.",
                scope="global",
            )
        return self

    @property
    def supports_speculative_decoding(self) -> bool:
        return _SPECULATIVE_DECODING_SUPPORT[self.algorithm]
