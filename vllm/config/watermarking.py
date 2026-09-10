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

_SPECULATIVE_DECODING_SUPPORT: dict[WatermarkingAlgorithm, bool] = {
    "gumbel": False,
}


@config
class WatermarkConfig:
    """Configuration for text watermark generation."""

    key: int = Field(ge=0, repr=False, exclude=True)
    """Secret key used to watermark generated text."""
    algorithm: WatermarkingAlgorithm = "gumbel"
    """Algorithm used to watermark generated text."""
    context_width: int = Field(default=4, ge=1)
    """Number of prior tokens used by the watermark PRF."""
    deduplicate_contexts: Literal["none", "single_turn", "all"] = "single_turn"
    """History scope used to identify repeated watermark contexts."""
    deduplicate_contexts_max_history: int | None = Field(default=8192, ge=1)
    """Maximum prior positions searched, or ``None`` for the full scope."""
    prf: WatermarkPRFName = "philox"
    """Pseudorandom function used by the watermarking algorithm."""

    @model_validator(mode="after")
    def validate_key(self) -> Self:
        if self.key > 2**64 - 1:
            raise ValueError("philox keys must fit in 64 bits")
        history_is_too_short = (
            self.deduplicate_contexts_max_history is not None
            and self.deduplicate_contexts_max_history < 256
        )
        if self.algorithm == "gumbel" and (
            self.deduplicate_contexts == "none" or history_is_too_short
        ):
            logger.warning_once(
                "Single-key Gumbel-max watermarking with context deduplication "
                "disabled or limited to fewer than 256 positions may increase the "
                "frequency of degenerate generations, including repetition loops. "
                "Use deduplicate_contexts='single_turn' or 'all' with "
                "deduplicate_contexts_max_history at least 256 or null to mitigate "
                "this.",
                scope="global",
            )
        return self

    @property
    def supports_speculative_decoding(self) -> bool:
        return _SPECULATIVE_DECODING_SUPPORT[self.algorithm]
