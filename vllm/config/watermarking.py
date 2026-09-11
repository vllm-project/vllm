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
    """Number of prior output tokens used by the watermark PRF."""
    prf: WatermarkPRFName = "philox"
    """Pseudorandom function used by the watermarking algorithm."""

    @model_validator(mode="after")
    def validate_key(self) -> Self:
        if self.key > 2**64 - 1:
            raise ValueError("philox keys must fit in 64 bits")
        if self.algorithm == "gumbel":
            logger.warning_once(
                "Single-key Gumbel-max watermarking may increase the frequency of "
                "degenerate generations, including repetition loops.",
                scope="global",
            )
        return self

    @property
    def supports_speculative_decoding(self) -> bool:
        return _SPECULATIVE_DECODING_SUPPORT[self.algorithm]
