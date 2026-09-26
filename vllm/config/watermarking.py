# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

WatermarkingAlgorithm = Literal["gumbel", "dual_key_gumbel", "sbw"]
WatermarkPRFName = Literal["philox"]
WatermarkContextScope = Literal["none", "single_turn", "all"]
SBWScheme = Literal["selfhash", "lefthash"]

_MIN_RECOMMENDED_DEDUP_HISTORY = 1024

# Per-scheme default context widths for SBW, matching the reference sbw library:
#   selfhash (anchored minhash PRF) → 4
#   lefthash (additive PRF)         → 1
_SBW_DEFAULT_CONTEXT_WIDTH: dict[SBWScheme, int] = {
    "selfhash": 4,
    "lefthash": 1,
}


def derive_watermark_key(key: int, domain: bytes) -> int:
    digest = hashlib.sha256(domain + key.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:8], "big")


@config
class WatermarkConfig:
    """Configuration for text watermark generation."""

    key: int = Field(ge=0, repr=False, exclude=True)
    """Secret key used to watermark generated text."""
    algorithm: WatermarkingAlgorithm = "gumbel"
    """Algorithm used to watermark generated text."""
    alpha: float = Field(default=0.1, ge=0, le=1)
    """Probability of selecting key B for dual-key watermarking."""
    context_width: int | None = Field(default=None, ge=1)
    """Number of prior output tokens used by the watermark PRF.
    Defaults: gumbel/dual_key_gumbel → 4; sbw selfhash → 4; sbw lefthash → 1."""
    deduplicate_contexts: WatermarkContextScope = "single_turn"
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
    allow_target_only_watermarking: bool = False
    """Allow speculative decoding without watermarking draft tokens."""

    # SBW-specific fields
    sbw_scheme: SBWScheme = "selfhash"
    """SBW seeding scheme: 'selfhash' (anchored minhash PRF, default) or
    'lefthash' (additive PRF, faster)."""
    sbw_gamma: float = Field(default=0.5, gt=0.0, lt=1.0)
    """SBW green-list fraction in (0, 1)."""
    sbw_delta: float = Field(default=2.0, ge=0.0)
    """SBW logit bias added to green tokens. Set to 0 to disable."""

    @property
    def supports_speculative_decoding(self) -> bool:
        return self.algorithm in ("dual_key_gumbel", "sbw")

    @model_validator(mode="after")
    def validate_watermark_settings(self) -> Self:
        if self.key > 2**64 - 1:
            raise ValueError("philox keys must fit in 64 bits")

        # Set default context_width per algorithm/scheme.
        if self.context_width is None:
            if self.algorithm in ("gumbel", "dual_key_gumbel"):
                object.__setattr__(self, "context_width", 4)
            elif self.algorithm == "sbw":
                default_cw = _SBW_DEFAULT_CONTEXT_WIDTH[self.sbw_scheme]
                object.__setattr__(self, "context_width", default_cw)

        # Warn about degenerate generation risk for Gumbel variants.
        history_is_too_short = (
            self.deduplicate_contexts_max_history is not None
            and self.deduplicate_contexts_max_history < _MIN_RECOMMENDED_DEDUP_HISTORY
        )
        if self.algorithm in ("gumbel", "dual_key_gumbel") and (
            self.deduplicate_contexts == "none" or history_is_too_short
        ):
            logger.warning_once(
                "Gumbel-max watermarking with context deduplication "
                "disabled or limited to fewer than "
                f"{_MIN_RECOMMENDED_DEDUP_HISTORY} positions may increase the "
                "frequency of degenerate generations, including repetition loops. "
                "Use deduplicate_contexts='single_turn' or 'all' with "
                "deduplicate_contexts_max_history at least "
                f"{_MIN_RECOMMENDED_DEDUP_HISTORY} or null to mitigate this.",
                scope="global",
            )
        return self
