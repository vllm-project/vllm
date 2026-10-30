# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from pathlib import Path
from typing import Any, Literal, Self

import regex as re
from pydantic import ConfigDict, Field, model_validator
from pydantic_core import ArgsKwargs

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

WatermarkingAlgorithm = Literal["gumbel", "dual_key_gumbel"]
WatermarkPRFName = Literal["philox"]
WatermarkContextScope = Literal["none", "single_turn", "all"]

_MIN_RECOMMENDED_DEDUP_HISTORY = 1024


def derive_watermark_key(key: int, domain: bytes) -> int:
    digest = hashlib.sha256(domain + key.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:8], "big")


_DECIMAL_KEY = re.compile(r"[0-9]{1,20}")


def _read_key_file(path: str) -> int:
    try:
        text = Path(path).read_text().strip()
    except OSError as error:
        raise ValueError(
            f"Cannot read watermark key file {path!r}: {error.strerror}"
        ) from None
    if not _DECIMAL_KEY.fullmatch(text) or int(text) > 2**64 - 1:
        raise ValueError(
            f"Watermark key file {path!r} must contain a decimal integer "
            "between 0 and 2**64 - 1."
        )
    return int(text)


@config(config=ConfigDict(hide_input_in_errors=True))
class WatermarkConfig:
    """Configuration for text watermark generation."""

    key: int = Field(ge=0, repr=False, exclude=True)
    """Secret key used to watermark generated text. Set either `key` or
    `key_file`."""
    key_file: str | None = None
    """Path to a file holding the secret key as a decimal integer, such as a
    Docker or Kubernetes secret. It is read once at startup."""
    algorithm: WatermarkingAlgorithm = "gumbel"
    """Algorithm used to watermark generated text."""
    alpha: float = Field(default=0.1, ge=0, le=1)
    """Probability of selecting key B for dual-key watermarking."""
    context_width: int = Field(default=4, ge=1)
    """Number of prior tokens used by the watermark PRF."""
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

    @property
    def supports_speculative_decoding(self) -> bool:
        return self.algorithm == "dual_key_gumbel"

    @model_validator(mode="before")
    @classmethod
    def read_key_file(cls, data: Any) -> Any:
        is_args_kwargs = isinstance(data, ArgsKwargs)
        values = data.kwargs if is_args_kwargs else data
        if not isinstance(values, dict) or values.get("key_file") is None:
            return data
        if values.get("key") is not None or (is_args_kwargs and data.args):
            raise ValueError("Set either key or key_file, not both.")
        values = {**values, "key": _read_key_file(values["key_file"])}
        return ArgsKwargs(data.args, values) if is_args_kwargs else values

    @model_validator(mode="after")
    def validate_watermark_settings(self) -> Self:
        if self.key > 2**64 - 1:
            raise ValueError("philox keys must fit in 64 bits")
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
