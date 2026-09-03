# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared utilities for benchmark dataset sampling.
"""

import logging
import math

import numpy as np

from vllm.tokenizers import TokenizerLike

logger = logging.getLogger(__name__)

# Type alias: a single float applies to both ISL and OSL; a dict allows
# specifying them independently via ``{"input": …, "output": …}``.
RangeRatio = float | dict[str, float]


def _resolve_range_ratios(
    range_ratio: RangeRatio,
) -> tuple[float, float]:
    """Return ``(input_range_ratio, output_range_ratio)`` from *range_ratio*.

    *range_ratio* is either a single float (used for both input and output)
    or a dict with ``"input"`` and ``"output"`` keys.
    """
    if isinstance(range_ratio, dict):
        try:
            return float(range_ratio["input"]), float(range_ratio["output"])
        except KeyError as exc:
            raise ValueError(
                "When range_ratio is a dict it must contain 'input' and "
                f"'output' keys, got: {sorted(range_ratio)}"
            ) from exc
    ratio = float(range_ratio)
    return ratio, ratio


def get_sampling_params(
    rng: np.random.Generator,
    num_requests: int,
    range_ratio: RangeRatio,
    input_len: int,
    output_len: int,
    tokenizer: TokenizerLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample per-request input/output token lengths and vocab offsets.

    Lengths are drawn uniformly from integer ranges around the configured
    means, controlled by *range_ratio*.  It may be a single ``float``
    (applied to both input and output) or a ``dict`` with ``"input"`` and
    ``"output"`` keys for independent control.

    Tokenizer special tokens are subtracted from ``input_len`` before
    computing the sampling interval.

    Returns:
        (input_lens, output_lens, offsets) – three 1-D ``np.ndarray`` of
        shape ``(num_requests,)``.
    """
    input_range_ratio, output_range_ratio = _resolve_range_ratios(range_ratio)

    if not (0.0 <= input_range_ratio < 1.0):
        raise ValueError("input_range_ratio must be in [0, 1).")
    if not (0.0 <= output_range_ratio < 1.0):
        raise ValueError("output_range_ratio must be in [0, 1).")
    num_special_tokens = int(tokenizer.num_special_tokens_to_add())
    real_input_len = max(0, int(input_len) - num_special_tokens)
    input_low = math.floor(real_input_len * (1 - input_range_ratio))
    input_high = math.ceil(real_input_len * (1 + input_range_ratio))
    output_low = math.floor(output_len * (1 - output_range_ratio))
    output_high = math.ceil(output_len * (1 + output_range_ratio))
    # Ensure the lower bound for output length is at least 1 to
    # prevent sampling 0 tokens.
    output_low = max(output_low, 1)
    output_high = max(output_high, 1)

    if input_low > input_high:
        raise ValueError(
            f"Invalid input sampling interval: low={input_low} > high={input_high}"
        )
    if output_low > output_high:
        raise ValueError(
            f"Invalid output sampling interval: low={output_low} > high={output_high}"
        )

    logger.info(
        "Sampling input_len from [%s, %s] and output_len from [%s, %s]",
        input_low,
        input_high,
        output_low,
        output_high,
    )

    input_lens = rng.integers(input_low, input_high + 1, size=num_requests)
    output_lens = rng.integers(output_low, output_high + 1, size=num_requests)
    offsets = rng.integers(0, tokenizer.vocab_size, size=num_requests)
    return input_lens, output_lens, offsets
