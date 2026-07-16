# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default bucket boundaries for the Prometheus histograms emitted by vLLM.

Every histogram created by the Prometheus stat logger draws its bucket
boundaries from exactly one of the families defined here, so each default
list has a single source of truth.
"""

from typing import Literal, get_args

BucketFamilyKey = Literal[
    "request_latency",
    "time_to_first_token",
    "inter_token_latency",
    "iteration_tokens",
    "request_params_n",
    "request_tokens",
    "kv_cache_residency",
]
"""Canonical keys for the histogram bucket families."""

BUCKET_FAMILY_KEYS: frozenset[str] = frozenset(get_args(BucketFamilyKey))
"""All bucket family keys, for membership checks."""

REQUEST_LATENCY_BUCKETS: tuple[float, ...] = (
    0.3,
    0.5,
    0.8,
    1.0,
    1.5,
    2.0,
    2.5,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    120.0,
    240.0,
    480.0,
    960.0,
    1920.0,
    7680.0,
)
"""Sub-second scheduling delays through multi-hour batch requests; shared by
the accumulated request phase-timing histograms (e2e, queue, inference,
prefill, decode)."""

TIME_TO_FIRST_TOKEN_BUCKETS: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
    640.0,
    2560.0,
)
"""Millisecond-scale prefill for tiny prompts up to ~40-minute worst cases,
with the densest resolution around interactive (sub-second) latencies."""

INTER_TOKEN_LATENCY_BUCKETS: tuple[float, ...] = (
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
)
"""Per-decode-step latencies: 10 ms fast decode up to multi-second stalls."""

ITERATION_TOKENS_BUCKETS: tuple[float, ...] = (
    1,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
)
"""Tokens processed per engine step: powers of two up to a typical
max-num-batched-tokens budget."""

REQUEST_PARAMS_N_BUCKETS: tuple[float, ...] = (1, 2, 5, 10, 20)
"""Small integer counts for the ``n`` sampling parameter."""

KV_CACHE_RESIDENCY_BUCKETS: tuple[float, ...] = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    20,
    30,
    60,
    120,
    300,
    600,
    1200,
    1800,
)
"""KV cache block residency times: millisecond-scale reuse gaps up to
30-minute block lifetimes."""

_STATIC_FAMILY_DEFAULTS: dict[str, tuple[float, ...]] = {
    "request_latency": REQUEST_LATENCY_BUCKETS,
    "time_to_first_token": TIME_TO_FIRST_TOKEN_BUCKETS,
    "inter_token_latency": INTER_TOKEN_LATENCY_BUCKETS,
    "iteration_tokens": ITERATION_TOKENS_BUCKETS,
    "request_params_n": REQUEST_PARAMS_N_BUCKETS,
    "kv_cache_residency": KV_CACHE_RESIDENCY_BUCKETS,
}


def build_buckets(mantissa_lst: list[int], max_value: int) -> list[float]:
    """Build buckets from mantissas scaled by increasing powers of 10.

    Args:
        mantissa_lst: Mantissa values multiplied by each power of 10.
        max_value: Largest bucket value to include.

    Returns:
        Bucket values in increasing order, capped at `max_value`.
    """
    exponent = 0
    buckets: list[float] = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


def build_1_2_5_buckets(max_value: int) -> list[float]:
    """Build a 1-2-5 series of buckets capped at `max_value`.

    Example:
        >>> build_1_2_5_buckets(100)
        [1, 2, 5, 10, 20, 50, 100]
    """
    return build_buckets([1, 2, 5], max_value)


def histogram_buckets(
    family: BucketFamilyKey,
    max_model_len: int | None = None,
) -> list[float]:
    """Return the default bucket boundaries for a histogram family.

    Args:
        family: Canonical bucket family key.
        max_model_len: Cap for the token-count series of the
            `request_tokens` family; required for that family and ignored
            otherwise.

    Returns:
        A fresh list of bucket upper bounds; callers may mutate it freely.

    Raises:
        ValueError: If `family` is `request_tokens` and `max_model_len`
            is None.
    """
    if family == "request_tokens":
        if max_model_len is None:
            raise ValueError(
                "max_model_len is required for the 'request_tokens' bucket family"
            )
        return build_1_2_5_buckets(max_model_len)
    return list(_STATIC_FAMILY_DEFAULTS[family])
