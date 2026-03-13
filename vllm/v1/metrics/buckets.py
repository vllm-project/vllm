# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Histogram bucket definitions for Prometheus metrics.

This module centralizes all histogram bucket configurations used across
vLLM's metrics system. Each bucket set is designed for a specific class
of measurements.
"""

from enum import Enum


class BucketType(str, Enum):
    """Named bucket types for histogram metrics."""

    TOKEN_STEP_LATENCY = "token_step_latency"
    """For inter-token latency and time_per_output_token metrics (seconds).
    Optimized for per-token timing in 10ms to 80s range.
    Used by: inter_token_latency_seconds, request_time_per_output_token_seconds
    """

    PREFILL_LATENCY = "prefill_latency"
    """For time-to-first-token metrics (seconds).
    Covers 1ms to 2560s with fine granularity at sub-second levels
    for interactive applications.
    Used by: time_to_first_token_seconds
    """

    ACCUMULATED_PHASE_LATENCY = "accumulated_phase_latency"
    """For end-to-end and phase-specific request latencies (seconds).
    Range: 100ms to 7680s (2+ hours) for long-running requests.
    Used by: e2e_request_latency, queue_time, inference_time,
             prefill_time, decode_time
    """

    CACHE_RESIDENCY = "cache_residency"
    """For KV cache block lifecycle metrics (seconds).
    Range: 1ms to 3600s (1 hour) for cache timing.
    Used by: kv_block_lifetime, kv_block_idle_before_evict, kv_block_reuse_gap
    """

    REQUEST_TOKEN_COUNT = "request_token_count"
    """For token count histograms (dynamic 1-2-5 buckets).
    Scales dynamically based on max_model_len.
    Used by: request_prompt_tokens, request_generation_tokens,
             request_max_num_generation_tokens, request_params_max_tokens,
             request_prefill_kv_computed_tokens
    """

    BATCH_SIZE = "batch_size"
    """For iteration token counts (tokens per scheduler step).
    Dynamic 1-2-5 buckets scaling to max_num_batched_tokens.
    Used by: iteration_tokens_total
    """

    COMPLETION_COUNT = "completion_count"
    """For request completion count (n parameter).
    Small integers: 1, 2, 5, 10, 20.
    Used by: request_params_n
    """


# Default bucket definitions
# Each bucket array is designed for specific measurement characteristics

DEFAULT_BUCKETS: dict[BucketType, tuple[float, ...]] = {
    # TOKEN_STEP_LATENCY: Per-token timing (10ms to 80s)
    # Fine granularity in 10-200ms range where most tokens complete,
    # then coarser buckets for outliers
    BucketType.TOKEN_STEP_LATENCY: (
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
    ),
    # PREFILL_LATENCY: Time to first token (1ms to 2560s)
    # Very fine granularity at sub-100ms for interactive apps,
    # extending to ~42 minutes for very long contexts
    BucketType.PREFILL_LATENCY: (
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
    ),
    # ACCUMULATED_PHASE_LATENCY: Request phase timings (100ms to 7680s)
    # Coarser granularity suitable for total request time.
    BucketType.ACCUMULATED_PHASE_LATENCY: (
        0.1,
        0.2,
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
    ),
    # CACHE_RESIDENCY: KV cache block lifetime (1ms to 3600s)
    # Fine granularity at millisecond level for cache hits
    BucketType.CACHE_RESIDENCY: (
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
        3600,
    ),
    # BATCH_SIZE: Dynamic (see get_buckets with max_num_batched_tokens)
    # COMPLETION_COUNT: Request n parameter (1 to 20)
    BucketType.COMPLETION_COUNT: (1, 2, 5, 10, 20),
}


def build_buckets(mantissa_lst: list[int], max_value: int) -> list[int]:
    """
    Builds a list of buckets with increasing powers of 10 multiplied by
    mantissa values until the value exceeds the specified maximum.

    Args:
        mantissa_lst: List of mantissa values (e.g., [1, 2, 5])
        max_value: Maximum bucket value to include

    Returns:
        Sorted list of unique bucket boundary values
    """
    if not mantissa_lst:
        return []

    buckets: set[int] = set()
    exponent = 0
    while True:
        smallest_val_in_exp = float("inf")
        for m in mantissa_lst:
            if m <= 0:
                continue
            value = m * 10**exponent
            if value < smallest_val_in_exp:
                smallest_val_in_exp = value
            if value <= max_value:
                buckets.add(value)

        if smallest_val_in_exp > max_value:
            break
        exponent += 1

    return sorted(buckets)


def build_1_2_5_buckets(max_value: int) -> list[int]:
    """
    Build dynamic buckets using 1-2-5 sequence up to max_value.

    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    return build_buckets([1, 2, 5], max_value)


def get_buckets(
    bucket_type: BucketType,
    max_model_len: int | None = None,
    max_num_batched_tokens: int | None = None,
) -> tuple[float, ...]:
    """
    Get bucket values for a given bucket type.

    Args:
        bucket_type: The type of buckets to retrieve.
        max_model_len: Required for REQUEST_TOKEN_COUNT type.
        max_num_batched_tokens: Optional for BATCH_SIZE type (defaults to 16384).

    Returns:
        Tuple of bucket boundary values.

    Raises:
        ValueError: If max_model_len is required but not provided.
    """
    if bucket_type == BucketType.REQUEST_TOKEN_COUNT:
        if max_model_len is None:
            raise ValueError(
                "max_model_len is required for REQUEST_TOKEN_COUNT buckets"
            )
        return tuple(build_1_2_5_buckets(max_model_len))

    if bucket_type == BucketType.BATCH_SIZE:
        max_batch = max_num_batched_tokens or 16384
        return tuple(build_1_2_5_buckets(max_batch))

    return DEFAULT_BUCKETS[bucket_type]


# Mapping of metric names to their bucket types
# This provides a reference for which bucket type each metric should use
METRIC_BUCKET_MAPPING: dict[str, BucketType] = {
    # TOKEN_STEP_LATENCY metrics
    "vllm:inter_token_latency_seconds": BucketType.TOKEN_STEP_LATENCY,
    "vllm:request_time_per_output_token_seconds": BucketType.TOKEN_STEP_LATENCY,
    # PREFILL_LATENCY metrics
    "vllm:time_to_first_token_seconds": BucketType.PREFILL_LATENCY,
    # ACCUMULATED_PHASE_LATENCY metrics
    "vllm:e2e_request_latency_seconds": BucketType.ACCUMULATED_PHASE_LATENCY,
    "vllm:request_queue_time_seconds": BucketType.ACCUMULATED_PHASE_LATENCY,
    "vllm:request_inference_time_seconds": BucketType.ACCUMULATED_PHASE_LATENCY,
    "vllm:request_prefill_time_seconds": BucketType.ACCUMULATED_PHASE_LATENCY,
    "vllm:request_decode_time_seconds": BucketType.ACCUMULATED_PHASE_LATENCY,
    # CACHE_RESIDENCY metrics
    "vllm:kv_block_lifetime_seconds": BucketType.CACHE_RESIDENCY,
    "vllm:kv_block_idle_before_evict_seconds": BucketType.CACHE_RESIDENCY,
    "vllm:kv_block_reuse_gap_seconds": BucketType.CACHE_RESIDENCY,
    # REQUEST_TOKEN_COUNT metrics
    "vllm:request_prompt_tokens": BucketType.REQUEST_TOKEN_COUNT,
    "vllm:request_generation_tokens": BucketType.REQUEST_TOKEN_COUNT,
    "vllm:request_max_num_generation_tokens": BucketType.REQUEST_TOKEN_COUNT,
    "vllm:request_params_max_tokens": BucketType.REQUEST_TOKEN_COUNT,
    "vllm:request_prefill_kv_computed_tokens": BucketType.REQUEST_TOKEN_COUNT,
    # BATCH_SIZE metrics
    "vllm:iteration_tokens_total": BucketType.BATCH_SIZE,
    # COMPLETION_COUNT metrics
    "vllm:request_params_n": BucketType.COMPLETION_COUNT,
}
