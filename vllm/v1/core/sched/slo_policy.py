# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.v1.request import Request
from vllm.v1.utils import record_function_or_nullcontext

SLO_PREFILL_LENGTH_BONUS = 10.0
SLO_TTFT_BASE_MS = 20_000.0
SLO_TTFT_SHORT_MAX_MS = 40_000.0
SLO_TTFT_LONG_MAX_MS = 120_000.0
SLO_BIAS_FAVORED_LIMIT = 1.0
SLO_BIAS_MAX = 10.0


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def compute_request_ttft_slo_ms(
    num_prompt_tokens: int,
    priority_bias: float,
    short_request_token_threshold: int,
) -> float:
    magnitude = abs(priority_bias)
    if magnitude <= SLO_BIAS_FAVORED_LIMIT:
        expansion = _smoothstep(magnitude)
        favored_ttft_ms = SLO_TTFT_BASE_MS
        nonfavored_ttft_ms = SLO_TTFT_BASE_MS + (
            SLO_TTFT_SHORT_MAX_MS - SLO_TTFT_BASE_MS
        ) * expansion
    else:
        normalized = (magnitude - SLO_BIAS_FAVORED_LIMIT) / (
            SLO_BIAS_MAX - SLO_BIAS_FAVORED_LIMIT
        )
        expansion = _smoothstep(normalized)
        favored_ttft_ms = SLO_TTFT_BASE_MS + (
            SLO_TTFT_SHORT_MAX_MS - SLO_TTFT_BASE_MS
        ) * expansion
        nonfavored_ttft_ms = SLO_TTFT_SHORT_MAX_MS + (
            SLO_TTFT_LONG_MAX_MS - SLO_TTFT_SHORT_MAX_MS
        ) * expansion

    short_is_favored = priority_bias >= 0
    short_ttft_ms = favored_ttft_ms if short_is_favored else nonfavored_ttft_ms
    long_ttft_ms = nonfavored_ttft_ms if short_is_favored else favored_ttft_ms
    return short_ttft_ms if num_prompt_tokens <= short_request_token_threshold else long_ttft_ms


def set_request_slo_constraints(request: Request, scheduler_config: Any) -> None:
    request.ttft_slo_ms = compute_request_ttft_slo_ms(
        request.num_prompt_tokens,
        scheduler_config.slo_priority_bias,
        scheduler_config.slo_short_request_token_threshold,
    )


def compute_slo_score(
    request: Request,
    now: float,
    max_model_len: int,
    prefill_length_bonus: float,
) -> float:
    with record_function_or_nullcontext("slo_policy: compute_slo_score"):
        score = 0.0
        if request.first_token_ts is None:
            wait_time_ms = (now - request.arrival_time) * 1000.0
            if request.ttft_slo_ms < float("inf") and request.ttft_slo_ms > 0:
                score += 200.0 * max(0.0, wait_time_ms / request.ttft_slo_ms)

        if (
            request.num_computed_tokens == 0
            and request.num_prompt_tokens > 0
            and prefill_length_bonus > 0
        ):
            prompt_fraction = min(request.num_prompt_tokens / max_model_len, 1.0)
            score += prefill_length_bonus * (1.0 - prompt_fraction)

        return score


def is_waiting_reserve_candidate(
    request: Request,
    score: float,
    now: float,
    prefill_length_bonus: float,
) -> bool:
    if request.first_token_ts is not None or request.ttft_slo_ms <= 0:
        return False
    if request.ttft_slo_ms == float("inf"):
        return False

    wait_time_ms = max(0.0, (now - request.arrival_time) * 1000.0)
    if wait_time_ms < 0.6 * request.ttft_slo_ms:
        return False
    return score >= 0.8 * prefill_length_bonus


def compute_waiting_token_reserve(
    max_num_scheduled_tokens: int,
    waiting_token_reserve_ratio: float,
    running_count: int,
    running_decode_count: int,
    reserve_candidates: list[tuple[float, int]],
) -> int:
    with record_function_or_nullcontext("slo_policy: compute_waiting_token_reserve"):
        if (
            waiting_token_reserve_ratio <= 0
            or not reserve_candidates
            or running_decode_count > max(1, running_count // 2)
        ):
            return 0

        reserve_cap = max(1, int(max_num_scheduled_tokens * waiting_token_reserve_ratio))
        _score, remaining_prompt = max(reserve_candidates)
        return min(remaining_prompt, reserve_cap)
