# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Sequence

import vllm.envs as envs  # cohere
from vllm.logger import init_logger  # cohere
from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)  # cohere


def _has_repeating_pattern(
    token_ids: Sequence[int],
    pattern_len: int,
    repetition_min_count: int,
) -> bool:
    """Check if the tail of token_ids contains a repeating pattern.

    Compares the last pattern_len tokens against the preceding
    (repetition_min_count - 1) repetitions of the same length.
    """
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
) -> bool:
    """Check if a sequence of token IDs has a repetition pattern.
    Args:
        token_ids: List of token IDs
        params: Repetition detection parameters.
    Returns:
        True if a repetition pattern is found, False otherwise.
    """
    max_pattern_size = params.max_pattern_size
    min_pattern_size = params.min_pattern_size
    min_count = params.min_count

    if min_pattern_size <= 0:
        min_pattern_size = 1

    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False

    for pattern_len in range(
        min_pattern_size,
        max_pattern_size + 1,
    ):
        if pattern_len * min_count > len(token_ids):
            return False

        if _has_repeating_pattern(token_ids, pattern_len, min_count):
            return True

    return False


# cohere start
def _has_hit_token_repetition_limit(
    request: Request,
    repetition_limit: int,
    max_sequence_length: int,
) -> bool:
    """Check if output tokens contain a repeating pattern.

    Uses incremental streak tracking stored on the request object.
    For each sequence length k (1 to max_sequence_length), tracks
    how many consecutive positions ending at the latest token
    satisfy tokens[i] == tokens[i - k]. When the streak reaches
    (repetition_limit - 1) * k, a repeating pattern is detected.

    On first call, bootstraps by scanning backwards. Subsequent
    calls process only the latest token in O(max_sequence_length).
    """
    tokens = request.output_token_ids
    num_tokens = len(tokens)
    if num_tokens < repetition_limit:
        return False

    max_possible = num_tokens // repetition_limit
    if max_sequence_length > max_possible:
        max_sequence_length = max_possible

    last_idx = num_tokens - 1
    streaks = request._repetition_streaks

    if streaks is None:
        streaks = []
        for k in range(1, max_sequence_length + 1):
            streak = 0
            i = last_idx
            while i >= k and tokens[i] == tokens[i - k]:
                streak += 1
                i -= 1
            streaks.append(streak)
        request._repetition_streaks = streaks
    else:
        while len(streaks) < max_sequence_length:
            k = len(streaks) + 1
            streak = 0
            i = last_idx - 1
            while i >= k and tokens[i] == tokens[i - k]:
                streak += 1
                i -= 1
            streaks.append(streak)
        for k_idx in range(min(max_sequence_length, last_idx)):
            k = k_idx + 1
            if tokens[last_idx] == tokens[last_idx - k]:
                streaks[k_idx] += 1
            else:
                streaks[k_idx] = 0

    for k_idx in range(max_sequence_length):
        k = k_idx + 1
        if streaks[k_idx] >= (repetition_limit - 1) * k:
            return True

    return False


# cohere end


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from
        items_to_remove: Set of items to remove

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]


def check_stop(request: Request, max_model_len: int) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    # cohere start
    if (
        envs.VLLM_REPETITION_LIMIT > 0
        and envs.VLLM_REPETITION_MAX_SEQUENCE_LENGTH > 0
        and _has_hit_token_repetition_limit(
            request,
            envs.VLLM_REPETITION_LIMIT,
            envs.VLLM_REPETITION_MAX_SEQUENCE_LENGTH,
        )
    ):
        logger.error(
            "Request %s hit token repetition limit "
            "(repetition_limit=%d, max_sequence_length=%d, "
            "num_output_tokens=%d). Stopping generation.",
            request.request_id,
            envs.VLLM_REPETITION_LIMIT,
            envs.VLLM_REPETITION_MAX_SEQUENCE_LENGTH,
            request.num_output_tokens,
        )
        request.status = RequestStatus.FINISHED_REPETITION
        return True
    # cohere end
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    repetition_detection = sampling_params.repetition_detection
    if repetition_detection is not None and (
        check_sequence_repetition(
            request.output_token_ids,
            repetition_detection,
        )
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False
