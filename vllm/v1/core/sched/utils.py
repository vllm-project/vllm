# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Sequence

from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.request import Request, RequestStatus


def _occurrence_rules(params: RepetitionDetectionParams) -> list[tuple[int, int]]:
    if params.occurrence_rules is not None:
        return params.occurrence_rules

    max_pattern_size = params.max_pattern_size
    min_pattern_size = params.min_pattern_size
    min_count = params.min_count

    if min_pattern_size <= 0:
        min_pattern_size = 1

    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return []

    return [
        (ngram_size, min_count)
        for ngram_size in range(min_pattern_size, max_pattern_size + 1)
    ]


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


def _has_repeated_ngram_occurrence(
    token_ids: Sequence[int],
    ngram_size: int,
    min_count: int,
) -> bool:
    """Check whether the tail N-gram has appeared min_count times.

    This is evaluated after each generated token. If an older N-gram crossed
    the threshold earlier, it would already have been the tail at that step.
    """
    if len(token_ids) < ngram_size:
        return False

    tail_ngram = tuple(token_ids[-ngram_size:])
    count = 0
    last_start = len(token_ids) - ngram_size
    for start in range(last_start + 1):
        if tuple(token_ids[start : start + ngram_size]) == tail_ngram:
            count += 1
            if count >= min_count:
                return True
    return False


def _check_request_ngram_occurrence(
    request: Request,
    ngram_size: int,
    min_count: int,
) -> bool:
    token_ids = request.output_token_ids
    if len(token_ids) < ngram_size:
        return False

    counts = request.repetition_ngram_counts.setdefault(ngram_size, {})
    next_start = request.repetition_ngram_next_start.get(ngram_size, 0)
    last_start = len(token_ids) - ngram_size

    while next_start <= last_start:
        ngram = tuple(token_ids[next_start : next_start + ngram_size])
        count = counts.get(ngram, 0) + 1
        counts[ngram] = count
        next_start += 1
        request.repetition_ngram_next_start[ngram_size] = next_start
        if count >= min_count:
            return True

    return False


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
    if params.mode == "occurrence":
        for ngram_size, min_count in _occurrence_rules(params):
            if _has_repeated_ngram_occurrence(token_ids, ngram_size, min_count):
                return True
        return False

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
        if pattern_len > len(token_ids):
            return False

        if pattern_len * min_count > len(token_ids):
            return False
        repeated = _has_repeating_pattern(token_ids, pattern_len, min_count)

        if repeated:
            return True

    return False


def check_request_repetition(
    request: Request,
    params: RepetitionDetectionParams,
) -> bool:
    if params.mode == "consecutive":
        return check_sequence_repetition(request.output_token_ids, params)

    for ngram_size, min_count in _occurrence_rules(params):
        if _check_request_ngram_occurrence(request, ngram_size, min_count):
            return True

    return False


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
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    repetition_detection = sampling_params.repetition_detection
    if repetition_detection is not None and (
        check_request_repetition(
            request,
            repetition_detection,
        )
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False
