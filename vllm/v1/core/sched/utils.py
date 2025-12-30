# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib

from vllm.v1.request import Request, RequestStatus


def check_sequence_repetition(
    token_ids: list[int],
    max_repetition_pattern_size: int,
    min_repetition_pattern_size: int,
    repetition_min_count: int,
) -> bool:
    """Check if a sequence of token IDs has a repetition pattern.
    Args:
        token_ids: List of token IDs
        max_repetition_pattern_size: Maximum size of the repetition pattern.
        min_repetition_pattern_size: Minimum size of the repetition pattern.
        repetition_min_count: Minimum number of repetitions to detect.
    Returns:
        True if a repetition pattern is found, False otherwise.
    """
    if min_repetition_pattern_size <= 0:
        min_repetition_pattern_size = 1

    if (
        max_repetition_pattern_size <= 0
        or repetition_min_count < 2
        or min_repetition_pattern_size > max_repetition_pattern_size
    ):
        return False

    for pattern_len in range(
        min_repetition_pattern_size, max_repetition_pattern_size + 1
    ):
        # Check if the pattern is repeated at least min_repetitions times
        if pattern_len * repetition_min_count > len(token_ids):
            return False

        has_repetition = True
        for n in range(1, pattern_len + 1):
            for m in range(1, repetition_min_count):
                if token_ids[-(pattern_len * m + n)] != token_ids[-n]:
                    has_repetition = False
                    break

            if not has_repetition:
                break

        if has_repetition:
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
    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
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

    if check_sequence_repetition(
        list(request.output_token_ids),
        max_repetition_pattern_size=sampling_params.max_repetition_pattern_size,
        min_repetition_pattern_size=sampling_params.min_repetition_pattern_size,
        repetition_min_count=sampling_params.repetition_min_count,
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False
