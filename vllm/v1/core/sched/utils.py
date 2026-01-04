# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib

from vllm.v1.request import Request, RequestStatus


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


def check_repetition(
    output_token_ids: list[int],
    max_consecutive_repeats: int = 0,
) -> bool:
    """Detect repetitive token patterns indicating potential hallucination.

    This function checks for consecutive identical tokens: if the same token
    appears N times in a row, it indicates the model may be stuck in a loop.

    Performance: O(1) in the common case (no repetition). Only performs the
    full check when the last two tokens are identical, which is the only case
    where a long consecutive run could exist.

    Args:
        output_token_ids: List of generated output token IDs
        max_consecutive_repeats: Stop if N consecutive identical tokens detected.
            Set to 0 to disable this check. Typical values: 3-5

    Returns:
        True if repetitive pattern detected (hallucination), False otherwise
    """
    # A single token is not a repetition. For max_consecutive_repeats=1,
    # the behavior is the same as 2 (detects two identical tokens).
    if max_consecutive_repeats == 1:
        max_consecutive_repeats = 2

    num_tokens = len(output_token_ids)

    # Early exit: need at least max_consecutive_repeats tokens
    if max_consecutive_repeats <= 0 or num_tokens < max_consecutive_repeats:
        return False

    # O(1) fast path: if last two tokens differ, no consecutive run exists
    last_token = output_token_ids[-1]
    if output_token_ids[-2] != last_token:
        return False

    # Only reach here if last two tokens are identical.
    # Now count backwards to see if we have max_consecutive_repeats in a row.
    for i in range(3, max_consecutive_repeats + 1):
        if output_token_ids[-i] != last_token:
            return False

    # If we reached here, it means we have max_consecutive_repeats identical tokens.
    return True


def check_stop(
    request: Request, max_model_len: int, pooler_output: torch.Tensor | None = None
) -> bool:
    assert not request.pooling_params
    if request.pooling_params:
        if pooler_output is not None:
            request.status = RequestStatus.FINISHED_STOPPED
            return True
        return False

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

    # Check for repetitive token patterns (hallucination detection)
    max_consecutive = getattr(sampling_params, "max_consecutive_repeats", 0)

    if max_consecutive > 0 and check_repetition(
        list(request.output_token_ids),
        max_consecutive_repeats=max_consecutive,
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False
