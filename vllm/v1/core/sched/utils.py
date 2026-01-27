# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Scheduler utilities for vLLM V1 engine.

This module provides utility functions for the scheduler, including:
- List manipulation helpers
- Stop condition checking with infinite loop detection

The infinite loop detection is particularly important for models like
PaddleOCR-VL that may enter repetitive generation patterns without
proper EOS token emission.
"""

import contextlib

from vllm.logger import init_logger
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


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


# =============================================================================
# Infinite Loop Detection Constants
# =============================================================================
# These constants are tuned based on empirical testing with multimodal models
# like PaddleOCR-VL. The detection algorithm uses a sliding window approach
# to identify when the model enters a degenerate state where it repeatedly
# generates the same token sequence without emitting an EOS token.

# Minimum number of output tokens before checking for infinite loops.
# This prevents false positives during normal table/structured content
# generation where legitimate repetition (e.g., cell delimiters) may occur.
INFINITE_LOOP_MIN_TOKENS = 60

# Size of the sliding window for pattern comparison.
# A window of 30 tokens provides a good balance between:
# - Avoiding false positives from short legitimate repetitions
# - Detecting actual infinite loops quickly enough to save compute
INFINITE_LOOP_WINDOW_SIZE = 30


def _compute_sequence_hash(tokens: list, start: int, length: int) -> int:
    """Compute a hash for a subsequence of tokens using polynomial rolling hash.

    This provides efficient comparison capability for detecting repeated patterns.
    The hash function uses a prime base with a large Mersenne prime modulus
    to minimize collision probability.

    Args:
        tokens: The full token sequence
        start: Starting index of the subsequence
        length: Length of the subsequence to hash

    Returns:
        Integer hash value for the subsequence

    Time Complexity: O(length)
    Space Complexity: O(1)
    """
    # Prime base for polynomial hashing - chosen to minimize collisions
    # Using 31 as it's a common choice that works well with ASCII-range values
    BASE = 31
    # Mersenne prime (2^61 - 1) provides excellent distribution and allows
    # efficient modulo operations on 64-bit systems
    MOD = 2305843009213693951  # 2**61 - 1

    hash_val = 0
    for i in range(length):
        hash_val = (hash_val * BASE + tokens[start + i]) % MOD
    return hash_val


def _detect_infinite_loop(output_token_ids: list) -> bool:
    """Detect if the model has entered an infinite loop state.

    Uses a sliding window approach with hash-based comparison for efficiency.
    The algorithm checks if the last N tokens are identical to the preceding
    N tokens, which indicates the model is stuck in a repetitive pattern.

    This is particularly important for multimodal models like PaddleOCR-VL
    that can get "stuck" on certain visual features and produce repetitive
    outputs like:
        - "6.3.1. 2 2 6.3.2 6.3.3. 6.3.1. 2 2 6.3.2 6.3.3. ..."
        - "곧 곧 곧 곧 곧 곧 곧 곧 곧 곧 ..."

    Algorithm:
        1. Early exit if not enough tokens (< 2 * window_size)
        2. Compute hash of last window (tokens[-window:])
        3. Compute hash of previous window (tokens[-2*window:-window])
        4. If hashes differ, no loop detected (fast path)
        5. If hashes match, verify with direct comparison (handle collisions)

    Args:
        output_token_ids: List of generated token IDs

    Returns:
        True if infinite loop detected, False otherwise

    Time Complexity: O(window_size) for hash computation
    Space Complexity: O(1) additional space
    """
    num_tokens = len(output_token_ids)

    # Early exit: need at least 2 windows worth of tokens
    if num_tokens < INFINITE_LOOP_MIN_TOKENS:
        return False

    window = INFINITE_LOOP_WINDOW_SIZE
    if num_tokens < 2 * window:
        return False

    # Define window boundaries
    # prev_window: tokens[num_tokens - 2*window : num_tokens - window]
    # curr_window: tokens[num_tokens - window : num_tokens]
    prev_start = num_tokens - 2 * window
    curr_start = num_tokens - window

    # Compute hashes for both windows
    prev_hash = _compute_sequence_hash(output_token_ids, prev_start, window)
    curr_hash = _compute_sequence_hash(output_token_ids, curr_start, window)

    # Fast path: if hashes don't match, definitely no infinite loop
    if prev_hash != curr_hash:
        return False

    # Hashes match - verify with direct comparison to handle hash collisions.
    # This is O(window_size) but only executed when hashes match, which is
    # rare for non-repetitive sequences.
    for i in range(window):
        if output_token_ids[prev_start + i] != output_token_ids[curr_start + i]:
            return False

    # Confirmed: last window is identical to previous window
    return True


def check_stop(
    request: Request, max_model_len: int, enable_infinite_loop_detection: bool = True
) -> bool:
    """Check if generation should stop for the given request.

    This function implements multiple stopping conditions in order of
    computational cost (cheapest checks first):

    1. Minimum token requirement not met (O(1))
    2. Infinite loop detected (O(window_size), only if enabled)
    3. EOS token generated (O(1))
    4. Stop token generated (O(k) where k = number of stop tokens)
    5. Maximum length reached (O(1))

    The infinite loop detection is particularly important for multimodal
    models like PaddleOCR-VL that may enter degenerate states. In these
    states, the model repeatedly generates the same token sequence without
    emitting an EOS token, which:
    - Wastes GPU compute resources
    - Blocks other requests in the batch
    - May run until max_tokens limit (e.g., 8192 tokens)

    Root Cause (PaddleOCR-VL Infinite Loop):
        The model's attention mechanism can get "stuck" on certain visual
        features, particularly when processing:
        - Complex table structures with repetitive cells
        - Images with repetitive visual patterns
        - Low-quality or ambiguous content

        When stuck, the model generates patterns like:
        - "6.3.1. 2 2 6.3.2 6.3.3. 6.3.1. 2 2 6.3.2 6.3.3. ..."
        - "곧 곧 곧 곧 곧 곧 곧 곧 곧 곧 ..." (Korean repetition)

        Note: The "2" in the first pattern is NOT the EOS token (which also
        has ID 2), but a different token representing the digit. The model
        never actually outputs the EOS token in these degenerate states.

    Args:
        request: The request to check
        max_model_len: Maximum model context length
        enable_infinite_loop_detection: Whether to check for infinite loops.
            Default is True. Can be disabled for models known not to have
            this issue or when precise control over stopping is needed.

    Returns:
        True if generation should stop, False otherwise

    Side Effects:
        Sets request.status and optionally request.stop_reason when stopping.
    """
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    # Check 1: Minimum tokens not yet generated
    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    # ==========================================================================
    # Check 2: Infinite Loop Detection
    # ==========================================================================
    # This check is placed early to prevent wasted computation on requests
    # stuck in an infinite loop. The detection is O(window_size) which is
    # much cheaper than continuing to generate tokens indefinitely.
    #
    # We check this BEFORE the EOS check because in infinite loop states,
    # the model never outputs EOS - it just keeps repeating the same pattern.
    if enable_infinite_loop_detection and _detect_infinite_loop(
        request.output_token_ids
    ):
        logger.warning(
            "Infinite loop detected for request %s. "
            "Last %d tokens repeat the previous %d tokens. "
            "Stopping generation to prevent resource exhaustion.",
            request.request_id,
            INFINITE_LOOP_WINDOW_SIZE,
            INFINITE_LOOP_WINDOW_SIZE,
        )
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = "infinite_loop_detected"
        return True

    last_token_id = request.output_token_ids[-1]

    # Check 3: EOS token from tokenizer
    # Note: request.eos_token_id is set from the tokenizer configuration,
    # which properly handles model-specific EOS tokens (e.g., token ID 2
    # for PaddleOCR-VL, LLaMA, and many other models).
    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    # Check 4: User-specified stop tokens
    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True

    # Check 5: Maximum length reached
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    return False
