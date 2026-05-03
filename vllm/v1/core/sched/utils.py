# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from array import array
from collections.abc import Sequence

from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.request import Request, RequestStatus


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


# Double polynomial rolling hash parameters. MOD1 is the Mersenne prime 2^61-1
# and MOD2 is 2^31-1; collision probability per pair is < 2^-90, negligible
# for repetition detection at LLM-output scales.
_RH_MOD1 = (1 << 61) - 1
_RH_MOD2 = (1 << 31) - 1
_RH_BASE1 = 131
_RH_BASE2 = 137


class RollingHashState:
    """Incremental double-rolling-hash state for repetition detection.

    Owned by a single ``Request`` for the duration of its decode; updated
    in O(1) per output token via :meth:`extend`. Range-hash queries over
    any sub-interval ``[l, r)`` are O(1).

    Storage: four ``array.array("q")`` buffers (signed int64, 8 bytes each
    plus amortized growth slack), so the memory cost is roughly
    ``4 * 8 * num_output_tokens`` bytes — about 7x smaller than the
    equivalent Python ``list[int]`` because raw int64 elements are not
    boxed as Python objects.
    """

    __slots__ = ("h1", "h2", "p1", "p2", "n")

    def __init__(self) -> None:
        # h*[i] = hash of token_ids[0..i); p*[i] = base^i mod _RH_MOD*.
        self.h1: array = array("q", [0])
        self.h2: array = array("q", [0])
        self.p1: array = array("q", [1])
        self.p2: array = array("q", [1])
        self.n: int = 0

    def extend(self, token_ids: Sequence[int], up_to: int) -> None:
        """Extend internal hashes to cover ``token_ids[:up_to]``.

        Idempotent and append-only: callers may pass the entire output
        sequence each step; only tokens beyond ``self.n`` are processed.
        """
        if up_to <= self.n:
            return
        h1, h2, p1, p2 = self.h1, self.h2, self.p1, self.p2
        last_h1 = h1[-1]
        last_h2 = h2[-1]
        last_p1 = p1[-1]
        last_p2 = p2[-1]
        for i in range(self.n, up_to):
            # +1 so token id 0 contributes a non-zero term.
            t = token_ids[i] + 1
            last_h1 = (last_h1 * _RH_BASE1 + t) % _RH_MOD1
            last_h2 = (last_h2 * _RH_BASE2 + t) % _RH_MOD2
            last_p1 = last_p1 * _RH_BASE1 % _RH_MOD1
            last_p2 = last_p2 * _RH_BASE2 % _RH_MOD2
            h1.append(last_h1)
            h2.append(last_h2)
            p1.append(last_p1)
            p2.append(last_p2)
        self.n = up_to


def _has_repeating_pattern_rolling_hash(
    token_ids: Sequence[int],
    max_pattern_size: int,
    min_pattern_size: int,
    min_count: int,
    state: RollingHashState | None = None,
) -> bool:
    """Detect tail repetition via double polynomial rolling hashes.

    For each candidate period L in ``[min_pattern_size, upper_l]``, checks
    whether the last ``min_count * L`` tokens are exactly ``min_count``
    copies of an L-block. A cheap single-token compare (Stage 1) prunes
    most L values; the hash equality (Stage 2) confirms full block match
    in O(min_count) per L using O(1) range hashes.

    ``max_pattern_size <= 0`` removes the upper bound (capped only by
    ``len(token_ids) // min_count``). Otherwise capped by
    ``min(max_pattern_size, len(token_ids) // min_count)``.

    When ``state`` is provided, prefix hashes are reused across calls and
    extended incrementally — total cost is O(1) per token plus O(L_max)
    scan per step. When ``state`` is ``None`` (e.g. direct invocations
    from tests), prefix hashes are recomputed locally.
    """
    n = len(token_ids)
    # Keep the request-attached state consistent with the latest output
    # tokens regardless of whether we end up scanning. The per-token
    # cost is amortized O(1), and downstream callers (and tests) can
    # assume ``state.n == len(token_ids)`` after this function returns.
    if state is not None and n > state.n:
        state.extend(token_ids, n)

    if n == 0 or min_count < 2:
        return False

    if min_pattern_size <= 0:
        min_pattern_size = 1

    upper_l = n // min_count
    if max_pattern_size > 0:
        upper_l = min(upper_l, max_pattern_size)

    if upper_l < min_pattern_size:
        return False

    if state is not None:
        h1, h2, p1, p2 = state.h1, state.h2, state.p1, state.p2
    else:
        h1 = [0] * (n + 1)
        h2 = [0] * (n + 1)
        p1 = [1] * (n + 1)
        p2 = [1] * (n + 1)
        for i in range(n):
            t = token_ids[i] + 1
            h1[i + 1] = (h1[i] * _RH_BASE1 + t) % _RH_MOD1
            h2[i + 1] = (h2[i] * _RH_BASE2 + t) % _RH_MOD2
            p1[i + 1] = p1[i] * _RH_BASE1 % _RH_MOD1
            p2[i + 1] = p2[i] * _RH_BASE2 % _RH_MOD2

    # Hot loop: read into locals to avoid attribute lookups.
    mod1 = _RH_MOD1
    mod2 = _RH_MOD2
    last_token = token_ids[-1]
    h1_n = h1[n]
    h2_n = h2[n]

    for length in range(min_pattern_size, upper_l + 1):
        # Stage 1: single-token prune. Eliminates most L cheaply.
        if token_ids[n - 1 - length] != last_token:
            continue
        # Stage 2: hash equality across all min_count blocks.
        p1_l = p1[length]
        p2_l = p2[length]
        ref1 = (h1_n - h1[n - length] * p1_l) % mod1
        ref2 = (h2_n - h2[n - length] * p2_l) % mod2
        match = True
        for k in range(1, min_count):
            r = n - length * k
            l_ = r - length
            if (h1[r] - h1[l_] * p1_l) % mod1 != ref1 or (
                h2[r] - h2[l_] * p2_l
            ) % mod2 != ref2:
                match = False
                break
        if match:
            return True
    return False


def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
    state: RollingHashState | None = None,
) -> bool:
    """Check if a sequence of token IDs has a repetition pattern.

    Args:
        token_ids: List of token IDs.
        params: Repetition detection parameters.
        state: Optional incremental rolling-hash state. Only consulted
            when ``params.algorithm == "rolling_hash"``. The ``check_stop``
            scheduler entry passes a per-request state for amortized O(1)
            per-token updates; direct callers may omit it.

    Returns:
        True if a repetition pattern is found, False otherwise.
    """
    if params.algorithm == "rolling_hash":
        return _has_repeating_pattern_rolling_hash(
            token_ids,
            params.max_pattern_size,
            params.min_pattern_size,
            params.min_count,
            state=state,
        )

    # naive (default)
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
    if repetition_detection is not None:
        state: RollingHashState | None = None
        if repetition_detection.algorithm == "rolling_hash":
            state = request.repetition_hash_state
            if state is None:
                state = RollingHashState()
                request.repetition_hash_state = state
        if check_sequence_repetition(
            request.output_token_ids,
            repetition_detection,
            state=state,
        ):
            request.status = RequestStatus.FINISHED_REPETITION
            request.stop_reason = "repetition_detected"
            return True

    return False
