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


def _build_powers(count: int, base: int, mod: int) -> array:
    powers = array("q", [0]) * count
    powers[0] = 1
    for i in range(1, count):
        powers[i] = powers[i - 1] * base % mod
    return powers


class RollingHashState:
    """Incremental double-rolling-hash state for repetition detection.

    Owned by a single ``Request`` for the duration of its decode; updated
    in O(1) per output token via :meth:`extend`. Range-hash queries over
    any sub-interval ``[l, r)`` are O(1).

    In **bounded** mode (``max_pattern_size > 0``) the search only ever
    queries hashes spanning the last ``W = max_pattern_size * min_count``
    tokens, and powers up to length ``max_pattern_size``. Two memory
    optimizations follow:

    * ``p1`` / ``p2`` are pre-filled to ``max_pattern_size + 1`` entries
      and never grow.
    * ``h1`` / ``h2`` slide: when the stored prefix exceeds ``2W + 1``
      entries, the leading ``W`` are dropped and ``self.start`` is
      advanced by ``W``. Indexing becomes ``h*[abs_pos - self.start]``.

    In **unbounded** mode (``max_pattern_size <= 0``) the prefix arrays
    grow per token (current behaviour).

    Storage uses ``array.array("q")`` (signed int64) — about 7x smaller
    than ``list[int]`` since raw int64 elements are not boxed.
    """

    __slots__ = ("h1", "h2", "p1", "p2", "n", "start", "_window")

    def __init__(self, max_pattern_size: int = 0, min_count: int = 0) -> None:
        self.h1: array = array("q", [0])
        self.h2: array = array("q", [0])
        self.n: int = 0
        # Absolute index of ``h*[0]``. Advances on window slides.
        self.start: int = 0
        if max_pattern_size > 0 and min_count >= 2:
            # Bounded mode: cap powers and enable sliding window.
            self._window: int = max_pattern_size * min_count
            self.p1: array = _build_powers(max_pattern_size + 1, _RH_BASE1, _RH_MOD1)
            self.p2: array = _build_powers(max_pattern_size + 1, _RH_BASE2, _RH_MOD2)
        else:
            self._window = 0
            self.p1 = array("q", [1])
            self.p2 = array("q", [1])

    def extend(self, token_ids: Sequence[int], up_to: int) -> None:
        """Sync internal hashes to cover ``token_ids[:up_to]`` (idempotent).

        Handles growth, truncation (speculative-decode rollback shrinks
        ``request.output_token_ids``), and — in bounded mode — periodic
        sliding to bound memory.
        """
        if up_to == self.n:
            return
        if up_to < self.n:
            if up_to < self.start:
                # Rollback exceeds the kept window; reset and rebuild.
                self.h1 = array("q", [0])
                self.h2 = array("q", [0])
                self.start = 0
                self.n = 0
            else:
                keep = up_to - self.start + 1
                del self.h1[keep:]
                del self.h2[keep:]
                self.n = up_to
                return
        # Growth path.
        h1, h2 = self.h1, self.h2
        last_h1 = h1[-1]
        last_h2 = h2[-1]
        for i in range(self.n, up_to):
            # +1 so token id 0 contributes a non-zero term.
            t = token_ids[i] + 1
            last_h1 = (last_h1 * _RH_BASE1 + t) % _RH_MOD1
            last_h2 = (last_h2 * _RH_BASE2 + t) % _RH_MOD2
            h1.append(last_h1)
            h2.append(last_h2)
        self.n = up_to
        # Bounded mode: slide head off when buffer exceeds 2W + 1.
        window = self._window
        if window > 0:
            stored = up_to - self.start + 1
            if stored > 2 * window + 1:
                drop = window
                del self.h1[:drop]
                del self.h2[:drop]
                self.start += drop
        else:
            # Unbounded mode: grow powers lazily to cover any L the
            # search may query (capped by ``up_to // 2`` since
            # ``min_count >= 2``).
            target = up_to // 2
            cur = len(self.p1) - 1
            if target > cur:
                p1, p2 = self.p1, self.p2
                last_p1 = p1[-1]
                last_p2 = p2[-1]
                for _ in range(cur, target):
                    last_p1 = last_p1 * _RH_BASE1 % _RH_MOD1
                    last_p2 = last_p2 * _RH_BASE2 % _RH_MOD2
                    p1.append(last_p1)
                    p2.append(last_p2)


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

    A per-request ``state`` reuses prefix hashes across decode steps
    (O(1) per new token). Direct test callers may omit it; in that case
    a temporary state is built once locally.
    """
    n = len(token_ids)
    if state is None:
        state = RollingHashState(max_pattern_size, min_count)
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

    h1, h2, p1, p2 = state.h1, state.h2, state.p1, state.p2
    start = state.start

    mod1 = _RH_MOD1
    mod2 = _RH_MOD2
    last_token = token_ids[-1]
    h1_n = h1[n - start]
    h2_n = h2[n - start]

    for length in range(min_pattern_size, upper_l + 1):
        if token_ids[n - 1 - length] != last_token:
            continue
        p1_l = p1[length]
        p2_l = p2[length]
        ref1 = (h1_n - h1[n - length - start] * p1_l) % mod1
        ref2 = (h2_n - h2[n - length - start] * p2_l) % mod2
        match = True
        for k in range(1, min_count):
            r = n - length * k - start
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
                state = RollingHashState(
                    repetition_detection.max_pattern_size,
                    repetition_detection.min_count,
                )
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
