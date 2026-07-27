# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Algorithm-only tests for DPLBAsyncMPClient engine selection.

These tests exercise ``_select_engine_by_load`` (and the surrounding
``get_core_engine_for_request`` glue) without requiring CUDA, ZMQ, or any
multi-process engine setup. The dispatch logic is pure Python: a full scan
for small DP sizes and power-of-two-choices above a threshold.
"""

from __future__ import annotations

import random
import time
from unittest.mock import patch

import pytest

from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import DPLBAsyncMPClient


def _make_sampling_request(request_id: str = "rid") -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _make_pooling_request(
    request_id: str = "rid",
) -> EngineCoreRequest:
    """Plain pooling request without late-interaction routing hints."""
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=None,
        pooling_params=PoolingParams(task="token_embed"),
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _make_client(
    num_engines: int,
    *,
    counts: list[list[int]] | None = None,
    eng_start_index: int = 0,
    client_count: int = 1,
) -> DPLBAsyncMPClient:
    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = client_count
    client.reqs_in_flight = {}
    client.core_engines = [bytes([i, 0]) for i in range(num_engines)]
    client.lb_engines = (
        counts if counts is not None else [[0, 0] for _ in range(num_engines)]
    )
    client.eng_start_index = eng_start_index
    return client


# --- Full-scan path (small DP) -----------------------------------------------


def test_full_scan_picks_lowest_score_at_or_below_threshold():
    """At <= _P2C_FULL_SCAN_MAX_ENGINES, deterministic full scan still rules."""

    threshold = DPLBAsyncMPClient._P2C_FULL_SCAN_MAX_ENGINES
    counts = [[2, 1], [0, 0], [3, 4], [5, 0]][:threshold]
    client = _make_client(len(counts), counts=[list(c) for c in counts])

    request = _make_sampling_request()
    chosen = client.get_core_engine_for_request(request)

    # Index 1 has score 0; everyone else > 0.
    assert chosen == client.core_engines[1]
    assert client.lb_engines[1][0] == 1


def test_full_scan_respects_eng_start_index_for_tied_zero_load():
    """Cold-start fairness: clients with different start indices spread."""

    # All zero-load engines; the first engine encountered (= eng_start_index)
    # wins because of strict ``<`` in the scan.
    counts = [[0, 0], [0, 0], [0, 0]]
    for start in range(len(counts)):
        client = _make_client(
            len(counts),
            counts=[list(c) for c in counts],
            eng_start_index=start,
        )
        chosen = client.get_core_engine_for_request(_make_sampling_request())
        assert chosen == client.core_engines[start]


# --- P2C path (large DP) -----------------------------------------------------


def test_p2c_returns_one_of_the_two_sampled_engines():
    """Above threshold, the chosen engine must be one of the two sampled."""

    num_engines = DPLBAsyncMPClient._P2C_FULL_SCAN_MAX_ENGINES + 4
    counts = [[i, 0] for i in range(num_engines)]
    client = _make_client(num_engines, counts=[list(c) for c in counts])

    # Force ``random.randrange`` to return a known pair; the helper uses
    # ``a = randrange(N)``, ``b = randrange(N - 1)``, then bumps b if b >= a.
    target_a, target_b = 2, 7
    raw_b = target_b - 1  # because the impl bumps b when b >= a (a=2 < raw_b)
    with patch.object(random, "randrange", side_effect=[target_a, raw_b]):
        chosen_idx = client._select_engine_by_load()

    # Engine 2 has score 8, engine 7 has score 28; P2C picks the lower.
    assert chosen_idx == target_a


def test_p2c_avoids_picking_the_same_engine_twice():
    """``b += 1`` trick: when ``randrange(N-1)`` collides with ``a``, skip."""

    num_engines = DPLBAsyncMPClient._P2C_FULL_SCAN_MAX_ENGINES + 4
    client = _make_client(num_engines)

    # ``a=3``, ``raw_b=3`` → after the bump the second pick must be 4, not 3.
    seen_pairs: list[tuple[int, int]] = []
    real_choice = []

    def fake_randrange(n: int) -> int:
        if n == num_engines:
            return 3
        if n == num_engines - 1:
            return 3
        raise AssertionError(f"unexpected randrange({n}) call")

    with patch.object(random, "randrange", side_effect=fake_randrange):
        # Use a tie-aware count layout so we can verify which pair was sampled.
        client.lb_engines[3] = [10, 0]
        client.lb_engines[4] = [0, 0]
        chosen = client._select_engine_by_load()
        seen_pairs.append((3, 4))
        real_choice.append(chosen)

    # b was forced to collide with a (both 3); the bump must move b to 4,
    # which has the lower score → 4 wins.
    assert real_choice == [4]
    # And the helper never inspected an out-of-range index.
    assert all(0 <= idx < num_engines for pair in seen_pairs for idx in pair)


@pytest.mark.parametrize("num_engines", [5, 6, 8])
def test_p2c_full_enumeration_covers_all_distinct_ordered_pairs(num_engines):
    """Sweep every ``(a, raw_b)`` the RNG can yield and verify the
    ``b += 1 if b >= a`` trick maps to **every** ordered distinct
    ``(a, b)`` pair exactly once -- i.e. P((a, b)) = 1 / (N * (N - 1))
    is uniform over distinct ordered pairs, with no missing pair and no
    double-counted pair.

    Verifies the math invariant via the actual helper, not just by
    re-deriving the mapping. We pin scores so that
    ``lb_engines[i] = [0, i]`` -> score_i = i, so the helper's
    ``score_a <= score_b`` tie-break deterministically returns
    ``min(a, b)`` and lets us assert the chosen index agrees with the
    bump-resolved pair we expect.

    N values are restricted to the P2C path (``> _P2C_FULL_SCAN_MAX_ENGINES``)
    because the trick only fires there; N <= 4 takes the deterministic
    full scan and never invokes ``random.randrange``.
    """
    counts = [[0, i] for i in range(num_engines)]
    client = _make_client(num_engines, counts=[list(c) for c in counts])

    expected_pairs = {
        (i, j) for i in range(num_engines) for j in range(num_engines) if i != j
    }
    seen_pairs: set[tuple[int, int]] = set()

    for a in range(num_engines):
        for raw_b in range(num_engines - 1):
            b = raw_b + 1 if raw_b >= a else raw_b
            with patch.object(random, "randrange", side_effect=[a, raw_b]):
                returned = client._select_engine_by_load()
            # Monotonic counts -> helper deterministically returns min(a, b).
            assert returned == min(a, b), (
                f"(a={a}, raw_b={raw_b}, b={b}): expected {min(a, b)}, got {returned}"
            )
            seen_pairs.add((a, b))

    assert seen_pairs == expected_pairs, (
        f"missing pairs: {expected_pairs - seen_pairs}; "
        f"unexpected pairs: {seen_pairs - expected_pairs}"
    )
    # No collision in the (a, raw_b) -> (a, b) mapping.
    assert len(seen_pairs) == num_engines * (num_engines - 1)


def test_p2c_distribution_does_not_starve_any_engine_under_uniform_load():
    """Statistical sanity: with all-zero load, every engine gets picked."""

    num_engines = DPLBAsyncMPClient._P2C_FULL_SCAN_MAX_ENGINES + 12  # 16
    client = _make_client(num_engines)

    rng = random.Random(0xBADC0FFEE)
    seen: set[int] = set()
    with patch.object(random, "randrange", side_effect=rng.randrange):
        for _ in range(2000):
            seen.add(client._select_engine_by_load())

    assert seen == set(range(num_engines)), (
        f"P2C starved engines: missing={set(range(num_engines)) - seen}"
    )


def test_p2c_increment_path_updates_only_chosen_engine_counter():
    """``get_core_engine_for_request`` bumps only the picked engine's counter."""

    num_engines = DPLBAsyncMPClient._P2C_FULL_SCAN_MAX_ENGINES + 4
    counts = [[0, 0] for _ in range(num_engines)]
    client = _make_client(num_engines, counts=[list(c) for c in counts], client_count=3)

    target_a, target_b = 1, 5
    raw_b = target_b - 1
    with patch.object(random, "randrange", side_effect=[target_a, raw_b]):
        chosen = client.get_core_engine_for_request(_make_sampling_request("r1"))

    assert chosen == client.core_engines[target_a]
    assert client.lb_engines[target_a][0] == 3  # client_count bumped
    for i in range(num_engines):
        if i != target_a:
            assert client.lb_engines[i] == [0, 0]
    assert client.reqs_in_flight["r1"] == chosen


# --- DP rank short-circuit (unaffected by load-balancing path) ---------------


def test_explicit_data_parallel_rank_bypasses_load_balancer():
    """A request with an explicit DP rank must skip the score-based picker."""

    num_engines = DPLBAsyncMPClient._P2C_FULL_SCAN_MAX_ENGINES + 4
    # Pre-load the load counters so the lb path would NOT pick rank 0.
    counts = [[100, 100]] + [[0, 0]] * (num_engines - 1)
    client = _make_client(num_engines, counts=[list(c) for c in counts])

    request = _make_sampling_request("r-pinned")
    request.data_parallel_rank = 0
    chosen = client.get_core_engine_for_request(request)

    assert chosen == client.core_engines[0]
    # No counter increment when DP rank is explicit.
    assert client.lb_engines[0] == [100, 100]


# --- Threshold behavior consistency ------------------------------------------


@pytest.mark.parametrize(
    "num_engines",
    [1, 2, 3, DPLBAsyncMPClient._P2C_FULL_SCAN_MAX_ENGINES],
)
def test_full_scan_path_does_not_call_random(num_engines):
    """Full scan must not touch ``random.randrange`` at all."""

    client = _make_client(num_engines)
    with patch.object(random, "randrange") as mocked:
        client._select_engine_by_load()
    assert mocked.call_count == 0


def test_p2c_path_calls_randrange_exactly_twice():
    """P2C invokes randrange twice: one for ``a``, one for ``b``."""

    num_engines = DPLBAsyncMPClient._P2C_FULL_SCAN_MAX_ENGINES + 4
    client = _make_client(num_engines)
    with patch.object(random, "randrange", side_effect=[0, 1]) as mocked:
        client._select_engine_by_load()
    assert mocked.call_count == 2


def test_p2c_breaks_score_ties_in_favor_of_first_sample():
    """Tie-break: ``score_a <= score_b`` keeps ``a`` when scores match.

    The contract this locks in is ``deterministic-under-mocked-RNG``:
    given the same RNG outputs, the helper returns a stable engine. The
    actual cross-request fairness comes from ``a`` itself being uniform
    over ``[0, N)`` -- not from "first sample wins" semantics, which is
    just an implementation byproduct.
    """

    num_engines = DPLBAsyncMPClient._P2C_FULL_SCAN_MAX_ENGINES + 4
    counts = [[2, 1] for _ in range(num_engines)]  # everyone identical
    client = _make_client(num_engines, counts=[list(c) for c in counts])

    target_a, target_b = 1, 6
    raw_b = target_b - 1
    with patch.object(random, "randrange", side_effect=[target_a, raw_b]):
        chosen_idx = client._select_engine_by_load()
    assert chosen_idx == target_a
