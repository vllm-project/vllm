# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import cast

import pytest

from vllm.utils.session_prefix import PrefixContext, SessionPrefixStore, token_bytes


@pytest.mark.parametrize(
    ("left", "right", "length", "expected"),
    [
        ([], [], 0, 0),
        ([], [1], 0, 0),
        ([1], [], 0, 0),
        ([1, 2], [1, 2], 0, 0),
        ([1, 2], [1, 2], 2, 2),
        ([1, 2], [9, 2], 2, 0),
        ([1, 2, 3], [1, 9, 3], 3, 1),
        ([1, 2, 3], [1, 2, 9], 2, 2),
        ([1, 2], [1, 2, 3], 2, 2),
        ([1, 2, 3], [1, 2], 2, 2),
    ],
)
def test_token_lcp_matches_bounded_prefix(left, right, length, expected):
    from vllm import _session_prefix

    before_left, before_right = left.copy(), right.copy()
    assert _session_prefix.token_lcp(left, right, length) == expected
    assert left == before_left and right == before_right


@pytest.mark.parametrize("length", [-1, 2, sys.maxsize])
def test_token_lcp_rejects_invalid_bounds(length):
    from vllm import _session_prefix

    for left, right in (([1], [1, 2]), ([1, 2], [1])):
        with pytest.raises(ValueError, match="bounded length"):
            _session_prefix.token_lcp(left, right, length)


@pytest.mark.parametrize("invalid", [None, (1,), "1"])
def test_token_lcp_rejects_non_lists(invalid):
    from vllm import _session_prefix

    for left, right in ((invalid, [1]), ([1], invalid)):
        with pytest.raises(ValueError, match="exact lists"):
            _session_prefix.token_lcp(cast(list[int], left), cast(list[int], right), 1)


@pytest.mark.parametrize(
    ("invalid", "error"),
    [
        (True, ValueError),
        ("1", ValueError),
        (1.0, ValueError),
        (-1, OverflowError),
        (1 << 300, OverflowError),
    ],
)
def test_token_lcp_rejects_invalid_elements(invalid, error):
    from vllm import _session_prefix

    # Reach the invalid value after a valid prefix, in either input list.
    invalid_ids = cast(list[int], [1, invalid])
    for left, right in ((invalid_ids, [1, 2]), ([1, 2], invalid_ids)):
        with pytest.raises(error):
            _session_prefix.token_lcp(left, right, 2)
        # A failed call leaves no error behind for subsequent valid calls.
        assert _session_prefix.token_lcp([1, 2], [1, 2], 2) == 2


def test_token_lcp_checks_exact_types_and_unsigned_long_range():
    from vllm import _session_prefix

    class ListSubclass(list):
        pass

    class IntSubclass(int):
        pass

    for left, right in ((ListSubclass([1]), [1]), ([1], ListSubclass([1]))):
        with pytest.raises(ValueError, match="exact lists"):
            _session_prefix.token_lcp(left, right, 1)
    for left, right in (([IntSubclass(1)], [1]), ([1], [IntSubclass(1)])):
        with pytest.raises(ValueError, match="integer token IDs"):
            _session_prefix.token_lcp(left, right, 1)

    maximum = (1 << (8 * struct.calcsize("L"))) - 1
    assert _session_prefix.token_lcp([maximum], [int(str(maximum))], 1) == 1
    for left, right in (([maximum + 1], [1]), ([1], [maximum + 1])):
        with pytest.raises(OverflowError):
            _session_prefix.token_lcp(left, right, 1)
    with pytest.raises(OverflowError):
        _session_prefix.token_lcp([], [], sys.maxsize + 1)


def test_token_accounting_never_understates_owned_list_and_elements():
    class IntSubclass(int):
        pass

    rng = random.Random(42)
    for ids in (
        [],
        [0, 0, 1, 151669],
        [0, (1 << 30) - 1, 1 << 30, (1 << 32) - 1],
        [-1, 1],
        [1 << 32, 1 << 300],
        [IntSubclass(7), True, 9],
        [rng.randrange(151670) for _ in range(1024)],
    ):
        reference = (
            sys.getsizeof([])
            + len(ids) * struct.calcsize("P")
            + sum(map(sys.getsizeof, ids))
        )
        assert token_bytes(ids) >= reference
        if all(type(i) is int and 0 <= i < (1 << 30) for i in ids):
            assert token_bytes(ids) == reference


def test_max_token_id_rejects_unsupported_elements_without_truncating():
    from vllm import _session_prefix

    assert _session_prefix.max_token_id([]) == 0
    assert _session_prefix.max_token_id([0, (1 << 32) - 1]) == (1 << 32) - 1
    for ids in ([-1], [1 << 32], [1 << 300], [True], ["7"]):
        assert _session_prefix.max_token_id(ids) is None


def test_retired_reader_stays_charged_until_release():
    store = SessionPrefixStore[str](100)
    reservation = store.reserve("a", 40)
    assert reservation is not None
    assert reservation.commit("a", 40)
    with store.borrow("a") as value:
        assert value == "a"
        assert store.reserve("b", 80) is None
        assert store.bytes_used == 40
        with store.borrow("a") as evicted:
            assert evicted is None
    assert store.bytes_used == 0
    reservation = store.reserve("b", 80)
    assert reservation is not None
    assert reservation.commit("b", 80)
    assert store.bytes_used == 80


def test_reset_and_close_reject_inflight_publication():
    store = SessionPrefixStore[str](100)
    old = store.reserve("session", 40)
    assert old is not None
    store.clear()
    assert not old.commit("stale", 40)
    assert store.bytes_used == 0
    old = store.reserve("session", 40)
    assert old is not None
    store.clear(close=True)
    assert not old.commit("closed", 40)
    assert store.reserve("session", 1) is None
    assert store.bytes_used == 0


def test_cancelled_worker_cannot_stage_after_cancellation():
    store = SessionPrefixStore[str](100)
    context = PrefixContext((None, "session"), store.epoch)
    reservation = store.reserve(context.key, 40)
    context.finish(publish=False)
    context.stage(reservation, "late result", 40)
    assert store.bytes_used == 0
    with store.borrow(context.key) as value:
        assert value is None


def test_concurrent_requests_publish_in_completion_order():
    store = SessionPrefixStore[str](100)
    a_ready, b_done = Event(), Event()

    def a():
        reservation = store.reserve("session", 40)
        assert reservation is not None
        a_ready.set()
        assert b_done.wait(5)
        assert reservation.commit("a", 40)

    def b():
        assert a_ready.wait(5)
        reservation = store.reserve("session", 40)
        assert reservation is not None
        assert reservation.commit("b", 40)
        b_done.set()

    with ThreadPoolExecutor(2) as pool:
        first, second = pool.submit(a), pool.submit(b)
        first.result(timeout=10)
        second.result(timeout=10)
    with store.borrow("session") as value:
        assert value == "a"
    assert store.bytes_used == 40
