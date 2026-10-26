# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import cast

from vllm.utils.session_prefix import (
    PrefixContext,
    SessionPrefixStore,
    lcp,
    token_bytes,
)


def test_lcp_matches_character_scan():
    def reference(a, b):
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        return i

    rng = random.Random(0)
    alphabet = "ab<>中文🙂é"
    cases = [("", ""), ("", "a"), ("abc", "abc"), ("abc", "abx"), ("abc", "ab")]
    cases.append(("🙂éa", "🙂éb"))
    long_text = "中文🙂é" * 4096
    cases.extend((long_text, long_text[:i] + "x") for i in (0, 1, 8192, len(long_text)))
    for _ in range(500):
        a = "".join(rng.choices(alphabet, k=rng.randrange(40)))
        b = a[: rng.randrange(len(a) + 1)] + "".join(
            rng.choices(alphabet, k=rng.randrange(10))
        )
        cases.append((a, b))
    for a, b in cases:
        assert lcp(a, b) == lcp(b, a) == reference(a, b)


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


def test_python_accounting_preserves_strict_admission_and_fallback():
    class IntSubclass(int):
        def __sizeof__(self):
            return super().__sizeof__() + 1024

    class IndexOnly:
        def __index__(self):
            raise AssertionError("accounting must not coerce token IDs")

    class ListSubclass(list):
        pass

    for ids in (
        [True, 0],
        [IntSubclass(7), 0],
        [-1, 0],
        [1 << 32, 0],
        [1 << 300, 0],
        [1.0, 0],
        ["7", 0],
        [IndexOnly(), 0],
        (1, 1 << 30),
        ListSubclass([1, 1 << 30]),
    ):
        reference = (
            sys.getsizeof([])
            + len(ids) * struct.calcsize("P")
            + sum(map(sys.getsizeof, ids))
        )
        assert token_bytes(cast(list[int], ids)) == reference
    ids = [0, (1 << 30) - 1, 1 << 30, (1 << 32) - 1]
    assert token_bytes(ids) == sys.getsizeof([]) + len(ids) * (
        struct.calcsize("P") + sys.getsizeof((1 << 32) - 1)
    )
