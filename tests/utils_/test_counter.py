# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

from vllm.utils.counter import AtomicCounter, Counter


class TestCounter:
    def test_initial_value(self):
        c = Counter()
        assert next(c) == 0

    def test_custom_start(self):
        c = Counter(start=5)
        assert next(c) == 5

    def test_sequential_increment(self):
        c = Counter()
        assert next(c) == 0
        assert next(c) == 1
        assert next(c) == 2
        assert next(c) == 3

    def test_reset(self):
        c = Counter()
        for _ in range(5):
            next(c)
        assert next(c) == 5
        c.reset()
        assert next(c) == 0

    def test_reset_to_zero(self):
        c = Counter(start=10)
        next(c)  # 10
        next(c)  # 11
        c.reset()
        assert next(c) == 0


class TestAtomicCounter:
    def test_initial_value(self):
        ac = AtomicCounter()
        assert ac.value == 0

    def test_custom_initial(self):
        ac = AtomicCounter(initial=42)
        assert ac.value == 42

    def test_inc_default(self):
        ac = AtomicCounter()
        assert ac.inc() == 1
        assert ac.value == 1

    def test_inc_by_value(self):
        ac = AtomicCounter()
        assert ac.inc(5) == 5
        assert ac.inc(3) == 8
        assert ac.value == 8

    def test_dec_default(self):
        ac = AtomicCounter(initial=10)
        assert ac.dec() == 9
        assert ac.value == 9

    def test_dec_by_value(self):
        ac = AtomicCounter(initial=10)
        assert ac.dec(3) == 7
        assert ac.dec(2) == 5
        assert ac.value == 5

    def test_inc_dec_sequence(self):
        ac = AtomicCounter(initial=0)
        assert ac.inc(10) == 10
        assert ac.dec(3) == 7
        assert ac.inc(1) == 8
        assert ac.dec(8) == 0

    def test_thread_safety(self):
        ac = AtomicCounter()
        num_threads = 8
        increments_per_thread = 1000

        def worker():
            for _ in range(increments_per_thread):
                ac.inc()

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ac.value == num_threads * increments_per_thread
