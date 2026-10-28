# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

from vllm.utils.counter import AtomicCounter, Counter


def test_counter_reset_restores_start():
    counter = Counter(start=5)
    assert [next(counter) for _ in range(3)] == [5, 6, 7]

    counter.reset()
    assert next(counter) == 5


def test_atomic_counter_concurrent_inc_dec():
    counter = AtomicCounter(10)
    num_threads, iters = 8, 1000

    def worker():
        for _ in range(iters):
            counter.inc(2)
            counter.dec()

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter.value == 10 + num_threads * iters
