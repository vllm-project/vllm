# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark: eager vs lazy deletion for PriorityRequestQueue.

Compares the old O(n) removal approach against the new O(1) lazy
deletion.  Run with:
    python tests/v1/core/bench_request_queue.py
"""

import heapq
import time
import uuid

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import PriorityRequestQueue
from vllm.v1.request import Request


def _make_request(priority: int = 0) -> Request:
    return Request(
        request_id=uuid.uuid4().hex,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(ignore_eos=False, max_tokens=10),
        pooling_params=None,
        priority=priority,
    )


def _bench_eager(n: int, k: int) -> float:
    """Simulate old eager removal: list.remove() + heapify."""
    heap: list[Request] = []
    reqs = [_make_request(priority=i) for i in range(n)]
    for r in reqs:
        heapq.heappush(heap, r)

    start = time.perf_counter()
    for r in reqs[:k]:
        heap.remove(r)
        heapq.heapify(heap)
    return time.perf_counter() - start


def _bench_lazy(n: int, k: int) -> float:
    """New lazy deletion via PriorityRequestQueue."""
    q = PriorityRequestQueue()
    reqs = [_make_request(priority=i) for i in range(n)]
    for r in reqs:
        q.add_request(r)

    start = time.perf_counter()
    q.remove_requests(reqs[:k])
    return time.perf_counter() - start


def _bench_mixed(n: int, k: int) -> float:
    """Mixed workload: remove k requests then pop all remaining."""
    q = PriorityRequestQueue()
    reqs = [_make_request(priority=i) for i in range(n)]
    for r in reqs:
        q.add_request(r)

    start = time.perf_counter()
    q.remove_requests(reqs[:k])
    while q:
        q.pop_request()
    return time.perf_counter() - start


def _bench_eager_mixed(n: int, k: int) -> float:
    """Eager mixed workload: remove k requests then pop all remaining."""
    heap: list[Request] = []
    reqs = [_make_request(priority=i) for i in range(n)]
    for r in reqs:
        heapq.heappush(heap, r)

    start = time.perf_counter()
    for r in reqs[:k]:
        heap.remove(r)
        heapq.heapify(heap)
    while heap:
        heapq.heappop(heap)
    return time.perf_counter() - start


if __name__ == "__main__":
    configs = [
        (64, 16),
        (64, 32),
        (256, 64),
        (256, 128),
        (1024, 256),
        (1024, 512),
        (4096, 1024),
    ]

    runs = 100

    print(
        f"{'n':>6} {'k':>6} {'eager(ms)':>12} {'lazy(ms)':>12} "
        f"{'speedup':>10}  |  {'eager_mixed':>14} {'lazy_mixed':>14} "
        f"{'speedup':>10}"
    )
    print("-" * 110)

    for n, k in configs:
        eager_times = [_bench_eager(n, k) * 1000 for _ in range(runs)]
        lazy_times = [_bench_lazy(n, k) * 1000 for _ in range(runs)]
        eager_avg = sum(eager_times) / runs
        lazy_avg = sum(lazy_times) / runs

        eager_mixed_times = [_bench_eager_mixed(n, k) * 1000 for _ in range(runs)]
        lazy_mixed_times = [_bench_mixed(n, k) * 1000 for _ in range(runs)]
        eager_mixed_avg = sum(eager_mixed_times) / runs
        lazy_mixed_avg = sum(lazy_mixed_times) / runs

        speedup = eager_avg / lazy_avg if lazy_avg > 0 else float("inf")
        speedup_mixed = (
            eager_mixed_avg / lazy_mixed_avg if lazy_mixed_avg > 0 else float("inf")
        )

        print(
            f"{n:>6} {k:>6} {eager_avg:>12.3f} {lazy_avg:>12.3f} "
            f"{speedup:>9.1f}x  |  {eager_mixed_avg:>14.3f} "
            f"{lazy_mixed_avg:>14.3f} {speedup_mixed:>9.1f}x"
        )
