# SPDX-License-Identifier: Apache-2.0
"""Benchmark runners for L2 adapter ops.

Each round issues ``in_flight`` submits sequentially from the calling
thread, then waits for ``in_flight`` eventfd notifications before
recording the round duration. This matches the real-world usage pattern
where multiple producers submit tasks and the L2 adapter's worker
coroutine processes them.

The benchmark itself is single-threaded on the producer side; the
adapter internally is free to use threads / coroutines / async I/O.
"""

# Future
from __future__ import annotations

# Standard
from typing import Callable
import time

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.memory_management import MemoryObj

# Local
from .data import wait_eventfd
from .result import BenchResult

# Logger callable type: takes a single string and prints / logs it.
LogFn = Callable[[str], None]

# Provider callable signatures used by the runners. They are invoked at
# the start of every round and must return ``in_flight`` lists, one per
# in-flight submit, of length ``num_keys`` each.
KeyProvider = Callable[[int], list[list[ObjectKey]]]
ObjProvider = Callable[[int], list[list[MemoryObj]]]

# TODO: bench passes a placeholder layout_desc; a real layout may be
# required here in the future (e.g. when benchmarking the P2P adapter).
_PLACEHOLDER_LAYOUT_DESC = MemoryLayoutDesc(shapes=[], dtypes=[])


def _bitmap_count(bitmap: Bitmap | None) -> int:
    """Count how many bits are set in *bitmap*. Returns 0 when None."""
    if bitmap is None:
        return 0
    return bitmap.popcount()


def _wait_store_finished(
    adapter, task_ids: list[int], timeout: float
) -> dict[int, L2StoreResult]:
    """Wait for all store tasks to finish.

    Returns the accumulated ``{task_id: L2StoreResult}`` dict.
    ``pop_completed_store_tasks`` consumes the adapter's completion
    dict, so we must accumulate the results here for the caller to
    use. On timeout, returns whatever was harvested so far (possibly
    empty or partial); the caller can detect timeout by comparing
    ``len(returned_dict)`` against ``len(task_ids)``.
    """
    unfinished = len(task_ids)
    efd = adapter.get_store_event_fd()
    completed: dict[int, L2StoreResult] = {}
    while unfinished > 0:
        if not wait_eventfd(efd, timeout=timeout):
            return completed
        batch = adapter.pop_completed_store_tasks()
        completed.update(batch)
        unfinished -= len(batch)
    return completed


def _wait_load_finished(
    adapter, task_ids: list[int], timeout: float
) -> dict[int, Bitmap]:
    """Wait for all load tasks to finish.

    Returns ``{task_id: bitmap}``. ``query_load_result`` consumes the
    per-task result, so we cache the bitmaps here for the caller.
    Already-finished tasks are removed from the pending set so
    subsequent wakeups don't re-query them. On timeout, returns
    whatever was harvested so far; the caller can detect timeout by
    comparing ``len(returned_dict)`` against ``len(task_ids)``.
    """
    pending = set(task_ids)
    efd = adapter.get_load_event_fd()
    results: dict[int, Bitmap] = {}
    while pending:
        if not wait_eventfd(efd, timeout=timeout):
            return results
        for task_id in list(pending):
            bitmap = adapter.query_load_result(task_id)
            if bitmap is not None:
                results[task_id] = bitmap
                pending.remove(task_id)
    return results


def _wait_lookup_finished(
    adapter, task_ids: list[int], timeout: float
) -> dict[int, Bitmap]:
    """Wait for all lookup-and-lock tasks to finish.

    Returns ``{task_id: bitmap}``. ``query_lookup_and_lock_result``
    consumes the per-task result, so we cache the bitmaps here for
    the caller. Already-finished tasks are removed from the pending
    set so subsequent wakeups don't re-query them. On timeout,
    returns whatever was harvested so far; the caller can detect
    timeout by comparing ``len(returned_dict)`` against
    ``len(task_ids)``.
    """
    pending = set(task_ids)
    efd = adapter.get_lookup_and_lock_event_fd()
    results: dict[int, Bitmap] = {}
    while pending:
        if not wait_eventfd(efd, timeout=timeout):
            return results
        for task_id in list(pending):
            bitmap = adapter.query_lookup_and_lock_result(task_id)
            if bitmap is not None:
                results[task_id] = bitmap
                pending.remove(task_id)
    return results


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def bench_store(
    adapter,
    in_flight: int,
    num_keys: int,
    data_size: int,
    rounds: int,
    keys_for_round: KeyProvider,
    objs_for_round: ObjProvider,
    log: LogFn,
) -> BenchResult:
    """Benchmark ``submit_store_task``.

    For each round, ``in_flight`` independent submits are issued; the
    round duration is the wall-clock time from the first submit until
    every submit of that round has completed.
    """
    result = BenchResult(
        operation="Store",
        in_flight=in_flight,
        num_keys=num_keys,
        data_size_bytes=data_size,
    )

    for r in range(rounds):
        keys_batches = keys_for_round(r)
        obj_batches = objs_for_round(r)
        assert len(keys_batches) == in_flight
        assert len(obj_batches) == in_flight

        t0 = time.perf_counter()
        task_ids: list[int] = []
        for i in range(in_flight):
            task_ids.append(adapter.submit_store_task(keys_batches[i], obj_batches[i]))

        completed = _wait_store_finished(adapter, task_ids, 120.0)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        timed_out = len(completed) < len(task_ids)

        success_keys = sum(
            len(keys_batches[i])
            for i, tid in enumerate(task_ids)
            if completed.get(tid, L2StoreResult(False, 0)).is_successful()
        )

        if timed_out:
            log(
                f"  [Store] Round {r + 1}: TIMEOUT "
                f"({len(completed)}/{len(task_ids)} tasks completed, "
                f"success_keys={success_keys}/{in_flight * num_keys})"
            )
            result.round_durations.append(float("inf"))
            result.success_counts.append(success_keys)
            continue

        result.round_durations.append(elapsed)
        result.success_counts.append(success_keys)
        log(
            f"  [Store] Round {r + 1}: {elapsed * 1000:.2f} ms, "
            f"success_keys={success_keys}/{in_flight * num_keys}"
        )

    return result


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def bench_lookup(
    adapter,
    in_flight: int,
    num_keys: int,
    rounds: int,
    keys_for_round: KeyProvider,
    log: LogFn,
    expected_max_hit_rate: float = 0.0,
    expected_hit_count: int = 0,
) -> BenchResult:
    """Benchmark ``submit_lookup_and_lock_task``."""
    result = BenchResult(
        operation="Lookup",
        in_flight=in_flight,
        num_keys=num_keys,
        data_size_bytes=0,  # lookup transfers no payload
        expected_max_hit_rate=expected_max_hit_rate,
        expected_hit_count=expected_hit_count,
    )

    log(
        "bench_lookup uses a placeholder MemoryLayoutDesc; this may need a "
        "real layout for layout-sensitive adapters"
    )

    for r in range(rounds):
        keys_batches = keys_for_round(r)
        assert len(keys_batches) == in_flight

        t0 = time.perf_counter()
        task_ids: list[int] = []
        for i in range(in_flight):
            task_ids.append(
                adapter.submit_lookup_and_lock_task(
                    keys_batches[i], _PLACEHOLDER_LAYOUT_DESC
                )
            )

        results = _wait_lookup_finished(adapter, task_ids, 60.0)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        timed_out = len(results) < len(task_ids)

        total_found = sum(_bitmap_count(results.get(tid)) for tid in task_ids)

        if timed_out:
            log(
                f"  [Lookup] Round {r + 1}: TIMEOUT "
                f"({len(results)}/{len(task_ids)} tasks completed, "
                f"found={total_found}/{in_flight * num_keys})"
            )
            result.round_durations.append(float("inf"))
            result.success_counts.append(total_found)
            continue

        result.round_durations.append(elapsed)
        result.success_counts.append(total_found)
        log(
            f"  [Lookup] Round {r + 1}: {elapsed * 1000:.2f} ms, "
            f"found={total_found}/{in_flight * num_keys}"
        )

    return result


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def bench_load(
    adapter,
    in_flight: int,
    num_keys: int,
    data_size: int,
    rounds: int,
    keys_for_round: KeyProvider,
    objs_for_round: ObjProvider,
    log: LogFn,
) -> BenchResult:
    """Benchmark ``submit_load_task``."""
    result = BenchResult(
        operation="Load",
        in_flight=in_flight,
        num_keys=num_keys,
        data_size_bytes=data_size,
    )

    for r in range(rounds):
        keys_batches = keys_for_round(r)
        obj_batches = objs_for_round(r)
        assert len(keys_batches) == in_flight
        assert len(obj_batches) == in_flight

        # Reset all load buffers before each round to ensure fresh reads.
        for objs in obj_batches:
            for obj in objs:
                obj.raw_data.zero_()

        t0 = time.perf_counter()
        task_ids: list[int] = []
        for i in range(in_flight):
            task_ids.append(adapter.submit_load_task(keys_batches[i], obj_batches[i]))

        results = _wait_load_finished(adapter, task_ids, 120.0)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        timed_out = len(results) < len(task_ids)

        total_loaded = sum(_bitmap_count(results.get(tid)) for tid in task_ids)

        if timed_out:
            log(
                f"  [Load] Round {r + 1}: TIMEOUT "
                f"({len(results)}/{len(task_ids)} tasks completed, "
                f"loaded={total_loaded}/{in_flight * num_keys})"
            )
            result.round_durations.append(float("inf"))
            result.success_counts.append(total_loaded)
            continue

        result.round_durations.append(elapsed)
        result.success_counts.append(total_loaded)
        log(
            f"  [Load] Round {r + 1}: {elapsed * 1000:.2f} ms, "
            f"loaded={total_loaded}/{in_flight * num_keys}"
        )

    return result
