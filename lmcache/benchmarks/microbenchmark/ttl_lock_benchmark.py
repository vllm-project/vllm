#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Performance benchmark for TTLLock.

This script tests the performance overhead of TTLLock under realistic conditions:
- 100K different TTLLock objects
- 10K+ lock/unlock operations per second
- 4-8 threads accessing different locks concurrently

Usage:
    python benchmarks/ttl_lock_benchmark.py [--num-locks NUM] [--threads THREADS]
                                            [--duration SECONDS] [--ops-per-batch OPS]
"""

# Standard
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable
import argparse
import random
import statistics
import threading
import time

# First Party
from lmcache.native_storage_ops import TTLLock


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    num_threads: int
    num_locks: int
    duration_seconds: float
    total_operations: int
    operations_per_second: float
    avg_latency_us: float
    p50_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    min_latency_us: float
    max_latency_us: float

    def __str__(self) -> str:
        return (
            f"\n{'=' * 60}\n"
            f"Benchmark: {self.name}\n"
            f"{'=' * 60}\n"
            f"Configuration:\n"
            f"  Threads:     {self.num_threads}\n"
            f"  Lock objects: {self.num_locks:,}\n"
            f"  Duration:    {self.duration_seconds:.2f}s\n"
            f"\nThroughput:\n"
            f"  Total ops:   {self.total_operations:,}\n"
            f"  Ops/second:  {self.operations_per_second:,.0f}\n"
            f"\nLatency (microseconds):\n"
            f"  Average:     {self.avg_latency_us:.2f} us\n"
            f"  P50:         {self.p50_latency_us:.2f} us\n"
            f"  P95:         {self.p95_latency_us:.2f} us\n"
            f"  P99:         {self.p99_latency_us:.2f} us\n"
            f"  Min:         {self.min_latency_us:.2f} us\n"
            f"  Max:         {self.max_latency_us:.2f} us\n"
        )


class TTLLockBenchmark:
    """Benchmark suite for TTLLock performance testing."""

    def __init__(
        self,
        num_locks: int = 100_000,
        default_threads: int = 4,
        default_duration: float = 5.0,
    ):
        self.num_locks = num_locks
        self.default_threads = default_threads
        self.default_duration = default_duration
        self.locks: list[TTLLock] = []

    def setup(self):
        """Initialize the lock objects."""
        print(f"Creating {self.num_locks:,} TTLLock objects...")
        start = time.perf_counter()
        self.locks = [TTLLock() for _ in range(self.num_locks)]
        elapsed = time.perf_counter() - start
        print(
            f"Created {self.num_locks:,} locks in {elapsed:.3f}s "
            f"({self.num_locks / elapsed:,.0f} locks/sec)"
        )
        print()

    def _run_benchmark(
        self,
        name: str,
        worker_fn: Callable[[int, list[float], threading.Event], None],
        num_threads: int,
        duration: float,
    ) -> BenchmarkResult:
        """Run a benchmark with the given worker function."""
        stop_event = threading.Event()
        latencies_per_thread: list[list[float]] = [[] for _ in range(num_threads)]
        ops_count = [0] * num_threads

        def timed_worker(thread_id: int):
            local_latencies = latencies_per_thread[thread_id]
            worker_fn(thread_id, local_latencies, stop_event)
            ops_count[thread_id] = len(local_latencies)

        # Start workers
        threads = []
        start_time = time.perf_counter()

        for i in range(num_threads):
            t = threading.Thread(target=timed_worker, args=(i,))
            t.start()
            threads.append(t)

        # Let benchmark run for specified duration
        time.sleep(duration)
        stop_event.set()

        # Wait for all threads to finish
        for t in threads:
            t.join()

        end_time = time.perf_counter()
        actual_duration = end_time - start_time

        # Aggregate results
        all_latencies = []
        for lat_list in latencies_per_thread:
            all_latencies.extend(lat_list)

        total_ops = sum(ops_count)

        if not all_latencies:
            return BenchmarkResult(
                name=name,
                num_threads=num_threads,
                num_locks=self.num_locks,
                duration_seconds=actual_duration,
                total_operations=0,
                operations_per_second=0,
                avg_latency_us=0,
                p50_latency_us=0,
                p95_latency_us=0,
                p99_latency_us=0,
                min_latency_us=0,
                max_latency_us=0,
            )

        # Convert to microseconds
        latencies_us = [lat * 1_000_000 for lat in all_latencies]
        latencies_us.sort()

        p50_idx = int(len(latencies_us) * 0.50)
        p95_idx = int(len(latencies_us) * 0.95)
        p99_idx = int(len(latencies_us) * 0.99)

        return BenchmarkResult(
            name=name,
            num_threads=num_threads,
            num_locks=self.num_locks,
            duration_seconds=actual_duration,
            total_operations=total_ops,
            operations_per_second=total_ops / actual_duration,
            avg_latency_us=statistics.mean(latencies_us),
            p50_latency_us=latencies_us[p50_idx],
            p95_latency_us=latencies_us[p95_idx],
            p99_latency_us=latencies_us[min(p99_idx, len(latencies_us) - 1)],
            min_latency_us=latencies_us[0],
            max_latency_us=latencies_us[-1],
        )

    def benchmark_lock_only(
        self, num_threads: int | None = None, duration: float | None = None
    ) -> BenchmarkResult:
        """Benchmark lock() operations only."""
        num_threads = num_threads or self.default_threads
        duration = duration or self.default_duration
        num_locks = self.num_locks

        def worker(thread_id: int, latencies: list[float], stop: threading.Event):
            # Each thread accesses a different range of locks to reduce contention
            rng = random.Random(thread_id)
            while not stop.is_set():
                lock_idx = rng.randint(0, num_locks - 1)
                start = time.perf_counter()
                self.locks[lock_idx].lock()
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)

        return self._run_benchmark("lock() only", worker, num_threads, duration)

    def benchmark_unlock_only(
        self, num_threads: int | None = None, duration: float | None = None
    ) -> BenchmarkResult:
        """Benchmark unlock() operations only."""
        num_threads = num_threads or self.default_threads
        duration = duration or self.default_duration
        num_locks = self.num_locks

        # Pre-lock all locks
        for lock in self.locks:
            for _ in range(100):  # Lock each 100 times to have room for unlocks
                lock.lock()

        def worker(thread_id: int, latencies: list[float], stop: threading.Event):
            rng = random.Random(thread_id)
            while not stop.is_set():
                lock_idx = rng.randint(0, num_locks - 1)
                start = time.perf_counter()
                self.locks[lock_idx].unlock()
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)

        result = self._run_benchmark("unlock() only", worker, num_threads, duration)

        # Reset locks for next benchmark
        for lock in self.locks:
            lock.reset()

        return result

    def benchmark_is_locked_only(
        self, num_threads: int | None = None, duration: float | None = None
    ) -> BenchmarkResult:
        """Benchmark is_locked() operations only (read-only)."""
        num_threads = num_threads or self.default_threads
        duration = duration or self.default_duration
        num_locks = self.num_locks

        # Pre-lock half the locks
        for i, lock in enumerate(self.locks):
            if i % 2 == 0:
                lock.lock()

        def worker(thread_id: int, latencies: list[float], stop: threading.Event):
            rng = random.Random(thread_id)
            while not stop.is_set():
                lock_idx = rng.randint(0, num_locks - 1)
                start = time.perf_counter()
                _ = self.locks[lock_idx].is_locked()
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)

        result = self._run_benchmark("is_locked() only", worker, num_threads, duration)

        # Reset locks for next benchmark
        for lock in self.locks:
            lock.reset()

        return result

    def benchmark_lock_unlock_pair(
        self, num_threads: int | None = None, duration: float | None = None
    ) -> BenchmarkResult:
        """Benchmark lock() + unlock() pairs (typical usage pattern)."""
        num_threads = num_threads or self.default_threads
        duration = duration or self.default_duration
        num_locks = self.num_locks

        def worker(thread_id: int, latencies: list[float], stop: threading.Event):
            rng = random.Random(thread_id)
            while not stop.is_set():
                lock_idx = rng.randint(0, num_locks - 1)
                start = time.perf_counter()
                self.locks[lock_idx].lock()
                self.locks[lock_idx].unlock()
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)

        return self._run_benchmark(
            "lock() + unlock() pair", worker, num_threads, duration
        )

    def benchmark_mixed_operations(
        self, num_threads: int | None = None, duration: float | None = None
    ) -> BenchmarkResult:
        """Benchmark mixed operations (realistic workload)."""
        num_threads = num_threads or self.default_threads
        duration = duration or self.default_duration
        num_locks = self.num_locks

        def worker(thread_id: int, latencies: list[float], stop: threading.Event):
            rng = random.Random(thread_id)
            while not stop.is_set():
                lock_idx = rng.randint(0, num_locks - 1)
                op = rng.randint(0, 9)

                start = time.perf_counter()
                if op < 4:  # 40% lock
                    self.locks[lock_idx].lock()
                elif op < 8:  # 40% unlock
                    self.locks[lock_idx].unlock()
                else:  # 20% is_locked
                    _ = self.locks[lock_idx].is_locked()
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)

        return self._run_benchmark("mixed operations", worker, num_threads, duration)

    def benchmark_high_contention(
        self, num_threads: int | None = None, duration: float | None = None
    ) -> BenchmarkResult:
        """Benchmark with high contention (few locks, many threads)."""
        num_threads = num_threads or self.default_threads
        duration = duration or self.default_duration

        # Use only 100 locks for high contention
        hot_locks = 100

        def worker(thread_id: int, latencies: list[float], stop: threading.Event):
            rng = random.Random(thread_id)
            while not stop.is_set():
                lock_idx = rng.randint(0, hot_locks - 1)
                start = time.perf_counter()
                self.locks[lock_idx].lock()
                self.locks[lock_idx].unlock()
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)

        result = self._run_benchmark(
            f"high contention ({hot_locks} locks)", worker, num_threads, duration
        )

        # Reset the hot locks
        for i in range(hot_locks):
            self.locks[i].reset()

        return result

    def benchmark_batch_operations(
        self,
        num_threads: int | None = None,
        duration: float | None = None,
        batch_size: int = 100,
    ) -> BenchmarkResult:
        """Benchmark batch lock operations (lock multiple, then unlock multiple)."""
        num_threads = num_threads or self.default_threads
        duration = duration or self.default_duration
        num_locks = self.num_locks

        def worker(thread_id: int, latencies: list[float], stop: threading.Event):
            rng = random.Random(thread_id)
            while not stop.is_set():
                # Pick a batch of random lock indices
                indices = [rng.randint(0, num_locks - 1) for _ in range(batch_size)]

                # Lock all
                start = time.perf_counter()
                for idx in indices:
                    self.locks[idx].lock()
                for idx in indices:
                    self.locks[idx].unlock()
                elapsed = time.perf_counter() - start

                # Record per-operation latency
                per_op_latency = elapsed / (batch_size * 2)
                for _ in range(batch_size * 2):
                    latencies.append(per_op_latency)

        return self._run_benchmark(
            f"batch operations (batch={batch_size})", worker, num_threads, duration
        )

    def run_all_benchmarks(
        self, thread_counts: list[int] | None = None, duration: float | None = None
    ) -> list[BenchmarkResult]:
        """Run all benchmarks for specified thread counts."""
        if thread_counts is None:
            thread_counts = [1, 2, 4, 8]
        if duration is None:
            duration = self.default_duration

        results = []

        for num_threads in thread_counts:
            print(f"\n{'#' * 60}")
            print(f"# Running benchmarks with {num_threads} threads")
            print(f"{'#' * 60}")

            # Run each benchmark
            benchmarks = [
                ("lock_only", self.benchmark_lock_only),
                ("unlock_only", self.benchmark_unlock_only),
                ("is_locked_only", self.benchmark_is_locked_only),
                ("lock_unlock_pair", self.benchmark_lock_unlock_pair),
                ("mixed_operations", self.benchmark_mixed_operations),
                ("high_contention", self.benchmark_high_contention),
                ("batch_operations", self.benchmark_batch_operations),
            ]

            for name, benchmark_fn in benchmarks:
                print(f"\nRunning: {name}...")
                result = benchmark_fn(  # type: ignore
                    num_threads=num_threads, duration=duration
                )
                results.append(result)
                print(result)

        return results


def print_summary_table(results: list[BenchmarkResult]):
    """Print a summary comparison table."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)

    # Group by benchmark name
    by_name: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        by_name[r.name].append(r)

    # Print header
    thread_counts = sorted(set(r.num_threads for r in results))
    header = f"{'Benchmark':<30}"
    for t in thread_counts:
        header += f" | {t} threads (ops/s)"
    print(header)
    print("-" * 100)

    # Print each benchmark
    for name in by_name:
        row = f"{name:<30}"
        results_by_threads = {r.num_threads: r for r in by_name[name]}
        for t in thread_counts:
            if t in results_by_threads:
                ops = results_by_threads[t].operations_per_second
                row += f" | {ops:>15,.0f}"
            else:
                row += f" | {'N/A':>15}"
        print(row)

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Performance benchmark for TTLLock",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-locks",
        type=int,
        default=100_000,
        help="Number of TTLLock objects to create",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="1,2,4,8",
        help="Comma-separated list of thread counts to test",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration in seconds for each benchmark",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks (1 second each, fewer thread counts)",
    )

    args = parser.parse_args()

    thread_counts = [int(t.strip()) for t in args.threads.split(",")]
    duration = args.duration

    if args.quick:
        thread_counts = [4, 8]
        duration = 1.0

    print("=" * 60)
    print("TTLLock Performance Benchmark")
    print("=" * 60)
    print(f"Number of locks: {args.num_locks:,}")
    print(f"Thread counts:   {thread_counts}")
    print(f"Duration:        {duration}s per benchmark")
    print()

    benchmark = TTLLockBenchmark(
        num_locks=args.num_locks,
        default_threads=thread_counts[0],
        default_duration=duration,
    )

    benchmark.setup()
    results = benchmark.run_all_benchmarks(
        thread_counts=thread_counts, duration=duration
    )

    print_summary_table(results)

    # Print key findings
    print("\nKEY FINDINGS:")
    print("-" * 60)

    # Find best throughput
    best = max(results, key=lambda r: r.operations_per_second)
    print(
        f"Best throughput: {best.operations_per_second:,.0f} ops/sec "
        f"({best.name}, {best.num_threads} threads)"
    )

    # Find typical workload performance
    mixed_results = [r for r in results if "mixed" in r.name]
    if mixed_results:
        for r in mixed_results:
            print(
                f"Mixed workload ({r.num_threads} threads): "
                f"{r.operations_per_second:,.0f} ops/sec, "
                f"avg latency: {r.avg_latency_us:.2f}us"
            )

    # Check if we can sustain 10K+ ops/sec with 4-8 threads
    target_ops = 10_000
    for t in [4, 8]:
        mixed = [r for r in results if "mixed" in r.name and r.num_threads == t]
        if mixed:
            r = mixed[0]
            if r.operations_per_second >= target_ops:
                print(
                    f"✓ {t} threads can sustain {target_ops:,}+ ops/sec "
                    f"(achieved: {r.operations_per_second:,.0f})"
                )
            else:
                print(
                    f"✗ {t} threads cannot sustain {target_ops:,} ops/sec "
                    f"(achieved: {r.operations_per_second:,.0f})"
                )


if __name__ == "__main__":
    main()
