# SPDX-License-Identifier: Apache-2.0

# First Party
from benchmarks.musa.bench_inprocess_transfer import (
    BenchmarkResult,
    compare_results,
)


def test_compare_results_requires_native_speedup() -> None:
    """A fast native result passes the Stage2 speedup gate."""
    torch_result = BenchmarkResult(name="torch", seconds_per_iter=1.0)
    native_result = BenchmarkResult(name="native", seconds_per_iter=0.7)

    passed, summary = compare_results(
        torch_result,
        native_result,
        min_speedup=1.2,
    )

    assert passed is True
    assert "speedup=1.429x" in summary


def test_compare_results_fails_when_native_is_slower() -> None:
    """A slower native result fails with the required speedup in the summary."""
    torch_result = BenchmarkResult(name="torch", seconds_per_iter=1.0)
    native_result = BenchmarkResult(name="native", seconds_per_iter=1.1)

    passed, summary = compare_results(
        torch_result,
        native_result,
        min_speedup=1.2,
    )

    assert passed is False
    assert "required>=1.200x" in summary
