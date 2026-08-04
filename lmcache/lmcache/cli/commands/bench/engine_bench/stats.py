# SPDX-License-Identifier: Apache-2.0
"""Stats collection, aggregation, and export for ``lmcache bench engine``."""

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import threading
import time

# First Party
from lmcache.cli.commands.bench.engine_bench.config import EngineBenchConfig
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class RequestResult:
    """Raw per-request result collected by the request sender."""

    request_id: str
    successful: bool
    ttft: float  # time to first token (seconds)
    request_latency: float  # total request time (seconds)
    num_input_tokens: int  # from server usage report
    num_output_tokens: int  # tokens generated
    decode_speed: float  # output tokens / decode time (tok/s)
    submit_time: float  # absolute timestamp
    first_token_time: float  # absolute timestamp
    finish_time: float  # absolute timestamp
    error: str  # empty string if successful


@dataclass
class AggregatedStats:
    """Snapshot of aggregated statistics (running totals)."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    elapsed_time: float  # seconds since benchmark start

    mean_ttft_ms: float
    mean_decode_speed: float  # tok/s
    mean_request_latency_ms: float

    input_throughput: float  # total input tokens / elapsed time
    output_throughput: float  # total output tokens / elapsed time

    total_input_tokens: int
    total_output_tokens: int


@dataclass
class FinalStats(AggregatedStats):
    """Final statistics with percentiles. Extends AggregatedStats."""

    p50_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    p50_decode_speed: float = 0.0
    p90_decode_speed: float = 0.0
    p99_decode_speed: float = 0.0
    p50_request_latency_ms: float = 0.0
    p90_request_latency_ms: float = 0.0
    p99_request_latency_ms: float = 0.0


class StatsCollector:
    """Thread-safe stats aggregation for benchmark results.

    Receives ``RequestResult`` objects from the request sender,
    maintains running totals, and produces final summaries.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: list[RequestResult] = []
        self._start_time: float = time.monotonic()

        # Running accumulators (updated under lock)
        self._successful: int = 0
        self._failed: int = 0
        self._sum_ttft: float = 0.0
        self._sum_decode_speed: float = 0.0
        self._sum_request_latency: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    def on_request_finished(self, result: RequestResult) -> None:
        """Record a completed request. Thread-safe."""
        with self._lock:
            self._results.append(result)
            if result.successful:
                self._successful += 1
                self._sum_ttft += result.ttft
                self._sum_decode_speed += result.decode_speed
                self._sum_request_latency += result.request_latency
            else:
                self._failed += 1
            self._total_input_tokens += result.num_input_tokens
            self._total_output_tokens += result.num_output_tokens
        logger.debug(
            "Recorded result for %s (successful=%s)",
            result.request_id,
            result.successful,
        )

    def reset(self) -> None:
        """Clear all accumulated results and restart the timer.

        Used between warmup and benchmark phases so warmup stats
        don't pollute benchmark results. Thread-safe.
        """
        with self._lock:
            self._results.clear()
            self._start_time = time.monotonic()
            self._successful = 0
            self._failed = 0
            self._sum_ttft = 0.0
            self._sum_decode_speed = 0.0
            self._sum_request_latency = 0.0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
        logger.debug("Stats collector reset")

    def get_current_stats(self) -> AggregatedStats:
        """Return current aggregated stats snapshot. Thread-safe."""
        with self._lock:
            successful = self._successful
            failed = self._failed
            elapsed = time.monotonic() - self._start_time
            sum_ttft = self._sum_ttft
            sum_decode = self._sum_decode_speed
            sum_latency = self._sum_request_latency
            total_in = self._total_input_tokens
            total_out = self._total_output_tokens

        safe_successful = max(successful, 1)

        return AggregatedStats(
            total_requests=successful + failed,
            successful_requests=successful,
            failed_requests=failed,
            elapsed_time=elapsed,
            mean_ttft_ms=(sum_ttft / safe_successful) * 1000.0,
            mean_decode_speed=sum_decode / safe_successful,
            mean_request_latency_ms=(sum_latency / safe_successful) * 1000.0,
            input_throughput=total_in / max(elapsed, 1e-9),
            output_throughput=total_out / max(elapsed, 1e-9),
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )

    def get_final_stats(self) -> FinalStats:
        """Compute and return final stats with percentiles.

        Should be called once after the benchmark completes.
        """
        with self._lock:
            results = list(self._results)

        successful_results = [r for r in results if r.successful]
        current = self.get_current_stats()

        if not successful_results:
            return FinalStats(
                total_requests=current.total_requests,
                successful_requests=current.successful_requests,
                failed_requests=current.failed_requests,
                elapsed_time=current.elapsed_time,
                mean_ttft_ms=current.mean_ttft_ms,
                mean_decode_speed=current.mean_decode_speed,
                mean_request_latency_ms=current.mean_request_latency_ms,
                input_throughput=current.input_throughput,
                output_throughput=current.output_throughput,
                total_input_tokens=current.total_input_tokens,
                total_output_tokens=current.total_output_tokens,
            )

        ttfts = sorted(r.ttft * 1000.0 for r in successful_results)
        decode_speeds = sorted(r.decode_speed for r in successful_results)
        latencies = sorted(r.request_latency * 1000.0 for r in successful_results)

        return FinalStats(
            total_requests=current.total_requests,
            successful_requests=current.successful_requests,
            failed_requests=current.failed_requests,
            elapsed_time=current.elapsed_time,
            mean_ttft_ms=current.mean_ttft_ms,
            mean_decode_speed=current.mean_decode_speed,
            mean_request_latency_ms=current.mean_request_latency_ms,
            input_throughput=current.input_throughput,
            output_throughput=current.output_throughput,
            total_input_tokens=current.total_input_tokens,
            total_output_tokens=current.total_output_tokens,
            p50_ttft_ms=_percentile(ttfts, 50),
            p90_ttft_ms=_percentile(ttfts, 90),
            p99_ttft_ms=_percentile(ttfts, 99),
            p50_decode_speed=_percentile(decode_speeds, 50),
            p90_decode_speed=_percentile(decode_speeds, 90),
            p99_decode_speed=_percentile(decode_speeds, 99),
            p50_request_latency_ms=_percentile(latencies, 50),
            p90_request_latency_ms=_percentile(latencies, 90),
            p99_request_latency_ms=_percentile(latencies, 99),
        )

    def get_all_results(self) -> list[RequestResult]:
        """Return all raw results for CSV export."""
        with self._lock:
            return list(self._results)

    def export_csv(self, path: str) -> None:
        """Write per-request results to a CSV file."""
        results = self.get_all_results()
        fieldnames = [
            "request_id",
            "successful",
            "ttft",
            "request_latency",
            "num_input_tokens",
            "num_output_tokens",
            "decode_speed",
            "submit_time",
            "first_token_time",
            "finish_time",
            "error",
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
        logger.debug("Exported %d results to CSV: %s", len(results), path)

    def export_json(self, path: str, config: EngineBenchConfig) -> None:
        """Write summary JSON with config and aggregated metrics."""
        final = self.get_final_stats()
        output = {
            "config": asdict(config),
            "results": asdict(final),
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        logger.debug("Exported JSON summary to: %s", path)


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute the p-th percentile using linear interpolation.

    Uses the Tensormesh-Benchmark V1 method:
    ``k = (len(sorted_data) - 1) * p / 100``

    Args:
        sorted_data: Pre-sorted list of values.
        p: Percentile value (0-100).

    Returns:
        Interpolated percentile value. Returns 0.0 for empty data.
    """
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (n - 1) * p / 100.0
    floor_k = int(k)
    ceil_k = min(floor_k + 1, n - 1)
    fraction = k - floor_k
    return sorted_data[floor_k] + fraction * (
        sorted_data[ceil_k] - sorted_data[floor_k]
    )
