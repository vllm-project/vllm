# SPDX-License-Identifier: Apache-2.0

"""Per-qualname latency statistics for trace replay.

The replay driver times every dispatched call.  Timings feed into this
collector, which computes count + mean + percentiles (p50, p90, p99)
per qualname.

The shape is deliberately simpler than
:class:`lmcache.cli.commands.bench.engine_bench.stats.StatsCollector`
— that one is tailored to OpenAI-style streaming inference (TTFT,
decode speed, etc.), which is not applicable to in-process storage
replay.  Sharing the computation code between the two would push a
small helper deep into the bench module; keeping this collector local
keeps the storage-replay code self-contained and easy to evolve.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import csv
import json
import statistics
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class OpStats:
    """Aggregate timing stats for one qualname.

    Attributes:
        qualname: The qualname these stats cover.
        count: Number of successful replays for this qualname.
        error_count: Number of replays that raised.
        total_s: Total wall time spent replaying this qualname.
        mean_ms: Mean per-call latency in milliseconds.
        p50_ms: 50th-percentile latency in milliseconds.
        p90_ms: 90th-percentile latency in milliseconds.
        p99_ms: 99th-percentile latency in milliseconds.
        min_ms: Minimum observed latency in milliseconds.
        max_ms: Maximum observed latency in milliseconds.
    """

    qualname: str
    count: int
    error_count: int
    total_s: float
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@dataclass
class _Bucket:
    """Internal per-qualname sample bucket."""

    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the *pct*-th percentile from an already-sorted list.

    Uses nearest-rank with no interpolation: for N samples, the
    p-th percentile is the sample at index ``ceil(p/100 * N) - 1``.
    Returns 0.0 for empty input.

    Args:
        sorted_values: Ascending-sorted list of values.
        pct: Percentile in ``[0, 100]``.

    Returns:
        The percentile value, or 0.0 for empty input.
    """
    if not sorted_values:
        return 0.0
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    # bisect_left gives the insertion index; the nearest-rank formula
    # maps p to ceil(p/100 * N) which equals floor((p/100 * N - eps) + 1).
    n = len(sorted_values)
    idx = max(0, min(n - 1, int((pct / 100.0) * n)))
    return sorted_values[idx]


class ReplayStatsCollector:
    """Thread-safe per-qualname latency collector.

    The replay driver is single-threaded at the dispatcher boundary,
    but the underlying StorageManager performs async work on helper
    threads whose timings may eventually feed back here.  A lock keeps
    concurrent ``record()`` calls safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}
        self._wall_start_s: float | None = None
        self._wall_end_s: float | None = None

    def mark_start(self, wall_time_s: float) -> None:
        """Record the wall-clock time when replay began.

        Called once before the first ``record()``; replay-duration
        metrics are derived from start/end marks.

        Args:
            wall_time_s: ``time.time()`` at replay start.
        """
        with self._lock:
            self._wall_start_s = wall_time_s

    def mark_end(self, wall_time_s: float) -> None:
        """Record the wall-clock time when replay finished.

        Args:
            wall_time_s: ``time.time()`` at replay end.
        """
        with self._lock:
            self._wall_end_s = wall_time_s

    def record(self, qualname: str, latency_s: float, failed: bool = False) -> None:
        """Record one replayed call.

        Args:
            qualname: Qualified name of the replayed function.
            latency_s: Elapsed seconds for the call.
            failed: ``True`` if the call raised.  Failed calls still
                contribute to the count but do not add a latency
                sample — the raising path's timing is not comparable
                to successful calls.
        """
        with self._lock:
            bucket = self._buckets.get(qualname)
            if bucket is None:
                bucket = _Bucket()
                self._buckets[qualname] = bucket
            if failed:
                bucket.errors += 1
                return
            # Append is O(1); the one-shot sort in :meth:`summary` is
            # O(N log N), which beats the O(N) shift from keeping the
            # list sorted on insert.  For large traces (>1M records
            # per qualname) the driver should sample or switch to an
            # approximation; for now, exact percentiles suffice.
            bucket.latencies_ms.append(latency_s * 1000.0)

    def total_duration_s(self) -> float:
        """Return replay wall-clock duration in seconds.

        Returns:
            ``mark_end - mark_start`` if both were set, else 0.0.
        """
        with self._lock:
            if self._wall_start_s is None or self._wall_end_s is None:
                return 0.0
            return max(0.0, self._wall_end_s - self._wall_start_s)

    def summary(self) -> dict[str, OpStats]:
        """Return a per-qualname :class:`OpStats` snapshot.

        Returns:
            A dict keyed by qualname.  A qualname with only errors
            still appears, with zero latency stats.
        """
        with self._lock:
            result: dict[str, OpStats] = {}
            for qualname, bucket in self._buckets.items():
                # Sort once per summary call — ``record`` keeps the
                # list unsorted (O(1) append) so the total work is
                # O(N log N) per summary rather than O(N) per insert.
                lats = sorted(bucket.latencies_ms)
                if lats:
                    mean = statistics.fmean(lats)
                    total_s = sum(lats) / 1000.0
                    p50 = _percentile(lats, 50)
                    p90 = _percentile(lats, 90)
                    p99 = _percentile(lats, 99)
                    lo, hi = lats[0], lats[-1]
                else:
                    mean = total_s = p50 = p90 = p99 = lo = hi = 0.0
                result[qualname] = OpStats(
                    qualname=qualname,
                    count=len(lats),
                    error_count=bucket.errors,
                    total_s=total_s,
                    mean_ms=mean,
                    p50_ms=p50,
                    p90_ms=p90,
                    p99_ms=p99,
                    min_ms=lo,
                    max_ms=hi,
                )
            return result

    def export_csv(self, path: str) -> None:
        """Write per-qualname stats to a CSV file.

        Columns: ``qualname, count, errors, mean_ms, p50_ms, p90_ms,
        p99_ms, min_ms, max_ms``.  One row per qualname.

        Args:
            path: File path to write.  Overwritten if it exists.
        """
        summary = self.summary()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "qualname",
                    "count",
                    "errors",
                    "mean_ms",
                    "p50_ms",
                    "p90_ms",
                    "p99_ms",
                    "min_ms",
                    "max_ms",
                ]
            )
            for qn in sorted(summary):
                s = summary[qn]
                w.writerow(
                    [
                        s.qualname,
                        s.count,
                        s.error_count,
                        f"{s.mean_ms:.6f}",
                        f"{s.p50_ms:.6f}",
                        f"{s.p90_ms:.6f}",
                        f"{s.p99_ms:.6f}",
                        f"{s.min_ms:.6f}",
                        f"{s.max_ms:.6f}",
                    ]
                )

    def export_json(self, path: str) -> None:
        """Write per-qualname stats + replay duration to a JSON file.

        Schema::

            {
              "duration_s": <float>,
              "ops": {
                "<qualname>": {
                  "count": int, "errors": int,
                  "mean_ms": float, "p50_ms": float,
                  "p90_ms": float, "p99_ms": float,
                  "min_ms": float, "max_ms": float
                },
                ...
              }
            }

        Args:
            path: File path to write.  Overwritten if it exists.
        """
        summary = self.summary()
        payload = {
            "duration_s": self.total_duration_s(),
            "ops": {
                qn: {
                    "count": s.count,
                    "errors": s.error_count,
                    "mean_ms": s.mean_ms,
                    "p50_ms": s.p50_ms,
                    "p90_ms": s.p90_ms,
                    "p99_ms": s.p99_ms,
                    "min_ms": s.min_ms,
                    "max_ms": s.max_ms,
                }
                for qn, s in summary.items()
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
