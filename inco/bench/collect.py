# SPDX-License-Identifier: Apache-2.0
"""Turn aiperf JSON exports into the tidy records a Pareto curve needs.

aiperf's ``profile_export_aiperf.json`` is a flat map of metric tag -> stats
block (``{"unit", "avg", "min", "max", "p50", ..., "std", "count"}``), plus
run-level keys like ``input_config``. Distribution metrics carry percentiles;
derived metrics such as ``output_token_throughput`` carry only ``avg``.

Two axes are extracted for every concurrency point:

* tokens/s/user (interactivity) - ``output_token_throughput_per_user``
* tokens/s/gpu (throughput)     - ``output_token_throughput`` / num_gpus
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Metrics pulled into every record, with the statistics kept for each.
DISTRIBUTION_METRICS = (
    "time_to_first_token",
    "inter_token_latency",
    "request_latency",
    "output_token_throughput_per_user",
    "output_sequence_length",
    "input_sequence_length",
)
SCALAR_METRICS = (
    "output_token_throughput",
    "total_token_throughput",
    "request_throughput",
    "request_count",
    "error_request_count",
    "benchmark_duration",
)
KEPT_STATS = ("avg", "p50", "p90", "p99", "min", "max", "std")


class MalformedExport(ValueError):
    pass


def _stat(export: dict[str, Any], metric: str, stat: str = "avg") -> float | None:
    block = export.get(metric)
    if not isinstance(block, dict):
        return None
    value = block.get(stat)
    return float(value) if isinstance(value, (int, float)) else None


@dataclass
class SweepPoint:
    """One concurrency point on the Pareto curve."""

    concurrency: int
    label: str
    tokens_per_s_per_user: float
    tokens_per_s_per_gpu: float
    output_token_throughput: float
    num_gpus: int
    ttft_ms: float | None = None
    ttft_p99_ms: float | None = None
    itl_ms: float | None = None
    request_latency_ms: float | None = None
    request_latency_p99_ms: float | None = None
    request_throughput: float | None = None
    output_sequence_length: float | None = None
    input_sequence_length: float | None = None
    request_count: float | None = None
    error_request_count: float | None = None
    benchmark_duration_s: float | None = None
    metrics: dict[str, float] | None = None

    @property
    def error_rate(self) -> float:
        total = self.request_count or 0.0
        if total <= 0:
            return 0.0
        return (self.error_request_count or 0.0) / total

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("metrics")
        row["error_rate"] = round(self.error_rate, 6)
        return row


def parse_export(
    export: dict[str, Any],
    concurrency: int,
    label: str,
    num_gpus: int = 1,
) -> SweepPoint:
    """Build a :class:`SweepPoint` from a decoded aiperf export."""
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")

    total_tps = _stat(export, "output_token_throughput")
    per_user = _stat(export, "output_token_throughput_per_user")
    if total_tps is None:
        raise MalformedExport(
            "export has no 'output_token_throughput'; the run probably failed "
            "before producing results"
        )
    if per_user is None:
        # Non-streaming runs have no per-token metrics; fall back to the
        # identity tokens/s/user = OSL / request_latency.
        osl = _stat(export, "output_sequence_length")
        latency_ms = _stat(export, "request_latency")
        if osl and latency_ms:
            per_user = osl / (latency_ms / 1000.0)
        else:
            raise MalformedExport(
                "export has neither 'output_token_throughput_per_user' nor the "
                "OSL/latency pair needed to derive it (run with --streaming)"
            )

    metrics: dict[str, float] = {}
    for metric in DISTRIBUTION_METRICS:
        for stat in KEPT_STATS:
            value = _stat(export, metric, stat)
            if value is not None:
                metrics[f"{metric}.{stat}"] = value
    for metric in SCALAR_METRICS:
        value = _stat(export, metric)
        if value is not None:
            metrics[metric] = value

    return SweepPoint(
        concurrency=concurrency,
        label=label,
        tokens_per_s_per_user=per_user,
        tokens_per_s_per_gpu=total_tps / num_gpus,
        output_token_throughput=total_tps,
        num_gpus=num_gpus,
        ttft_ms=_stat(export, "time_to_first_token"),
        ttft_p99_ms=_stat(export, "time_to_first_token", "p99"),
        itl_ms=_stat(export, "inter_token_latency"),
        request_latency_ms=_stat(export, "request_latency"),
        request_latency_p99_ms=_stat(export, "request_latency", "p99"),
        request_throughput=_stat(export, "request_throughput"),
        output_sequence_length=_stat(export, "output_sequence_length"),
        input_sequence_length=_stat(export, "input_sequence_length"),
        request_count=_stat(export, "request_count"),
        error_request_count=_stat(export, "error_request_count"),
        benchmark_duration_s=_stat(export, "benchmark_duration"),
        metrics=metrics,
    )


def load_export(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise MalformedExport(f"{path}: expected a JSON object")
    return payload


def concurrency_from_path(path: Path) -> int | None:
    """Recover the concurrency from a ``concurrency0032`` style directory."""
    for part in reversed(path.parts):
        if part.startswith("concurrency"):
            suffix = part[len("concurrency") :]
            if suffix.isdigit():
                return int(suffix)
    return None


def collect_run(
    run_dir: str | Path,
    label: str | None = None,
    num_gpus: int = 1,
) -> list[SweepPoint]:
    """Collect every concurrency point written under ``run_dir``."""
    run_dir = Path(run_dir)
    exports = sorted(run_dir.rglob("profile_export_aiperf.json"))
    points: list[SweepPoint] = []
    for export_path in exports:
        concurrency = concurrency_from_path(export_path)
        if concurrency is None:
            continue
        export = load_export(export_path)
        points.append(
            parse_export(
                export,
                concurrency=concurrency,
                label=label or run_dir.name,
                num_gpus=num_gpus,
            )
        )
    points.sort(key=lambda p: p.concurrency)
    return _dedupe_by_concurrency(points)


def _dedupe_by_concurrency(points: Iterable[SweepPoint]) -> list[SweepPoint]:
    """Keep the last point per concurrency (re-runs overwrite earlier ones)."""
    by_concurrency: dict[int, SweepPoint] = {}
    for point in points:
        by_concurrency[point.concurrency] = point
    return [by_concurrency[k] for k in sorted(by_concurrency)]
