# SPDX-License-Identifier: Apache-2.0
"""Render a sweep as CSV, a markdown table, and the Pareto curve.

The curve is the deliverable the assignment asks for: x = tokens/s/user
(interactivity), y = tokens/s/gpu (throughput), one point per client
concurrency. Comparing runs means overlaying two such curves - an optimization
is only real if it moves the whole frontier up and/or right.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

from bench.collect import SweepPoint

CSV_COLUMNS = (
    "label",
    "concurrency",
    "tokens_per_s_per_user",
    "tokens_per_s_per_gpu",
    "output_token_throughput",
    "ttft_ms",
    "ttft_p99_ms",
    "itl_ms",
    "request_latency_ms",
    "request_latency_p99_ms",
    "request_throughput",
    "input_sequence_length",
    "output_sequence_length",
    "request_count",
    "error_request_count",
    "error_rate",
    "benchmark_duration_s",
    "num_gpus",
)

_MD_COLUMNS = (
    ("concurrency", "Conc", "{:.0f}"),
    ("tokens_per_s_per_user", "tok/s/user", "{:.1f}"),
    ("tokens_per_s_per_gpu", "tok/s/gpu", "{:.0f}"),
    ("ttft_ms", "TTFT ms", "{:.1f}"),
    ("ttft_p99_ms", "TTFT p99", "{:.1f}"),
    ("itl_ms", "ITL ms", "{:.2f}"),
    ("request_latency_ms", "e2e ms", "{:.0f}"),
    ("output_sequence_length", "OSL", "{:.1f}"),
    ("error_rate", "err", "{:.2%}"),
)


def write_csv(points: Sequence[SweepPoint], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for point in points:
            row = point.as_row()
            writer.writerow({k: row.get(k) for k in CSV_COLUMNS})
    return path


def render_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A markdown table; the divider is sized from the header."""
    divider = "|" + "|".join("---" for _ in header) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(["| " + " | ".join(header) + " |", divider, *body])


def _fmt(value, spec: str) -> str:
    return "-" if value is None else spec.format(value)


def markdown_table(points: Sequence[SweepPoint]) -> str:
    """Human-readable sweep summary for pasting into the report."""
    if not points:
        return "_no results_"
    rows = [
        [_fmt(point.as_row().get(key), spec) for key, _, spec in _MD_COLUMNS]
        for point in points
    ]
    return render_table([head for _, head, _ in _MD_COLUMNS], rows)


def pareto_frontier(points: Sequence[SweepPoint]) -> list[SweepPoint]:
    """Points not dominated on both axes (higher is better on each).

    A point is dominated when another point is at least as good on
    tokens/s/user *and* tokens/s/gpu, and strictly better on one of them.
    """
    frontier = []
    for candidate in points:
        dominated = any(
            other is not candidate
            and other.tokens_per_s_per_user >= candidate.tokens_per_s_per_user
            and other.tokens_per_s_per_gpu >= candidate.tokens_per_s_per_gpu
            and (
                other.tokens_per_s_per_user > candidate.tokens_per_s_per_user
                or other.tokens_per_s_per_gpu > candidate.tokens_per_s_per_gpu
            )
            for other in points
        )
        if not dominated:
            frontier.append(candidate)
    frontier.sort(key=lambda p: p.tokens_per_s_per_user)
    return frontier


def throughput_at_interactivity(
    points: Sequence[SweepPoint], min_tokens_per_s_per_user: float
) -> SweepPoint | None:
    """Best tokens/s/gpu subject to an interactivity SLO.

    This is the scalar that makes two Pareto curves comparable: "at >= 30
    tok/s/user, how much throughput does a GPU deliver?"
    """
    eligible = [
        p for p in points if p.tokens_per_s_per_user >= min_tokens_per_s_per_user
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda p: p.tokens_per_s_per_gpu)


def interactivity_at_load(
    points: Sequence[SweepPoint], concurrency: int
) -> SweepPoint | None:
    for point in points:
        if point.concurrency == concurrency:
            return point
    return None


def plot_pareto(
    runs: dict[str, Sequence[SweepPoint]],
    path: str | Path,
    title: str = "tokens/s/gpu vs tokens/s/user",
    annotate: bool = True,
) -> Path | None:
    """Save the Pareto plot. Returns None when matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    for label, points in runs.items():
        ordered = sorted(points, key=lambda p: p.concurrency)
        xs = [p.tokens_per_s_per_user for p in ordered]
        ys = [p.tokens_per_s_per_gpu for p in ordered]
        ax.plot(xs, ys, marker="o", label=label)
        if annotate:
            for point, x, y in zip(ordered, xs, ys):
                ax.annotate(
                    f"c={point.concurrency}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=7,
                )
    ax.set_xlabel("tokens/s/user  (interactivity)")
    ax.set_ylabel("tokens/s/gpu  (throughput)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if len(runs) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def integrity_warnings(
    points: Sequence[SweepPoint], min_waves: float = 4.0
) -> list[str]:
    """Measurement defects that invalidate a curve rather than merely add noise.

    Two checks, both learned the hard way:

    * tokens/s/user rising with concurrency is physically impossible -- a
      wider batch cannot decode faster per sequence. It means the points are
      dominated by something other than the engine.
    * Too few waves through the batch leaves the fill/drain transient a large
      fraction of the sample, which is what produced the impossible ordering.
    """
    found = []
    ordered = sorted(points, key=lambda p: p.concurrency)
    for earlier, later in zip(ordered, ordered[1:]):
        if later.tokens_per_s_per_user > earlier.tokens_per_s_per_user:
            found.append(
                f"tokens/s/user rose from {earlier.tokens_per_s_per_user:.1f} at "
                f"concurrency {earlier.concurrency} to "
                f"{later.tokens_per_s_per_user:.1f} at {later.concurrency}: "
                "impossible, so these points do not measure the engine"
            )
    for point in ordered:
        if point.request_count and point.concurrency:
            waves = point.request_count / point.concurrency
            if waves < min_waves:
                found.append(
                    f"concurrency {point.concurrency} saw only {waves:.1f} waves "
                    f"({point.request_count:.0f} requests): raise max_requests"
                )
    return found


def summarize(
    points: Sequence[SweepPoint],
    slo_tokens_per_s_per_user: Iterable[float] = (10, 20, 30, 50),
) -> str:
    """Markdown block with the table plus throughput-at-SLO headlines."""
    chunks = [markdown_table(points), ""]
    if problems := integrity_warnings(points):
        chunks.append("**Measurement integrity warnings**\n")
        chunks += [f"- {problem}" for problem in problems]
        chunks.append("")
    chunks.append("| interactivity SLO | best tok/s/gpu | at concurrency |")
    chunks.append("|---|---|---|")
    for slo in slo_tokens_per_s_per_user:
        best = throughput_at_interactivity(points, slo)
        if best is None:
            chunks.append(f"| >= {slo:g} tok/s/user | not reached | - |")
        else:
            chunks.append(
                f"| >= {slo:g} tok/s/user | {best.tokens_per_s_per_gpu:.0f} "
                f"| {best.concurrency} |"
            )
    return "\n".join(chunks)
