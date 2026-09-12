# SPDX-License-Identifier: Apache-2.0
"""Overlay two or more sweeps: `python -m bench.compare baseline optimized`.

Reports the two scalars that make Pareto curves comparable:
  * throughput at a fixed interactivity SLO (tok/s/gpu at >= N tok/s/user)
  * interactivity at a fixed load (tok/s/user at a given concurrency)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench import report
from bench.collect import SweepPoint, collect_run
from bench.config import DEFAULT_ARTIFACT_ROOT

DEFAULT_SLOS = (10.0, 20.0, 30.0, 50.0)


def _ratio(point: SweepPoint | None, base: SweepPoint | None, suffix: str = "") -> str:
    """Throughput ratio of ``point`` to ``base``, or "-" if not comparable."""
    if base is None or point is None or base.tokens_per_s_per_gpu == 0:
        return "-"
    ratio = point.tokens_per_s_per_gpu / base.tokens_per_s_per_gpu
    return f"{ratio:.3f}x{suffix.format(pct=(ratio - 1) * 100)}"


def load_runs(
    labels: list[str], artifact_root: str | Path, num_gpus: int = 1
) -> dict[str, list[SweepPoint]]:
    runs: dict[str, list[SweepPoint]] = {}
    for label in labels:
        run_dir = Path(artifact_root) / label
        points = collect_run(run_dir, label=label, num_gpus=num_gpus)
        if not points:
            raise FileNotFoundError(f"no aiperf exports under {run_dir}")
        runs[label] = points
    return runs


def delta_table(
    runs: dict[str, list[SweepPoint]], slos: tuple[float, ...] = DEFAULT_SLOS
) -> str:
    """Throughput-at-SLO for each run, with speedup vs the first run."""
    labels = list(runs)
    baseline = labels[0]
    header = [
        "interactivity SLO",
        *(f"{label} tok/s/gpu" for label in labels),
        *(f"{label} vs {baseline}" for label in labels[1:]),
    ]

    rows = []
    for slo in slos:
        best = {
            label: report.throughput_at_interactivity(points, slo)
            for label, points in runs.items()
        }
        cells = [
            f"{best[label].tokens_per_s_per_gpu:.0f} (c={best[label].concurrency})"
            if best[label]
            else "not reached"
            for label in labels
        ]
        deltas = [
            _ratio(best[label], best[baseline], " ({pct:+.1f}%)")
            for label in labels[1:]
        ]
        rows.append([f">= {slo:g}", *cells, *deltas])
    return report.render_table(header, rows)


def per_concurrency_table(runs: dict[str, list[SweepPoint]]) -> str:
    """Both axes plus TTFT for every run at every concurrency they share."""
    labels = list(runs)
    baseline = labels[0]
    by_label = {
        label: {p.concurrency: p for p in points} for label, points in runs.items()
    }
    header = [
        "Conc",
        *(
            f"{label} {column}"
            for label in labels
            for column in ("tok/s/gpu", "tok/s/user", "TTFT ms")
        ),
        *(f"{label} thr vs {baseline}" for label in labels[1:]),
    ]

    rows = []
    for concurrency in sorted({c for table in by_label.values() for c in table}):
        at = {label: by_label[label].get(concurrency) for label in labels}
        cells = []
        for label in labels:
            point = at[label]
            cells += (
                ["-", "-", "-"]
                if point is None
                else [
                    f"{point.tokens_per_s_per_gpu:.0f}",
                    f"{point.tokens_per_s_per_user:.1f}",
                    f"{point.ttft_ms:.0f}" if point.ttft_ms is not None else "-",
                ]
            )
        deltas = [_ratio(at[label], at[baseline]) for label in labels[1:]]
        rows.append([str(concurrency), *cells, *deltas])
    return report.render_table(header, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "labels", nargs="+", help="run labels; the first is the baseline"
    )
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--slo",
        type=float,
        nargs="+",
        default=list(DEFAULT_SLOS),
        help="interactivity SLOs in tokens/s/user",
    )
    parser.add_argument("--out", default=None, help="directory for comparison outputs")
    args = parser.parse_args(argv)

    runs = load_runs(args.labels, args.artifact_root, args.num_gpus)
    out_dir = Path(args.out or Path(args.artifact_root) / "comparisons")
    name = "-vs-".join(args.labels)

    body = "\n\n".join(
        [
            f"# {name}",
            "## Throughput at interactivity SLO",
            delta_table(runs, tuple(args.slo)),
            "## Per-concurrency detail",
            per_concurrency_table(runs),
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.md").write_text(body + "\n")
    flat = [p for points in runs.values() for p in points]
    report.write_csv(flat, out_dir / f"{name}.csv")
    plot = report.plot_pareto(runs, out_dir / f"{name}.png", title=name)

    print(body)
    print(f"\n[out] {out_dir / f'{name}.md'}")
    print(f"[out] {out_dir / f'{name}.csv'}")
    if plot:
        print(f"[out] {plot}")
    else:
        print("[warn] matplotlib missing, no plot", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
