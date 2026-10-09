#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate post-benchmark figures from completed recipe sweep results."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlotSpec:
    var_x: str
    var_y: str
    fig_name: str


STAGE_PLOTS: dict[str, tuple[PlotSpec, ...]] = {
    "parallel-layout": (
        PlotSpec("recipe_candidate", "output_throughput", "parallel_throughput"),
        PlotSpec("recipe_candidate", "p99_ttft_ms", "parallel_p99_ttft"),
        PlotSpec("recipe_candidate", "p99_tpot_ms", "parallel_p99_tpot"),
    ),
    "concurrency-tuning": (
        PlotSpec("max_concurrency", "output_throughput", "concurrency_throughput"),
        PlotSpec("max_concurrency", "p99_ttft_ms", "concurrency_p99_ttft"),
        PlotSpec("max_concurrency", "p99_tpot_ms", "concurrency_p99_tpot"),
        PlotSpec("output_throughput", "p99_ttft_ms", "throughput_p99_ttft"),
    ),
    "runtime-tuning": (
        PlotSpec("recipe_candidate", "output_throughput", "scheduler_throughput"),
        PlotSpec("recipe_candidate", "p99_ttft_ms", "scheduler_p99_ttft"),
        PlotSpec("recipe_candidate", "p99_tpot_ms", "scheduler_p99_tpot"),
    ),
}


def _serve_candidate(summary_path: Path, record: dict[str, Any]) -> str:
    directory_name = summary_path.parent.name
    prefix = "SERVE--"
    separator = "-BENCH--"
    if directory_name.startswith(prefix) and separator in directory_name:
        return directory_name[len(prefix) :].split(separator, 1)[0]

    benchmark_name = record.get("_benchmark_name")
    if isinstance(benchmark_name, str) and benchmark_name:
        return benchmark_name
    return "fixed_configuration"


def _load_stage_records(stage_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted(stage_dir.rglob("summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for record in data:
            if not isinstance(record, dict):
                continue
            normalized = dict(record)
            normalized["recipe_candidate"] = _serve_candidate(
                summary_path,
                normalized,
            )
            records.append(normalized)
    return records


def _has_numeric_metric(records: list[dict[str, Any]], key: str) -> bool:
    return any(
        isinstance(record.get(key), (int, float))
        and not isinstance(record.get(key), bool)
        for record in records
    )


def _run_plot(
    normalized_dir: Path,
    figure_dir: Path,
    spec: PlotSpec,
    *,
    dry_run: bool,
) -> None:
    command = [
        "vllm",
        "bench",
        "sweep",
        "plot",
        str(normalized_dir),
        "--fig-dir",
        str(figure_dir.resolve()),
        "--var-x",
        spec.var_x,
        "--var-y",
        spec.var_y,
        "--fig-name",
        spec.fig_name,
    ]
    if dry_run:
        command.append("--dry-run")

    environment = dict(os.environ)
    environment.setdefault("MPLBACKEND", "Agg")
    subprocess.run(command, check=True, env=environment)


def generate_figures(
    results_dir: Path,
    *,
    selected_stages: list[str] | None = None,
    dry_run: bool = False,
) -> list[Path]:
    if shutil.which("vllm") is None:
        raise RuntimeError(
            "The vllm executable was not found. Run this analysis inside the "
            "vLLM benchmark environment."
        )

    stages = selected_stages or list(STAGE_PLOTS)
    discovered: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="vllm-recipe-plot-") as temp_dir:
        temporary_root = Path(temp_dir)
        for stage in stages:
            stage_dir = results_dir / stage
            if not stage_dir.is_dir():
                continue

            records = _load_stage_records(stage_dir)
            if not records:
                continue
            discovered.append(stage_dir)

            normalized_dir = temporary_root / stage
            normalized_dir.mkdir(parents=True, exist_ok=True)
            (normalized_dir / "summary.json").write_text(
                json.dumps(records, indent=2) + "\n",
                encoding="utf-8",
            )

            figure_dir = stage_dir / "figures"
            plots = list(STAGE_PLOTS[stage])
            if stage == "concurrency-tuning" and _has_numeric_metric(
                records, "request_goodput"
            ):
                plots.append(
                    PlotSpec(
                        "max_concurrency",
                        "request_goodput",
                        "concurrency_goodput",
                    )
                )

            for spec in plots:
                _run_plot(
                    normalized_dir,
                    figure_dir,
                    spec,
                    dry_run=dry_run,
                )

    if not discovered:
        requested = ", ".join(stages)
        raise ValueError(
            f"No completed sweep summaries found under {results_dir} "
            f"for stages: {requested}."
        )
    return discovered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot completed recipe sweep results without running a sweep."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results root (default: results beside this script).",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=tuple(STAGE_PLOTS),
        help="Plot only this stage; repeat to select multiple stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview vLLM plot operations without writing figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    results_dir = (
        Path(args.results_dir).resolve()
        if args.results_dir is not None
        else script_dir / "results"
    )
    stages = generate_figures(
        results_dir,
        selected_stages=args.stage,
        dry_run=args.dry_run,
    )
    print()
    if args.dry_run:
        print("Visualization dry run completed for:")
    else:
        print("Wrote post-benchmark figures under:")
    for stage_dir in stages:
        print(f"  {stage_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
