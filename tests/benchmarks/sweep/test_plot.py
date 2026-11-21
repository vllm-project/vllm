# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from pathlib import Path

import pytest

from vllm.benchmarks.sweep.plot import SweepPlotArgs, main
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _plot_args(
    experiment_dir: Path, names: tuple[str, str], *extra_args: str
) -> argparse.Namespace:
    records = [
        {
            "_benchmark_name": name,
            "total_token_throughput": 100 + index,
            "median_ttft_ms": 10 + index,
        }
        for index, name in enumerate(names)
    ]
    (experiment_dir / "summary.json").write_text(json.dumps(records), encoding="utf-8")
    parser = SweepPlotArgs.add_cli_args(FlexibleArgumentParser())
    return parser.parse_args(
        [
            str(experiment_dir),
            "--fig-by",
            "_benchmark_name",
            "--fig-dir",
            "figures",
            "--no-error-bars",
            *extra_args,
        ]
    )


@pytest.mark.parametrize("names", [("case/a", "case_a"), ("case..a", "case__a")])
def test_cli_rejects_colliding_figure_paths(
    tmp_path: Path, names: tuple[str, str]
) -> None:
    args = _plot_args(tmp_path, names, "--dry-run")

    with pytest.raises(ValueError, match="same figure output path"):
        main(args)

    assert not (tmp_path / "figures").exists()


def test_cli_preserves_distinct_figure_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")

    main(_plot_args(tmp_path, ("case_a", "case_b")))

    figures = sorted((tmp_path / "figures").glob("*.png"))
    assert [figure.name for figure in figures] == [
        "FIGURE-_benchmark_name=case_a.png",
        "FIGURE-_benchmark_name=case_b.png",
    ]
    assert all(
        figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for figure in figures
    )


def test_cli_allows_collisions_removed_by_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    args = _plot_args(
        tmp_path,
        ("case/a", "case_a"),
        "--filter-by",
        "_benchmark_name==case_a",
    )

    main(args)

    figures = list((tmp_path / "figures").glob("*.png"))
    assert [figure.name for figure in figures] == ["FIGURE-_benchmark_name=case_a.png"]
    assert figures[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
