# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from pathlib import Path

import pandas as pd
import seaborn as sns

from vllm.utils.collections import full_groupby


def _json_load_bytes(path: Path) -> list[dict[str, object]]:
    with path.open("rb") as f:
        return json.load(f)


def _plot_fig(
    fig_path: Path,
    fig_data: list[dict[str, object]],
    curve_by: list[str],
    *,
    var_x: str,
    var_y: str,
    bin_y: float,
    dry_run: bool,
):
    print("[BEGIN FIGURE]")
    print(f"Output file: {fig_path}")

    if dry_run:
        print("[END FIGURE]")
        return

    df = pd.DataFrame.from_records(fig_data)
    df[var_y] = df[var_y] // bin_y * bin_y

    if len(curve_by) <= 3:
        hue, style, size, *_ = (*curve_by, None, None)
        ax = sns.lineplot(
            df,
            x=var_x,
            y=var_y,
            hue=hue,
            style=style,
            size=size,
            markers=True,
        )
    else:
        df["params"] = df[list(curve_by)].agg("-".join, axis=1)
        ax = sns.lineplot(
            df,
            x=var_x,
            y=var_y,
            hue="params",
            markers=True,
        )

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    fig = ax.get_figure()
    assert fig is not None

    fig.tight_layout()
    fig.savefig(fig_path)

    print("[END FIGURE]")


def plot(
    output_dir: Path,
    fig_by: list[str],
    curve_by: list[str],
    *,
    var_x: str,
    var_y: str,
    bin_y: float,
    dry_run: bool,
):
    all_data = [
        run_data
        for path in output_dir.rglob("**/summary.json")
        for run_data in _json_load_bytes(path)
    ]

    for fig_group, fig_data in full_groupby(
        all_data,
        key=lambda item: tuple((k, str(item[k])) for k in fig_by),
    ):
        fig_path = output_dir / (
            "-".join(
                (
                    "FIGURE",
                    *(f"{k}={v}" for k, v in fig_group),
                )
            )
            .replace("/", "_")
            .replace("..", "__")  # Sanitize
            + ".png"
        )

        _plot_fig(
            fig_path,
            fig_data,
            curve_by,
            var_x=var_x,
            var_y=var_y,
            bin_y=bin_y,
            dry_run=dry_run,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot performance curves from parameter sweep results."
    )
    parser.add_argument(
        "OUTPUT_DIR",
        type=str,
        default="results",
        help="The directory containing the results to plot. "
        "Figures will be saved to the same directory.",
    )
    parser.add_argument(
        "--curve-by",
        type=str,
        required=True,
        help="A comma-separated list of variables, such that a separate curve "
        "is created for each combination of these variables.",
    )
    parser.add_argument(
        "--fig-by",
        type=str,
        default="",
        help="A comma-separated list of variables, such that a separate figure "
        "is created for each combination of these variables.",
    )
    parser.add_argument(
        "--var-x",
        type=str,
        default="request_throughput",
        help="The variable for the x-axis.",
    )
    parser.add_argument(
        "--var-y",
        type=str,
        default="p99_e2el_ms",
        help="The variable for the y-axis",
    )
    parser.add_argument(
        "--bin-y",
        type=float,
        default=1,
        help="Points with y-axis values in the same bin are grouped togther "
        "to reduce noise.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, prints the location of the figures without drawing them.",
    )

    args = parser.parse_args()

    curve_by = [] if not args.curve_by else args.curve_by.split(",")
    fig_by = [] if not args.fig_by else args.fig_by.split(",")

    plot(
        output_dir=Path(args.OUTPUT_DIR),
        fig_by=fig_by,
        curve_by=curve_by,
        var_x=args.var_x,
        var_y=args.var_y,
        bin_y=args.bin_y,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
