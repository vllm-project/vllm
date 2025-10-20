# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from vllm.utils.collections import full_groupby


def _json_load_bytes(path: Path) -> list[dict[str, object]]:
    with path.open("rb") as f:
        return json.load(f)


def _plot_fig(
    fig_path: Path,
    fig_title: str,
    fig_data: list[dict[str, object]],
    curve_by: list[str],
    *,
    var_x: str,
    var_y: str,
    max_x: float | None,
    bin_x: float | None,
    log_y: bool,
    dry_run: bool,
):
    print("[BEGIN FIGURE]")
    print(f"Output file: {fig_path}")

    if dry_run:
        print("[END FIGURE]")
        return

    df = pd.DataFrame.from_records(fig_data)

    if max_x is not None:
        df = df[df[var_x] <= max_x]

    if bin_x is not None:
        df[var_x] = df[var_x] // bin_x * bin_x

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

    ax.set_title(fig_title)

    if log_y:
        ax.set_yscale("log")

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    fig = ax.get_figure()
    assert fig is not None

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)

    print("[END FIGURE]")


def plot(
    output_dir: Path,
    fig_dir: Path,
    fig_by: list[str],
    curve_by: list[str],
    *,
    var_x: str,
    var_y: str,
    max_x: float | None,
    bin_x: float | None,
    log_y: bool,
    dry_run: bool,
):
    all_data = [
        run_data
        for path in output_dir.rglob("**/summary.json")
        for run_data in _json_load_bytes(path)
    ]

    fig_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor() as pool:
        tasks = list[Future[None]]()

        for fig_group, fig_data in full_groupby(
            all_data,
            key=lambda item: tuple((k, str(item[k])) for k in fig_by),
        ):
            fig_group = tuple(fig_group)

            fig_path = fig_dir / (
                "-".join(
                    (
                        "FIGURE-",
                        *(f"{k}={v}" for k, v in fig_group),
                    )
                )
                .replace("/", "_")
                .replace("..", "__")  # Sanitize
                + ".png"
            )
            fig_title = (
                ", ".join(f"{k}={v}" for k, v in fig_group)
                if fig_group
                else "(All data)"
            )

            task = pool.submit(
                _plot_fig,
                fig_path,
                fig_title,
                fig_data,
                curve_by,
                var_x=var_x,
                var_y=var_y,
                max_x=max_x,
                bin_x=bin_x,
                log_y=log_y,
                dry_run=dry_run,
            )
            tasks.append(task)

        for f in tasks:
            f.result()


def main():
    parser = argparse.ArgumentParser(
        description="Plot performance curves from parameter sweep results."
    )
    parser.add_argument(
        "OUTPUT_DIR",
        type=str,
        default="results",
        help="The directory containing the results to plot, "
        "i.e., the `--output-dir` argument to the parameter sweep script.",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=None,
        help="The directory to save the figures. "
        "By default, this is set to `OUTPUT_DIR`.",
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
        "--max-x",
        type=float,
        default=None,
        help="The maximum value to plot for the x-axis.",
    )
    parser.add_argument(
        "--bin-x",
        type=float,
        default=None,
        help="Group together points with x-axis values in the same bin "
        "to reduce noise.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use logarithmic scaling for the y-axis.",
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
        fig_dir=Path(args.fig_dir or args.OUTPUT_DIR),
        fig_by=fig_by,
        curve_by=curve_by,
        var_x=args.var_x,
        var_y=args.var_y,
        max_x=args.max_x,
        bin_x=args.bin_x,
        log_y=args.log_y,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
