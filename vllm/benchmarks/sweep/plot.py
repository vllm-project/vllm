# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from types import TracebackType

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing_extensions import Self

from vllm.utils.collections import full_groupby


def _json_load_bytes(path: Path) -> list[dict[str, object]]:
    with path.open("rb") as f:
        return json.load(f)


def _get_metric(run_data: dict[str, object], metric_key: str):
    try:
        return run_data[metric_key]
    except KeyError as exc:
        raise ValueError(f"Cannot find metric {metric_key!r} in {run_data=}") from exc


def _get_group(run_data: dict[str, object], group_keys: list[str]):
    return tuple((k, str(_get_metric(run_data, k))) for k in group_keys)


def _get_fig_path(fig_dir: Path, group: tuple[tuple[str, str], ...]):
    return fig_dir / (
        "-".join(
            (
                "FIGURE" + ("-" if group else ""),
                *(f"{k}={v}" for k, v in group),
            )
        )
        .replace("/", "_")
        .replace("..", "__")  # Sanitize
        + ".png"
    )


class DummyExecutor:
    map = map

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        return None


def _plot_fig(
    fig_dir: Path,
    fig_group_data: tuple[tuple[tuple[str, str], ...], list[dict[str, object]]],
    row_by: list[str],
    col_by: list[str],
    curve_by: list[str],
    *,
    var_x: str,
    var_y: str,
    max_x: float | None,
    bin_x: float | None,
    log_y: bool,
    dry_run: bool,
):
    fig_group, fig_data = fig_group_data

    row_groups = full_groupby(
        fig_data,
        key=lambda item: _get_group(item, row_by),
    )
    num_rows = len(row_groups)
    num_cols = max(
        len(full_groupby(row_data, key=lambda item: _get_group(item, col_by)))
        for _, row_data in row_groups
    )

    fig_path = _get_fig_path(fig_dir, fig_group)

    print("[BEGIN FIGURE]")
    print(f"Group: {dict(fig_group)}")
    print(f"Grid: {num_rows} rows x {num_cols} cols")
    print(f"Output file: {fig_path}")

    if dry_run:
        print("[END FIGURE]")
        return

    df = pd.DataFrame.from_records(fig_data)

    if var_x not in df.columns:
        raise ValueError(
            f"Cannot find {var_x=!r} in parameter sweep results. "
            f"Available variables: {df.columns.tolist()}"
        )
    if var_y not in df.columns:
        raise ValueError(
            f"Cannot find {var_y=!r} in parameter sweep results. "
            f"Available variables: {df.columns.tolist()}"
        )

    # TODO: Support <KEY><OP><VALUE> syntax
    # e.g. request_rate<=1024%2 means max of 1024 and bin size of 2
    if max_x is not None:
        df = df[df[var_x] <= max_x]

    if bin_x is not None:
        df[var_x] = df[var_x] // bin_x * bin_x

    df["row_group"] = (
        pd.concat(
            [k + "=" + df[k].astype(str) for k in row_by],
            axis=1,
        ).agg("\n".join, axis=1)
        if row_by
        else "(All)"
    )

    df["col_group"] = (
        pd.concat(
            [k + "=" + df[k].astype(str) for k in col_by],
            axis=1,
        ).agg("\n".join, axis=1)
        if col_by
        else "(All)"
    )

    g = sns.FacetGrid(df, row="row_group", col="col_group")

    if row_by and col_by:
        g.set_titles("{row_name}\n{col_name}")
    elif row_by:
        g.set_titles("{row_name}")
    elif col_by:
        g.set_titles("{col_name}")
    else:
        g.set_titles("")

    if log_y:
        g.set(yscale="log")

    if len(curve_by) <= 3:
        hue, style, size, *_ = (*curve_by, None, None, None)

        g.map_dataframe(
            sns.lineplot,
            x=var_x,
            y=var_y,
            hue=hue,
            style=style,
            size=size,
            markers=True,
        )
    else:
        df["curve_group"] = (
            pd.concat(
                [k + "=" + df[k].astype(str) for k in curve_by],
                axis=1,
            ).agg("\n".join, axis=1)
            if curve_by
            else "(All)"
        )

        g.map_dataframe(
            sns.lineplot,
            x=var_x,
            y=var_y,
            hue="curve_group",
            markers=True,
        )

    g.add_legend()

    g.savefig(fig_path)
    plt.close(g.figure)

    print("[END FIGURE]")


def plot(
    output_dir: Path,
    fig_dir: Path,
    fig_by: list[str],
    row_by: list[str],
    col_by: list[str],
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

    if not all_data:
        raise ValueError(f"Did not find any parameter sweep results under {output_dir}")

    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_groups = full_groupby(
        all_data,
        key=lambda item: _get_group(item, fig_by),
    )

    with DummyExecutor() if len(fig_groups) <= 1 else ProcessPoolExecutor() as executor:
        # Resolve the iterable to ensure that the workers are run
        all(
            executor.map(
                partial(
                    _plot_fig,
                    fig_dir,
                    row_by=row_by,
                    col_by=col_by,
                    curve_by=curve_by,
                    var_x=var_x,
                    var_y=var_y,
                    max_x=max_x,
                    bin_x=bin_x,
                    log_y=log_y,
                    dry_run=dry_run,
                ),
                fig_groups,
            )
        )


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
        default=None,
        help="A comma-separated list of variables, such that a separate curve "
        "is created for each combination of these variables.",
    )
    parser.add_argument(
        "--col-by",
        type=str,
        default="",
        help="A comma-separated list of variables, such that a separate column "
        "is created for each combination of these variables.",
    )
    parser.add_argument(
        "--row-by",
        type=str,
        default="",
        help="A comma-separated list of variables, such that a separate row "
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
    row_by = [] if not args.row_by else args.row_by.split(",")
    col_by = [] if not args.col_by else args.col_by.split(",")
    fig_by = [] if not args.fig_by else args.fig_by.split(",")

    plot(
        output_dir=Path(args.OUTPUT_DIR),
        fig_dir=Path(args.fig_dir or args.OUTPUT_DIR),
        fig_by=fig_by,
        row_by=row_by,
        col_by=col_by,
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
