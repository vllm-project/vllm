# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import ClassVar

from typing_extensions import Self, override

from vllm.utils.collection_utils import full_groupby
from vllm.utils.import_utils import PlaceholderModule

from .utils import sanitize_filename

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
except ImportError:
    plt = PlaceholderModule("matplotlib").placeholder_attr("pyplot")
    pd = PlaceholderModule("pandas")
    seaborn = PlaceholderModule("seaborn")


@dataclass
class PlotFilterBase(ABC):
    var: str
    target: str

    @classmethod
    def parse_str(cls, s: str):
        for op_key in PLOT_FILTERS:
            if op_key in s:
                key, value = s.split(op_key)
                return PLOT_FILTERS[op_key](
                    key,
                    value.removeprefix(op_key).strip("'").strip('"'),
                )
        else:
            raise ValueError(
                f"Invalid operator for plot filter '{s}'. "
                f"Valid operators are: {sorted(PLOT_FILTERS)}",
            )

    @abstractmethod
    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Applies this filter to a DataFrame."""
        raise NotImplementedError


@dataclass
class PlotEqualTo(PlotFilterBase):
    @override
    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        try:
            target = float(self.target)
        except ValueError:
            target = self.target

        return df[df[self.var] == target]


@dataclass
class PlotLessThan(PlotFilterBase):
    @override
    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        return df[df[self.var] < float(self.target)]


@dataclass
class PlotLessThanOrEqualTo(PlotFilterBase):
    @override
    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        return df[df[self.var] <= float(self.target)]


@dataclass
class PlotGreaterThan(PlotFilterBase):
    @override
    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        return df[df[self.var] > float(self.target)]


@dataclass
class PlotGreaterThanOrEqualTo(PlotFilterBase):
    @override
    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        return df[df[self.var] >= float(self.target)]


# NOTE: The ordering is important! Match longer op_keys first
PLOT_FILTERS: dict[str, type[PlotFilterBase]] = {
    "==": PlotEqualTo,
    "<=": PlotLessThanOrEqualTo,
    ">=": PlotGreaterThanOrEqualTo,
    "<": PlotLessThan,
    ">": PlotGreaterThan,
}


class PlotFilters(list[PlotFilterBase]):
    @classmethod
    def parse_str(cls, s: str):
        if not s:
            return cls()

        return cls(PlotFilterBase.parse_str(e) for e in s.split(","))

    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        for item in self:
            df = item.apply(df)

        return df


@dataclass
class PlotBinner:
    var: str
    bin_size: float

    @classmethod
    def parse_str(cls, s: str):
        for op_key in PLOT_BINNERS:
            if op_key in s:
                key, value = s.split(op_key)
                return PLOT_BINNERS[op_key](key, float(value.removeprefix(op_key)))
        else:
            raise ValueError(
                f"Invalid operator for plot binner '{s}'. "
                f"Valid operators are: {sorted(PLOT_BINNERS)}",
            )

    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Applies this binner to a DataFrame."""
        df = df.copy()
        df[self.var] = df[self.var] // self.bin_size * self.bin_size
        return df


PLOT_BINNERS: dict[str, type[PlotBinner]] = {
    "%": PlotBinner,
}


class PlotBinners(list[PlotBinner]):
    @classmethod
    def parse_str(cls, s: str):
        if not s:
            return cls()

        return cls(PlotBinner.parse_str(e) for e in s.split(","))

    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        for item in self:
            df = item.apply(df)

        return df


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
    parts = list[str]()
    if group:
        parts.extend(("FIGURE-", *(f"{k}={v}" for k, v in group)))
    else:
        parts.append("figure")

    return fig_dir / sanitize_filename("-".join(parts) + ".png")


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
    filter_by: PlotFilters,
    bin_by: PlotBinners,
    scale_x: str | None,
    scale_y: str | None,
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
    for k in row_by:
        if k not in df.columns:
            raise ValueError(
                f"Cannot find row_by={k!r} in parameter sweep results. "
                f"Available variables: {df.columns.tolist()}"
            )
    for k in col_by:
        if k not in df.columns:
            raise ValueError(
                f"Cannot find col_by={k!r} in parameter sweep results. "
                f"Available variables: {df.columns.tolist()}"
            )
    for k in curve_by:
        if k not in df.columns:
            raise ValueError(
                f"Cannot find curve_by={k!r} in parameter sweep results. "
                f"Available variables: {df.columns.tolist()}"
            )

    df = filter_by.apply(df)
    df = bin_by.apply(df)

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

    if scale_x:
        g.set(xscale=scale_x)
    if scale_y:
        g.set(yscale=scale_y)

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

        g.add_legend(title=hue)
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
    filter_by: PlotFilters,
    bin_by: PlotBinners,
    scale_x: str | None,
    scale_y: str | None,
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
                    filter_by=filter_by,
                    bin_by=bin_by,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    dry_run=dry_run,
                ),
                fig_groups,
            )
        )


@dataclass
class SweepPlotArgs:
    output_dir: Path
    fig_dir: Path
    fig_by: list[str]
    row_by: list[str]
    col_by: list[str]
    curve_by: list[str]
    var_x: str
    var_y: str
    filter_by: PlotFilters
    bin_by: PlotBinners
    scale_x: str | None
    scale_y: str | None
    dry_run: bool

    parser_name: ClassVar[str] = "plot"
    parser_help: ClassVar[str] = "Plot performance curves from parameter sweep results."

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        output_dir = Path(args.OUTPUT_DIR)
        if not output_dir.exists():
            raise ValueError(f"No parameter sweep results under {output_dir}")

        curve_by = [] if not args.curve_by else args.curve_by.split(",")
        row_by = [] if not args.row_by else args.row_by.split(",")
        col_by = [] if not args.col_by else args.col_by.split(",")
        fig_by = [] if not args.fig_by else args.fig_by.split(",")

        return cls(
            output_dir=output_dir,
            fig_dir=output_dir / args.fig_dir,
            fig_by=fig_by,
            row_by=row_by,
            col_by=col_by,
            curve_by=curve_by,
            var_x=args.var_x,
            var_y=args.var_y,
            filter_by=PlotFilters.parse_str(args.filter_by),
            bin_by=PlotBinners.parse_str(args.bin_by),
            scale_x=args.scale_x,
            scale_y=args.scale_y,
            dry_run=args.dry_run,
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
            default="",
            help="The directory to save the figures, relative to `OUTPUT_DIR`. "
            "By default, the same directory is used.",
        )
        parser.add_argument(
            "--fig-by",
            type=str,
            default="",
            help="A comma-separated list of variables, such that a separate figure "
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
            "--col-by",
            type=str,
            default="",
            help="A comma-separated list of variables, such that a separate column "
            "is created for each combination of these variables.",
        )
        parser.add_argument(
            "--curve-by",
            type=str,
            default=None,
            help="A comma-separated list of variables, such that a separate curve "
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
            "--filter-by",
            type=str,
            default="",
            help="A comma-separated list of statements indicating values to filter by. "
            "This is useful to remove outliers. "
            "Example: `max_concurrency<1000,max_num_batched_tokens<=4096` means "
            "plot only the points where `max_concurrency` is less than 1000 and "
            "`max_num_batched_tokens` is no greater than 4096.",
        )
        parser.add_argument(
            "--bin-by",
            type=str,
            default="",
            help="A comma-separated list of statements indicating values to bin by. "
            "This is useful to avoid plotting points that are too close together. "
            "Example: `request_throughput%%1` means "
            "use a bin size of 1 for the `request_throughput` variable.",
        )
        parser.add_argument(
            "--scale-x",
            type=str,
            default=None,
            help="The scale to use for the x-axis. "
            "Currently only accepts string values such as 'log' and 'sqrt'. "
            "See also: https://seaborn.pydata.org/generated/seaborn.objects.Plot.scale.html",
        )
        parser.add_argument(
            "--scale-y",
            type=str,
            default=None,
            help="The scale to use for the y-axis. "
            "Currently only accepts string values such as 'log' and 'sqrt'. "
            "See also: https://seaborn.pydata.org/generated/seaborn.objects.Plot.scale.html",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="If set, prints the information about each figure to plot, "
            "then exits without drawing them.",
        )

        return parser


def run_main(args: SweepPlotArgs):
    return plot(
        output_dir=args.output_dir,
        fig_dir=args.fig_dir,
        fig_by=args.fig_by,
        row_by=args.row_by,
        col_by=args.col_by,
        curve_by=args.curve_by,
        var_x=args.var_x,
        var_y=args.var_y,
        filter_by=args.filter_by,
        bin_by=args.bin_by,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        dry_run=args.dry_run,
    )


def main(args: argparse.Namespace):
    run_main(SweepPlotArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=SweepPlotArgs.parser_help)
    SweepPlotArgs.add_cli_args(parser)

    main(parser.parse_args())
