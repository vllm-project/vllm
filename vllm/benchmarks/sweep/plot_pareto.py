# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar

from vllm.utils.collection_utils import full_groupby
from vllm.utils.import_utils import PlaceholderModule

from .plot import DummyExecutor, _json_load_bytes
from .utils import sanitize_filename

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
except ImportError:
    plt = PlaceholderModule("matplotlib").placeholder_attr("pyplot")
    pd = PlaceholderModule("pandas")
    sns = PlaceholderModule("seaborn")


def _first_present(run_data: dict[str, object], keys: list[str]):
    for key in keys:
        for candidate in {key, key.replace("_", "-"), key.replace("-", "_")}:
            if candidate in run_data:
                return run_data[candidate]
    return None


def _get_numeric(
    run_data: dict[str, object],
    keys: list[str],
    *,
    allow_zero: bool = True,
) -> float | None:
    value = _first_present(run_data, keys)
    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected numeric value for one of {keys}, "
            f"but found {value!r} in {run_data=}"
        ) from exc

    if not allow_zero and numeric == 0:
        return None

    return numeric


def _infer_user_count(
    run_data: dict[str, object],
    user_count_var: str | None,
) -> float | None:
    candidates = [user_count_var] if user_count_var else []
    candidates.extend(["request_rate"])
    user_count = _get_numeric(run_data, candidates, allow_zero=False)
    if user_count is not None:
        return user_count

    # Fallback to the observed peak if configured value is missing.
    return _get_numeric(run_data, ["max_concurrent_requests"], allow_zero=False)


def _infer_gpu_count(
    run_data: dict[str, object],
    gpu_count_var: str | None,
) -> float:
    direct_candidates = [gpu_count_var] if gpu_count_var else []
    direct_gpu_count = _get_numeric(run_data, direct_candidates, allow_zero=False)
    if direct_gpu_count:
        return direct_gpu_count

    tp_size = _get_numeric(run_data, ["tensor_parallel_size", "tp"])
    pp_size = _get_numeric(run_data, ["pipeline_parallel_size", "pp"])
    dp_size = _get_numeric(run_data, ["data_parallel_size", "dp"])
    world_size = 1.0
    if tp_size:
        world_size *= tp_size
    if pp_size:
        world_size *= pp_size
    if dp_size:
        world_size *= dp_size

    return world_size


def _get_throughput(
    run_data: dict[str, object],
    throughput_var: str,
) -> float:
    throughput = _get_numeric(run_data, [throughput_var])
    if throughput is None:
        raise ValueError(
            f"Cannot find throughput metric {throughput_var!r} in run data. "
            f"Available keys: {sorted(run_data)}"
        )

    return throughput


def _prepare_records(
    all_data: list[dict[str, object]],
    *,
    user_count_var: str | None,
    gpu_count_var: str | None,
) -> tuple[list[dict[str, object]], int]:
    prepared = []
    skipped_missing_users = 0

    for record in all_data:
        throughput = _get_throughput(record, "output_throughput")
        user_count = _infer_user_count(record, user_count_var)
        if user_count is None:
            skipped_missing_users += 1
            continue

        gpu_count = _infer_gpu_count(record, gpu_count_var)
        tokens_per_user = throughput / user_count
        tokens_per_gpu = throughput / gpu_count

        prepared.append(
            {
                **record,
                "tokens_per_user": tokens_per_user,
                "tokens_per_gpu": tokens_per_gpu,
                "user_count_estimate": user_count,
                "gpu_count": gpu_count,
            }
        )

    return prepared, skipped_missing_users


def _pareto_frontier(
    df: "pd.DataFrame",
    x_col: str,
    y_col: str,
    *,
    epsilon: float = 1e-9,
) -> "pd.DataFrame":
    sorted_df = df.sort_values([x_col, y_col], ascending=[False, False])
    frontier_indices = []
    best_y = -math.inf

    for idx, row in sorted_df.iterrows():
        y_val = row[y_col]
        if y_val >= best_y - epsilon:
            frontier_indices.append(idx)
            best_y = max(best_y, y_val)

    return df.loc[frontier_indices]


def _get_fig_path(
    fig_dir: Path,
    fig_group: tuple[tuple[str, str], ...],
) -> Path:
    parts = ["PARETO"]
    if fig_group:
        parts.extend(f"{k}={v}" for k, v in fig_group)
    filename = sanitize_filename("-".join(parts) + ".png")
    return fig_dir / filename


def _plot_fig(
    fig_dir: Path,
    fig_group_data: tuple[tuple[tuple[str, str], ...], list[dict[str, object]]],
    label_by: list[str],
    *,
    dry_run: bool,
):
    fig_group, fig_data = fig_group_data
    fig_path = _get_fig_path(fig_dir, fig_group)

    print("[BEGIN FIGURE]")
    print(f"Group: {dict(fig_group)}")
    print(f"Output file: {fig_path}")

    if dry_run:
        print("[END FIGURE]")
        return

    df = pd.DataFrame.from_records(fig_data)
    df = df.dropna(subset=["tokens_per_user", "tokens_per_gpu"])

    if df.empty:
        print("No data points available after filtering; skipping.")
        print("[END FIGURE]")
        return

    frontier = _pareto_frontier(df, "tokens_per_user", "tokens_per_gpu")
    frontier = frontier.sort_values("tokens_per_user")

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="tokens_per_user",
        y="tokens_per_gpu",
        color="0.5",
        alpha=0.6,
        ax=ax,
        label="All runs",
    )
    sns.lineplot(
        data=frontier,
        x="tokens_per_user",
        y="tokens_per_gpu",
        marker="o",
        ax=ax,
        label="Pareto frontier",
    )

    if label_by:
        for _, row in frontier.iterrows():
            label_parts = []
            for key in label_by:
                if key in row:
                    label_parts.append(f"{key}={row[key]}")
            if label_parts:
                ax.text(
                    row["tokens_per_user"],
                    row["tokens_per_gpu"],
                    "\n".join(label_parts),
                    fontsize=8,
                )

    ax.set_xlabel("Tokens/s/user")
    ax.set_ylabel("Tokens/s/GPU")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)

    print(
        f"Plotted {len(df)} points; Pareto frontier size: {len(frontier)}.",
    )
    print("[END FIGURE]")


def plot_pareto(
    output_dir: Path,
    user_count_var: str | None,
    gpu_count_var: str | None,
    label_by: list[str],
    *,
    dry_run: bool,
):
    fig_dir = output_dir / "pareto"
    raw_data = [
        run_data
        for path in output_dir.rglob("**/summary.json")
        for run_data in _json_load_bytes(path)
    ]

    if not raw_data:
        raise ValueError(f"Did not find any parameter sweep results under {output_dir}")

    fig_dir.mkdir(parents=True, exist_ok=True)

    prepared_data, skipped_missing_users = _prepare_records(
        raw_data,
        user_count_var=user_count_var,
        gpu_count_var=gpu_count_var,
    )

    if skipped_missing_users:
        print(
            f"Skipped {skipped_missing_users} runs without a user count "
            "(`max_concurrency` or `max_concurrent_requests`).",
        )

    if not prepared_data:
        raise ValueError(
            "No data points with both throughput and user count available "
            "to plot Pareto frontier.",
        )

    fig_groups = full_groupby(
        prepared_data,
        key=lambda item: tuple(),
    )

    with DummyExecutor() if len(fig_groups) <= 1 else ProcessPoolExecutor() as executor:
        all(
            executor.map(
                partial(
                    _plot_fig,
                    fig_dir,
                    label_by=label_by,
                    dry_run=dry_run,
                ),
                fig_groups,
            )
        )


@dataclass
class SweepPlotParetoArgs:
    output_dir: Path
    user_count_var: str | None
    gpu_count_var: str | None
    label_by: list[str]
    dry_run: bool

    parser_name: ClassVar[str] = "plot_pareto"
    parser_help: ClassVar[str] = (
        "Plot Pareto frontier between tokens/s/user and tokens/s/GPU "
        "from parameter sweep results."
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        output_dir = Path(args.OUTPUT_DIR)
        if not output_dir.exists():
            raise ValueError(f"No parameter sweep results under {output_dir}")

        label_by = [] if not args.label_by else args.label_by.split(",")

        return cls(
            output_dir=output_dir,
            user_count_var=args.user_count_var,
            gpu_count_var=args.gpu_count_var,
            label_by=label_by,
            dry_run=args.dry_run,
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "OUTPUT_DIR",
            type=str,
            default="results",
            help="The directory containing the sweep results to plot.",
        )
        parser.add_argument(
            "--user-count-var",
            type=str,
            default="max_concurrency",
            help="Result key that stores concurrent user count. "
            "Falls back to max_concurrent_requests if missing.",
        )
        parser.add_argument(
            "--gpu-count-var",
            type=str,
            default=None,
            help="Result key that stores GPU count. "
            "If not provided, falls back to num_gpus/gpu_count "
            "or tensor_parallel_size * pipeline_parallel_size.",
        )
        parser.add_argument(
            "--label-by",
            type=str,
            default="max_concurrency,gpu_count",
            help="Comma-separated list of fields to annotate on Pareto frontier "
            "points.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="If set, prints the figures to plot without drawing them.",
        )

        return parser


def run_main(args: SweepPlotParetoArgs):
    return plot_pareto(
        output_dir=args.output_dir,
        user_count_var=args.user_count_var,
        gpu_count_var=args.gpu_count_var,
        label_by=args.label_by,
        dry_run=args.dry_run,
    )


def main(args: argparse.Namespace):
    run_main(SweepPlotParetoArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=SweepPlotParetoArgs.parser_help)
    SweepPlotParetoArgs.add_cli_args(parser)

    main(parser.parse_args())
