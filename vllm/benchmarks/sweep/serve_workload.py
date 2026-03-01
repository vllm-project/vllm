# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Literal, get_args

import numpy as np
from typing_extensions import assert_never

from vllm.benchmarks.datasets import DEFAULT_NUM_PROMPTS
from vllm.utils.import_utils import PlaceholderModule

from .param_sweep import ParameterSweep, ParameterSweepItem
from .serve import (
    SweepServeArgs,
    _get_comb_base_path,
    run_comb,
    server_ctx,
)
from .server import ServerProcess

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")


WorkloadVariable = Literal["request_rate", "max_concurrency"]


def _estimate_workload_value(
    run_data: dict[str, object],
    workload_var: WorkloadVariable,
):
    request_throughput = float(run_data["request_throughput"])  # type: ignore
    if workload_var == "request_rate":
        return request_throughput
    if workload_var == "max_concurrency":
        mean_latency_ms = float(run_data["mean_e2el_ms"])  # type: ignore
        return request_throughput * mean_latency_ms / 1000

    assert_never(workload_var)


def _estimate_workload_avg(
    runs: list[dict[str, object]],
    workload_var: WorkloadVariable,
):
    total = sum(_estimate_workload_value(run, workload_var) for run in runs)
    return total / len(runs)


def run_comb_workload(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
    link_vars: list[tuple[str, str]],
    experiment_dir: Path,
    num_runs: int,
    dry_run: bool,
    workload_var: WorkloadVariable,
    workload_value: int,
) -> list[dict[str, object]] | None:
    bench_comb_workload = bench_comb | {workload_var: workload_value}

    return run_comb(
        server,
        bench_cmd,
        serve_comb=serve_comb,
        bench_comb=bench_comb_workload,
        link_vars=link_vars,
        base_path=_get_comb_base_path(
            experiment_dir,
            serve_comb,
            bench_comb,
            extra_parts=("WL-", f"{workload_var}={workload_value}"),
        ),
        num_runs=num_runs,
        dry_run=dry_run,
    )


def explore_comb_workloads(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
    link_vars: list[tuple[str, str]],
    workload_var: WorkloadVariable,
    workload_iters: int,
    experiment_dir: Path,
    num_runs: int,
    dry_run: bool,
):
    print("[WL START]")
    print(f"Serve parameters: {serve_comb.as_text() or '(None)'}")
    print(f"Bench parameters: {bench_comb.as_text() or '(None)'}")
    print(f"Number of workload iterations: {workload_iters}")

    if workload_iters < 2:
        raise ValueError("`workload_iters` should be at least 2")

    dataset_size = DEFAULT_NUM_PROMPTS
    if "num_prompts" in bench_comb:
        dataset_size = int(bench_comb["num_prompts"])  # type: ignore
    else:
        for i, arg in enumerate(bench_cmd):
            if arg == "--num-prompts" and i + 1 < len(bench_cmd):
                dataset_size = int(bench_cmd[i + 1])
                break
            elif arg.startswith("--num-prompts="):
                dataset_size = int(arg.split("=", 1)[1])
                break

    print(f"Dataset size: {dataset_size}")

    serial_workload_data = run_comb_workload(
        server,
        bench_cmd,
        serve_comb=serve_comb,
        bench_comb=bench_comb | {"max_concurrency": 1},
        link_vars=link_vars,
        experiment_dir=experiment_dir,
        num_runs=num_runs,
        dry_run=dry_run,
        workload_var=workload_var,
        workload_value=1,
    )
    batch_workload_data = run_comb_workload(
        server,
        bench_cmd,
        serve_comb=serve_comb,
        bench_comb=bench_comb | {"max_concurrency": dataset_size},
        link_vars=link_vars,
        experiment_dir=experiment_dir,
        num_runs=num_runs,
        dry_run=dry_run,
        workload_var=workload_var,
        workload_value=dataset_size,
    )

    if serial_workload_data is None or batch_workload_data is None:
        if dry_run:
            print("Omitting intermediate Workload iterations.")
            print("[WL END]")

        return

    serial_workload_value = math.ceil(
        _estimate_workload_avg(serial_workload_data, workload_var)
    )
    print(f"Serial inference: {workload_var}={serial_workload_value}")

    batch_workload_value = math.floor(
        _estimate_workload_avg(batch_workload_data, workload_var)
    )
    print(f"Batch inference: {workload_var}={batch_workload_value}")

    # Avoid duplicated runs for intermediate values if the range between
    # `serial_workload_value` and `batch_workload_value` is small
    inter_workload_values = np.linspace(
        serial_workload_value, batch_workload_value, workload_iters
    )[1:-1]
    inter_workload_values = sorted(set(map(round, inter_workload_values)))

    inter_workloads_data: list[dict[str, object]] = []
    for inter_workload_value in inter_workload_values:
        print(f"Exploring: {workload_var}={inter_workload_value}")
        inter_workload_data = run_comb_workload(
            server,
            bench_cmd,
            serve_comb=serve_comb,
            bench_comb=bench_comb,
            link_vars=link_vars,
            experiment_dir=experiment_dir,
            num_runs=num_runs,
            dry_run=dry_run,
            workload_var=workload_var,
            workload_value=inter_workload_value,
        )
        if inter_workload_data is not None:
            inter_workloads_data.extend(inter_workload_data)

    print("[WL END]")

    return serial_workload_data + inter_workloads_data + batch_workload_data


def explore_combs_workloads(
    serve_cmd: list[str],
    bench_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    server_ready_timeout: int,
    serve_params: ParameterSweep,
    bench_params: ParameterSweep,
    link_vars: list[tuple[str, str]],
    workload_var: WorkloadVariable,
    workload_iters: int,
    experiment_dir: Path,
    num_runs: int,
    dry_run: bool,
):
    if any(bench_comb.has_param(workload_var) for bench_comb in bench_params):
        raise ValueError(
            f"You should not override `{workload_var}` in `bench_params` "
            "since it is supposed to be explored automatically."
        )

    all_data = list[dict[str, object]]()
    for serve_comb in serve_params:
        with server_ctx(
            serve_cmd,
            after_bench_cmd,
            show_stdout=show_stdout,
            server_ready_timeout=server_ready_timeout,
            serve_comb=serve_comb,
            bench_params=bench_params,
            experiment_dir=experiment_dir,
            dry_run=dry_run,
        ) as server:
            for bench_comb in bench_params:
                comb_data = explore_comb_workloads(
                    server,
                    bench_cmd,
                    serve_comb=serve_comb,
                    bench_comb=bench_comb,
                    link_vars=link_vars,
                    workload_var=workload_var,
                    workload_iters=workload_iters,
                    experiment_dir=experiment_dir,
                    num_runs=num_runs,
                    dry_run=dry_run,
                )

                if comb_data is not None:
                    all_data.extend(comb_data)

    if dry_run:
        return None

    combined_df = pd.DataFrame.from_records(all_data)
    combined_df.to_csv(experiment_dir / "summary.csv")

    return combined_df


@dataclass
class SweepServeWorkloadArgs(SweepServeArgs):
    workload_var: WorkloadVariable
    workload_iters: int

    parser_name: ClassVar[str] = "serve_workload"
    parser_help: ClassVar[str] = (
        "Explore the latency-throughput tradeoff for different workload levels."
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # NOTE: Don't use super() as `from_cli_args` calls `cls()`
        base_args = SweepServeArgs.from_cli_args(args)

        return cls(
            **asdict(base_args),
            workload_var=args.workload_var,
            workload_iters=args.workload_iters,
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().add_cli_args(parser)

        workload_group = parser.add_argument_group("workload options")
        workload_group.add_argument(
            "--workload-var",
            type=str,
            choices=get_args(WorkloadVariable),
            default="request_rate",
            help="The variable to adjust in each iteration.",
        )
        workload_group.add_argument(
            "--workload-iters",
            type=int,
            default=10,
            help="Number of workload levels to explore. "
            "This includes the first two iterations used to interpolate the value of "
            "`workload_var` for remaining iterations.",
        )

        return parser


def run_main(args: SweepServeWorkloadArgs):
    experiment_dir = args.resolve_experiment_dir()

    with args.run_ctx(experiment_dir):
        return explore_combs_workloads(
            serve_cmd=args.serve_cmd,
            bench_cmd=args.bench_cmd,
            after_bench_cmd=args.after_bench_cmd,
            show_stdout=args.show_stdout,
            server_ready_timeout=args.server_ready_timeout,
            serve_params=args.serve_params,
            bench_params=args.bench_params,
            link_vars=args.link_vars,
            workload_var=args.workload_var,
            workload_iters=args.workload_iters,
            experiment_dir=experiment_dir,
            num_runs=args.num_runs,
            dry_run=args.dry_run,
        )


def main(args: argparse.Namespace):
    run_main(SweepServeWorkloadArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=SweepServeWorkloadArgs.parser_help)
    SweepServeWorkloadArgs.add_cli_args(parser)

    main(parser.parse_args())
