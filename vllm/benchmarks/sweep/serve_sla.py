# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal, get_args

import numpy as np
from typing_extensions import assert_never

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


SLAVariable = Literal["request_rate", "max_concurrency"]


def _estimate_sla_value(run_data: dict[str, object], sla_variable: SLAVariable):
    request_throughput = float(run_data["request_throughput"])  # type: ignore
    if sla_variable == "request_rate":
        return request_throughput
    if sla_variable == "max_concurrency":
        mean_latency_ms = float(run_data["mean_e2el_ms"])  # type: ignore
        return request_throughput * mean_latency_ms / 1000

    assert_never(sla_variable)


def _estimate_sla_avg(runs: list[dict[str, object]], sla_variable: SLAVariable):
    return sum(_estimate_sla_value(run, sla_variable) for run in runs) / len(runs)


def run_comb_sla(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
    link_vars: list[tuple[str, str]],
    sla_variable: SLAVariable,
    sla_value: int,
) -> list[dict[str, object]] | None:
    bench_comb_sla = bench_comb | {sla_variable: sla_value}

    return run_comb(
        server,
        bench_cmd,
        serve_comb=serve_comb,
        bench_comb=bench_comb_sla,
        base_path=_get_comb_base_path(output_dir, serve_comb, bench_comb_sla),
        num_runs=num_runs,
        dry_run=dry_run,
        link_vars=link_vars,
    )


def explore_sla(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
    sla_variable: SLAVariable,
    sla_iters: int,
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
    link_vars: list[tuple[str, str]],
):
    print("[SLA START]")
    print(f"Serve parameters: {serve_comb.as_text() or '(None)'}")
    print(f"Bench parameters: {bench_comb.as_text() or '(None)'}")
    print(f"Number of SLA iterations: {sla_iters}")

    if sla_iters < 2:
        raise ValueError("`sla_iters` should be at least 2")

    serial_comb_data = run_comb_sla(
        server,
        bench_cmd,
        serve_comb=serve_comb,
        bench_comb=bench_comb,
        output_dir=output_dir,
        num_runs=num_runs,
        dry_run=dry_run,
        link_vars=link_vars,
        sla_variable=sla_variable,
        sla_value=1,
    )
    batch_comb_data = run_comb_sla(
        server,
        bench_cmd,
        serve_comb=serve_comb,
        bench_comb=bench_comb,
        output_dir=output_dir,
        num_runs=num_runs,
        dry_run=dry_run,
        link_vars=link_vars,
        sla_variable=sla_variable,
        sla_value=int(bench_comb.get("num_prompts", 1000)),  # type: ignore
    )

    if serial_comb_data is None or batch_comb_data is None:
        if dry_run:
            print("Omitting intermediate SLA iterations.")
            print("[SLA END]")

        return

    serial_sla_value = math.ceil(_estimate_sla_avg(serial_comb_data, sla_variable))
    print(f"Serial inference: {sla_variable}={serial_sla_value}")

    batch_sla_value = math.floor(_estimate_sla_avg(batch_comb_data, sla_variable))
    print(f"Batch inference: {sla_variable}={batch_sla_value}")

    # Avoid duplicated runs for intermediate values if the range between
    # `serial_sla_value` and `batch_sla_value` is small
    inter_sla_values = np.linspace(serial_sla_value, batch_sla_value, sla_iters)[1:-1]
    inter_sla_values = sorted(set(map(round, inter_sla_values)))

    inter_combs_data: list[dict[str, object]] = []
    for inter_sla_value in inter_sla_values:
        print(f"Exploring: {sla_variable}={inter_sla_value}")
        inter_comb_data = run_comb_sla(
            server,
            bench_cmd,
            serve_comb=serve_comb,
            bench_comb=bench_comb,
            output_dir=output_dir,
            num_runs=num_runs,
            dry_run=dry_run,
            link_vars=link_vars,
            sla_variable=sla_variable,
            sla_value=inter_sla_value,
        )
        if inter_comb_data is not None:
            inter_combs_data.extend(inter_comb_data)

    print("[SLA END]")

    return serial_comb_data + inter_combs_data + batch_comb_data


def run_slas(
    serve_cmd: list[str],
    bench_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    server_ready_timeout: int,
    serve_params: ParameterSweep,
    bench_params: ParameterSweep,
    sla_variable: SLAVariable,
    sla_iters: int,
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
    link_vars: list[tuple[str, str]],
):
    if any(bench_comb.has_param(sla_variable) for bench_comb in bench_params):
        raise ValueError(
            f"You should not override `{sla_variable}` in `bench_params` in SLA mode, "
            "since it is supposed to be determined automatically."
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
            output_dir=output_dir,
            dry_run=dry_run,
        ) as server:
            for bench_comb in bench_params:
                comb_data = explore_sla(
                    server,
                    bench_cmd,
                    serve_comb=serve_comb,
                    bench_comb=bench_comb,
                    sla_variable=sla_variable,
                    sla_iters=sla_iters,
                    output_dir=output_dir,
                    num_runs=num_runs,
                    dry_run=dry_run,
                    link_vars=link_vars,
                )

                if comb_data is not None:
                    all_data.extend(comb_data)

    if dry_run:
        return None

    combined_df = pd.DataFrame.from_records(all_data)
    combined_df.to_csv(output_dir / "summary.csv")

    return combined_df


@dataclass
class SweepServeSLAArgs(SweepServeArgs):
    sla_variable: SLAVariable
    sla_iters: int

    parser_name: ClassVar[str] = "serve_sla"
    parser_help: ClassVar[str] = (
        "Explore the latency-throughput space for determining SLAs."
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # NOTE: Don't use super() as `from_cli_args` calls `cls()`
        base_args = SweepServeArgs.from_cli_args(args)

        return cls(
            **asdict(base_args),
            sla_variable=args.sla_variable,
            sla_iters=args.sla_iters,
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().add_cli_args(parser)

        sla_group = parser.add_argument_group("sla options")
        sla_group.add_argument(
            "--sla-variable",
            type=str,
            choices=get_args(SLAVariable),
            default="request_rate",
            help="The variable to adjust in each iteration.",
        )
        sla_group.add_argument(
            "--sla-iters",
            type=int,
            default=10,
            help="Number of iterations used to explore the latency-throughput space. "
            "This includes the first two iterations used to interpolate the value of "
            "`sla_variable` for remaining iterations.",
        )

        return parser


def run_main(args: SweepServeSLAArgs):
    timestamp = args.resume or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp

    if args.resume and not output_dir.exists():
        raise ValueError(f"Cannot resume from non-existent directory ({output_dir})")

    try:
        return run_slas(
            serve_cmd=args.serve_cmd,
            bench_cmd=args.bench_cmd,
            after_bench_cmd=args.after_bench_cmd,
            show_stdout=args.show_stdout,
            server_ready_timeout=args.server_ready_timeout,
            serve_params=args.serve_params,
            bench_params=args.bench_params,
            sla_variable=args.sla_variable,
            sla_iters=args.sla_iters,
            output_dir=output_dir,
            num_runs=args.num_runs,
            dry_run=args.dry_run,
            link_vars=args.link_vars,
        )
    except BaseException as exc:
        raise RuntimeError(
            f"The script was terminated early. Use `--resume {timestamp}` "
            f"to continue the script from its last checkpoint."
        ) from exc


def main(args: argparse.Namespace):
    run_main(SweepServeSLAArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=SweepServeSLAArgs.parser_help)
    SweepServeSLAArgs.add_cli_args(parser)

    main(parser.parse_args())
