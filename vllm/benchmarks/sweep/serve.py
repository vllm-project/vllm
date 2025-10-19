# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import contextlib
import json
import math
import shlex
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args

import pandas as pd
from typing_extensions import assert_never

from .param_sweep import ParameterSweep, ParameterSweepItem
from .server import ServerProcess
from .sla_sweep import SLASweep, SLASweepItem


@contextlib.contextmanager
def _run_server(
    serve_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    serve_overrides: ParameterSweepItem,
    dry_run: bool,
):
    server_cmd = serve_overrides.apply_to_cmd(serve_cmd)

    print("[BEGIN SERVER]")
    print(f"Server overrides: {serve_overrides}")
    print(f"Server command: {server_cmd}")

    if dry_run:
        yield None
        print("[END SERVER]")
        return

    with ServerProcess(server_cmd, after_bench_cmd, show_stdout=show_stdout) as server:
        yield server

    print("[END SERVER]")


def _run_benchmark(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_overrides: ParameterSweepItem,
    bench_overrides: ParameterSweepItem,
    run_number: int,
    output_path: Path,
    dry_run: bool,
):
    benchmark_cmd = [
        *bench_overrides.apply_to_cmd(bench_cmd),
        "--save-result",
        "--result-dir",
        str(output_path.parent),
        "--result-filename",
        output_path.name,
    ]

    print("[BEGIN BENCHMARK]")
    print(f"Benchmark overrides: {bench_overrides}")
    print(f"Run Number: {run_number}")
    print(f"Benchmark command: {benchmark_cmd}")
    print(f"Output file: {output_path}")

    run_data: dict[str, object]

    if output_path.exists():
        print("Found existing results. Skipping.")

        with output_path.open("rb") as f:
            run_data = json.load(f)
            return run_data

    if server is None:
        assert dry_run
        print("[END BENCHMARK]")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    server.run_subcommand(benchmark_cmd)
    server.after_bench()

    with output_path.open("rb") as f:
        run_data = json.load(f)

    run_data["run_number"] = run_number
    run_data.update(serve_overrides)

    with output_path.open("w") as f:
        json.dump(run_data, f, indent=4)

    print("[END BENCHMARK]")

    return run_data


def _get_comb_base_path(
    output_dir: Path,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
):
    return output_dir / "-".join(
        (
            "SERVE",
            serve_comb.as_text(sep="-"),
            "BENCH",
            bench_comb.as_text(sep="-"),
        )
    ).replace("/", "_").replace("..", "__")  # Sanitize


def _get_comb_run_path(base_path: Path, run_number: int | None):
    if run_number is None:
        return base_path / "summary.json"

    return base_path / f"run={run_number}.json"


def _comb_needs_server(
    serve_comb: ParameterSweepItem,
    bench_combs: ParameterSweep,
    output_dir: Path,
):
    for bench_comb in bench_combs:
        base_path = _get_comb_base_path(output_dir, serve_comb, bench_comb)
        if not _get_comb_run_path(base_path, run_number=None).exists():
            return True

    return False


def _run_comb(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
    base_path: Path,
    num_runs: int,
    dry_run: bool,
):
    comb_data = list[dict[str, object]]()

    for run_number in range(num_runs):
        run_data = _run_benchmark(
            server,
            bench_cmd,
            serve_overrides=serve_comb,
            bench_overrides=bench_comb,
            run_number=run_number,
            output_path=_get_comb_run_path(base_path, run_number),
            dry_run=dry_run,
        )

        if run_data is not None:
            comb_data.append(run_data)

    if dry_run:
        return None

    with _get_comb_run_path(base_path, run_number=None).open("w") as f:
        json.dump(comb_data, f, indent=4)

    return comb_data


def run_combs(
    serve_cmd: list[str],
    bench_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    serve_params: ParameterSweep,
    bench_params: ParameterSweep,
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
):
    all_data = list[dict[str, object]]()
    for serve_comb in serve_params:
        with (
            _run_server(
                serve_cmd,
                after_bench_cmd,
                show_stdout=show_stdout,
                serve_overrides=serve_comb,
                dry_run=dry_run,
            )
            if _comb_needs_server(serve_comb, bench_params, output_dir)
            else contextlib.nullcontext()
        ) as server:
            for bench_comb in bench_params:
                base_path = _get_comb_base_path(output_dir, serve_comb, bench_comb)

                comb_data = _run_comb(
                    server,
                    bench_cmd,
                    serve_comb=serve_comb,
                    bench_comb=bench_comb,
                    base_path=base_path,
                    num_runs=num_runs,
                    dry_run=dry_run,
                )

                if comb_data is not None:
                    all_data.extend(comb_data)

    if dry_run:
        return None

    combined_df = pd.DataFrame.from_records(all_data)
    combined_df.to_csv(output_dir / "summary.csv")

    return combined_df


def _get_sla_base_path(
    output_dir: Path,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
):
    return output_dir / "-".join(
        (
            "SERVE",
            serve_comb.as_text(sep="-"),
            "BENCH",
            bench_comb.as_text(sep="-"),
        )
    ).replace("/", "_").replace("..", "__")  # Sanitize


def _get_sla_iter_path(
    base_path: Path,
    sla_comb: SLASweepItem,
    sla_variable: str,
    sla_value: int | None,
):
    if sla_value is None:
        prefix = sla_comb.as_text(sep="-")
        return base_path / f"SLA-{prefix}.json"

    return base_path / f"{sla_variable}={sla_value}"


def _get_sla_run_path(iter_path: Path, run_number: int | None):
    if run_number is None:
        return iter_path / "summary.json"

    return iter_path / f"run={run_number}.json"


def _sla_needs_server(
    serve_comb: ParameterSweepItem,
    bench_combs: ParameterSweep,
    sla_combs: SLASweep,
    sla_variable: str,
    output_dir: Path,
):
    for bench_comb in bench_combs:
        base_path = _get_sla_base_path(output_dir, serve_comb, bench_comb)
        for sla_comb in sla_combs:
            if not _get_sla_iter_path(
                base_path,
                sla_comb,
                sla_variable,
                sla_value=None,
            ).exists():
                return True

    return False


def _run_sla(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
    iter_path: Path,
    num_runs: int,
    dry_run: bool,
):
    iter_data = list[dict[str, object]]()

    for run_number in range(num_runs):
        run_data = _run_benchmark(
            server,
            bench_cmd,
            serve_overrides=serve_comb,
            bench_overrides=bench_comb,
            run_number=run_number,
            output_path=_get_sla_run_path(iter_path, run_number),
            dry_run=dry_run,
        )

        if run_data is not None:
            iter_data.append(run_data)

    if dry_run:
        return None

    with _get_sla_run_path(iter_path, run_number=None).open("w") as f:
        json.dump(iter_data, f, indent=4)

    return iter_data


SLAVariable = Literal["request_rate", "max_concurrency"]


def _estimate_sla_value(run_data: dict[str, object], sla_variable: SLAVariable):
    request_throughput = float(run_data["request_throughput"])  # type: ignore
    if sla_variable == "request_rate":
        return request_throughput
    if sla_variable == "max_concurrency":
        mean_latency_ms = float(run_data["mean_e2el_ms"])  # type: ignore
        return request_throughput * mean_latency_ms / 1000

    assert_never(sla_variable)


def _estimate_sla_bounds(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
    sla_comb: SLASweepItem,
    base_path: Path,
    num_runs: int,
    dry_run: bool,
    sla_variable: SLAVariable,
    init_value: int,
    max_value: int,
):
    sla_data = list[dict[str, object]]()

    max_passing: int = 0
    min_failing: int = 0

    val: int = init_value
    assert val > 0

    while True:
        print(f"Testing {sla_variable}: {val} req/s")

        iter_data = _run_sla(
            server,
            bench_cmd,
            serve_comb=serve_comb,
            bench_comb=bench_comb | {sla_variable: val},
            iter_path=_get_sla_iter_path(base_path, sla_comb, sla_variable, val),
            num_runs=num_runs,
            dry_run=dry_run,
        )

        assert iter_data is not None
        sla_data.extend(iter_data)

        iter_data_mean = {
            k: sum(float(run_data[k]) for run_data in iter_data) / len(iter_data)  # type: ignore
            for k in sla_comb
        }

        sla_results = [
            criterion.print_and_validate(iter_data_mean, k)
            for k, criterion in sla_comb.items()
        ]

        if all(sla_results):
            print("SLA criteria are met.")
            max_passing = val
            val *= 2
        else:
            print("SLA criteria are not met.")
            min_failing = val
            break

        if val >= max_value:
            break

    return sla_data, (max_passing, min_failing)


def _find_sla_value(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
    sla_comb: SLASweepItem,
    base_path: Path,
    num_runs: int,
    dry_run: bool,
    sla_variable: SLAVariable,
    min_value: int,
    max_value: int,
):
    sla_data = list[dict[str, object]]()

    left: int = min_value
    right: int = max_value

    while True:
        val = (left + right) // 2
        print(f"Testing {sla_variable}: {val} req/s")

        iter_data = _run_sla(
            server,
            bench_cmd,
            serve_comb=serve_comb,
            bench_comb=bench_comb | {sla_variable: val},
            iter_path=_get_sla_iter_path(base_path, sla_comb, sla_variable, val),
            num_runs=num_runs,
            dry_run=dry_run,
        )

        assert iter_data is not None
        sla_data.extend(iter_data)

        iter_data_mean = {
            k: sum(float(run_data[k]) for run_data in iter_data) / len(iter_data)  # type: ignore
            for k in sla_comb
        }

        sla_results = [
            criterion.print_and_validate(iter_data_mean, k)
            for k, criterion in sla_comb.items()
        ]

        if all(sla_results):
            print("SLA criteria are met.")
            left = val
        else:
            print("SLA criteria are not met.")
            right = val

        if right - left <= 1:
            break

    return sla_data, left


def _search_sla(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
    sla_comb: SLASweepItem,
    sla_variable: SLAVariable,
    sla_inf_value: int = 65536,  # The value that represents infinite QPS
    base_path: Path,
    num_runs: int,
    dry_run: bool,
):
    print("[SLA START]")
    print(f"SLA criteria: {sla_comb.as_text()}")

    sla_data_0 = _run_sla(
        server,
        bench_cmd,
        serve_comb=serve_comb,
        bench_comb=bench_comb | {sla_variable: sla_inf_value},
        iter_path=_get_sla_iter_path(base_path, sla_comb, sla_variable, sla_inf_value),
        num_runs=num_runs,
        dry_run=dry_run,
    )
    if sla_data_0 is None:
        assert dry_run
        print("Omitting SLA search.")
        print("[SLA END]")
        return None

    sla_init_value = math.ceil(
        sum(_estimate_sla_value(item, sla_variable) for item in sla_data_0)
        / len(sla_data_0)
    )
    print(f"Initial {sla_variable} to search: {sla_init_value} req/s.")

    sla_data_1, (sla_min, sla_max) = _estimate_sla_bounds(
        server,
        bench_cmd,
        serve_comb=serve_comb,
        bench_comb=bench_comb,
        sla_comb=sla_comb,
        base_path=base_path,
        num_runs=num_runs,
        dry_run=dry_run,
        sla_variable=sla_variable,
        init_value=sla_init_value,
        max_value=sla_inf_value,
    )
    print(f"Range of {sla_variable} to search: [{sla_min}, {sla_max}] req/s.")

    sla_data_2, sla_value = _find_sla_value(
        server,
        bench_cmd,
        serve_comb=serve_comb,
        bench_comb=bench_comb,
        sla_comb=sla_comb,
        base_path=base_path,
        num_runs=num_runs,
        dry_run=dry_run,
        sla_variable=sla_variable,
        min_value=sla_min,
        max_value=sla_max,
    )

    sla_data = sla_data_0 + sla_data_1 + sla_data_2
    print(f"Maximum {sla_variable} for SLA: {sla_value} req/s.")

    with _get_sla_iter_path(
        base_path,
        sla_comb,
        sla_variable,
        sla_value=None,
    ).open("w") as f:
        json.dump(sla_data, f, indent=4)

    print("[SLA END]")

    return sla_data


def run_slas(
    serve_cmd: list[str],
    bench_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    serve_params: ParameterSweep,
    bench_params: ParameterSweep,
    sla_params: SLASweep,
    sla_variable: SLAVariable,
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
):
    if any(bench_comb.has_param(sla_variable) for bench_comb in bench_params):
        raise ValueError(
            f"You should not override `{sla_variable}` in `bench_params` in SLA mode, "
            "since it is supposed to be determined automatically."
        )

    all_data = list[dict[str, object]]()
    for serve_comb in serve_params:
        with (
            _run_server(
                serve_cmd,
                after_bench_cmd,
                show_stdout=show_stdout,
                serve_overrides=serve_comb,
                dry_run=dry_run,
            )
            if _sla_needs_server(
                serve_comb,
                bench_params,
                sla_params,
                sla_variable,
                output_dir,
            )
            else contextlib.nullcontext()
        ) as server:
            for bench_comb in bench_params:
                for sla_comb in sla_params:
                    base_path = _get_sla_base_path(output_dir, serve_comb, bench_comb)

                    comb_data = _search_sla(
                        server,
                        bench_cmd,
                        serve_comb=serve_comb,
                        bench_comb=bench_comb,
                        sla_comb=sla_comb,
                        sla_variable=sla_variable,
                        base_path=base_path,
                        num_runs=num_runs,
                        dry_run=dry_run,
                    )

                    if comb_data is not None:
                        all_data.extend(comb_data)

    if dry_run:
        return None

    combined_df = pd.DataFrame.from_records(all_data)
    combined_df.to_csv(output_dir / "summary.csv")

    return combined_df


def _run_main(
    serve_cmd: list[str],
    bench_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    serve_params: ParameterSweep,
    bench_params: ParameterSweep,
    sla_params: SLASweep,
    sla_variable: SLAVariable,
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
):
    if sla_params:
        return run_slas(
            serve_cmd=serve_cmd,
            bench_cmd=bench_cmd,
            after_bench_cmd=after_bench_cmd,
            show_stdout=show_stdout,
            serve_params=serve_params,
            bench_params=bench_params,
            sla_params=sla_params,
            sla_variable=sla_variable,
            output_dir=output_dir,
            num_runs=num_runs,
            dry_run=dry_run,
        )

    return run_combs(
        serve_cmd=serve_cmd,
        bench_cmd=bench_cmd,
        after_bench_cmd=after_bench_cmd,
        show_stdout=show_stdout,
        serve_params=serve_params,
        bench_params=bench_params,
        output_dir=output_dir,
        num_runs=num_runs,
        dry_run=dry_run,
    )


def run_main(
    serve_cmd: list[str],
    bench_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    serve_params: ParameterSweep,
    bench_params: ParameterSweep,
    sla_params: SLASweep,
    sla_variable: SLAVariable,
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
    resume: str | None,
):
    timestamp = resume or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp

    if resume and not output_dir.exists():
        raise ValueError(f"Cannot resume from non-existent directory ({output_dir})")

    try:
        return _run_main(
            serve_cmd=serve_cmd,
            bench_cmd=bench_cmd,
            after_bench_cmd=after_bench_cmd,
            show_stdout=show_stdout,
            serve_params=serve_params,
            bench_params=bench_params,
            sla_params=sla_params,
            sla_variable=sla_variable,
            output_dir=output_dir,
            num_runs=num_runs,
            dry_run=dry_run,
        )
    except BaseException as exc:
        raise RuntimeError(
            f"The script was terminated early. Use `--resume {timestamp}` "
            f"to continue the script from its last checkpoint."
        ) from exc


def main():
    parser = argparse.ArgumentParser(
        description="Run vLLM server benchmark under multiple settings."
    )
    parser.add_argument(
        "--serve-cmd",
        type=str,
        required=True,
        help="The command used to run the server: `vllm serve ...`",
    )
    parser.add_argument(
        "--bench-cmd",
        type=str,
        required=True,
        help="The command used to run the benchmark: `vllm bench serve ...`",
    )
    parser.add_argument(
        "--after-bench-cmd",
        type=str,
        default=None,
        help="After a benchmark run is complete, invoke this command instead of the "
        "default `ServerWrapper.clear_cache()`.",
    )
    parser.add_argument(
        "--show-stdout",
        action="store_true",
        help="If set, logs the standard output of subcommands. "
        "Useful for debugging but can be quite spammy.",
    )
    parser.add_argument(
        "--serve-params",
        type=str,
        default=None,
        help="Path to JSON file containing a list of parameter combinations "
        "for the `vllm serve` command. "
        "If both `serve_params` and `bench_params` are given, "
        "this script will iterate over their Cartesian product.",
    )
    parser.add_argument(
        "--bench-params",
        type=str,
        default=None,
        help="Path to JSON file containing a list of parameter combinations "
        "for the `vllm bench serve` command. "
        "If both `serve_params` and `bench_params` are given, "
        "this script will iterate over their Cartesian product.",
    )
    parser.add_argument(
        "--sla-params",
        type=str,
        default=None,
        help="Path to JSON file containing a list of SLA constraints to satisfy. "
        'Each constraint is expressed in `{"<KEY>": "<OP><VALUE>"}` format, '
        'e.g.: `{"p99_e2el_ms": "<=500"}` means that '
        "the E2E latency should be less than 500ms 99% of the time. "
        "Setting this option runs this script in SLA mode, which searches for the "
        "maximum `sla_variable` that satisfies the constraints for each combination "
        "of `serve_params`, `bench_params`, and `sla_params`.",
    )
    parser.add_argument(
        "--sla-variable",
        type=str,
        choices=get_args(SLAVariable),
        default="request_rate",
        help="Whether to tune request rate or maximum concurrency to satisfy "
        "the SLA constraints.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="results",
        help="The directory to which results are written.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per parameter combination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, prints the commands to run then exits without running them.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Set this to the name of a directory under `output_dir` (which is a "
        "timestamp) to resume a previous execution of this script, i.e., only run "
        "parameter combinations for which there are still no output files.",
    )

    args = parser.parse_args()

    serve_cmd = shlex.split(args.serve_cmd)
    bench_cmd = shlex.split(args.bench_cmd)
    after_bench_cmd = (
        [] if args.after_bench_cmd is None else shlex.split(args.after_bench_cmd)
    )

    if args.serve_params:
        serve_params = ParameterSweep.read_json(args.serve_params)
    else:
        # i.e.: run serve_cmd without any modification
        serve_params = ParameterSweep.from_records([{}])

    if args.bench_params:
        bench_params = ParameterSweep.read_json(args.bench_params)
    else:
        # i.e.: run bench_cmd without any modification
        bench_params = ParameterSweep.from_records([{}])

    if args.sla_params:
        sla_params = SLASweep.read_json(args.sla_params)
    else:
        sla_params = SLASweep.from_records([])

    num_runs = args.num_runs
    if num_runs < 1:
        raise ValueError("`num_runs` should be at least 1.")

    run_main(
        serve_cmd=serve_cmd,
        bench_cmd=bench_cmd,
        after_bench_cmd=after_bench_cmd,
        show_stdout=args.show_stdout,
        serve_params=serve_params,
        bench_params=bench_params,
        sla_params=sla_params,
        sla_variable=args.sla_variable,
        output_dir=Path(args.output_dir),
        num_runs=num_runs,
        dry_run=args.dry_run,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
