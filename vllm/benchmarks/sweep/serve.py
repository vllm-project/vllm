# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import contextlib
import json
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from vllm.utils.import_utils import PlaceholderModule

from .param_sweep import ParameterSweep, ParameterSweepItem
from .server import ServerProcess
from .utils import sanitize_filename

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")


@contextlib.contextmanager
def run_server(
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


def _update_run_data(
    run_data: dict[str, object],
    serve_overrides: ParameterSweepItem,
    bench_overrides: ParameterSweepItem,
    run_number: int,
):
    run_data["run_number"] = run_number
    run_data.update(serve_overrides)
    run_data.update(bench_overrides)

    return run_data


def run_benchmark(
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
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
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
            return _update_run_data(
                run_data,
                serve_overrides,
                bench_overrides,
                run_number,
            )

    if server is None:
        if not dry_run:
            raise ValueError(f"Cannot find results at {output_path}")

        print("[END BENCHMARK]")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    server.run_subcommand(benchmark_cmd)
    server.after_bench()

    with output_path.open("rb") as f:
        run_data = json.load(f)

    run_data = _update_run_data(
        run_data,
        serve_overrides,
        bench_overrides,
        run_number,
    )

    with output_path.open("w") as f:
        json.dump(run_data, f, indent=4)

    print("[END BENCHMARK]")

    return run_data


def _get_comb_base_path(
    output_dir: Path,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
):
    parts = list[str]()
    if serve_comb:
        parts.extend(("SERVE-", serve_comb.as_text(sep="-")))
    if bench_comb:
        parts.extend(("BENCH-", bench_comb.as_text(sep="-")))

    return output_dir / sanitize_filename("-".join(parts))


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


def run_comb(
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
        run_data = run_benchmark(
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
            run_server(
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

                comb_data = run_comb(
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


@dataclass
class SweepServeArgs:
    serve_cmd: list[str]
    bench_cmd: list[str]
    after_bench_cmd: list[str]
    show_stdout: bool
    serve_params: ParameterSweep
    bench_params: ParameterSweep
    output_dir: Path
    num_runs: int
    dry_run: bool
    resume: str | None

    parser_name: ClassVar[str] = "serve"
    parser_help: ClassVar[str] = "Run vLLM server benchmark under multiple settings."

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
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

        num_runs = args.num_runs
        if num_runs < 1:
            raise ValueError("`num_runs` should be at least 1.")

        return cls(
            serve_cmd=serve_cmd,
            bench_cmd=bench_cmd,
            after_bench_cmd=after_bench_cmd,
            show_stdout=args.show_stdout,
            serve_params=serve_params,
            bench_params=bench_params,
            output_dir=Path(args.output_dir),
            num_runs=num_runs,
            dry_run=args.dry_run,
            resume=args.resume,
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
            help="After a benchmark run is complete, invoke this command instead of "
            "the default `ServerWrapper.clear_cache()`.",
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
            help="If set, prints the commands to run, "
            "then exits without executing them.",
        )
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="Set this to the name of a directory under `output_dir` (which is a "
            "timestamp) to resume a previous execution of this script, i.e., only run "
            "parameter combinations for which there are still no output files.",
        )

        return parser


def run_main(args: SweepServeArgs):
    timestamp = args.resume or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp

    if args.resume and not output_dir.exists():
        raise ValueError(f"Cannot resume from non-existent directory ({output_dir})")

    try:
        return run_combs(
            serve_cmd=args.serve_cmd,
            bench_cmd=args.bench_cmd,
            after_bench_cmd=args.after_bench_cmd,
            show_stdout=args.show_stdout,
            serve_params=args.serve_params,
            bench_params=args.bench_params,
            output_dir=output_dir,
            num_runs=args.num_runs,
            dry_run=args.dry_run,
        )
    except BaseException as exc:
        raise RuntimeError(
            f"The script was terminated early. Use `--resume {timestamp}` "
            f"to continue the script from its last checkpoint."
        ) from exc


def main(args: argparse.Namespace):
    run_main(SweepServeArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=SweepServeArgs.parser_help)
    SweepServeArgs.add_cli_args(parser)

    main(parser.parse_args())
