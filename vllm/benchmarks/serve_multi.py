# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import contextlib
import json
import os
import shlex
import signal
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

_BAD_PARAMS_TYPE_MSG = (
    "The parameters to vary should be expressed as a JSON list of dictionaries."
)


def _validate_combs(params: list[dict[str, object]]):
    if not isinstance(params, list):
        raise TypeError(f"{_BAD_PARAMS_TYPE_MSG} Found JSON type {type(params)}")

    for comb in params:
        if not isinstance(comb, dict):
            raise TypeError(f"{_BAD_PARAMS_TYPE_MSG} Found item type {type(comb)}")

    return params


def _iter_cmd_key_candidates(param_key: str):
    # We prefer "-" instead of "_", but the user-inputted command may contain "_"
    yield "--" + param_key.replace("_", "-")
    yield "--" + param_key.replace("-", "_")
    yield "--" + param_key


def _override_args(cmd: list[str], params: dict[str, object]):
    cmd = list(cmd)

    for k, v in params.items():
        for k_candidate in _iter_cmd_key_candidates(k):
            try:
                k_idx = cmd.index(k_candidate)
                cmd[k_idx + 1] = str(v)

                break
            except ValueError:
                continue
        else:
            cmd.extend([next(_iter_cmd_key_candidates(k)), str(v)])

    return cmd


def _get_path_one_comb(output_dir: Path, params: dict[str, object]):
    return output_dir / "_".join(f"{k}={v}" for k, v in params.items())


def _get_path_one_run(result_dir: Path, run_number: int):
    return result_dir / f"run={run_number}.json"


def benchmark_one_run(
    serve_cmd: list[str],
    bench_cmd: list[str],
    serve_comb: dict[str, object],
    run_number: int,
    result_dir: Path,
    dry_run: bool,
):
    result_path = _get_path_one_run(result_dir, run_number)

    server_cmd = _override_args(serve_cmd, serve_comb)
    benchmark_cmd = [
        *bench_cmd,
        "--save-result",
        "--result-dir",
        result_dir,
        "--result-filename",
        result_path.name,
    ]

    print("=" * 60)
    print(f"Parameter Combination: {serve_comb}")
    print(f"Run Number: {run_number}")
    print(f"Server command: {server_cmd}")
    print(f"Benchmark command: {benchmark_cmd}")
    print(f"Output file: {result_path}")

    if dry_run:
        return None

    # Create new process group for clean termination
    server_process = subprocess.Popen(server_cmd, start_new_session=True)

    try:
        subprocess.run(benchmark_cmd, check=True)
    finally:
        if server_process.poll() is None:
            # In case some processes have been terminated
            with contextlib.suppress(ProcessLookupError):
                # We need to kill both API Server and Engine processes
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)

    with result_path.open("rb") as f:
        run_data = json.load(f)

    run_data["run_number"] = run_number
    run_data.update(serve_comb)

    return run_data


def benchmark_one_comb(
    serve_cmd: list[str],
    bench_cmd: list[str],
    serve_comb: dict[str, object],
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
):
    result_dir = _get_path_one_comb(output_dir, serve_comb)
    if not dry_run:
        result_dir.mkdir(parents=True, exist_ok=True)

    comb_data = [
        benchmark_one_run(
            serve_cmd=serve_cmd,
            bench_cmd=bench_cmd,
            serve_comb=serve_comb,
            run_number=run_number,
            result_dir=result_dir,
            dry_run=dry_run,
        )
        for run_number in range(num_runs)
    ]

    if dry_run:
        return None

    with (result_dir / "summary.json").open("w") as f:
        json.dump(comb_data, f)

    return pd.DataFrame.from_records(comb_data)  # type: ignore[arg-type]


def benchmark_all(
    serve_cmd: list[str],
    bench_cmd: list[str],
    serve_params: list[dict[str, object]],
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp

    result_dfs = [
        benchmark_one_comb(
            serve_cmd=serve_cmd,
            bench_cmd=bench_cmd,
            serve_comb=serve_comb,
            output_dir=output_dir,
            num_runs=num_runs,
            dry_run=dry_run,
        )
        for serve_comb in _validate_combs(serve_params)
    ]

    if dry_run:
        return None

    combined_df = pd.concat(result_dfs)
    combined_df.to_csv(output_dir / "summary.csv")

    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description="Run vLLM server benchmark on a parameter grid of settings."
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
        help="The command used to run the benchmark: `vllm bench serve...`",
    )
    parser.add_argument(
        "--serve-params",
        type=str,
        default=None,
        help="Path to JSON file containing parameter combinations for the "
        "`vllm serve` command.",
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
        help="Number of runs per parameter combination",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, only prints the commands to run.",
    )

    args = parser.parse_args()

    serve_cmd = shlex.split(args.serve_cmd)
    bench_cmd = shlex.split(args.bench_cmd)

    if args.serve_params:
        with open(args.serve_params, "rb") as f:
            serve_params = json.load(f)
    else:
        # i.e.: run serve_cmd without any modification
        serve_params = [{}]

    benchmark_all(
        serve_cmd=serve_cmd,
        bench_cmd=bench_cmd,
        serve_params=serve_params,
        output_dir=Path(args.output_dir),
        num_runs=args.num_runs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
