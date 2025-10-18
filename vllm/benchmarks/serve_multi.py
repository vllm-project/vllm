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
import requests

_BAD_PARAMS_TYPE_MSG = (
    "The parameters to vary should be expressed as a JSON list of dictionaries."
)


def _validate_params(params: list[dict[str, object]]):
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

                if isinstance(v, bool):
                    cmd[k_idx] = k if v else "--no-" + k
                else:
                    cmd[k_idx + 1] = str(v)

                break
            except ValueError:
                continue
        else:
            if isinstance(v, bool):
                cmd.append(k if v else "--no-" + k)
            else:
                cmd.extend([next(_iter_cmd_key_candidates(k)), str(v)])

    return cmd


def _get_path_one_comb(
    output_dir: Path,
    serve_comb: dict[str, object],
    bench_comb: dict[str, object],
):
    return output_dir / "_".join(
        (
            *(f"s_{k}={v}" for k, v in serve_comb.items()),
            *(f"b_{k}={v}" for k, v in bench_comb.items()),
        )
    )


def _get_path_one_run(result_dir: Path, run_number: int):
    return result_dir / f"run={run_number}.json"


def _needs_server(
    serve_comb: dict[str, object],
    bench_combs: list[dict[str, object]],
    output_dir: Path,
    num_runs: int,
):
    for bench_comb in bench_combs:
        result_dir = _get_path_one_comb(
            output_dir,
            serve_comb=serve_comb,
            bench_comb=bench_comb,
        )
        for run_number in range(num_runs):
            result_path = _get_path_one_run(result_dir, run_number)
            if not result_path.exists():
                return True

    return False


@contextlib.contextmanager
def _run_server(
    serve_cmd: list[str],
    *,
    serve_overrides: dict[str, object],
    dry_run: bool,
):
    server_cmd = _override_args(serve_cmd, serve_overrides)

    for port_key in ("-p", "--port"):
        if port_key in server_cmd:
            port = int(server_cmd[server_cmd.index(port_key) + 1])
            break
    else:
        port = 8000  # The default value in vllm serve

    print("[Running Server]")
    print(f"Server overrides: {serve_overrides}")
    print(f"Server command: {server_cmd}")
    print(f"Server port: {port}")

    if dry_run:
        yield port
        return

    # Create new process group for clean termination
    server_process = subprocess.Popen(
        server_cmd,
        start_new_session=True,
        # Need VLLM_SERVER_DEV_MODE=1 for /reset_prefix_cache
        env={**os.environ, "VLLM_SERVER_DEV_MODE": "1"},
    )

    try:
        yield port
    finally:
        if server_process.poll() is None:
            # In case only some processes have been terminated
            with contextlib.suppress(ProcessLookupError):
                # We need to kill both API Server and Engine processes
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)


def _reset_caches(port: int, *, dry_run: bool):
    print("Resetting caches...")

    if dry_run:
        return

    res = requests.post(f"http://0.0.0.0:{port}/reset_prefix_cache")
    res.raise_for_status()

    res = requests.post(f"http://0.0.0.0:{port}/reset_mm_cache")
    res.raise_for_status()


def _run_benchmark(
    bench_cmd: list[str],
    *,
    serve_overrides: dict[str, object],
    bench_overrides: dict[str, object],
    run_number: int,
    output_dir: Path,
    dry_run: bool,
):
    result_dir = _get_path_one_comb(
        output_dir,
        serve_comb=serve_overrides,
        bench_comb=bench_overrides,
    )
    result_path = _get_path_one_run(result_dir, run_number)
    benchmark_cmd = [
        *_override_args(bench_cmd, bench_overrides),
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_path.name,
    ]

    print("[Running Benchmark]")
    print(f"Benchmark overrides: {bench_overrides}")
    print(f"Run Number: {run_number}")
    print(f"Benchmark command: {benchmark_cmd}")
    print(f"Output file: {result_path}")

    run_data: dict[str, object]

    if result_path.exists():
        print("Found existing results. Skipping.")

        with result_path.open("rb") as f:
            run_data = json.load(f)
            return run_data

    if dry_run:
        return None

    result_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(benchmark_cmd, check=True)

    with result_path.open("rb") as f:
        run_data = json.load(f)

    run_data["run_number"] = run_number
    run_data.update(serve_overrides)

    with result_path.open("w") as f:
        json.dump(run_data, f)

    return run_data


def run_all(
    serve_cmd: list[str],
    bench_cmd: list[str],
    *,
    serve_params: list[dict[str, object]],
    bench_params: list[dict[str, object]],
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
    resume: str | None,
):
    timestamp = resume or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp

    serve_combs = _validate_params(serve_params)
    bench_combs = _validate_params(bench_params)

    all_data = list[dict[str, object]]()
    for serve_comb in serve_combs:
        if _needs_server(serve_comb, bench_combs, output_dir, num_runs):
            with _run_server(
                serve_cmd,
                serve_overrides=serve_comb,
                dry_run=dry_run,
            ) as port:
                for bench_comb in bench_combs:
                    for run_number in range(num_runs):
                        run_data = _run_benchmark(
                            bench_cmd,
                            serve_overrides=serve_comb,
                            bench_overrides=bench_comb,
                            run_number=run_number,
                            output_dir=output_dir,
                            dry_run=dry_run,
                        )
                        _reset_caches(port, dry_run=dry_run)

                        if run_data is not None:
                            all_data.append(run_data)

    if dry_run:
        assert len(all_data) == 0
        return None

    assert len(all_data) == len(serve_combs) * len(bench_combs)
    combined_df = pd.DataFrame.from_records(all_data)
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
        "--bench-params",
        type=str,
        default=None,
        help="Path to JSON file containing parameter combinations for the "
        "`vllm bench serve` command.",
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
        help="If set, prints the commands to run and then exits without running them.",
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

    if args.serve_params:
        with open(args.serve_params, "rb") as f:
            serve_params = json.load(f)
    else:
        # i.e.: run serve_cmd without any modification
        serve_params = [{}]

    if args.bench_params:
        with open(args.bench_params, "rb") as f:
            bench_params = json.load(f)
    else:
        # i.e.: run bench_cmd without any modification
        bench_params = [{}]

    run_all(
        serve_cmd=serve_cmd,
        bench_cmd=bench_cmd,
        serve_params=serve_params,
        bench_params=bench_params,
        output_dir=Path(args.output_dir),
        num_runs=args.num_runs,
        dry_run=args.dry_run,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
