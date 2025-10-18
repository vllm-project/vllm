# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import contextlib
import json
import os
import shlex
import signal
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from typing_extensions import override

_BAD_PARAMS_TYPE_MSG = (
    "The parameters to vary should be expressed as a JSON list of dictionaries."
)


def _parse_params(params: list[dict[str, object]]):
    if not isinstance(params, list):
        raise TypeError(f"{_BAD_PARAMS_TYPE_MSG} Found JSON type {type(params)}")

    for comb in params:
        if not isinstance(comb, dict):
            raise TypeError(f"{_BAD_PARAMS_TYPE_MSG} Found item type {type(comb)}")

    return params


class SLACriterionBase(ABC):
    def __init__(self, target: float) -> None:
        super().__init__()

        self.target = target

    @abstractmethod
    def validate(self, actual: float) -> bool:
        """Return `True` if this criterion is met; otherwise `False`."""
        raise NotImplementedError

    @abstractmethod
    def format_cond(self, lhs: str) -> str:
        raise NotImplementedError

    def print_and_validate(
        self,
        metrics: dict[str, float],
        metrics_key: str,
    ) -> bool:
        metric = metrics[metrics_key]
        result = self.validate(metric)

        cond = self.format_cond(f"{metrics_key} = {metric:.2f}")
        print(f"Validating SLA: {cond} | " + ("PASSED" if result else "FAILED"))

        return result


class SLALessThan(SLACriterionBase):
    @override
    def validate(self, actual: float) -> bool:
        return actual < self.target

    @override
    def format_cond(self, lhs: str) -> str:
        return f"{lhs}<{self.target:.2f}"


class SLALessThanOrEqual(SLACriterionBase):
    @override
    def validate(self, actual: float) -> bool:
        return actual <= self.target

    @override
    def format_cond(self, lhs: str) -> str:
        return f"{lhs}<={self.target:.2f}"


class SLAGreaterThan(SLACriterionBase):
    @override
    def validate(self, actual: float) -> bool:
        return actual > self.target

    @override
    def format_cond(self, lhs: str) -> str:
        return f"{lhs}>{self.target:.2f}"


class SLAGreaterThanOrEqual(SLACriterionBase):
    @override
    def validate(self, actual: float) -> bool:
        return actual >= self.target

    @override
    def format_cond(self, lhs: str) -> str:
        return f"{lhs}>={self.target:.2f}"


# NOTE: The ordering is important! Match longer op_keys first
SLA_CRITERIA: dict[str, type[SLACriterionBase]] = {
    "<=": SLALessThanOrEqual,
    ">=": SLAGreaterThanOrEqual,
    "<": SLALessThan,
    ">": SLAGreaterThan,
}


def _parse_sla_item(sla_item: dict[str, str]):
    sla_criteria: dict[str, SLACriterionBase] = {}

    for metric_key, metric_value in sla_item.items():
        for op_key in SLA_CRITERIA:
            if metric_value.startswith(op_key):
                sla_criteria[metric_key] = SLA_CRITERIA[op_key](
                    float(metric_value.removeprefix(op_key))
                )
                break
        else:
            raise ValueError(
                f"Invalid operator for SLA constraint '{metric_key}={metric_value}'. "
                f"Valid operators are: {set(SLA_CRITERIA)}",
            )

    return sla_criteria


def _parse_sla(sla: list[dict[str, str]]):
    return [_parse_sla_item(item) for item in sla]


# In JSON, we prefer "_"
def _iter_param_key_candidates(param_key: str):
    yield param_key
    yield param_key.replace("-", "_")
    yield param_key.replace("_", "-")


# In CLI, we prefer "-"
def _iter_cmd_key_candidates(param_key: str):
    for k in reversed(tuple(_iter_param_key_candidates(param_key))):
        yield "--" + k


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

    print("[BEGIN SERVER]")
    print(f"Server overrides: {serve_overrides}")
    print(f"Server command: {server_cmd}")
    print(f"Server port: {port}")

    if dry_run:
        yield None
        print("[END SERVER]")
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

        print("[END SERVER]")


def _reset_caches(port: int):
    print("Resetting caches...")

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
    output_path: Path,
    dry_run: bool,
):
    benchmark_cmd = [
        *_override_args(bench_cmd, bench_overrides),
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

    if dry_run:
        print("[END BENCHMARK]")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(benchmark_cmd, check=True)

    with output_path.open("rb") as f:
        run_data = json.load(f)

    run_data["run_number"] = run_number
    run_data.update(serve_overrides)

    with output_path.open("w") as f:
        json.dump(run_data, f)

    print("[END BENCHMARK]")

    return run_data


def _get_comb_base_path(
    output_dir: Path,
    serve_comb: dict[str, object],
    bench_comb: dict[str, object],
):
    return output_dir / "-".join(
        (
            "serve",
            *(f"{k}={v}" for k, v in serve_comb.items()),
            "bench",
            *(f"{k}={v}" for k, v in bench_comb.items()),
        )
    )


def _get_comb_run_path(base_path: Path, run_number: int | None):
    if run_number is None:
        return base_path / "summary.json"

    return base_path / f"run={run_number}.json"


def _comb_needs_server(
    serve_comb: dict[str, object],
    bench_combs: list[dict[str, object]],
    output_dir: Path,
):
    for bench_comb in bench_combs:
        base_path = _get_comb_base_path(output_dir, serve_comb, bench_comb)
        if not _get_comb_run_path(base_path, run_number=None).exists():
            return True

    return False


def _run_comb(
    port: int | None,
    bench_cmd: list[str],
    *,
    serve_comb: dict[str, object],
    bench_comb: dict[str, object],
    base_path: Path,
    num_runs: int,
    dry_run: bool,
):
    comb_data = list[dict[str, object]]()

    for run_number in range(num_runs):
        run_path = _get_comb_run_path(base_path, run_number)

        run_data = _run_benchmark(
            bench_cmd,
            serve_overrides=serve_comb,
            bench_overrides=bench_comb,
            run_number=run_number,
            output_path=run_path,
            dry_run=dry_run,
        )

        if port is not None:
            _reset_caches(port)

        if run_data is not None:
            comb_data.append(run_data)

    if dry_run:
        return None

    with _get_comb_run_path(base_path, run_number=None).open("w") as f:
        json.dump(comb_data, f)

    return comb_data


def run_combs(
    serve_cmd: list[str],
    bench_cmd: list[str],
    *,
    serve_params: list[dict[str, object]],
    bench_params: list[dict[str, object]],
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
):
    all_data = list[dict[str, object]]()
    for serve_comb in serve_params:
        with (
            _run_server(
                serve_cmd,
                serve_overrides=serve_comb,
                dry_run=dry_run,
            )
            if _comb_needs_server(serve_comb, bench_params, output_dir)
            else contextlib.nullcontext()
        ) as port:
            for bench_comb in bench_params:
                base_path = _get_comb_base_path(output_dir, serve_comb, bench_comb)

                comb_data = _run_comb(
                    port,
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
    serve_comb: dict[str, object],
    bench_comb: dict[str, object],
    sla_comb: dict[str, SLACriterionBase],
):
    return output_dir / "-".join(
        (
            "serve",
            *(f"{k}={v}" for k, v in serve_comb.items()),
            "bench",
            *(f"{k}={v}" for k, v in bench_comb.items()),
            "sla",
            *(v.format_cond(k) for k, v in sla_comb.items()),
        )
    )


def _get_sla_iter_path(base_path: Path, request_rate: int | None):
    if request_rate is None:
        return base_path / "summary.json"

    return base_path / f"request_rate={request_rate}"


def _get_sla_run_path(iter_path: Path, run_number: int | None):
    if run_number is None:
        return iter_path / "summary.json"

    return iter_path / f"run={run_number}.json"


def _sla_needs_server(
    serve_comb: dict[str, object],
    bench_combs: list[dict[str, object]],
    sla_combs: list[dict[str, SLACriterionBase]],
    output_dir: Path,
):
    for bench_comb in bench_combs:
        for sla_comb in sla_combs:
            base_path = _get_sla_base_path(output_dir, serve_comb, bench_comb, sla_comb)
            if not _get_sla_iter_path(base_path, request_rate=None).exists():
                return True

    return False


def _run_sla(
    port: int | None,
    bench_cmd: list[str],
    *,
    serve_comb: dict[str, object],
    bench_comb: dict[str, object],
    iter_path: Path,
    num_runs: int,
    dry_run: bool,
):
    iter_data = list[dict[str, object]]()

    for run_number in range(num_runs):
        run_path = _get_sla_run_path(iter_path, run_number)

        run_data = _run_benchmark(
            bench_cmd,
            serve_overrides=serve_comb,
            bench_overrides=bench_comb,
            run_number=run_number,
            output_path=run_path,
            dry_run=dry_run,
        )

        if port is not None:
            _reset_caches(port)

        if run_data is not None:
            iter_data.append(run_data)

    if dry_run:
        return None

    with _get_sla_run_path(iter_path, run_number=None).open("w") as f:
        json.dump(iter_data, f)

    return iter_data


def _iter_sla(
    port: int | None,
    bench_cmd: list[str],
    *,
    serve_comb: dict[str, object],
    bench_comb: dict[str, object],
    sla_comb: dict[str, SLACriterionBase],
    base_path: Path,
    num_runs: int,
    dry_run: bool,
    init_request_rate: int = 4096,
):
    print("[SLA START]")
    print(f"SLA criteria: {', '.join(v.format_cond(k) for k, v in sla_comb.items())}")

    sla_data = list[dict[str, object]]()

    # Binary search
    request_rate_left: int = 0
    request_rate_right: int = init_request_rate * 2

    while request_rate_right > request_rate_left:
        request_rate = (request_rate_left + request_rate_right) // 2
        print(f"Search bounds: [{request_rate_left}, {request_rate_right}] req/s")
        print(f"Testing request rate: {request_rate} req/s")

        iter_path = _get_sla_iter_path(base_path, request_rate)

        iter_data = _run_sla(
            port,
            bench_cmd,
            serve_comb=serve_comb,
            bench_comb={**bench_comb, "request_rate": request_rate},
            iter_path=iter_path,
            num_runs=num_runs,
            dry_run=dry_run,
        )

        if port is not None:
            _reset_caches(port)

        if iter_data is not None:
            sla_data.extend(iter_data)

        if iter_data is None:
            assert dry_run
            print("Omitting binary search iterations.")
            break

        iter_data_mean = {
            k: sum(float(run_data[k]) for run_data in iter_data) / len(iter_data)  # type: ignore
            for k in sla_comb
        }

        sla_results = [
            criterion.print_and_validate(iter_data_mean, k)
            for k, criterion in sla_comb.items()
        ]

        if all(sla_results):
            if request_rate_right == init_request_rate:
                print("SLA is satisfied even with unbounded request rate.")
                break

            print("SLA criteria has been met.")
            request_rate_left = request_rate
        else:
            print("SLA criteria has not been met.")
            request_rate_right = request_rate

        # Use `<= 1` because `(odd + even) // 2 == 1`
        if abs(request_rate_left - request_rate_right) <= 1:
            print("Binary search has converged.")
            break

    if dry_run:
        print("[SLA END]")
        return None

    with _get_sla_iter_path(base_path, request_rate=None).open("w") as f:
        json.dump(sla_data, f)

    print("[SLA END]")

    return sla_data


def run_slas(
    serve_cmd: list[str],
    bench_cmd: list[str],
    *,
    serve_params: list[dict[str, object]],
    bench_params: list[dict[str, object]],
    sla_params: list[dict[str, SLACriterionBase]],
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
):
    if any(
        k in bench_comb
        for bench_comb in bench_params
        for k in _iter_param_key_candidates("request_rate")
    ):
        raise ValueError(
            "You should not override `request_rate` in `bench_params` in SLA mode, "
            "since `request_rate` is supposed to be determined automatically."
        )

    all_data = list[dict[str, object]]()
    for serve_comb in serve_params:
        with (
            _run_server(
                serve_cmd,
                serve_overrides=serve_comb,
                dry_run=dry_run,
            )
            if _sla_needs_server(serve_comb, bench_params, sla_params, output_dir)
            else contextlib.nullcontext()
        ) as port:
            for bench_comb in bench_params:
                for sla_comb in sla_params:
                    base_path = _get_sla_base_path(
                        output_dir, serve_comb, bench_comb, sla_comb
                    )

                    comb_data = _iter_sla(
                        port,
                        bench_cmd,
                        serve_comb=serve_comb,
                        bench_comb=bench_comb,
                        sla_comb=sla_comb,
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


def run_main(
    serve_cmd: list[str],
    bench_cmd: list[str],
    *,
    serve_params: list[dict[str, object]],
    bench_params: list[dict[str, object]],
    sla_params: list[dict[str, SLACriterionBase]],
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
    resume: str | None,
):
    timestamp = resume or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp

    if resume and not output_dir.exists():
        raise ValueError(f"Cannot resume from non-existent directory ({output_dir})")

    if sla_params:
        return run_slas(
            serve_cmd=serve_cmd,
            bench_cmd=bench_cmd,
            serve_params=serve_params,
            bench_params=bench_params,
            sla_params=sla_params,
            output_dir=output_dir,
            num_runs=num_runs,
            dry_run=dry_run,
        )

    return run_combs(
        serve_cmd=serve_cmd,
        bench_cmd=bench_cmd,
        serve_params=serve_params,
        bench_params=bench_params,
        output_dir=output_dir,
        num_runs=num_runs,
        dry_run=dry_run,
    )


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
        "Setting this option runs this script in SLA mode, where the request rate "
        "is iteratively reduced to satisfy the constraints for each combination "
        "of `serve_params`, `bench_params`, and `sla_params`.",
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

    serve_params: list[dict[str, object]]
    if args.serve_params:
        with open(args.serve_params, "rb") as f:
            serve_params = _parse_params(json.load(f))
    else:
        # i.e.: run serve_cmd without any modification
        serve_params = [{}]

    bench_params: list[dict[str, object]]
    if args.bench_params:
        with open(args.bench_params, "rb") as f:
            bench_params = _parse_params(json.load(f))
    else:
        # i.e.: run bench_cmd without any modification
        bench_params = [{}]

    sla_params: list[dict[str, SLACriterionBase]]
    if args.sla_params:
        with open(args.sla_params, "rb") as f:
            sla_params = _parse_sla(json.load(f))
    else:
        sla_params = []

    num_runs = args.num_runs
    if num_runs < 1:
        raise ValueError("`num_runs` should be at least 1.")

    run_main(
        serve_cmd=serve_cmd,
        bench_cmd=bench_cmd,
        serve_params=serve_params,
        bench_params=bench_params,
        sla_params=sla_params,
        output_dir=Path(args.output_dir),
        num_runs=num_runs,
        dry_run=args.dry_run,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
