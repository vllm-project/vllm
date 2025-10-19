# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import contextlib
import json
import math
import os
import shlex
import signal
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args

import pandas as pd
import requests
import seaborn as sns
from typing_extensions import assert_never, override

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


def _normalize_cmd_key(param_key: str):
    return next(_iter_cmd_key_candidates(param_key))


def _override_args(cmd: list[str], params: dict[str, object]):
    cmd = list(cmd)

    for k, v in params.items():
        for k_candidate in _iter_cmd_key_candidates(k):
            try:
                k_idx = cmd.index(k_candidate)

                if isinstance(v, bool):
                    cmd[k_idx] = _normalize_cmd_key(k if v else "no-" + k)
                else:
                    cmd[k_idx + 1] = str(v)

                break
            except ValueError:
                continue
        else:
            if isinstance(v, bool):
                cmd.append(_normalize_cmd_key(k if v else "no-" + k))
            else:
                cmd.extend([_normalize_cmd_key(k), str(v)])

    return cmd


class ServerWrapper:
    def __init__(
        self,
        server_cmd: list[str],
        after_bench_cmd: list[str],
        *,
        show_stdout: bool,
    ) -> None:
        super().__init__()

        self.server_cmd = server_cmd
        self.after_bench_cmd = after_bench_cmd
        self.show_stdout = show_stdout

    def run_subcommand(self, cmd: list[str]):
        return subprocess.run(
            cmd,
            stdout=None if self.show_stdout else subprocess.DEVNULL,
            check=True,
        )

    def after_bench(self) -> None:
        if not self.after_bench_cmd:
            self.reset_caches()
            return

        self.run_subcommand(self.after_bench_cmd)

    def _get_vllm_server_address(self) -> str:
        server_cmd = self.server_cmd

        for host_key in ("--host",):
            if host_key in server_cmd:
                host = server_cmd[server_cmd.index(host_key) + 1]
                break
        else:
            host = "localhost"

        for port_key in ("-p", "--port"):
            if port_key in server_cmd:
                port = int(server_cmd[server_cmd.index(port_key) + 1])
                break
        else:
            port = 8000  # The default value in vllm serve

        return f"http://{host}:{port}"

    def reset_caches(self) -> None:
        server_cmd = self.server_cmd

        # Use `.endswith()` to match `/bin/...`
        if server_cmd[0].endswith("vllm"):
            server_address = self._get_vllm_server_address()
            print(f"Resetting caches at {server_address}")

            res = requests.post(f"{server_address}/reset_prefix_cache")
            res.raise_for_status()

            res = requests.post(f"{server_address}/reset_mm_cache")
            res.raise_for_status()
        elif server_cmd[0].endswith("infinity_emb"):
            if "--vector-disk-cache" in server_cmd:
                raise NotImplementedError(
                    "Infinity server uses caching but does not expose a method "
                    "to reset the cache"
                )
        else:
            raise NotImplementedError(
                f"No implementation of `reset_caches` for `{server_cmd[0]}` server. "
                "Please specify a custom command via `--after-bench-cmd`."
            )


@contextlib.contextmanager
def _run_server(
    serve_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    serve_overrides: dict[str, object],
    dry_run: bool,
):
    server_cmd = _override_args(serve_cmd, serve_overrides)

    print("[BEGIN SERVER]")
    print(f"Server overrides: {serve_overrides}")
    print(f"Server command: {server_cmd}")

    if dry_run:
        yield None
        print("[END SERVER]")
        return

    # Create new process group for clean termination
    server_process = subprocess.Popen(
        server_cmd,
        start_new_session=True,
        stdout=None if show_stdout else subprocess.DEVNULL,
        # Need VLLM_SERVER_DEV_MODE=1 for `_reset_caches`
        env={**os.environ, "VLLM_SERVER_DEV_MODE": "1"},
    )

    try:
        yield ServerWrapper(
            server_cmd,
            after_bench_cmd,
            show_stdout=show_stdout,
        )
    finally:
        if server_process.poll() is None:
            # In case only some processes have been terminated
            with contextlib.suppress(ProcessLookupError):
                # We need to kill both API Server and Engine processes
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)

        print("[END SERVER]")


def _run_benchmark(
    server: ServerWrapper | None,
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
    serve_comb: dict[str, object],
    bench_comb: dict[str, object],
):
    return output_dir / "-".join(
        (
            "SERVE",
            *(f"{k}={v}" for k, v in serve_comb.items()),
            "BENCH",
            *(f"{k}={v}" for k, v in bench_comb.items()),
        )
    ).replace("/", "_").replace("..", "__")  # Sanitize


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
    server: ServerWrapper | None,
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
    serve_comb: dict[str, object],
    bench_comb: dict[str, object],
):
    return output_dir / "-".join(
        (
            "SERVE",
            *(f"{k}={v}" for k, v in serve_comb.items()),
            "BENCH",
            *(f"{k}={v}" for k, v in bench_comb.items()),
        )
    ).replace("/", "_").replace("..", "__")  # Sanitize


def _get_sla_iter_path(
    base_path: Path,
    sla_comb: dict[str, SLACriterionBase],
    sla_variable: str,
    sla_value: int | None,
):
    if sla_value is None:
        prefix = "-".join(v.format_cond(k) for k, v in sla_comb.items())
        return base_path / f"SLA-{prefix}.json"

    return base_path / f"{sla_variable}={sla_value}"


def _get_sla_run_path(iter_path: Path, run_number: int | None):
    if run_number is None:
        return iter_path / "summary.json"

    return iter_path / f"run={run_number}.json"


def _sla_needs_server(
    serve_comb: dict[str, object],
    bench_combs: list[dict[str, object]],
    sla_combs: list[dict[str, SLACriterionBase]],
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
    server: ServerWrapper | None,
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
    server: ServerWrapper | None,
    bench_cmd: list[str],
    *,
    serve_comb: dict[str, object],
    bench_comb: dict[str, object],
    sla_comb: dict[str, SLACriterionBase],
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

    while True:
        print(f"Testing {sla_variable}: {val} req/s")

        iter_data = _run_sla(
            server,
            bench_cmd,
            serve_comb=serve_comb,
            bench_comb={**bench_comb, sla_variable: val},
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
            val = 2 if val <= 1 else val * 2
        else:
            print("SLA criteria are not met.")
            min_failing = val
            break

        if val >= max_value:
            break

    return sla_data, (min_failing, max_passing)


def _find_sla_value(
    server: ServerWrapper | None,
    bench_cmd: list[str],
    *,
    serve_comb: dict[str, object],
    bench_comb: dict[str, object],
    sla_comb: dict[str, SLACriterionBase],
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
            bench_comb={**bench_comb, sla_variable: val},
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
    server: ServerWrapper | None,
    bench_cmd: list[str],
    *,
    serve_comb: dict[str, object],
    bench_comb: dict[str, object],
    sla_comb: dict[str, SLACriterionBase],
    sla_variable: SLAVariable,
    sla_inf_value: int = 8192,  # The value that represents infinite QPS
    base_path: Path,
    num_runs: int,
    dry_run: bool,
):
    print("[SLA START]")
    print(f"SLA criteria: {', '.join(v.format_cond(k) for k, v in sla_comb.items())}")

    sla_data_0 = _run_sla(
        server,
        bench_cmd,
        serve_comb=serve_comb,
        bench_comb={**bench_comb, sla_variable: sla_inf_value},
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


def _plot_throughput_latency_curve(
    all_data: list[dict[str, object]],
    serve_combs: list[dict[str, object]],
    bench_comb: dict[str, object],
    output_dir: Path,
):
    fig_path = output_dir / "-".join(
        (
            "BENCH",
            *(f"{k}={v}" for k, v in bench_comb.items()),
        )
    ).replace("/", "_").replace("..", "__")  # Sanitize

    df = pd.DataFrame.from_records(
        [item for item in all_data if all(item[k] == bench_comb[k] for k in bench_comb)]
    )

    # Group together points with similar throughput
    df["request_throughput"] = df["request_throughput"].round()

    # Preserve the key order using dictionary
    all_comb_keys = {k: None for comb in serve_combs for k in comb}
    for k in all_comb_keys:
        df[k] = df[k].astype(str)

    keys_per_comb = [comb.keys() for comb in serve_combs]
    if (
        all(ks == keys_per_comb[0] for ks in keys_per_comb)
        and len(keys_per_comb[0]) <= 3
    ):
        hue, style, size, *_ = (*keys_per_comb[0], None, None)
        ax = sns.lineplot(
            df,
            x="request_throughput",
            y="p99_e2el_ms",
            hue=hue,
            style=style,
            size=size,
            markers=True,
        )
    else:
        df["category"] = df[list(all_comb_keys)].agg("-".join, axis=1)
        ax = sns.lineplot(
            df,
            x="request_throughput",
            y="p99_e2el_ms",
            hue="category",
            markers=True,
        )

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    fig = ax.get_figure()
    assert fig is not None

    fig.tight_layout()
    fig.savefig(fig_path)


def _plot_throughput_latency_curves(
    all_data: list[dict[str, object]],
    serve_combs: list[dict[str, object]],
    bench_combs: list[dict[str, object]],
    output_dir: Path,
):
    for bench_comb in bench_combs:
        _plot_throughput_latency_curve(all_data, serve_combs, bench_comb, output_dir)


def run_slas(
    serve_cmd: list[str],
    bench_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    serve_params: list[dict[str, object]],
    bench_params: list[dict[str, object]],
    sla_params: list[dict[str, SLACriterionBase]],
    sla_variable: SLAVariable,
    output_dir: Path,
    num_runs: int,
    dry_run: bool,
):
    if any(
        k in bench_comb
        for bench_comb in bench_params
        for k in _iter_param_key_candidates(sla_variable)
    ):
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

    _plot_throughput_latency_curves(all_data, serve_params, bench_params, output_dir)

    return combined_df


def _run_main(
    serve_cmd: list[str],
    bench_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    serve_params: list[dict[str, object]],
    bench_params: list[dict[str, object]],
    sla_params: list[dict[str, SLACriterionBase]],
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
    serve_params: list[dict[str, object]],
    bench_params: list[dict[str, object]],
    sla_params: list[dict[str, SLACriterionBase]],
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
