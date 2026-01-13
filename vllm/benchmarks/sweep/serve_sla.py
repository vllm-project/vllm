# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import contextlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal, get_args

from vllm.utils.import_utils import PlaceholderModule

from .param_sweep import ParameterSweep, ParameterSweepItem
from .serve import SweepServeArgs, run_benchmark, run_server
from .server import ServerProcess
from .sla_sweep import SLASweep, SLASweepItem
from .utils import sanitize_filename

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    PchipInterpolator = (
        PlaceholderModule("scipy")
        .placeholder_attr("interpolate")
        .placeholder_attr("PchipInterpolator")
    )


def _get_sla_base_path(
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


def _get_sla_iter_path(
    base_path: Path,
    sla_comb: SLASweepItem,
    sla_variable: str,
    sla_value: int | None,
):
    if sla_value is None:
        prefix = sla_comb.as_text(sep="-")
        return base_path / f"SLA--{prefix}.json"

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


def run_sla(
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
        run_data = run_benchmark(
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


class SLAHistory(dict[int, float]):
    def __init__(self, min_value: int, max_value: int) -> None:
        super().__init__()

        self.min_value = min_value
        self.max_value = max_value

    def get_xy(self) -> tuple[list[int], list[float]]:
        xs = list[int]()
        ys = list[float]()
        for x, y in sorted(self.items()):
            xs.append(x)
            ys.append(y)

        return xs, ys

    def get_max_passing(self) -> float:
        return max(
            (val for val, margin in self.items() if margin <= 0),
            default=self.min_value,
        )

    def get_min_failing(self) -> float:
        return min(
            (val for val, margin in self.items() if margin > 0),
            default=self.max_value,
        )


def solve_sla(
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
    sla_min_value: int = 1,
    sla_max_value: int = 8192,  # The value that represents infinite QPS
):
    sla_data = list[dict[str, object]]()
    history = SLAHistory(min_value=sla_min_value, max_value=sla_max_value)

    # NOTE: We don't use equality here to be more robust against noisy results
    while history.get_max_passing() + 1 < history.get_min_failing():
        if len(history) == 0:
            val = sla_max_value
        elif len(history) == 1:
            val = sla_min_value
        else:
            spl = PchipInterpolator(*history.get_xy(), extrapolate=False)
            spl_roots = spl.solve()
            if len(spl_roots) == 0:
                # Fallback to binary search
                val = int((history.get_max_passing() + history.get_min_failing()) / 2)
            else:
                val = int(spl_roots[0])

            if val in history:
                # Cover both sides (floor and ceil) of the root to be sure
                # that it is indeed the target value
                val += 1

        val = max(sla_min_value, min(val, sla_max_value))
        print(f"Testing {sla_variable}: {val} req/s")

        iter_data = run_sla(
            server,
            bench_cmd,
            serve_comb=serve_comb,
            bench_comb=bench_comb | {sla_variable: val},
            iter_path=_get_sla_iter_path(base_path, sla_comb, sla_variable, val),
            num_runs=num_runs,
            dry_run=dry_run,
        )
        if iter_data is None:
            return None

        sla_data.extend(iter_data)

        iter_data_mean = {
            k: sum(float(run_data[k]) for run_data in iter_data) / len(iter_data)  # type: ignore
            for k in sla_comb
        }

        sla_margins = [
            criterion.print_and_compute_margin(iter_data_mean, k)
            for k, criterion in sla_comb.items()
        ]
        history[val] = margin = max(sla_margins)

        if margin <= 0:
            print(f"SLA criteria are met. ({margin=:.2f})")
        else:
            print(f"SLA criteria are not met. ({margin=:.2f})")

    return sla_data, history


def search_sla(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
    sla_comb: SLASweepItem,
    sla_variable: SLAVariable,
    base_path: Path,
    num_runs: int,
    dry_run: bool,
):
    print("[SLA START]")
    print(f"SLA criteria: {sla_comb.as_text()}")

    result = solve_sla(
        server,
        bench_cmd,
        serve_comb=serve_comb,
        bench_comb=bench_comb,
        sla_comb=sla_comb,
        base_path=base_path,
        num_runs=num_runs,
        dry_run=dry_run,
        sla_variable=sla_variable,
    )
    if result is None:
        assert dry_run
        print("Omitting SLA search.")
        print("[SLA END]")
        return

    sla_data, sla_history = result
    sla_value = sla_history.get_max_passing()
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
            run_server(
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

                    comb_data = search_sla(
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


@dataclass
class SweepServeSLAArgs(SweepServeArgs):
    sla_params: SLASweep
    sla_variable: SLAVariable

    parser_name: ClassVar[str] = "serve_sla"
    parser_help: ClassVar[str] = "Tune a variable to meet SLAs under multiple settings."

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # NOTE: Don't use super() as `from_cli_args` calls `cls()`
        base_args = SweepServeArgs.from_cli_args(args)

        if args.sla_params:
            sla_params = SLASweep.read_json(args.sla_params)
        else:
            sla_params = SLASweep.from_records([])

        return cls(
            **asdict(base_args),
            sla_params=sla_params,
            sla_variable=args.sla_variable,
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().add_cli_args(parser)

        sla_group = parser.add_argument_group("sla options")
        sla_group.add_argument(
            "--sla-params",
            type=str,
            required=True,
            help="Path to JSON file containing a list of SLA constraints to satisfy. "
            'Each constraint is expressed in `{"<KEY>": "<OP><VALUE>"}` format, '
            'e.g.: `{"p99_e2el_ms": "<=500"}` means that '
            "the E2E latency should be less than 500ms 99%% of the time. "
            "Setting this option runs this script in SLA mode, which searches for "
            "the maximum `sla_variable` that satisfies the constraints for "
            "each combination of `serve_params`, `bench_params`, and `sla_params`.",
        )
        sla_group.add_argument(
            "--sla-variable",
            type=str,
            choices=get_args(SLAVariable),
            default="request_rate",
            help="Whether to tune request rate or maximum concurrency to satisfy "
            "the SLA constraints.",
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
            serve_params=args.serve_params,
            bench_params=args.bench_params,
            sla_params=args.sla_params,
            sla_variable=args.sla_variable,
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
    run_main(SweepServeSLAArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=SweepServeSLAArgs.parser_help)
    SweepServeSLAArgs.add_cli_args(parser)

    main(parser.parse_args())
