# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`vllm bench sweep simulate`: the `vllm bench sweep serve` workflow in
simulated time.

Each server configuration runs in its own process as an in-process engine with
simulated execution (see docs/features/simulated_execution.md), fed the
requests `vllm bench serve` would send, so configurations run in parallel and
without the target GPUs. With `--slo`, the highest request rate that meets the
latency targets is searched for each configuration.
"""

import argparse
import json
import math
import multiprocessing
import os
import random
import shlex
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.import_utils import PlaceholderModule

from .param_sweep import ParameterSweep, ParameterSweepItem
from .serve import _get_comb_base_path

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")

_COMPLETION_BACKENDS = ("vllm", "openai")
_CHAT_BACKENDS = ("openai-chat",)


def _subcommand_args(cmd: list[str], subcommand: tuple[str, ...]) -> list[str]:
    n = len(subcommand)
    if tuple(cmd[:n]) != subcommand:
        raise ValueError(f"Expected a `{' '.join(subcommand)}` command, got {cmd}")
    return cmd[n:]


def _flatten(config: dict, prefix: str) -> dict[str, object]:
    """Dotted CLI keys for the leaves of a nested config: unlike a JSON value,
    they merge with the dotted keys of a parameter combination."""
    flat: dict[str, object] = {}
    for k, v in config.items():
        if isinstance(v, dict) and v:
            flat |= _flatten(v, f"{prefix}.{k}")
        else:
            flat[f"{prefix}.{k}"] = json.dumps(v) if isinstance(v, list) else v
    return flat


def simulated_serve_cmd(
    serve_cmd: list[str],
    simulation_config: dict,
    serve_comb: ParameterSweepItem,
) -> list[str]:
    """`serve_cmd` with simulated execution, then `serve_comb`, applied."""
    simulation = ParameterSweepItem(
        _flatten({"clock": "virtual"} | simulation_config, "simulation_config")
        | {"simulation_config.simulate_forward": True}
    )
    return serve_comb.apply_to_cmd(simulation.apply_to_cmd(serve_cmd))


def max_rate_meeting_slo(
    evaluate: Callable[[float], dict[str, object]],
    slo: dict[str, float],
    search_iters: int,
) -> float:
    """The highest request rate whose run meets every `slo` upper bound,
    bracketed from the throughput of a run with all requests sent at once and
    refined by `search_iters` geometric bisections; 0 if none was found."""

    def meets(run_data: dict[str, object]) -> bool:
        missing = slo.keys() - run_data.keys()
        if missing:
            raise ValueError(f"SLO metrics {sorted(missing)} are not reported.")
        return all(float(run_data[k]) <= v for k, v in slo.items())  # type: ignore[arg-type]

    batch = evaluate(math.inf)
    if meets(batch):
        return math.inf
    rate = float(batch["request_throughput"])  # type: ignore[arg-type]
    lo, hi = 0.0, math.inf
    for _ in range(8):
        if meets(evaluate(rate)):
            lo, rate = rate, rate * 2
        else:
            hi, rate = rate, rate / 2
        if lo > 0 and hi < math.inf:
            break
    if lo == 0 or hi == math.inf:
        return lo
    for _ in range(search_iters):
        mid = math.sqrt(lo * hi)
        if meets(evaluate(mid)):
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class _CombTask:
    serve_cmd: list[str]
    bench_cmd: list[str]
    serve_comb: ParameterSweepItem
    bench_comb: ParameterSweepItem
    comb_dir: Path
    slo: dict[str, float] | None
    search_iters: int
    index: int


def _prompt_token_ids(prompt: object, tokenizer, backend: str) -> list[int]:
    if backend in _CHAT_BACKENDS:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    if backend not in _COMPLETION_BACKENDS:
        raise ValueError(f"Simulated sweeps do not support the {backend} backend.")
    if isinstance(prompt, str):
        return tokenizer.encode(prompt)
    if isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
        return prompt
    raise ValueError(f"Simulated sweeps do not support prompts like {prompt!r}.")


def _select_devices(index: int, num_ranks: int) -> None:
    """Spread combinations over the visible devices: a simulated rank still
    launches small kernels, which processes sharing a device serialize."""
    from vllm.platforms import current_platform

    env_var = current_platform.device_control_env_var
    devices = os.environ.get(env_var, "").split(",")
    num_groups = len(devices) // num_ranks
    if num_groups > 1:
        start = index % num_groups * num_ranks
        os.environ[env_var] = ",".join(devices[start : start + num_ranks])


def _simulate_comb(task: _CombTask) -> None:
    """Simulate one combination in this (fresh) process, writing each run's
    `summary.json` and, after an SLO search, `result.json`."""
    from vllm.benchmarks.datasets import get_samples
    from vllm.benchmarks.serve import add_cli_args as add_bench_cli_args
    from vllm.benchmarks.serve import get_request_delays
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.sampling_params import SamplingParams
    from vllm.tokenizers import get_tokenizer
    from vllm.v1.worker.simulated.offline import TimedRequest, make_engine, run

    serve_args = make_arg_parser(FlexibleArgumentParser()).parse_args(
        _subcommand_args(task.serve_cmd, ("vllm", "serve"))
    )
    if serve_args.model_tag is not None:
        serve_args.model = serve_args.model_tag
    engine_args = EngineArgs.from_cli_args(serve_args)
    _select_devices(
        task.index,
        engine_args.tensor_parallel_size
        * engine_args.pipeline_parallel_size
        * engine_args.data_parallel_size,
    )

    bench_parser = FlexibleArgumentParser()
    add_bench_cli_args(bench_parser)
    bench_args = bench_parser.parse_args(
        _subcommand_args(task.bench_cmd, ("vllm", "bench", "serve"))
    )
    if bench_args.max_concurrency is not None:
        raise ValueError("Simulated sweeps do not support --max-concurrency.")
    bench_args.model = bench_args.model or engine_args.model
    if bench_args.self_timed is None:
        bench_args.self_timed = bench_args.dataset_name == "timed_trace"
    tokenizer = get_tokenizer(
        bench_args.tokenizer or engine_args.tokenizer or bench_args.model,
        tokenizer_mode=bench_args.tokenizer_mode,
        trust_remote_code=bench_args.trust_remote_code,
    )
    # Seeded as `vllm bench serve` is, so that the requests and their send
    # times are the same as in a real benchmark.
    random.seed(bench_args.seed)
    np.random.seed(bench_args.seed)
    samples = get_samples(bench_args, tokenizer)
    rng_state = np.random.get_state()
    if any(s.multi_modal_data for s in samples):
        raise ValueError("Simulated sweeps do not support multimodal requests.")
    prompts = [
        _prompt_token_ids(s.prompt, tokenizer, bench_args.backend) for s in samples
    ]

    engine = make_engine(engine_args)
    parallel_config = engine.vllm_config.parallel_config
    layout = {
        "tensor_parallel_size": parallel_config.tensor_parallel_size,
        "pipeline_parallel_size": parallel_config.pipeline_parallel_size,
        "data_parallel_size": parallel_config.data_parallel_size,
    }
    runs: dict[float, dict[str, object]] = {}

    def evaluate(rate: float) -> dict[str, object]:
        if rate in runs:
            return runs[rate]
        np.random.set_state(rng_state)
        delays, _ = get_request_delays(
            samples,
            rate,
            bench_args.burstiness,
            bench_args.ramp_up_strategy,
            bench_args.ramp_up_start_rps,
            bench_args.ramp_up_end_rps,
            bench_args.self_timed,
        )
        requests = [
            TimedRequest(
                prompt_token_ids=prompt,
                sampling_params=SamplingParams(
                    max_tokens=sample.expected_output_len, ignore_eos=True
                ),
                arrival_time=delay,
            )
            for prompt, sample, delay in zip(prompts, samples, delays)
        ]
        # As `vllm bench sweep serve` does between runs.
        engine.reset_prefix_cache(reset_connector=True)
        result = run(engine, requests)
        run_data = (
            dict(task.serve_comb)
            | dict(task.bench_comb)
            | layout
            | {"request_rate": rate, "wall_time_s": result.wall_time}
            | result.summary()
        )
        rate_dir = task.comb_dir / f"request_rate={rate:g}"
        rate_dir.mkdir(parents=True, exist_ok=True)
        with open(rate_dir / "summary.json", "w") as f:
            json.dump([run_data], f, indent=4)
        runs[rate] = run_data
        return run_data

    try:
        if task.slo is None:
            evaluate(bench_args.request_rate)
        else:
            max_rate = max_rate_meeting_slo(evaluate, task.slo, task.search_iters)
            best = (
                dict(task.serve_comb) | dict(task.bench_comb) | layout
                if max_rate == 0
                else runs[max_rate]
            )
            with open(task.comb_dir / "result.json", "w") as f:
                json.dump(best | {"max_request_rate": max_rate}, f, indent=4)
    finally:
        engine.engine_core.shutdown()


def _run_task(task: _CombTask) -> None:
    task.comb_dir.mkdir(parents=True, exist_ok=True)
    with open(task.comb_dir / "simulate.log", "w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        _simulate_comb(task)


@dataclass
class SweepSimulateArgs:
    serve_cmd: list[str]
    bench_cmd: list[str]
    serve_params: ParameterSweep
    bench_params: ParameterSweep
    simulation_config: dict
    slo: dict[str, float] | None
    search_iters: int
    num_workers: int
    output_dir: Path
    experiment_name: str
    continue_on_error: bool
    dry_run: bool
    resume: bool

    parser_name: ClassVar[str] = "simulate"
    parser_help: ClassVar[str] = (
        "Run `vllm bench sweep serve` combinations with simulated execution."
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        slo = None
        if args.slo:
            slo = {
                k.strip(): float(v)
                for k, v in (item.split("=") for item in args.slo.split(","))
            }
        bench_params = (
            ParameterSweep.read_json(args.bench_params)
            if args.bench_params
            else ParameterSweep.from_records([{}])
        )
        if slo and any(comb.has_param("request_rate") for comb in bench_params):
            raise ValueError("`--slo` searches request_rate; do not set it.")
        if args.num_workers < 1:
            raise ValueError("`num_workers` should be at least 1.")
        return cls(
            serve_cmd=shlex.split(args.serve_cmd),
            bench_cmd=shlex.split(args.bench_cmd),
            serve_params=(
                ParameterSweep.read_json(args.serve_params)
                if args.serve_params
                else ParameterSweep.from_records([{}])
            ),
            bench_params=bench_params,
            simulation_config=json.loads(args.simulation_config),
            slo=slo,
            search_iters=args.search_iters,
            num_workers=args.num_workers,
            output_dir=Path(args.output_dir),
            experiment_name=(
                args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
            ),
            continue_on_error=args.continue_on_error,
            dry_run=args.dry_run,
            resume=args.resume,
        )

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser.add_argument(
            "--serve-cmd",
            type=str,
            required=True,
            help="The `vllm serve ...` command of the deployment to simulate.",
        )
        parser.add_argument(
            "--bench-cmd",
            type=str,
            required=True,
            help="The `vllm bench serve ...` command whose requests to simulate.",
        )
        parser.add_argument(
            "--serve-params",
            type=str,
            default=None,
            help="JSON file of parameter combinations for `--serve-cmd`, as for "
            "`vllm bench sweep serve`.",
        )
        parser.add_argument(
            "--bench-params",
            type=str,
            default=None,
            help="JSON file of parameter combinations for `--bench-cmd`, as for "
            "`vllm bench sweep serve`.",
        )
        parser.add_argument(
            "--simulation-config",
            type=str,
            default="{}",
            help="JSON SimulationConfig applied to every combination (the clock "
            'defaults to "virtual"); combinations can override its keys as '
            "`simulation_config.<key>`.",
        )
        parser.add_argument(
            "--slo",
            type=str,
            default=None,
            help="Comma-separated upper bounds on run metrics, e.g. "
            "p99_ttft_ms=500,mean_tpot_ms=30. If set, searches each "
            "combination for the highest request rate meeting them.",
        )
        parser.add_argument(
            "--search-iters",
            type=int,
            default=5,
            help="Bisection steps of the request-rate search, after bracketing.",
        )
        parser.add_argument(
            "-j",
            "--num-workers",
            type=int,
            default=1,
            help="Number of combinations simulated in parallel.",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            type=str,
            default="results",
            help="The main directory to which results are written.",
        )
        parser.add_argument(
            "-e",
            "--experiment-name",
            type=str,
            default=None,
            help="The name of this experiment (defaults to current timestamp). "
            "Results will be stored under `output_dir/experiment_name`.",
        )
        parser.add_argument(
            "--continue-on-error",
            action="store_true",
            help="Record and skip a failed combination instead of terminating "
            "the entire sweep.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the commands to simulate, then exit.",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Only run the combinations without results under "
            "`output_dir/experiment_name`.",
        )
        return parser


def run_main(args: SweepSimulateArgs):
    experiment_dir = args.output_dir / args.experiment_name
    if args.resume and not experiment_dir.exists():
        raise ValueError(f"Cannot resume from non-existent {experiment_dir=}")
    if not args.resume and not args.dry_run and experiment_dir.exists():
        raise ValueError(f"Cannot overwrite existing {experiment_dir=}")

    tasks: list[_CombTask] = []
    for serve_comb in args.serve_params:
        serve_cmd = simulated_serve_cmd(
            args.serve_cmd, args.simulation_config, serve_comb
        )
        for bench_comb in args.bench_params:
            comb_dir = _get_comb_base_path(experiment_dir, serve_comb, bench_comb)
            if args.resume and (
                (comb_dir / "result.json").exists()
                if args.slo
                else any(comb_dir.rglob("summary.json"))
            ):
                continue
            task = _CombTask(
                serve_cmd=serve_cmd,
                bench_cmd=bench_comb.apply_to_cmd(args.bench_cmd),
                serve_comb=serve_comb,
                bench_comb=bench_comb,
                comb_dir=comb_dir,
                slo=args.slo,
                search_iters=args.search_iters,
                index=len(tasks),
            )
            print(f"[SIMULATE] {shlex.join(task.serve_cmd)}")
            print(f"           {shlex.join(task.bench_cmd)}")
            tasks.append(task)
    if args.dry_run:
        print(f"Experiment will be saved at: {experiment_dir}")
        return None

    # One process per combination: each builds an engine with its own
    # process-global simulation clock.
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    ) as pool:
        futures = {pool.submit(_run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
                print(f"[DONE] {task.comb_dir}")
            except Exception as exc:
                if not args.continue_on_error:
                    raise RuntimeError(
                        f"Simulating {task.comb_dir} failed; see its simulate.log"
                    ) from exc
                print(f"[FAILED] {task.comb_dir}: {exc}")
                with open(task.comb_dir / ".failure.json", "w") as f:
                    json.dump({"error": traceback.format_exc()}, f, indent=4)

    all_data = [
        run_data
        for path in sorted(experiment_dir.rglob("summary.json"))
        for run_data in json.loads(path.read_text())
    ]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(all_data).to_csv(experiment_dir / "summary.csv")
    if args.slo:
        df = pd.DataFrame.from_records(
            [
                json.loads(path.read_text())
                for path in sorted(experiment_dir.rglob("result.json"))
            ]
        )
        gpus = (
            df["tensor_parallel_size"]
            * df["pipeline_parallel_size"]
            * df["data_parallel_size"]
        )
        df["output_throughput_per_gpu"] = df.get("output_throughput", 0) / gpus
        df = df.sort_values("output_throughput_per_gpu", ascending=False)
        df.to_csv(experiment_dir / "slo_summary.csv")
        params = {k for comb in (*args.serve_params, *args.bench_params) for k in comb}
        columns = [c for c in df.columns if c in params]
        columns += ["max_request_rate", *args.slo, "output_throughput_per_gpu"]
        print(df[[c for c in columns if c in df.columns]].to_string(index=False))
    print(f"Experiment has been saved at: {experiment_dir}")
    return all_data


def main(args: argparse.Namespace):
    run_main(SweepSimulateArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=SweepSimulateArgs.parser_help)
    SweepSimulateArgs.add_cli_args(parser)

    main(parser.parse_args())
