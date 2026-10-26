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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.import_utils import PlaceholderModule

from .param_sweep import ParameterSweepItem
from .serve import SweepServeArgs, _comb_is_valid, _get_comb_base_path

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.offline import SimulationResult, TimedRequest

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")

_COMPLETION_BACKENDS = ("vllm", "openai")
_CHAT_BACKENDS = ("openai-chat",)
# Rate doublings or halvings that bracket the SLO search.
_BRACKET_STEPS = 8
# Requests that fill the KV caches, as a multiple of their capacity.
_KV_FILL_FACTOR = 1.2
# Fraction of k-NN lookups beyond the recorded range that warrants a warning.
_MAX_OUTSIDE_RANGE = 0.01


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
    if not rate > 0:
        return 0.0
    lo, hi = 0.0, math.inf
    for _ in range(_BRACKET_STEPS):
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


_NUM_INSTANCES = ("_num_prefill", "_num_decode")
_ROLES = ("prefill", "decode")


def split_roles(
    serve_comb: ParameterSweepItem,
) -> tuple[ParameterSweepItem, dict[str, ParameterSweepItem], dict[str, int]]:
    """A combination's parameters for every instance, those of each role of a
    disaggregated deployment (keys `_prefill.<param>` and `_decode.<param>`),
    and its instance counts (`_num_prefill` and `_num_decode`, each 1 if only
    the other is given; empty for an aggregated deployment)."""
    common = ParameterSweepItem(
        {
            k: v
            for k, v in serve_comb.items()
            if k not in _NUM_INSTANCES and not k.startswith(("_prefill.", "_decode."))
        }
    )
    roles = {
        role: ParameterSweepItem(
            {
                k.removeprefix(f"_{role}."): v
                for k, v in serve_comb.items()
                if k.startswith(f"_{role}.")
            }
        )
        for role in _ROLES
    }
    counts = {}
    if any(k in serve_comb for k in _NUM_INSTANCES) or any(roles.values()):
        counts = {
            role: int(serve_comb.get(f"_num_{role}", 1))  # type: ignore[call-overload]
            for role in _ROLES
        }
    return common, roles, counts


@dataclass
class _CombTask:
    serve_cmds: dict[str, list[str]]
    """`vllm serve` command per role: "serve" for an aggregated deployment,
    "prefill" and "decode" for a disaggregated one."""
    num_instances: dict[str, int]
    bench_cmd: list[str]
    serve_comb: ParameterSweepItem
    bench_comb: ParameterSweepItem
    comb_dir: Path
    slo: dict[str, float] | None
    search_iters: int
    frontend_latency_ms: float
    handoff_latency_ms: float
    fill_kv_caches: bool
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


def _engine_args(cmd: list[str]) -> "EngineArgs":
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.launchers.cli_args import make_arg_parser

    serve_args = make_arg_parser(FlexibleArgumentParser()).parse_args(
        _subcommand_args(cmd, ("vllm", "serve"))
    )
    if serve_args.model_tag is not None:
        serve_args.model = serve_args.model_tag
    return EngineArgs.from_cli_args(serve_args)


def _bench_args(task: _CombTask, model: str) -> argparse.Namespace:
    from vllm.benchmarks.serve import add_cli_args

    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(
        _subcommand_args(task.bench_cmd, ("vllm", "bench", "serve"))
    )
    if args.max_concurrency is not None:
        raise ValueError("Simulated sweeps do not support --max-concurrency.")
    if task.slo is not None and (
        args.ramp_up_strategy is not None
        or args.dataset_name == "timed_trace"
        or args.self_timed
    ):
        raise ValueError(
            "`--slo` searches the request rate, which ramp-up and timed "
            "traces do not use."
        )
    args.model = args.model or model
    if args.self_timed is None:
        args.self_timed = args.dataset_name == "timed_trace"
    return args


class _CombSimulation:
    """One combination's engines and requests, in its own process."""

    def __init__(self, task: _CombTask):
        from vllm.benchmarks.datasets import get_samples
        from vllm.benchmarks.serve import (
            GENERATION_PERCENTILE_METRICS,
            check_goodput_args,
        )
        from vllm.tokenizers import get_tokenizer

        self.task = task
        disaggregated = bool(task.num_instances)
        role_args = {role: _engine_args(cmd) for role, cmd in task.serve_cmds.items()}
        for args in role_args.values():
            # Under the virtual clock (which disaggregated instances always
            # use) host time comes only from the timing model, so stats
            # collection costs wall time and changes nothing.
            if disaggregated or args.simulation_config.clock == "virtual":
                args.disable_log_stats = True
        num_ranks = {
            role: args.tensor_parallel_size
            * args.pipeline_parallel_size
            * args.data_parallel_size
            for role, args in role_args.items()
        }
        if disaggregated and any(n > 1 for n in num_ranks.values()):
            raise ValueError(
                "Disaggregated combinations run their instances in one process, "
                "with one rank each; serve multi-rank instances with "
                "`vllm serve --simulate-forward` instead."
            )
        _select_devices(task.index, 1 if disaggregated else num_ranks["serve"])
        engine_args = next(iter(role_args.values()))

        bench = _bench_args(task, engine_args.model)
        self.bench_args = bench
        self.percentile_metrics = (
            bench.percentile_metrics or GENERATION_PERCENTILE_METRICS
        ).split(",")
        self.percentiles = [float(p) for p in bench.metric_percentiles.split(",")]
        self.goodput = check_goodput_args(bench)
        self.tokenizer = get_tokenizer(
            bench.tokenizer or engine_args.tokenizer or bench.model,
            tokenizer_mode=bench.tokenizer_mode,
            trust_remote_code=bench.trust_remote_code,
        )
        # Seeded as `vllm bench serve` is, so that the requests and their send
        # times are the same as in a real benchmark.
        random.seed(bench.seed)
        np.random.seed(bench.seed)
        self.samples = get_samples(bench, self.tokenizer)
        self.rng_state = np.random.get_state()
        if any(s.multi_modal_data for s in self.samples):
            raise ValueError("Simulated sweeps do not support multimodal requests.")
        self.prompts = [
            _prompt_token_ids(s.prompt, self.tokenizer, bench.backend)
            for s in self.samples
        ]
        self._build_engines(role_args)
        self.runs: dict[float, dict[str, object]] = {}

    def _build_engines(self, role_args: dict[str, "EngineArgs"]) -> None:
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.worker.simulated.disaggregated import make_instance
        from vllm.v1.worker.simulated.offline import make_engine

        # Unset scheduler limits default as for `vllm serve`.
        serving = UsageContext.OPENAI_API_SERVER
        counts = self.task.num_instances
        if counts:
            self.prefill = [
                make_instance(
                    role_args["prefill"], "kv_producer", f"prefill{i}", serving
                )
                for i in range(counts["prefill"])
            ]
            self.decode = [
                make_instance(role_args["decode"], "kv_consumer", f"decode{i}", serving)
                for i in range(counts["decode"])
            ]
            self.engines = [*self.prefill, *self.decode]
            self.layout: dict[str, object] = {"gpu_count": sum(counts.values())}
            return
        engine = make_engine(role_args["serve"], usage_context=serving)
        self.engines = [engine]
        parallel_config = engine.vllm_config.parallel_config
        self.layout = {
            "tensor_parallel_size": parallel_config.tensor_parallel_size,
            "pipeline_parallel_size": parallel_config.pipeline_parallel_size,
            "data_parallel_size": parallel_config.data_parallel_size,
            "gpu_count": parallel_config.world_size_across_dp,
        }

    def _run(self, requests: list["TimedRequest"]) -> "SimulationResult":
        from vllm.v1.worker.simulated.disaggregated import run_disaggregated
        from vllm.v1.worker.simulated.offline import run

        frontend_latency_s = self.task.frontend_latency_ms * 1e-3
        if self.task.num_instances:
            return run_disaggregated(
                self.prefill,
                self.decode,
                requests,
                handoff_latency_s=self.task.handoff_latency_ms * 1e-3,
                frontend_latency_s=frontend_latency_s,
            )
        return run(self.engines[0], requests, frontend_latency_s=frontend_latency_s)

    def fill_kv_caches(self) -> None:
        """Use up every instance's KV cache once, with one-token requests of
        the dataset's mean prompt length."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.worker.simulated.offline import TimedRequest

        cache_tokens = max(
            (e.vllm_config.cache_config.num_gpu_blocks or 0)
            * e.vllm_config.cache_config.block_size
            for e in self.engines
        )
        prompt_len = max(1, round(float(np.mean([len(p) for p in self.prompts]))))
        per_instance = max(self.task.num_instances.values(), default=1)
        num_fill = math.ceil(_KV_FILL_FACTOR * per_instance * cache_tokens / prompt_len)
        rng = np.random.default_rng(self.bench_args.seed)
        vocab_size = self.tokenizer.vocab_size
        params = SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False)
        self._run(
            [
                TimedRequest(rng.integers(0, vocab_size, prompt_len).tolist(), params)
                for _ in range(num_fill)
            ]
        )

    def evaluate(self, rate: float) -> dict[str, object]:
        """Simulate the benchmark at `rate` and write its `summary.json`."""
        from vllm.benchmarks.serve import get_request_delays
        from vllm.sampling_params import SamplingParams
        from vllm.v1.worker.simulated.offline import TimedRequest
        from vllm.v1.worker.simulated.timing import knn_lookup_counts

        if rate in self.runs:
            return self.runs[rate]
        task, bench = self.task, self.bench_args
        np.random.set_state(self.rng_state)
        delays, _ = get_request_delays(
            self.samples,
            rate,
            bench.burstiness,
            bench.ramp_up_strategy,
            bench.ramp_up_start_rps,
            bench.ramp_up_end_rps,
            bench.self_timed,
        )
        requests = [
            TimedRequest(
                prompt_token_ids=prompt,
                # A server detokenizes in its frontend, off the engine core.
                sampling_params=SamplingParams(
                    max_tokens=sample.expected_output_len,
                    ignore_eos=True,
                    detokenize=False,
                ),
                arrival_time=delay,
            )
            for prompt, sample, delay in zip(self.prompts, self.samples, delays)
        ]
        # As `vllm bench sweep serve` does between runs.
        for e in self.engines:
            e.reset_prefix_cache(reset_connector=True)
        lookups_before, outside_before = knn_lookup_counts()
        result = self._run(requests)
        lookups, outside = knn_lookup_counts()
        run_data = (
            dict(task.serve_comb)
            | dict(task.bench_comb)
            | self.layout
            | {
                "request_rate": rate,
                "frontend_latency_ms": task.frontend_latency_ms,
                "kv_caches_filled": task.fill_kv_caches,
                **(
                    {"handoff_latency_ms": task.handoff_latency_ms}
                    if task.num_instances
                    else {}
                ),
                "wall_time_s": result.wall_time,
                # Steps timed by k-NN lookups beyond the recorded range.
                "timing_lookups_outside_range": outside - outside_before,
                "timing_lookups": lookups - lookups_before,
            }
            | result.summary(self.percentile_metrics, self.percentiles, self.goodput)
        )
        rate_dir = task.comb_dir / f"request_rate={rate:g}"
        rate_dir.mkdir(parents=True, exist_ok=True)
        with open(rate_dir / "summary.json", "w") as f:
            json.dump([run_data], f, indent=4)
        self.runs[rate] = run_data
        return run_data

    def shutdown(self) -> None:
        for e in self.engines:
            e.engine_core.shutdown()


def _simulate_comb(task: _CombTask) -> None:
    """Simulate one combination in this (fresh) process, writing each run's
    `summary.json` and, after an SLO search, `result.json`."""
    sim = _CombSimulation(task)
    try:
        if task.fill_kv_caches:
            sim.fill_kv_caches()
        if task.slo is None:
            sim.evaluate(sim.bench_args.request_rate)
            return
        max_rate = max_rate_meeting_slo(sim.evaluate, task.slo, task.search_iters)
        best = (
            dict(task.serve_comb) | dict(task.bench_comb) | sim.layout
            if max_rate == 0
            else sim.runs[max_rate]
        )
        with open(task.comb_dir / "result.json", "w") as f:
            json.dump(best | {"max_request_rate": max_rate}, f, indent=4)
    finally:
        sim.shutdown()


def _run_task(task: _CombTask) -> None:
    task.comb_dir.mkdir(parents=True, exist_ok=True)
    with open(task.comb_dir / "simulate.log", "w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        _simulate_comb(task)


@dataclass
class SweepSimulateArgs(SweepServeArgs):
    simulation_config: dict
    slo: dict[str, float] | None
    search_iters: int
    frontend_latency_ms: float
    handoff_latency_ms: float
    fill_kv_caches: bool
    num_workers: int

    parser_name: ClassVar[str] = "simulate"
    parser_help: ClassVar[str] = (
        "Run `vllm bench sweep serve` combinations with simulated execution. "
        "Options that only concern a real server or repeated runs are ignored."
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # NOTE: Don't use super() as `from_cli_args` calls `cls()`
        base_args = SweepServeArgs.from_cli_args(args)
        slo = None
        if args.slo:
            slo = {
                k.strip(): float(v)
                for k, v in (item.split("=") for item in args.slo.split(","))
            }
        if slo and any(
            comb.has_param("request_rate") for comb in base_args.bench_params
        ):
            raise ValueError("`--slo` searches request_rate; do not set it.")
        if args.frontend_latency_ms < 0 or args.handoff_latency_ms < 0:
            raise ValueError("Latencies should be non-negative.")
        if args.num_workers < 1:
            raise ValueError("`num_workers` should be at least 1.")
        return cls(
            **{f.name: getattr(base_args, f.name) for f in fields(base_args)},
            simulation_config=json.loads(args.simulation_config),
            slo=slo,
            search_iters=args.search_iters,
            frontend_latency_ms=args.frontend_latency_ms,
            handoff_latency_ms=args.handoff_latency_ms,
            fill_kv_caches=args.fill_kv_caches,
            num_workers=args.num_workers,
        )

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser = super().add_cli_args(parser)
        group = parser.add_argument_group("simulation options")
        group.add_argument(
            "--simulation-config",
            type=str,
            default="{}",
            help="JSON SimulationConfig applied to every combination (the clock "
            'defaults to "virtual"); combinations can override its keys as '
            "`simulation_config.<key>`.",
        )
        group.add_argument(
            "--slo",
            type=str,
            default=None,
            help="Comma-separated upper bounds on run metrics, e.g. "
            "p99_ttft_ms=500,mean_tpot_ms=30. If set, searches each "
            "combination for the highest request rate meeting them.",
        )
        group.add_argument(
            "--search-iters",
            type=int,
            default=5,
            help="Bisection steps of the request-rate search, after bracketing.",
        )
        group.add_argument(
            "--frontend-latency-ms",
            type=float,
            default=0.0,
            help="Per-request latency of the server outside the engine core "
            "(HTTP, API server, IPC), added to TTFT and end-to-end latency.",
        )
        group.add_argument(
            "--handoff-latency-ms",
            type=float,
            default=0.0,
            help="For disaggregated combinations (`_num_prefill`/`_num_decode`): "
            "the proxy's time from a prefill instance finishing a request to the "
            "decode instance receiving it.",
        )
        group.add_argument(
            "--fill-kv-caches",
            action="store_true",
            help="Before a combination's first run, use up each instance's KV "
            "cache once, so that the runs see a long-running server's "
            "scattered KV blocks.",
        )
        group.add_argument(
            "-j",
            "--num-workers",
            type=int,
            default=1,
            help="Number of combinations simulated in parallel.",
        )
        return parser


def _make_tasks(args: SweepSimulateArgs, experiment_dir: Path) -> list[_CombTask]:
    tasks: list[_CombTask] = []
    for serve_comb in args.serve_params:
        common, roles, num_instances = split_roles(serve_comb)
        serve_cmds = {
            role: simulated_serve_cmd(
                args.serve_cmd, args.simulation_config, common | roles.get(role, {})
            )
            for role in (num_instances or ["serve"])
        }
        for bench_comb in args.bench_params:
            if not _comb_is_valid(serve_comb, bench_comb, args.link_vars):
                continue
            comb_dir = _get_comb_base_path(experiment_dir, serve_comb, bench_comb)
            if args.resume and (
                (comb_dir / "result.json").exists()
                if args.slo
                else any(comb_dir.rglob("summary.json"))
            ):
                continue
            task = _CombTask(
                serve_cmds=serve_cmds,
                num_instances=num_instances,
                bench_cmd=bench_comb.apply_to_cmd(args.bench_cmd),
                serve_comb=serve_comb,
                bench_comb=bench_comb,
                comb_dir=comb_dir,
                slo=args.slo,
                search_iters=args.search_iters,
                frontend_latency_ms=args.frontend_latency_ms,
                handoff_latency_ms=args.handoff_latency_ms,
                fill_kv_caches=args.fill_kv_caches,
                index=len(tasks),
            )
            for role, cmd in serve_cmds.items():
                count = f" x{num_instances[role]}" if num_instances else ""
                print(f"[SIMULATE {role}{count}] {shlex.join(cmd)}")
            print(f"[BENCH] {shlex.join(task.bench_cmd)}")
            tasks.append(task)
    return tasks


def _run_tasks(args: SweepSimulateArgs, tasks: list[_CombTask]) -> None:
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


def _summarize(args: SweepSimulateArgs, experiment_dir: Path) -> list[dict]:
    all_data = [
        run_data
        for path in sorted(experiment_dir.rglob("summary.json"))
        for run_data in json.loads(path.read_text())
    ]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(all_data).to_csv(experiment_dir / "summary.csv")
    extrapolated = [
        run_data
        for run_data in all_data
        if run_data.get("timing_lookups_outside_range", 0)
        > _MAX_OUTSIDE_RANGE * run_data.get("timing_lookups", 0)
    ]
    if extrapolated:
        print(
            f"WARNING: {len(extrapolated)} of {len(all_data)} runs timed more than "
            f"{_MAX_OUTSIDE_RANGE:.0%} of their k-NN lookups beyond the recorded "
            "range, where step times are underestimated "
            "(timing_lookups_outside_range in their summary.json). Record the "
            "deployment under a load that covers the simulated one."
        )
    results = [
        json.loads(path.read_text())
        for path in sorted(experiment_dir.rglob("result.json"))
    ]
    if args.slo and results:
        df = pd.DataFrame.from_records(results)
        df["output_throughput_per_gpu"] = (
            df.get("output_throughput", 0) / df["gpu_count"]
        )
        if "timing_lookups" in df:
            df["timing_outside_range"] = (
                df["timing_lookups_outside_range"] / df["timing_lookups"]
            ).fillna(0.0)
        df = df.sort_values("output_throughput_per_gpu", ascending=False)
        df.to_csv(experiment_dir / "slo_summary.csv")
        params = {k for comb in (*args.serve_params, *args.bench_params) for k in comb}
        # The parameters that tell the combinations apart.
        columns = [c for c in df.columns if c in params and df[c].nunique() > 1]
        columns += ["gpu_count", "max_request_rate", *args.slo]
        columns += ["output_throughput_per_gpu", "timing_outside_range"]
        print(df[[c for c in columns if c in df.columns]].to_string(index=False))
    return all_data


def run_main(args: SweepSimulateArgs):
    experiment_dir = args.resolve_experiment_dir()
    with args.run_ctx(experiment_dir):
        tasks = _make_tasks(args, experiment_dir)
        if args.dry_run:
            return None
        _run_tasks(args, tasks)
        return _summarize(args, experiment_dir)


def main(args: argparse.Namespace):
    run_main(SweepSimulateArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=SweepSimulateArgs.parser_help)
    SweepSimulateArgs.add_cli_args(parser)

    main(parser.parse_args())
