# SPDX-License-Identifier: Apache-2.0
"""Drive a concurrency sweep against an already-running vLLM server.

Usage:
    python -m bench.sweep --label baseline

The server is launched once, at its maximum batch size, and only the *client*
concurrency is swept. That keeps a single engine configuration (one set of CUDA
graphs, one KV cache) under test across the whole curve, which is what makes
the points comparable to each other and to a later optimized run.
"""

from __future__ import annotations

import argparse
import sys
import time

from bench import report
from bench.aiperf import (
    AiperfNotInstalled,
    aiperf_help,
    build_aiperf_command,
    run_concurrency_point,
    streaming_runner,
)
from bench.collect import collect_run, load_export, parse_export
from bench.config import DEFAULT_CONCURRENCIES, RunManifest, SweepConfig, Workload
from bench.server import (
    audit_perf_features,
    fetch_server_info,
    reset_prefix_cache,
    wait_for_server,
)

_STARTED = time.monotonic()


def log(message: str, stream=None) -> None:
    """Timestamped, flushed progress line.

    A sweep spends minutes per point inside aiperf. Without elapsed-time
    markers a healthy run is indistinguishable from a hang, and flushing is
    mandatory because stdout is a pipe under Modal and CI.
    """
    # Resolved per call, not as a default: a default would bind the stdout
    # object at import time and ignore any later redirection.
    mins, secs = divmod(time.monotonic() - _STARTED, 60)
    print(
        f"[{int(mins):02d}:{secs:04.1f}] {message}",
        file=stream or sys.stdout,
        flush=True,
    )


def estimate_seconds(workload: Workload, sweep: SweepConfig, concurrency: int) -> float:
    """Rough wall-clock estimate for one point, for progress reporting only.

    Assumes decode dominates and per-user speed degrades slowly with batch
    size; accurate enough to tell "slow but fine" from "stuck".
    """
    requests = sweep.request_count(concurrency) + sweep.warmup_count(concurrency)
    per_user_tps = max(5.0, 45.0 / (1 + concurrency / 64))
    seconds_per_request = workload.osl / per_user_tps
    return requests * seconds_per_request / concurrency


def build_parser() -> argparse.ArgumentParser:
    workload = Workload.from_env()
    sweep = SweepConfig.from_env()
    parser = argparse.ArgumentParser(
        description="vLLM + aiperf concurrency sweep -> Pareto curve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=workload.model)
    parser.add_argument("--tokenizer", default=workload.tokenizer)
    parser.add_argument("--isl", type=int, default=workload.isl)
    parser.add_argument("--osl", type=int, default=workload.osl)
    parser.add_argument("--num-gpus", type=int, default=workload.num_gpus)
    parser.add_argument("--random-seed", type=int, default=workload.random_seed)
    parser.add_argument(
        "--no-ignore-eos",
        dest="ignore_eos",
        action="store_false",
        default=workload.ignore_eos,
        help="let the model stop early instead of pinning OSL",
    )
    parser.add_argument("--url", default=sweep.url)
    parser.add_argument("--label", default=sweep.label)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=list(sweep.concurrencies or DEFAULT_CONCURRENCIES),
    )
    parser.add_argument(
        "--requests-per-concurrency", type=int, default=sweep.requests_per_concurrency
    )
    parser.add_argument("--min-requests", type=int, default=sweep.min_requests)
    parser.add_argument("--max-requests", type=int, default=sweep.max_requests)
    parser.add_argument("--warmup-requests", type=int, default=sweep.warmup_requests)
    parser.add_argument(
        "--benchmark-duration", type=float, default=sweep.benchmark_duration
    )
    parser.add_argument("--artifact-root", default=sweep.artifact_root)
    parser.add_argument("--aiperf-bin", default=sweep.aiperf_bin)
    parser.add_argument("--ui", default=sweep.ui)
    parser.add_argument("--extra-aiperf-args", default=sweep.extra_aiperf_args)
    parser.add_argument(
        "--keep-prefix-cache",
        dest="reset_prefix_cache",
        action="store_false",
        default=sweep.reset_prefix_cache,
        help="do not flush the server prefix cache between points",
    )
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=1800.0,
        help="seconds to wait for /health",
    )
    parser.add_argument(
        "--allow-degraded-server",
        action="store_true",
        help="proceed even if the perf-feature audit finds problems",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the aiperf commands without touching the server",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="re-render CSV/plot/markdown from artifacts already on disk",
    )
    return parser


def configs_from_args(args: argparse.Namespace) -> tuple[Workload, SweepConfig]:
    workload = Workload(
        model=args.model,
        tokenizer=args.tokenizer,
        isl=args.isl,
        osl=args.osl,
        ignore_eos=args.ignore_eos,
        random_seed=args.random_seed,
        num_gpus=args.num_gpus,
    )
    sweep = SweepConfig(
        url=args.url,
        label=args.label,
        concurrencies=tuple(args.concurrency),
        requests_per_concurrency=args.requests_per_concurrency,
        min_requests=args.min_requests,
        max_requests=args.max_requests,
        warmup_requests=args.warmup_requests,
        benchmark_duration=args.benchmark_duration,
        reset_prefix_cache=args.reset_prefix_cache,
        artifact_root=args.artifact_root,
        aiperf_bin=args.aiperf_bin,
        ui=args.ui,
        extra_aiperf_args=args.extra_aiperf_args,
    )
    return workload, sweep


def report_kv_capacity(audit, workload: Workload, sweep: SweepConfig) -> float | None:
    """Print the engine's real KV capacity and flag queue-bound sweep points.

    For a 30B MoE in bf16 the weights leave little room for KV cache, so the
    high-concurrency end of the sweep can exceed what the engine can hold
    resident. Those points still measure something real, but it is the
    scheduler queue rather than the engine, and they must be labelled as such.
    """
    seq_len = workload.isl + workload.osl
    capacity = audit.servable_concurrency(seq_len)
    if capacity is None:
        return None

    print(
        f"[server] KV cache holds {audit.kv_cache_size_tokens:,} tokens "
        f"= {capacity:.0f} resident requests at ISL+OSL={seq_len}",
        flush=True,
    )
    queue_bound = [c for c in sweep.concurrencies if c > capacity]
    if queue_bound:
        print(
            f"[warn] concurrency {queue_bound} exceeds KV capacity "
            f"({capacity:.0f}); those points measure the queue, not the engine. "
            "Lower the sweep, raise --gpu-memory-utilization, or use a bigger GPU.",
            file=sys.stderr,
            flush=True,
        )
    return capacity


def audit_or_exit(
    workload: Workload, sweep: SweepConfig, allow_degraded: bool
) -> dict | None:
    """Fetch /server_info and refuse to benchmark a degraded server."""
    server_info = fetch_server_info(sweep.url)
    if server_info is None:
        print(
            "[warn] /server_info unavailable - launch the server with "
            "VLLM_SERVER_DEV_MODE=1 to record the effective config",
            file=sys.stderr,
            flush=True,
        )
        return None

    audit = audit_perf_features(server_info)
    print("[server] effective configuration:")
    for key, value in vars(audit).items():
        print(f"           {key} = {value}")
    sys.stdout.flush()
    report_kv_capacity(audit, workload, sweep)

    problems = audit.problems(max_concurrency=max(sweep.concurrencies))
    if problems:
        for problem in problems:
            print(f"[warn] {problem}", file=sys.stderr, flush=True)
        if not allow_degraded:
            raise SystemExit(
                "refusing to benchmark a degraded server; fix the flags above or "
                "pass --allow-degraded-server if this is intentional"
            )
    else:
        print("[server] perf-feature audit passed", flush=True)
    return server_info


def render_report(workload: Workload, sweep: SweepConfig) -> None:
    points = collect_run(
        sweep.run_dir / workload.slug, label=sweep.label, num_gpus=workload.num_gpus
    )
    if not points:
        print(
            f"[warn] no results under {sweep.run_dir / workload.slug}", file=sys.stderr
        )
        return
    out_dir = sweep.run_dir
    csv_path = report.write_csv(points, out_dir / "pareto.csv")
    for problem in report.integrity_warnings(points):
        log(f"[warn] {problem}", stream=sys.stderr)
    md = report.summarize(points)
    md_path = out_dir / "summary.md"
    md_path.write_text(
        f"# {sweep.label}: {workload.model} "
        f"ISL={workload.isl} OSL={workload.osl} TP={workload.num_gpus}\n\n{md}\n"
    )
    plot_path = report.plot_pareto(
        {sweep.label: points},
        out_dir / "pareto.png",
        title=f"{workload.model}  ISL={workload.isl} OSL={workload.osl}",
    )
    print("\n" + md + "\n")
    print(f"[out] {csv_path}")
    print(f"[out] {md_path}")
    print(f"[out] {plot_path}" if plot_path else "[warn] matplotlib missing, no plot")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workload, sweep = configs_from_args(args)

    if args.analyze_only:
        render_report(workload, sweep)
        return 0

    try:
        help_text = aiperf_help(sweep.aiperf_bin)
    except AiperfNotInstalled as exc:
        if not args.dry_run:
            raise
        print(f"[warn] {exc}; rendering commands with default flag names")
        help_text = None

    log(f"[aiperf] resolved flags against {sweep.aiperf_bin}")
    if args.dry_run:
        for concurrency in sweep.concurrencies:
            cmd = build_aiperf_command(workload, sweep, concurrency, help_text)
            print(" ".join(cmd))
        return 0

    log(f"[server] waiting for {sweep.url}")
    waited = wait_for_server(
        sweep.url,
        timeout_s=args.server_timeout,
        on_wait=lambda seconds: log(f"[server] not up yet ({seconds:.0f}s)"),
    )
    log(f"[server] healthy after {waited:.1f}s")
    server_info = audit_or_exit(workload, sweep, args.allow_degraded_server)

    manifest = RunManifest.create(workload, sweep)
    manifest.server_info = server_info
    failures: list[int] = []
    points_done: list[int] = []
    total_estimate = sum(
        estimate_seconds(workload, sweep, c) for c in sweep.concurrencies
    )
    log(
        f"[plan] {len(sweep.concurrencies)} points {list(sweep.concurrencies)} "
        f"~{total_estimate / 60:.0f} min total"
    )

    for concurrency in sweep.concurrencies:
        if sweep.reset_prefix_cache and not reset_prefix_cache(sweep.url):
            print(
                "[warn] could not reset prefix cache (needs VLLM_SERVER_DEV_MODE=1)",
                file=sys.stderr,
            )
        estimate = estimate_seconds(workload, sweep, concurrency)
        log(
            f"[run] concurrency={concurrency} "
            f"requests={sweep.request_count(concurrency)} "
            f"(+{sweep.warmup_count(concurrency)} warmup) ~{estimate:.0f}s expected"
        )
        point_started = time.monotonic()
        result = run_concurrency_point(
            workload,
            sweep,
            concurrency,
            help_text,
            runner=streaming_runner(lambda line: log(f"  | {line}")),
        )
        elapsed = time.monotonic() - point_started
        manifest.commands.append(result.command)
        if not result.ok:
            failures.append(concurrency)
            log(
                f"[fail] concurrency={concurrency} rc={result.returncode} "
                f"after {elapsed:.0f}s export={result.export_json}",
                stream=sys.stderr,
            )
            continue
        point = parse_export(
            load_export(result.export_json),
            concurrency=concurrency,
            label=sweep.label,
            num_gpus=workload.num_gpus,
        )
        log(
            f"[ok]   concurrency={concurrency} in {elapsed:.0f}s | "
            f"{point.tokens_per_s_per_user:.1f} tok/s/user | "
            f"{point.tokens_per_s_per_gpu:.0f} tok/s/gpu | "
            f"TTFT {point.ttft_ms:.0f}ms | OSL {point.output_sequence_length} | "
            f"{len(points_done) + 1}/{len(sweep.concurrencies)} points"
        )
        points_done.append(concurrency)
        if point.error_rate > 0:
            log(f"[warn] error rate {point.error_rate:.2%}", stream=sys.stderr)

    manifest.write(sweep.run_dir / "manifest.json")
    render_report(workload, sweep)

    if failures:
        print(f"[fail] concurrencies failed: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
