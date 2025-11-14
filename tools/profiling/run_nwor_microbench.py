#!/usr/bin/env python3
"""
NWOR microbenchmark harness for speculative decoding.

Example:
  python tools/profiling/run_nwor_microbench.py \
      --scenario short --batches 4 --requests 8 --draft-tokens 4 \
      --temperature 0.0 --output results.json

Environment overrides:
  TARGET_MODEL=... DRAFT_MODEL=... python ...
"""

import argparse
import gc
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, List

from datasets import load_dataset

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter as MetricCounter, Gauge as MetricGauge
from vllm.v1.metrics.reader import Vector as MetricVector


DEFAULT_TARGET_MODEL = os.getenv(
    "TARGET_MODEL", "meta-llama/Llama-3.2-3B-Instruct"
)
DEFAULT_DRAFT_MODEL = os.getenv(
    "DRAFT_MODEL", "linborui/EAGLE-Llama-3.2-3B-Instruct"
)

SCENARIOS = {
    "short": dict(
        dataset="OpenAssistant/oasst1",
        split="train",
        fields=["prompt", "text", "instruction"],
        min_chars=1,
        max_chars=800,
    ),
    "medium": dict(
        dataset="abisee/cnn_dailymail",
        name="3.0.0",
        split="train",
        fields=["article", "text"],
        min_chars=800,
        max_chars=2000,
    ),
    "long": dict(
        dataset="abisee/cnn_dailymail",
        name="3.0.0",
        split="train",
        fields=["article", "text"],
        min_chars=2000,
        max_chars=None,
    ),
    "mixed": dict(
        dataset="Open-Orca/OpenOrca",
        split="train",
        fields=["text", "response", "output"],
        min_chars=1,
        max_chars=None,
    ),
}


@dataclass
class RunConfig:
    target_model: str
    drafter_model: str
    scenario: str
    num_requests: int
    draft_tokens: int
    batches: int
    temperature: float
    top_p: float
    tensor_parallel_size: int
    prompt_count: int
    prompt_shuffle_seed: int
    max_model_len: int | None
    max_new_tokens: int
    warmup_steps: int
    measure_steps: int
    spec_method: str
    nwor_modes: List[str]
    scv_modes: List[str]
    enable_ncu: bool
    ncu_metrics: str
    enable_nsys: bool
    profile_only: bool
    output_path: str


def pick_prompts(config: RunConfig) -> List[str]:
    info = SCENARIOS[config.scenario]
    ds = load_dataset(
        info["dataset"],
        info.get("name"),
        split=info["split"],
    )
    min_chars = info.get("min_chars") or 0
    max_chars = info.get("max_chars") or 1_000_000

    candidates = []
    for record in ds:
        texts: List[str] = []
        for field in info["fields"]:
            value = record.get(field)
            if isinstance(value, str):
                texts.append(value)
        if not texts:
            continue
        text = "\n".join(t.strip() for t in texts if t)
        if min_chars <= len(text) <= max_chars:
            candidates.append(text)
        if len(candidates) >= config.prompt_count * config.num_requests:
            break

    if not candidates:
        raise RuntimeError(
            f"No prompts found for scenario '{config.scenario}'. "
            "Consider lowering min/max char filters."
        )

    random.seed(config.prompt_shuffle_seed)
    random.shuffle(candidates)
    total_needed = (config.warmup_steps + config.batches) * config.num_requests
    if len(candidates) < total_needed:
        raise RuntimeError(
            f"Not enough prompts ({len(candidates)}) for warmup + measurement "
            f"needs ({total_needed}). Increase --prompt-count or adjust batching."
        )
    return candidates[:total_needed]


def build_engine(config: RunConfig) -> LLM:
    speculative_config = {
        "method": config.spec_method,
        "model": config.drafter_model,
        "num_speculative_tokens": config.draft_tokens,
    }
    llm_kwargs: dict[str, Any] = {
        "model": config.target_model,
        "tensor_parallel_size": config.tensor_parallel_size,
        "speculative_config": speculative_config,
        # Enable Prometheus stats so NWOR metrics appear in microbench output.
        "disable_log_stats": False,
    }
    if config.max_model_len is not None:
        llm_kwargs["max_model_len"] = config.max_model_len
    return LLM(**llm_kwargs)


def run_batch(
    engine: LLM,
    prompts: Iterable[str],
    config: RunConfig,
    nwor_mode: str,
    batch_index: int,
    scv_mode: str,
) -> dict[str, Any]:
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
    )

    prompt_list = list(prompts)
    start = time.time()
    request_outputs = engine.generate(prompt_list, sampling_params=sampling_params, use_tqdm=False)
    duration = time.time() - start

    texts = [
        output.outputs[0].text if output.outputs else ""
        for output in request_outputs
    ]

    return {
        "nwor_mode": nwor_mode,
        "scv_mode": scv_mode,
        "batch_index": batch_index,
        "latency_s": duration,
        "outputs": texts,
        "sampling_params": {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_tokens,
        },
    }


def snapshot_metrics(engine: LLM | None = None) -> dict[str, float | list[int]]:
    totals: dict[str, float | list[int]] = defaultdict(float)
    metrics = engine.get_metrics() if engine is not None else []
    if engine is None:
        # Fallback path if an engine handle is not available.
        try:
            from vllm.v1.metrics.reader import get_metrics_snapshot  # type: ignore
        except ImportError:
            metrics = []
        else:
            metrics = get_metrics_snapshot()

    for metric in metrics:
        if isinstance(metric, MetricCounter):
            totals[metric.name] += metric.value
        elif isinstance(metric, MetricGauge):
            totals[metric.name] += metric.value
        elif isinstance(metric, MetricVector):
            if metric.name not in totals:
                totals[metric.name] = [0] * len(metric.values)
            current = totals[metric.name]
            assert isinstance(current, list)
            for idx, val in enumerate(metric.values):
                current[idx] += val
    return totals


def diff_metrics(
    after: dict[str, float | list[int]],
    before: dict[str, float | list[int]],
) -> dict[str, float]:
    diff: dict[str, float] = {}
    keys = set(before.keys()) | set(after.keys())
    for name in keys:
        after_val = after.get(name)
        before_val = before.get(name)
        if isinstance(after_val, list) or isinstance(before_val, list):
            # Skip vector metrics for now.
            continue
        base_value = float(after_val or 0.0) - float(before_val or 0.0)
        diff[name] = base_value
        if name.endswith("_total"):
            base_name = name[: -len("_total")]
            diff.setdefault(base_name, base_value)
    return diff


def run_microbenchmark(config: RunConfig) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, float]]]:
    prompts = pick_prompts(config)
    results: list[dict[str, Any]] = []
    metrics_delta: dict[tuple[str, str], dict[str, float]] = {}

    for scv_mode in config.scv_modes:
        os.environ["VLLM_SCV_MODE"] = scv_mode or "off"

        for nwor_mode in config.nwor_modes:
            # Backward compatibility: translate nwor_mode to new env vars
            if nwor_mode == "off":
                os.environ["VLLM_NWOR_ADAPTIVE_DRAFT_LENGTH"] = "0"
                os.environ["VLLM_NWOR_CONFIDENCE_THRESHOLD"] = "0.0"
            else:  # "on" or default
                os.environ["VLLM_NWOR_ADAPTIVE_DRAFT_LENGTH"] = "1"
                os.environ["VLLM_NWOR_CONFIDENCE_THRESHOLD"] = "0.5"
            engine = build_engine(config)

            prompt_offset = 0
            # Warmup (not recorded)
            for _ in range(config.warmup_steps):
                warm_prompts = prompts[prompt_offset : prompt_offset + config.num_requests]
                prompt_offset += config.num_requests
                run_batch(engine, warm_prompts, config, nwor_mode, -1, scv_mode)

            metrics_before = snapshot_metrics(engine)

            for batch_idx in range(config.batches):
                start = prompt_offset + batch_idx * config.num_requests
                end = start + config.num_requests
                batch_prompts = prompts[start:end]
                result = run_batch(
                    engine, batch_prompts, config, nwor_mode, batch_idx, scv_mode
                )
                results.append(result)

            metrics_after = snapshot_metrics(engine)
            delta = diff_metrics(metrics_after, metrics_before)
            metrics_delta[(scv_mode, nwor_mode)] = delta

            # Explicitly delete engine to free GPU memory before next iteration
            del engine
            gc.collect()

    return results, metrics_delta


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="NWOR microbenchmark harness")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="short")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--prompt-count", type=int, default=100)
    parser.add_argument("--prompt-shuffle-seed", type=int, default=1234)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=1)
    parser.add_argument(
        "--nwor-modes",
        default="off,stage",
        help="Comma-separated list of NWOR modes to benchmark (default: off,stage)",
    )
    parser.add_argument(
        "--scv-modes",
        default="off",
        help="Comma-separated list of SCV modes to benchmark (default: off)",
    )
    parser.add_argument(
        "--spec-method",
        default="eagle",
        help="Speculative method to use (default: eagle).",
    )
    parser.add_argument(
        "--enable-ncu",
        action="store_true",
        help="Run an additional pass under Nsight Compute (nv-nsight-cu-cli).",
    )
    parser.add_argument(
        "--ncu-metrics",
        default="dram__bytes_write.sum,lts__t_sectors_op_write.sum",
        help="Comma-separated Nsight Compute metrics to collect when --enable-ncu is set.",
    )
    parser.add_argument(
        "--enable-nsys",
        action="store_true",
        help="Run an additional pass under Nsight Systems.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output", default="nwor_microbench.json")
    args = parser.parse_args()

    nwor_modes = [mode.strip() for mode in args.nwor_modes.split(",") if mode.strip()]
    scv_modes = [mode.strip() for mode in args.scv_modes.split(",") if mode.strip()]

    return RunConfig(
        target_model=args.target_model,
        drafter_model=args.draft_model,
        scenario=args.scenario,
        num_requests=args.requests,
        draft_tokens=args.draft_tokens,
        batches=args.batches,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        prompt_count=args.prompt_count,
        prompt_shuffle_seed=args.prompt_shuffle_seed,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        spec_method=args.spec_method,
        nwor_modes=nwor_modes or ["off"],
        scv_modes=scv_modes or ["off"],
        enable_ncu=args.enable_ncu,
        ncu_metrics=args.ncu_metrics,
        enable_nsys=args.enable_nsys,
        profile_only=args.profile_only,
        output_path=args.output,
    )


def summarize_results(
    results: list[dict[str, Any]],
    metrics_delta: dict[tuple[str, str], dict[str, float]],
    ncu_metrics: dict[tuple[str, str], dict[str, float]] | None = None,
) -> dict[str, Any]:
    summary: dict[tuple[str, str], dict[str, Any]] = {}

    for result in results:
        key = (result["scv_mode"], result["nwor_mode"])
        entry = summary.setdefault(
            key,
            {
                "latencies": [],
                "batches": 0,
            },
        )
        entry["latencies"].append(result["latency_s"])
        entry["batches"] += 1

    summary_output = []
    for (scv_mode, nwor_mode), entry in summary.items():
        latencies = entry["latencies"]
        latency_avg = statistics.mean(latencies) if latencies else 0.0
        if len(latencies) >= 2:
            p50 = statistics.quantiles(latencies, n=100, method="inclusive")[49]
            p95 = statistics.quantiles(latencies, n=100, method="inclusive")[94]
        else:
            p50 = latencies[0] if latencies else 0.0
            p95 = p50

        metrics = metrics_delta.get((scv_mode, nwor_mode), {})
        committed = int(
            metrics.get(
                "vllm:nwor_committed_tokens",
                metrics.get("vllm:nwor_committed_tokens_total", 0),
            )
        )
        rejected = int(
            metrics.get(
                "vllm:nwor_rejected_tokens",
                metrics.get("vllm:nwor_rejected_tokens_total", 0),
            )
        )
        staged = committed + rejected
        writes_saved_pct = (
            (1 - committed / staged) * 100.0 if staged > 0 else 0.0
        )

        spec_drafts = int(metrics.get("vllm:spec_decode_num_drafts", 0))
        spec_draft_tokens = int(metrics.get("vllm:spec_decode_num_draft_tokens", 0))
        spec_accepted_tokens = int(metrics.get("vllm:spec_decode_num_accepted_tokens", 0))
        avg_acceptance_per_window = (
            spec_accepted_tokens / spec_drafts if spec_drafts > 0 else 0.0
        )
        acceptance_ratio = (
            spec_accepted_tokens / spec_draft_tokens
            if spec_draft_tokens > 0
            else 0.0
        )

        metrics_extra = (ncu_metrics or {}).get((scv_mode, nwor_mode), {})
        summary_output.append(
            {
                "scv_mode": scv_mode,
                "nwor_mode": nwor_mode,
                "batches": entry["batches"],
                "latency_avg_s": latency_avg,
                "latency_p50_s": p50,
                "latency_p95_s": p95,
                "nwor_tokens_committed": committed,
                "nwor_tokens_staged": staged,
                "nwor_writes_saved_pct": writes_saved_pct,
                "spec_num_drafts": spec_drafts,
                "spec_num_draft_tokens": spec_draft_tokens,
                "spec_num_accepted_tokens": spec_accepted_tokens,
                "spec_avg_accepted_per_window": avg_acceptance_per_window,
                "spec_acceptance_ratio": acceptance_ratio,
                "ncu_metrics": metrics_extra,
            }
        )

    return {"per_mode": summary_output}


def write_markdown_summary(config: RunConfig, summary: dict[str, Any], path: Path) -> None:
    lines = []
    lines.append(f"# NWOR/SCV Microbenchmark\n")
    lines.append("## Configuration\n")
    lines.append("```json")
    lines.append(json.dumps(config.__dict__, indent=2))
    lines.append("```")
    lines.append("\n## Summary\n")
    # Determine optional NCU metric columns
    metric_names: list[str] = []
    for row in summary["per_mode"]:
        for metric_name in row.get("ncu_metrics", {}):
            if metric_name not in metric_names:
                metric_names.append(metric_name)

    header_cols = [
        "SCV Mode",
        "NWOR Mode",
        "Batches",
        "Avg Latency (s)",
        "P50 (s)",
        "P95 (s)",
        "Tokens Staged",
        "Tokens Committed",
        "Writes Saved %",
        "Avg Accepted/window",
        "Acceptance Ratio",
    ] + metric_names
    header = "| " + " | ".join(header_cols) + " |"
    separator = "| " + " | ".join("---" for _ in header_cols) + " |"
    lines.append(header)
    lines.append(separator)
    for row in summary["per_mode"]:
        values = [
            row["scv_mode"],
            row["nwor_mode"],
            str(row["batches"]),
            f"{row['latency_avg_s']:.4f}",
            f"{row['latency_p50_s']:.4f}",
            f"{row['latency_p95_s']:.4f}",
            str(row["nwor_tokens_staged"]),
            str(row["nwor_tokens_committed"]),
            f"{row['nwor_writes_saved_pct']:.2f}",
            f"{row['spec_avg_accepted_per_window']:.2f}",
            f"{row['spec_acceptance_ratio']:.2f}",
        ]
        metrics_extra = row.get("ncu_metrics", {})
        for name in metric_names:
            value = metrics_extra.get(name)
            values.append(f"{value:.3e}" if value is not None else "")
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def config_to_args(
    config: RunConfig,
    *,
    output_path: str,
    profile_only: bool = False,
    override_modes: tuple[str, str] | None = None,
) -> list[str]:
    args = [
        "--target-model",
        config.target_model,
        "--draft-model",
        config.drafter_model,
        "--scenario",
        config.scenario,
        "--requests",
        str(config.num_requests),
        "--draft-tokens",
        str(config.draft_tokens),
        "--batches",
        str(config.batches),
        "--temperature",
        str(config.temperature),
        "--top-p",
        str(config.top_p),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--prompt-count",
        str(config.prompt_count),
        "--prompt-shuffle-seed",
        str(config.prompt_shuffle_seed),
    ]
    if config.max_model_len is not None:
        args.extend(["--max-model-len", str(config.max_model_len)])
    args.extend([
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--warmup-steps",
        str(config.warmup_steps),
        "--measure-steps",
        str(config.measure_steps),
        "--nwor-modes",
        ",".join(override_modes and [override_modes[1]] or config.nwor_modes),
        "--scv-modes",
        ",".join(override_modes and [override_modes[0]] or config.scv_modes),
        "--output",
        output_path,
    ])
    if profile_only:
        args.append("--profile-only")
    return args


def run_ncu_profiles(config: RunConfig, output_json: Path) -> dict[tuple[str, str], dict[str, float]]:
    metrics_map: dict[tuple[str, str], dict[str, float]] = {}
    script_path = Path(__file__).resolve()
    env = os.environ.copy()
    metric_names = [m.strip() for m in config.ncu_metrics.split(",") if m.strip()]

    for scv_mode in config.scv_modes:
        for nwor_mode in config.nwor_modes:
            suffix = f".{scv_mode or 'off'}-{nwor_mode or 'off'}"
            csv_path = output_json.with_suffix(f"{suffix}.ncu.csv")
            rep_path = output_json.with_suffix(f"{suffix}.ncu")
            profile_json = output_json.with_suffix(f"{suffix}.ncu.json")
            args = config_to_args(
                config,
                output_path=str(profile_json),
                profile_only=True,
                override_modes=(scv_mode, nwor_mode),
            )
            # Try ncu first (modern CUDA), fallback to nv-nsight-cu-cli (older)
            ncu_cmd = "ncu" if shutil.which("ncu") else "nv-nsight-cu-cli"
            cmd = [
                ncu_cmd,
                "-f",  # Force overwrite existing report files
                "--csv",
                "--log-file",
                str(csv_path),
                "--metrics",
                ",".join(metric_names),
                "--target-processes",
                "all",
                "-o",
                str(rep_path),
                sys.executable,
                str(script_path),
            ] + args
            try:
                subprocess.run(cmd, check=True, env=env)
            except FileNotFoundError as exc:
                print(f"[WARN] {ncu_cmd} not found: {exc}. Skipping NCU collection.")
                return {}
            except subprocess.CalledProcessError as exc:
                print(f"[WARN] nv-nsight-cu-cli failed for modes {scv_mode}/{nwor_mode}: {exc}")
                continue

            metrics = parse_ncu_csv(csv_path, metric_names)
            metrics_map[(scv_mode, nwor_mode)] = metrics
    return metrics_map


def parse_ncu_csv(path: Path, metric_names: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            name, _unit, value = parts[:3]
            if name in metric_names:
                try:
                    metrics[name] = float(value)
                except ValueError:
                    pass
    return metrics


def main() -> None:
    config = parse_args()
    results, metrics_delta = run_microbenchmark(config)
    ncu_metrics_map: dict[tuple[str, str], dict[str, float]] | None = None
    output_json = Path(config.output_path)

    if config.enable_ncu and not config.profile_only:
        ncu_metrics_map = run_ncu_profiles(config, output_json)

    summary = summarize_results(results, metrics_delta, ncu_metrics=ncu_metrics_map)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config.__dict__,
                "summary": summary,
                "results": results,
            },
            f,
            indent=2,
        )

    output_md = output_json.with_suffix(".md")
    write_markdown_summary(config, summary, output_md)
    print(f"Wrote benchmark output to {output_json} and {output_md}")

    if config.enable_nsys and not config.profile_only:
        # Run Nsight Systems once over all modes
        script_path = Path(__file__).resolve()
        env = os.environ.copy()
        nsys_output = output_json.with_suffix(".nsys")
        args = config_to_args(
            config,
            output_path=str(output_json.with_suffix(".nsys.json")),
            profile_only=True,
        )
        cmd = [
            "nsys",
            "profile",
            "-t",
            "cuda,nvtx,osrt",
            "-o",
            str(nsys_output),
            sys.executable,
            str(script_path),
        ] + args
        try:
            subprocess.run(cmd, check=True, env=env)
        except FileNotFoundError as exc:
            print(f"[WARN] nsys not found: {exc}. Skipping Nsight Systems collection.")
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] nsys failed: {exc}")


if __name__ == "__main__":
    main()
