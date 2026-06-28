# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stress SimpleCPUOffloadConnector under repeated CPU-hit batches.

This script is intended as a reproduction/measurement harness for scheduler
behavior under CPU KV offload. It first runs a cold batch to populate the CPU
offload cache, resets the GPU prefix cache, then repeatedly runs the same
prompts so subsequent batches should load prompt KV from CPU.

Example:
    python benchmarks/benchmark_cpu_offload_stress.py +        --model facebook/opt-125m +        --num-requests 8 +        --num-batches 5 +        --prompt-repeats 512 +        --max-num-seqs 2 +        --max-num-batched-tokens 1024 +        --stall-timeout-s 120
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import statistics
import sys
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class GenerateTimeoutError(TimeoutError):
    pass


@contextmanager
def alarm_timeout(seconds: float):
    if seconds <= 0:
        yield
        return

    def _handle_timeout(signum: int, frame: Any) -> None:
        raise GenerateTimeoutError(
            f"generation did not finish within {seconds:.1f} seconds"
        )

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def run_timed(
    label: str,
    timeout_s: float,
    fn: Callable[[], T],
) -> tuple[T, float]:
    start = time.perf_counter()
    try:
        with alarm_timeout(timeout_s):
            result = fn()
    except GenerateTimeoutError as exc:
        elapsed = time.perf_counter() - start
        print(
            json.dumps(
                {
                    "event": "stall_detected",
                    "label": label,
                    "elapsed_s": elapsed,
                    "timeout_s": timeout_s,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    elapsed = time.perf_counter() - start
    return result, elapsed


def build_prompts(num_requests: int, prompt_repeats: int) -> list[str]:
    prompts = []
    for idx in range(num_requests):
        group_idx = idx // 2
        prefix = f"Group {group_idx} " + ("hi " * prompt_repeats)
        prompts.append(f"{prefix} Request {idx}. Please answer with one short token.")

    return prompts




def output_texts(outputs: list[Any]) -> list[str]:
    return [out.outputs[0].text for out in outputs]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress SimpleCPUOffloadConnector with repeated CPU-hit batches."
    )
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--cpu-bytes-to-use", type=int, default=1 << 30)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--prompt-repeats", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--stall-timeout-s", type=float, default=120.0)
    parser.add_argument("--sleep-after-cold-s", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--disable-flashinfer-sampler",
        action="store_true",
        help="Set VLLM_USE_FLASHINFER_SAMPLER=0 before importing vLLM.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Enable vLLM engine stats logging when supported by the engine args.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.disable_flashinfer_sampler:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    # Make sure imports resolve to the checked-out vLLM repo.
    sys.path.insert(0, os.getcwd())

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    prompts = build_prompts(args.num_requests, args.prompt_repeats)
    kv_transfer_config = KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        engine_id=str(uuid.uuid4()),
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": args.cpu_bytes_to_use,
            "lazy_offload": False,
        },
    )

    print(
        json.dumps(
            {
                "event": "config",
                "model": args.model,
                "num_requests": args.num_requests,
                "num_batches": args.num_batches,
                "prompt_repeats": args.prompt_repeats,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_model_len": args.max_model_len,
                "cpu_bytes_to_use": args.cpu_bytes_to_use,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        kv_transfer_config=kv_transfer_config,
        disable_log_stats=not args.log_stats,
    )

    summary: dict[str, Any] = {}
    try:
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )

        cold_outputs, cold_s = run_timed(
            "cold_populate",
            args.stall_timeout_s,
            lambda: llm.generate(prompts, sampling_params, use_tqdm=False),
        )
        expected = output_texts(cold_outputs)
        print(
            json.dumps(
                {"event": "batch", "label": "cold_populate", "elapsed_s": cold_s},
                sort_keys=True,
            ),
            flush=True,
        )

        if args.sleep_after_cold_s > 0:
            time.sleep(args.sleep_after_cold_s)

        reset_ok = llm.reset_prefix_cache()
        print(
            json.dumps({"event": "reset_prefix_cache", "ok": reset_ok}),
            flush=True,
        )
        if not reset_ok:
            raise RuntimeError("reset_prefix_cache returned False")

        batch_latencies: list[float] = []
        mismatched_batches: list[int] = []
        for batch_idx in range(args.num_batches):
            outputs, elapsed_s = run_timed(
                f"cpu_hit_batch_{batch_idx}",
                args.stall_timeout_s,
                lambda: llm.generate(prompts, sampling_params, use_tqdm=False),
            )
            actual = output_texts(outputs)
            if actual != expected:
                mismatched_batches.append(batch_idx)
            batch_latencies.append(elapsed_s)
            print(
                json.dumps(
                    {
                        "event": "batch",
                        "label": "cpu_hit",
                        "batch_idx": batch_idx,
                        "elapsed_s": elapsed_s,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        summary = {
            "status": "pass" if not mismatched_batches else "output_mismatch",
            "cold_s": cold_s,
            "num_batches": args.num_batches,
            "num_requests": args.num_requests,
            "batch_latency_s": batch_latencies,
            "mean_batch_latency_s": statistics.mean(batch_latencies),
            "p50_batch_latency_s": percentile(batch_latencies, 50),
            "p95_batch_latency_s": percentile(batch_latencies, 95),
            "max_batch_latency_s": max(batch_latencies),
            "mismatched_batches": mismatched_batches,
        }
        print(json.dumps({"event": "summary", **summary}, sort_keys=True), flush=True)
        if args.json_output:
            args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "pass" else 1
    except GenerateTimeoutError:
        return 2
    finally:
        del llm
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
