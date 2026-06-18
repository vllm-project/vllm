# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark for Scheduler.update_from_output CPU overhead.

Measures per-step scheduler post-processing time for decode-only workloads
at various batch sizes. Useful for validating optimizations to the hot loop
in vllm/v1/core/sched/scheduler.py.

Example:
    .venv/bin/python benchmarks/overheads/benchmark_scheduler_update_from_output.py
    .venv/bin/python benchmarks/overheads/benchmark_scheduler_update_from_output.py \
        --num-requests 256 --warmup 50 --iters 200
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.v1.core.utils import create_requests, create_scheduler  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402


def _make_decode_model_runner_output(
    scheduler_output,
    token_id: int = 1,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[token_id] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
    )


def _setup_decode_batch(num_requests: int, prompt_tokens: int):
    scheduler = create_scheduler(
        max_num_seqs=num_requests,
        max_num_batched_tokens=max(num_requests * prompt_tokens, 8192),
        enable_prefix_caching=False,
    )
    requests = create_requests(
        num_requests=num_requests,
        num_tokens=prompt_tokens,
        max_tokens=64,
        ignore_eos=True,
    )
    for request in requests:
        scheduler.add_request(request)

    prefill_output = scheduler.schedule()
    scheduler.update_from_output(
        prefill_output,
        _make_decode_model_runner_output(prefill_output, token_id=0),
    )

    for request in requests:
        request.num_computed_tokens = len(request.prompt_token_ids)

    return scheduler, requests


def _bench_update_from_output(
    num_requests: int,
    prompt_tokens: int,
    warmup: int,
    iters: int,
) -> list[float]:
    scheduler, _ = _setup_decode_batch(num_requests, prompt_tokens)

    for _ in range(warmup):
        sched_out = scheduler.schedule()
        mro = _make_decode_model_runner_output(sched_out)
        scheduler.update_from_output(sched_out, mro)

    timings_ms: list[float] = []
    for _ in range(iters):
        sched_out = scheduler.schedule()
        mro = _make_decode_model_runner_output(sched_out)
        start = time.perf_counter()
        scheduler.update_from_output(sched_out, mro)
        timings_ms.append((time.perf_counter() - start) * 1000)

    return timings_ms


def _summarize(timings_ms: list[float]) -> str:
    return (
        f"median={statistics.median(timings_ms):.3f}ms "
        f"p95={sorted(timings_ms)[int(0.95 * len(timings_ms))]:.3f}ms "
        f"mean={statistics.mean(timings_ms):.3f}ms"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Scheduler.update_from_output decode overhead.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        nargs="+",
        default=[16, 64, 128, 256],
        help="Batch sizes to benchmark.",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=32,
        help="Prompt length per request (prefill done before timing).",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    print(
        "Scheduler.update_from_output decode micro-benchmark "
        f"(prompt_tokens={args.prompt_tokens}, warmup={args.warmup}, "
        f"iters={args.iters})"
    )
    print(f"{'batch':>8}  {'timings':>40}")
    print("-" * 52)
    for num_requests in args.num_requests:
        timings = _bench_update_from_output(
            num_requests=num_requests,
            prompt_tokens=args.prompt_tokens,
            warmup=args.warmup,
            iters=args.iters,
        )
        print(f"{num_requests:>8}  {_summarize(timings)}")


if __name__ == "__main__":
    main()
