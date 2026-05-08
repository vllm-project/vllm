# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for KV connector early-prefetch (#41784).

Reproduces the failure mode jiangxiaosheng described in the issue: a saturated
running queue prevents the waiting-queue scheduling loop from ever invoking
the connector for newly arrived requests, so async lookups never start until
the running queue drains, leaving the GPU idle waiting on the just-started
lookup.

The benchmark uses a synthetic `SlowMockKVConnector` that simulates a
disk-backed KV store with a fixed wall-clock lookup latency. We:

  1) Drown the engine in long-decode "warmup" requests that saturate
     `--max-num-batched-tokens`.
  2) Slightly later, submit short "probe" requests and measure their TTFT.

With the prefetch optimization disabled (``--budget 0``), each probe pays
the full simulated lookup latency right before its first token. With the
optimization enabled (``--budget`` >= total prompt-token sum of probes),
the lookup timer starts at the top of the next schedule step rather than
when the probe finally reaches the head of the waiting queue, hiding the
latency behind ongoing GPU work.

Run from the repo root:

    .venv/bin/python -m benchmarks.kv_connector_prefetch.run_microbench \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --max-num-batched-tokens 512 \\
        --num-warmup 6 --num-probe 12 \\
        --latency-ms 200 \\
        --compare

The ``--compare`` flag runs the benchmark twice (budget=0 then
budget=large) and prints a side-by-side delta. Without ``--compare`` the
benchmark runs once with ``--budget``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics
import time
import uuid
from dataclasses import dataclass

from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

SLOW_MOCK_MODULE = "benchmarks.kv_connector_prefetch.slow_mock_connector"


@dataclass
class ProbeResult:
    request_id: str
    submit_time: float
    first_token_time: float
    finish_time: float

    @property
    def ttft(self) -> float:
        return self.first_token_time - self.submit_time

    @property
    def e2e(self) -> float:
        return self.finish_time - self.submit_time


def _make_engine(args: argparse.Namespace, budget: int) -> AsyncLLM:
    kv_transfer_config = KVTransferConfig(
        kv_connector="SlowMockKVConnector",
        kv_role="kv_both",
        kv_connector_module_path=SLOW_MOCK_MODULE,
    )

    engine_args = AsyncEngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
        enable_prefix_caching=False,
        kv_transfer_config=kv_transfer_config,
        kv_connector_prefetch_token_budget=budget,
    )
    return AsyncLLM.from_engine_args(engine_args, usage_context=UsageContext.LLM_CLASS)


_RNG = random.Random(0xC0FFEE)


def _build_prompt(num_tokens: int) -> TokensPrompt:
    # Random token ids give us exact length control (no tokenizer variance)
    # and cheap uniqueness across requests. We avoid id 0/1/2 which tend
    # to be special tokens in most tokenizers.
    token_ids = [_RNG.randint(100, 30000) for _ in range(num_tokens)]
    return TokensPrompt(prompt_token_ids=token_ids)


async def _run_request(
    engine: AsyncLLM,
    prompt: TokensPrompt,
    sampling: SamplingParams,
    submit_time: float,
) -> ProbeResult:
    rid = str(uuid.uuid4())
    first_token_time: float | None = None
    async for output in engine.generate(
        prompt=prompt, sampling_params=sampling, request_id=rid
    ):
        if first_token_time is None and output.outputs and output.outputs[0].token_ids:
            first_token_time = time.perf_counter()
        if output.finished:
            finish_time = time.perf_counter()
            assert first_token_time is not None
            return ProbeResult(
                request_id=rid,
                submit_time=submit_time,
                first_token_time=first_token_time,
                finish_time=finish_time,
            )
    raise RuntimeError(f"request {rid} ended without `finished`")


async def _drive_one(engine: AsyncLLM, args: argparse.Namespace) -> list[ProbeResult]:
    # Phase 1: kick off warmup requests that will saturate the running queue.
    warmup_sampling = SamplingParams(
        max_tokens=args.warmup_max_tokens, temperature=0.0, ignore_eos=True
    )
    probe_sampling = SamplingParams(
        max_tokens=args.probe_max_tokens, temperature=0.0, ignore_eos=True
    )

    # Burn-in: one tiny request to make the first compile/cudagraph cost
    # land outside our measurement window.
    burn_prompt = _build_prompt(8)
    burn_sampling = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    await _run_request(engine, burn_prompt, burn_sampling, time.perf_counter())

    warmup_tasks = []
    for _ in range(args.num_warmup):
        prompt = _build_prompt(args.warmup_prompt_tokens)
        warmup_tasks.append(
            asyncio.create_task(
                _run_request(engine, prompt, warmup_sampling, time.perf_counter())
            )
        )

    # Let the running queue fill before we submit the probes. This is what
    # makes the prefetch pass observable: by the time probes arrive, the
    # running queue is already saturating max_num_batched_tokens, so the
    # waiting-queue scheduling loop would never run -- and without the
    # early prefetch pass, the connector lookup never starts until later.
    await asyncio.sleep(args.warmup_settle_s)

    probe_tasks = []
    for _ in range(args.num_probe):
        prompt = _build_prompt(args.probe_prompt_tokens)
        submit = time.perf_counter()
        probe_tasks.append(
            asyncio.create_task(_run_request(engine, prompt, probe_sampling, submit))
        )

    probes = await asyncio.gather(*probe_tasks)
    # Drain warmups so engine shutdown is clean.
    await asyncio.gather(*warmup_tasks)
    return probes


def _summarize(probes: list[ProbeResult], label: str) -> dict[str, float]:
    ttfts = sorted(p.ttft for p in probes)
    n = len(ttfts)
    p50 = ttfts[n // 2]
    p90 = ttfts[min(n - 1, int(n * 0.9))]
    p99 = ttfts[min(n - 1, int(n * 0.99))]
    mean = statistics.fmean(ttfts)
    print(f"\n=== {label} (n={n}) ===")
    print(f"  TTFT mean : {mean * 1000:8.2f} ms")
    print(f"  TTFT p50  : {p50 * 1000:8.2f} ms")
    print(f"  TTFT p90  : {p90 * 1000:8.2f} ms")
    print(f"  TTFT p99  : {p99 * 1000:8.2f} ms")
    print(f"  TTFT min  : {ttfts[0] * 1000:8.2f} ms")
    print(f"  TTFT max  : {ttfts[-1] * 1000:8.2f} ms")
    return {"mean": mean, "p50": p50, "p90": p90, "p99": p99}


async def _amain(args: argparse.Namespace) -> None:
    os.environ["VLLM_SLOWMOCK_LATENCY_MS"] = str(args.latency_ms)
    os.environ["VLLM_SLOWMOCK_MATCHED_TOK"] = str(args.matched_tokens)

    if args.compare:
        # Run twice: budget=0 (disabled) then budget=large (enabled).
        # Use a generous budget so every probe gets its hint in step 1.
        big_budget = max(
            args.num_probe * args.probe_prompt_tokens * 4,
            args.max_num_batched_tokens * 4,
        )

        engine = _make_engine(args, budget=0)
        try:
            off = await _drive_one(engine, args)
        finally:
            engine.shutdown()
        off_stats = _summarize(off, label="BUDGET=0 (disabled)")

        # Brief gap so any lingering background work settles.
        await asyncio.sleep(1.0)

        engine = _make_engine(args, budget=big_budget)
        try:
            on = await _drive_one(engine, args)
        finally:
            engine.shutdown()
        on_stats = _summarize(on, label=f"BUDGET={big_budget} (enabled)")

        print("\n=== DELTA (enabled - disabled) ===")
        for k in ("mean", "p50", "p90", "p99"):
            d_ms = (on_stats[k] - off_stats[k]) * 1000
            print(f"  TTFT {k}: {d_ms:+8.2f} ms")
        print(
            "\nExpected sign: negative (TTFT improves) when the simulated "
            "lookup latency was being paid in front of the GPU under "
            "BUDGET=0 and is hidden behind running-queue work under "
            "BUDGET>0."
        )
    else:
        engine = _make_engine(args, budget=args.budget)
        try:
            probes = await _drive_one(engine, args)
        finally:
            engine.shutdown()
        _summarize(probes, label=f"BUDGET={args.budget}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-batched-tokens", type=int, default=512)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--enforce-eager", action="store_true", default=True)

    # The "warmup" phase is what saturates the per-step token budget. We
    # rely on prefill, not decode, to do the saturating: prefill consumes
    # `warmup_prompt_tokens` per step per request until each prompt is
    # fully ingested, while decode only consumes 1 token per step per
    # request. Pick num_warmup * warmup_prompt_tokens >> max_num_batched
    # _tokens so several consecutive schedule steps run with the running
    # queue completely full -- that is the window in which the bug bites.
    p.add_argument("--num-warmup", type=int, default=4)
    p.add_argument("--warmup-prompt-tokens", type=int, default=384)
    p.add_argument("--warmup-max-tokens", type=int, default=64)
    p.add_argument("--warmup-settle-s", type=float, default=0.5)

    p.add_argument("--num-probe", type=int, default=8)
    p.add_argument("--probe-prompt-tokens", type=int, default=64)
    p.add_argument("--probe-max-tokens", type=int, default=1)

    p.add_argument("--latency-ms", type=float, default=200.0)
    p.add_argument("--matched-tokens", type=int, default=0)
    p.add_argument("--budget", type=int, default=0)
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run twice (budget=0 then large budget) and print delta.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
