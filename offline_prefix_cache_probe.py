#!/usr/bin/env python3
"""Run a real vLLM offline generation and print prefix-cache hit stats.

This is intentionally not an HTTP/API-server test. It loads a model with
vllm.LLM, calls LLM.generate(), and records scheduler prefix-cache stats from
inside the engine.

Example:
    python offline_prefix_cache_probe.py --model facebook/opt-125m
    python offline_prefix_cache_probe.py --model facebook/opt-125m --cache-salt org-a
"""

from __future__ import annotations

import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any


if sys.version_info < (3, 10):
    raise SystemExit(
        "This vLLM checkout requires Python >= 3.10. "
        f"Current interpreter: {sys.version.split()[0]}"
    )

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm import LLM, SamplingParams


@dataclass
class PrefixCacheSnapshot:
    requests: int = 0
    queries: int = 0
    hits: int = 0
    hit_rate: float = 0.0


class PrefixCacheRecorder:
    """Small research logger for collecting per-generate prefix-cache stats."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.requests = 0
        self.queries = 0
        self.hits = 0

    def record(
        self,
        scheduler_stats: Any | None,
        iteration_stats: Any | None,
        mm_cache_stats: Any | None = None,
        engine_idx: int = 0,
    ) -> None:
        del iteration_stats, mm_cache_stats, engine_idx
        if scheduler_stats is None:
            return
        stats = scheduler_stats.prefix_cache_stats
        self.requests += stats.requests
        self.queries += stats.queries
        self.hits += stats.hits
        self.requests += stats.preempted_requests
        self.queries += stats.preempted_queries
        self.hits += stats.preempted_hits

    def log(self) -> None:
        pass

    def log_engine_initialized(self) -> None:
        pass

    def record_sleep_state(self, is_awake: int, level: int) -> None:
        del is_awake, level

    def snapshot(self) -> PrefixCacheSnapshot:
        hit_rate = self.hits / self.queries if self.queries else 0.0
        return PrefixCacheSnapshot(
            requests=self.requests,
            queries=self.queries,
            hits=self.hits,
            hit_rate=hit_rate,
        )


def attach_prefix_cache_recorder(llm: LLM) -> PrefixCacheRecorder:
    recorder = PrefixCacheRecorder()
    logger_manager = llm.llm_engine.logger_manager
    if logger_manager is None:
        raise RuntimeError(
            "llm_engine.logger_manager is None. Construct LLM with "
            "disable_log_stats=False."
        )
    logger_manager.stat_loggers.append(recorder)  # type: ignore[arg-type]
    return recorder


def make_prompt(shared_prefix: str, query: str, cache_salt: str | None) -> Any:
    prompt = shared_prefix + "\n\n" + query
    if cache_salt is None:
        return prompt
    return {"prompt": prompt, "cache_salt": cache_salt}


def run_one(
    llm: LLM,
    recorder: PrefixCacheRecorder,
    label: str,
    prompt: Any,
    sampling_params: SamplingParams,
) -> tuple[str, PrefixCacheSnapshot, float]:
    recorder.reset()
    start = time.perf_counter()
    outputs = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    text = outputs[0].outputs[0].text
    stats = recorder.snapshot()

    print(f"\n[{label}]")
    print(f"elapsed_seconds = {elapsed:.4f}")
    print(f"prefix_requests = {stats.requests}")
    print(f"prefix_queries  = {stats.queries}")
    print(f"prefix_hits     = {stats.hits}")
    print(f"prefix_hit_rate = {stats.hit_rate:.4f}")
    print(f"generated_text  = {text!r}")
    return text, stats, elapsed


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--cache-salt", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--shared-repeat",
        type=int,
        default=128,
        help="How many repeated sentences to put in the shared prefix.",
    )
    args = parser.parse_args()

    shared_prefix = (
        "You are a deterministic research assistant. "
        "Remember this reusable shared prefix sentence. "
    ) * args.shared_repeat

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        enable_prefix_caching=True,
        disable_log_stats=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    recorder = attach_prefix_cache_recorder(llm)

    warm_prompt = make_prompt(
        shared_prefix,
        "Warmup query: answer with the word warmup.",
        args.cache_salt,
    )
    probe_prompt = make_prompt(
        shared_prefix,
        "Probe query: answer with the word probe.",
        args.cache_salt,
    )

    _, warm_stats, _ = run_one(
        llm, recorder, "warmup", warm_prompt, sampling_params
    )
    _, probe_stats, _ = run_one(
        llm, recorder, "probe_same_prefix", probe_prompt, sampling_params
    )

    print("\n[summary]")
    print("The warmup request should usually have near-zero prefix hits.")
    print("The probe request should have non-zero hits if prefix caching works.")
    print(f"warmup_hits = {warm_stats.hits}")
    print(f"probe_hits  = {probe_stats.hits}")


if __name__ == "__main__":
    main()
