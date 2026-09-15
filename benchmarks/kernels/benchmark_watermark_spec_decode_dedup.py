#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.platforms import current_platform
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.watermarking.gumbel import GumbelWatermarker
from vllm.v1.watermarking.spec_decode import DraftWatermarker
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample


@dataclass
class PairedTiming:
    baseline_us: float
    deduplicated_us: float
    delta_us: float
    delta_p10_us: float
    delta_p90_us: float
    baseline_gpu_us: float
    deduplicated_gpu_us: float
    gpu_delta_us: float


@dataclass
class BenchmarkResult:
    batch_size: int
    history_length: int
    num_speculative_tokens: int
    context_width: int
    vocab_size: int
    draft_block: PairedTiming
    target_mask: PairedTiming
    resampler: PairedTiming
    added_us_per_engine_step: float
    added_gpu_us_per_engine_step: float
    added_us_per_draft_step: float


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _measure_once(fn: Callable[[], object], iterations: int) -> tuple[float, float]:
    torch.accelerator.synchronize()
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    for _ in range(iterations):
        fn()
    end_event.record()
    torch.accelerator.synchronize()
    wall_us = (time.perf_counter() - started) * 1e6 / iterations
    gpu_us = start_event.elapsed_time(end_event) * 1e3 / iterations
    return wall_us, gpu_us


def _measure_pair(
    baseline: Callable[[], object],
    deduplicated: Callable[[], object],
    warmup: int,
    repeats: int,
    iterations: int,
) -> PairedTiming:
    for _ in range(warmup):
        baseline()
        deduplicated()
    torch.accelerator.synchronize()

    baseline_wall: list[float] = []
    deduplicated_wall: list[float] = []
    baseline_gpu: list[float] = []
    deduplicated_gpu: list[float] = []
    wall_deltas: list[float] = []
    gpu_deltas: list[float] = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            baseline_sample = _measure_once(baseline, iterations)
            deduplicated_sample = _measure_once(deduplicated, iterations)
        else:
            deduplicated_sample = _measure_once(deduplicated, iterations)
            baseline_sample = _measure_once(baseline, iterations)
        baseline_wall.append(baseline_sample[0])
        baseline_gpu.append(baseline_sample[1])
        deduplicated_wall.append(deduplicated_sample[0])
        deduplicated_gpu.append(deduplicated_sample[1])
        wall_deltas.append(deduplicated_sample[0] - baseline_sample[0])
        gpu_deltas.append(deduplicated_sample[1] - baseline_sample[1])

    return PairedTiming(
        baseline_us=statistics.median(baseline_wall),
        deduplicated_us=statistics.median(deduplicated_wall),
        delta_us=statistics.median(wall_deltas),
        delta_p10_us=_percentile(wall_deltas, 0.1),
        delta_p90_us=_percentile(wall_deltas, 0.9),
        baseline_gpu_us=statistics.median(baseline_gpu),
        deduplicated_gpu_us=statistics.median(deduplicated_gpu),
        gpu_delta_us=statistics.median(gpu_deltas),
    )


def _make_request_state(
    batch_size: int,
    history_length: int,
    context_width: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    all_token_ids = torch.randint(
        vocab_size,
        (batch_size, history_length),
        dtype=torch.int32,
        device=device,
    )
    prompt_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    total_lens = torch.full(
        (batch_size,), history_length, dtype=torch.int32, device=device
    )
    contexts = all_token_ids[:, -context_width:].to(torch.int64)
    return all_token_ids, prompt_lens, total_lens, contexts


def _make_draft_pair(
    batch_size: int,
    history_length: int,
    num_speculative_tokens: int,
    context_width: int,
    vocab_size: int,
    max_history: int | None,
    device: torch.device,
) -> tuple[Callable[[], object], Callable[[], object]]:
    all_token_ids, prompt_lens, total_lens, initial_contexts = _make_request_state(
        batch_size, history_length, context_width, vocab_size, device
    )
    logits = torch.randn(batch_size, vocab_size, dtype=torch.bfloat16, device=device)
    ordinary_sampled = torch.randint(
        vocab_size, (batch_size,), dtype=torch.int64, device=device
    )
    idx_mapping = torch.arange(batch_size, dtype=torch.int64, device=device)
    temperature = torch.ones(batch_size, dtype=torch.float32, device=device)
    enabled = torch.ones(batch_size, dtype=torch.bool, device=device)
    draft_steps = torch.arange(num_speculative_tokens, dtype=torch.int64, device=device)

    baseline_watermarker = DraftWatermarker(
        GumbelWatermarker(key=42, context_width=context_width),
        max_num_reqs=batch_size,
        device=device,
        num_speculative_steps=num_speculative_tokens,
        deduplicate_contexts="none",
        deduplicate_contexts_max_history=max_history,
    )
    deduplicated_watermarker = DraftWatermarker(
        GumbelWatermarker(key=42, context_width=context_width),
        max_num_reqs=batch_size,
        device=device,
        num_speculative_steps=num_speculative_tokens,
        deduplicate_contexts="single_turn",
        deduplicate_contexts_max_history=max_history,
    )

    def baseline() -> torch.Tensor:
        baseline_watermarker.prepare(
            initial_contexts,
            enabled,
            all_token_ids,
            prompt_lens,
            total_lens,
        )
        sampled = ordinary_sampled
        for draft_step in range(num_speculative_tokens):
            sampled = baseline_watermarker.sample(
                logits,
                ordinary_sampled,
                idx_mapping,
                temperature,
                draft_steps[draft_step],
            )
        return sampled

    def deduplicated() -> torch.Tensor:
        deduplicated_watermarker.prepare(
            initial_contexts,
            enabled,
            all_token_ids,
            prompt_lens,
            total_lens,
        )
        sampled = ordinary_sampled
        for draft_step in range(num_speculative_tokens):
            sampled = deduplicated_watermarker.sample(
                logits,
                ordinary_sampled,
                idx_mapping,
                temperature,
                draft_steps[draft_step],
            )
        return sampled

    return baseline, deduplicated


def _make_target_mask_pair(
    batch_size: int,
    history_length: int,
    num_speculative_tokens: int,
    context_width: int,
    vocab_size: int,
    max_history: int | None,
    device: torch.device,
) -> tuple[Callable[[], object], Callable[[], object]]:
    all_token_ids, prompt_lens, total_lens, _ = _make_request_state(
        batch_size, history_length, context_width, vocab_size, device
    )
    num_rows = batch_size * (num_speculative_tokens + 1)
    expanded_idx_mapping = torch.arange(
        batch_size, dtype=torch.int32, device=device
    ).repeat_interleave(num_speculative_tokens + 1)
    expanded_local_pos = torch.arange(
        num_speculative_tokens + 1, dtype=torch.int32, device=device
    ).repeat(batch_size)
    draft_sampled = torch.randint(
        vocab_size, (num_rows,), dtype=torch.int64, device=device
    )

    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = GumbelWatermarker(key=42, context_width=context_width)
    sampler.deduplicate_contexts = "single_turn"
    sampler.deduplicate_contexts_max_history = max_history
    sampler.num_speculative_tokens = num_speculative_tokens
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=all_token_ids),
        prompt_len=SimpleNamespace(gpu=prompt_lens),
        total_len=SimpleNamespace(gpu=total_lens),
    )

    def baseline() -> torch.Tensor:
        return sampler._get_contexts(
            expanded_idx_mapping, expanded_local_pos, draft_sampled
        )

    def deduplicated() -> torch.Tensor:
        contexts = sampler._get_contexts(
            expanded_idx_mapping, expanded_local_pos, draft_sampled
        )
        return sampler._get_repeated_contexts(
            expanded_idx_mapping, contexts, expanded_local_pos
        )

    return baseline, deduplicated


def _make_rejection_inputs(
    batch_size: int,
    num_speculative_tokens: int,
    context_width: int,
    vocab_size: int,
    repeated_fraction: float,
    device: torch.device,
) -> tuple[dict[str, object], torch.Tensor]:
    num_rows = batch_size * (num_speculative_tokens + 1)
    logits = torch.randn(vocab_size, dtype=torch.bfloat16, device=device)
    target_logits = logits.expand(num_rows, -1).contiguous()
    draft_logits = logits.expand(batch_size, num_speculative_tokens, -1).contiguous()
    draft_sampled = torch.zeros(num_rows, dtype=torch.int64, device=device)
    drafts = torch.randint(
        vocab_size,
        (batch_size, num_speculative_tokens),
        dtype=torch.int64,
        device=device,
    )
    draft_sampled.view(batch_size, num_speculative_tokens + 1)[:, 1:] = drafts
    cu_num_logits = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * (
        num_speculative_tokens + 1
    )
    expanded_idx_mapping = torch.arange(
        batch_size, dtype=torch.int32, device=device
    ).repeat_interleave(num_speculative_tokens + 1)
    expanded_local_pos = torch.arange(
        num_speculative_tokens + 1, dtype=torch.int32, device=device
    ).repeat(batch_size)
    contexts = torch.randint(
        vocab_size,
        (num_rows, context_width),
        dtype=torch.int64,
        device=device,
    )
    skip_mask = torch.zeros(num_rows, dtype=torch.bool, device=device)
    num_repeated = round(batch_size * repeated_fraction)
    if num_repeated:
        bonus_rows = (
            torch.arange(num_repeated, device=device) * (num_speculative_tokens + 1)
            + num_speculative_tokens
        )
        skip_mask[bonus_rows] = True

    inputs: dict[str, object] = {
        "target_logits": target_logits,
        "draft_logits": draft_logits,
        "draft_sampled": draft_sampled,
        "cu_num_logits": cu_num_logits,
        "pos": torch.arange(num_rows, dtype=torch.int32, device=device),
        "idx_mapping": torch.arange(batch_size, dtype=torch.int32, device=device),
        "expanded_idx_mapping": expanded_idx_mapping,
        "expanded_local_pos": expanded_local_pos,
        "temperature": torch.ones(batch_size, dtype=torch.float32, device=device),
        "seed": torch.arange(batch_size, dtype=torch.int64, device=device),
        "num_speculative_steps": num_speculative_tokens,
        "contexts": contexts,
        "watermarking": torch.ones(batch_size, dtype=torch.bool, device=device),
        "watermark_key": 42,
    }
    return inputs, skip_mask


def _make_resampler_pair(
    batch_size: int,
    num_speculative_tokens: int,
    context_width: int,
    vocab_size: int,
    repeated_fraction: float,
    device: torch.device,
) -> tuple[Callable[[], object], Callable[[], object]]:
    inputs, skip_mask = _make_rejection_inputs(
        batch_size,
        num_speculative_tokens,
        context_width,
        vocab_size,
        repeated_fraction,
        device,
    )
    return (
        lambda: rejection_sample(**inputs),
        lambda: rejection_sample(**inputs, watermarking_skip_mask=skip_mask),
    )


def run_case(
    batch_size: int,
    history_length: int,
    args: argparse.Namespace,
    device: torch.device,
) -> BenchmarkResult:
    timing_args = (args.warmup, args.repeats, args.iterations)
    draft_block = _measure_pair(
        *_make_draft_pair(
            batch_size,
            history_length,
            args.num_speculative_tokens,
            args.context_width,
            args.vocab_size,
            args.max_history,
            device,
        ),
        *timing_args,
    )
    target_mask = _measure_pair(
        *_make_target_mask_pair(
            batch_size,
            history_length,
            args.num_speculative_tokens,
            args.context_width,
            args.vocab_size,
            args.max_history,
            device,
        ),
        *timing_args,
    )
    resampler = _measure_pair(
        *_make_resampler_pair(
            batch_size,
            args.num_speculative_tokens,
            args.context_width,
            args.vocab_size,
            args.repeated_fraction,
            device,
        ),
        *timing_args,
    )
    added_us = draft_block.delta_us + target_mask.delta_us + resampler.delta_us
    added_gpu_us = (
        draft_block.gpu_delta_us + target_mask.gpu_delta_us + resampler.gpu_delta_us
    )
    return BenchmarkResult(
        batch_size=batch_size,
        history_length=history_length,
        num_speculative_tokens=args.num_speculative_tokens,
        context_width=args.context_width,
        vocab_size=args.vocab_size,
        draft_block=draft_block,
        target_mask=target_mask,
        resampler=resampler,
        added_us_per_engine_step=added_us,
        added_gpu_us_per_engine_step=added_gpu_us,
        added_us_per_draft_step=added_us / args.num_speculative_tokens,
    )


def _print_results(results: list[BenchmarkResult]) -> None:
    print(
        "batch history draft_delta_us target_delta_us resample_delta_us "
        "added_step_us added_draft_step_us gpu_added_step_us"
    )
    for result in results:
        print(
            f"{result.batch_size:>5} {result.history_length:>7} "
            f"{result.draft_block.delta_us:>14.2f} "
            f"{result.target_mask.delta_us:>15.2f} "
            f"{result.resampler.delta_us:>17.2f} "
            f"{result.added_us_per_engine_step:>13.2f} "
            f"{result.added_us_per_draft_step:>19.2f} "
            f"{result.added_gpu_us_per_engine_step:>17.2f}"
        )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark speculative watermark context-deduplication overhead."
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument(
        "--history-lengths", type=int, nargs="+", default=[256, 2048, 8192]
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--context-width", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--max-history", type=int, default=8192)
    parser.add_argument("--repeated-fraction", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.accelerator.is_available():
        raise RuntimeError("An accelerator is required for this benchmark")
    if args.num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be positive")
    if not 0 <= args.repeated_fraction <= 1:
        raise ValueError("repeated_fraction must be between 0 and 1")

    device = torch.accelerator.current_accelerator()
    device_name = current_platform.get_device_name()
    torch.manual_seed(0)
    results = [
        run_case(batch_size, history_length, args, device)
        for history_length in args.history_lengths
        for batch_size in args.batch_sizes
    ]
    print(f"device={device_name} torch={torch.__version__}")
    _print_results(results)
    if args.json_output is not None:
        payload = {
            "commit": _git_commit(),
            "device": device_name,
            "torch_version": torch.__version__,
            "config": {
                "batch_sizes": args.batch_sizes,
                "history_lengths": args.history_lengths,
                "num_speculative_tokens": args.num_speculative_tokens,
                "context_width": args.context_width,
                "vocab_size": args.vocab_size,
                "max_history": args.max_history,
                "repeated_fraction": args.repeated_fraction,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "iterations": args.iterations,
            },
            "results": [asdict(result) for result in results],
        }
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
