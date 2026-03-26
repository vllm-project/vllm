#!/usr/bin/env python3
"""Comprehensive TurboQuant benchmark: quality + latency + batching.

Measures:
  - Output quality: exact match rate vs baseline
  - TTFT (time to first token): single request latency
  - Throughput: tokens/sec under batching (1, 4, 8, 16 concurrent)
  - Per-bit-width comparison (baseline, 4-bit, 3-bit, 2-bit)

Usage:
    python tests/quantization/bench_turboquant_vllm.py
    python tests/quantization/bench_turboquant_vllm.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import time

import torch
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

QUALITY_PROMPTS = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light in meters per second?",
    "Name the largest planet in our solar system.",
    "What year did World War II end?",
    "What is 15 * 23?",
    "If a train travels at 60 mph for 2.5 hours, how far does it go?",
    "What comes next: 2, 4, 8, 16, ?",
    "Write a haiku about autumn.",
    "Explain quantum computing in one sentence.",
    "Briefly explain how photosynthesis works.",
    "What are three benefits of regular exercise?",
]

BATCH_PROMPT = "Explain the theory of relativity in simple terms."


def create_llm(model_name: str, bits: int | None, gpu_util: float = 0.5):
    """Create LLM instance with optional TurboQuant."""
    if bits is not None:
        import vllm.model_executor.layers.quantization.turboquant as tq
        tq.TurboQuantConfig.__init__.__defaults__ = (bits, False, 42)
        kv_dtype = "turboquant"
    else:
        kv_dtype = "auto"

    return LLM(
        model=model_name,
        enforce_eager=True,
        kv_cache_dtype=kv_dtype,
        gpu_memory_utilization=gpu_util,
    )


def measure_ttft(llm: LLM, prompt: str, n_runs: int = 5) -> dict:
    """Measure TTFT by generating 1 token."""
    sampling_1 = SamplingParams(max_tokens=1, temperature=0)

    # Warmup
    llm.generate([prompt], sampling_1)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        llm.generate([prompt], sampling_1)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    return {
        "mean_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


def measure_itl(llm: LLM, prompt: str, max_tokens: int = 50,
                n_runs: int = 3) -> dict:
    """Measure Inter-Token Latency (ITL) via single-request generation."""
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0)
    # Warmup
    llm.generate([prompt], sampling)

    itls = []
    e2e_latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        t1 = time.perf_counter()

        n_gen = len(outputs[0].outputs[0].token_ids)
        e2e = (t1 - t0) * 1000  # ms
        e2e_latencies.append(e2e)
        if n_gen > 1:
            # ITL = (E2E - TTFT) / (n_gen - 1), approximate TTFT as small
            itls.append(e2e / n_gen)  # ms per token

    return {
        "itl_mean_ms": sum(itls) / len(itls) if itls else 0,
        "e2e_mean_ms": sum(e2e_latencies) / len(e2e_latencies),
        "e2e_min_ms": min(e2e_latencies),
        "tokens_generated": len(outputs[0].outputs[0].token_ids),
    }


def measure_prefill_throughput(llm: LLM, n_runs: int = 3) -> dict:
    """Measure prefill throughput with varying prompt lengths."""
    sampling_1 = SamplingParams(max_tokens=1, temperature=0)
    results = {}

    for length_name, prompt in [
        ("short (32tok)", "What is the meaning of life?"),
        ("medium (128tok)",
         "Explain the complete history of computing from Charles Babbage "
         "to modern GPUs, including all major milestones, key figures, "
         "and technological breakthroughs that shaped the industry. "
         "Cover mechanical computers, vacuum tubes, transistors, "
         "integrated circuits, microprocessors, and parallel computing."),
        ("long (512tok)",
         ("Describe the solar system in detail. " * 30).strip()),
    ]:
        # Warmup
        llm.generate([prompt], sampling_1)
        latencies = []
        prompt_lens = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = llm.generate([prompt], sampling_1)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
            prompt_lens.append(len(out[0].prompt_token_ids))

        avg_ms = sum(latencies) / len(latencies)
        avg_len = sum(prompt_lens) / len(prompt_lens)
        results[length_name] = {
            "prompt_tokens": avg_len,
            "latency_ms": avg_ms,
            "prefill_tok_per_s": avg_len / (avg_ms / 1000),
        }

    return results


def measure_throughput(llm: LLM, prompt: str, batch_sizes: list[int],
                       max_tokens: int = 128) -> dict:
    """Measure throughput at different batch sizes."""
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0)
    results = {}

    for bs in batch_sizes:
        prompts = [prompt] * bs
        # Warmup
        llm.generate(prompts[:1], sampling)

        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        t1 = time.perf_counter()

        total_gen = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_prompt = sum(len(o.prompt_token_ids) for o in outputs)
        elapsed = t1 - t0

        results[bs] = {
            "elapsed_s": elapsed,
            "prompt_tokens": total_prompt,
            "gen_tokens": total_gen,
            "gen_tok_per_s": total_gen / elapsed,
            "total_tok_per_s": (total_prompt + total_gen) / elapsed,
            "prefill_tok_per_s": total_prompt / elapsed,
        }

    return results


def measure_quality(llm: LLM) -> list[str]:
    """Run quality prompts, return responses."""
    sampling = SamplingParams(max_tokens=100, temperature=0)
    outputs = llm.generate(QUALITY_PROMPTS, sampling)
    return [o.outputs[0].text.strip() for o in outputs]


def run_benchmark(model_name: str):
    configs = [
        ("baseline", None),
        ("tq-4bit", 4),
        ("tq-3bit", 3),
        ("tq-2bit", 2),
    ]

    batch_sizes = [1, 4, 8, 16]
    all_results = {}

    for config_name, bits in configs:
        print(f"\n{'='*70}")
        print(f"  Config: {config_name}")
        print(f"{'='*70}")

        llm = create_llm(model_name, bits)

        # Quality
        print("\n  [1/3] Quality evaluation...")
        responses = measure_quality(llm)
        for i, (q, a) in enumerate(zip(QUALITY_PROMPTS, responses)):
            print(f"    Q: {q}")
            print(f"    A: {a[:100]}{'...' if len(a) > 100 else ''}")
            print()

        # TTFT
        print("  [2/3] TTFT measurement (5 runs)...")
        ttft_short = measure_ttft(llm, "Hi", n_runs=5)
        ttft_long = measure_ttft(
            llm,
            "Explain the complete history of the Roman Empire from its "
            "founding to its fall, including all major events.",
            n_runs=5,
        )
        print(f"    Short prompt TTFT: {ttft_short['mean_ms']:.1f} ms "
              f"(min={ttft_short['min_ms']:.1f}, max={ttft_short['max_ms']:.1f})")
        print(f"    Long prompt TTFT:  {ttft_long['mean_ms']:.1f} ms "
              f"(min={ttft_long['min_ms']:.1f}, max={ttft_long['max_ms']:.1f})")

        # Throughput under batching
        # ITL + E2E latency
        print("\n  [3/5] ITL + E2E latency (3 runs, 50 tokens)...")
        itl = measure_itl(llm, "Explain how a car engine works.", n_runs=3)
        print(f"    ITL: {itl['itl_mean_ms']:.1f} ms/token")
        print(f"    E2E: {itl['e2e_mean_ms']:.1f} ms "
              f"(min={itl['e2e_min_ms']:.1f})")
        print(f"    Tokens generated: {itl['tokens_generated']}")

        # Prefill throughput
        print("\n  [4/5] Prefill throughput (varying prompt lengths)...")
        prefill = measure_prefill_throughput(llm, n_runs=3)
        for name, data in prefill.items():
            print(f"    {name}: {data['prefill_tok_per_s']:.0f} tok/s "
                  f"({data['prompt_tokens']:.0f} tokens, "
                  f"{data['latency_ms']:.1f} ms)")

        # Throughput under batching
        print(f"\n  [5/5] Throughput (batch sizes: {batch_sizes})...")
        throughput = measure_throughput(llm, BATCH_PROMPT, batch_sizes)
        for bs, data in throughput.items():
            print(f"    batch={bs:>2}: {data['gen_tok_per_s']:>8.1f} gen tok/s, "
                  f"{data['total_tok_per_s']:>8.1f} total tok/s, "
                  f"{data['elapsed_s']:.2f}s")

        all_results[config_name] = {
            "responses": responses,
            "ttft_short": ttft_short,
            "ttft_long": ttft_long,
            "itl": itl,
            "prefill": prefill,
            "throughput": throughput,
        }

        del llm
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # Summary tables
    # ---------------------------------------------------------------------------
    baseline_responses = all_results["baseline"]["responses"]

    print(f"\n{'='*70}")
    print("  SUMMARY: Quality")
    print(f"{'='*70}")
    print(f"  {'Config':<12} {'1st-sent match':>15} {'Exact match':>12}")
    print(f"  {'-'*42}")
    for config_name, data in all_results.items():
        first_match = sum(
            1 for b, r in zip(baseline_responses, data["responses"])
            if b.split('.')[0] == r.split('.')[0]
        )
        exact_match = sum(
            1 for b, r in zip(baseline_responses, data["responses"])
            if b == r
        )
        n = len(baseline_responses)
        print(f"  {config_name:<12} {first_match}/{n:>13} {exact_match}/{n:>11}")

    print(f"\n{'='*70}")
    print("  SUMMARY: TTFT (ms)")
    print(f"{'='*70}")
    print(f"  {'Config':<12} {'Short prompt':>14} {'Long prompt':>14} {'Overhead':>10}")
    print(f"  {'-'*52}")
    base_short = all_results["baseline"]["ttft_short"]["mean_ms"]
    base_long = all_results["baseline"]["ttft_long"]["mean_ms"]
    for config_name, data in all_results.items():
        short = data["ttft_short"]["mean_ms"]
        long_ = data["ttft_long"]["mean_ms"]
        overhead = ((short + long_) / 2) / ((base_short + base_long) / 2)
        overhead_str = f"{overhead:.2f}x" if config_name != "baseline" else "-"
        print(f"  {config_name:<12} {short:>13.1f} {long_:>13.1f} {overhead_str:>10}")

    print(f"\n{'='*70}")
    print("  SUMMARY: ITL + E2E Latency")
    print(f"{'='*70}")
    print(f"  {'Config':<12} {'ITL (ms/tok)':>13} {'E2E (ms)':>10} {'Overhead':>10}")
    print(f"  {'-'*47}")
    base_itl = all_results["baseline"]["itl"]["itl_mean_ms"]
    base_e2e = all_results["baseline"]["itl"]["e2e_mean_ms"]
    for config_name, data in all_results.items():
        itl_ms = data["itl"]["itl_mean_ms"]
        e2e_ms = data["itl"]["e2e_mean_ms"]
        overhead = itl_ms / base_itl if base_itl > 0 else 0
        overhead_str = f"{overhead:.2f}x" if config_name != "baseline" else "-"
        print(f"  {config_name:<12} {itl_ms:>12.1f} {e2e_ms:>9.1f} {overhead_str:>10}")

    print(f"\n{'='*70}")
    print("  SUMMARY: Prefill Throughput (tok/s)")
    print(f"{'='*70}")
    pfill_keys = list(next(iter(all_results.values()))["prefill"].keys())
    header = f"  {'Config':<12}"
    for k in pfill_keys:
        header += f" {k:>16}"
    print(header)
    print(f"  {'-'*(12 + 17 * len(pfill_keys))}")
    for config_name, data in all_results.items():
        row = f"  {config_name:<12}"
        for k in pfill_keys:
            tok_s = data["prefill"][k]["prefill_tok_per_s"]
            row += f" {tok_s:>16.0f}"
        print(row)

    print(f"\n{'='*70}")
    print("  SUMMARY: Throughput (gen tok/s)")
    print(f"{'='*70}")
    header = f"  {'Config':<12}"
    for bs in batch_sizes:
        header += f" {'bs='+str(bs):>10}"
    print(header)
    print(f"  {'-'*(12 + 11 * len(batch_sizes))}")
    for config_name, data in all_results.items():
        row = f"  {config_name:<12}"
        for bs in batch_sizes:
            tok_s = data["throughput"][bs]["gen_tok_per_s"]
            row += f" {tok_s:>10.1f}"
        print(row)

    # Diffs
    print(f"\n{'='*70}")
    print("  DIFFERENCES vs baseline")
    print(f"{'='*70}")
    for config_name, data in all_results.items():
        if config_name == "baseline":
            continue
        diffs = [(i, b, r) for i, (b, r) in enumerate(
            zip(baseline_responses, data["responses"])) if b != r]
        if diffs:
            print(f"\n  {config_name}: {len(diffs)}/{len(baseline_responses)} differ")
            for i, b, r in diffs[:3]:
                print(f"    Q: {QUALITY_PROMPTS[i]}")
                print(f"    Base: {b[:80]}")
                print(f"    TQ:   {r[:80]}")
        else:
            print(f"\n  {config_name}: ALL match baseline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()
    run_benchmark(args.model)
