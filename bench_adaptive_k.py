"""
Benchmark: Fixed K=5 vs Adaptive K for speculative decoding.
Target: Qwen2.5-7B-Instruct-AWQ  (7B Q4)
Draft:  Qwen2.5-0.5B-Instruct-AWQ  (0.5B AWQ)

Each config runs in a separate subprocess to isolate GPU state.
"""

import os
import sys
import subprocess

import json

COMMON_KWARGS = dict(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    spec_model="Qwen/Qwen2.5-0.5B-Instruct-AWQ",
    spec_tokens=5,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.70,
    max_model_len=2048,
    enforce_eager=True,
    cpu_offload_gb=3,
)

FIXED_CFG = {}
ADAPTIVE_CFG = {
    "speculative_config": {
        "enable_adaptive_k": True,
        "adaptive_k_ema_alpha": 0.3,
        "adaptive_k_c_draft": 0.1144,
        "adaptive_k_min_tokens": 1,
        "adaptive_k_alpha_prior": 0.85,
    },
}

PROMPTS = [
    "Write a Python factorial function.",
    "Explain relativity simply.",
    "What is the capital of France and what is its most famous landmark?",
    "Write a haiku about artificial intelligence.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What are the three laws of thermodynamics?",
    "Write a recursive binary search in Python.",
    "Explain how a transformer neural network works.",
    "What is the difference between TCP and UDP?",
    "Write a short poem about the ocean.",
]


def make_runner_code(label: str, extra_args: dict) -> str:
    """Generate Python code for a subprocess that runs one benchmark."""
    # Build kwargs dict with correct Python bool/None syntax
    kwargs_items = []
    for k, v in COMMON_KWARGS.items():
        kwargs_items.append(f"    {k!r}: {v!r}")
    for k, v in extra_args.items():
        if isinstance(v, dict):
            inner = ", ".join(f"{kk!r}: {vv!r}" for kk, vv in v.items())
            kwargs_items.append(f"    {k!r}: {{{inner}}}")
        else:
            kwargs_items.append(f"    {k!r}: {v!r}")
    kwargs_str = ",\n".join(kwargs_items)

    prompts_str = ",\n".join(f"    {p!r}" for p in PROMPTS)

    return f"""
import os, json, time, gc, torch
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.llm import LLM

PROMPTS = [
{prompts_str},
]
sp = SamplingParams(temperature=0.0, max_tokens=128)

kwargs = {{
{kwargs_str},
}}

llm = LLM(**kwargs)

# Warmup
_ = llm.generate(["Hello world"] * 2, sp)

# Timed run
torch.cuda.synchronize()
t0 = time.perf_counter()
outputs = llm.generate(PROMPTS, sp)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

total_tokens = sum(
    len(completion.token_ids)
    for out in outputs
    for completion in out.outputs
)

tpot_ms = (elapsed / total_tokens) * 1000
throughput = total_tokens / elapsed

result = {{
    "label": {label!r},
    "elapsed_s": round(elapsed, 3),
    "total_tokens": total_tokens,
    "tpot_ms": round(tpot_ms, 2),
    "throughput": round(throughput, 2),
}}

print(json.dumps(result))

del llm
gc.collect()
torch.cuda.empty_cache()
"""


def run_single(label: str, extra_args: dict) -> dict:
    """Run a single benchmark in a subprocess, return stats."""
    code = make_runner_code(label, extra_args)
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )

    # Print subprocess stdout/stderr for visibility
    if proc.stdout:
        for line in proc.stdout.strip().splitlines():
            print(line)
    if proc.stderr:
        for line in proc.stderr.strip().splitlines():
            if "WARNING" in line or "ERROR" in line or "Traceback" in line:
                print(f"  [stderr] {line}")

    # Parse the JSON result from output
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        try:
            data = json.loads(line)
            if "label" in data and "tpot_ms" in data:
                return data
        except json.JSONDecodeError:
            pass

    # If no JSON found, show full output
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    raise RuntimeError(f"Benchmark '{label}' failed — no JSON result found")


def main():
    results = []

    for label, extra in [("Fixed K=5", FIXED_CFG), ("Adaptive K", ADAPTIVE_CFG)]:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        r = run_single(label, extra)
        results.append(r)
        print(f"  Total time:      {r['elapsed_s']:.2f}s")
        print(f"  Total tokens:    {r['total_tokens']}")
        print(f"  TPOT:            {r['tpot_ms']:.2f} ms/tok")
        print(f"  Throughput:      {r['throughput']:.2f} tok/s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'TPOT (ms/tok)':<20} {'Throughput (tok/s)':<20}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['label']:<20} {r['tpot_ms']:<20.2f} {r['throughput']:<20.2f}")

    if len(results) == 2:
        speedup = (results[1]["throughput"] / results[0]["throughput"] - 1) * 100
        print(f"\n  Adaptive K throughput change vs Fixed K=5: {speedup:+.1f}%")


if __name__ == "__main__":
    main()
