"""
Profile speculative decoding: compare acceptance depth vs adaptive K selection.

Measures per-step:
  - Fixed K=5: k_proposed=5, k_accepted = actual acceptance depth (max K that works)
  - Adaptive K:  adaptive_k_for_step = what algorithm chose

Outputs aggregate stats and distribution comparisons.
"""

import os, sys, json, subprocess, gc, time, torch
from collections import Counter
from pathlib import Path

STATS_FILE = "/tmp/adaptive_k_per_step.jsonl"

COMMON_KWARGS = dict(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    spec_model="Qwen/Qwen2.5-0.5B-Instruct-AWQ",
    spec_tokens=5,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.70,
    max_model_len=2048,
    enforce_eager=True,
    cpu_offload_gb=3,
    disable_log_stats=True,
)

FIXED_CFG = {}
ADAPTIVE_CFG = {
    "speculative_config": {
        "enable_adaptive_k": True,
        "adaptive_k_ema_alpha": 0.3,
        "adaptive_k_c_draft": 0.05,
        "adaptive_k_min_tokens": 1,
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
os.environ["VLLM_ADAPTIVE_K_STATS"] = {STATS_FILE!r}
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
total_tokens = sum(len(c.token_ids) for out in outputs for c in out.outputs)
tpot_ms = (elapsed / total_tokens) * 1000
throughput = total_tokens / elapsed
result = {{"label": {label!r}, "elapsed_s": round(elapsed, 3), "total_tokens": total_tokens, "tpot_ms": round(tpot_ms, 2), "throughput": round(throughput, 2)}}
print(json.dumps(result))
del llm; gc.collect(); torch.cuda.empty_cache()
"""


def run_single(label: str, extra_args: dict) -> dict:
    # Clear stats file from previous run
    if os.path.exists(STATS_FILE):
        os.remove(STATS_FILE)

    code = make_runner_code(label, extra_args)
    env = os.environ.copy()
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=env)

    if proc.stdout:
        for line in proc.stdout.strip().splitlines():
            print(line)
    if proc.stderr:
        for line in proc.stderr.strip().splitlines():
            if "WARNING" in line or "ERROR" in line or "Traceback" in line:
                print(f"  [stderr] {line}")

    # Parse benchmark result
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        try:
            data = json.loads(line)
            if "label" in data and "tpot_ms" in data:
                return data
        except json.JSONDecodeError:
            pass

    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    raise RuntimeError(f"Benchmark '{label}' failed")


def load_stats() -> list[dict]:
    if not os.path.exists(STATS_FILE):
        return []
    with open(STATS_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(mode_label: str, mode: str, stats: list[dict]):
    """Analyze per-step stats for one mode."""
    if not stats:
        print(f"  No stats collected for {mode_label}")
        return

    # Filter steps where speculative decoding happened
    steps = [s for s in stats if s.get("step_type") == "sd_step"]

    if mode == "fixed":
        # In fixed mode, k_proposed is always num_spec_tokens.
        # k_accepted = acceptance depth = "max K that works for this position"
        accepted_vals = [s["k_accepted"] for s in steps]
        proposed_vals = [s["k_proposed"] for s in steps]

        print(f"\n  === {mode_label}: Acceptance Depth (max K that works) ===")
        print(f"  Total steps (req×step):  {len(accepted_vals)}")
        print(f"  Average k_proposed:      {sum(proposed_vals)/len(proposed_vals):.2f}")
        print(f"  Average k_accepted:      {sum(accepted_vals)/len(accepted_vals):.2f}")
        print(f"  Acceptance rate:         {sum(accepted_vals)/sum(proposed_vals)*100:.1f}%")

        # Distribution of acceptance depths
        dist = Counter(accepted_vals)
        print(f"\n  Acceptance depth distribution:")
        for depth in sorted(dist.keys()):
            pct = dist[depth] / len(accepted_vals) * 100
            bar = "█" * int(pct / 2)
            print(f"    K={depth}: {dist[depth]:>4} steps ({pct:5.1f}%)  {bar}")

    elif mode == "adaptive":
        # For adaptive: adaptive_k_for_step = what K was actually chosen
        # k_proposed = always num_spec_tokens, k_accepted = actual acceptance
        k_chosen = [s.get("adaptive_k_for_step") or 5 for s in steps]
        accepted_vals = [s["k_accepted"] for s in steps]
        invalid_counts = [s.get("num_invalid") or 0 for s in steps]

        # The "real" K (not counting padding) = adaptive_k_for_step
        real_k = k_chosen

        print(f"\n  === {mode_label}: Adaptive K Selection ===")
        print(f"  Total steps (req×step):  {len(steps)}")
        print(f"  Average K chosen:        {sum(real_k)/len(real_k):.2f}")
        print(f"  Average k_accepted:      {sum(accepted_vals)/len(accepted_vals):.2f}")

        # Distribution of chosen K values
        dist = Counter(real_k)
        print(f"\n  Adaptive K distribution:")
        for k in sorted(dist.keys()):
            pct = dist[k] / len(real_k) * 100
            bar = "█" * int(pct / 2)
            print(f"    K={k}: {dist[k]:>4} steps ({pct:5.1f}%)  {bar}")

        # How often does adaptive choose a suboptimal K?
        # "Optimal" for a step would be: acceptance_depth from the same step
        # If adaptive K > actual_accepted+1: wasted compute (over-speculated)
        # If adaptive K < actual_accepted+1: left throughput on table (under-speculated)
        wasted = sum(1 for k, a in zip(real_k, accepted_vals) if k > a + 1)
        under = sum(1 for k, a in zip(real_k, accepted_vals) if k < a + 1)
        exact = sum(1 for k, a in zip(real_k, accepted_vals) if k == a + 1 or k == a)
        print(f"\n  Optimality (vs actual acceptance):")
        print(f"    K ≈ optimal   (k ≈ accept+1): {exact:>4} steps ({exact/max(len(real_k),1)*100:5.1f}%)")
        print(f"    Over-spec     (k > accept+1): {wasted:>4} steps ({wasted/max(len(real_k),1)*100:5.1f}%)")
        print(f"    Under-spec    (k < accept+1): {under:>4} steps ({under/max(len(real_k),1)*100:5.1f}%)")


def main():
    results = []

    for label, extra, mode in [
        ("Fixed K=5", FIXED_CFG, "fixed"),
        ("Adaptive K", ADAPTIVE_CFG, "adaptive"),
    ]:
        print(f"\n{'='*65}")
        print(f"  {label}")
        print(f"{'='*65}")
        r = run_single(label, extra)
        results.append(r)
        print(f"  Total time:   {r['elapsed_s']:.2f}s")
        print(f"  Throughput:   {r['throughput']:.2f} tok/s")
        print(f"  TPOT:         {r['tpot_ms']:.2f} ms/tok")

        stats = load_stats()
        analyze(label, mode, stats)

    # Summary comparison
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"{'Config':<20} {'TPOT':<20} {'Throughput':<20}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['label']:<20} {r['tpot_ms']:<20.2f} {r['throughput']:<20.2f}")

    if len(results) == 2:
        speedup = (results[1]["throughput"] / results[0]["throughput"] - 1) * 100
        print(f"\n  Adaptive K vs Fixed K=5: {speedup:+.1f}% throughput")


if __name__ == "__main__":
    main()
