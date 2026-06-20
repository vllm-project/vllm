"""
Full diagnostic: collects and reports all 9 metrics for adaptive K validation.

Usage:  python _diagnose_adaptive_k.py

Outputs:
  M1: K* distribution
  M2: Per-position conditional acceptance (alpha_i)
  M3: E_acc predicted vs realized
  M4: Goodput(K) curve snapshot (last step)
  M5: Per-prompt acceptance profile
  M6: Exploration outcomes
  M7: K stability (consecutive same-K runs)
  M8: Cooldown rate
  M9: Throughput comparison
"""

import os, sys, json, subprocess, statistics, time
from collections import Counter, defaultdict
from pathlib import Path

STATS_FILE = "/tmp/adaptive_k_diag.jsonl"

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
        "adaptive_k_ema_alpha": 0.5,
        "adaptive_k_c_draft": 0.05,
        "adaptive_k_min_tokens": 1,
        "adaptive_k_alpha_prior": 0.5,
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
PROMPTS = [{prompts_str},]
sp = SamplingParams(temperature=0.0, max_tokens=128)
kwargs = {{{kwargs_str},}}
llm = LLM(**kwargs)
_ = llm.generate(["Hello world"] * 2, sp)
torch.cuda.synchronize()
t0 = time.perf_counter()
outputs = llm.generate(PROMPTS, sp)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
total_tokens = sum(len(c.token_ids) for out in outputs for c in out.outputs)
tpot_ms = (elapsed / total_tokens) * 1000
throughput = total_tokens / elapsed
result = {{"label": {label!r},"elapsed_s": round(elapsed,3),"total_tokens": total_tokens,"tpot_ms": round(tpot_ms,2),"throughput": round(throughput,2)}}
print(json.dumps(result))
del llm; gc.collect(); torch.cuda.empty_cache()
"""


def run_single(label: str, extra_args: dict) -> dict | None:
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
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        try:
            data = json.loads(line)
            if "label" in data and "tpot_ms" in data:
                return data
        except json.JSONDecodeError:
            pass
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return None


def load_stats():
    if not os.path.exists(STATS_FILE):
        return []
    with open(STATS_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


# ─── Report ───

def report_fixed(steps):
    print("\n" + "=" * 65)
    print("  FIXED K=5: ACCEPTANCE BASELINE (M1, M2)")
    print("=" * 65)
    accepted_vals = [s["num_accepted"] for s in steps]
    n = len(accepted_vals)
    if n == 0:
        print("  No data.")
        return accepted_vals
    print(f"  Total steps: {n}")
    print(f"  Average k_accepted: {sum(accepted_vals)/n:.2f}")
    print(f"  Acceptance rate:    {sum(accepted_vals)/(n*5)*100:.1f}%")
    print(f"\n  ── M1: Acceptance depth distribution (max K that works) ──")
    dist = Counter(accepted_vals)
    for depth in sorted(dist.keys()):
        pct = dist[depth] / n * 100
        bar = "█" * max(1, int(pct / 2))
        label = f"K={depth}" if depth < 5 else "K=5 (cap)"
        print(f"    {label:>12}: {dist[depth]:>4} ({pct:5.1f}%) {bar}")
    print(f"\n  ── M2: Per-position conditional acceptance ──")
    # Compute from the LAST snapshot of pos_reached/pos_accepted
    final_step = steps[-1]
    pre = final_step.get("pos_reached", [])
    pac = final_step.get("pos_accepted", [])
    if pre:
        for i in range(len(pre)):
            if pre[i] > 0:
                rate = pac[i] / pre[i]
                print(f"    α_{i+1}: {rate:.4f}  ({pac[i]:>4}/{pre[i]:>4})")
    return accepted_vals


def report_adaptive(steps):
    print("\n" + "=" * 65)
    print("  ADAPTIVE K: FULL DIAGNOSTIC (M1-M8)")
    print("=" * 65)
    if not steps:
        print("  No data.")
        return
    n = len(steps)

    # ── M1: K* distribution ──
    k_vals = [s["selected_k"] for s in steps]
    adaptive_k_vals = [s.get("adaptive_k_fs") or 5 for s in steps]
    # For adaptive mode, selected_k is the previous_adaptive_k which might be stale.
    # adaptive_k_fs is the actual K used this step.
    actual_k_vals = adaptive_k_vals
    k_dist = Counter(actual_k_vals)
    print(f"\n  ── M1: K* DISTRIBUTION ({n} steps) ──")
    for k in sorted(k_dist.keys()):
        pct = k_dist[k] / n * 100
        bar = "█" * max(1, int(pct / 2))
        print(f"    K={k:2d}: {k_dist[k]:>4} ({pct:5.1f}%) {bar}")
    chosen = statistics.mean(actual_k_vals) if actual_k_vals else 0
    print(f"    Average K chosen: {chosen:.2f}")

    accepted_vals = [s["num_accepted"] for s in steps]
    avg_acc = sum(accepted_vals) / n
    print(f"    Average k_accepted: {avg_acc:.2f}")

    # ── M2: Per-position conditional acceptance ──
    print(f"\n  ── M2: PER-POSITION CONDITIONAL ACCEPTANCE (α_i) ──")
    final_step = steps[-1]
    pre = final_step.get("pos_reached", [])
    pac = final_step.get("pos_accepted", [])
    if pre:
        for i in range(len(pre)):
            if pre[i] > 0:
                rate = pac[i] / pre[i]
                print(f"    pos[{i:2d}] α={rate:.4f}  ({pac[i]:>4}/{pre[i]:>4})")
            else:
                print(f"    pos[{i:2d}] α=     N/A  (0/0)")

    # ── M3: E_acc predicted vs realized ──
    print(f"\n  ── M3: E_acc PREDICTED vs REALIZED ──")
    mape_values = []
    for s in steps:
        ema = s.get("ema", [])
        ak = s.get("adaptive_k_fs", 5) or 5
        actual_acc = s.get("num_accepted", 0)
        if not ema:
            continue
        # Compute E_acc(K)
        prod = 1.0
        e_acc = 1.0
        for i in range(min(ak, len(ema))):
            a = max(0.001, min(0.999, ema[i]))
            prod *= a
            e_acc += prod
        if actual_acc > 0 or e_acc > 0.001:
            err = abs(e_acc - actual_acc) / max(actual_acc, 0.001)
            mape_values.append(err)
    if mape_values:
        print(f"    Steps with data: {len(mape_values)}")
        print(f"    Median APE: {statistics.median(mape_values)*100:.1f}%")
        if len(mape_values) > 10:
            p90 = sorted(mape_values)[int(len(mape_values)*0.9)]
            print(f"    90th percentile APE: {p90*100:.1f}%")
            print(f"    Mean APE: {statistics.mean(mape_values)*100:.1f}%")

    # ── M4: Goodput(K) curve (last step snapshot) ──
    print(f"\n  ── M4: GOODPUT(K) CURVE (last step) ──")
    last = steps[-1]
    gc = last.get("goodput_curve", [])
    if gc:
        best_k_gc = max(range(len(gc)), key=lambda i: gc[i]) + 1
        print(f"    {'K':>3}  {'Goodput':>10}")
        for i, g in enumerate(gc):
            marker = " ← best" if i+1 == best_k_gc else ""
            print(f"    K={i+1:2d}  {g:>8.4f}{marker}")

    # ── M5: Per-prompt acceptance (approximate: all prompts together since
    #       we don't track prompt type in the scheduler) ──
    print(f"\n  ── M5: PER-PROMPT ACCEPTANCE (overall, all 10 prompts) ──")
    print(f"    (Requires prompt-type grouping for per-type breakdown)")
    print(f"    Average accepted: {avg_acc:.2f}")

    # ── M6: Exploration outcomes ──
    print(f"\n  ── M6: EXPLORATION OUTCOMES ──")
    expl_steps = [s for s in steps if s.get("was_exploration")]
    if expl_steps:
        expl_k = [s.get("adaptive_k_fs", 5) or 5 for s in expl_steps]
        expl_acc = [s["num_accepted"] for s in expl_steps]
        print(f"    {len(expl_steps)} exploration steps ({len(expl_steps)/n*100:.1f}%)")
        print(f"    Avg accepted at max K: {statistics.mean(expl_acc):.1f}")
        # How often did exploration discover useful high-K?
        useful = sum(1 for a in expl_acc if a >= 2)
        print(f"    Useful (≥2 accepted): {useful}/{len(expl_steps)} ({useful/len(expl_steps)*100:.0f}%)")
    else:
        print(f"    0 exploration steps — ε-greedy may not be triggering")

    # ── M7: K stability ──
    print(f"\n  ── M7: K STABILITY ──")
    stability_runs = []
    run_k = None
    run_len = 0
    for k in actual_k_vals:
        if k == run_k:
            run_len += 1
        else:
            if run_k is not None:
                stability_runs.append((run_k, run_len))
            run_k = k
            run_len = 1
    if run_k is not None:
        stability_runs.append((run_k, run_len))
    print(f"    Total K changes: {len(stability_runs)-1}")
    avg_stable = statistics.mean([r[1] for r in stability_runs]) if stability_runs else 0
    print(f"    Avg consecutive steps at same K: {avg_stable:.1f}")
    if stability_runs:
        print(f"    Longest runs:")
        for k, cnt in sorted(stability_runs, key=lambda x: -x[1])[:5]:
            qual = " ⬿ exploration" if k == steps[0].get("num_draft_tokens", 5) else ""
            print(f"      K={k}: {cnt} steps{qual}")

    # ── M8: Cooldown rate ──
    print(f"\n  ── M8: COOLDOWN RATE ──")
    cd_steps = sum(1 for s in steps if s.get("in_cooldown"))
    print(f"    Steps in cooldown: {cd_steps} / {n} ({cd_steps/n*100:.1f}%)")

    # Optimality check (adaptive vs acceptance)
    print(f"\n  ── OPTIMALITY CHECK ──")
    over = sum(1 for k, a in zip(actual_k_vals, accepted_vals) if k > a + 1)
    under = sum(1 for k, a in zip(actual_k_vals, accepted_vals) if k < a + 1)
    exact = sum(1 for k, a in zip(actual_k_vals, accepted_vals) if k == a + 1 or k == a)
    print(f"    K ≈ optimal   (k ≈ accept+1): {exact:>4} ({exact/n*100:5.1f}%)")
    print(f"    Over-spec     (k > accept+1): {over:>4} ({over/n*100:5.1f}%)")
    print(f"    Under-spec    (k < accept+1): {under:>4} ({under/n*100:5.1f}%)")


def main():
    results = []
    for label, extra in [("Fixed K=5", FIXED_CFG), ("Adaptive K", ADAPTIVE_CFG)]:
        print(f"\n{'='*65}\n  RUNNING: {label}\n{'='*65}")
        r = run_single(label, extra)
        if r:
            results.append(r)
            print(f"  Throughput: {r['throughput']:.2f} tok/s | TPOT: {r['tpot_ms']:.2f} ms/tok")
        stats = load_stats()
        sd_steps = [s for s in stats if s.get("step_type") == "sd_step_diag"]
        if label == "Fixed K=5":
            report_fixed(sd_steps)
        else:
            report_adaptive(sd_steps)

    if len(results) == 2:
        print(f"\n{'='*65}")
        print(f"  M9: THROUGHPUT COMPARISON")
        print(f"{'='*65}")
        print(f"  {'Config':<20} {'TPOT':>10} {'Throughput':>12}")
        print(f"  {'-'*44}")
        for r in results:
            print(f"  {r['label']:<20} {r['tpot_ms']:>8.2f} ms/tok  {r['throughput']:>8.2f} tok/s")
        speedup = (results[1]["throughput"] / results[0]["throughput"] - 1) * 100
        print(f"\n  Adaptive K vs Fixed K=5: {speedup:+.1f}% throughput")


if __name__ == "__main__":
    main()
