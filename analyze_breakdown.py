"""
Per-step acceptance breakdown per chosen K.
Runs Fixed K=5 and Adaptive K, captures per-step stats, prints detailed tables.
"""

import os, sys, json, subprocess, tempfile

sys.path.insert(0, os.path.expanduser("~/vllm-adaptive-k"))
from bench_adaptive_k import COMMON_KWARGS, FIXED_CFG, ADAPTIVE_CFG, PROMPTS


def py_json(obj) -> str:
    """JSON string with Python-style booleans/none."""
    s = json.dumps(obj, indent=2)
    s = s.replace(": false", ": False")
    s = s.replace(": true", ": True")
    s = s.replace(": null", ": None")
    return s


def make_runner_code(label: str, extra_args: dict) -> str:
    kwargs = {**COMMON_KWARGS, **extra_args}
    args_str = py_json(kwargs)
    prompts_str = json.dumps(PROMPTS, indent=2)
    return f"""\
import os, json, time, sys
from vllm import LLM, SamplingParams
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

llm = LLM(**{args_str})
sp = SamplingParams(temperature=0.0, max_tokens=128)

t0 = time.time()
outputs = llm.generate({prompts_str}, sp)
elapsed = time.time() - t0

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
tpot_ms = (elapsed / total_tokens) * 1000
throughput = total_tokens / elapsed

result = {{
    "label": "{label}",
    "elapsed_s": round(elapsed, 3),
    "total_tokens": total_tokens,
    "tpot_ms": round(tpot_ms, 2),
    "throughput": round(throughput, 2),
}}
print("__BENCH_RESULT__" + json.dumps(result) + "__END__")
"""


def run_single(label: str, extra_args: dict, stats_file: str) -> dict:
    code = make_runner_code(label, extra_args)
    env = {**os.environ, "VLLM_PER_STEP_STATS": stats_file}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=600,
        env=env,
    )
    if proc.returncode != 0:
        print(f"[{label}] STDERR:", proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Benchmark '{label}' failed with code {proc.returncode}")
    for line in proc.stdout.splitlines():
        if "__BENCH_RESULT__" in line:
            return json.loads(line.split("__BENCH_RESULT__")[1].split("__END__")[0])
    raise RuntimeError(f"No result in output for '{label}'")


def load_stats(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def print_breakdown(data: list[dict], label: str, is_adaptive: bool = False):
    if not data:
        print(f"\n  [No per-step data for {label}]")
        return
    total = len(data)

    if is_adaptive:
        by_k = {}
        for r in data:
            by_k.setdefault(r["k_chosen"], []).append(r)

        print(f"\n  ── Adaptive K selection frequency ──")
        print(f"  {'K':>3s} | {'steps':>5s} | {'pct':>6s}")
        print(f"  {'─'*3}-+-{'─'*5}-+-{'─'*6}")
        for k in sorted(by_k):
            n = len(by_k[k])
            print(f"  {k:>3d} | {n:>5d} | {n/total*100:>5.1f}%")

        print(f"\n  ── Acceptance breakdown per chosen K ──")
        hdr = f"  {'K':>3s} | {'steps':>5s} |"
        for a in range(6):
            hdr += f" {'acc='+str(a):>8s}"
        hdr += f" | {'all-K':>7s}"
        print(hdr)
        print(f"  {'─'*3}-+-{'─'*5}-+-" + "-+-".join(["─"*8]*6 + ["─"*7]))

        for k in sorted(by_k):
            rows = by_k[k]
            n = len(rows)
            counts = {}
            for r in rows:
                counts[r["num_accepted"]] = counts.get(r["num_accepted"], 0) + 1
            line = f"  {k:>3d} | {n:>5d} |"
            for a in range(6):
                cnt = counts.get(a, 0)
                line += f" {cnt/n*100:>7.1f}%"
            all_k = counts.get(k, 0)
            line += f" |  {all_k/n*100:>5.1f}%"
            print(line)
    else:
        counts = {}
        for r in data:
            counts[r["num_accepted"]] = counts.get(r["num_accepted"], 0) + 1

        print(f"\n  ── Fixed K=5 Acceptance Distribution ──")
        print(f"  {'acceptance':>10s} | {'steps':>5s} | {'pct':>6s}")
        print(f"  {'─'*10}-+-{'─'*5}-+-{'─'*6}")
        for a in range(6):
            cnt = counts.get(a, 0)
            pct_label = "all 5 ✅" if a == 5 else f""
            print(f"  {'acc='+str(a):>10s} | {cnt:>5d} | {cnt/total*100:>5.1f}%" + (f"  ← {pct_label}" if pct_label else ""))


def main():
    tmpdir = tempfile.gettempdir()
    fixed_file = os.path.join(tmpdir, "stats_fixed.jsonl")
    adapt_file = os.path.join(tmpdir, "stats_adaptive.jsonl")
    for f in [fixed_file, adapt_file]:
        if os.path.exists(f):
            os.remove(f)

    print("=" * 65)
    print("  Fixed K=5")
    print("=" * 65)
    res_f = run_single("Fixed K=5", FIXED_CFG, fixed_file)
    print(f"  ✓ {res_f['throughput']:.1f} tok/s  ({res_f['tpot_ms']:.2f} ms/tok)")

    print("\n" + "=" * 65)
    print("  Adaptive K")
    print("=" * 65)
    res_a = run_single("Adaptive K", ADAPTIVE_CFG, adapt_file)
    print(f"  ✓ {res_a['throughput']:.1f} tok/s  ({res_a['tpot_ms']:.2f} ms/tok)")

    fixed_data = load_stats(fixed_file)
    adapt_data = load_stats(adapt_file)

    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    print(f"\n  ── Throughput ──")
    print(f"  {'Config':<20s} {'TPOT':>6s}    {'Throughput':>10s}")
    print(f"  {'─'*20} {'─'*6}    {'─'*10}")
    for label, res in [("Fixed K=5", res_f), ("Adaptive K", res_a)]:
        print(f"  {label:<20s} {res['tpot_ms']:>5.2f} ms/tok    {res['throughput']:>6.2f} tok/s")

    print_breakdown(fixed_data, "Fixed K=5", is_adaptive=False)
    print_breakdown(adapt_data, "Adaptive K", is_adaptive=True)

    speedup = ((res_a['throughput'] / res_f['throughput']) - 1) * 100
    print(f"\n  Adaptive K vs Fixed K=5: {speedup:+.2f}%")


if __name__ == "__main__":
    main()
