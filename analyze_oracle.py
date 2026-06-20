"""
Oracle analysis: run K=12, K=5, and Adaptive K to compare acceptance curves.

For each config, collects per-step (req_id, num_accepted, k_chosen) via
VLLM_PER_STEP_STATS env var. Then generates:
  - Per-prompt acceptance curve (steps on X, accepted tokens on Y)
  - Average acceptance curve across all prompts
  - K=12 oracle, K=5 fixed, Adaptive K overlay
  - HTML page with interactive Chart.js plots
"""
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

# ── common benchmark params ─────────────────────────────────────────
TARGET = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"

PROMPTS = [
    # 1-10 (original)
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
    # 11-20
    "Describe the water cycle in four sentences.",
    "Write a function to check if a string is a palindrome.",
    "Explain how photosynthesis works.",
    "What is the Pythagorean theorem? Give an example.",
    "Write a limerick about machine learning.",
    "Explain the difference between HTTP and HTTPS.",
    "What are the main causes of World War I?",
    "Write a Python class for a bank account with deposit and withdraw methods.",
    "Explain how a database index works.",
    "What is the meaning of life according to various philosophers?",
]

ADAPTIVE_CFG = {
    "enable_adaptive_k": True,
    "adaptive_k_ema_alpha": 0.3,
    "adaptive_k_c_draft": 0.1144,
    "adaptive_k_min_tokens": 1,
    "adaptive_k_alpha_prior": 0.85,
    "adaptive_k_cooldown_steps": 2,
}

NP = len(PROMPTS)


def make_runner_code(spec_tokens: int, extra_cfg: dict | None,
                      stats_path: str, label: str) -> str:
    """Generate Python runner subprocess code."""
    kwargs_items = [
        f"    'model': {TARGET!r}",
        f"    'spec_model': {DRAFT!r}",
        f"    'spec_tokens': {spec_tokens}",
        f"    'tensor_parallel_size': 1",
        f"    'gpu_memory_utilization': 0.70",
        f"    'max_model_len': 2048",
        f"    'enforce_eager': True",
        f"    'cpu_offload_gb': 3",
    ]
    if extra_cfg:
        inner = ", ".join(f"{k!r}: {v!r}" for k, v in extra_cfg.items())
        kwargs_items.append(f"    'speculative_config': {{{inner}}}")

    kwargs_str = ",\n".join(kwargs_items)
    prompts_str = ",\n".join(f"    {p!r}" for p in PROMPTS)

    return f'''
import os, json, time, gc, torch
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_PER_STEP_STATS"] = {stats_path!r}
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.llm import LLM

PROMPTS = [
{prompts_str},
]
sp = SamplingParams(temperature=0.0, max_tokens=128, seed=42)

kwargs = {{
{kwargs_str},
}}

llm = LLM(**kwargs)
_ = llm.generate(["Hello world"] * 2, sp)

torch.cuda.synchronize()
t0 = time.perf_counter()
outputs = llm.generate(PROMPTS, sp)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

total_tokens = sum(len(c.token_ids) for o in outputs for c in o.outputs)
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
del llm; gc.collect(); torch.cuda.empty_cache()
'''


def run_config(label: str, spec_tokens: int, extra_cfg: dict | None,
               stats_path: str) -> dict:
    """Run one benchmark config; returns {label, elapsed_s, ...}."""
    code = make_runner_code(spec_tokens, extra_cfg, stats_path, label)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=900,
    )
    # Print stderr for progress
    for line in proc.stderr.strip().split("\n"):
        if any(k in line for k in
               ("AdaptiveK:", "WARNING", "INFO", "ERROR")):
            print(f"  [{label}] {line}", file=sys.stderr)
    # Find last JSON line
    for line in reversed(proc.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    print(f"  STDOUT: {proc.stdout[:500]}", file=sys.stderr)
    print(f"  STDERR (last 2KB): {proc.stderr[-2000:]}", file=sys.stderr)
    raise RuntimeError(f"No JSON result for {label}")


def load_stats(path: str) -> list[dict]:
    """Load per-step stats from JSONL file."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_req(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["req_id"]].append(r)
    # Sort each group by step_idx
    for rid in groups:
        groups[rid].sort(key=lambda x: x["step_idx"])
    return dict(groups)


def compute_avg_curve(groups: dict[str, list[dict]],
                      field: str = "num_accepted") -> tuple[list[int], list[float]]:
    """Compute the average curve across all prompts.

    Returns (positions, avg_values_at_each_position).
    We align by step index within each prompt (position in generation).
    """
    max_span = max(len(rows) for rows in groups.values())
    sums = [0.0] * max_span
    counts = [0] * max_span
    for rid, rows in groups.items():
        for i, r in enumerate(rows):
            if i < max_span:
                sums[i] += r.get(field, 0)
                counts[i] += 1
    avg = []
    positions = []
    for i in range(max_span):
        if counts[i] > 0:
            avg.append(sums[i] / counts[i])
            positions.append(i)
    return positions, avg


def step_to_token_curves(groups: dict[str, list[dict]],
                          max_tokens: int = 128,
                          field: str = "num_accepted",
                          ) -> dict[str, list[float]]:
    """Convert per-step acceptance to per-token curves.

    Each step with num_accepted=N generates 1+N tokens. We broadcast the
    step's value to each token position. All curves padded to max_tokens.
    """
    curves = {}
    for req_id, rows in groups.items():
        token_accepted = []
        for r in rows:
            n = r.get(field, 0)
            if n is None:
                n = 0.0
            step_len = max(1, 1 + int(n))
            for _ in range(step_len):
                token_accepted.append(float(n) if n else 0.0)
        if len(token_accepted) < max_tokens:
            token_accepted.extend([0.0] * (max_tokens - len(token_accepted)))
        else:
            token_accepted = token_accepted[:max_tokens]
        curves[req_id] = token_accepted
    return curves
def main():
    print("=" * 60, file=sys.stderr)
    print("ORACLE ANALYSIS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    results = {}
    stats = {}

    for label, spec_tokens, extra_cfg in [
        ("Oracle K=12", 12, None),
        ("Fixed K=5", 5, None),
        ("Adaptive K", 5, ADAPTIVE_CFG),
    ]:
        print(f"\n  --- {label} ---", file=sys.stderr)
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False) as f:
            spath = f.name
        try:
            r = run_config(label, spec_tokens, extra_cfg, spath)
            results[label] = r
            rows = load_stats(spath)
            stats[label] = {"rows": rows, "groups": group_by_req(rows)}
            print(f"  ✓ {r['throughput']:.1f} tok/s  "
                  f"({r['tpot_ms']:.2f} ms/tok, "
                  f"{len(rows)} spec-decode steps)", file=sys.stderr)
        finally:
            os.unlink(spath)

    # ── Compute curves ────────────────────────────────────────────
    oracle_groups = stats["Oracle K=12"]["groups"]
    fixed_groups = stats["Fixed K=5"]["groups"]
    adapt_groups = stats["Adaptive K"]["groups"]

    # Per-prompt curves for key analysis
    # 1. Average oracle curve
    oracle_pos, oracle_avg = compute_avg_curve(oracle_groups, "num_accepted")
    fixed_pos, fixed_avg = compute_avg_curve(fixed_groups, "num_accepted")
    adapt_pos_accepted, adapt_avg_accepted = compute_avg_curve(
        adapt_groups, "num_accepted")
    adapt_pos_k, adapt_avg_k = compute_avg_curve(adapt_groups, "k_chosen")

    # 2. Overall stats
    all_oracle = [r["num_accepted"] for rows in oracle_groups.values()
                  for r in rows]
    all_fixed = [r["num_accepted"] for rows in fixed_groups.values()
                 for r in rows]
    all_adapt_acc = [r["num_accepted"] for rows in adapt_groups.values()
                     for r in rows]
    all_adapt_k = [r["k_chosen"] for rows in adapt_groups.values()
                   for r in rows]

    avg_oracle = sum(all_oracle) / len(all_oracle) if all_oracle else 0
    avg_fixed = sum(all_fixed) / len(all_fixed) if all_fixed else 0
    avg_adapt_k = sum(all_adapt_k) / len(all_adapt_k) if all_adapt_k else 0
    avg_adapt_acc = sum(all_adapt_acc) / len(all_adapt_acc) if all_adapt_acc else 0

    # distribution of oracle values
    oracle_dist = defaultdict(int)
    for v in all_oracle:
        oracle_dist[v] += 1
    fixed_dist = defaultdict(int)
    for v in all_fixed:
        fixed_dist[v] += 1

    # ── Print report ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print(f"""\n  ── Throughput ──
  {'Config':<20} {'TPOT':>12} {'Throughput':>15}
  {'─'*20} {'─'*12} {'─'*15}""")
    for label in ["Oracle K=12", "Fixed K=5", "Adaptive K"]:
        r = results[label]
        print(f"  {label:<20} {r['tpot_ms']:>10.2f} ms/tok  "
              f"{r['throughput']:>8.2f} tok/s")

    print(f"""\n  ── Average Acceptance Depth ──
  Oracle (K=12): {avg_oracle:.3f} tokens  ← theoretical upper bound
  Fixed  (K=5):  {avg_fixed:.3f} tokens  ({avg_fixed/avg_oracle*100:.0f}% of oracle)
  Adaptive K:    {avg_adapt_acc:.3f} tokens  ({avg_adapt_acc/avg_oracle*100:.0f}% of oracle)
    Avg K chosen: {avg_adapt_k:.2f}""")

    print(f"""\n  ── Cold Start (first 5 vs last 5 steps) ──""")
    # For each prompt, compare first 5 steps vs last 5 steps
    oracle_first = []
    oracle_last = []
    for rid, rows in oracle_groups.items():
        if len(rows) >= 10:
            first5 = [r["num_accepted"] for r in rows[:5]]
            last5 = [r["num_accepted"] for r in rows[-5:]]
            oracle_first.append(sum(first5)/5)
            oracle_last.append(sum(last5)/5)
    if oracle_first:
        print(f"    Oracle avg first 5 steps: {sum(oracle_first)/len(oracle_first):.2f}")
        print(f"    Oracle avg last  5 steps: {sum(oracle_last)/len(oracle_last):.2f}")

    print(f"\n  ── Oracle K=12 Acceptance Distribution ──")
    for k in sorted(oracle_dist):
        print(f"    K={k:<2}: {oracle_dist[k]:>4} ({oracle_dist[k]/len(all_oracle)*100:>5.1f}%)")

    print(f"\n  ── Fixed K=5 Acceptance Distribution ──")
    for k in sorted(fixed_dist):
        print(f"    K={k:<2}: {fixed_dist[k]:>4} ({fixed_dist[k]/len(all_fixed)*100:>5.1f}%)")

    print(f"\n  ── Adaptive K Distribution ──")
    adapt_dist = defaultdict(int)
    for v in all_adapt_k:
        adapt_dist[v] += 1
    for k in sorted(adapt_dist):
        print(f"    K={k:<2}: {adapt_dist[k]:>4} ({adapt_dist[k]/len(all_adapt_k)*100:>5.1f}%)")

    # ── What percent of the time does Adaptive K match oracle? ──
    # For each step in adaptive run, check what the oracle accepted at that step
    # We can't do exact per-step matching since step indices differ between runs.
    # Instead, we compare the distributions.

    # ── Per-token curves (broadcast step acceptance to each token) ──
    oracle_tok_curves = step_to_token_curves(oracle_groups)
    fixed_tok_curves = step_to_token_curves(fixed_groups)
    adapt_tok_curves = step_to_token_curves(adapt_groups)
    adapt_k_tok_curves = step_to_token_curves(adapt_groups, field="k_chosen")

    # Average per-token oracle curve
    n_prompts = len(oracle_tok_curves)
    oracle_tok_avg = [sum(oracle_tok_curves[rid][i] for rid in oracle_tok_curves) / n_prompts
                      for i in range(128)]
    fixed_tok_avg = [sum(fixed_tok_curves[rid][i] for rid in fixed_tok_curves) / n_prompts
                     for i in range(128)]
    adapt_tok_avg = [sum(adapt_tok_curves[rid][i] for rid in adapt_tok_curves) / n_prompts
                     for i in range(128)]
    adapt_k_tok_avg = [sum(adapt_k_tok_curves[rid][i] for rid in adapt_k_tok_curves) / n_prompts
                       for i in range(128)]
    # ── Draft confidence curves (from Adaptive K run) ──
    adapt_confidence_tok = step_to_token_curves(adapt_groups, field="draft_confidence")
    adapt_confidence_avg = [
        sum(adapt_confidence_tok[rid][i] for rid in adapt_confidence_tok) / max(len(adapt_confidence_tok), 1)
        for i in range(128)
    ] if adapt_confidence_tok else [0.0] * 128
    # ── Generate HTML plot ────────────────────────────────────────
    html = generate_html(
        oracle_tok_curves, oracle_tok_avg,
        fixed_tok_curves, fixed_tok_avg,
        adapt_tok_curves, adapt_tok_avg,
        adapt_k_tok_avg,
        adapt_confidence_avg,
        results, avg_oracle, avg_fixed, avg_adapt_acc, avg_adapt_k,
        oracle_dist, fixed_dist, adapt_dist,
    )
    html_path = os.path.expanduser("~/vllm-adaptive-k/oracle_analysis.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\n  📈 HTML plot saved to: {html_path}", file=sys.stderr)

    # ── Key conclusion ────────────────────────────────────────────
    efficiency = avg_adapt_acc / avg_oracle * 100 if avg_oracle > 0 else 0
    fixed_efficiency = avg_fixed / avg_oracle * 100 if avg_oracle > 0 else 0
    print(f"""
  ── VERDICT ──
  Oracle acceptance: {avg_oracle:.2f} tokens/step
  Fixed K=5:         {avg_fixed:.2f} tokens/step ({fixed_efficiency:.0f}% of oracle)
  Adaptive K:        {avg_adapt_acc:.2f} tokens/step ({efficiency:.0f}% of oracle)
  Avg K chosen:      {avg_adapt_k:.2f}

  Adaptive K captures {efficiency:.0f}% of the oracle's acceptance vs
  Fixed K=5 at {fixed_efficiency:.0f}%. """)
    if efficiency >= fixed_efficiency - 5:
        print("  ✅ Adaptive K matches fixed K=5 in acceptance efficiency.")
    else:
        print("  ⚠️  Adaptive K trails fixed K=5 in acceptance efficiency.")
    if avg_adapt_k > 1:
        print("  ✅ K distribution is spread (K=1 is not dominant).")
    else:
        print("  ⚠️  K distribution is stuck at low values.")


def generate_html(
    oracle_tok_curves, oracle_tok_avg,
    fixed_tok_curves, fixed_tok_avg,
    adapt_tok_curves, adapt_tok_avg,
    adapt_k_tok_avg,
    adapt_confidence_avg,
    results, avg_o, avg_f, avg_aa, avg_ak,
    od, fd, ad,
):
    """Generate interactive HTML with per-token oracle acceptance curves."""
    max_pos = 128
    positions = list(range(max_pos))
    pos_json = json.dumps(positions)

    # Per-prompt oracle curves: semi-transparent lines
    prompt_datasets = []
    colors = [
        "255,99,132", "255,159,64", "255,205,86", "75,192,192",
        "54,162,235", "153,102,255", "201,203,207", "231,76,60",
        "46,204,113", "52,152,219", "155,89,182", "241,196,15",
        "230,126,34", "26,188,156", "149,165,166", "243,156,18",
        "142,68,173", "44,62,80", "22,160,133", "39,174,96",
    ]
    for i, rid in enumerate(sorted(oracle_tok_curves.keys())):
        c = colors[i % len(colors)]
        data = oracle_tok_curves[rid]
        prompt_datasets.append(f"""        {{
            label: '',
            data: {json.dumps([round(v,4) for v in data])},
            borderColor: 'rgba({c},0.2)',
            borderWidth: 1,
            pointRadius: 0,
            tension: 0.2,
        }}""")
    prompt_datasets_str = ",\n".join(prompt_datasets)

    # Average curves
    oa_json = json.dumps([round(v, 4) for v in oracle_tok_avg])
    fa_json = json.dumps([round(v, 4) for v in fixed_tok_avg])
    aa_json = json.dumps([round(v, 4) for v in adapt_tok_avg])
    ak_json = json.dumps([round(v, 4) for v in adapt_k_tok_avg])
    # Confidence average
    ac_clean = [v if v is not None else 0.5 for v in adapt_confidence_avg]
    ac_json = json.dumps([round(v, 4) for v in ac_clean])

    # Distribution data
    o_keys = sorted(od)
    o_vals = [od[k] for k in o_keys]
    f_keys = sorted(fd)
    f_vals = [fd[k] for k in f_keys]
    a_keys = sorted(ad)
    a_vals = [ad[k] for k in a_keys]

    r_o = results["Oracle K=12"]
    r_f = results["Fixed K=5"]
    r_a = results["Adaptive K"]

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Oracle Analysis — Per-Token Acceptance Curves</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
  h1 {{ color: #1a1a2e; }}
  .summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 20px 0; }}
  .card {{ background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .card h3 {{ margin: 0 0 8px; font-size: 14px; color: #666; }}
  .card .value {{ font-size: 28px; font-weight: 700; }}
  .card .sub {{ font-size: 13px; color: #888; margin-top: 4px; }}
  .oracle .value {{ color: #e74c3c; }}
  .fixed .value {{ color: #3498db; }}
  .adaptive .value {{ color: #2ecc71; }}
  .chart-container {{ background: white; border-radius: 8px; padding: 16px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 700px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
  .legend {{ font-size: 13px; padding: 10px; background: #f0f0f0; border-radius: 6px; margin-top: 8px; }}
  .legend span {{ display: inline-block; margin-right: 16px; }}
  .legend .line {{ display: inline-block; width: 20px; height: 3px; vertical-align: middle; margin-right: 4px; }}
</style>
</head>
<body>
<h1>Speculative Decoding: Per-Token Oracle Acceptance</h1>
<p>Qwen2.5-7B-Instruct-AWQ + Qwen2.5-0.5B-Instruct-AWQ · RTX 4050 (6GB) · {len(oracle_tok_curves)} prompts</p>

<div class="summary">
  <div class="card oracle">
    <h3>Oracle K=12</h3>
    <div class="value">{r_o["throughput"]:.1f} tok/s</div>
    <div class="sub">{r_o["tpot_ms"]:.2f} ms/tok · {avg_o:.3f} avg accept</div>
  </div>
  <div class="card fixed">
    <h3>Fixed K=5</h3>
    <div class="value">{r_f["throughput"]:.1f} tok/s</div>
    <div class="sub">{r_f["tpot_ms"]:.2f} ms/tok · {avg_f:.3f} avg accept</div>
  </div>
  <div class="card adaptive">
    <h3>Adaptive K</h3>
    <div class="value">{r_a["throughput"]:.1f} tok/s</div>
    <div class="sub">{r_a["tpot_ms"]:.2f} ms/tok · avg K={avg_ak:.2f}, accept={avg_aa:.3f}</div>
  </div>
</div>

<div class="chart-container">
  <h2>Per-Token Oracle Acceptance: Each Prompt vs Average</h2>
  <p style="color:#888;font-size:13px">
    X-axis: token position in generation (0..127) &nbsp;|&nbsp;
    Y-axis: oracle acceptance at the step that produced this token
  </p>
  <div class="legend">
    <span><span class="line" style="background:#e74c3c"></span><strong>Oracle avg</strong></span>
    <span><span class="line" style="background:#e74c3c;opacity:0.2"></span>Individual prompts</span>
    <span><span class="line" style="background:#3498db"></span>Fixed K=5 avg</span>
    <span><span class="line" style="background:#2ecc71"></span>Adaptive K (accepted)</span>
    <span><span class="line" style="background:#2ecc71;border:2px dashed #27ae60"></span>Adaptive K (k chosen)</span>
    <span><span class="line" style="background:#f39c12;border:1px dashed #e67e22"></span>Draft confidence</span>
  </div>
  <canvas id="mainChart" height="360"></canvas>
</div>

<div class="chart-row">
  <div class="chart-container">
    <h2>Acceptance Distribution</h2>
    <canvas id="distChart" height="240"></canvas>
  </div>
  <div class="chart-container">
    <h2>Throughput Comparison</h2>
    <canvas id="throughputChart" height="240"></canvas>
  </div>
</div>

<div class="chart-container">
  <h2>Average: Oracle vs Fixed K=5 vs Adaptive K</h2>
  <canvas id="avgComparison" height="280"></canvas>
</div>

<div class="chart-container">
  <h2>Draft Confidence vs Oracle Acceptance</h2>
  <p style="color:#888;font-size:13px">
    Dual Y-axes: left = oracle acceptance (tokens per step), right = draft model confidence (avg max prob)<br>
    When confidence is high but oracle is low: draft was confident but WRONG → wasted draft compute.<br>
    When confidence is low but oracle is high: draft was uncertain but correct (rare).
  </p>

<script>
new Chart(document.getElementById('mainChart'), {{
  type: 'line',
  data: {{
    labels: {pos_json},
    datasets: [
{prompt_datasets_str},
      {{
        label: 'Oracle avg (K=12)',
        data: {oa_json},
        borderColor: '#e74c3c',
        backgroundColor: 'rgba(231,76,60,0.08)',
        borderWidth: 3,
        fill: true,
        tension: 0.3,
        pointRadius: 0,
      }},
      {{
        label: 'Fixed K=5 avg',
        data: {fa_json},
        borderColor: '#3498db',
        borderWidth: 2.5,
        tension: 0.3,
        pointRadius: 0,
      }},
      {{
        label: 'Adaptive K (k chosen avg)',
        data: {ak_json},
        borderColor: '#2ecc71',
        borderWidth: 2,
        borderDash: [6, 3],
        tension: 0.3,
        pointRadius: 0,
      }},
      {{
        label: 'Adaptive K (accepted avg)',
        data: {aa_json},
        borderColor: '#27ae60',
        borderWidth: 2.5,
        tension: 0.3,
        pointRadius: 0,
      }},
      {{
        label: 'Draft confidence (avg)',
        data: {ac_json},
        borderColor: '#f39c12',
        borderWidth: 2,
        borderDash: [3, 3],
        tension: 0.3,
        pointRadius: 0,
      }},
    ]
  }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ title: {{ display: true, text: 'Token position in generation' }} }},
      y: {{ title: {{ display: true, text: 'Oracle acceptance' }}, min: 0, max: 14 }}
    }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});

new Chart(document.getElementById('distChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(o_keys)},
    datasets: [
      {{
        label: 'Oracle K=12',
        data: {json.dumps(o_vals)},
        backgroundColor: 'rgba(231,76,60,0.7)',
      }},
      {{
        label: 'Fixed K=5',
        data: {json.dumps(f_vals)},
        backgroundColor: 'rgba(52,152,219,0.7)',
      }},
      {{
        label: 'Adaptive K (k_chosen)',
        data: {json.dumps(a_vals)},
        backgroundColor: 'rgba(46,204,113,0.7)',
      }},
    ]
  }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ title: {{ display: true, text: 'Accepted K' }} }},
      y: {{ title: {{ display: true, text: 'Steps' }}, beginAtZero: true }}
    }},
    plugins: {{ legend: {{ position: 'bottom' }} }}
  }}
}});

new Chart(document.getElementById('throughputChart'), {{
  type: 'bar',
  data: {{
    labels: ['Oracle K=12', 'Fixed K=5', 'Adaptive K'],
    datasets: [{{
      label: 'Throughput (tok/s)',
      data: [{r_o["throughput"]}, {r_f["throughput"]}, {r_a["throughput"]}],
      backgroundColor: ['rgba(231,76,60,0.7)', 'rgba(52,152,219,0.7)', 'rgba(46,204,113,0.7)'],
    }}]
  }},
  options: {{
    responsive: true,
    scales: {{ y: {{ title: {{ display: true, text: 'tok/s' }}, beginAtZero: true }} }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});

new Chart(document.getElementById('avgComparison'), {{
  type: 'line',
  data: {{
    labels: {pos_json},
    datasets: [
      {{
        label: 'Oracle avg (K=12)',
        data: {oa_json},
        borderColor: '#e74c3c',
        borderWidth: 2.5,
        fill: true,
        backgroundColor: 'rgba(231,76,60,0.06)',
        tension: 0.3,
        pointRadius: 0,
      }},
      {{
        label: 'Fixed K=5 avg',
        data: {fa_json},
        borderColor: '#3498db',
        borderWidth: 2,
        tension: 0.3,
        pointRadius: 0,
      }},
      {{
        label: 'Adaptive K (accepted)',
        data: {aa_json},
        borderColor: '#27ae60',
        borderWidth: 2,
        tension: 0.3,
        pointRadius: 0,
      }},
      {{
        label: 'Adaptive K (chosen)',
        data: {ak_json},
        borderColor: '#2ecc71',
        borderWidth: 1.5,
        borderDash: [5, 3],
        tension: 0.3,
        pointRadius: 0,
      }},
    ]
  }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ title: {{ display: true, text: 'Token position' }} }},
      y: {{ title: {{ display: true, text: 'Accepted tokens / K chosen' }}, min: 0 }}
    }},
    plugins: {{ legend: {{ position: 'bottom' }} }}
  }}
}});

new Chart(document.getElementById('confidenceVsOracle'), {{
  type: 'line',
  data: {{
    labels: {pos_json},
    datasets: [
      {{
        label: 'Oracle avg acceptance',
        data: {oa_json},
        borderColor: '#e74c3c',
        borderWidth: 2,
        tension: 0.3,
        pointRadius: 0,
        yAxisID: 'y',
      }},
      {{
        label: 'Draft confidence (avg)',
        data: {ac_json},
        borderColor: '#f39c12',
        borderWidth: 2,
        borderDash: [3, 3],
        tension: 0.3,
        yAxisID: 'y1',
        pointRadius: 0,
      }},
    ]
  }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ title: {{ display: true, text: 'Token position' }} }},
      y: {{
        type: 'linear',
        display: true,
        position: 'left',
        title: {{ display: true, text: 'Accepted tokens' }},
        min: 0,
      }},
      y1: {{
        type: 'linear',
        display: true,
        position: 'right',
        title: {{ display: true, text: 'Confidence [0-1]' }},
        min: 0,
        max: 1,
        grid: {{ drawOnChartArea: false }},
      }},
    }},
    plugins: {{ legend: {{ position: 'bottom' }} }}
  }}
}});
</script>
</body>
</html>'''

if __name__ == "__main__":
    main()
