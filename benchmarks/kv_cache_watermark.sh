#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Reproducible demonstration of the KV cache watermark (`--watermark`) for
# reducing preemption thrashing.
#
# The watermark is the fraction of total KV cache blocks the scheduler keeps
# free when admitting a waiting/preempted request into the running queue.
#
# Why this workload triggers thrashing:
#   Requests are admitted based on the KV cache they need *at admission time*.
#   With `--scheduler-reserve-full-isl` (default) the input length is reserved up
#   front, but the *output* length is unknown and unreserved. A decode-heavy
#   workload (output >> input) at high concurrency therefore over-admits while
#   requests are short, then runs out of KV cache as they all grow during decode
#   -> the scheduler preempts (recompute) recently-admitted requests, re-prefills
#   them later, and repeats. The watermark keeps a block of KV cache free so
#   running requests can grow into it instead of triggering this churn.
#
# This script launches `vllm serve` under a deliberately KV-constrained config
# and a decode-heavy workload, sweeping the watermark across several values, and
# reports the preemption count (scraped from /metrics), throughput, and latency
# percentiles for each. It then plots the results.
#
# Default workload: concurrency 200, input ~300 tokens, output ~4000 tokens
# (+/- 20% variance), sized to run each config for ~5 minutes.
#
# Usage:
#   benchmarks/kv_cache_watermark.sh
#   MODEL=Qwen/Qwen2.5-14B-Instruct TP=2 benchmarks/kv_cache_watermark.sh
#
# Run inside the vLLM virtualenv (so `vllm` and `python` resolve to it).
set -euo pipefail

# ---- Config (override via environment) -------------------------------------
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TP=${TP:-1}
PORT=${PORT:-8000}
URL="http://127.0.0.1:${PORT}"
# Constrain the KV cache to a *near-critical* size: large enough that the engine
# can run stably, but small enough that greedy over-admission tips it into
# preemption thrashing. (Independent of GPU size, so the demo is reproducible.)
# At the default workload this fits ~1.5x the mean concurrent KV demand.
KV_CACHE_MEMORY_GB=${KV_CACHE_MEMORY_GB:-16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
# Optional weight loader (e.g. fastsafetensors on the GCP cluster).
LOAD_FORMAT=${LOAD_FORMAT:-auto}
# Decode-heavy workload: moderate input, long output, with length variance. The
# long output means preempted requests have generated a lot before eviction, so
# resuming them re-prefills a long sequence (high recomputation cost).
INPUT_LEN=${INPUT_LEN:-1000}
OUTPUT_LEN=${OUTPUT_LEN:-5000}
RANGE_RATIO=${RANGE_RATIO:-0.2}
CONCURRENCY=${CONCURRENCY:-128}
# Enough prompts to keep each config saturated for ~5+ minutes.
NUM_PROMPTS=${NUM_PROMPTS:-450}
OUTDIR=${OUTDIR:-./watermark_bench_results}
# Watermark fractions compared. "label value" per line; value=0 disables it.
CONFIGS=${CONFIGS:-"off    0
w0.02  0.02
w0.05  0.05
w0.10  0.10
w0.15  0.15"}

KV_CACHE_MEMORY_BYTES=$((KV_CACHE_MEMORY_GB * 1024 * 1024 * 1024))
mkdir -p "$OUTDIR"

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

scrape_preemptions() {
  # Sum the vllm:num_preemptions_total counter across engines.
  python - "${URL}/metrics" <<'PY'
import sys, urllib.request
total = 0.0
try:
    body = urllib.request.urlopen(sys.argv[1], timeout=10).read().decode("utf-8", "replace")
    for line in body.splitlines():
        if line.startswith("vllm:num_preemptions_total"):
            total += float(line.rsplit(" ", 1)[-1])
except Exception as e:  # noqa: BLE001
    print(f"scrape error: {e}", file=sys.stderr)
print(int(total))
PY
}

wait_for_server() {
  for _ in $(seq 1 300); do
    if curl -s "${URL}/health" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "ERROR: server process exited during startup" >&2; return 1
    fi
    sleep 5
  done
  echo "ERROR: server did not become ready" >&2; return 1
}

run_one() {
  local label=$1 watermark=$2
  echo
  echo "==================== watermark: ${label} (${watermark}) ===================="
  vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --load-format "$LOAD_FORMAT" \
    --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --no-enable-prefix-caching \
    --watermark "$watermark" \
    --port "$PORT" >"${OUTDIR}/serve_${label}.log" 2>&1 &
  SERVER_PID=$!
  wait_for_server
  sleep 5

  local pre post
  pre=$(scrape_preemptions)
  vllm bench serve \
    --backend vllm \
    --base-url "$URL" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio "$RANGE_RATIO" \
    --ignore-eos \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --metric-percentiles "50,90,99" \
    --save-result \
    --result-dir "$OUTDIR" \
    --result-filename "bench_${label}.json"
  post=$(scrape_preemptions)
  echo "${label} ${watermark} $((post - pre))" >>"${OUTDIR}/preemptions.txt"

  kill "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do curl -s "${URL}/health" >/dev/null 2>&1 || break; sleep 2; done
  SERVER_PID=""
  sleep 10
}

: >"${OUTDIR}/preemptions.txt"
while read -r label watermark; do
  [[ -z "${label:-}" ]] && continue
  run_one "$label" "$watermark"
done <<<"$CONFIGS"

echo
echo "==================== summary ===================="
python - "$OUTDIR" <<'PY'
import json, os, sys
outdir = sys.argv[1]
pre = {}
order = []
for line in open(os.path.join(outdir, "preemptions.txt")):
    label, watermark, n = line.split()
    pre[label] = (float(watermark), int(n))
    order.append(label)

def g(d, *names):
    for n in names:
        if d.get(n) is not None:
            return d[n]
    return float("nan")

cols = ["watermark", "frac", "preempt", "out_tok/s", "req/s",
        "TTFT_p50", "TTFT_p99", "ITL_p99", "E2EL_p50"]
print("  ".join(f"{c:>10}" for c in cols))
rows = []
for label in order:
    watermark, n = pre[label]
    d = json.load(open(os.path.join(outdir, f"bench_{label}.json")))
    rows.append(dict(
        label=label, watermark=watermark, preempt=n,
        out_tok_s=g(d, "output_throughput"),
        req_s=g(d, "request_throughput"),
        ttft_p50=g(d, "p50_ttft_ms", "median_ttft_ms"),
        ttft_p99=g(d, "p99_ttft_ms"),
        itl_p99=g(d, "p99_itl_ms"),
        e2el_p50=g(d, "p50_e2el_ms", "median_e2el_ms"),
    ))
    print("  ".join(f"{str(v):>10}" for v in [
        label, watermark, n,
        f"{rows[-1]['out_tok_s']:.0f}",
        f"{rows[-1]['req_s']:.3f}",
        f"{rows[-1]['ttft_p50']/1000:.2f}",
        f"{rows[-1]['ttft_p99']/1000:.2f}",
        f"{rows[-1]['itl_p99']:.2f}",
        f"{rows[-1]['e2el_p50']/1000:.1f}",
    ]))
print("\n(TTFT/E2EL in seconds; ITL in ms. Lower preempt is better.)")

# ---- Plot -------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # noqa: BLE001
    print(f"\n(skip plot: matplotlib unavailable: {e})")
    sys.exit(0)

x = [r["watermark"] for r in rows]
xt = [f"{r['watermark']:g}\n({r['label']})" for r in rows]
idx = list(range(len(rows)))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    f"KV cache watermark sweep — {os.path.basename(os.path.abspath(outdir))}",
    fontsize=12,
)

ax = axes[0][0]
ax.bar(idx, [r["preempt"] for r in rows], color="tab:red")
ax.set_title("Preemptions (lower is better)")
ax.set_ylabel("preemptions")
ax.set_xticks(idx); ax.set_xticklabels(xt)

ax = axes[0][1]
ax.plot(idx, [r["out_tok_s"] for r in rows], "o-", color="tab:green")
ax.set_title("Output throughput (higher is better)")
ax.set_ylabel("tokens/s")
ax.set_xticks(idx); ax.set_xticklabels(xt)

ax = axes[1][0]
ax.plot(idx, [r["itl_p99"] for r in rows], "o-", color="tab:blue")
ax.set_title("Inter-token latency p99 (lower is better)")
ax.set_ylabel("ITL p99 (ms)")
ax.set_xlabel("watermark fraction")
ax.set_xticks(idx); ax.set_xticklabels(xt)

ax = axes[1][1]
ax.plot(idx, [r["ttft_p50"] / 1000 for r in rows], "o-", label="TTFT p50")
ax.plot(idx, [r["ttft_p99"] / 1000 for r in rows], "o-", label="TTFT p99")
ax.plot(idx, [r["e2el_p50"] / 1000 for r in rows], "o-", label="E2EL p50")
ax.set_title("Latency (lower is better)")
ax.set_ylabel("seconds")
ax.set_xlabel("watermark fraction")
ax.set_xticks(idx); ax.set_xticklabels(xt)
ax.legend()

fig.tight_layout(rect=(0, 0, 1, 0.95))
out_png = os.path.join(outdir, "watermark_results.png")
fig.savefig(out_png, dpi=120)
print(f"\nWrote plot: {out_png}")
PY
