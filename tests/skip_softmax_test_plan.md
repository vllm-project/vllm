# Skip-Softmax Threshold Testing Plan

This document describes how to test the
`skip_softmax_threshold_scale_factor_prefill` and
`skip_softmax_threshold_scale_factor_decode` features across three models,
several threshold configurations (including mixed prefill/decode pairs), two
KV cache dtypes, multiple ISL/concurrency combinations, and three accuracy
benchmarks (GSM8K, MMLU Pro, LongBench-E).

## Prerequisites

```bash
# Install lm-eval harness for accuracy benchmarks
uv pip install "lm-eval[api]>=0.4.11"

# Extra deps required by the LongBench-E accuracy tasks.
#   - jieba, fuzzywuzzy, rouge: pulled in (transitively) by lm_eval's
#     longbench metric module; without them lm_eval aborts at task-load
#     time with an ImportError pointing at this list.
#   - python-Levenshtein: optional but strongly recommended — without it
#     fuzzywuzzy falls back to a pure-Python SequenceMatcher and prints
#     a warning on every import, and the F1/fuzzy scoring is noticeably
#     slower on the longer contexts.
uv pip install jieba fuzzywuzzy rouge python-Levenshtein

# Make sure the vLLM CLI is available
vllm --help
```

---

## Test Matrix

| Dimension | Values |
| --------- | ------ |
| **Models** | `nvidia/DeepSeek-R1-NVFP4` (MLA) · `nvidia/DeepSeek-V3.2-NVFP4` (sparse MLA) · `openai/gpt-oss-120b` (GQA) |
| **Parallelism** | Attention DP (`-dp 8`) + Expert Parallel (`--enable-expert-parallel` for MoE models) |
| **KV cache dtype** | `auto` (default) · `fp8` |
| **Thresholds** | `prefill:decode` pairs — `none:none` · `none:0.2` · `0.2:0.2` · `0.4:0.4` (where `none` means the flag is not passed) |
| **ISL** | 1024 · 8192 · 65536 · 131072 |
| **OSL** | 500 |
| **Concurrency** | 1 · 4 · 16 · 64 |
| **Accuracy tasks** | GSM8K (5-shot) · MMLU Pro (5-shot) · LongBench-E (0-shot, 13 length-stratified subtasks) |

---

## 1 — Performance Benchmarks

> **The runnable scripts are already extracted as files alongside this plan.**
> Run the full sweep with:
>
> ```bash
> bash tests/skip_softmax_run_all.sh
> ```
>
> It sources `tests/skip_softmax_perf.sh` (§1.2) and
> `tests/skip_softmax_accuracy.sh` (§2) automatically.

### 1.1 Start the server (orchestration script)

Each model × threshold × kv-cache-dtype combination needs its own server
instance. The orchestration script (`tests/skip_softmax_run_all.sh`) kills
the previous server before starting a new one.

```bash
#!/usr/bin/env bash
set -euo pipefail

# Skip-Softmax full test sweep
# Starts a server per (model × threshold × kv-cache-dtype) combination,
# runs perf + accuracy benchmarks, then tears the server down.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── tunables ──────────────────────────────────────────────────────
MODELS=(
  "nvidia/DeepSeek-R1-NVFP4"
  "nvidia/DeepSeek-V3.2-NVFP4"
  "openai/gpt-oss-120b"
)

# MoE models get --enable-expert-parallel; dense models do not
MOE_MODELS=("nvidia/DeepSeek-R1-NVFP4" "nvidia/DeepSeek-V3.2-NVFP4")

# Threshold combinations to sweep, formatted as "<prefill>:<decode>".
# Use "none" for either side to mean "don't pass that flag at all".
THRESHOLD_PAIRS=(
  "none:none"
  "none:0.2"
  "0.2:0.2"
  "0.4:0.4"
)

# KV cache data types to test
KV_CACHE_DTYPES=("auto" "fp8")

DP=4                 # attention data-parallel size – adjust for your node
PORT=8000
MAX_MODEL_LEN=140000 # must cover the largest ISL + OSL
HEALTH_TIMEOUT=120   # seconds to wait for the server to become healthy
# ──────────────────────────────────────────────────────────────────

is_moe_model() {
  local model="$1"
  for m in "${MOE_MODELS[@]}"; do
    [[ "$model" == "$m" ]] && return 0
  done
  return 1
}

wait_for_server() {
  local port="$1"
  local timeout="$2"
  echo "Waiting for server on port ${port} (timeout ${timeout}s) …"
  for i in $(seq 1 "$timeout"); do
    if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
      echo "Server ready after ~${i}s"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: Server did not become healthy within ${timeout}s" >&2
  return 1
}

for MODEL in "${MODELS[@]}"; do
  for KV_DTYPE in "${KV_CACHE_DTYPES[@]}"; do
    for PAIR in "${THRESHOLD_PAIRS[@]}"; do

      THRESH_PREFILL="${PAIR%%:*}"
      THRESH_DECODE="${PAIR##*:}"

      echo "============================================================"
      echo "MODEL         : $MODEL"
      echo "KV_DTYPE      : $KV_DTYPE"
      echo "THRESH prefill: $THRESH_PREFILL"
      echo "THRESH decode : $THRESH_DECODE"
      echo "============================================================"

      # ── build the extra flags ──
      EXTRA_FLAGS=()
      if [[ "$THRESH_PREFILL" != "none" ]]; then
        EXTRA_FLAGS+=(
          --attention-config.skip_softmax_threshold_scale_factor_prefill \
          "$THRESH_PREFILL"
        )
      fi
      if [[ "$THRESH_DECODE" != "none" ]]; then
        EXTRA_FLAGS+=(
          --attention-config.skip_softmax_threshold_scale_factor_decode \
          "$THRESH_DECODE"
        )
      fi

      # ── expert parallel for MoE models ──
      if is_moe_model "$MODEL"; then
        EXTRA_FLAGS+=(--enable-expert-parallel)
      fi

      # ── launch server in background ──
      VLLM_ENGINE_READY_TIMEOUT_S=7200 vllm serve "$MODEL" \
        --data-parallel-size "$DP" \
        --kv-cache-dtype "$KV_DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --port "$PORT" \
        --trust-remote-code \
        --no-enable-log-requests \
        "${EXTRA_FLAGS[@]}" &
      SERVER_PID=$!

      # ── wait for the server to be ready (fail if it never is) ──
      if ! wait_for_server "$PORT" "$HEALTH_TIMEOUT"; then
        echo "Killing unhealthy server (PID $SERVER_PID) and skipping this config."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        continue
      fi

      # ── run perf benchmarks (see §1.2) ──
      bash "${SCRIPT_DIR}/skip_softmax_perf.sh" \
        "$MODEL" "$THRESH_PREFILL" "$THRESH_DECODE" "$KV_DTYPE" "$PORT"

      # ── run accuracy benchmarks (see §2) ──
      bash "${SCRIPT_DIR}/skip_softmax_accuracy.sh" \
        "$MODEL" "$THRESH_PREFILL" "$THRESH_DECODE" "$KV_DTYPE" "$PORT"

      # ── tear down ──
      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
      echo "Server stopped."
      echo ""

    done
  done
done
```

### 1.2 Throughput / latency sweep

File: **`tests/skip_softmax_perf.sh`** (already created):

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="$1"
THRESH_PREFILL="$2"
THRESH_DECODE="$3"
KV_DTYPE="$4"
PORT="$5"

ISLS=(1024 8192 65536 131072)
OSL=500
CONCURRENCIES=(1 4 16 64)
NUM_PROMPTS=100           # adjust for longer runs
RESULTS_DIR="results/perf"
mkdir -p "$RESULTS_DIR"

# Sanitise model name for filenames
MODEL_TAG="${MODEL//\//_}"
THRESH_TAG="thresh-pf${THRESH_PREFILL}-dc${THRESH_DECODE}"

for ISL in "${ISLS[@]}"; do
  for CONC in "${CONCURRENCIES[@]}"; do
    TAG="${MODEL_TAG}_${THRESH_TAG}_kvdtype-${KV_DTYPE}_isl-${ISL}_osl-${OSL}_conc-${CONC}"
    echo ">>> $TAG"

    vllm bench serve \
      --model "$MODEL" \
      --backend openai \
      --port "$PORT" \
      --endpoint /v1/completions \
      --dataset-name random \
      --input-len "$ISL" \
      --output-len "$OSL" \
      --num-prompts "$NUM_PROMPTS" \
      --max-concurrency "$CONC" \
      --ignore-eos \
      --save-result \
      --result-dir "$RESULTS_DIR" \
      --result-filename "${TAG}.json" \
      2>&1 | tee "${RESULTS_DIR}/${TAG}.log"

    echo ""
  done
done
```

---

## 2 — Accuracy Benchmarks

File: **`tests/skip_softmax_accuracy.sh`** (already created). The script
honours a `SKIP_SOFTMAX_ACCURACY_TASKS` env var (space-separated list of
task keys: `gsm8k`, `mmlu_pro`, `longbench_e`). If unset, all three run:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="$1"
THRESH_PREFILL="$2"
THRESH_DECODE="$3"
KV_DTYPE="$4"
PORT="$5"

TASKS="${SKIP_SOFTMAX_ACCURACY_TASKS:-gsm8k mmlu_pro longbench_e}"

RESULTS_DIR="results/accuracy"
mkdir -p "$RESULTS_DIR"

MODEL_TAG="${MODEL//\//_}"
THRESH_TAG="thresh-pf${THRESH_PREFILL}-dc${THRESH_DECODE}"

run_task() {
  local task_key="$1"
  case "$task_key" in
    gsm8k)
      local TAG="${MODEL_TAG}_${THRESH_TAG}_kvdtype-${KV_DTYPE}_gsm8k"
      lm_eval \
        --model local-completions \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=16,tokenized_requests=False" \
        --tasks gsm8k --num_fewshot 5 --limit 250 --batch_size auto \
        --output_path "${RESULTS_DIR}/${TAG}" \
        2>&1 | tee "${RESULTS_DIR}/${TAG}.log"
      ;;
    mmlu_pro)
      local TAG="${MODEL_TAG}_${THRESH_TAG}_kvdtype-${KV_DTYPE}_mmlu_pro"
      lm_eval \
        --model local-completions \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=16,tokenized_requests=False" \
        --tasks mmlu_pro --num_fewshot 5 --limit 250 --batch_size auto \
        --output_path "${RESULTS_DIR}/${TAG}" \
        2>&1 | tee "${RESULTS_DIR}/${TAG}.log"
      ;;
    longbench_e)
      # LongBench-E: 13 length-stratified subtasks (0-4k / 4-8k / 8k+).
      # 0-shot; requires the extra deps listed in the Prerequisites.
      local TAG="${MODEL_TAG}_${THRESH_TAG}_kvdtype-${KV_DTYPE}_longbench_e"
      lm_eval \
        --model local-completions \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=16,tokenized_requests=False" \
        --tasks longbench_tasks_e --num_fewshot 0 --limit 100 --batch_size auto \
        --output_path "${RESULTS_DIR}/${TAG}" \
        2>&1 | tee "${RESULTS_DIR}/${TAG}.log"
      ;;
  esac
}

for t in $TASKS; do run_task "$t"; done
```

### 2.1 Re-running only LongBench-E across all configs

LongBench-E is the long-context accuracy signal that pairs with the
64k/128k ISL rows of the perf matrix. A dedicated driver runs it for every
`(model × threshold × kv-cache-dtype)` combination without re-executing the
perf sweep or GSM8K/MMLU-Pro:

```bash
bash tests/skip_softmax_longbench_only.sh
```

Under the hood it mirrors `skip_softmax_run_all.sh`'s server orchestration
(spin up `vllm serve` per config, wait for `/health`, tear down afterwards),
but it calls `skip_softmax_accuracy.sh` with `SKIP_SOFTMAX_ACCURACY_TASKS=longbench_e`
so only the LongBench block runs. The per-config server log is written to
`results/logs/<tag>_longbench_server.log` and results land in
`results/accuracy/<tag>_longbench_e/` alongside the full-sweep outputs.

Useful ad-hoc variations (no new script needed — just reuse the env var):

```bash
# Against an already-running server on port 8000, run *only* LongBench-E:
SKIP_SOFTMAX_ACCURACY_TASKS="longbench_e" \
  bash tests/skip_softmax_accuracy.sh "$MODEL" none none auto 8000

# GSM8K + LongBench-E, skip MMLU Pro:
SKIP_SOFTMAX_ACCURACY_TASKS="gsm8k longbench_e" \
  bash tests/skip_softmax_accuracy.sh "$MODEL" 10000 500 auto 8000
```

---

## 3 — Quick Smoke Test (single model, single ISL)

Use this to verify the feature end-to-end before launching the
full sweep.

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="openai/gpt-oss-120b"
PORT=8000
DP=8

# ── baseline (no threshold, default kv-cache-dtype) ──
vllm serve "$MODEL" -dp "$DP" --port "$PORT" \
  --max-model-len 16384 --disable-log-requests &
PID=$!
sleep 60  # wait for server

vllm bench serve --model "$MODEL" --port "$PORT" \
  --dataset-name random --input-len 8192 --output-len 500 \
  --num-prompts 20 --max-concurrency 4 --ignore-eos
kill $PID; wait $PID 2>/dev/null

# ── with skip-softmax threshold 0.2 for both prefill and decode,
#    default kv-cache-dtype ──
vllm serve "$MODEL" -dp "$DP" --port "$PORT" \
  --max-model-len 16384 --disable-log-requests \
  --attention-config.skip_softmax_threshold_scale_factor_prefill 0.2 \
  --attention-config.skip_softmax_threshold_scale_factor_decode 0.2 &
PID=$!
sleep 60

vllm bench serve --model "$MODEL" --port "$PORT" \
  --dataset-name random --input-len 8192 --output-len 500 \
  --num-prompts 20 --max-concurrency 4 --ignore-eos
kill $PID; wait $PID 2>/dev/null

# ── with skip-softmax threshold 0.2 (decode-only),
#    fp8 kv-cache-dtype ──
vllm serve "$MODEL" -dp "$DP" --port "$PORT" \
  --max-model-len 16384 --disable-log-requests \
  --kv-cache-dtype fp8 \
  --attention-config.skip_softmax_threshold_scale_factor_decode 0.2 &
PID=$!
sleep 60

vllm bench serve --model "$MODEL" --port "$PORT" \
  --dataset-name random --input-len 8192 --output-len 500 \
  --num-prompts 20 --max-concurrency 4 --ignore-eos
kill $PID; wait $PID 2>/dev/null
```

---

## 4 — Collecting and Comparing Results

After the full sweep, every JSON result lives under `results/`.
The script below tabulates the key metrics.

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "model,threshold,kv_cache_dtype,isl,osl,concurrency,\
mean_ttft_ms,mean_tpot_ms,req_throughput_rps,tok_throughput_tps"

for f in results/perf/*.json; do
  python3 -c "
import json, sys, os
d = json.load(open('$f'))
base = os.path.basename('$f').replace('.json','')
parts = base.split('_')
# parse tag components
model = '/'.join(parts[0:2]) if '/' not in parts[0] else parts[0]
thresh  = [p for p in parts if p.startswith('thresh-')][0].split('-',1)[1]
kvdtype = [p for p in parts if p.startswith('kvdtype-')][0].split('-',1)[1]
isl     = [p for p in parts if p.startswith('isl-')][0].split('-',1)[1]
osl     = [p for p in parts if p.startswith('osl-')][0].split('-',1)[1]
conc    = [p for p in parts if p.startswith('conc-')][0].split('-',1)[1]
ttft    = d.get('mean_ttft_ms', 'N/A')
tpot    = d.get('mean_tpot_ms', 'N/A')
rps     = d.get('request_throughput', 'N/A')
tps     = d.get('output_throughput', 'N/A')
print(f'{model},{thresh},{kvdtype},{isl},{osl},{conc},{ttft},{tpot},{rps},{tps}')
"
done

echo ""
echo "=== Accuracy ==="
echo "model,threshold,kv_cache_dtype,task,score"
for f in results/accuracy/*/*.json results/accuracy/*/results.json; do
  [ -f "$f" ] || continue
  python3 -c "
import json, os
d = json.load(open('$f'))
results = d.get('results', {})
base = os.path.dirname('$f').split('/')[-1]
for task, metrics in results.items():
    acc = metrics.get('acc,none', metrics.get('acc_norm,none', 'N/A'))
    print(f'{base},{task},{acc}')
" 2>/dev/null
done
```

---

## Expected Outcomes

| Threshold | KV dtype | Performance (TTFT / TPOT) | Accuracy (GSM8K / MMLU Pro / LongBench-E) |
| --------- | -------- | ------------------------- | ------------------------------------------ |
| *(unset)* | `auto` | Baseline | Baseline |
| *(unset)* | `fp8` | Baseline (fp8 KV) | Baseline (fp8 KV) |
| `0.0` | `auto` | ≈ baseline (no blocks skipped) | ≈ baseline |
| `0.0` | `fp8` | ≈ fp8 baseline (no blocks skipped) | ≈ fp8 baseline |
| `0.2` | `auto` | Moderate speedup at long ISLs | Slight degradation possible; LongBench-E most sensitive at 8k+ bucket |
| `0.2` | `fp8` | Moderate speedup at long ISLs | Slight degradation possible; LongBench-E most sensitive at 8k+ bucket |
| `0.4` | `auto` | Larger speedup at long ISLs | More noticeable degradation; watch LongBench-E 8k+ bucket closely |
| `0.4` | `fp8` | Larger speedup at long ISLs | More noticeable degradation; watch LongBench-E 8k+ bucket closely |

Key things to look for:

- **TTFT** should decrease at 64k/128k ISL with higher thresholds.
- **TPOT** may also improve at high concurrency + long context.
- **Accuracy** at threshold `0.0` must match baseline exactly (no blocks skipped)
  for both `auto` and `fp8` KV cache dtypes.
- **Accuracy** degradation at `0.2`/`0.4` should be within acceptable bounds
  for the use-case. GSM8K and MMLU Pro are short-context signals; LongBench-E
  is the primary long-context signal and should be read per length bucket
  (0-4k, 4-8k, 8k+) — degradation, if any, typically concentrates in the
  8k+ bucket, which is exactly what the prefill threshold targets.
- **fp8 vs auto**: Compare each threshold across both KV cache dtypes to
  isolate the effect of quantized KV caches on skip-softmax behaviour.
