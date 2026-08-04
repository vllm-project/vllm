#!/usr/bin/env bash
# Run long_doc_qa for L2 skip_l1 mode with mock L2 adapter.
#
# This script:
#   1. Kills the existing LMCache MP server
#   2. Relaunches it with L2 config (skip_l1 + mock L2 at 2 GB/s)
#   3. Waits for vLLM to reconnect
#   4. Runs long_doc_qa against baseline (vLLM only) and L2-enabled vLLM
#   5. Verifies L2 query is faster than baseline and warmup overhead is bounded
#
# Expects the following env vars from run-mp-test.sh:
#   VLLM_PORT, VLLM_BASELINE_PORT, MODEL, BUILD_ID, RESULTS_DIR, LMCACHE_DIR,
#   LMCACHE_PORT, CPU_BUFFER_SIZE, MAX_WORKERS, GPU_FOR_VLLM (optional)
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
LMCACHE_DIR="${LMCACHE_DIR:-$REPO_ROOT}"
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
MAX_WORKERS="${MAX_WORKERS:-4}"

DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-10000}"
NUM_DOCUMENTS="${NUM_DOCUMENTS:-30}"
OUTPUT_LEN="${OUTPUT_LEN:-200}"
REPEAT_COUNT="${REPEAT_COUNT:-2}"
REPEAT_MODE="${REPEAT_MODE:-tile}"
SHUFFLE_SEED="${SHUFFLE_SEED:-0}"
MAX_INFLIGHT_REQUESTS="${MAX_INFLIGHT_REQUESTS:-5}"

# Mock L2 config
L2_MAX_SIZE_GB="${L2_MAX_SIZE_GB:-80}"
L2_BANDWIDTH_GB="${L2_BANDWIDTH_GB:-4}"

# L2 performance thresholds
# Recent CI runs show ~1.51-1.67x query speedup, ~1.77-2.02x TTFT speedup,
# and ~0.87-0.99x warmup overhead. Tighten from the previous pass-anything
# thresholds (1.0x/1.0x/2.0x) while leaving headroom for variance.
MIN_L2_SPEEDUP="${MIN_L2_SPEEDUP:-1.3}"
MIN_L2_TTFT_SPEEDUP="${MIN_L2_TTFT_SPEEDUP:-1.5}"
MAX_WARMUP_OVERHEAD="${MAX_WARMUP_OVERHEAD:-1.2}"

L2_RESULTS_DIR="$RESULTS_DIR/long_doc_qa_l2"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
# /metrics is now served by the LMCache FastAPI HTTP server (port 8080
# by default) — the legacy ``--prometheus-port`` standalone server was
# disabled for the ``lmcache server`` entrypoint by #3164.  Defined here
# (not just in Step 4) so the relaunch and the curl scrape agree.
METRICS_HTTP_PORT="${METRICS_HTTP_PORT:-8080}"

echo "=== Long Doc QA L2 Performance Test ==="
echo "Model: $MODEL"
echo "L2 adapter: mock (${L2_MAX_SIZE_GB}GB, ${L2_BANDWIDTH_GB}GB/s)"
echo "Store policy: skip_l1 | Eviction: noop"
echo "Thresholds: speedup>=${MIN_L2_SPEEDUP}x, TTFT speedup>=${MIN_L2_TTFT_SPEEDUP}x, overhead<=${MAX_WARMUP_OVERHEAD}x"
echo "Results: $L2_RESULTS_DIR"
echo ""

mkdir -p "$L2_RESULTS_DIR"

# ---------------------------------------------------------------------------
# Step 1: Kill existing LMCache + vLLM, relaunch both with L2 config
# ---------------------------------------------------------------------------

echo "--- Stopping existing LMCache MP server and vLLM ---"
# PID file layout: line1=LMCache, line2=vLLM w/ LMCache, line3=vLLM baseline.
# These processes were launched by an earlier script (launch-processes.sh)
# and are not children of this shell, so ``wait $pid`` is a no-op here.
# We instead poll until each PID actually exits, then poll until the
# Prometheus port is free, otherwise the LMCache relaunch below would
# fail to bind /metrics and the metrics check would fail spuriously.
if [ -f "$PID_FILE" ]; then
    LMCACHE_PID=$(sed -n '1p' "$PID_FILE")
    VLLM_PID=$(sed -n '2p' "$PID_FILE")
    for pid in $LMCACHE_PID $VLLM_PID; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid"
            kill "$pid" 2>/dev/null || true
            for _ in $(seq 1 60); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.5
            done
            # Last resort: SIGKILL if SIGTERM didn't take after 30s.
            if kill -0 "$pid" 2>/dev/null; then
                echo "PID $pid still alive after SIGTERM; sending SIGKILL"
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
    done
    # Poll until the Prometheus port is fully released so the new server
    # below can bind it cleanly.
    for _ in $(seq 1 30); do
        if ! (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) \
                | awk '{print $4}' | grep -qE ":${METRICS_HTTP_PORT}$"; then
            break
        fi
        sleep 0.5
    done
fi

echo "--- Launching LMCache MP server with L2 config ---"
L2_ADAPTER_JSON="{\"type\":\"mock\",\"max_size_gb\":${L2_MAX_SIZE_GB},\"mock_bandwidth_gb\":${L2_BANDWIDTH_GB}}"

# Determine GPU to use
GPU_DEVICE="${GPU_FOR_VLLM:-0}"

CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
lmcache server \
    --l1-size-gb "$CPU_BUFFER_SIZE" \
    --eviction-policy noop \
    --l2-store-policy skip_l1 \
    --l2-prefetch-policy default \
    --l2-adapter "$L2_ADAPTER_JSON" \
    --max-workers "$MAX_WORKERS" \
    --metrics-sample-rate 1.0 \
    --http-port "$METRICS_HTTP_PORT" \
    --port "$LMCACHE_PORT" \
    > "/tmp/build_${BUILD_ID}_lmcache_l2.log" 2>&1 &

NEW_LMCACHE_PID=$!
echo "LMCache L2 server started (PID=$NEW_LMCACHE_PID)"

echo "Waiting for LMCache L2 to initialize..."
sleep 10

echo "--- Launching vLLM with LMCache ---"
# Compute GPU memory utilization for large GPUs
GPU_MEMORY_UTIL_ARG=""
GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPU_DEVICE}" | tr -d ' ')
GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
if [ "$GPU_MEMORY_GB" -gt 90 ]; then
    GPU_MEMORY_UTIL_ARG="--gpu-memory-utilization 0.5"
fi

# Unset VLLM_PORT in child env so vLLM's torch.distributed picks a free port
env -u VLLM_PORT \
    CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_BATCH_INVARIANT=1 \
    PYTHONHASHSEED=0 \
vllm serve "$MODEL" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 10}}" \
    --attention-backend FLASH_ATTN \
    --port "$VLLM_PORT" \
    --no-async-scheduling \
    $GPU_MEMORY_UTIL_ARG \
    > "/tmp/build_${BUILD_ID}_vllm_l2.log" 2>&1 &

NEW_VLLM_PID=$!
echo "vLLM started (PID=$NEW_VLLM_PID)"

# Update PID file (replace lines 1 and 2, keep baseline on line 3)
if [ -f "$PID_FILE" ]; then
    sed -i "1s/.*/$NEW_LMCACHE_PID/" "$PID_FILE"
    sed -i "2s/.*/$NEW_VLLM_PID/" "$PID_FILE"
else
    echo "$NEW_LMCACHE_PID" > "$PID_FILE"
    echo "$NEW_VLLM_PID" >> "$PID_FILE"
fi

# Wait for vLLM to be ready (needs time to load model)
echo "--- Waiting for vLLM to be ready ---"
if ! wait_for_server "$VLLM_PORT" 300; then
    echo "vLLM failed to start after restart"
    echo "LMCache L2 log (last 50 lines):"
    tail -50 "/tmp/build_${BUILD_ID}_lmcache_l2.log" || true
    echo "vLLM log (last 50 lines):"
    tail -50 "/tmp/build_${BUILD_ID}_vllm_l2.log" || true
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

run_long_doc_qa() {
    local port="$1"
    local result_file="$2"
    local description="$3"

    echo "--- Running long_doc_qa ($description) on port $port ---"
    local output_file="$L2_RESULTS_DIR/${description}_output.txt"

    python3 "$LMCACHE_DIR/benchmarks/long_doc_qa/long_doc_qa.py" \
        --port "$port" \
        --model "$MODEL" \
        --document-length "$DOCUMENT_LENGTH" \
        --num-documents "$NUM_DOCUMENTS" \
        --output-len "$OUTPUT_LEN" \
        --repeat-count "$REPEAT_COUNT" \
        --repeat-mode "$REPEAT_MODE" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --max-inflight-requests "$MAX_INFLIGHT_REQUESTS" \
        --output "$output_file" \
        --json-output \
        2>>"$output_file" | tee "$result_file"

    echo "Completed: $description"
    echo ""
}

extract_json_field() {
    local json_file="$1"
    local field="$2"
    tail -n 1 "$json_file" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    v = data.get('$field')
    print(v if v is not None else 'null')
except Exception:
    print('null')
"
}

# ---------------------------------------------------------------------------
# Step 2: Run benchmarks
# ---------------------------------------------------------------------------

# Phase 1: Baseline -- reuse results from step 5 (same port, same params)
STEP5_BASELINE="$RESULTS_DIR/long_doc_qa/baseline_result.json"
if [ -f "$STEP5_BASELINE" ]; then
    echo "============================================"
    echo "=== Phase 1: Reusing baseline from step 5 ==="
    echo "============================================"
    cp "$STEP5_BASELINE" "$L2_RESULTS_DIR/baseline_result.json"
    echo "Copied baseline results from $STEP5_BASELINE"
    echo ""
else
    echo "============================================"
    echo "=== Phase 1: Baseline vLLM (no LMCache) ==="
    echo "============================================"
    run_long_doc_qa "$VLLM_BASELINE_PORT" "$L2_RESULTS_DIR/baseline_result.json" "baseline"
fi

# Phase 2+3: L2 warmup + query (repeat_count=2, tile mode)
#   Round 1 (warmup): prompts -> L1 write buffer -> L2 store -> L1 delete
#   Round 2 (query):  prompts -> L1 miss -> L2 prefetch -> L1 load -> serve
echo "============================================"
echo "=== Phase 2+3: vLLM + LMCache L2 ==="
echo "============================================"
run_long_doc_qa "$VLLM_PORT" "$L2_RESULTS_DIR/l2_result.json" "l2"

# ---------------------------------------------------------------------------
# Step 3: Verify thresholds
# ---------------------------------------------------------------------------

echo "============================================"
echo "=== Verifying L2 Performance ==="
echo "============================================"

baseline_query_ttft=$(extract_json_field "$L2_RESULTS_DIR/baseline_result.json" "query_ttft_per_prompt")
baseline_query_round_time=$(extract_json_field "$L2_RESULTS_DIR/baseline_result.json" "query_round_time_per_prompt")
baseline_warmup_round_time=$(extract_json_field "$L2_RESULTS_DIR/baseline_result.json" "warmup_round_time_per_prompt")

l2_query_ttft=$(extract_json_field "$L2_RESULTS_DIR/l2_result.json" "query_ttft_per_prompt")
l2_query_round_time=$(extract_json_field "$L2_RESULTS_DIR/l2_result.json" "query_round_time_per_prompt")
l2_warmup_round_time=$(extract_json_field "$L2_RESULTS_DIR/l2_result.json" "warmup_round_time_per_prompt")

python3 << EOF
import sys

def sf(val):
    try: return float(val)
    except: return None

bqt  = sf("$baseline_query_ttft")
bqrt = sf("$baseline_query_round_time")
bwrt = sf("$baseline_warmup_round_time")
lqt  = sf("$l2_query_ttft")
lqrt = sf("$l2_query_round_time")
lwrt = sf("$l2_warmup_round_time")

min_spd  = float("$MIN_L2_SPEEDUP")
min_ttft = float("$MIN_L2_TTFT_SPEEDUP")
max_oh   = float("$MAX_WARMUP_OVERHEAD")

failed = False

print("=" * 60)
print("L2 Performance Summary")
print("=" * 60)
print(f"{'Metric':<35} {'Baseline':>12} {'L2':>12}")
print("-" * 60)
for name, bv, lv in [
    ("query_ttft_per_prompt (s)", bqt, lqt),
    ("query_round_time_per_prompt (s)", bqrt, lqrt),
    ("warmup_round_time_per_prompt (s)", bwrt, lwrt),
]:
    bs = f"{bv:.4f}" if bv else "N/A"
    ls = f"{lv:.4f}" if lv else "N/A"
    print(f"{name:<35} {bs:>12} {ls:>12}")

print()
print("=" * 60)
print("Threshold Verification")
print("=" * 60)

# 1. L2 query round-time speedup
if lqrt and bqrt and lqrt > 0:
    s = bqrt / lqrt
    ok = s >= min_spd
    print(f"[{'PASS' if ok else 'FAIL'}] L2 query speedup: {s:.2f}x (need >= {min_spd}x)")
    if not ok: failed = True
else:
    print("[FAIL] Cannot compute L2 query speedup"); failed = True

# 2. L2 TTFT speedup
if lqt and bqt and lqt > 0:
    s = bqt / lqt
    ok = s >= min_ttft
    print(f"[{'PASS' if ok else 'FAIL'}] L2 TTFT speedup: {s:.2f}x (need >= {min_ttft}x)")
    if not ok: failed = True
else:
    print("[FAIL] Cannot compute L2 TTFT speedup"); failed = True

# 3. Warmup overhead
if lwrt and bwrt and bwrt > 0:
    o = lwrt / bwrt
    ok = o <= max_oh
    print(f"[{'PASS' if ok else 'FAIL'}] Warmup overhead: {o:.2f}x (need <= {max_oh}x)")
    if not ok: failed = True
else:
    print("[FAIL] Cannot compute warmup overhead"); failed = True

print()
if failed:
    print("[FAIL] L2 performance verification FAILED")
    sys.exit(1)
else:
    print("[PASS] All L2 performance thresholds passed")
EOF

# ---------------------------------------------------------------------------
# Step 4: Verify L2 data flow via Prometheus metrics
# ---------------------------------------------------------------------------

echo "============================================"
echo "=== Verifying L2 Data Flow (Metrics) ==="
echo "============================================"

L2_METRICS_FILE="$L2_RESULTS_DIR/prometheus_metrics.txt"
# Retry briefly: when LMCache is relaunched on the same port as the
# previous instance, the Prometheus socket can take a moment to come
# back up, and a single-shot curl loses the metrics check silently.
> "$L2_METRICS_FILE"
for i in 1 2 3 4 5; do
    if curl -sf "http://localhost:${METRICS_HTTP_PORT}/metrics" \
            > "$L2_METRICS_FILE" 2>/dev/null && [ -s "$L2_METRICS_FILE" ]; then
        break
    fi
    sleep 2
done

if [ ! -s "$L2_METRICS_FILE" ]; then
    echo "FAIL: could not fetch /metrics from LMCache HTTP server (port $METRICS_HTTP_PORT)."
    echo "       /metrics being unreachable means we cannot verify the L2"
    echo "       data flow or the observability surface; failing the test"
    echo "       rather than silently skipping."
    echo ""
    echo "--- LMCache L2 server log (last 50 lines) ---"
    tail -50 "/tmp/build_${BUILD_ID}_lmcache_l2.log" 2>&1 || true
    echo ""
    echo "--- Listening sockets on port ${METRICS_HTTP_PORT} ---"
    (ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null || true) \
        | awk -v p=":${METRICS_HTTP_PORT}" '$0 ~ p'
    exit 1
fi

python3 -c "
import sys

with open('$L2_METRICS_FILE') as f:
    metrics_text = f.read()

def get_counter(name):
    for line in metrics_text.splitlines():
        if line.startswith(name + ' ') or line.startswith(name + '{'):
            return float(line.rsplit(' ', 1)[-1])
    return 0.0

# L1 metrics
l1_write_keys = get_counter('lmcache_mp_l1_write_chunks_total')

# L2 metrics
store_keys = get_counter('lmcache_mp_l2_store_submitted_objects_chunks_total')
store_succeeded = get_counter('lmcache_mp_l2_store_completed_objects_chunks_total')
prefetch_lookups = get_counter('lmcache_mp_l2_prefetch_lookup_requests_total')
prefetch_hits = get_counter('lmcache_mp_l2_prefetch_hit_chunks_total')
prefetch_loaded = get_counter('lmcache_mp_l2_prefetch_load_completed_chunks_total')

print('=' * 60)
print('Data Flow Metrics')
print('=' * 60)
print(f'  L1 write keys:               {l1_write_keys:.0f}')
print(f'  L2 store keys submitted:     {store_keys:.0f}')
print(f'  L2 store keys succeeded:     {store_succeeded:.0f}')
print(f'  L2 prefetch lookups:         {prefetch_lookups:.0f}')
print(f'  L2 prefetch prefix hits:     {prefetch_hits:.0f}')
print(f'  L2 prefetch keys loaded:     {prefetch_loaded:.0f}')
print()

failed = False

def check(cond, pass_msg, fail_msg):
    global failed
    if cond:
        print(f'[PASS] {pass_msg}')
    else:
        print(f'[FAIL] {fail_msg}')
        failed = True

# 1. L1 store activity (warmup writes KV to L1 before L2 store)
check(l1_write_keys > 0,
      f'L1 store: {l1_write_keys:.0f} keys written',
      'No keys written to L1 (expected > 0 from warmup)')

# 2. L2 store submitted and completed
check(store_keys > 0,
      f'L2 store: {store_keys:.0f} keys submitted',
      'No keys submitted to L2 store')
check(store_succeeded > 0,
      f'L2 store: {store_succeeded:.0f} keys succeeded',
      'No keys successfully stored to L2')

# 3. L2 prefetch submitted and completed (query round: L1 cold, L2 has data)
check(prefetch_lookups > 0,
      f'L2 prefetch: {prefetch_lookups:.0f} lookup requests',
      'No prefetch lookups (expected > 0 from query round)')
check(prefetch_hits > 0,
      f'L2 prefetch: {prefetch_hits:.0f} prefix hits',
      'No prefix hits from L2 lookup')
check(prefetch_loaded > 0,
      f'L2 prefetch: {prefetch_loaded:.0f} keys loaded',
      'No keys loaded from L2')

print()
if failed:
    print('[FAIL] Data flow verification FAILED')
    sys.exit(1)
else:
    print('[PASS] All data flow checks passed')
"

# ---------------------------------------------------------------------------
# Step 5: Verify the rest of the MP observability surface
# ---------------------------------------------------------------------------
# The data-flow block above is L2-focused.  This block goes wider — it
# asserts that every metric we publish from MP mode actually advances
# during the run.  ``--metrics-sample-rate 1.0`` was set on the relaunch
# above so the histograms record on every event (the default 0.01 would
# leave them empty in this short workload and flake the assertions).

echo ""
echo "============================================"
echo "=== Verifying full MP observability surface ==="
echo "============================================"

python3 - "$L2_METRICS_FILE" <<'PYEOF'
import re
import sys

with open(sys.argv[1]) as f:
    text = f.read()


def counter_total(name: str) -> float:
    """Sum a counter across all label combinations."""
    total = 0.0
    pat = re.compile(rf"^{re.escape(name)}(\{{[^}}]*\}})?\s+([0-9eE+\-.]+)\s*$", re.M)
    for _, value in pat.findall(text):
        try:
            total += float(value)
        except ValueError:
            pass
    return total


def histogram_count(base_name: str) -> float:
    """Sum the ``_count`` series across all label combinations.

    Non-zero means the histogram observed at least one sample.

    The OTel→Prometheus bridge appends the OTel ``unit`` to the metric
    name (e.g. unit ``GB/s`` → ``GB_per_second``), so the actual series
    looks like ``<base>_GB_per_second_count``.  Match that as a suffix
    so this works whether or not the unit is present.
    """
    pat = re.compile(
        rf"^{re.escape(base_name)}(?:_[A-Za-z_]+)?_count(?:\{{[^}}]*\}})?\s+"
        rf"([0-9eE+\-.]+)\s*$",
        re.M,
    )
    return sum(float(v) for v in pat.findall(text))


def has_label(base_name: str, label: str) -> bool:
    """Check that at least one sample of `base_name` carries the named label.

    Tolerates the OTel unit suffix that Prometheus appends to histograms.
    """
    pat = re.compile(
        rf"^{re.escape(base_name)}(?:_[A-Za-z_]+)?(?:_count|_sum|_bucket)?"
        rf"\{{[^}}]*\b{re.escape(label)}=",
        re.M,
    )
    return bool(pat.search(text))


# (kind, metric_name, optional_label_to_assert_present_or_None)
checks = [
    # ── Newer counters (with label dimensions) ─────────────────────
    ("counter", "lmcache_mp_l2_store_completed_requests_total", "l2_name"),
    ("counter", "lmcache_mp_l2_load_completed_requests_total", "l2_name"),
    ("counter", "lmcache_mp_lookup_requested_tokens_total", "model_name"),
    ("counter", "lmcache_mp_lookup_hit_tokens_total", "model_name"),
    ("counter", "lmcache_mp_num_chunks_loaded_total", "worker_id"),
    # ── Histograms.  The OTel→Prometheus bridge appends the OTel
    # ``unit`` to the series name, so a histogram declared with
    # ``unit="GB/s"`` actually reports as
    # ``<name>_GB_per_second_count`` / ``..._sum`` / ``..._bucket``.
    # Match by base name and let the helper tolerate the unit suffix.
    ("hist", "lmcache_mp_l0_l1_store_throughput", None),
    ("hist", "lmcache_mp_l0_l1_load_throughput", None),
    ("hist", "lmcache_mp_l2_store_throughput", "l2_name"),
    ("hist", "lmcache_mp_l2_load_throughput", "l2_name"),
]

failed = False
for kind, name, label in checks:
    if kind == "counter":
        value = counter_total(name)
        ok = value > 0
        detail = f"total={value:.0f}"
    else:
        value = histogram_count(name)
        ok = value > 0
        detail = f"_count={value:.0f}"

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not ok:
        failed = True
        continue

    if label is not None:
        if has_label(name, label):
            print(f"       └─ label '{label}' present")
        else:
            print(f"[FAIL] {name}: expected label '{label}' is missing")
            failed = True

print()
if failed:
    print("[FAIL] Observability metric verification FAILED")
    print("       (some metric did not advance, or its label dimension is missing)")
    sys.exit(1)
print("[PASS] All observability metrics populated.")
PYEOF

echo "============================================"
echo "=== L2 Long Doc QA test completed ==="
echo "============================================"
echo "Results: $L2_RESULTS_DIR"
