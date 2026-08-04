#!/usr/bin/env bash
# Test LMCache server restart recovery: vLLM workers should re-register
# their KV caches with the new LMCache server (driven by the heartbeat
# thread's recover callback) and resume successful stores.
#
# Flow:
#   1. Run a `lmcache bench engine --workload random-prefill` round.
#   2. Snapshot lmcache_mp_l1_write_chunks_total via /metrics.
#   3. Kill the LMCache server, relaunch on the same port.
#   4. Wait for the new server to be ready and for the worker to
#      re-register (poll /status until cache_context_meta is non-empty).
#   5. Run the same bench round again.
#   6. Snapshot the metric again.
#   7. Assert run2 > 0 and run2 >= 0.8 * run1.
#
# The metric counter is reset to 0 by the new server process, so
# run2 is the absolute count for the post-restart benchmark only.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# ── Configuration (inherited from run-single-test.sh) ─────────
VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
MAX_WORKERS="${MAX_WORKERS:-4}"

# Bench parameters — small enough for CI but big enough to register
# a non-trivial l1_write_keys count per round.
NUM_REQUESTS="${RR_NUM_REQUESTS:-100}"
REQUEST_LEN="${RR_REQUEST_LEN:-5000}"
KV_CACHE_VOLUME="${RR_KV_CACHE_VOLUME:-5}"

# Recovery timing
RECOVER_TIMEOUT="${RR_RECOVER_TIMEOUT:-150}"

# Output
OUT_DIR="$RESULTS_DIR/restart_recovery"
mkdir -p "$OUT_DIR"

PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"

echo "=== Restart Recovery Test ==="
echo "Model: $MODEL"
echo "vLLM URL: http://localhost:${VLLM_PORT}"
echo "LMCache HTTP URL: http://localhost:${LMCACHE_HTTP_PORT}"
echo "Bench: ${NUM_REQUESTS} requests x ${REQUEST_LEN} tokens"
echo "Recovery timeout: ${RECOVER_TIMEOUT}s"
echo ""

# ── Helpers ──────────────────────────────────────────────────

run_bench_round() {
    local label="$1"
    local seed=$2
    local out_subdir="$OUT_DIR/$label"
    mkdir -p "$out_subdir"
    echo "--- bench round: $label ---"

    if ! lmcache bench engine \
        --engine-url "http://localhost:${VLLM_PORT}" \
        --lmcache-url "http://localhost:${LMCACHE_HTTP_PORT}" \
        --workload random-prefill \
        --rp-num-requests "$NUM_REQUESTS" \
        --rp-request-length "$REQUEST_LEN" \
        --kv-cache-volume "$KV_CACHE_VOLUME" \
        --no-interactive \
        --no-csv \
        --json \
        --quiet \
        --seed $seed \
        --output-dir "$out_subdir" \
        2>&1 | tee "$out_subdir/bench.log"; then
        echo "FAIL: bench round '$label' returned non-zero (failed requests)"
        return 1
    fi

    # Quick sanity from JSON summary
    if [ -f "$out_subdir/bench_summary.json" ]; then
        local successful failed
        successful=$(python3 -c "import json; print(json.load(open('$out_subdir/bench_summary.json')).get('successful_requests', 0))")
        failed=$(python3 -c "import json; print(json.load(open('$out_subdir/bench_summary.json')).get('failed_requests', 0))")
        echo "$label: successful=$successful failed=$failed"
        if [ "$failed" -ne 0 ]; then
            echo "FAIL: $failed requests failed in '$label'"
            return 1
        fi
    fi
    return 0
}

scrape_l1_write_keys() {
    # Print the latest lmcache_mp_l1_write_chunks_total value (single sample,
    # unlabeled) from the LMCache HTTP server's /metrics endpoint.
    python3 - <<EOF
import sys, urllib.request
url = "http://localhost:${LMCACHE_HTTP_PORT}/metrics"
try:
    body = urllib.request.urlopen(url, timeout=10).read().decode()
except Exception as e:
    print(f"ERROR fetching {url}: {e}", file=sys.stderr)
    sys.exit(1)
total = None
for line in body.splitlines():
    if line.startswith("#"):
        continue
    if not line.startswith("lmcache_mp_l1_write_chunks_total"):
        continue
    parts = line.rsplit(" ", 1)
    if len(parts) != 2:
        continue
    try:
        val = float(parts[1])
    except ValueError:
        continue
    total = (total or 0.0) + val
if total is None:
    print(f"ERROR: lmcache_mp_l1_write_chunks_total not found at {url}", file=sys.stderr)
    sys.exit(1)
print(int(total))
EOF
}

wait_for_lmcache_http() {
    local deadline=$(( $(date +%s) + 60 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if curl -sf "http://localhost:${LMCACHE_HTTP_PORT}/healthcheck" > /dev/null 2>&1; then
            echo "LMCache HTTP healthy"
            return 0
        fi
        sleep 2
    done
    echo "FAIL: LMCache HTTP did not come back within 60s"
    return 1
}

wait_for_worker_reregister() {
    # Poll /status until cache_context_meta has at least one entry,
    # which proves the vLLM worker re-registered with the new server.
    local deadline=$(( $(date +%s) + RECOVER_TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local count
        count=$(python3 - <<EOF 2>/dev/null
import json, urllib.request
try:
    body = urllib.request.urlopen("http://localhost:${LMCACHE_HTTP_PORT}/status", timeout=5).read()
    data = json.loads(body)
    print(len(data.get("cache_context_meta", {})))
except Exception:
    print(0)
EOF
)
        if [ "$count" -gt 0 ]; then
            echo "Worker re-registered (cache_context_meta entries: $count)"
            return 0
        fi
        sleep 2
    done
    echo "FAIL: worker did not re-register within ${RECOVER_TIMEOUT}s"
    return 1
}

restart_lmcache() {
    local old_pid
    old_pid=$(sed -n '1p' "$PID_FILE" 2>/dev/null || true)
    if [ -z "$old_pid" ]; then
        echo "FAIL: no LMCache PID in $PID_FILE"
        return 1
    fi
    echo "Killing LMCache PID $old_pid..."
    if kill -0 "$old_pid" 2>/dev/null; then
        kill "$old_pid" 2>/dev/null || true
        wait "$old_pid" 2>/dev/null || true
    fi
    sleep 60

    echo "Relaunching LMCache on port ${LMCACHE_PORT} / HTTP ${LMCACHE_HTTP_PORT}..."
    lmcache server \
        --l1-size-gb "$CPU_BUFFER_SIZE" \
        --eviction-policy LRU \
        --max-workers "$MAX_WORKERS" \
        --port "$LMCACHE_PORT" \
        --http-port "$LMCACHE_HTTP_PORT" \
        > "/tmp/build_${BUILD_ID}_lmcache_restart.log" 2>&1 &

    local new_pid=$!
    echo "$new_pid" > /tmp/.new_lmcache_pid
    # Replace line 1 of the PID file so cleanup.sh kills the new process.
    sed -i "1s/.*/$new_pid/" "$PID_FILE"
    echo "LMCache restarted (PID=$new_pid)"
    return 0
}

# ── Step 1: Bench round 1 ────────────────────────────────────
echo "============================================"
echo "=== Round 1: bench against original server ==="
echo "============================================"
if ! run_bench_round "round1" "41"; then
    echo "FAIL: round 1 bench failed"
    exit 1
fi

ROUND1_WRITES=$(scrape_l1_write_keys) || {
    echo "FAIL: could not scrape /metrics after round 1"
    exit 1
}
echo "Round 1 lmcache_mp_l1_write_chunks_total = $ROUND1_WRITES"

if [ "$ROUND1_WRITES" -le 0 ]; then
    echo "FAIL: round 1 produced no L1 writes; benchmark setup is broken"
    exit 1
fi

# ── Step 2: Kill + relaunch LMCache ──────────────────────────
echo ""
echo "============================================"
echo "=== Restarting LMCache server ==="
echo "============================================"
if ! restart_lmcache; then
    exit 1
fi

if ! wait_for_lmcache_http; then
    echo "--- new lmcache log (last 80 lines) ---"
    tail -80 "/tmp/build_${BUILD_ID}_lmcache_restart.log" 2>/dev/null || true
    exit 1
fi

if ! wait_for_worker_reregister; then
    echo "--- new lmcache log (last 80 lines) ---"
    tail -80 "/tmp/build_${BUILD_ID}_lmcache_restart.log" 2>/dev/null || true
    echo "--- vllm log (last 80 lines) ---"
    tail -80 "/tmp/build_${BUILD_ID}_vllm.log" 2>/dev/null || true
    exit 1
fi

# ── Step 3: Bench round 2 ────────────────────────────────────
echo ""
echo "============================================"
echo "=== Round 2: bench against restarted server ==="
echo "============================================"
if ! run_bench_round "round2" "42"; then
    echo "FAIL: round 2 bench failed"
    echo "--- vllm log (last 80 lines) ---"
    tail -80 "/tmp/build_${BUILD_ID}_vllm.log" 2>/dev/null || true
    exit 1
fi

ROUND2_WRITES=$(scrape_l1_write_keys) || {
    echo "FAIL: could not scrape /metrics after round 2"
    exit 1
}
echo "Round 2 lmcache_mp_l1_write_chunks_total = $ROUND2_WRITES"

# ── Step 4: Compare metrics ──────────────────────────────────
echo ""
echo "============================================"
echo "=== Verification ==="
echo "============================================"
echo "Round 1 writes: $ROUND1_WRITES"
echo "Round 2 writes: $ROUND2_WRITES"

if [ "$ROUND2_WRITES" -le 0 ]; then
    echo "FAIL: round 2 produced no L1 writes — re-registration likely failed"
    exit 1
fi

# Floor-divide: round2 >= 0.8 * round1  <=>  round2 * 10 >= round1 * 8
if [ $((ROUND2_WRITES * 10)) -lt $((ROUND1_WRITES * 8)) ]; then
    pct=$(python3 -c "print(f'{$ROUND2_WRITES / $ROUND1_WRITES * 100:.1f}')")
    echo "FAIL: round 2 only ${pct}% of round 1 writes (threshold: 80%)"
    exit 1
fi

pct=$(python3 -c "print(f'{$ROUND2_WRITES / $ROUND1_WRITES * 100:.1f}')")
echo "PASS: round 2 = ${pct}% of round 1 (>= 80%)"
echo ""
echo "============================================"
echo "=== Restart Recovery Test PASSED ==="
echo "============================================"
