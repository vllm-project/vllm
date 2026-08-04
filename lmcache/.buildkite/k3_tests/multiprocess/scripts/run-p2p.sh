#!/usr/bin/env bash
# P2P end-to-end test: two LMCache MP instances + two vLLM instances + one
# coordinator, all in a single 2-GPU pod over localhost.
#
# Topology:
#   coordinator           : http://127.0.0.1:12300
#   lmcache A (GPU 0)      : zmq 6555, http 7555, p2p-advertise 127.0.0.1:8555
#   lmcache B (GPU 1)      : zmq 6556, http 7556, p2p-advertise 127.0.0.1:8556
#   vLLM A    (GPU 0)      : port 8000, connector -> lmcache A (mp.port 6555)
#   vLLM B    (GPU 1)      : port 8001, connector -> lmcache B (mp.port 6556)
#
# Flow:
#   1. Both lmcache servers register with the coordinator and discover each
#      other; each creates a P2P L2 adapter for its peer.
#   2. Send a long prompt to vLLM A (cold) -> stored in lmcache A's L1.
#   3. Send the same prompt to vLLM B. B's L1 is empty and its only L2 adapter
#      is the P2P adapter pointing at A, so any LMCache hit on B must have been
#      fetched from A over P2P.
#   4. Assert B's num_lmcache_cached_tokens > 0.
#
# Self-contained: launches every process itself and records PIDs in the shared
# PID file so cleanup.sh tears them down on exit.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-300}"
CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
MAX_WORKERS="${MAX_WORKERS:-4}"

# Ports (localhost, single pod).
COORDINATOR_PORT="${COORDINATOR_PORT:-12300}"
COORDINATOR_URL="http://127.0.0.1:${COORDINATOR_PORT}"
LMCACHE_A_PORT=6555; LMCACHE_A_HTTP=7555; LMCACHE_A_P2P="127.0.0.1:8555"
LMCACHE_B_PORT=6556; LMCACHE_B_HTTP=7556; LMCACHE_B_P2P="127.0.0.1:8556"
VLLM_A_PORT=8000
VLLM_B_PORT=8001

# Tell cleanup.sh which ports to free (PID-file kill covers the rest).
export VLLM_PORT="$VLLM_A_PORT"
export VLLM_BASELINE_PORT="$VLLM_B_PORT"
export LMCACHE_PORT="$LMCACHE_A_PORT"

# GPUs: K8s assigns devices 0 and 1.
GPU_A="${GPU_FOR_A:-0}"
GPU_B="${GPU_FOR_B:-1}"

P2P_DIR="$RESULTS_DIR/p2p"
mkdir -p "$P2P_DIR"

PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
> "$PID_FILE"

# vLLM GPU-memory fraction: clamp on very large GPUs so APC does not cover the
# whole prefix and hide the LMCache path (mirrors launch-processes.sh).
GPU_MEMORY_UTIL_ARG=""
GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPU_A}" | tr -d ' ')
GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
echo "Detected GPU memory: ${GPU_MEMORY_GB}GB"
if [ -n "${GPU_MEMORY_UTILIZATION:-}" ]; then
    GPU_MEMORY_UTIL_ARG="--gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}"
elif [ "$GPU_MEMORY_GB" -gt 90 ]; then
    GPU_MEMORY_UTIL_ARG="--gpu-memory-utilization 0.5"
fi

CONNECTOR_CONFIG() {
    # $1 = lmcache mp port
    echo "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $1, \"lmcache.mp.mq_timeout\": 10}}"
}

launch_lmcache() {
    # $1=gpu $2=zmq_port $3=http_port $4=p2p_advertise $5=instance_id $6=logname
    CUDA_VISIBLE_DEVICES="$1" \
    lmcache server \
        --l1-size-gb "$CPU_BUFFER_SIZE" \
        --eviction-policy LRU \
        --max-workers "$MAX_WORKERS" \
        --port "$2" \
        --http-port "$3" \
        --host 127.0.0.1 \
        --instance-id "$5" \
        --coordinator-url "$COORDINATOR_URL" \
        --coordinator-advertise-ip 127.0.0.1 \
        --p2p-advertise-url "$4" \
        > "/tmp/build_${BUILD_ID}_${6}.log" 2>&1 &
    local pid=$!
    echo "$pid" >> "$PID_FILE"
    echo "$pid"
}

launch_vllm() {
    # $1=gpu $2=serving_port $3=lmcache_mp_port $4=logname
    # Unset VLLM_PORT so vLLM's get_open_port() picks a random internal port
    # for torch.distributed instead of colliding on serving_port+1.
    env -u VLLM_PORT \
    CUDA_VISIBLE_DEVICES="$1" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_BATCH_INVARIANT=1 \
    PYTHONHASHSEED=0 \
    vllm serve "$MODEL" \
        --kv-transfer-config "$(CONNECTOR_CONFIG "$3")" \
        --attention-backend FLASH_ATTN \
        --port "$2" \
        --no-async-scheduling \
        --max-model-len auto \
        $GPU_MEMORY_UTIL_ARG \
        > "/tmp/build_${BUILD_ID}_${4}.log" 2>&1 &
    local pid=$!
    echo "$pid" >> "$PID_FILE"
    echo "$pid"
}

# ── Step 1: coordinator ──────────────────────────────────────
echo "=== Launching coordinator on ${COORDINATOR_URL} ==="
lmcache coordinator --host 0.0.0.0 --port "$COORDINATOR_PORT" \
    > "/tmp/build_${BUILD_ID}_coordinator.log" 2>&1 &
echo "$!" >> "$PID_FILE"
echo "Waiting for coordinator health..."
for _ in $(seq 1 30); do
    if curl -sf "${COORDINATOR_URL}/healthz" > /dev/null 2>&1; then break; fi
    sleep 1
done

# ── Step 2: two LMCache MP servers ───────────────────────────
echo "=== Launching LMCache servers ==="
launch_lmcache "$GPU_A" "$LMCACHE_A_PORT" "$LMCACHE_A_HTTP" "$LMCACHE_A_P2P" \
    "p2p-a-${BUILD_ID}" "lmcache_a" > /dev/null
launch_lmcache "$GPU_B" "$LMCACHE_B_PORT" "$LMCACHE_B_HTTP" "$LMCACHE_B_P2P" \
    "p2p-b-${BUILD_ID}" "lmcache_b" > /dev/null
echo "Waiting for LMCache servers to initialize..."
sleep 10

# ── Step 3: wait for P2P discovery (A must see B as a peer) ──
echo "=== Waiting for P2P peer discovery ==="
discovered=false
for _ in $(seq 1 "$MAX_WAIT_SECONDS"); do
    peers=$(curl -sf "http://127.0.0.1:${LMCACHE_A_HTTP}/status" 2>/dev/null \
        | python3 -c "import json,sys; print(json.load(sys.stdin).get('p2p_peer_count', 0))" \
        2>/dev/null || echo 0)
    if [ "${peers:-0}" -ge 1 ]; then
        echo "lmcache A discovered ${peers} P2P peer(s)."
        discovered=true
        break
    fi
    sleep 2
done
if [ "$discovered" != "true" ]; then
    echo "FAIL: lmcache A did not discover its P2P peer in time"
    echo "--- lmcache A log (tail) ---"; tail -50 "/tmp/build_${BUILD_ID}_lmcache_a.log" || true
    echo "--- lmcache B log (tail) ---"; tail -50 "/tmp/build_${BUILD_ID}_lmcache_b.log" || true
    echo "--- coordinator instances ---"; curl -s "${COORDINATOR_URL}/instances" || true
    exit 1
fi

# ── Step 4: two vLLM servers ─────────────────────────────────
echo "=== Launching vLLM servers ==="
launch_vllm "$GPU_A" "$VLLM_A_PORT" "$LMCACHE_A_PORT" "vllm_a" > /dev/null
launch_vllm "$GPU_B" "$VLLM_B_PORT" "$LMCACHE_B_PORT" "vllm_b" > /dev/null

wait_for_vllm() {
    local port="$1" name="$2" logfile="$3"
    echo "Waiting for $name on port $port..."
    local deadline=$(( $(date +%s) + MAX_WAIT_SECONDS ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1 \
           || curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "$name is ready."
            return 0
        fi
        sleep 5
    done
    echo "FAIL: $name did not become ready in ${MAX_WAIT_SECONDS}s"
    tail -100 "$logfile" 2>/dev/null || true
    return 1
}
wait_for_vllm "$VLLM_A_PORT" "vLLM A" "/tmp/build_${BUILD_ID}_vllm_a.log"
wait_for_vllm "$VLLM_B_PORT" "vLLM B" "/tmp/build_${BUILD_ID}_vllm_b.log"

# ── Step 5: long prompt to A (cold), then B (P2P hit) ────────
LONG_CONTENT="Explain the history of computer science in great detail. $(printf 'The Turing machine is a fundamental concept in theoretical computer science that defines an abstract machine capable of manipulating symbols on a strip of tape according to a table of rules. %.0s' {1..20})"

send_request() {
    # $1=port $2=label $3=output_file
    local http_code
    http_code=$(curl -s -o "$3" -w "%{http_code}" \
        -X POST "http://localhost:${1}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"messages\": [{\"role\": \"user\", \"content\": $(python3 -c "import json;print(json.dumps('$LONG_CONTENT'))")}],
            \"max_tokens\": 1,
            \"kv_transfer_params\": {\"cached_token_stats\": true}
        }")
    if [ "$http_code" -ne 200 ]; then
        echo "FAIL: $2 returned HTTP $http_code"; cat "$3"; return 1
    fi
    echo "$2: HTTP 200 OK"
}

echo "=== Step 5a: cold request to vLLM A (populates lmcache A) ==="
send_request "$VLLM_A_PORT" "engine-A-cold" "$P2P_DIR/a_cold.json"

# Let A's store complete and become lockable before B looks it up over P2P.
sleep 8

echo "=== Step 5b: same request to vLLM B (should hit A over P2P) ==="
send_request "$VLLM_B_PORT" "engine-B-p2p" "$P2P_DIR/b_p2p.json"

# ── Step 6: assert B got a P2P cache hit ─────────────────────
echo "=== Step 6: verify P2P cache hit on engine B ==="
python3 -c "
import json, sys

with open('$P2P_DIR/b_p2p.json') as f:
    data = json.load(f)

stats = (data.get('kv_transfer_params') or {}).get('cached_token_stats')
if stats is None:
    print('FAIL: cached_token_stats missing from engine B response')
    print(data)
    sys.exit(1)

lmcache_hits = stats['num_lmcache_cached_tokens']
vllm_hits = stats['num_vllm_cached_tokens']
print(f'engine B num_vllm_cached_tokens:    {vllm_hits}')
print(f'engine B num_lmcache_cached_tokens: {lmcache_hits}')

# B never saw this prompt and its only L2 adapter is the P2P adapter to A, so a
# positive LMCache hit can only have come from A over P2P.
if lmcache_hits <= 0:
    print('FAIL: engine B had no LMCache hits; P2P fetch did not happen')
    sys.exit(1)

print(f'PASS: engine B served {lmcache_hits} tokens from peer A over P2P')
"

echo "============================================"
echo "=== P2P test PASSED ==="
echo "============================================"
