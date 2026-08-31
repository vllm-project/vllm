#!/usr/bin/env bash
# Disaggregated P/D serving benchmark for the NIXL / HiSparse path.
#
# Launches prefill engine(s), decode engine(s) and the toy PD proxy, waits for
# readiness, drives `vllm bench serve` through the proxy over a
# (ISL:OSL) x concurrency grid, writes one JSON per point plus engine logs,
# then tears everything down.
#
# One arm per invocation: what the decode side does with imported KV is
# selected by HOST_POOL_GIB / DECODE_ATTENTION_CONFIG (both empty = plain
# GPU-resident decode, i.e. the GPU-KV baseline arm of the #46326 A/B).
#
# Replicates the measurement shape of:
#   - #53781 e2e: GLM ISL/OSL points 20k/10k, 32k/8k, 60k/10k, concurrency sweep
#   - #46326: same-topology A/B, HiSparse vs plain GPU-KV decode
#
# Everything is environment driven; see README.md in this directory.
# Example (single node, 8 GPUs, 1P tp4 + 1D tp4):
#   MODEL=zai-org/GLM-5.2-FP8 P_TP=4 D_TP=4 HOST_POOL_GIB=64 \
#     ./benchmarks/hisparse_pd/pd_bench.sh

set -euo pipefail

# ---------------------------------------------------------------- config ----
MODEL=${MODEL:?Set MODEL (e.g. zai-org/GLM-5.2-FP8)}
ARM_TAG=${ARM_TAG:-$( [[ -n "${HOST_POOL_GIB:-}" ]] && echo hisparse || echo gpu-kv )}

# topology
NUM_PREFILL=${NUM_PREFILL:-1}
NUM_DECODE=${NUM_DECODE:-1}
P_TP=${P_TP:-1}
P_PP=${P_PP:-1}          # >1 requires KV_CONNECTOR=NixlPushConnector
D_TP=${D_TP:-1}
BLOCK_SIZE=${BLOCK_SIZE:-128}
GPU_MEM_UTIL_P=${GPU_MEM_UTIL_P:-0.90}
GPU_MEM_UTIL_D=${GPU_MEM_UTIL_D:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-81920}
MAX_NUM_SEQS_D=${MAX_NUM_SEQS_D:-96}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
KV_CONNECTOR=${KV_CONNECTOR:-NixlConnector}

# HiSparse arm selection. Explicit DECODE_ATTENTION_CONFIG wins; otherwise
# HOST_POOL_GIB builds one. Empty both = plain GPU-KV decode.
DECODE_ATTENTION_CONFIG=${DECODE_ATTENTION_CONFIG:-}
PREFILL_ATTENTION_CONFIG=${PREFILL_ATTENTION_CONFIG:-}
HOST_POOL_GIB=${HOST_POOL_GIB:-}
DEVICE_BUFFER_SIZE=${DEVICE_BUFFER_SIZE:-}
# Set on the pre-rework branch only (hisparse_config had to be enabled on P
# there or the NIXL handshake fails).
PREFILL_HISPARSE=${PREFILL_HISPARSE:-0}

# extra serve flags, space separated
P_EXTRA_ARGS=${P_EXTRA_ARGS:-}
D_EXTRA_ARGS=${D_EXTRA_ARGS:-}

# traffic grid: whitespace-separated ISL:OSL pairs
ISL_OSL_PAIRS=${ISL_OSL_PAIRS:-"20000:10000 32000:8000 60000:10000"}
CONCURRENCIES=${CONCURRENCIES:-"16 32 64 128"}
NUM_PROMPTS=${NUM_PROMPTS:-100}
BENCH_EXTRA_ARGS=${BENCH_EXTRA_ARGS:-}
BENCH_TIMEOUT=${BENCH_TIMEOUT:-7200}

# outputs / ports
OUTPUT_DIR=${OUTPUT_DIR:-bench_results/hisparse_pd/${ARM_TAG}_$(date +%Y%m%d_%H%M%S)}
PROXY_PORT=${PROXY_PORT:-8192}
P_PORT_BASE=${P_PORT_BASE:-8100}
D_PORT_BASE=${D_PORT_BASE:-8200}
P_INTERNAL_PORT_BASE=${P_INTERNAL_PORT_BASE:-20000}
D_INTERNAL_PORT_BASE=${D_INTERNAL_PORT_BASE:-30000}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-3600}

# ---------------------------------------------------------------- resolve ----
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
PROXY_SCRIPT="$REPO_ROOT/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"

VLLM_BIN="$REPO_ROOT/.venv/bin/vllm"
PY_BIN="$REPO_ROOT/.venv/bin/python"
[[ -x "$VLLM_BIN" ]] || VLLM_BIN=$(command -v vllm || true)
[[ -x "$PY_BIN" ]] || PY_BIN=$(command -v python3 || true)
if [[ -z "$VLLM_BIN" || -z "$PY_BIN" ]]; then
    echo "ERROR: cannot find vllm / python binaries. Activate the venv or set PATH."
    exit 1
fi

if [[ -z "$DECODE_ATTENTION_CONFIG" && -n "$HOST_POOL_GIB" ]]; then
    DECODE_ATTENTION_CONFIG='{"hisparse_config":{"host_pool_gib":'"$HOST_POOL_GIB"
    if [[ -n "$DEVICE_BUFFER_SIZE" ]]; then
        DECODE_ATTENTION_CONFIG="$DECODE_ATTENTION_CONFIG"',"device_buffer_size":'"$DEVICE_BUFFER_SIZE"
    fi
    DECODE_ATTENTION_CONFIG="$DECODE_ATTENTION_CONFIG"'}}'
fi
if [[ -z "$PREFILL_ATTENTION_CONFIG" && "$PREFILL_HISPARSE" == "1" && -n "$HOST_POOL_GIB" ]]; then
    PREFILL_ATTENTION_CONFIG="$DECODE_ATTENTION_CONFIG"
fi

mkdir -p "$OUTPUT_DIR"
LOGDIR="$OUTPUT_DIR/logs"
mkdir -p "$LOGDIR"

# ------------------------------------------------------------- gpu layout ----
ALL_GPUS=${CUDA_VISIBLE_DEVICES:-}
if [[ -z "$ALL_GPUS" ]]; then
    ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
fi
IFS=',' read -r -a GPU_LIST <<< "$ALL_GPUS"
NUM_GPUS=${#GPU_LIST[@]}
P_WORLD=$((NUM_PREFILL * P_TP * P_PP))
D_WORLD=$((NUM_DECODE * D_TP))
if (( P_WORLD + D_WORLD > NUM_GPUS )); then
    echo "ERROR: topology needs $((P_WORLD + D_WORLD)) GPUs, only $NUM_GPUS visible: ${GPU_LIST[*]}"
    exit 1
fi

gpu_chunk() { # <start> <count> -> comma-joined device list
    local start=$1 count=$2 j out=()
    for ((j = 0; j < count; j++)); do out+=("${GPU_LIST[start + j]}"); done
    local IFS=','
    echo "${out[*]}"
}

# ---------------------------------------------------------------- helpers ----
PIDS=()
cleanup() {
    echo "Cleaning up engines and proxy..."
    local pid
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    fi
    pkill -TERM -f "toy_proxy_server.py.*--port $PROXY_PORT" 2>/dev/null || true
    sleep 3
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    fi
    pkill -9 -f "toy_proxy_server.py.*--port $PROXY_PORT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local port=$1
    echo "Waiting for server on port $port (timeout ${SERVER_START_TIMEOUT}s)..."
    timeout "$SERVER_START_TIMEOUT" bash -c "
        until curl -s localhost:${port}/v1/models > /dev/null; do sleep 2; done" \
        && return 0
    echo "ERROR: server on port $port did not come up; see $LOGDIR"
    return 1
}

wait_for_gpus_free() { # best effort before launching engines
    command -v nvidia-smi > /dev/null || return 0
    local i dev used
    for i in $(seq 1 60); do
        local busy=0
        for dev in "${GPU_LIST[@]}"; do
            used=$(nvidia-smi --query-gpu=memory.used --id="$dev" --format=csv,noheader,nounits 2>/dev/null || echo 0)
            if (( used > 1000 )); then busy=1; fi
        done
        if (( busy == 0 )); then return 0; fi
        sleep 5
    done
    echo "WARNING: GPUs still busy after 300s; launching anyway."
}

# ---------------------------------------------------------------- manifest ----
{
    echo "date: $(date -Is)"
    echo "model: $MODEL"
    echo "arm: $ARM_TAG"
    echo "decode_attention_config: ${DECODE_ATTENTION_CONFIG:-<none>}"
    echo "prefill_attention_config: ${PREFILL_ATTENTION_CONFIG:-<none>}"
    echo "topology: ${NUM_PREFILL}P(tp=${P_TP},pp=${P_PP}) + ${NUM_DECODE}D(tp=${D_TP})"
    echo "block_size: $BLOCK_SIZE  max_model_len: $MAX_MODEL_LEN  max_num_seqs_d: $MAX_NUM_SEQS_D"
    echo "grid: pairs=[$ISL_OSL_PAIRS] concurrencies=[$CONCURRENCIES] num_prompts=$NUM_PROMPTS"
    if git -C "$REPO_ROOT" rev-parse HEAD > /dev/null 2>&1; then
        echo "git_rev: $(git -C "$REPO_ROOT" rev-parse HEAD)"
        echo "git_branch: $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
        echo "git_status_dirty: $(git -C "$REPO_ROOT" status --porcelain | wc -l) files"
    fi
} | tee "$OUTPUT_DIR/manifest.txt"

# ---------------------------------------------------------------- launch ----
wait_for_gpus_free

EAGER_FLAGS=()
if [[ "$ENFORCE_EAGER" == "1" ]]; then EAGER_FLAGS=(--enforce-eager); fi

P_AC_FLAGS=()
if [[ -n "$PREFILL_ATTENTION_CONFIG" ]]; then P_AC_FLAGS=(--attention-config "$PREFILL_ATTENTION_CONFIG"); fi
D_AC_FLAGS=()
if [[ -n "$DECODE_ATTENTION_CONFIG" ]]; then D_AC_FLAGS=(--attention-config "$DECODE_ATTENTION_CONFIG"); fi

for i in $(seq 0 $((NUM_PREFILL - 1))); do
    DEV_START=$((i * P_TP * P_PP))
    PORT=$((P_PORT_BASE + i))
    SIDE=$((5559 + i))
    INTERNAL=$((P_INTERNAL_PORT_BASE + i * 100))
    echo "Starting prefill $i: GPUs [$(gpu_chunk "$DEV_START" $((P_TP * P_PP)))], port $PORT, side $SIDE"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$(gpu_chunk "$DEV_START" $((P_TP * P_PP))) \
    VLLM_PORT=$INTERNAL \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE \
    "$VLLM_BIN" serve "$MODEL" \
        --port "$PORT" \
        --block-size "$BLOCK_SIZE" \
        --gpu-memory-utilization "$GPU_MEM_UTIL_P" \
        --tensor-parallel-size "$P_TP" \
        --pipeline-parallel-size "$P_PP" \
        --max-model-len "$MAX_MODEL_LEN" \
        --kv-transfer-config "{\"kv_connector\":\"$KV_CONNECTOR\",\"kv_role\":\"kv_producer\"}" \
        "${EAGER_FLAGS[@]}" "${P_AC_FLAGS[@]}" $P_EXTRA_ARGS \
        > "$LOGDIR/prefill_${i}.log" 2>&1 &
    PIDS+=($!)
done

for i in $(seq 0 $((NUM_DECODE - 1))); do
    DEV_START=$((P_WORLD + i * D_TP))
    PORT=$((D_PORT_BASE + i))
    SIDE=$((5659 + i * D_TP))
    INTERNAL=$((D_INTERNAL_PORT_BASE + i * 100))
    echo "Starting decode $i: GPUs [$(gpu_chunk "$DEV_START" "$D_TP")], port $PORT, side $SIDE"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$(gpu_chunk "$DEV_START" "$D_TP") \
    VLLM_PORT=$INTERNAL \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE \
    "$VLLM_BIN" serve "$MODEL" \
        --port "$PORT" \
        --block-size "$BLOCK_SIZE" \
        --gpu-memory-utilization "$GPU_MEM_UTIL_D" \
        --tensor-parallel-size "$D_TP" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS_D" \
        --kv-transfer-config "{\"kv_connector\":\"$KV_CONNECTOR\",\"kv_role\":\"kv_consumer\"}" \
        "${EAGER_FLAGS[@]}" "${D_AC_FLAGS[@]}" $D_EXTRA_ARGS \
        > "$LOGDIR/decode_${i}.log" 2>&1 &
    PIDS+=($!)
done

P_PORTS=(); for i in $(seq 0 $((NUM_PREFILL - 1))); do P_PORTS+=($((P_PORT_BASE + i))); done
D_PORTS=(); for i in $(seq 0 $((NUM_DECODE - 1))); do D_PORTS+=($((D_PORT_BASE + i))); done

for PORT in "${P_PORTS[@]}"; do wait_for_server "$PORT"; done
for PORT in "${D_PORTS[@]}"; do wait_for_server "$PORT"; done

"$PY_BIN" "$PROXY_SCRIPT" \
    --port "$PROXY_PORT" \
    --prefiller-ports "${P_PORTS[@]}" \
    --decoder-ports "${D_PORTS[@]}" \
    > "$LOGDIR/proxy.log" 2>&1 &
PIDS+=($!)

timeout 120 bash -c "
    until curl -s localhost:${PROXY_PORT}/healthcheck > /dev/null; do sleep 1; done" \
    || { echo "ERROR: proxy did not come up; see $LOGDIR/proxy.log"; exit 1; }

# ---------------------------------------------------------------- smoke ----
# One request through P -> NIXL transfer -> D. Catches handshake / transfer
# breakage before burning the sweep. Garbled-but-nonempty output is not caught
# here; inspect generated text in the result JSONs for that.
echo "Smoke check: one completion through the proxy..."
SMOKE=$(curl -s --max-time 600 -X POST "localhost:${PROXY_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":8}")
echo "$SMOKE" | grep -q '"text"' \
    || { echo "ERROR: smoke request failed through the PD stack:"; echo "$SMOKE"; exit 1; }
echo "Smoke check passed."

# ---------------------------------------------------------------- sweep ----
TOTAL=0
for _ in $CONCURRENCIES; do for _ in $ISL_OSL_PAIRS; do TOTAL=$((TOTAL + 1)); done; done
RUN=0

echo ""
echo "========================================="
echo " PD sweep: arm=$ARM_TAG"
echo "   pairs=[$ISL_OSL_PAIRS]"
echo "   concurrencies=[$CONCURRENCIES]  num_prompts=$NUM_PROMPTS"
echo "   proxy=localhost:$PROXY_PORT  results=$OUTPUT_DIR"
echo "========================================="

for CONC in $CONCURRENCIES; do
    for PAIR in $ISL_OSL_PAIRS; do
        ISL=${PAIR%%:*}
        OSL=${PAIR##*:}
        RUN=$((RUN + 1))
        FNAME="conc${CONC}_isl${ISL}_osl${OSL}.json"
        FPATH="$OUTPUT_DIR/$FNAME"

        if [[ -f "$FPATH" ]] && "$PY_BIN" -c "
import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if d.get('completed',0)>0 else 1)
" "$FPATH" 2>/dev/null; then
            echo "--- [$RUN/$TOTAL] conc=$CONC isl=$ISL osl=$OSL -> SKIP (already done) ---"
            continue
        fi

        echo "--- [$RUN/$TOTAL] conc=$CONC isl=$ISL osl=$OSL -> $FNAME ---"
        # shellcheck disable=SC2086
        timeout "$BENCH_TIMEOUT" "$VLLM_BIN" bench serve \
            --backend openai \
            --base-url "http://localhost:${PROXY_PORT}" \
            --endpoint /v1/completions \
            --model "$MODEL" \
            --trust-remote-code \
            --dataset-name random \
            --random-input-len "$ISL" \
            --random-output-len "$OSL" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$CONC" \
            --save-result \
            --result-dir "$OUTPUT_DIR" \
            --result-filename "$FNAME" \
            --metadata "arm=$ARM_TAG" "concurrency=$CONC" "input_len=$ISL" "output_len=$OSL" \
            $BENCH_EXTRA_ARGS \
            || echo "WARNING: bench run conc=$CONC isl=$ISL osl=$OSL failed"
    done
done

# ---------------------------------------------------------------- summary ----
echo ""
"$PY_BIN" - "$OUTPUT_DIR" <<'EOF'
import glob
import json
import sys

rows = []
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    if f.endswith(".pytorch.json"):
        continue
    d = json.load(open(f))
    m = d.get("metadata", {}) or {}
    rows.append((
        m.get("concurrency", "?"), m.get("input_len", "?"), m.get("output_len", "?"),
        d.get("completed", 0), d.get("failed", 0),
        round(d.get("output_throughput") or 0, 1),
        round(d.get("mean_ttft_ms") or 0),
        round(d.get("p99_ttft_ms") or 0),
        round(d.get("median_itl_ms") or 0, 2),
    ))
if rows:
    hdr = ("conc", "isl", "osl", "done", "fail", "out_tok_s", "ttft_mean", "ttft_p99", "itl_med")
    print("{:>6} {:>7} {:>6} {:>5} {:>5} {:>10} {:>10} {:>10} {:>8}".format(*hdr))
    for r in rows:
        print("{:>6} {:>7} {:>6} {:>5} {:>5} {:>10} {:>10} {:>10} {:>8}".format(*r))
else:
    print("No result JSONs found.")
EOF

echo ""
echo "Preemption-ish log lines (grep -ci preempt; #46326 compares arms on preemptions):"
for f in "$LOGDIR"/decode_*.log; do
    [[ -e "$f" ]] || continue
    echo "  $(basename "$f"): $(grep -ci preempt "$f" || true)"
done

echo ""
echo "Results in: $OUTPUT_DIR"
echo "Engine logs in: $LOGDIR"
