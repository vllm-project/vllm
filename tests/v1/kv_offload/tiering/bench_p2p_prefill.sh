#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Benchmark: P2P-connector KV pull vs ordinary GPU prefill.
#
# Question answered: for a variable-length prompt with a 1-token decode, is it
# faster to PULL already-computed KV from a remote peer's CPU cache over the P2P
# connector than to RECOMPUTE the prefill on the local GPU?
#
#   P2P arm      producer (pre-warmed, holds the run's KV in CPU) + consumer
#                (serves the run by pulling KV over P2P; no prefill on the
#                measured path). Measured TTFT = lookup + NIXL transfer +
#                load-into-GPU + 1 decode token.
#   Baseline arm single plain vLLM that computes prefill on the GPU and serves
#                the same run. Measured TTFT = GPU prefill + 1 decode token.
#
# With --random-output-len 1, TTFT is the headline metric. Identical --seed =>
# byte-identical prompts, so both arms serve the exact same run.
#
# Topology (defaults):
#   consumer-pod (2 GPUs): P2P consumer (GPU 0) + baseline (GPU 1) + injector proxy
#   producer-pod (>=1 GPU): P2P producer (GPU 0)
#
# Usage:
#   bash bench_p2p_prefill.sh
#   bash bench_p2p_prefill.sh --input-len 1024,4096 --concurrency 1,16 --num-prompts 200
#   bash bench_p2p_prefill.sh --consumer-pod llmd-decoder --producer-pod llmd-prefiller

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONSUMER_POD="llmd-decoder"
PRODUCER_POD="llmd-prefiller"
MODEL="meta-llama/Llama-3.1-8B"
GPU_MEM_UTIL="0.90"
MAX_MODEL_LEN="8192"
BLOCK_SIZE="16"
INPUT_LEN_LIST="1024,4096"
CONCURRENCY_LIST="1,16"
NUM_PROMPTS="200"
RANGE_RATIO="0.0"
TRIALS="1"
CPU_BYTES=""                 # producer CPU KV bytes; auto-sized when empty
CONSUMER_CPU_BYTES="8589934592"   # 8 GiB — pulled blocks land in GPU as a hit
PER_TOKEN_KV_BYTES=""        # override the analytic per-token KV estimate
NO_BASELINE=0
SKIP_DEPLOY=0
NO_PLOT=0
RESULTS_DIR=""
RESULTS_DIR_OVERRIDE=0
PYTHONHASHSEED_VAL="42"
LOG_LEVEL="INFO"
HEALTH_TIMEOUT="600"
HIT_RATE_MIN="90.0"

VLLM_BIN="${VLLM_BIN:-vllm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Ports (consumer pod co-locates several services)
CONSUMER_HTTP_PORT="8200"
BASELINE_HTTP_PORT="8300"
PROXY_PORT="8192"
PRODUCER_HTTP_PORT="8100"
P2P_PORT="${VLLM_P2P_SIDE_CHANNEL_PORT:-5710}"
PRODUCER_NIXL_PORT="5559"
CONSUMER_NIXL_PORT="5659"

PRODUCER_GPUS="0"
CONSUMER_GPUS="0"
BASELINE_GPUS="1"

PRODUCER_LOG="/tmp/p2p_producer.log"
CONSUMER_LOG="/tmp/p2p_consumer.log"
BASELINE_LOG="/tmp/p2p_baseline.log"
PROXY_LOG="/tmp/p2p_bench_proxy.log"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------------------
# Arg parse
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --consumer-pod)   CONSUMER_POD="$2"; shift 2 ;;
        --producer-pod)   PRODUCER_POD="$2"; shift 2 ;;
        --model)          MODEL="$2"; shift 2 ;;
        --gpu-mem-util)   GPU_MEM_UTIL="$2"; shift 2 ;;
        --max-model-len)  MAX_MODEL_LEN="$2"; shift 2 ;;
        --block-size)     BLOCK_SIZE="$2"; shift 2 ;;
        --input-len)      INPUT_LEN_LIST="$2"; shift 2 ;;
        --concurrency)    CONCURRENCY_LIST="$2"; shift 2 ;;
        --num-prompts)    NUM_PROMPTS="$2"; shift 2 ;;
        --range-ratio)    RANGE_RATIO="$2"; shift 2 ;;
        --trials)         TRIALS="$2"; shift 2 ;;
        --cpu-bytes)      CPU_BYTES="$2"; shift 2 ;;
        --consumer-cpu-bytes) CONSUMER_CPU_BYTES="$2"; shift 2 ;;
        --per-token-kv-bytes) PER_TOKEN_KV_BYTES="$2"; shift 2 ;;
        --pythonhashseed) PYTHONHASHSEED_VAL="$2"; shift 2 ;;
        --hit-rate-min)   HIT_RATE_MIN="$2"; shift 2 ;;
        --no-baseline)    NO_BASELINE=1; shift ;;
        --skip-deploy)    SKIP_DEPLOY=1; shift ;;
        --no-plot)        NO_PLOT=1; shift ;;
        --results-dir)    RESULTS_DIR="$2"; RESULTS_DIR_OVERRIDE=1; shift 2 ;;
        --log-level)      LOG_LEVEL="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

INPUT_LENS="${INPUT_LEN_LIST//,/ }"
CONCURRENCIES="${CONCURRENCY_LIST//,/ }"

if [[ "${RESULTS_DIR_OVERRIDE}" -ne 1 ]]; then
    RESULTS_DIR="${SCRIPT_DIR}/../results/p2p_prefill_${TIMESTAMP}_i${INPUT_LEN_LIST//,/-}_c${CONCURRENCY_LIST//,/-}_n${NUM_PROMPTS}"
fi
mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Helpers (adapted from deploy.sh)
# ---------------------------------------------------------------------------
get_pod_ip() { oc get pod "$1" -o jsonpath='{.status.podIP}'; }

# Kill any process bound to an HTTP port match on a pod, wait for GPU release.
cleanup_role() {
    local pod="$1" match="$2" role="$3"
    echo "--- cleanup ${role} (pod=${pod}) ---"
    oc exec "${pod}" -- pkill -9 -f "${match}"        2>/dev/null || true
    oc exec "${pod}" -- pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
}

wait_for_health() {
    local pod="$1" name="$2" addr="$3" port="$4" log="$5" path="${6:-/health}"
    local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
    echo -n "Waiting for ${name} (${addr}:${port}${path}) ..."
    while true; do
        if oc exec "${pod}" -- curl -sf "http://${addr}:${port}${path}" >/dev/null 2>&1; then
            echo " ready"; return 0
        fi
        if [[ "$(date +%s)" -ge "${deadline}" ]]; then
            echo ""
            echo "ERROR: ${name} did not become healthy within ${HEALTH_TIMEOUT}s" >&2
            echo "--- last 25 lines of ${log} ---" >&2
            oc exec "${pod}" -- tail -25 "${log}" 2>/dev/null >&2 || true
            return 1
        fi
        sleep 5; echo -n "."
    done
}

# ---------------------------------------------------------------------------
# Resolve pod IPs
# ---------------------------------------------------------------------------
PRODUCER_ADDR="$(get_pod_ip "${PRODUCER_POD}")"
CONSUMER_ADDR="$(get_pod_ip "${CONSUMER_POD}")"
if [[ -z "${PRODUCER_ADDR}" || -z "${CONSUMER_ADDR}" ]]; then
    echo "ERROR: failed to resolve pod IPs (producer='${PRODUCER_ADDR}' consumer='${CONSUMER_ADDR}')" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step A — per-token KV bytes + producer CPU-cache fit sizing
# ---------------------------------------------------------------------------
MAX_INPUT_LEN=0
for il in ${INPUT_LENS}; do (( il > MAX_INPUT_LEN )) && MAX_INPUT_LEN=$il; done

if [[ -z "${PER_TOKEN_KV_BYTES}" ]]; then
    echo "=== Computing per-token KV bytes for ${MODEL} (on ${PRODUCER_POD}) ==="
    # `oc exec` does not forward stdin without -i, so use a `python3 -c` program
    # (self-contained, model passed via $MODEL) rather than a stdin heredoc.
    KV_PY='
import sys
from transformers import AutoConfig
c = AutoConfig.from_pretrained(sys.argv[1])
layers = getattr(c, "num_hidden_layers", None) or getattr(c, "n_layer", 0)
kv_heads = (getattr(c, "num_key_value_heads", None)
            or getattr(c, "num_attention_heads", None) or 0)
head_dim = getattr(c, "head_dim", None)
if not head_dim:
    hs = getattr(c, "hidden_size", 0)
    nh = getattr(c, "num_attention_heads", 0) or 1
    head_dim = hs // nh
dtype = str(getattr(c, "torch_dtype", None) or getattr(c, "dtype", "bfloat16"))
dbytes = 4 if ("float32" in dtype or dtype == "float") else 2
print(2 * int(layers) * int(kv_heads) * int(head_dim) * dbytes)
'
    PER_TOKEN_KV_BYTES=$(oc exec "${PRODUCER_POD}" -- ${PYTHON_BIN} -c "${KV_PY}" "${MODEL}" 2>/dev/null | tail -1 || true)
fi

if ! [[ "${PER_TOKEN_KV_BYTES}" =~ ^[0-9]+$ ]] || [[ "${PER_TOKEN_KV_BYTES}" -eq 0 ]]; then
    echo "WARN: could not compute per-token KV bytes for ${MODEL}." >&2
    if [[ -z "${CPU_BYTES}" ]]; then
        echo "ERROR: pass --cpu-bytes <N> (or --per-token-kv-bytes <N>) explicitly." >&2
        exit 1
    fi
    PER_TOKEN_KV_BYTES=""
fi

REQUIRED_CPU_BYTES=""
if [[ -n "${PER_TOKEN_KV_BYTES}" ]]; then
    # 15% headroom over the largest cell's total prompt KV.
    REQUIRED_CPU_BYTES=$(( MAX_INPUT_LEN * NUM_PROMPTS * PER_TOKEN_KV_BYTES * 115 / 100 ))
fi

if [[ -z "${CPU_BYTES}" ]]; then
    # Round up to the next whole GiB.
    GIB=$(( 1024 * 1024 * 1024 ))
    CPU_BYTES=$(( (REQUIRED_CPU_BYTES + GIB - 1) / GIB * GIB ))
    echo "  Auto-sized producer CPU_BYTES=${CPU_BYTES} ($(( CPU_BYTES / GIB )) GiB)"
elif [[ -n "${REQUIRED_CPU_BYTES}" && "${CPU_BYTES}" -lt "${REQUIRED_CPU_BYTES}" ]]; then
    echo "ERROR: --cpu-bytes=${CPU_BYTES} is smaller than the required ${REQUIRED_CPU_BYTES}" >&2
    echo "       (max_input_len=${MAX_INPUT_LEN} x num_prompts=${NUM_PROMPTS} x" >&2
    echo "        ${PER_TOKEN_KV_BYTES} B/token x 1.15). The run's KV would not fit;" >&2
    echo "        LRU eviction would make the consumer fall back to local prefill." >&2
    exit 1
fi

cat <<HEADER

=== bench_p2p_prefill.sh ===
  Consumer pod:  ${CONSUMER_POD} (${CONSUMER_ADDR})  P2P consumer GPU ${CONSUMER_GPUS} + baseline GPU ${BASELINE_GPUS} + proxy
  Producer pod:  ${PRODUCER_POD} (${PRODUCER_ADDR})  P2P producer GPU ${PRODUCER_GPUS}
  Model:         ${MODEL}  gpu_mem=${GPU_MEM_UTIL}  max_len=${MAX_MODEL_LEN}  block=${BLOCK_SIZE}
  Input lens:    ${INPUT_LENS}
  Concurrency:   ${CONCURRENCIES}
  Prompts/cell:  ${NUM_PROMPTS}   range_ratio=${RANGE_RATIO}   output_len=1
  Trials:        ${TRIALS}
  Per-token KV:  ${PER_TOKEN_KV_BYTES:-<unknown>} bytes
  Producer CPU:  ${CPU_BYTES} bytes
  Consumer CPU:  ${CONSUMER_CPU_BYTES} bytes
  PYTHONHASHSEED:${PYTHONHASHSEED_VAL} (shared across all instances)
  Baseline arm:  $([[ ${NO_BASELINE} -eq 1 ]] && echo disabled || echo enabled)
  Results dir:   ${RESULTS_DIR}
HEADER

# ---------------------------------------------------------------------------
# KV-transfer configs (identical p2p secondary tier on producer + consumer)
# ---------------------------------------------------------------------------
PRODUCER_KV_CONFIG="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":${CPU_BYTES},\"secondary_tiers\":[{\"type\":\"p2p\"}]}}"
CONSUMER_KV_CONFIG="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":${CONSUMER_CPU_BYTES},\"secondary_tiers\":[{\"type\":\"p2p\"}]}}"

# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------
if [[ "${SKIP_DEPLOY}" -eq 0 ]]; then
    echo ""
    echo "=== Stopping existing instances ==="
    oc exec "${CONSUMER_POD}" -- pkill -9 -f "p2p_bench_proxy.py" 2>/dev/null || true
    cleanup_role "${PRODUCER_POD}" "vllm serve.*${PRODUCER_HTTP_PORT}" "producer"
    cleanup_role "${CONSUMER_POD}" "vllm serve.*${CONSUMER_HTTP_PORT}" "consumer"
    cleanup_role "${CONSUMER_POD}" "vllm serve.*${BASELINE_HTTP_PORT}" "baseline"
    echo -n "Waiting for GPU memory to be released ..."
    for pod in "${PRODUCER_POD}" "${CONSUMER_POD}"; do
        oc exec "${pod}" -- bash -c '
            deadline=$(( $(date +%s) + 60 ))
            while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q "[0-9]"; do
                [[ $(date +%s) -ge $deadline ]] && break
                sleep 2
            done' 2>/dev/null || true
    done
    echo " done"

    # Purge stale offload regions / NIXL shm left by prior runs, else the
    # producer's SharedOffloadRegion (sized to the largest cell) can exceed
    # free /dev/shm. Safe now that our processes are dead.
    for pod in "${PRODUCER_POD}" "${CONSUMER_POD}"; do
        oc exec "${pod}" -- bash -c \
            'rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null' \
            2>/dev/null || true
    done
    echo "/dev/shm purged"

    echo "=== Copying injector proxy to ${CONSUMER_POD} ==="
    oc cp "${SCRIPT_DIR}/p2p_bench_proxy.py" "${CONSUMER_POD}:/tmp/p2p_bench_proxy.py"

    echo "=== Writing launcher scripts ==="
    # Producer launcher (holds KV in CPU, serves over P2P).
    oc exec "${PRODUCER_POD}" -- bash -c "cat > /tmp/start_p2p_producer.sh << 'LAUNCHER'
#!/usr/bin/env bash
export VLLM_LOGGING_LEVEL=${LOG_LEVEL}
export CUDA_VISIBLE_DEVICES=${PRODUCER_GPUS}
export PYTHONHASHSEED=${PYTHONHASHSEED_VAL}
export VLLM_NIXL_SIDE_CHANNEL_HOST=${PRODUCER_ADDR}
export VLLM_P2P_SIDE_CHANNEL_HOST=${PRODUCER_ADDR}
export VLLM_NIXL_SIDE_CHANNEL_PORT=${PRODUCER_NIXL_PORT}
export VLLM_P2P_SIDE_CHANNEL_PORT=${P2P_PORT}
export UCX_NET_DEVICES=all
cd /tmp
exec ${VLLM_BIN} serve '${MODEL}' \
    --port ${PRODUCER_HTTP_PORT} \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --block-size ${BLOCK_SIZE} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --kv-transfer-config '${PRODUCER_KV_CONFIG}' \
    ${EXTRA_VLLM_ARGS:-}
LAUNCHER
chmod +x /tmp/start_p2p_producer.sh"

    # Consumer launcher (pulls KV from producer over P2P).
    oc exec "${CONSUMER_POD}" -- bash -c "cat > /tmp/start_p2p_consumer.sh << 'LAUNCHER'
#!/usr/bin/env bash
export VLLM_LOGGING_LEVEL=${LOG_LEVEL}
export CUDA_VISIBLE_DEVICES=${CONSUMER_GPUS}
export PYTHONHASHSEED=${PYTHONHASHSEED_VAL}
export VLLM_NIXL_SIDE_CHANNEL_HOST=${CONSUMER_ADDR}
export VLLM_P2P_SIDE_CHANNEL_HOST=${CONSUMER_ADDR}
export VLLM_NIXL_SIDE_CHANNEL_PORT=${CONSUMER_NIXL_PORT}
export VLLM_P2P_SIDE_CHANNEL_PORT=${P2P_PORT}
export VLLM_SERVER_DEV_MODE=1
export UCX_NET_DEVICES=all
cd /tmp
exec ${VLLM_BIN} serve '${MODEL}' \
    --port ${CONSUMER_HTTP_PORT} \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --block-size ${BLOCK_SIZE} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --kv-transfer-config '${CONSUMER_KV_CONFIG}' \
    ${EXTRA_VLLM_ARGS:-}
LAUNCHER
chmod +x /tmp/start_p2p_consumer.sh"

    # Baseline launcher (ordinary vLLM, no KV transfer, computes prefill on GPU).
    oc exec "${CONSUMER_POD}" -- bash -c "cat > /tmp/start_p2p_baseline.sh << 'LAUNCHER'
#!/usr/bin/env bash
export VLLM_LOGGING_LEVEL=${LOG_LEVEL}
export CUDA_VISIBLE_DEVICES=${BASELINE_GPUS}
export PYTHONHASHSEED=${PYTHONHASHSEED_VAL}
export VLLM_SERVER_DEV_MODE=1
cd /tmp
exec ${VLLM_BIN} serve '${MODEL}' \
    --port ${BASELINE_HTTP_PORT} \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --block-size ${BLOCK_SIZE} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    ${EXTRA_VLLM_ARGS:-}
LAUNCHER
chmod +x /tmp/start_p2p_baseline.sh"

    echo "=== Starting producer (${PRODUCER_POD}) ==="
    oc exec "${PRODUCER_POD}" -- bash -c "nohup /tmp/start_p2p_producer.sh > ${PRODUCER_LOG} 2>&1 & echo \$!" >/dev/null

    echo "=== Starting consumer + baseline (${CONSUMER_POD}) ==="
    oc exec "${CONSUMER_POD}" -- bash -c "nohup /tmp/start_p2p_consumer.sh > ${CONSUMER_LOG} 2>&1 & echo \$!" >/dev/null
    if [[ "${NO_BASELINE}" -eq 0 ]]; then
        oc exec "${CONSUMER_POD}" -- bash -c "nohup /tmp/start_p2p_baseline.sh > ${BASELINE_LOG} 2>&1 & echo \$!" >/dev/null
    fi

    # ---------------------------------------------------------------------
    # Health + connectivity
    # ---------------------------------------------------------------------
    wait_for_health "${PRODUCER_POD}" "Producer" "${PRODUCER_ADDR}" "${PRODUCER_HTTP_PORT}" "${PRODUCER_LOG}" || exit 1

    echo -n "Preflight: ${CONSUMER_POD} -> ${PRODUCER_ADDR}:${PRODUCER_HTTP_PORT} ... "
    if oc exec "${CONSUMER_POD}" -- curl -sf --max-time 5 "http://${PRODUCER_ADDR}:${PRODUCER_HTTP_PORT}/health" >/dev/null 2>&1; then
        echo "ok"
    else
        echo "FAILED" >&2
        echo "ERROR: consumer pod cannot reach producer HTTP port. Check NetworkPolicy / routing." >&2
        exit 1
    fi

    wait_for_health "${CONSUMER_POD}" "Consumer" "${CONSUMER_ADDR}" "${CONSUMER_HTTP_PORT}" "${CONSUMER_LOG}" || exit 1
    if [[ "${NO_BASELINE}" -eq 0 ]]; then
        wait_for_health "${CONSUMER_POD}" "Baseline" "127.0.0.1" "${BASELINE_HTTP_PORT}" "${BASELINE_LOG}" || exit 1
    fi

    echo "=== Starting injector proxy (${CONSUMER_POD}) ==="
    oc exec "${CONSUMER_POD}" -- bash -c "
        nohup ${PYTHON_BIN} /tmp/p2p_bench_proxy.py \
            --port ${PROXY_PORT} --host 127.0.0.1 \
            --target-host 127.0.0.1 --target-port ${CONSUMER_HTTP_PORT} \
            --producer-p2p-host ${PRODUCER_ADDR} --producer-p2p-port ${P2P_PORT} \
            > ${PROXY_LOG} 2>&1 & echo \$!" >/dev/null
    wait_for_health "${CONSUMER_POD}" "Injector proxy" "127.0.0.1" "${PROXY_PORT}" "${PROXY_LOG}" "/healthcheck" || exit 1
fi

# ---------------------------------------------------------------------------
# Bench command template (saved for the report)
# ---------------------------------------------------------------------------
cat > "${RESULTS_DIR}/bench_command.txt" <<BENCHCMD
# P2P arm  (injector -> consumer, pulls KV from producer)
oc exec ${CONSUMER_POD} -- ${VLLM_BIN} bench serve \\
    --backend vllm --model ${MODEL} --endpoint /v1/completions \\
    --base-url http://127.0.0.1:${PROXY_PORT} \\
    --dataset-name random \\
    --random-input-len <input_len> --random-output-len 1 \\
    --random-range-ratio ${RANGE_RATIO} \\
    --num-prompts ${NUM_PROMPTS} --max-concurrency <concurrency> \\
    --request-rate inf --seed <seed> --ignore-eos \\
    --save-result --save-detailed --result-dir <dir> \\
    --metadata arm=p2p input_len=<input_len>

# Baseline arm  (ordinary vLLM, GPU prefill)
oc exec ${CONSUMER_POD} -- ${VLLM_BIN} bench serve \\
    ... --base-url http://127.0.0.1:${BASELINE_HTTP_PORT} ... --metadata arm=baseline ...
BENCHCMD

# ---------------------------------------------------------------------------
# One bench run: $1 pod, $2 base_url, $3 arm, $4 conc, $5 input_len, $6 seed,
#                $7 local_out_dir, $8 save(0/1)
# ---------------------------------------------------------------------------
run_bench() {
    local pod="$1" base_url="$2" arm="$3" conc="$4" input_len="$5" seed="$6" out_dir="$7" save="$8"
    local save_flags=""
    local pod_dir=""
    if [[ "${save}" -eq 1 ]]; then
        pod_dir="/tmp/p2p_bench_${TIMESTAMP}/${arm}/c${conc}_i${input_len}_s${seed}"
        oc exec "${pod}" -- mkdir -p "${pod_dir}"
        save_flags="--save-result --save-detailed --result-dir ${pod_dir} --metadata arm=${arm} input_len=${input_len}"
    fi

    if [[ "${save}" -eq 1 ]]; then
        oc exec "${pod}" -- ${VLLM_BIN} bench serve \
            --backend vllm --model "${MODEL}" --endpoint /v1/completions \
            --base-url "${base_url}" --dataset-name random \
            --random-input-len "${input_len}" --random-output-len 1 \
            --random-range-ratio "${RANGE_RATIO}" \
            --num-prompts "${NUM_PROMPTS}" --max-concurrency "${conc}" \
            --request-rate inf --seed "${seed}" --ignore-eos \
            --percentile-metrics ttft,e2el \
            ${save_flags}
    else
        oc exec "${pod}" -- ${VLLM_BIN} bench serve \
            --backend vllm --model "${MODEL}" --endpoint /v1/completions \
            --base-url "${base_url}" --dataset-name random \
            --random-input-len "${input_len}" --random-output-len 1 \
            --random-range-ratio "${RANGE_RATIO}" \
            --num-prompts "${NUM_PROMPTS}" --max-concurrency "${conc}" \
            --request-rate inf --seed "${seed}" --ignore-eos \
            >/dev/null 2>&1 || echo "  WARN: warm run failed (continuing)" >&2
    fi

    if [[ "${save}" -eq 1 ]]; then
        mkdir -p "${out_dir}"
        oc cp "${pod}:${pod_dir}" "${out_dir}/" 2>/dev/null || \
            echo "WARN: failed to copy ${pod_dir} from ${pod}" >&2
    fi
}

# Best-effort reset of a served instance's LOCAL prefix cache so the measured
# run serves the cell's prompts cold, independent of what earlier cells or a
# prior --skip-deploy invocation left cached. Requires VLLM_SERVER_DEV_MODE=1
# on that instance (set in its launcher). The producer is never reset — its
# warmed CPU tier is exactly what the consumer pulls from. Unique per-cell
# seeds already guarantee disjoint prompts; this makes it deterministic too.
reset_prefix_cache() {
    local port="$1" name="$2" ok=""
    for _ in 1 2 3; do
        ok=$(oc exec "${CONSUMER_POD}" -- curl -sf -X POST \
            "http://127.0.0.1:${port}/reset_prefix_cache" 2>/dev/null || true)
        if [[ "${ok}" == *'"success":true'* ]]; then
            echo "  reset ${name} prefix cache"
            return 0
        fi
        sleep 1
    done
    echo "  WARN: could not reset ${name} prefix cache (dev mode off?); relying on unique seed" >&2
}

# Read the consumer's latest external prefix cache hit rate (KV-transfer proof).
consumer_hit_rate() {
    oc exec "${CONSUMER_POD}" -- bash -c \
        "grep -oE 'External prefix cache hit rate: [0-9.]+' ${CONSUMER_LOG} | tail -1 | awk '{print \$NF}'" \
        2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Trial loop: warm producer, measure P2P, measure baseline
# ---------------------------------------------------------------------------
BASE_SEED="$(date +%s)"
run_index=0
declare -a HIT_RATE_NOTES=()

for (( trial=1; trial<=TRIALS; trial++ )); do
    for input_len in ${INPUT_LENS}; do
        for conc in ${CONCURRENCIES}; do
            run_index=$(( run_index + 1 ))
            seed=$(( BASE_SEED + run_index ))   # never 0

            echo ""
            echo "############################################################"
            echo "# trial=${trial} input_len=${input_len} concurrency=${conc} seed=${seed}"
            echo "############################################################"

            echo "--- warm producer (${PRODUCER_POD}) ---"
            run_bench "${PRODUCER_POD}" "http://127.0.0.1:${PRODUCER_HTTP_PORT}" \
                "warm" "${conc}" "${input_len}" "${seed}" "" 0
            echo "  warm done"

            # Open the consumer->producer P2P session (ZMQ + NIXL handshake) with
            # a throwaway request on a DISTINCT seed, so its one-time setup cost
            # is not charged to the first measured request. The distinct seed
            # misses at the producer (session still opens on lookup) and uses
            # different prompts, so it does not pollute the measured GPU cache.
            echo "--- warm P2P session (consumer -> producer handshake) ---"
            oc exec "${CONSUMER_POD}" -- ${VLLM_BIN} bench serve \
                --backend vllm --model "${MODEL}" --endpoint /v1/completions \
                --base-url "http://127.0.0.1:${PROXY_PORT}" --dataset-name random \
                --random-input-len 128 --random-output-len 1 --random-range-ratio 0.0 \
                --num-prompts 2 --max-concurrency 1 --request-rate inf \
                --seed $(( seed + 500000 )) --ignore-eos \
                >/dev/null 2>&1 || echo "  WARN: P2P session warmup failed (continuing)" >&2
            echo "  session warm done"

            # Drop anything cached on the consumer (session-warmup blocks, prior
            # cells) so this cell's prompts are served cold and truly pulled.
            reset_prefix_cache "${CONSUMER_HTTP_PORT}" "consumer"

            echo "--- measure P2P (injector -> consumer) ---"
            run_bench "${CONSUMER_POD}" "http://127.0.0.1:${PROXY_PORT}" \
                "p2p" "${conc}" "${input_len}" "${seed}" "${RESULTS_DIR}/p2p" 1

            # KV-transfer gate.
            sleep 12   # let one periodic stats line land past the P2P run
            hr="$(consumer_hit_rate)"
            if [[ -n "${hr}" ]] && awk -v r="${hr}" -v m="${HIT_RATE_MIN}" 'BEGIN{exit !(r+0 >= m+0)}'; then
                echo "  KV-transfer OK: external prefix cache hit rate = ${hr}% (>= ${HIT_RATE_MIN}%)"
            else
                echo "  WARN: external prefix cache hit rate = ${hr:-<missing>}% (< ${HIT_RATE_MIN}%)" >&2
                echo "        P2P numbers for this cell may reflect local prefill fallback." >&2
                HIT_RATE_NOTES+=("i=${input_len} c=${conc}: hit_rate=${hr:-missing}%")
            fi

            if [[ "${NO_BASELINE}" -eq 0 ]]; then
                # Cold baseline: no prior copy of this cell's prompts in its cache.
                reset_prefix_cache "${BASELINE_HTTP_PORT}" "baseline"
                echo "--- measure baseline (ordinary GPU prefill) ---"
                run_bench "${CONSUMER_POD}" "http://127.0.0.1:${BASELINE_HTTP_PORT}" \
                    "baseline" "${conc}" "${input_len}" "${seed}" "${RESULTS_DIR}/baseline" 1
            fi
        done
    done
done

# ---------------------------------------------------------------------------
# Setup metadata for the report
# ---------------------------------------------------------------------------
cat > "${RESULTS_DIR}/setup.json" <<SETUP
{
  "model": "${MODEL}",
  "consumer_pod": "${CONSUMER_POD}",
  "producer_pod": "${PRODUCER_POD}",
  "max_model_len": ${MAX_MODEL_LEN},
  "block_size": ${BLOCK_SIZE},
  "gpu_memory_utilization": ${GPU_MEM_UTIL},
  "num_prompts": ${NUM_PROMPTS},
  "range_ratio": "${RANGE_RATIO}",
  "trials": ${TRIALS},
  "per_token_kv_bytes": ${PER_TOKEN_KV_BYTES:-0},
  "producer_cpu_bytes": ${CPU_BYTES},
  "consumer_cpu_bytes": ${CONSUMER_CPU_BYTES}
}
SETUP

# ---------------------------------------------------------------------------
# Aggregate + compare (baseline is the base; Δ shows P2P vs baseline)
# ---------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# Aggregating results"
echo "############################################################"

ARMS="baseline p2p"
[[ "${NO_BASELINE}" -eq 1 ]] && ARMS="p2p"

ARMS_STR="${ARMS}" \
RESULTS_DIR="${RESULTS_DIR}" \
MODEL="${MODEL}" \
NUM_PROMPTS="${NUM_PROMPTS}" \
${PYTHON_BIN} - <<'PYEOF'
import csv
import json
import os
from pathlib import Path

arms = os.environ["ARMS_STR"].split()
results_dir = Path(os.environ["RESULTS_DIR"])
model = os.environ["MODEL"]
num_prompts = os.environ["NUM_PROMPTS"]

def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

rows = {}
for arm in arms:
    adir = results_dir / arm
    if not adir.is_dir():
        continue
    for jpath in adir.rglob("*.json"):
        if jpath.name in ("setup.json",):
            continue
        try:
            d = json.loads(jpath.read_text())
        except Exception as e:
            print(f"WARN: failed to read {jpath}: {e}")
            continue
        conc = int(d.get("max_concurrency") or 0)
        try:
            input_len = int(d.get("input_len"))
        except (TypeError, ValueError):
            try:
                input_len = int(jpath.parent.name.split("_i")[-1].split("_s")[0])
            except ValueError:
                input_len = -1
        arm_tag = d.get("arm") or arm
        key = (arm_tag, conc, input_len)
        # Average over trials that share (arm, conc, input_len).
        acc = rows.setdefault(key, {"n": 0, "vals": {}})
        acc["n"] += 1
        for k in ("duration", "completed", "failed", "request_throughput",
                  "output_throughput", "total_token_throughput",
                  "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
                  "mean_e2el_ms", "p99_e2el_ms"):
            v = _num(d.get(k))
            if v is not None:
                acc["vals"].setdefault(k, []).append(v)

def avg(lst):
    return sum(lst) / len(lst) if lst else None

flat = {}
for (arm, conc, il), acc in rows.items():
    m = {k: avg(v) for k, v in acc["vals"].items()}
    flat[(arm, conc, il)] = {
        "arm": arm, "concurrency": conc, "input_len": il,
        "trials": acc["n"], "num_prompts": int(num_prompts),
        "duration_s": m.get("duration"),
        "completed": int(m.get("completed") or 0),
        "failed": int(m.get("failed") or 0),
        "req_throughput": m.get("request_throughput"),
        "output_tok_s": m.get("output_throughput"),
        "total_tok_s": m.get("total_token_throughput"),
        "mean_ttft_ms": m.get("mean_ttft_ms"),
        "median_ttft_ms": m.get("median_ttft_ms"),
        "p99_ttft_ms": m.get("p99_ttft_ms"),
        "mean_e2el_ms": m.get("mean_e2el_ms"),
        "p99_e2el_ms": m.get("p99_e2el_ms"),
    }

# ---------- CSV ----------
csv_path = results_dir / "summary.csv"
fields = ["arm", "concurrency", "input_len", "trials", "num_prompts",
          "duration_s", "completed", "failed",
          "req_throughput", "output_tok_s", "total_tok_s",
          "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
          "mean_e2el_ms", "p99_e2el_ms"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for key in sorted(flat.keys()):
        w.writerow(flat[key])
print(f"Wrote {csv_path}  ({len(flat)} rows)")

# ---------- Markdown ----------
metrics = [
    ("mean_ttft_ms", "TTFT mean ms", "lower"),
    ("median_ttft_ms", "TTFT median ms", "lower"),
    ("p99_ttft_ms", "TTFT p99 ms", "lower"),
    ("mean_e2el_ms", "E2E mean ms", "lower"),
    ("req_throughput", "req/s", "higher"),
    ("output_tok_s", "out tok/s", "higher"),
]
cells = sorted({(c, il) for (_, c, il) in flat.keys()})
base = "baseline" if "baseline" in arms else arms[0]

def fmt(v):
    if v is None:
        return "—"
    return f"{v:.1f}" if abs(v) >= 100 else f"{v:.2f}"

md = []
md.append("# P2P KV pull vs ordinary GPU prefill")
md.append("")
md.append(f"- Model: `{model}`")
md.append(f"- num_prompts/cell: {num_prompts}, output_len: 1 (TTFT is the headline metric)")
md.append(f"- Arms: {', '.join(arms)}  (Δ = P2P vs {base}; negative TTFT Δ = P2P faster)")
sp = results_dir / "setup.json"
if sp.is_file():
    try:
        s = json.loads(sp.read_text())
        md.append(f"- Producer CPU KV: {s.get('producer_cpu_bytes',0)/2**30:.1f} GiB, "
                  f"per-token KV: {s.get('per_token_kv_bytes',0)} B")
    except Exception:
        pass
md.append("")

hdr = ["concurrency", "input_len", "metric"] + arms
show_delta = "baseline" in arms and "p2p" in arms
if show_delta:
    hdr.append("Δ (p2p vs baseline)")
md.append("| " + " | ".join(hdr) + " |")
md.append("|" + "|".join(["---"] * len(hdr)) + "|")

warnings = []
for (conc, il) in cells:
    for mkey, mlabel, _dir in metrics:
        vals = [flat.get((arm, conc, il), {}).get(mkey) for arm in arms]
        if all(v is None for v in vals):
            continue
        line = [str(conc), str(il), mlabel] + [fmt(v) for v in vals]
        if show_delta:
            bv = flat.get(("baseline", conc, il), {}).get(mkey)
            pv = flat.get(("p2p", conc, il), {}).get(mkey)
            if bv in (None, 0) or pv is None:
                line.append("—")
            else:
                line.append(f"{(pv - bv) / bv * 100:+.1f}%")
        md.append("| " + " | ".join(line) + " |")
    for arm in arms:
        r = flat.get((arm, conc, il))
        if r and r.get("failed"):
            warnings.append(f"{arm} c={conc} i={il}: {r['failed']} failed requests")

if warnings:
    md.append("")
    md.append("## Warnings")
    md.append("")
    for w in warnings:
        md.append(f"- {w}")

md_path = results_dir / "summary.md"
md_path.write_text("\n".join(md) + "\n")
print()
print("\n".join(md))
print()
print(f"CSV:      {csv_path}")
print(f"Markdown: {md_path}")
PYEOF

if [[ ${#HIT_RATE_NOTES[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: low external prefix cache hit rate on these cells (P2P may have"
    echo "         fallen back to local prefill — treat their P2P numbers with care):"
    for note in "${HIT_RATE_NOTES[@]}"; do echo "  - ${note}"; done
fi

echo ""
echo "Raw JSONs + summary: ${RESULTS_DIR}"
