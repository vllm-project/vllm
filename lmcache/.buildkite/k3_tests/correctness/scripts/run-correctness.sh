#!/usr/bin/env bash
# Self-contained correctness test for K8s pods.
# Replaces the old vllm-correctness.sh which assumed a pre-existing venv
# and pick-free-gpu.sh. Here everything runs natively in the pod.
#
# Tests that LMCache produces identical output to base vLLM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"
source .buildkite/k3_tests/common_scripts/helpers.sh

###############
# CONFIG      #
###############

BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"
MODEL="Qwen/Qwen2.5-14B-Instruct"
WORK_LOG="/tmp/build_${BUILD_ID}_correctness.log"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm.log"
ARTIFACT="build_${BUILD_ID}.log"
SERVER_WAIT_TIMEOUT=180
CORRECTNESS_DIR=".buildkite/correctness"
REQUEST_NUMBER=100
MAX_CONCURRENCY=40

# K8s assigns GPUs via device plugin
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

###############
# DATA SETUP  #
###############

SHAREGPT_PATH="/root/correctness/.ShareGPT_V3_unfiltered_cleaned_split.json"
if [[ ! -f "$SHAREGPT_PATH" ]]; then
    echo "[INFO] ShareGPT dataset not found, downloading..."
    mkdir -p /root/correctness
    wget -q \
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" \
        -O "$SHAREGPT_PATH"
fi

if [[ ! -s "$SHAREGPT_PATH" ]]; then
    echo "[ERROR] ShareGPT dataset file is empty: $SHAREGPT_PATH"
    exit 1
fi

###############
# HELPERS     #
###############

VLLM_PID=""

CI_CACHE_DIR="$PWD/.vllm_cache_${BUILD_ID}"
mkdir -p "$CI_CACHE_DIR"
export FLASHINFER_WORKSPACE_DIR="$CI_CACHE_DIR/flashinfer"

collect_artifact() {
    echo "[INFO] Collecting logs into ${ARTIFACT}"
    cat "${WORK_LOG}" "${VLLM_LOG}" > "${ARTIFACT}" 2>/dev/null || true
}

stop_vllm() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "[INFO] Stopping vLLM process (PID: ${VLLM_PID})"
        kill "${VLLM_PID}" >/dev/null 2>&1 || true
        wait "${VLLM_PID}" 2>/dev/null || true
        VLLM_PID=""
        sleep 5
    fi
}

trap 'rc=$?; stop_vllm; collect_artifact; exit $rc' EXIT INT TERM

exec > >(tee -a "${WORK_LOG}") 2>&1

echo "=== DIAGNOSTICS: GPU STATE before CI ==="
nvidia-smi
echo "[INFO] Using GPU(s): ${CUDA_VISIBLE_DEVICES}"

echo "[INFO] Converting ShareGPT dataset to OpenAI format..."
python "${CORRECTNESS_DIR}/sharegpt2openai.py" -i "$SHAREGPT_PATH" -o "./shareGPT_dataset.json"

###############
# PHASE 1     #
###############
# Base Server (Baseline) -- vLLM WITHOUT LMCache

PORT=$(find_free_port 8000)
echo "[INFO] Starting BASE vLLM server on port ${PORT}..."

VLLM_SERVER_DEV_MODE=1 \
VLLM_BATCH_INVARIANT=1 \
vllm serve "${MODEL}" \
    --port "${PORT}" \
    --trust-remote-code \
    --enforce-eager \
    --attention-backend FLASH_ATTN \
    --gpu-memory-utilization 0.8 \
    >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

echo "[INFO] Waiting for Base server readiness..."
if ! wait_for_server "$PORT" "$SERVER_WAIT_TIMEOUT"; then
    echo "[ERROR] Base vLLM failed to start"
    exit 1
fi

echo "[TEST] Running ShareGPT Batch (Base)..."
python "${CORRECTNESS_DIR}/async_request.py" \
    --model "${MODEL}" \
    --endpoint "http://localhost:${PORT}/v1/chat/completions" \
    --output-file "sharegpt_base.txt" \
    --dataset_file "./shareGPT_dataset.json" \
    --request-number "${REQUEST_NUMBER}" \
    --max-concurrency "${MAX_CONCURRENCY}"

stop_vllm

###############
# PHASE 2     #
###############
# LMCache Server -- vLLM WITH LMCache

echo "[INFO] Preparing LMCache config (cpu.yaml)..."
cat <<EOF > cpu.yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 50
EOF

PORT=$(find_free_port 8000)
echo "[INFO] Starting LMCACHE vLLM server on port ${PORT}..."

LMCACHE_CONFIG_FILE=cpu.yaml \
VLLM_SERVER_DEV_MODE=1 \
VLLM_BATCH_INVARIANT=1 \
vllm serve "${MODEL}" \
    --port "${PORT}" \
    --trust-remote-code \
    --enforce-eager \
    --attention-backend FLASH_ATTN \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    >>"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

echo "[INFO] Waiting for LMCache server readiness..."
if ! wait_for_server "$PORT" "$SERVER_WAIT_TIMEOUT"; then
    echo "[ERROR] LMCache vLLM failed to start"
    exit 1
fi

echo "[TEST] Running ShareGPT Batch (LMCache)..."
python "${CORRECTNESS_DIR}/async_request.py" \
    --model "${MODEL}" \
    --endpoint "http://localhost:${PORT}/v1/chat/completions" \
    --output-file "sharegpt_lmcache.txt" \
    --dataset_file "./shareGPT_dataset.json" \
    --request-number "${REQUEST_NUMBER}" \
    --max-concurrency "${MAX_CONCURRENCY}"

###############
# PHASE 3     #
###############
# Man bash correctness test (on the LMCache server)

echo "[TEST] Running technical man bash correctness test..."
CONTEXT="$(man bash | sed 's/.\x08//g' | tr -s '[:space:]' ' ' | awk '{for(i=1;i<=NF;i++){printf "%s ",$i; if(++c==5000) exit}}')"
HALF_CONTEXT="$(man bash | sed 's/.\x08//g' | tr -s '[:space:]' ' ' | awk '{for(i=1;i<=NF;i++){printf "%s ",$i; if(++c==2500) exit}}')"

send_completion() {
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg model "${MODEL}" --arg content "$1" '{model: $model, temperature: 0, max_tokens: 100, messages: [{role:"user",content:$content}]}')" |
        jq -r '.choices[0].message.content'
}

echo "[STEP 1] Full context (LMCache)"
OUT1="$(send_completion "${CONTEXT}")"

echo "[STEP 2] Reset prefix cache"
curl -s -X POST "http://localhost:${PORT}/reset_prefix_cache" >/dev/null

echo "[STEP 3] Half context"
send_completion "${HALF_CONTEXT}" >/dev/null

echo "[STEP 4] Full context again"
OUT2="$(send_completion "${CONTEXT}")"

if [[ "${OUT1}" != "${OUT2}" ]]; then
    echo "[FAIL] man bash output mismatch!"
    exit 1
fi

###############
# PHASE 4     #
###############
# ShareGPT File Comparison

echo "[INFO] Comparing ShareGPT results..."
COMPARE_OUT=$(python "${CORRECTNESS_DIR}/compare_files.py" --file1 "sharegpt_base.txt" --file2 "sharegpt_lmcache.txt")

echo "--- COMPARISON RESULTS ---"
echo "${COMPARE_OUT}"
echo "--------------------------"

DIFFERENT_COUNT=$(echo "${COMPARE_OUT}" | grep "^Different IDs:" | grep -oE '[0-9]+' | tail -1 || echo "0")
ONLY_FILE1_COUNT=$(echo "${COMPARE_OUT}" | awk '/^—— Only in File 1 ——$/,/^—— Only in File 2 ——$/ {print}' | grep -cE "^chatcmpl-" || echo "0")
ONLY_FILE2_COUNT=$(echo "${COMPARE_OUT}" | awk '/^—— Only in File 2 ——$/,/^$/ {print}' | grep -cE "^chatcmpl-" || echo "0")

echo "[INFO] Analysis: Different=${DIFFERENT_COUNT}, Only in Base=${ONLY_FILE1_COUNT}, Only in LMCache=${ONLY_FILE2_COUNT}"

if [[ "${DIFFERENT_COUNT}" -gt 0 ]] || [[ "${ONLY_FILE1_COUNT}" -gt 0 ]] || [[ "${ONLY_FILE2_COUNT}" -gt 0 ]]; then
    echo "[FAIL] Inconsistency detected between Base and LMCache outputs."
    exit 1
fi

echo "[PASS] All correctness tests passed -- identical output verified."
