#!/usr/bin/env bash
set -euo pipefail

# This script requires some pre-configuration on the CI machine: 
# 1. already has a virtual environment activated
# 2. the model weights already exists on the CI machine
# 3. the sharegpt dataset already exists on the CI machine

# This script runs two tests: 
# 1. a single request test for APC + LMCache hybrid KV Cache Retrieval
# 2. a batch test for high concurrency / preemption to see if LMCache handles this case correctly
#######################################
# Required input
#######################################
: "${BUILD_ID:?BUILD_ID must be set}"

#######################################
# Configuration
#######################################
MODEL="Qwen/Qwen2.5-14B-Instruct"
WORK_LOG="/tmp/build_${BUILD_ID}_correctness.log"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm.log"
ARTIFACT="build_${BUILD_ID}.log"
SERVER_WAIT_TIMEOUT=180
CORRECTNESS_DIR=".buildkite/correctness"
REQUEST_NUMBER=100
MAX_CONCURRENCY=40

#######################################
# Prerequisite for this script: 
# Requires manual configuration on CI machines
#######################################

# 1. uv environment already exists

# persist uv’s cache somewhere stable:
export UV_CACHE_DIR="$HOME/.cache/uv"

# we will try to reuse as much uv cache as possible across jobs
# while pulling latest changes from vllm, LMCache, and other wheel dependencies
source "$HOME/correctness/.venv/bin/activate"

# update dependencies (nightly vllm and LMCache from the PR)
# --refresh-package tells uv to revalidate cached data for that dependency.
# --reinstall would reinstall all dependencies
uv pip install -U vllm \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --refresh-package vllm

# override previous lmcache from previous jobs
# the source installation is from this PR
uv pip install -e . --reinstall-package lmcache --no-build-isolation

# additional dependencies (please update manually if needed)
# these packages are pretty stable so should not need to
uv pip install aiohttp tqdm pandas huggingface_hub

# 3. the sharegpt dataset already exists on the CI machine
# wget -q \
#   https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \
#   -O "$HOME/.ShareGPT_V3_unfiltered_cleaned_split.json"

# ShareGPT dataset must already exist on the CI machine
SHAREGPT_PATH="$HOME/correctness/.ShareGPT_V3_unfiltered_cleaned_split.json"

echo "[INFO] Verifying ShareGPT dataset exists on the CI machine..."

if [[ ! -f "$SHAREGPT_PATH" ]]; then
    echo "[ERROR] ShareGPT dataset not found at: $SHAREGPT_PATH"
    echo "[FIX] Pre-download it on the CI machine, e.g.:"
    echo "      wget -q https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -O '$SHAREGPT_PATH'"
    exit 1
fi

if [[ ! -s "$SHAREGPT_PATH" ]]; then
    echo "[ERROR] ShareGPT dataset file is empty: $SHAREGPT_PATH"
    exit 1
fi


# 1. Setup local writable sandbox
CI_CACHE_DIR="$PWD/.vllm_cache_${BUILD_ID}"
mkdir -p "$CI_CACHE_DIR"

# not sure if this is needed on H100 (it was on L40s)
export FLASHINFER_WORKSPACE_DIR="$CI_CACHE_DIR/flashinfer"

#######################################
# Helpers
#######################################
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

find_free_port() {
    for p in $(seq 8000 9000); do
        if ! lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
            echo "$p"; return
        fi
    done
    exit 1
}

trap 'rc=$?; stop_vllm; collect_artifact; exit $rc' EXIT INT TERM

exec > >(tee -a "${WORK_LOG}") 2>&1

echo "=== DIAGNOSTICS: GPU STATE before CI ==="
nvidia-smi

echo "[INFO] Selecting free GPU for this build..."
source .buildkite/scripts/pick-free-gpu.sh 90000 1
echo "[INFO] Using GPU(s): ${CUDA_VISIBLE_DEVICES}"

echo "[INFO] Converting ShareGPT dataset to OpenAI format..."
python "${CORRECTNESS_DIR}/sharegpt2openai.py" -i "$SHAREGPT_PATH" -o "./shareGPT_dataset.json"

#######################################
# Phase 1: Base Server (Baseline)
#######################################
PORT=$(find_free_port)
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
READY=false
START_TIME=$(date +%s)
while [ $(($(date +%s) - START_TIME)) -lt $SERVER_WAIT_TIMEOUT ]; do
    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "${MODEL//\//\\/}"; then
        READY=true; break
    fi
    sleep 5
done

if [ "$READY" = false ]; then
    echo "[ERROR] Base vLLM failed to start"; exit 1
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

#######################################
# Phase 2: LMCache Server (Comparison)
#######################################
echo "[INFO] Preparing LMCache config (cpu.yaml)..."
cat <<EOF > cpu.yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 50
EOF

PORT=$(find_free_port)
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
READY=false
START_TIME=$(date +%s)
while [ $(($(date +%s) - START_TIME)) -lt $SERVER_WAIT_TIMEOUT ]; do
    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "${MODEL//\//\\/}"; then
        READY=true; break
    fi
    sleep 5
done

if [ "$READY" = false ]; then
    echo "[ERROR] LMCache vLLM failed to start"; exit 1
fi

echo "[TEST] Running ShareGPT Batch (LMCache)..."
python "${CORRECTNESS_DIR}/async_request.py" \
    --model "${MODEL}" \
    --endpoint "http://localhost:${PORT}/v1/chat/completions" \
    --output-file "sharegpt_lmcache.txt" \
    --dataset_file "./shareGPT_dataset.json" \
    --request-number "${REQUEST_NUMBER}" \
    --max-concurrency "${MAX_CONCURRENCY}"

#######################################
# Phase 3: Final man bash Correctness (On LMCache Server)
#######################################
echo "[TEST] Running technical man bash correctness test..."
CONTEXT="$(man bash | col -b | tr -s '[:space:]' ' ' | awk '{for(i=1;i<=NF;i++){printf "%s ",$i; if(++c==5000) exit}}')"
HALF_CONTEXT="$(man bash | col -b | tr -s '[:space:]' ' ' | awk '{for(i=1;i<=NF;i++){printf "%s ",$i; if(++c==2500) exit}}')"

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

#######################################
# Phase 4: ShareGPT File Comparison
#######################################
echo "[INFO] Comparing ShareGPT results..."
COMPARE_OUT=$(python "${CORRECTNESS_DIR}/compare_files.py" --file1 "sharegpt_base.txt" --file2 "sharegpt_lmcache.txt")

echo "--- COMPARISON RESULTS ---"
echo "${COMPARE_OUT}"
echo "--------------------------"

# Extract counts from the statistics output
DIFFERENT_COUNT=$(echo "${COMPARE_OUT}" | grep "^Different IDs:" | grep -oE '[0-9]+' || echo "0")
ONLY_FILE1_COUNT=$(echo "${COMPARE_OUT}" | awk '/^—— Only in File 1 ——$/,/^—— Only in File 2 ——$/ {print}' | grep -E "^chatcmpl-" | wc -l || echo "0")
ONLY_FILE2_COUNT=$(echo "${COMPARE_OUT}" | awk '/^—— Only in File 2 ——$/,/^$/ {print}' | grep -E "^chatcmpl-" | wc -l || echo "0")

echo "[INFO] Analysis: Different=${DIFFERENT_COUNT}, Only in Base=${ONLY_FILE1_COUNT}, Only in LMCache=${ONLY_FILE2_COUNT}"

if [[ "${DIFFERENT_COUNT}" -gt 0 ]] || [[ "${ONLY_FILE1_COUNT}" -gt 0 ]] || [[ "${ONLY_FILE2_COUNT}" -gt 0 ]]; then
    echo "[FAIL] Inconsistency detected between Base and LMCache outputs."
    exit 1
fi

echo "[PASS] All correctness tests passed identical output verified."