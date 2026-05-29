#!/usr/bin/env bash
# Same protocol as run_vllm_qwen3_next_profile_demo.sh, but routes the
# BlockScale GEMM dispatch through the *production* Qwen3-Next tune
# (a8w8_blockscale_tuned_gemm_qwen3_next_80b_a3b.csv).  That CSV contains
# 1046 legacy-CK rows (88 of them with splitK>0) in addition to the 436
# cktile rows, so it exercises *both* the cktile and the newly-plumbed
# legacy-CK zero-init fusion paths.
#
# Both modes use the SAME CSV; the only difference between splitk and
# splitk_fused is the BlockScaleSplitKZeroInitFusionPass flag.
#
# Usage:
#   bash benchmarks/run_vllm_qwen3_next_profile_master_csv.sh [<results_dir>]
set -euo pipefail

VLLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AITER_ROOT="${AITER_ROOT:-${HOME}/dev/aiter}"
RESULTS_DIR="${1:-${VLLM_ROOT}/benchmarks/zero_init_demo_results/profile_qwen3_next_master_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RESULTS_DIR}"

# >>> Only difference from run_vllm_qwen3_next_profile_demo.sh: <<<
# point at the production tuned CSV that has CK splitK rows.  We use
# the "filtered_built" copy: it preserves every CK row from the master
# tune (including all 88 CK rows with splitK>0 that exercise the new
# legacy-CK zero-init fusion) but drops cktile rows whose kernel was
# not built into the current aiter cktile module so they fall back to
# the untuned default cktile kernel instead of crashing the engine.
TUNED_CSV="${TUNED_CSV:-${AITER_ROOT}/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_next_80b_a3b_filtered_built.csv}"

MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
SERVER_PORT="${SERVER_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
NUM_PROMPTS="${NUM_PROMPTS:-4}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-2}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

PYTHONPATH_BASE="${AITER_ROOT}:${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
SERVER_BOOT_TIMEOUT="${SERVER_BOOT_TIMEOUT:-3600}"

export AMDGCN_USE_BUFFER_OPS="${AMDGCN_USE_BUFFER_OPS:-0}"
echo "# AMDGCN_USE_BUFFER_OPS=${AMDGCN_USE_BUFFER_OPS} (workaround for triton mamba conv1d crash)"

export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-1}"
export VLLM_ROCM_USE_SKINNY_GEMM="${VLLM_ROCM_USE_SKINNY_GEMM:-0}"
export VLLM_ROCM_USE_AITER_TRITON_GEMM="${VLLM_ROCM_USE_AITER_TRITON_GEMM:-1}"

VLLM_COMPILE_CACHE="${HOME}/.cache/vllm/torch_compile_cache"

nuke_torch_compile_cache() {
    if [[ -d "${VLLM_COMPILE_CACHE}" ]]; then
        echo "# nuking vllm torch.compile cache at ${VLLM_COMPILE_CACHE}"
        rm -rf "${VLLM_COMPILE_CACHE}"
    fi
}

wait_for_server() {
    local pid="$1"
    local deadline=$(( SECONDS + SERVER_BOOT_TIMEOUT ))
    while (( SECONDS < deadline )); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: server PID ${pid} died during boot" >&2
            return 1
        fi
        if curl -fsS "http://${HOST}:${SERVER_PORT}/health" >/dev/null 2>&1; then
            echo "# server up after $(( SECONDS - (deadline - SERVER_BOOT_TIMEOUT) ))s"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: server did not become ready within ${SERVER_BOOT_TIMEOUT}s" >&2
    return 1
}

run_config() {
    local mode="$1"
    local csv="$2"
    local fuse_flag="$3"   # "true" or "false"

    local mode_dir="${RESULTS_DIR}/${mode}"
    mkdir -p "${mode_dir}"
    local server_log="${RESULTS_DIR}/${mode}.server.log"
    local bench_log="${RESULTS_DIR}/${mode}.bench.log"

    echo "=========================================================="
    echo "# config: mode=${mode}"
    echo "#   csv         = ${csv}"
    echo "#   fuse pass   = ${fuse_flag}"
    echo "#   trace dir  -> ${mode_dir}"
    echo "#   server log -> ${server_log}"
    echo "#   bench log  -> ${bench_log}"
    echo "=========================================================="

    nuke_torch_compile_cache

    local compilation_config
    compilation_config=$(printf '{"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true,"fuse_mla_dual_rms_norm":true,"fuse_blockscale_splitk_zero_init":%s}}' "${fuse_flag}")

    local profiler_config
    profiler_config=$(python3 -c "
import json
print(json.dumps({
    'profiler': 'torch',
    'torch_profiler_dir': '${mode_dir}',
    'torch_profiler_with_stack': False,
    'torch_profiler_use_gzip': False,
    'torch_profiler_dump_cuda_time_total': True,
    'warmup_iterations': 2,
    'active_iterations': 5,
    'wait_iterations': 0,
}))")

    local server_pid
    setsid bash -c "
        cd \"${VLLM_ROOT}\"
        exec env \
            PYTHONPATH=\"${PYTHONPATH_BASE}\" \
            AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=\"${csv}\" \
            VLLM_ROCM_USE_AITER=\"${VLLM_ROCM_USE_AITER}\" \
            VLLM_USE_TRITON_FLASH_ATTN=\"${VLLM_USE_TRITON_FLASH_ATTN}\" \
            VLLM_ROCM_USE_SKINNY_GEMM=\"${VLLM_ROCM_USE_SKINNY_GEMM}\" \
            VLLM_ROCM_USE_AITER_TRITON_GEMM=\"${VLLM_ROCM_USE_AITER_TRITON_GEMM}\" \
            AMDGCN_USE_BUFFER_OPS=\"${AMDGCN_USE_BUFFER_OPS}\" \
            vllm serve \"${MODEL}\" \
                --tensor-parallel-size \"${TP_SIZE}\" \
                --host \"${HOST}\" --port \"${SERVER_PORT}\" \
                --gpu-memory-utilization \"${GPU_MEM_UTIL}\" \
                --compilation-config '${compilation_config}' \
                --profiler-config '${profiler_config}' \
                >\"${server_log}\" 2>&1
    " &
    server_pid=$!
    echo "# server pid (pgid leader): ${server_pid}"
    trap "kill -TERM -${server_pid} 2>/dev/null || true" EXIT

    if ! wait_for_server "${server_pid}"; then
        kill -TERM -"${server_pid}" 2>/dev/null || true
        sleep 3
        kill -KILL -"${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
        trap - EXIT
        return 1
    fi

    PYTHONPATH="${PYTHONPATH_BASE}" \
    vllm bench serve \
        --model "${MODEL}" --backend vllm \
        --base-url "http://${HOST}:${SERVER_PORT}" \
        --dataset-name random \
        --random-input-len "${INPUT_LEN}" \
        --random-output-len "${OUTPUT_LEN}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency "${MAX_CONCURRENCY}" \
        --request-rate inf --ignore-eos \
        --profile \
        --save-result --percentile-metrics "ttft,tpot,itl,e2el" \
        --result-dir "${RESULTS_DIR}" \
        --result-filename "${mode}.json" \
        2>&1 | tee "${bench_log}"

    sleep 15

    echo "# stopping server pgid ${server_pid}"
    kill -TERM -"${server_pid}" 2>/dev/null || true
    for _i in $(seq 1 30); do
        if ! kill -0 -"${server_pid}" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    kill -KILL -"${server_pid}" 2>/dev/null || true
    sleep 3
    for _i in $(seq 1 30); do
        local port_busy=0
        local vram_used_mib=0
        if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${SERVER_PORT}\$"; then
            port_busy=1
        fi
        vram_used_mib=$(rocm-smi --showmeminfo vram 2>/dev/null \
            | awk '/VRAM Total Used Memory \(B\)/ {print int($NF/1024/1024)}' \
            | head -1)
        vram_used_mib=${vram_used_mib:-0}
        if (( port_busy == 0 && vram_used_mib < 2048 )); then
            break
        fi
        sleep 2
    done
    trap - EXIT
}

echo "# tuned CSV = ${TUNED_CSV}"
echo "# CK rows (libtype=ck): $(awk -F, 'NR>1 && $6=="ck"' "${TUNED_CSV}" | wc -l)"
echo "# CK rows with splitK>0: $(awk -F, 'NR>1 && $6=="ck" && $8>0' "${TUNED_CSV}" | wc -l)"
echo "# cktile rows: $(awk -F, 'NR>1 && $6=="cktile"' "${TUNED_CSV}" | wc -l)"

run_config splitk       "${TUNED_CSV}"   "false"
run_config splitk_fused "${TUNED_CSV}"   "true"

echo
echo "=========================================================="
echo "# Trace artifacts:"
echo "=========================================================="
for mode in splitk splitk_fused; do
    echo "[$mode]"
    ls -lh "${RESULTS_DIR}/${mode}" 2>/dev/null | sed 's/^/    /'
done

echo
echo "# All artifacts saved under: ${RESULTS_DIR}"
