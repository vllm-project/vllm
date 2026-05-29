#!/usr/bin/env bash
# Real serving benchmark of the BlockScale SplitK Zero-Init fusion pass on
# Qwen3-Next-80B-A3B-Instruct-FP8, on top of the full kernel set (CK + cktile)
# with FSE on. Closed-loop, NO torch profiler -- this captures steady-state
# serving metrics (throughput, TTFT, TPOT, ITL, E2EL) without the 20-50%
# profiler overhead.
#
# Bench protocol:
#   - ISL = OSL = 1024 tokens (random_input_len = random_output_len = 1024)
#   - num_prompts = 32  (8 sequential waves of concurrency=4)
#   - max_concurrency = 4
#   - request_rate = inf  (closed-loop; server saturated at conc=4)
#   - --ignore-eos so OSL is exactly 1024 per request
#   - --save-detailed so per-request TTFT/TPOT/E2EL land in the result JSON
#
# Only difference between modes:
#   splitk        : fuse_blockscale_splitk_zero_init = false  (baseline)
#   splitk_fused  : fuse_blockscale_splitk_zero_init = true   (with fusion)
#
# torch.compile cache is nuked between modes so the FX-graph compile is
# fully re-run for each fusion setting (~+90s boot per mode).
#
# Usage:
#   bash benchmarks/run_vllm_qwen3_next_serve_isl1024.sh [<results_dir>]
set -euo pipefail

VLLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AITER_ROOT="${AITER_ROOT:-${HOME}/dev/aiter}"
RESULTS_DIR="${1:-${VLLM_ROOT}/benchmarks/zero_init_demo_results/serve_qwen3_next_isl1024_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RESULTS_DIR}"

TUNED_CSV="${TUNED_CSV:-${AITER_ROOT}/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_next_80b_a3b_filtered_built.csv}"
MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
SERVER_PORT="${SERVER_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

PYTHONPATH_BASE="${AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
SERVER_BOOT_TIMEOUT="${SERVER_BOOT_TIMEOUT:-3600}"

export AMDGCN_USE_BUFFER_OPS="${AMDGCN_USE_BUFFER_OPS:-0}"
echo "# AMDGCN_USE_BUFFER_OPS=${AMDGCN_USE_BUFFER_OPS} (workaround for triton mamba conv1d crash)"

export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS="${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:-1}"
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

    local server_log="${RESULTS_DIR}/${mode}.server.log"
    local bench_log="${RESULTS_DIR}/${mode}.bench.log"

    echo "=========================================================="
    echo "# config: mode=${mode}"
    echo "#   csv            = ${csv}"
    echo "#   fuse pass      = ${fuse_flag}"
    echo "#   FSE            = ${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS}"
    echo "#   ISL/OSL        = ${INPUT_LEN}/${OUTPUT_LEN}"
    echo "#   num_prompts    = ${NUM_PROMPTS}"
    echo "#   max_concurrency= ${MAX_CONCURRENCY}"
    echo "#   server log    -> ${server_log}"
    echo "#   bench log     -> ${bench_log}"
    echo "=========================================================="

    nuke_torch_compile_cache

    local compilation_config
    compilation_config=$(printf '{"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true,"fuse_mla_dual_rms_norm":true,"fuse_blockscale_splitk_zero_init":%s}}' "${fuse_flag}")

    # NOTE: no --profiler-config; this is a pure serving benchmark.
    local server_pid
    setsid bash -c "
        cd \"${VLLM_ROOT}\"
        exec env \
            PYTHONPATH=\"${PYTHONPATH_BASE}\" \
            AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=\"${csv}\" \
            VLLM_ROCM_USE_AITER=\"${VLLM_ROCM_USE_AITER}\" \
            VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=\"${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS}\" \
            VLLM_USE_TRITON_FLASH_ATTN=\"${VLLM_USE_TRITON_FLASH_ATTN}\" \
            VLLM_ROCM_USE_SKINNY_GEMM=\"${VLLM_ROCM_USE_SKINNY_GEMM}\" \
            VLLM_ROCM_USE_AITER_TRITON_GEMM=\"${VLLM_ROCM_USE_AITER_TRITON_GEMM}\" \
            AMDGCN_USE_BUFFER_OPS=\"${AMDGCN_USE_BUFFER_OPS}\" \
            vllm serve \"${MODEL}\" \
                --tensor-parallel-size \"${TP_SIZE}\" \
                --host \"${HOST}\" --port \"${SERVER_PORT}\" \
                --gpu-memory-utilization \"${GPU_MEM_UTIL}\" \
                --compilation-config '${compilation_config}' \
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
        --save-result --save-detailed \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        --result-dir "${RESULTS_DIR}" \
        --result-filename "${mode}.json" \
        2>&1 | tee "${bench_log}"

    sleep 5

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
        if curl -fsS --max-time 1 "http://${HOST}:${SERVER_PORT}/health" >/dev/null 2>&1; then
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
echo "# CK rows (libtype=ck):    $(awk -F, 'NR>1 && $6=="ck"' "${TUNED_CSV}" | wc -l)"
echo "# CK rows with splitK>0:   $(awk -F, 'NR>1 && $6=="ck" && $8>0' "${TUNED_CSV}" | wc -l)"
echo "# cktile rows:             $(awk -F, 'NR>1 && $6=="cktile"' "${TUNED_CSV}" | wc -l)"
echo "# FSE                     = ${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS}"

run_config splitk       "${TUNED_CSV}"   "false"
run_config splitk_fused "${TUNED_CSV}"   "true"

echo
echo "=========================================================="
echo "# Result artifacts:"
echo "=========================================================="
for mode in splitk splitk_fused; do
    echo "[$mode]"
    ls -lh "${RESULTS_DIR}/${mode}".* 2>/dev/null | sed 's/^/    /'
done

echo
echo "# All artifacts saved under: ${RESULTS_DIR}"
