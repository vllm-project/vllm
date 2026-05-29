#!/usr/bin/env bash
# Diagnostic-only sibling of run_vllm_qwen3_next_profile_master_csv.sh.
#
# Boots vLLM with the production Qwen3-Next master tune CSV and the
# blockscale-splitk-zero-init fusion pass enabled, JUST long enough for
# the engine to finish compilation, then shuts the server down without
# running a decode profile. The new per-pair attribution + pre/post FX
# graph dumps wired into BlockScaleSplitKZeroInitFusionPass are what we
# read off the server log and the dump directory respectively.
#
# Outputs:
#   <results_dir>/diag.server.log    -- full server boot log
#   <results_dir>/debug_dump/
#       rank_0_dp_0/
#           blockscale_splitk_zero_init_fusion_pass.before.<seq>.<i>.fx.txt
#           blockscale_splitk_zero_init_fusion_pass.after.<seq>.<i>.fx.txt
#           patterns.blockscale_splitk_zero_init_fusion_pass.<i>.py
#
# Usage:
#   bash benchmarks/diagnose_blockscale_splitk_fusion.sh [<results_dir>]
set -euo pipefail

VLLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AITER_ROOT="${AITER_ROOT:-${HOME}/dev/aiter}"
RESULTS_DIR="${1:-${VLLM_ROOT}/benchmarks/zero_init_demo_results/diagnose_blockscale_$(date +%Y%m%d_%H%M%S)}"
DEBUG_DUMP_DIR="${RESULTS_DIR}/debug_dump"
SERVER_LOG="${RESULTS_DIR}/diag.server.log"
mkdir -p "${RESULTS_DIR}" "${DEBUG_DUMP_DIR}"

TUNED_CSV="${TUNED_CSV:-${AITER_ROOT}/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_next_80b_a3b_filtered_built.csv}"
MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
SERVER_PORT="${SERVER_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
PYTHONPATH_BASE="${AITER_ROOT}:${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
SERVER_BOOT_TIMEOUT="${SERVER_BOOT_TIMEOUT:-3600}"
COMPILE_DONE_MARKER="BlockScaleSplitKZeroInitFusionPass replaced"
# Wait for at least N markers + a final settle period. The original
# 100801 run produced 6 markers; we wait for 8 + 60s settle so we don't
# miss a late-arriving range (and so the FX dumps include every compile
# range, including the 6th that the first diagnostic capture missed).
EXPECTED_MARKER_COUNT="${EXPECTED_MARKER_COUNT:-8}"
EXTRA_SETTLE_SECONDS="${EXTRA_SETTLE_SECONDS:-60}"
# After EXTRA_SETTLE_SECONDS of no marker movement we declare "done" even
# if EXPECTED_MARKER_COUNT was not reached (engine has fewer ranges than
# we assumed). The post-grad fusion pass log is the source of truth.
STALL_TIMEOUT="${STALL_TIMEOUT:-90}"

export AMDGCN_USE_BUFFER_OPS="${AMDGCN_USE_BUFFER_OPS:-0}"
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-1}"
export VLLM_ROCM_USE_SKINNY_GEMM="${VLLM_ROCM_USE_SKINNY_GEMM:-0}"
export VLLM_ROCM_USE_AITER_TRITON_GEMM="${VLLM_ROCM_USE_AITER_TRITON_GEMM:-1}"

# IMPORTANT: VLLM_DEBUG_DUMP_PATH overrides CompilationConfig.debug_dump_path
# (see vllm/config/vllm.py around line 1491). Setting it here ensures the
# fusion pass's _dump_graph_to_file writes to <DEBUG_DUMP_DIR>/rank_0_dp_0/.
export VLLM_DEBUG_DUMP_PATH="${DEBUG_DUMP_DIR}"

VLLM_COMPILE_CACHE="${HOME}/.cache/vllm/torch_compile_cache"
if [[ -d "${VLLM_COMPILE_CACHE}" ]]; then
    echo "# nuking vllm torch.compile cache at ${VLLM_COMPILE_CACHE}"
    rm -rf "${VLLM_COMPILE_CACHE}"
fi

echo "# tuned CSV    = ${TUNED_CSV}"
echo "# results dir  = ${RESULTS_DIR}"
echo "# debug dump   = ${DEBUG_DUMP_DIR}"
echo "# server log   = ${SERVER_LOG}"

compilation_config=$(printf '{"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true,"fuse_mla_dual_rms_norm":true,"fuse_blockscale_splitk_zero_init":true}}')

setsid bash -c "
    cd \"${VLLM_ROOT}\"
    exec env \
        PYTHONPATH=\"${PYTHONPATH_BASE}\" \
        AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=\"${TUNED_CSV}\" \
        VLLM_ROCM_USE_AITER=\"${VLLM_ROCM_USE_AITER}\" \
        VLLM_USE_TRITON_FLASH_ATTN=\"${VLLM_USE_TRITON_FLASH_ATTN}\" \
        VLLM_ROCM_USE_SKINNY_GEMM=\"${VLLM_ROCM_USE_SKINNY_GEMM}\" \
        VLLM_ROCM_USE_AITER_TRITON_GEMM=\"${VLLM_ROCM_USE_AITER_TRITON_GEMM}\" \
        VLLM_DEBUG_DUMP_PATH=\"${DEBUG_DUMP_DIR}\" \
        AMDGCN_USE_BUFFER_OPS=\"${AMDGCN_USE_BUFFER_OPS}\" \
        vllm serve \"${MODEL}\" \
            --tensor-parallel-size \"${TP_SIZE}\" \
            --host \"${HOST}\" --port \"${SERVER_PORT}\" \
            --gpu-memory-utilization \"${GPU_MEM_UTIL}\" \
            --compilation-config '${compilation_config}' \
            >\"${SERVER_LOG}\" 2>&1
" &
server_pid=$!
echo "# server pid (pgid leader): ${server_pid}"
trap "kill -TERM -${server_pid} 2>/dev/null || true" EXIT

# Wait until either (a) we see N "replaced ... patterns" markers
# indicating compilation has finished for the desired number of ranges,
# or (b) the boot timeout elapses, or (c) the server process dies.
deadline=$(( SECONDS + SERVER_BOOT_TIMEOUT ))
last_marker_count=0
last_marker_time=$SECONDS
while (( SECONDS < deadline )); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "ERROR: server PID ${server_pid} died during boot" >&2
        tail -50 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    marker_count=$(grep -c "${COMPILE_DONE_MARKER}" "${SERVER_LOG}" 2>/dev/null) || marker_count=0
    if (( marker_count != last_marker_count )); then
        echo "# saw ${marker_count} fusion-pass markers"
        last_marker_count=${marker_count}
        last_marker_time=$SECONDS
    fi
    if (( marker_count >= EXPECTED_MARKER_COUNT )); then
        echo "# reached expected ${EXPECTED_MARKER_COUNT} markers; settling ${EXTRA_SETTLE_SECONDS}s"
        sleep "${EXTRA_SETTLE_SECONDS}"
        break
    fi
    # If we've seen at least one marker and the count has been stable
    # for STALL_TIMEOUT seconds, treat it as "engine has no more
    # compile work to do" and stop waiting.
    if (( marker_count > 0 )) && (( SECONDS - last_marker_time >= STALL_TIMEOUT )); then
        echo "# marker count stable at ${marker_count} for ${STALL_TIMEOUT}s; treating as done"
        sleep "${EXTRA_SETTLE_SECONDS}"
        break
    fi
    sleep 5
done

if (( last_marker_count == 0 )); then
    echo "ERROR: never saw any '${COMPILE_DONE_MARKER}' lines" >&2
    tail -100 "${SERVER_LOG}" >&2 || true
fi

echo "# stopping server pgid ${server_pid}"
kill -TERM -"${server_pid}" 2>/dev/null || true
for _i in $(seq 1 30); do
    if ! kill -0 -"${server_pid}" 2>/dev/null; then
        break
    fi
    sleep 1
done
kill -KILL -"${server_pid}" 2>/dev/null || true
sleep 5
trap - EXIT

echo
echo "=========================================================="
echo "# Fusion-pass attribution lines:"
echo "=========================================================="
grep -E "BlockScaleSplitKZeroInitFusionPass" "${SERVER_LOG}" || true

echo
echo "=========================================================="
echo "# FX graph dumps:"
echo "=========================================================="
find "${DEBUG_DUMP_DIR}" -type f 2>/dev/null | sort | head -40

echo
echo "# Diagnostic artifacts under: ${RESULTS_DIR}"
