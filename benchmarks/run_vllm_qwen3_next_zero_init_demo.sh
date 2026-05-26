#!/usr/bin/env bash
# End-to-end Qwen3-Next-80B-A3B FP8 demo of the BlockScale SplitK zero-init
# fusion under vLLM (mirror of aiter/op_tests/run_atom_qwen3_next_demo.sh,
# but targeting vLLM instead of ATOM).
#
# For each of three configs (none / splitk / splitk_fused) this:
#   1. Sets AITER_CONFIG_GEMM_A8W8_BLOCKSCALE to the matching tuned CSV so
#      the AITER blockscale GEMM dispatch picks CKTile (avoiding the
#      legacy CK invoker's unconditional hipMemsetAsync path).
#   2. Launches `vllm serve` with -tp 1 in the background and waits for
#      /health.
#   3. Runs `vllm bench serve` against it.
#   4. Kills the server and saves the bench result to a per-config JSON
#      under the timestamped results directory.
#
# Usage:
#   bash benchmarks/run_vllm_qwen3_next_zero_init_demo.sh [<results_dir>]
set -euo pipefail

VLLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AITER_ROOT="${AITER_ROOT:-${HOME}/dev/aiter}"
RESULTS_DIR="${1:-${VLLM_ROOT}/benchmarks/zero_init_demo_results/vllm_qwen3_next_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RESULTS_DIR}"

# CKTile-only tuned CSVs for Qwen3-Next 80B-A3B per-1x128 FP8 on gfx950
# (shipped with aiter at aiter/configs/zero_init_demo/robust/). The
# nosplitk variant uses splitK=0 everywhere; the splitk_yz variant uses
# splitK>0 for the shapes where SplitK reduces TPOT.
NOSPLITK_CSV="${AITER_ROOT}/aiter/configs/zero_init_demo/robust/qwen3_next_80b_a3b_per1x128_cktile_nosplitk_gfx950.csv"
SPLITK_CSV="${AITER_ROOT}/aiter/configs/zero_init_demo/robust/qwen3_next_80b_a3b_per1x128_cktile_splitk_yz_gfx950.csv"

MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
SERVER_PORT="${SERVER_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

PYTHONPATH_BASE="${AITER_ROOT}:${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Default 1 hour; bump higher for a cold HF download of the 80B FP8 weights
# (set SERVER_BOOT_TIMEOUT externally to 7200+ if running from a cold cache).
SERVER_BOOT_TIMEOUT="${SERVER_BOOT_TIMEOUT:-3600}"

# Workaround for the same triton 3.6.0+rocm7.2.2 + gfx950 mamba conv1d
# crash documented in the ATOM demo driver: disable AMD buffer-ops codegen
# so TritonAMDGPUCanonicalizePointers doesn't run on the conv1d kernels.
# The FP8 blockscale GEMM measured here is CK/CKTile in C++ (not a triton
# kernel) so the zero-init fusion timings are unaffected.
export AMDGCN_USE_BUFFER_OPS="${AMDGCN_USE_BUFFER_OPS:-0}"
echo "# AMDGCN_USE_BUFFER_OPS=${AMDGCN_USE_BUFFER_OPS} (workaround for triton mamba conv1d crash)"

# Force vLLM/AITER ROCm paths
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-1}"

# Disable vLLM's hand-written ROCm skinny-GEMM custom op (`wvSplitKrc` and
# friends in csrc/rocm/skinny_gemms.cu). It segfaults during CUDA-graph
# capture on gfx950 for Qwen3-Next-80B-A3B's M=128-256 BF16 GEMMs. With
# this off and VLLM_ROCM_USE_AITER_TRITON_GEMM=True (default), small-M
# unquantized GEMMs route to aiter's `gemm_a16w16` triton kernel; the rest
# falls back to `torch.nn.functional.linear`. The FP8 blockscale GEMM
# measured by this demo is unaffected -- it goes through AITER's CKTile
# blockscale dispatch independently.
export VLLM_ROCM_USE_SKINNY_GEMM="${VLLM_ROCM_USE_SKINNY_GEMM:-0}"
export VLLM_ROCM_USE_AITER_TRITON_GEMM="${VLLM_ROCM_USE_AITER_TRITON_GEMM:-1}"

# Nuke the torch.compile cache between modes; the fusion-pass enable flag
# is a config that may not be part of the cache key, so a graph compiled
# in mode A can be silently re-used in mode B if we don't clear it.
VLLM_COMPILE_CACHE="${HOME}/.cache/vllm/torch_compile_cache"
LAST_CSV=""

nuke_torch_compile_cache() {
    if [[ -d "${VLLM_COMPILE_CACHE}" ]]; then
        echo "# nuking vllm torch.compile cache at ${VLLM_COMPILE_CACHE}"
        rm -rf "${VLLM_COMPILE_CACHE}"
    fi
}

# We don't rebuild the AITER blockscale CKTile .so per CSV here, because
# vLLM's call site does NOT use the bpreshuffle CKTile variant; it goes
# through aiter.gemm_a8w8_blockscale (non-preshuffle CKTile), which loads
# the CSV at op-call time. No JIT rebuild is needed when the CSV path
# changes -- just the env var.

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
    local result_json="${RESULTS_DIR}/${mode}.json"
    local server_log="${RESULTS_DIR}/${mode}.server.log"
    local bench_log="${RESULTS_DIR}/${mode}.bench.log"
    echo "=========================================================="
    echo "# config: mode=${mode}"
    echo "#   csv         = ${csv}"
    echo "#   fuse pass   = ${fuse_flag}"
    echo "#   server log -> ${server_log}"
    echo "#   bench log  -> ${bench_log}"
    echo "#   result     -> ${result_json}"
    echo "=========================================================="

    nuke_torch_compile_cache

    # Build the --compilation-config payload. The zero-init fusion pass
    # only fires when fuse_blockscale_splitk_zero_init=true; otherwise
    # vLLM's PostGradPassManager skips registering the pass entirely.
    local compilation_config
    compilation_config=$(printf '{"pass_config":{"fuse_blockscale_splitk_zero_init":%s}}' "${fuse_flag}")

    local server_pid
    # Launch the server in its own process group via setsid so we can
    # kill the whole tree (the openai_server python + its inductor
    # workers) atomically with `kill -- -<pgid>`.
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

    # NB: vllm bench serve uses --random-range-ratio differently than ATOM's
    # benchmark_serving. With ratio=0.0 (vllm's default) both ISL and OSL are
    # held at the requested length; ratio=1.0 would let the sampler return
    # 0-token requests which vllm rejects. We pin to fixed length so each
    # request is exactly INPUT_LEN/OUTPUT_LEN tokens, matching the conc=4 +
    # 1k/1k workload the ATOM run used.
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
        --save-result --percentile-metrics "ttft,tpot,itl,e2el" \
        --result-dir "${RESULTS_DIR}" \
        --result-filename "${mode}.json" \
        2>&1 | tee "${bench_log}"

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
    # Wait until both the port AND VRAM are actually released before the
    # next iteration tries to bind / load weights.
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

run_config none         "${NOSPLITK_CSV}" "false"
run_config splitk       "${SPLITK_CSV}"   "false"
run_config splitk_fused "${SPLITK_CSV}"   "true"

echo
echo "=========================================================="
echo "# Summary across configs:"
echo "=========================================================="
python3 - <<PY
import json, os
results_dir = "${RESULTS_DIR}"
for mode in ("none", "splitk", "splitk_fused"):
    path = os.path.join(results_dir, f"{mode}.json")
    if not os.path.exists(path):
        print(f"{mode}: (missing)")
        continue
    with open(path) as f:
        d = json.load(f)
    keys = ("request_throughput", "output_throughput", "median_ttft_ms",
            "median_tpot_ms", "median_itl_ms", "median_e2el_ms")
    line = f"{mode:>14} | "
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            line += f"{k}={v:8.2f}  "
        else:
            line += f"{k}=n/a       "
    print(line)
PY

echo
echo "# All artifacts saved under: ${RESULTS_DIR}"
