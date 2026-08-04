#!/usr/bin/env bash
# Orchestrator for a single multiprocessing test (native, no Docker).
# Usage: run-single-test.sh <test_name>
#   test_name: lm_eval | lm_eval_preemption | hma_lm_eval_gemma4 | vllm_bench
#              | long_doc_qa | long_doc_qa_l2 | fault_tolerance | deadlock
#              | restart_recovery | gds_smoke_test
#
# Each invocation is self-contained: launches servers, runs one test, cleans up.
# This mirrors the comprehensive tests' run-single-config.sh pattern.
set -o pipefail

TEST_NAME="${1:?Usage: $0 <test_name>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"
source .buildkite/k3_tests/common_scripts/helpers.sh

# ── Configuration ────────────────────────────────────────────
export LMCACHE_PORT="${LMCACHE_PORT:-6555}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
export MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-300}"
export BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"

# gds_smoke_test enables the GDS L1 NVMe-slab tier
GDS_SCRATCH="${GDS_SCRATCH:-/scratch}"
if [ "$TEST_NAME" = "gds_smoke_test" ]; then
    export GDS_L1_PATH="${GDS_SCRATCH}/lmcache-gds-${BUILD_ID}-${TEST_NAME}"
    echo "GDS L1 tier enabled (slab dir: $GDS_L1_PATH)"
fi

# Per-test default model (overridable via the MODEL env var). The HMA test needs
# a hybrid model whose KV cache groups have different block sizes, so the
# connector exercises the per-group hybrid-memory-allocator path.
if [ "$TEST_NAME" = "hma_lm_eval_gemma4" ]; then
    # gemma-4-31B-it is public (no gating, so no HF token check) and has
    # heterogeneous head dims (head_dim 256 / global_head_dim 512), so vLLM
    # gives its KV cache groups different block sizes -- this is what exercises
    # LMCache's per-group block-size handling. It forces TRITON_ATTN, so the
    # pipeline sets ATTENTION_BACKEND=auto; its ~63GB of weights also need a
    # higher GPU_MEMORY_UTILIZATION than the default (all set in pipeline.yml).
    export MODEL="${MODEL:-google/gemma-4-31B-it}"
elif [ "$TEST_NAME" = "hma_lm_eval_qwen3_5" ]; then
    # Qwen3.5-0.8B is a Mamba/GDN + full-attention hybrid (caches re-viewed at
    # registration; see lmcache/integration/vllm/kv_cache_group_edits.py).
    export MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
    export ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
    # LMCache chunk size must be a multiple of the unified vLLM block size (544).
    export CHUNK_SIZE="${CHUNK_SIZE:-544}"
    # GDN supports only the 'align' Mamba cache mode.
    export MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
    # 'align' snapshots the Mamba state only at scheduler-step boundaries; cap
    # the step at the chunk size for one reusable snapshot per chunk.
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-544}"
    # GDN has no batch-invariant mode, so runs are not bit-exact; compare within
    # a score tolerance and use enough samples to shrink run-to-run drift
    # (~1/sqrt(LIMIT)) well inside it.
    export BATCH_INVARIANT="${BATCH_INVARIANT:-0}"
    export SCORE_TOLERANCE="${SCORE_TOLERANCE:-0.05}"
    export LIMIT="${LIMIT:-300}"
elif [ "$TEST_NAME" = "kimi_linear_tp" ]; then
    # Self-contained test: run-kimi-linear-tp.sh owns the server lifecycle and
    # all launch flags (TP=2, trust-remote-code, align, chunk/batch sizes). Only
    # the model name is declared here so the banner and the script's ${MODEL:-}
    # fallback both resolve to Kimi-Linear rather than the generic default below.
    export MODEL="${MODEL:-moonshotai/Kimi-Linear-48B-A3B-Instruct}"
else
    export MODEL="${MODEL:-Qwen/Qwen3-14B}"
fi
export CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
export MAX_WORKERS="${MAX_WORKERS:-4}"
export LMCACHE_DIR="$REPO_ROOT"
export RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"

mkdir -p "$RESULTS_DIR"

# Cleanup: always kill background processes on exit
trap '"${SCRIPT_DIR}/cleanup.sh"' EXIT

echo "============================================"
echo "=== LMCache Multiprocessing Test: ${TEST_NAME} ==="
echo "============================================"
echo "Build ID: $BUILD_ID"
echo "Model: $MODEL"
echo "LMCache port: $LMCACHE_PORT"
echo "vLLM port: $VLLM_PORT"
echo "vLLM baseline port: $VLLM_BASELINE_PORT"
echo "Results dir: $RESULTS_DIR"
echo ""

# Tests that handle their own server lifecycle (different GPU/model config)
SELF_CONTAINED_TESTS=" deadlock p2p kimi_linear_tp "

# Tests that compare against a baseline vLLM (no LMCache) on a second GPU.
# Only these need the baseline server (and thus a 2-GPU pod); everything
# else runs on GPU 0 alone, so launch-processes.sh skips the baseline.
BASELINE_TESTS=" vllm_bench long_doc_qa long_doc_qa_l2 "
if [[ "$BASELINE_TESTS" == *" $TEST_NAME "* ]]; then
    export LAUNCH_BASELINE=true
else
    export LAUNCH_BASELINE=false
fi

if [[ "$SELF_CONTAINED_TESTS" != *" $TEST_NAME "* ]]; then
    # ── Step 1: Launch native processes ──────────────────────────
    echo "============================================"
    echo "=== Launching native processes ==="
    echo "============================================"
    if ! "${SCRIPT_DIR}/launch-processes.sh"; then
        echo "Failed to launch processes"
        exit 1
    fi
    echo ""

    # ── Step 2: Wait for vLLM to be ready ───────────────────────
    echo "============================================"
    echo "=== Waiting for vLLM to be ready ==="
    echo "============================================"
    if ! "${SCRIPT_DIR}/wait-for-servers.sh"; then
        echo "vLLM failed to become ready"
        exit 1
    fi
    echo ""
fi

# ── Step 3: Run the requested test ──────────────────────────
echo "============================================"
echo "=== Running test: ${TEST_NAME} ==="
echo "============================================"

case "$TEST_NAME" in
    lm_eval)
        exec_script="${SCRIPT_DIR}/run-lm-eval.sh"
        ;;
    lm_eval_preemption)
        export LM_EVAL_VERIFY_MODE=preemption
        exec_script="${SCRIPT_DIR}/run-lm-eval.sh"
        ;;
    hma_lm_eval_gemma4)
        exec_script="${SCRIPT_DIR}/run-hma-lm-eval.sh"
        ;;
    hma_lm_eval_qwen3_5)
        exec_script="${SCRIPT_DIR}/run-hma-lm-eval.sh"
        ;;
    vllm_bench)
        exec_script="${SCRIPT_DIR}/run-vllm-bench.sh"
        ;;
    long_doc_qa)
        exec_script="${SCRIPT_DIR}/run-long-doc-qa.sh"
        ;;
    long_doc_qa_l2)
        exec_script="${SCRIPT_DIR}/run-long-doc-qa-l2.sh"
        ;;
    fault_tolerance)
        exec_script="${SCRIPT_DIR}/run-fault-tolerance.sh"
        ;;
    deadlock)
        exec_script="${SCRIPT_DIR}/run-deadlock.sh"
        ;;
    restart_recovery)
        exec_script="${SCRIPT_DIR}/run-restart-recovery.sh"
        ;;
    cache_stats)
        exec_script="${SCRIPT_DIR}/run-cache-stats.sh"
        ;;
    p2p)
        exec_script="${SCRIPT_DIR}/run-p2p.sh"
        ;;
    kimi_linear_tp)
        exec_script="${SCRIPT_DIR}/run-kimi-linear-tp.sh"
        ;;
    http_api)
        exec_script="${SCRIPT_DIR}/run-http-api.sh"
        ;;
    gds_smoke_test)
        exec_script="${SCRIPT_DIR}/run-gds-smoke.sh"
        ;;
    *)
        echo "Unknown test: $TEST_NAME"
        echo "Valid tests: lm_eval, lm_eval_preemption, hma_lm_eval_gemma4, vllm_bench, long_doc_qa, long_doc_qa_l2, fault_tolerance, deadlock, restart_recovery, cache_stats, http_api, gds_smoke_test, p2p, kimi_linear_tp"
        exit 1
        ;;
esac

if ! "$exec_script"; then
    echo "${TEST_NAME} test failed"
    exit 1
fi

echo ""
echo "============================================"
echo "=== Test ${TEST_NAME} passed! ==="
echo "============================================"
