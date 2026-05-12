#!/bin/bash
# ============================================================================
# Start vLLM server with a chosen video backend (deepstream or opencv).
# Run this first, then use run_bench_client.sh in another terminal.
#
# Usage:
#   ./start_server.sh                  # defaults: deepstream backend
#   ./start_server.sh opencv           # opencv backend
#   ./start_server.sh deepstream       # deepstream backend
#
# Environment overrides:
#   PORT=8000  MODEL=/path/to/model  GPU_MEM=0.8  NUM_FRAMES=8
# ============================================================================
set -euo pipefail

BACKEND="${1:-deepstream}"
MODEL="${MODEL:-/work/deepstream_9.0_vllm/Qwen2-VL-2B-Instruct}"
#MODEL="${MODEL:-/work/deepstream_9.0_vllm/Qwen3-VL-2B-Instruct}"
#MODEL="${MODEL:-/work/deepstream_9.0_vllm/InternVL3_5-1B}"
#MODEL="${MODEL:-/work/deepstream_9.0_vllm/Qwen3-VL-4B-Instruct}"
PORT="${PORT:-8000}"
GPU_MEM="${GPU_MEM:-0.8}"
NUM_FRAMES="${NUM_FRAMES:-8}"
# export VLLM_ATTENTION_BACKEND=FLASHINFER
# Pre-computed KV cache block count — skips the 2+ min GPU memory profiling step.
# Calibrated at GPU_MEM=0.7 on H200 (91 GiB free → 3,412,496 tokens / 16 = 213,281 blocks).
# Clear this (set to "") when changing GPU_MEM or the model so it re-profiles once.
#NUM_GPU_BLOCKS="${NUM_GPU_BLOCKS:-213281}"
LOG_DIR="/work/deepstream_9.0_vllm/bench_results"
SERVER_LOG="${LOG_DIR}/server_${BACKEND}.log"

mkdir -p "$LOG_DIR"

# Stage video if needed
mkdir -p /data/video 2>/dev/null || true
VIDEO_SRC="/work/via-engine/tests/alerts/verifyAlerts_clips/verifyAlerts_raw/drive_sim_collision_1080p_2025-08-01T04-18-22.000752Z_clip_5.mp4"
VIDEO="/data/video/drivesim.mp4"
if [ ! -f "$VIDEO" ] && [ -f "$VIDEO_SRC" ]; then
    echo "Staging video: $VIDEO_SRC -> $VIDEO"
    cp "$VIDEO_SRC" "$VIDEO"
fi

export HF_HUB_OFFLINE=1
export VLLM_MULTIMODAL_TENSOR_IPC=1
export TRANSFORMERS_OFFLINE=1
export CUDA_MODULE_LOADING=LAZY
export VLLM_MEDIA_LOADING_THREAD_COUNT=16
# H200 is SM 9.0 — vLLM's default is FLASH_ATTN (FlashAttention-3 with Hopper TMA/wgmma).
# FlashInfer is only preferred on SM 10.0 (Blackwell). Don't override unless benchmarking.
# export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_VIDEO_LOADER_BACKEND=$BACKEND

# Persistent kernel caches — survives container restarts, avoids Triton JIT recompile
export TRITON_CACHE_DIR=/work/deepstream_9.0_vllm/.triton_cache
export VLLM_CACHE_ROOT=/work/deepstream_9.0_vllm/.vllm_cache

MEDIA_IO_KWARGS="{\"video\": {\"num_frames\": ${NUM_FRAMES}}}"

echo "============================================================"
echo "  vLLM Server"
echo "============================================================"
echo "  Backend        : $BACKEND"
echo "  Model          : $MODEL"
echo "  Port           : $PORT"
echo "  GPU mem util   : $GPU_MEM"
echo "  Frames/video   : $NUM_FRAMES"
echo "  Server log     : $SERVER_LOG"
echo "============================================================"
echo ""

# Kill any existing server on this port
if curl -s --max-time 1 http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "  Killing existing server on port $PORT..."
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "EngineCore" 2>/dev/null || true
    sleep 3
fi

echo "  Starting vLLM server..."
echo "  (logs → $SERVER_LOG)"
echo ""

GPU_BLOCKS_ARG=""
if [ -n "${NUM_GPU_BLOCKS:-}" ]; then
    GPU_BLOCKS_ARG="--num-gpu-blocks-override $NUM_GPU_BLOCKS"
fi

vllm serve "$MODEL" \
    --limit-mm-per-prompt '{"video": 1}' \
    --media-io-kwargs "$MEDIA_IO_KWARGS" \
    --gpu-memory-utilization $GPU_MEM \
    --host 0.0.0.0 \
    --port $PORT \
    --dtype bfloat16 \
    --enforce-eager \
    --served-model-name bench-model \
    --max-model-len 20000 \
    --max-num-seqs 16 \
    --trust-remote-code \
    --allowed-local-media-path /data \
    --skip-mm-profiling \
    $GPU_BLOCKS_ARG \
    2>&1 | tee "$SERVER_LOG"
