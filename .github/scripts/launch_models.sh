#!/bin/bash

# Unified vLLM Model Launcher
# Usage: ./launch_unified.sh [model_name] [model_path_override]
# Available models:
#   - deepseek-r1
#   - deepseek-r1-fp8
#   - qwen25vl-72b
#   - qwen25vl-72b-fp8
#   - qwen3vl-235b
#   - qwen3-coder-480b

MODEL_NAME="${1:-deepseek-r1-fp8}"
MODEL_PATH_OVERRIDE="$2"

# Clean cache
rm -rf /root/.cache/vllm

# Common environment variables
export VLLM_USE_V1=1
export VLLM_RPC_TIMEOUT=1800000

# Model-specific configurations
case "$MODEL_NAME" in
  "deepseekr1_ptpc_fp8")
    export GPU_ARCHS=gfx942
    export AITER_ENABLE_VSKIP=0
    export VLLM_ROCM_USE_AITER=1
    export SAFETENSORS_FAST_GPU=1
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    export VLLM_USE_TRITON_FLASH_ATTN=0
    export NCCL_DEBUG=WARN
    #export VLLM_LOGGING_LEVEL=DEBUG
    export VLLM_ROCM_USE_AITER_MHA=0
    export VLLM_ROCM_USE_TRITON_ROPE=1
    
    export VLLM_TORCH_PROFILER_DIR="deepseek_in3k_out1k"
    export VLLM_TORCH_PROFILER_WITH_STACK=1
    export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
    
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-EmbeddedLLM/deepseek-r1-FP8-Dynamic/}"
    TP_SIZE=8
    EXTRA_ARGS=(
      --disable-log-requests
      --max-num-batched-tokens 32768
      --trust-remote-code
      --no-enable-prefix-caching
      --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}'
      --gpu_memory_utilization 0.9
      --block-size 1
    )
    ;;
    
  "deepseekr1")
    export GPU_ARCHS=gfx942
    export AITER_ENABLE_VSKIP=0
    export VLLM_ROCM_USE_AITER=1
    export SAFETENSORS_FAST_GPU=1
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    export VLLM_USE_TRITON_FLASH_ATTN=0
    export NCCL_DEBUG=WARN
    export VLLM_RPC_TIMEOUT=1800000
    export VLLM_ROCM_USE_AITER_MHA=0
    export VLLM_ROCM_USE_TRITON_ROPE=1 # add for acc
    export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 
    
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-deepseek-ai/DeepSeek-R1}"
    TP_SIZE=8
    EXTRA_ARGS=(
      --disable-log-requests
      --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}'
      --trust-remote-code
      --block-size 1
    )
    ;;
    
  "qwen3next")
    export VLLM_ROCM_USE_AITER=1
    export SAFETENSORS_FAST_GPU=1
    export VLLM_ROCM_USE_AITER_MHA=0 
    export HIP_VISIBLE_DEVICES=0,1,2,3

    pip install flash-linear-attention
    git clone https://github.com/Dao-AILab/causal-conv1d.git
    cd causal-conv1d
    python3 setup.py install

    # cudagraph|tp=4|bf16
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
    TP_SIZE=4
    EXTRA_ARGS=(
      --max-model-len 262114
      --disable-log-requests
      --trust-remote-code
      --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}'
    )
    ;;
    
  "qwen25vl-72b")
    export VLLM_ROCM_USE_AITER=1
    export SAFETENSORS_FAST_GPU=1
    
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-Qwen/Qwen2.5-VL-72B-Instruct}"
    TP_SIZE=4
    EXTRA_ARGS=(
      --mm-encoder-tp-mode "data"
      --trust-remote-code
      --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}'
    )
    ;;
    
  "qwen25vl-72b-fp8")
    export VLLM_ROCM_USE_AITER=1
    export SAFETENSORS_FAST_GPU=1
    
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic}"
    TP_SIZE=2
    EXTRA_ARGS=(
      --mm-encoder-tp-mode "data"
      --gpu-memory-utilization 0.8
      --trust-remote-code
    )
    ;;
    
  "qwen3vl-235b")
    export AITER_ENABLE_VSKIP=0
    export AITER_ONLINE_TUNE=1
    export VLLM_ROCM_USE_AITER=1
    export SAFETENSORS_FAST_GPU=1
    
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-Qwen/Qwen3-VL-235B-A22B-Instruct}"
    TP_SIZE=8
    EXTRA_ARGS=(
      --mm-encoder-tp-mode "data"
      --gpu-memory-utilization 0.8
      --trust-remote-code
    )
    ;;
    
  "qwen3coder_ptpc_fp8")
    export GPU_ARCHS=gfx942
    export AITER_ENABLE_VSKIP=0
    export AITER_ONLINE_TUNE=1
    export VLLM_ROCM_USE_AITER=1
    
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-EmbeddedLLM/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic}"
    TP_SIZE=8
    EXTRA_ARGS=(
      --max-model-len 65536
      --disable-log-requests
      --trust-remote-code
      --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}'
    )
    ;;
    
  *)
    echo "Error: Unknown model name '$MODEL_NAME'"
    echo "Available models:"
    echo "  - deepseekr1"
    echo "  - deepseekr1_ptpc_fp8"
    echo "  - qwen25vl-72b"
    echo "  - qwen25vl-72b-fp8"
    echo "  - qwen3vl-235b"
    echo "  - qwen3coder_ptpc_fp8"
    exit 1
    ;;
esac

# Build the command
CMD="vllm serve $MODEL_PATH"
CMD="$CMD --tensor-parallel-size $TP_SIZE"

# Add extra arguments
for arg in "${EXTRA_ARGS[@]}"; do
  CMD="$CMD $arg"
done

# Print configuration
echo "=========================================="
echo "vLLM Unified Launcher"
echo "=========================================="
echo "Model Name:       $MODEL_NAME"
echo "Model Path:       $MODEL_PATH"
echo "Tensor Parallel:  $TP_SIZE"
echo "Command:          $CMD"
echo "=========================================="
echo ""

# Execute
echo "Starting vLLM server..."
eval "$CMD"

