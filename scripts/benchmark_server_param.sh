#!/bin/bash
set -x

# Usage: benchmark_server_param.sh NUM_NODES MAX_MODEL_LEN MAX_NUM_SEQS TP_SIZE PP_SIZE \
#                                  COMM_BACKEND [PP_LAYER_PARTITION] [KV_CACHE_DTYPE] \
#                                  [DO_WARMUP] [DO_PROFILE] [HOST] [PORT] [MODEL_PATH] [RESULTS_DIR]
#
# Arguments:
#   NUM_NODES          Number of nodes to use for the server.
#   MAX_MODEL_LEN      Maximum model length (number of tokens).
#   MAX_NUM_SEQS       Maximum number of sequences to process concurrently.
#   TP_SIZE            Tensor parallelism size.
#   PP_SIZE            Pipeline parallelism size.
#   COMM_BACKEND       Communication backend to use (e.g., hccl, gloo).
#   PP_LAYER_PARTITION (Optional) Layer partitioning for pipeline parallelism (comma-separated list).
#   KV_CACHE_DTYPE     (Optional) Data type for KV cache (e.g., auto, fp8_inc). Default: auto.
#   DO_WARMUP          (Optional) Whether to perform warmup before benchmarking (true/false). Default: true.
#   DO_PROFILE         (Optional) Whether to enable profiling (true/false). Default: false.
#   HOST               (Optional) Host address for the server. Default: 127.0.0.1.
#   PORT               (Optional) Port for the server. Default: 8688.
#   MODEL_PATH         (Optional) Path to the model. Default: /root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2.
#   RESULTS_DIR        (Optional) Directory to store results and logs. Default: logs/test-results.
#
# Description:
#   This script launches a vLLM server with the specified configuration for benchmarking.
#   It supports various parallelism configurations (tensor and pipeline), communication backends,
#   and optional profiling. The script sets up the environment, configures memory and scheduling
#   parameters, and starts the server with the provided arguments.
#
#   Use this script as part of a benchmarking workflow to evaluate the performance of vLLM
#   under different configurations.

NUM_NODES=$1
MAX_MODEL_LEN=$2
MAX_NUM_SEQS=$3
TP_SIZE=$4
PP_SIZE=$5
COMM_BACKEND=$6
PP_LAYER_PARTITION=${7:-}
KV_CACHE_DTYPE=${8:-auto}
DO_WARMUP=${9:-true}
DO_PROFILE=${10:-false}
HOST=${11:-127.0.0.1}
PORT=${12:-8688}
MODEL_PATH=${13:-${MODEL_PATH:-/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2}}
RESULTS_DIR=${14:-logs/test-results}

if [ "$DO_PROFILE" == "true" ]; then
  hl-prof-config --use-template profile_api --hw-trace off
  export HABANA_PROFILE=1
  export VLLM_PROFILER_ENABLED=full
  export VLLM_TORCH_PROFILER_DIR=${RESULTS_DIR}/profiler/
fi

# Environment settings
export HABANA_VISIBLE_DEVICES="ALL"
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_RAY_DISABLE_LOG_TO_DRIVER="1"
export RAY_IGNORE_UNHANDLED_ERRORS="1"
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_MODULES="0,1,2,3,4,5,6,7"

if [ "$DO_WARMUP" == "true" ]; then
  export VLLM_SKIP_WARMUP=false
else
  export VLLM_SKIP_WARMUP=true
fi
export VLLM_MLA_DISABLE_REQUANTIZATION=1
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export VLLM_DELAYED_SAMPLING="false"

# memory footprint tunning params
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.75}
export VLLM_GRAPH_RESERVED_MEM=${VLLM_GRAPH_RESERVED_MEM:-0.4}
export VLLM_GRAPH_PROMPT_RATIO=0

export VLLM_EP_SIZE=$TP_SIZE
if [ "$PP_SIZE" -gt 1 ]; then
  if [ -n "$PP_LAYER_PARTITION" ]; then
    echo "PP_SIZE = ${PP_SIZE}, PP_LAYER_PARTITION = ${PP_LAYER_PARTITION}"
    export VLLM_PP_LAYER_PARTITION=$PP_LAYER_PARTITION
  else
    echo "Warning: PP_SIZE > 1 but PP_LAYER_PARTITION not provided"
  fi
fi

if [ "$COMM_BACKEND" == "gloo" ]; then
  export VLLM_PP_USE_CPU_COMS=1
fi

if [ "$KV_CACHE_DTYPE" == "fp8_inc" ]; then
  # Required to improve performance with FP8 KV cache.
  export VLLM_USE_FP8_MATMUL="true"
fi

# Bucketing configuration
BLOCK_SIZE=128
export PT_HPU_RECIPE_CACHE_CONFIG="/data/${MAX_MODEL_LEN}_cache,false,${MAX_MODEL_LEN}"
MAX_NUM_BATCHED_TOKENS=$MAX_MODEL_LEN

prompt_bs_min=1
prompt_bs_step=$(( MAX_NUM_SEQS > 32 ? 32 : MAX_NUM_SEQS ))
prompt_bs_max=$(( MAX_NUM_SEQS > 64 ? 64 : MAX_NUM_SEQS ))
export VLLM_PROMPT_BS_BUCKET_MIN=${VLLM_PROMPT_BS_BUCKET_MIN:-$prompt_bs_min}
export VLLM_PROMPT_BS_BUCKET_STEP=${VLLM_PROMPT_BS_BUCKET_STEP:-$prompt_bs_step}
export VLLM_PROMPT_BS_BUCKET_MAX=${VLLM_PROMPT_BS_BUCKET_MAX:-$prompt_bs_max}

prompt_seq_min=128
prompt_seq_step=128
prompt_seq_max=$MAX_NUM_BATCHED_TOKENS
export VLLM_PROMPT_SEQ_BUCKET_MIN=${VLLM_PROMPT_SEQ_BUCKET_MIN:-$prompt_seq_min}
export VLLM_PROMPT_SEQ_BUCKET_STEP=${VLLM_PROMPT_SEQ_BUCKET_STEP:-$prompt_seq_step}
export VLLM_PROMPT_SEQ_BUCKET_MAX=${VLLM_PROMPT_SEQ_BUCKET_MAX:-$prompt_seq_max}

decode_bs_min=1
decode_bs_step=$(( MAX_NUM_SEQS > 32 ? 32 : MAX_NUM_SEQS ))
decode_bs_max=$MAX_NUM_SEQS
export VLLM_DECODE_BS_BUCKET_MIN=${VLLM_DECODE_BS_BUCKET_MIN:-$decode_bs_min}
export VLLM_DECODE_BS_BUCKET_STEP=${VLLM_DECODE_BS_BUCKET_STEP:-$decode_bs_step}
export VLLM_DECODE_BS_BUCKET_MAX=${VLLM_DECODE_BS_BUCKET_MAX:-$decode_bs_max}

decode_block_min=128
decode_block_step=128
decode_block_max=$(( ((MAX_NUM_SEQS * MAX_MODEL_LEN / BLOCK_SIZE) > 128) ? (MAX_NUM_SEQS * MAX_MODEL_LEN / BLOCK_SIZE) : 128 ))
export VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN:-$decode_block_min}
export VLLM_DECODE_BLOCK_BUCKET_STEP=${VLLM_DECODE_BLOCK_BUCKET_STEP:-$decode_block_step}
export VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX:-$decode_block_max}

echo "Environments set for ${NUM_NODES}-node server: MAX_MODEL_LEN=${MAX_MODEL_LEN}, MAX_NUM_SEQS=${MAX_NUM_SEQS}, TP_SIZE=${TP_SIZE}, PP_SIZE=${PP_SIZE}, COMM_BACKEND=${COMM_BACKEND}"
env | grep VLLM

python3 -m vllm.entrypoints.openai.api_server --host $HOST --port $PORT \
  --block-size $BLOCK_SIZE \
  --model $MODEL_PATH \
  --device hpu \
  --dtype bfloat16 \
  --kv-cache-dtype $KV_CACHE_DTYPE \
  --tensor-parallel-size $TP_SIZE \
  --pipeline-parallel-size $PP_SIZE \
  --trust-remote-code \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs $MAX_NUM_SEQS \
  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
  --disable-log-requests \
  --use-padding-aware-scheduling \
  --use-v2-block-manager \
  --distributed_executor_backend ray \
  --gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
  --enable-reasoning \
  --reasoning-parser deepseek_r1
