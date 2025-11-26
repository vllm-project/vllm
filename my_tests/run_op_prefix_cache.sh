#!/bin/bash

PORT=8235
TP=2
MAX_MODEL_LEN=262144
# MAX_MODEL_LEN=131072

DO_NSYS=0


MODEL_DIR=/mnt/disk0/huanghaoyan.hhy/Qwen3-Next-80B-A3B-Instruct/
echo "MODEL_DIR: $MODEL_DIR"

NSYS_OUTPUT="qwen_next_h20_tp1_nreq1_fp8_mtp1_prefixcache_2"
NSYS=""
if (( DO_NSYS == 1 )); then
    NSYS="nsys profile -c cudaProfilerApi --cuda-graph-trace node -o $NSYS_OUTPUT"
fi

env_vars=(
    # "CUDA_LAUNCH_BLOCKING=0"
    "CUDA_VISIBLE_DEVICES=0,1,2,3"
    "VLLM_USE_LIGHTER_MAMBA_CACHE=1"
    # "CUDA_VISIBLE_DEVICES=6,7"
    # "VLLM_ATTENTION_BACKEND=FLASH_ATTN"
    # "VLLM_FLASH_ATTN_VERSION=3"
    # "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
    # "OMP_NUM_THREADS=1"
    # "VLLM_USE_V1=1"
    # "VLLM_LOG_REQ_KV_LENS=1"
    # "VLLM_USE_FLASHINFER_SAMPLER=0"
)

for var in "${env_vars[@]}"; do
    var_name="${var%%=*}"
    var_value="${var#*=}"
    echo -e "\t$var_name=$var_value"
done

CMD=( env )
for var in "${env_vars[@]}"; do
    CMD+=( "$var" )
done
CMD+=(
    $NSYS vllm serve
    $MODEL_DIR
    # --trust-remote-code
    --port "$PORT"
    --gpu-memory-utilization 0.9
    -tp $TP
    --enforce-eager
    # --no-enable-prefix-caching
    --enable-prefix-caching
    # --no-enable-chunked-prefill
    --enable-chunked-prefill
    --max-num-batched-tokens 8192
    --distributed-executor-backend mp
    --block-size 64
    --max-num-seqs 128
    # --max-num-seqs 16
    # --max-model-len $MAX_MODEL_LEN
    # --max-seq-len-to-capture $MAX_MODEL_LEN
    # --compilation-config "{\"use_inductor\": false, \"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"custom_ops\": [\"all\"]}"
    # --speculative-config "{\"method\": \"qwen3_next_mtp\", \"num_speculative_tokens\": 3}"
    # --hf_overrides "{\"max_position_embeddings\": $MAX_MODEL_LEN}"
)

echo -e "\nExecuting command:"
printf " %s" "${CMD[@]}"
echo -e "\n"

"${CMD[@]}"
