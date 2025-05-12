#!/bin/bash
set -x
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source ./pd_xpyd/dp_d_env.sh
export MOONCAKE_CONFIG_PATH=./pd_xpyd/2p2d_mooncake_d0.json

export VLLM_DP_SIZE=32
export VLLM_EP_SIZE=32

TOTAL_INSTANCES=8

if [ -z "$1" ]; then
    echo "please input the dp size per node, for example, 16dp on 2 node, run the xxx.sh 8"
    echo "run with default mode n=8"
    NUM_GROUPS=8
else
    NUM_GROUPS=${1:-1}
fi

NUM_INSTANCES=$((TOTAL_INSTANCES / NUM_GROUPS))

dp_size=$((4 * NUM_GROUPS))
export VLLM_DP_SIZE=$dp_size

for ((i=0; i<NUM_GROUPS; i++))
do
  
  RANK=$((0 + i))
  port=$((8200 + i))
  
  VLLM_DP_RANK=$RANK python3 -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --port "$port" \
    --max-model-len "$model_len" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    -tp $NUM_INSTANCES \
    --max-num-seqs "$max_num_seqs" \
    --trust-remote-code \
    --kv-cache-dtype fp8_inc \
    --disable-log-requests \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    --use-padding-aware-scheduling \
    --use-v2-block-manager \
    --distributed_executor_backend mp \
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}' &
done

wait

