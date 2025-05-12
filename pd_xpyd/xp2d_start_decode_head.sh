#set -x
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/dp_d_env.sh

export VLLM_EP_SIZE=16
export MOONCAKE_CONFIG_PATH=./pd_xpyd/2p2d_mooncake_d0.json

unset VLLM_DP_SIZE
unset VLLM_USE_V1
unset VLLM_DP_MASTER_IP
unset VLLM_DP_MASTER_PORT

ray start --head --port=8826

while true; do
    read -p "Continue? (y): " answer
    if [[ "$answer" == [yY] ]]; then
        echo "Proceeding..."
        break
    else
        echo "Invalid input. Please enter y or n."
    fi
done

python3 -m vllm.entrypoints.openai.api_server --model $model_path --port 8200 --max-model-len $model_len --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION -tp 16 --max-num-seqs $max_num_seqs --trust-remote-code --kv-cache-dtype fp8_inc --disable-log-requests --max-num-batched-tokens $max_num_batched_tokens --use-padding-aware-scheduling --use-v2-block-manager --distributed_executor_backend ray --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
