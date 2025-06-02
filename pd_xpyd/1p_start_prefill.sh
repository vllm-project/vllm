
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

if [ -z "$1" ] || [ "$1" == "g10" ]; then
    source "$BASH_DIR"/start_etc_mooncake_master.sh
    echo "source "$BASH_DIR"/start_etc_mooncake_master.sh"
fi


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export MOONCAKE_CONFIG_PATH="$BASH_DIR"/mooncake_${1:-g10}.json

echo "Using Mooncake config: $MOONCAKE_CONFIG_PATH"

source "$BASH_DIR"/dp_p_env.sh

#unset VLLM_SKIP_WARMUP
#export PT_HPU_RECIPE_CACHE_CONFIG=./_prefill_cache,false,16384

python3 -m vllm.entrypoints.openai.api_server --model $model_path --port 8100 --max-model-len $model_len --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION -tp 8  --max-num-seqs $max_num_seqs --trust-remote-code --disable-async-output-proc --disable-log-requests --max-num-batched-tokens $max_num_batched_tokens --use-padding-aware-scheduling --use-v2-block-manager --distributed_executor_backend mp --kv-cache-dtype fp8_inc --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}'

