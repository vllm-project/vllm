#set -x

source ./pd_xpyd/start_etc_mooncake_master.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export MOONCAKE_CONFIG_PATH=./pd_xpyd/2p2d_mooncake_p0.json 

source ./pd_xpyd/dp_p_env.sh

ray start --head --port=6886


while true; do
    read -p "Continue? (y): " answer
    if [[ "$answer" == [yY] ]]; then
        echo "Proceeding..."
        break
    else
        echo "Invalid input. Please enter y or n."
    fi
done

#unset VLLM_SKIP_WARMUP
#export PT_HPU_RECIPE_CACHE_CONFIG=./_prefill_cache,false,16384

python3 -m vllm.entrypoints.openai.api_server --model $model_path --port 8100 --max-model-len $model_len --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION -tp 16  --max-num-seqs $max_num_seqs --trust-remote-code --disable-async-output-proc --kv-cache-dtype fp8_inc --disable-log-requests --max-num-batched-tokens $max_num_batched_tokens --use-padding-aware-scheduling --use-v2-block-manager --distributed_executor_backend ray --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}'
