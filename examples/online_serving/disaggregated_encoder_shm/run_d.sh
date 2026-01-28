unset ftp_proxy
unset https_proxy
unset http_proxy
export VLLM_NIXL_SIDE_CHANNEL_PORT=5601

CUDA_VISIBLE_DEVICES=2 vllm serve "/model_path" \
    --gpu-memory-utilization 0.7 \
    --port "43001" \
    --enforce-eager \
    --enable-request-id-headers \
    --served-model-name model_name \
    --max-model-len 32768  \
    --max-num-seqs 128 \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer"
    }'