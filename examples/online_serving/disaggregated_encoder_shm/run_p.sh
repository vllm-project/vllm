unset ftp_proxy
unset https_proxy
unset http_proxy
EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/user/ec_cache}"
export VLLM_NIXL_SIDE_CHANNEL_PORT=5600

CUDA_VISIBLE_DEVICES=1 vllm serve "/model_path" \
    --gpu-memory-utilization 0.7 \
    --port "33001" \
    --enforce-eager \
    --enable-request-id-headers \
    --served-model-name model_name \
    --max-model-len 32768  \
    --max-num-seqs 128 \
    --ec-transfer-config '{
      "ec_connector": "SHMConnector",
      "ec_role": "ec_consumer",
      "ec_ip": "127.0.0.1",
      "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'",
            "listen_ports": [30161]
        }
    }' \
    --kv-transfer-config '{
      "kv_connector": "NixlConnector",
      "kv_role": "kv_producer"
    }' 



