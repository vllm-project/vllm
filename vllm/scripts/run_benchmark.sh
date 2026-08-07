cd villum
VLLM_USE_PRECOMPILED=1 uv pip install --editable .

export HF_TOKEN=""

#export MODEL="QuixiAI/Qwen3-30B-A3B-AWQ"
export MODEL="unsloth/gpt-oss-20b"

export HF_HOME="/data/huggingface"

VLLM_NIXL_SIDE_CHANNEL_PORT=5600 UCX_NET_DEVICES=all CUDA_VISIBLE_DEVICES=0 \
VLLM_NIXL_SIDE_CHANNEL_HOST=[IP_ADDRESS] vllm serve "$MODEL" \
  --port 8100 --enforce-eager --gpu-memory-utilization 0.9 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'

CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
VLLM_NIXL_SIDE_CHANNEL_HOST=[IP_ADDRESS] vllm serve "$MODEL" \
  --port 8200 --enforce-eager --gpu-memory-utilization 0.9 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'