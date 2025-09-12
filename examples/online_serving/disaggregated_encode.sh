#!/bin/bash
# This file demonstrates the example usage of disaggregated multi-modal encoding with NIXLConnector.
# We will launch 2 vllm instances (1 for encode and 1 for decode),
# and then transfer the KV cache between them.

set -xe

echo "🚧🚧 Warning: The usage of disaggregated encoding is experimental and subject to change 🚧🚧"
sleep 1

MODEL_NAME=${HF_MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# install quart first -- required for disagg encode proxy serve
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi 

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/chat/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


# You can also adjust --kv-ip and --kv-port for distributed inference.

# prefilling instance, which is the KV producer
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5557 vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --port 8100 \
    --max-model-len 256 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_encoder_mode":"epd_encoder"}' &

# decoding instance, which is the KV consumer
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5558 vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --port 8200 \
    --max-model-len 256 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

# launch a proxy server that opens the service at port 8000
# the workflow of this proxy:
# - send the request to encode vLLM instance (port 8100), change max_tokens 
#   to 1 and add "do_remote_decode": true in kv_transfer_params
# - after the encode vLLM finishes encoding, send the request to decode vLLM 
#   instance
# NOTE: the usage of this API is subject to change --- in the future we will 
# introduce "vllm connect" to connect between encode and decode instances
python3 ../../tests/v1/kv_connector/nixl_integration/toy_epd_proxy_server.py &
sleep 1

# serve two example requests
output1=$(curl -X POST -s http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"messages": [{
  "role": "user",
    "content": [
      {"type": "text", "text": "Describe the image."},
      {"type": "image_url", "image_url": {"url": "https://picsum.photos/id/238/200/300"}}
    ]
  }]
}')

output2=$(curl -X POST -s http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"messages": [{
  "role": "user",
    "content": [
      {"type": "text", "text": "Compare two images."},
      {"type": "image_url", "image_url": {"url": "https://picsum.photos/id/200/300/300"}},
      {"type": "image_url", "image_url": {"url": "https://picsum.photos/id/201/300/300"}}
    ]
  }]
}')


# Cleanup commands
pgrep python | xargs kill -9
pkill -f python

echo ""

sleep 1

# Print the outputs of the curl requests
echo ""
echo "Output of first request: $output1"
echo "Output of second request: $output2"

echo "🎉🎉 Successfully finished 2 test requests! 🎉🎉"
echo ""
