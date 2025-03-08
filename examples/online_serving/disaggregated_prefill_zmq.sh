#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling with ZMQ
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

set -xe

echo "🚧🚧 Warning: The usage of disaggregated prefill is experimental and subject to change 🚧🚧"
sleep 1

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

# a function that waits vLLM connect to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


# a function that waits vLLM disagg to start
wait_for_disagg_server() {
  local log_file=$1
  timeout 1200 bash -c "
    until grep -q 'zmq Server started at' $log_file; do
      sleep 1
    done" && return 0 || return 1
}


# You can also adjust --kv-ip and --kv-port for distributed inference.

# prefilling instance, which is the KV producer
CUDA_VISIBLE_DEVICES=0 vllm disagg meta-llama/Meta-Llama-3.1-8B-Instruct \
    --zmq-server-addr testipc0 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' > vllm_disagg_prefill.log 2>&1 &

# decoding instance, which is the KV consumer
CUDA_VISIBLE_DEVICES=1 vllm disagg meta-llama/Meta-Llama-3.1-8B-Instruct \
    --zmq-server-addr testipc1 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' > vllm_disagg_decode.log 2>&1 &

# launch a proxy server that opens the service at port 8000
# the workflow of this proxy:
# - send the request to prefill vLLM instance (via zmq addr testipc0), change max_tokens
#   to 1
# - after the prefill vLLM finishes prefill, send the request to decode vLLM 
#   instance (via zmq addr testipc1)
vllm connect --port 8000 \
    --prefill-addr testipc0 \
    --decode-addr testipc1 &

# wait until prefill, decode instances and proxy are ready
wait_for_server 8000
wait_for_disagg_server vllm_disagg_prefill.log
wait_for_disagg_server vllm_disagg_decode.log 

# serve two example requests
output1=$(curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}')

output2=$(curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
"prompt": "Santa Clara is a",
"max_tokens": 10,
"temperature": 0
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
