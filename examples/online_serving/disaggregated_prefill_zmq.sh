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
    until grep -q 'PDWorker is ready' $log_file; do
      sleep 1
    done" && return 0 || return 1
}


# You can also adjust --kv-ip and --kv-port for distributed inference.
MODEL=meta-llama/Llama-3.1-8B-Instruct
CONTROLLER_ADDR=controller.ipc
PREFILL_WORKER_ADDR=prefill.ipc
DECODE_WORKER_ADDR=decode.ipc
PORT=8001

# prefilling instance, which is the KV producer
CUDA_VISIBLE_DEVICES=0 python3 ../../vllm/entrypoints/disaggregated/worker.py \
    --model $MODEL \
    --controller-addr $CONTROLLER_ADDR \
    --worker-addr $PREFILL_WORKER_ADDR \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' > vllm_disagg_prefill.log 2>&1 &

# decoding instance, which is the KV consumer
CUDA_VISIBLE_DEVICES=1 python3 ../../vllm/entrypoints/disaggregated/worker.py \
    --model $MODEL \
    --controller-addr $CONTROLLER_ADDR \
    --worker-addr $DECODE_WORKER_ADDR \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' > vllm_disagg_decode.log 2>&1 &

# launch a proxy server that opens the service at port 8000
# the workflow of this proxy:
# - Send req to prefill instance, wait until complete.
# - Send req to decode instance, streaming tokens.
python3 ../../vllm/entrypoints/disaggregated/api_server.py \
    --port $PORT \
    --model $MODEL \
    --controller-addr $CONTROLLER_ADDR \
    --prefill-addr $PREFILL_WORKER_ADDR \
    --decode-addr $DECODE_WORKER_ADDR &

# wait until prefill, decode instances and proxy are ready
wait_for_server $PORT
wait_for_disagg_server vllm_disagg_prefill.log
wait_for_disagg_server vllm_disagg_decode.log 

# serve two example requests
output1=$(curl -X POST -s http://localhost:8001/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Llama-3.1-8B-Instruct",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}')

output2=$(curl -X POST -s http://localhost:8001/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Llama-3.1-8B-Instruct",
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
