#!/bin/bash

# Requirement: 3x GPUs.

# Model: Qwen/Qwen2.5-VL-3B-Instruct
# Query:  QPS 2/4/6/8, 100 requests
# Resource: 3x GPU
# Approaches:
# 1. Disaggregated Encode - Prefill+Decode: 1 encode instance, 1 prefill+decode instance
# 2. Disaggregated Encode - Prefill - Decode: 1 encode instance, 1 prefill instance, 1 decode instance

set -ex

kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  # vLLM now names the process with VLLM prefix after https://github.com/vllm-project/vllm/pull/21445
  pgrep VLLM | xargs -r kill -9
  for port in 8000 8100 8200 8300; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


launch_e_pd_benchmark() {
  model="Qwen/Qwen2.5-VL-3B-Instruct"
  
  # disagg encode
  CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=5560 vllm serve $model \
    --port 8100 \
    --trust-remote-code --enable-request-id-headers \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_encoder_mode":"encoder_only"}' &

  # disagg prefill + decode
  CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=5561 vllm serve $model \
    --port 8200 \
    --trust-remote-code --enable-request-id-headers \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

  wait_for_server 8100
  wait_for_server 8200
  python3 disagg_encode_route.py &
  sleep 1
}


launch_e_p_d_benchmark() {
  model="Qwen/Qwen2.5-VL-3B-Instruct"

  # disagg encode
  CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=5560 vllm serve $model \
    --port 8100 \
    --trust-remote-code --enable-request-id-headers \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_encoder_mode":"encoder_only"}' &

  # disagg prefill
  CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=5561 vllm serve $model \
    --port 8200 \
    --trust-remote-code --enable-request-id-headers \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_encoder_mode":"ep_encoder"}' &
  
  # disagg decode
  CUDA_VISIBLE_DEVICES=2 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=5562 vllm serve $model \
    --port 8300 \
    --trust-remote-code --enable-request-id-headers \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

  wait_for_server 8100
  wait_for_server 8200
  wait_for_server 8300
  python3 disagg_epd_route.py &
  sleep 1
}


benchmark() {
  results_folder="./results"
  model="Qwen/Qwen2.5-VL-3B-Instruct"
  dataset_name="hf"
  dataset_path="lmarena-ai/VisionArena-Chat"
  num_prompts=100
  qps=$1
  tag=$2

  vllm bench serve \
    --backend vllm \
    --model $model \
    --dataset-name $dataset_name \
    --dataset-path $dataset_path \
    --hf-split train \
    --endpoint-type openai-chat \
    --endpoint /v1/chat/completions \
    --num-prompts $num_prompts \
    --port 8000 \
    --save-result \
    --result-dir $results_folder \
    --result-filename "$tag"-qps-"$qps".json \
    --request-rate "$qps"

  sleep 2
}


main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)
  (which lsof) || (apt-get -y install lsof)

  pip install quart httpx matplotlib aiohttp datasets

  cd "$(dirname "$0")"

  rm -rf results
  mkdir results

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  launch_e_pd_benchmark
  for qps in 2; do
  benchmark $qps chunked_prefill
  done
  kill_gpu_processes

  launch_e_p_d_benchmark
  for qps in 2; do
  benchmark $qps disagg_prefill
  done
  kill_gpu_processes

  python3 visualize_benchmark_results.py
}


main "$@"
