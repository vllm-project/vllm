#!/bin/bash

# benchmark the overhead of disaggregated prefill.
# methodology:
# - send all request to prefill vLLM instance. It will buffer KV cache.
# - then send all request to decode instance.
# - The TTFT of decode instance is the overhead.

set -ex

kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  # vLLM now names the process with VLLM prefix after https://github.com/vllm-project/vllm/pull/21445
  pgrep VLLM | xargs -r kill -9
  sleep 10

  # remove vllm config file
  rm -rf ~/.config/vllm

  # Print the GPU memory usage
  # so that we know if all GPU processes are killed.
  gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  # The memory usage should be 0 MB.
  echo "GPU 0 Memory Usage: $gpu_memory_usage MB"
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


benchmark() {

  export VLLM_LOGGING_LEVEL=DEBUG
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  # compare chunked prefill with disaggregated prefill

  results_folder="./results"
  model="meta-llama/Meta-Llama-3.1-8B-Instruct"
  dataset_name="sonnet"
  dataset_path="../sonnet_4x.txt"
  num_prompts=10
  qps=$1
  prefix_len=50
  input_len=2048
  output_len=$2


  CUDA_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9}' &


  CUDA_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  wait_for_server 8100
  wait_for_server 8200

  # let the prefill instance finish prefill
  vllm bench serve \
    --backend vllm \
    --model $model \
    --dataset-name $dataset_name \
    --dataset-path $dataset_path \
    --sonnet-input-len $input_len \
    --sonnet-output-len "$output_len" \
    --sonnet-prefix-len $prefix_len \
    --num-prompts $num_prompts \
    --port 8100 \
    --save-result \
    --result-dir $results_folder \
    --result-filename disagg_prefill_tp1.json \
    --request-rate "inf"


  # send the request to decode.
  # The TTFT of this command will be the overhead of disagg prefill impl.
  vllm bench serve \
    --backend vllm \
    --model $model \
    --dataset-name $dataset_name \
    --dataset-path $dataset_path \
    --sonnet-input-len $input_len \
    --sonnet-output-len "$output_len" \
    --sonnet-prefix-len $prefix_len \
    --num-prompts $num_prompts \
    --port 8200 \
    --save-result \
    --result-dir $results_folder \
    --result-filename disagg_prefill_tp1_overhead.json \
    --request-rate "$qps"
  kill_gpu_processes

}


main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)

  pip install quart httpx datasets

  cd "$(dirname "$0")"

  cd ..
  # create sonnet-4x.txt
  echo "" > sonnet_4x.txt
  for _ in {1..4}
  do
    cat sonnet.txt >> sonnet_4x.txt
  done
  cd disagg_benchmarks

  rm -rf results
  mkdir results

  default_qps=1
  default_output_len=1
  benchmark $default_qps $default_output_len

}


main "$@"
