#!/bin/bash

# Currently FP8 benchmark is NOT enabled.

set -x
server_params=$1
common_params=$2

json2args() {
  # transforms the JSON string to command line args, and '_' is replaced to '-'
  # example:
  # input: { "model": "meta-llama/Llama-2-7b-chat-hf", "tensor_parallel_size": 1 }
  # output: --model meta-llama/Llama-2-7b-chat-hf --tensor-parallel-size 1
  local json_string=$1
  local args=$(
    echo "$json_string" | jq -r '
      to_entries |
      map("--" + (.key | gsub("_"; "-")) + " " + (.value | tostring)) |
      join(" ")
    '
  )
  echo "$args"
}

launch_trt_server() {

  model_path=$(echo "$common_params" | jq -r '.model')
  model_name="${model_path#*/}"
  model_type=$(echo "$server_params" | jq -r '.model_type')
  model_dtype=$(echo "$server_params" | jq -r '.model_dtype')
  model_tp_size=$(echo "$common_params" | jq -r '.tp')
  max_batch_size=$(echo "$server_params" | jq -r '.max_batch_size')
  max_input_len=$(echo "$server_params" | jq -r '.max_input_len')
  max_seq_len=$(echo "$server_params" | jq -r '.max_seq_len')
  max_num_tokens=$(echo "$server_params" | jq -r '.max_num_tokens')
  trt_llm_version=$(echo "$server_params" | jq -r '.trt_llm_version')

  # create model caching directory
  cd ~
  rm -rf models
  mkdir -p models
  cd models
  models_dir=$(pwd)
  trt_model_path=${models_dir}/${model_name}-trt-ckpt
  trt_engine_path=${models_dir}/${model_name}-trt-engine

  # clone tensorrt backend
  cd /
  rm -rf tensorrtllm_backend
  git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
  git lfs install
  cd tensorrtllm_backend
  git checkout "$trt_llm_version"
  git submodule update --init --recursive

  # build trtllm engine
  cd /tensorrtllm_backend
  cd "./tensorrt_llm/examples/${model_type}"
  python3 convert_checkpoint.py \
    --model_dir "${model_path}" \
    --dtype "${model_dtype}" \
    --tp_size "${model_tp_size}" \
    --output_dir "${trt_model_path}"
  trtllm-build \
    --checkpoint_dir "${trt_model_path}" \
    --use_fused_mlp \
    --reduce_fusion disable \
    --workers 8 \
    --gpt_attention_plugin "${model_dtype}" \
    --gemm_plugin "${model_dtype}" \
    --tp_size "${model_tp_size}" \
    --max_batch_size "${max_batch_size}" \
    --max_input_len "${max_input_len}" \
    --max_seq_len "${max_seq_len}" \
    --max_num_tokens "${max_num_tokens}" \
    --output_dir "${trt_engine_path}"

  # handle triton protobuf files and launch triton server
  cd /tensorrtllm_backend
  mkdir triton_model_repo
  cp -r all_models/inflight_batcher_llm/* triton_model_repo/
  cd triton_model_repo
  rm -rf ./tensorrt_llm/1/*
  cp -r "${trt_engine_path}"/* ./tensorrt_llm/1
  python3 ../tools/fill_template.py -i tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,engine_dir:/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1,decoupled_mode:true,batching_strategy:inflight_fused_batching,batch_scheduler_policy:guaranteed_no_evict,exclude_input_in_output:true,triton_max_batch_size:2048,max_queue_delay_microseconds:0,max_beam_width:1,max_queue_size:2048,enable_kv_cache_reuse:false
  python3 ../tools/fill_template.py -i preprocessing/config.pbtxt "triton_max_batch_size:2048,tokenizer_dir:$model_path,preprocessing_instance_count:5"
  python3 ../tools/fill_template.py -i postprocessing/config.pbtxt "triton_max_batch_size:2048,tokenizer_dir:$model_path,postprocessing_instance_count:5,skip_special_tokens:false"
  python3 ../tools/fill_template.py -i ensemble/config.pbtxt triton_max_batch_size:"$max_batch_size"
  python3 ../tools/fill_template.py -i tensorrt_llm_bls/config.pbtxt "triton_max_batch_size:$max_batch_size,decoupled_mode:true,accumulate_tokens:False,bls_instance_count:1"
  cd /tensorrtllm_backend
  python3 scripts/launch_triton_server.py \
    --world_size="${model_tp_size}" \
    --model_repo=/tensorrtllm_backend/triton_model_repo &

}

launch_tgi_server() {
  model=$(echo "$common_params" | jq -r '.model')
  tp=$(echo "$common_params" | jq -r '.tp')
  port=$(echo "$common_params" | jq -r '.port')
  server_args=$(json2args "$server_params")

  if echo "$common_params" | jq -e 'has("fp8")' >/dev/null; then
    echo "Key 'fp8' exists in common params."
    server_command="/tgi-entrypoint.sh \
                --model-id $model \
                --num-shard $tp \
                --port $port \
                --quantize fp8 \
                $server_args"
  else
    echo "Key 'fp8' does not exist in common params."
    server_command="/tgi-entrypoint.sh \
                --model-id $model \
                --num-shard $tp \
                --port $port \
                $server_args"
  fi

  echo "Server command: $server_command"
  eval "$server_command" &

}

launch_lmdeploy_server() {
  model=$(echo "$common_params" | jq -r '.model')
  tp=$(echo "$common_params" | jq -r '.tp')
  port=$(echo "$common_params" | jq -r '.port')
  server_args=$(json2args "$server_params")

  server_command="lmdeploy serve api_server $model \
    --tp $tp \
    --server-port $port \
    $server_args"

  # run the server
  echo "Server command: $server_command"
  bash -c "$server_command" &
}

launch_sglang_server() {

  model=$(echo "$common_params" | jq -r '.model')
  tp=$(echo "$common_params" | jq -r '.tp')
  port=$(echo "$common_params" | jq -r '.port')
  server_args=$(json2args "$server_params")

  if echo "$common_params" | jq -e 'has("fp8")' >/dev/null; then
    echo "Key 'fp8' exists in common params. Use neuralmagic fp8 model for convenience."
    model=$(echo "$common_params" | jq -r '.neuralmagic_quantized_model')
    server_command="python3 \
        -m sglang.launch_server \
        --tp $tp \
        --model-path $model \
        --port $port \
        $server_args"
  else
    echo "Key 'fp8' does not exist in common params."
    server_command="python3 \
        -m sglang.launch_server \
        --tp $tp \
        --model-path $model \
        --port $port \
        $server_args"
  fi

  # run the server
  echo "Server command: $server_command"
  eval "$server_command" &
}

launch_vllm_server() {

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  model=$(echo "$common_params" | jq -r '.model')
  tp=$(echo "$common_params" | jq -r '.tp')
  port=$(echo "$common_params" | jq -r '.port')
  server_args=$(json2args "$server_params")

  if echo "$common_params" | jq -e 'has("fp8")' >/dev/null; then
    echo "Key 'fp8' exists in common params. Use neuralmagic fp8 model for convenience."
    model=$(echo "$common_params" | jq -r '.neuralmagic_quantized_model')
    server_command="vllm serve $model \
        -tp $tp \
        --port $port \
        $server_args"
  else
    echo "Key 'fp8' does not exist in common params."
    server_command="vllm serve $model \
        -tp $tp \
        --port $port \
        $server_args"
  fi

  # run the server
  echo "Server command: $server_command"
  eval "$server_command" &
}

main() {

  if [[ "$CURRENT_LLM_SERVING_ENGINE" == "trt" ]]; then
    launch_trt_server
  fi

  if [[ "$CURRENT_LLM_SERVING_ENGINE" == "tgi" ]]; then
    launch_tgi_server
  fi

  if [[ "$CURRENT_LLM_SERVING_ENGINE" == "lmdeploy" ]]; then
    launch_lmdeploy_server
  fi

  if [[ "$CURRENT_LLM_SERVING_ENGINE" == "sglang" ]]; then
    launch_sglang_server
  fi

  if [[ "$CURRENT_LLM_SERVING_ENGINE" == *"vllm"* ]]; then
    launch_vllm_server
  fi
}

main
