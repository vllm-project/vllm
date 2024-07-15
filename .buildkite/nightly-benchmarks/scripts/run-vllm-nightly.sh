#!/bin/bash

set -o pipefail

check_gpus() {
  # check the number of GPUs and GPU type.
  declare -g gpu_count=$(nvidia-smi --list-gpus | wc -l)
  if [[ $gpu_count -gt 0 ]]; then
    echo "GPU found."
  else
    echo "Need at least 1 GPU to run benchmarking."
    exit 1
  fi
  declare -g gpu_type=$(echo $(nvidia-smi --query-gpu=name --format=csv,noheader) | awk '{print $2}')
  echo "GPU type is $gpu_type"
}

kill_gpu_processes() {
  # kill all processes on GPU.
  pkill pt_main_thread
  sleep 10

  # remove vllm config file
  rm -rf ~/.config/vllm

  # Print the GPU memory usage
  # so that we know if all GPU processes are killed.
  gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  # The memory usage should be 0 MB.
  echo "GPU 0 Memory Usage: $gpu_memory_usage MB"
}

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

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  timeout 1200 bash -c '
    until curl -s localhost:8000/v1/completions > /dev/null; do
      sleep 1
    done' && return 0 || return 1
}

run_serving_tests() {
  # run serving tests using `benchmark_serving.py`
  # $1: a json file specifying serving test cases

  local serving_test_file
  serving_test_file=$1

  # Iterate over serving tests
  jq -c '.[]' "$serving_test_file" | while read -r params; do
    # get the test name, and append the GPU type back to it.
    test_name=$(echo "$params" | jq -r '.test_name')

    # if TEST_SELECTOR is set, only run the test cases that match the selector
    if [[ -n "$TEST_SELECTOR" ]] && [[ ! "$test_name" =~ $TEST_SELECTOR ]]; then
      echo "Skip test case $test_name."
      continue
    fi

    # append vllm to the test name
    test_name=vllm_$test_name


    # get common parameters
    common_params=$(echo "$params" | jq -r '.common_parameters')
    model=$(echo "$common_params" | jq -r '.model')
    tp=$(echo "$common_params" | jq -r '.tp')
    dataset_name=$(echo "$common_params" | jq -r '.dataset_name')
    dataset_path=$(echo "$common_params" | jq -r '.dataset_path')
    port=$(echo "$common_params" | jq -r '.port')
    num_prompts=$(echo "$common_params" | jq -r '.num_prompts')

    # get client and server arguments
    server_params=$(echo "$params" | jq -r '.vllm_server_parameters')
    client_params=$(echo "$params" | jq -r '.vllm_client_parameters')
    server_args=$(json2args "$server_params")
    client_args=$(json2args "$client_params")
    qps_list=$(echo "$params" | jq -r '.qps_list')
    qps_list=$(echo "$qps_list" | jq -r '.[] | @sh')
    echo "Running over qps list $qps_list"

    # check if there is enough GPU to run the test
    if [[ $gpu_count -lt $tp ]]; then
      echo "Required tensor-parallel-size $tp but only $gpu_count GPU found. Skip testcase $test_name."
      continue
    fi

    if echo "$common_params" | jq -e 'has("fp8")' > /dev/null; then
      echo "Key 'fp8' exists in common params. Use neuralmagic fp8 model for convenience."
      model=$(echo "$common_params" | jq -r '.neuralmagic_quantized_model')
      server_command="python3 \
        -m vllm.entrypoints.openai.api_server \
        -tp $tp \
        --model $model \
        --port $port \
        $server_args"
    else
      echo "Key 'fp8' does not exist in common params."
      server_command="python3 \
        -m vllm.entrypoints.openai.api_server \
        -tp $tp \
        --model $model \
        --port $port \
        $server_args"
    fi

    # run the server
    echo "Running test case $test_name"
    echo "Server command: $server_command"
    eval "$server_command" &

    # wait until the server is alive
    wait_for_server
    if [ $? -eq 0 ]; then
      echo ""
      echo "vllm server is up and running."
    else
      echo ""
      echo "vllm failed to start within the timeout period."
      break
    fi

    # iterate over different QPS
    for qps in $qps_list; do
      # remove the surrounding single quote from qps
      if [[ "$qps" == *"inf"* ]]; then
        echo "qps was $qps"
        qps="inf"
        echo "now qps is $qps"
      fi

      new_test_name=$test_name"_qps_"$qps

      client_command="python3 benchmark_serving.py \
        --backend vllm \
        --model $model \
        --dataset-name $dataset_name \
        --dataset-path $dataset_path \
        --num-prompts $num_prompts \
        --port $port \
        --save-result \
        --result-dir $RESULTS_FOLDER \
        --result-filename ${new_test_name}.json \
        --request-rate $qps \
        $client_args"

      echo "Running test case $test_name with qps $qps"
      echo "Client command: $client_command"

      eval "$client_command"

      # record the benchmarking commands
      jq_output=$(jq -n \
        --arg server "$server_command" \
        --arg client "$client_command" \
        --arg gpu "$gpu_type" \
        --arg engine "vllm" \
        '{
          server_command: $server,
          client_command: $client,
          gpu_type: $gpu,
          engine: $engine
        }')
      echo "$jq_output" >"$RESULTS_FOLDER/${new_test_name}.commands"

    done

    # clean up
    kill_gpu_processes
    rm -rf /root/.cache/huggingface/*
  done
}


upload_to_buildkite() {
  # upload the benchmarking results to buildkite

  # if the agent binary is not found, skip uploading the results, exit 0
  if [ ! -f /workspace/buildkite-agent ]; then
    echo "buildkite-agent binary not found. Skip uploading the results."
    return 0
  fi
  # /workspace/buildkite-agent annotate --style "success" --context "benchmark-results" --append < $RESULTS_FOLDER/${CURRENT_LLM_SERVING_ENGINE}_nightly_results.md
  /workspace/buildkite-agent artifact upload "$RESULTS_FOLDER/*"
}

main() {

  check_gpus
  # enter vllm directory
  cd $VLLM_SOURCE_CODE_LOC/benchmarks
  declare -g RESULTS_FOLDER=results/
  mkdir -p $RESULTS_FOLDER
  BENCHMARK_ROOT=../.buildkite/nightly-benchmarks/

  export CURRENT_LLM_SERVING_ENGINE=vllm
  run_serving_tests $BENCHMARK_ROOT/tests/nightly-tests.json

  python3 -m pip install tabulate pandas
  python3 $BENCHMARK_ROOT/scripts/summary-nightly-results.py
  upload_to_buildkite

}

main "$@"
