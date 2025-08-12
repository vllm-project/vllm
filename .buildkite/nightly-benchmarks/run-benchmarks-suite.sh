#!/bin/bash

# This script should be run inside the CI process
# This script assumes that we are already inside the vllm/ directory
# Benchmarking results will be available inside vllm/benchmarks/results/

# Do not set -e, as the mixtral 8x22B model tends to crash occasionally
# and we still want to see other benchmarking results even when mixtral crashes.
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

check_hf_token() {
  # check if HF_TOKEN is available and valid
  if [[ -z "$HF_TOKEN" ]]; then
    echo "Error: HF_TOKEN is not set."
    exit 1
  elif [[ ! "$HF_TOKEN" =~ ^hf_ ]]; then
    echo "Error: HF_TOKEN does not start with 'hf_'."
    exit 1
  else
    echo "HF_TOKEN is set and valid."
  fi
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
    until curl localhost:8000/v1/completions; do
      sleep 1
    done' && return 0 || return 1
}

kill_gpu_processes() {
  # kill all processes on GPU.
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  if [ -z "$pids" ]; then
      echo "No GPU processes found."
  else
      for pid in $pids; do
          kill -9 "$pid"
          echo "Killed process with PID: $pid"
      done

      echo "All GPU processes have been killed."
  fi

  # waiting for GPU processes to be fully killed
  sleep 10

  # remove vllm config file
  rm -rf ~/.config/vllm

  # Print the GPU memory usage
  # so that we know if all GPU processes are killed.
  gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  # The memory usage should be 0 MB.
  echo "GPU 0 Memory Usage: $gpu_memory_usage MB"
}

upload_to_buildkite() {
  # upload the benchmarking results to buildkite

  # if the agent binary is not found, skip uploading the results, exit 0
  if [ ! -f /workspace/buildkite-agent ]; then
    echo "buildkite-agent binary not found. Skip uploading the results."
    return 0
  fi
  /workspace/buildkite-agent annotate --style "info" --context "benchmark-results" < $RESULTS_FOLDER/benchmark_results.md
  /workspace/buildkite-agent artifact upload "$RESULTS_FOLDER/*"
}

run_latency_tests() {
  # run latency tests using `benchmark_latency.py`
  # $1: a json file specifying latency test cases

  local latency_test_file
  latency_test_file=$1

  # Iterate over latency tests
  jq -c '.[]' "$latency_test_file" | while read -r params; do
    # get the test name, and append the GPU type back to it.
    test_name=$(echo "$params" | jq -r '.test_name')
    if [[ ! "$test_name" =~ ^latency_ ]]; then
      echo "In latency-test.json, test_name must start with \"latency_\"."
      exit 1
    fi

    # if TEST_SELECTOR is set, only run the test cases that match the selector
    if [[ -n "$TEST_SELECTOR" ]] && [[ ! "$test_name" =~ $TEST_SELECTOR ]]; then
      echo "Skip test case $test_name."
      continue
    fi

    # get arguments
    latency_params=$(echo "$params" | jq -r '.parameters')
    latency_args=$(json2args "$latency_params")

    # check if there is enough GPU to run the test
    tp=$(echo "$latency_params" | jq -r '.tensor_parallel_size')
    if [[ $gpu_count -lt $tp ]]; then
      echo "Required tensor-parallel-size $tp but only $gpu_count GPU found. Skip testcase $testname."
      continue
    fi

    latency_command="python3 benchmark_latency.py \
      --output-json $RESULTS_FOLDER/${test_name}.json \
      $latency_args"

    echo "Running test case $test_name"
    echo "Latency command: $latency_command"

    # recoding benchmarking command ang GPU command
    jq_output=$(jq -n \
      --arg latency "$latency_command" \
      --arg gpu "$gpu_type" \
      '{
        latency_command: $latency,
        gpu_type: $gpu
      }')
    echo "$jq_output" > "$RESULTS_FOLDER/$test_name.commands"

    # run the benchmark
    eval "$latency_command"

    kill_gpu_processes

  done
}


run_throughput_tests() {
  # run throughput tests using `benchmark_throughput.py`
  # $1: a json file specifying throughput test cases

  local throughput_test_file
  throughput_test_file=$1

  # Iterate over throughput tests
  jq -c '.[]' "$throughput_test_file" | while read -r params; do
    # get the test name, and append the GPU type back to it.
    test_name=$(echo "$params" | jq -r '.test_name')
    if [[ ! "$test_name" =~ ^throughput_ ]]; then
      echo "In throughput-test.json, test_name must start with \"throughput_\"."
      exit 1
    fi

    # if TEST_SELECTOR is set, only run the test cases that match the selector
    if [[ -n "$TEST_SELECTOR" ]] && [[ ! "$test_name" =~ $TEST_SELECTOR ]]; then
      echo "Skip test case $test_name."
      continue
    fi

    # get arguments
    throughput_params=$(echo "$params" | jq -r '.parameters')
    throughput_args=$(json2args "$throughput_params")

    # check if there is enough GPU to run the test
    tp=$(echo $throughput_params | jq -r '.tensor_parallel_size')
    if [[ $gpu_count -lt $tp ]]; then
      echo "Required tensor-parallel-size $tp but only $gpu_count GPU found. Skip testcase $testname."
      continue
    fi

    throughput_command="python3 benchmark_throughput.py \
      --output-json $RESULTS_FOLDER/${test_name}.json \
      $throughput_args"

    echo "Running test case $test_name"
    echo "Throughput command: $throughput_command"
    # recoding benchmarking command ang GPU command
    jq_output=$(jq -n \
      --arg command "$throughput_command" \
      --arg gpu "$gpu_type" \
      '{
        throughput_command: $command,
        gpu_type: $gpu
      }')
    echo "$jq_output" > "$RESULTS_FOLDER/$test_name.commands"

    # run the benchmark
    eval "$throughput_command"

    kill_gpu_processes

  done
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
    if [[ ! "$test_name" =~ ^serving_ ]]; then
      echo "In serving-test.json, test_name must start with \"serving_\"."
      exit 1
    fi

    # if TEST_SELECTOR is set, only run the test cases that match the selector
    if [[ -n "$TEST_SELECTOR" ]] && [[ ! "$test_name" =~ $TEST_SELECTOR ]]; then
      echo "Skip test case $test_name."
      continue
    fi


    # get client and server arguments
    server_params=$(echo "$params" | jq -r '.server_parameters')
    client_params=$(echo "$params" | jq -r '.client_parameters')
    server_args=$(json2args "$server_params")
    client_args=$(json2args "$client_params")
    qps_list=$(echo "$params" | jq -r '.qps_list')
    qps_list=$(echo "$qps_list" | jq -r '.[] | @sh')
    echo "Running over qps list $qps_list"

    # check if there is enough GPU to run the test
    tp=$(echo "$server_params" | jq -r '.tensor_parallel_size')
    if [[ $gpu_count -lt $tp ]]; then
      echo "Required tensor-parallel-size $tp but only $gpu_count GPU found. Skip testcase $testname."
      continue
    fi

    # check if server model and client model is aligned
    server_model=$(echo "$server_params" | jq -r '.model')
    client_model=$(echo "$client_params" | jq -r '.model')
    if [[ $server_model != "$client_model" ]]; then
      echo "Server model and client model must be the same. Skip testcase $testname."
      continue
    fi

    server_command="python3 \
      -m vllm.entrypoints.openai.api_server \
      $server_args"

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
        '{
          server_command: $server,
          client_command: $client,
          gpu_type: $gpu
        }')
      echo "$jq_output" > "$RESULTS_FOLDER/${new_test_name}.commands"

    done

    # clean up
    kill_gpu_processes
  done
}

main() {
  check_gpus
  check_hf_token

  # dependencies
  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get update && apt-get -y install jq)

  # get the current IP address, required by benchmark_serving.py
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  # turn of the reporting of the status of each request, to clean up the terminal output
  export VLLM_LOG_LEVEL="WARNING"

  # prepare for benchmarking
  cd benchmarks || exit 1
  wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
  declare -g RESULTS_FOLDER=results/
  mkdir -p $RESULTS_FOLDER
  QUICK_BENCHMARK_ROOT=../.buildkite/nightly-benchmarks/

  # benchmarking
  run_serving_tests $QUICK_BENCHMARK_ROOT/tests/serving-tests.json
  run_latency_tests $QUICK_BENCHMARK_ROOT/tests/latency-tests.json
  run_throughput_tests $QUICK_BENCHMARK_ROOT/tests/throughput-tests.json


  # postprocess benchmarking results
  pip install tabulate pandas
  python3 $QUICK_BENCHMARK_ROOT/scripts/convert-results-json-to-markdown.py

  upload_to_buildkite
}

main "$@"
