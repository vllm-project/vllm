#!/bin/bash

# This script should be run inside the CI process
# This script assumes that we are already inside the vllm/ directory
# Benchmarking results will be available inside vllm/benchmarks/results/

# Do not set -e, as the mixtral 8x22B model tends to crash occasionally
# and we still want to see other benchmarking results even when mixtral crashes.
set -x
set -o pipefail

source tests/cohere/scripts/run-helper.sh

run_latency_tests() {
  # run latency tests using `vllm bench latency` command
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
    latency_environment_variables=$(echo "$params" | jq -r '.environment_variables')
    latency_envs=$(json2envs "$latency_environment_variables")

    # check if there is enough GPU to run the test
    tp=$(echo "$latency_params" | jq -r '.tensor_parallel_size')
    if [ "$ON_CPU" == "1" ]; then
      pp=$(echo "$latency_params" | jq -r '.pipeline_parallel_size')
      world_size=$(($tp*$pp))
      if [[ $numa_count -lt $world_size  && -z "${REMOTE_HOST}" ]]; then
        echo "Required world-size $world_size but only $numa_count NUMA nodes found. Skip testcase $test_name."
        continue
      fi
    else
      if [[ $gpu_count -lt $tp ]]; then
        echo "Required tensor-parallel-size $tp but only $gpu_count GPU found. Skip testcase $test_name."
        continue
      fi
    fi

    latency_command=" $latency_envs vllm bench latency \
      ${VLLM_HARDWARE_PROFILE_ARGS:-} \
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
    echo "$jq_output" >"$RESULTS_FOLDER/$test_name.commands"

    # run the benchmark
    eval "$latency_command"

    kill_gpu_processes

  done
}

run_throughput_tests() {
  # run throughput tests using `vllm bench throughput`
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
    throughput_environment_variables=$(echo "$params" | jq -r '.environment_variables')
    throughput_envs=$(json2envs "$throughput_environment_variables")

    # check if there is enough GPU to run the test
    tp=$(echo "$throughput_params" | jq -r '.tensor_parallel_size')
    if [ "$ON_CPU" == "1" ]; then
      pp=$(echo "$throughput_params" | jq -r '.pipeline_parallel_size')
      world_size=$(($tp*$pp))
      if [[ $numa_count -lt $world_size  && -z "${REMOTE_HOST}" ]]; then
        echo "Required world-size $world_size but only $numa_count NUMA nodes found. Skip testcase $test_name."
        continue
      fi
    else
      if [[ $gpu_count -lt $tp ]]; then
        echo "Required tensor-parallel-size $tp but only $gpu_count GPU found. Skip testcase $test_name."
        continue
      fi
    fi

    throughput_command=" $throughput_envs vllm bench throughput \
      ${VLLM_HARDWARE_PROFILE_ARGS:-} \
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
    echo "$jq_output" >"$RESULTS_FOLDER/$test_name.commands"

    # run the benchmark
    eval "$throughput_command"

    kill_gpu_processes

  done
}

run_serving_tests() {
  # run serving tests using `vllm bench serve` command
  # $1: a json file specifying serving test cases
  #
  # The server is only restarted when server_command changes between test
  # cases, avoiding redundant load cycles when multiple benchmarks share
  # the same server configuration.

  local serving_test_file
  serving_test_file=$1

  local current_server_command=""
  local server_pid=""

  # Process substitution (< <(...)) keeps the loop in the current shell so
  # that server_pid and current_server_command persist across iterations and
  # remain accessible after the loop for final cleanup.
  while read -r params; do
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
    server_envs=$(echo "$params" | jq -r '.server_environment_variables')
    client_params=$(echo "$params" | jq -r '.client_parameters')
    server_args=$(json2args "$server_params")
    server_envs=$(json2envs "$server_envs")
    client_args=$(json2args "$client_params")
    qps_list=$(echo "$params" | jq -r '.qps_list')
    qps_list=$(echo "$qps_list" | jq -r '.[] | @sh')
    echo "Running over qps list $qps_list"
    max_concurrency_list=$(echo "$params" | jq -r '.max_concurrency_list')
    if [[ -z "$max_concurrency_list" || "$max_concurrency_list" == "null" ]]; then
        num_prompts=$(echo "$client_params" | jq -r '.num_prompts')
        max_concurrency_list="[$num_prompts]"
    fi
    max_concurrency_list=$(echo "$max_concurrency_list" | jq -r '.[] | @sh')
    echo "Running over max concurrency list $max_concurrency_list"

    # check if there is enough resources to run the test
    tp=$(echo "$server_params" | jq -r '.tensor_parallel_size')
    if [ "$ON_CPU" == "1" ]; then
      pp=$(echo "$server_params" | jq -r '.pipeline_parallel_size')
      world_size=$(($tp*$pp))
      if [[ $numa_count -lt $world_size  && -z "${REMOTE_HOST}" ]]; then
        echo "Required world-size $world_size but only $numa_count NUMA nodes found. Skip testcase $test_name."
        continue
      fi
    else
      if [[ $gpu_count -lt $tp ]]; then
        echo "Required tensor-parallel-size $tp but only $gpu_count GPU found. Skip testcase $test_name."
        continue
      fi
    fi

    # check if server model and client model is aligned
    server_model=$(echo "$server_params" | jq -r '.model')
    client_model=$(echo "$client_params" | jq -r '.model')
    if [[ $server_model != "$client_model" ]]; then
      echo "Server model and client model must be the same. Skip testcase $test_name."
      continue
    fi

    server_command="$server_envs vllm serve \
      ${VLLM_HARDWARE_PROFILE_ARGS:-} \
      $server_args"

    echo "Running test case $test_name"
    echo "Server command: $server_command"

    # support remote vllm server
    client_remote_args=""
    if [[ -z "${REMOTE_HOST}" ]]; then
      if [[ "$server_command" != "$current_server_command" ]]; then
        # Server config changed (or first iteration) -- (re)start the server.
        if [[ -n "$server_pid" ]]; then
          echo "Server config changed, stopping previous server (pid $server_pid)..."
          kill -9 $server_pid
          kill_gpu_processes
        fi

        # Use VLLM_LOGGING_LEVEL=INFO for the server process so KV cache
        # info lines are captured, while keeping WARNING for client benchmarks.
        # tee sends output to both terminal (visible in CI) and a log file.
        server_config_key="$(basename "$server_model")_tp${tp}"
        server_log="$RESULTS_FOLDER/${server_config_key}_server.log"
        VLLM_LOGGING_LEVEL=INFO bash -c "$server_command" 2>&1 | tee "$server_log" &
        server_pid=$!
        if wait_for_server; then
          echo ""
          echo "vLLM server is up and running."
        else
          echo ""
          echo "vLLM failed to start within the timeout period."
        fi

        # Extract KV cache info from server startup logs
        kv_mem_gib=$(grep -oP 'Available KV cache memory: \K[\d.]+' "$server_log" | head -1)
        kv_tokens=$(grep -oP 'GPU KV cache size: \K[\d,]+' "$server_log" | head -1 | tr -d ',')
        if [[ -n "$kv_mem_gib" || -n "$kv_tokens" ]]; then
          jq -n \
            --arg mem "${kv_mem_gib:-null}" \
            --arg tok "${kv_tokens:-null}" \
            '{kv_cache_memory_gib: ($mem | if . == "null" then null else tonumber end),
              kv_cache_tokens: ($tok | if . == "null" then null else tonumber end)}' \
            > "$RESULTS_FOLDER/${server_config_key}_server_info.json"
          echo "Captured KV cache info for ${server_config_key}: memory=${kv_mem_gib:-?} GiB, tokens=${kv_tokens:-?}"
        else
          echo "Warning: could not extract KV cache info from server logs for ${server_config_key}"
        fi
        current_server_command="$server_command"
      else
        echo "Reusing running server (pid $server_pid) for $test_name"
      fi
    else
      server_command="Using Remote Server $REMOTE_HOST $REMOTE_PORT"
      if [[ ${REMOTE_PORT} ]]; then
        client_remote_args=" --host=$REMOTE_HOST --port=$REMOTE_PORT "
      else
        client_remote_args=" --host=$REMOTE_HOST "
      fi
    fi

    # iterate over different QPS
    for qps in $qps_list; do
      # remove the surrounding single quote from qps
      if [[ "$qps" == *"inf"* ]]; then
        echo "qps was $qps"
        qps="inf"
        echo "now qps is $qps"
      fi

      # iterate over different max_concurrency
      for max_concurrency in $max_concurrency_list; do
        new_test_name=$test_name"_qps_"$qps"_concurrency_"$max_concurrency
        echo " new test name $new_test_name"
        # pass the tensor parallel size to the client so that it can be displayed
        # on the benchmark dashboard
        client_command="vllm bench serve \
          --save-result \
          --result-dir $RESULTS_FOLDER \
          --result-filename ${new_test_name}.json \
          --request-rate $qps \
          --max-concurrency $max_concurrency \
          --metadata "tensor_parallel_size=$tp" \
          $client_args $client_remote_args "

        echo "Running test case $test_name with qps $qps"
        echo "Client command: $client_command"

        bash -c "$client_command"

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
        echo "$jq_output" >"$RESULTS_FOLDER/${new_test_name}.commands"

      done
    done

  done < <(jq -c '.[]' "$serving_test_file")

  # Final cleanup: stop the last server
  if [[ -n "$server_pid" ]]; then
    echo "All serving tests complete, stopping server (pid $server_pid)..."
    kill -9 $server_pid
    kill_gpu_processes
  fi
}

main() {
  local ARCH
  ARCH=''
  if [ "$ON_CPU" == "1" ];then
     check_cpus
     ARCH='-cpu'
  else
     check_gpus
     ARCH="$arch_suffix"
  fi
  check_hf_token

  # dependencies
  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get update && apt-get -y install jq)
  (which lsof) || (apt-get update && apt-get install -y lsof)

  # get the current IP address, required by `vllm bench serve` command
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  # turn of the reporting of the status of each request, to clean up the terminal output
  export VLLM_LOGGING_LEVEL="WARNING"

  # prepare for benchmarking
  cd benchmarks || exit 1
  declare -g RESULTS_FOLDER=results/
  mkdir -p $RESULTS_FOLDER
  QUICK_BENCHMARK_ROOT=../.buildkite/performance-benchmarks/

  # dump vllm info via vllm collect-env
  env_output=$(vllm collect-env)
  echo "$env_output" >"$RESULTS_FOLDER/vllm_env.txt"

  # COHERE START
  # generate configs for serving test sweep
  MODEL_NAME=$1 MODEL_PATH=$2 python3 ../tests/cohere/scripts/generate-serving-config.py

  # benchmarking
  run_serving_tests ../tests/cohere/configs/"${SERVING_JSON:-serving-cohere-tests$ARCH.json}"
  # we're not currently using the data from latency/throughput so skip them for now to save time
  # run_latency_tests $QUICK_BENCHMARK_ROOT/tests/"${LATENCY_JSON:-latency-cohere-tests$ARCH.json}"
  # run_throughput_tests $QUICK_BENCHMARK_ROOT/tests/"${THROUGHPUT_JSON:-throughput-cohere-tests$ARCH.json}"

  # postprocess benchmarking results
  pip install tabulate pandas
  python3 $QUICK_BENCHMARK_ROOT/scripts/convert-results-json-to-markdown.py

  # COHERE END
}

main "$@"
