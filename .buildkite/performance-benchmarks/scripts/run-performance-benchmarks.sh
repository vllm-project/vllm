#!/bin/bash

# This script should be run inside the CI process
# This script assumes that we are already inside the vllm/ directory
# Benchmarking results will be available inside vllm/benchmarks/results/

# Do not set -e, as the mixtral 8x22B model tends to crash occasionally
# and we still want to see other benchmarking results even when mixtral crashes.
set -x
set -o pipefail

check_gpus() {
  if command -v nvidia-smi; then
    # check the number of GPUs and GPU type.
    declare -g gpu_count=$(nvidia-smi --list-gpus | wc -l)
  elif command -v amd-smi; then
    declare -g gpu_count=$(amd-smi list | grep 'GPU' | wc -l)
  elif command -v hl-smi; then
    declare -g gpu_count=$(hl-smi --list | grep -i "Module ID" | wc -l)
  fi

  if [[ $gpu_count -gt 0 ]]; then
    echo "GPU found."
  else
    echo "Need at least 1 GPU to run benchmarking."
    exit 1
  fi
  
  declare -g arch_suffix=''
  
  if command -v nvidia-smi; then
    declare -g gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '{print $2}')
  elif command -v amd-smi; then
    declare -g gpu_type=$(amd-smi static -g 0 -a | grep 'MARKET_NAME' | awk '{print $2}')
  elif command -v hl-smi; then
    declare -g gpu_type=$(hl-smi -q | grep "Product Name" | head -n 1 | awk -F ':' '{print $2}' | sed 's/^ *//')
    arch_suffix='-hpu'
  fi
  echo "GPU type is $gpu_type"
}

check_cpus() {
  # check the number of CPUs and NUMA Node and GPU type.
  declare -g numa_count=$(lscpu | grep "NUMA node(s):" | awk '{print $3}')
  if [[ $numa_count -gt 0 ]]; then
    echo "NUMA found."
    echo $numa_count
  else
    echo "Need at least 1 NUMA to run benchmarking."
    exit 1
  fi
  if [[ "$(uname -m)" == "aarch64" ]] || [[ "$(uname -m)" == "arm64" ]]; then
    declare -g gpu_type="arm64-cpu"
  else
    declare -g gpu_type="cpu"
  fi
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

ensure_sharegpt_downloaded() {
  local FILE=ShareGPT_V3_unfiltered_cleaned_split.json
  if [ ! -f "$FILE" ]; then
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/$FILE
  else
    echo "$FILE already exists."
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

json2envs() {
  # transforms the JSON string to environment variables.
  # example:
  # input: { "VLLM_CPU_KVCACHE_SPACE": 5 }
  # output: VLLM_CPU_KVCACHE_SPACE=5
  local json_string=$1
  local args=$(
    echo "$json_string" | jq -r '
      to_entries |
      map((.key ) + "=" + (.value | tostring)) |
      join(" ")
    '
  )
  echo "$args"
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local timeout_val="1200"
  timeout "$timeout_val" bash -c '
    until curl -X POST localhost:8000/v1/completions; do
      sleep 1
    done' && return 0 || return 1
}

kill_processes_launched_by_current_bash() {
  # Kill all python processes launched from current bash script
  current_shell_pid=$$
  processes=$(ps -eo pid,ppid,command | awk -v ppid="$current_shell_pid" -v proc="$1" '$2 == ppid && $3 ~ proc {print $1}')
  if [ -n "$processes" ]; then
    echo "Killing the following processes matching '$1':"
    echo "$processes"
    echo "$processes" | xargs kill -9
  else
    echo "No processes found matching '$1'."
  fi
}

kill_gpu_processes() {

  ps -aux
  lsof -t -i:8000 | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  # vLLM now names the process with VLLM prefix after https://github.com/vllm-project/vllm/pull/21445
  pgrep VLLM | xargs -r kill -9

  # wait until GPU memory usage smaller than 1GB
  if command -v nvidia-smi; then
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -ge 1000 ]; do
      sleep 1
    done
  elif command -v amd-smi; then
    while [ "$(amd-smi metric -g 0 | grep 'USED_VRAM' | awk '{print $2}')" -ge 1000 ]; do
      sleep 1
    done
  elif command -v hl-smi; then
    while [ "$(hl-smi -q | grep "Used" | head -n 1 | awk '{print $3}')" -ge 1000 ]; do
      sleep 1
    done
  fi

  # remove vllm config file
  rm -rf ~/.config/vllm

}

upload_to_buildkite() {
  # upload the benchmarking results to buildkite

  # if the agent binary is not found, skip uploading the results, exit 0
  # Check if buildkite-agent is available in the PATH or at /workspace/buildkite-agent
  if command -v buildkite-agent >/dev/null 2>&1; then
    BUILDKITE_AGENT_COMMAND="buildkite-agent"
  elif [ -f /workspace/buildkite-agent ]; then
    BUILDKITE_AGENT_COMMAND="/workspace/buildkite-agent"
  else
    echo "buildkite-agent binary not found. Skip uploading the results."
    return 0
  fi

  # Use the determined command to annotate and upload artifacts
  $BUILDKITE_AGENT_COMMAND annotate --style "info" --context "$BUILDKITE_LABEL-benchmark-results" < "$RESULTS_FOLDER/benchmark_results.md"
  $BUILDKITE_AGENT_COMMAND artifact upload "$RESULTS_FOLDER/*"
}

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
    if [[ "$ON_CPU" == "1" ]]; then
      pp=$(echo "$latency_params" | jq -r '.pipeline_parallel_size // 1')
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
    if [[ "$ON_CPU" == "1" ]]; then
      pp=$(echo "$throughput_params" | jq -r '.pipeline_parallel_size // 1')
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
  # Supported JSON formats:
  # 1) Plain format: top-level array
  #    [ { "test_name": "...", "server_parameters": {...}, ... }, ... ]
  #
  # 2) Default parameters field + plain format tests
  #    {
  #      "defaults": { ... },
  #      "tests": [ { "test_name": "...", "server_parameters": {...}, ... }, ... ]
  #    }

  local serving_test_file
  serving_test_file=$1

  # Iterate over serving tests
  jq -c '
    if type == "array" then
      # Plain format: test cases array
      .[]
    elif (type == "object" and has("tests")) then
      # merge the default parameters into each test cases
      . as $root
      | ($root.defaults // {}) as $d
      | ($root.tests // [])[]
      # default qps / max_concurrency from defaults if missing
      | .qps_list = (.qps_list // $d.qps_list)
      | .max_concurrency_list = (.max_concurrency_list // $d.max_concurrency_list)
      # merge envs / params: test overrides defaults
      | .server_environment_variables =
          (($d.server_environment_variables // {}) + (.server_environment_variables // {}))
      | .server_parameters =
          (($d.server_parameters // {}) + (.server_parameters // {}))
      | .client_parameters =
          (($d.client_parameters // {}) + (.client_parameters // {}))
    else
      error("Unsupported serving test file format: must be array or object with .tests")
    end
  ' "$serving_test_file" | while read -r params; do
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

    # get client and server arguments (after merged the default parameters)
    server_params=$(echo "$params" | jq -r '.server_parameters')
    server_envs=$(echo "$params" | jq -r '.server_environment_variables')
    client_params=$(echo "$params" | jq -r '.client_parameters')

    server_args=$(json2args "$server_params")
    server_envs=$(json2envs "$server_envs")
    client_args=$(json2args "$client_params")

    # qps_list
    qps_list=$(echo "$params" | jq -r '.qps_list')
    qps_list=$(echo "$qps_list" | jq -r '.[] | @sh')
    echo "Running over qps list $qps_list"

    # max_concurrency_list (fallback to num_prompts if missing)
    max_concurrency_list=$(echo "$params" | jq -r '.max_concurrency_list')
    if [[ -z "$max_concurrency_list" || "$max_concurrency_list" == "null" ]]; then
      num_prompts=$(echo "$client_params" | jq -r '.num_prompts')
      max_concurrency_list="[$num_prompts]"
    fi
    max_concurrency_list=$(echo "$max_concurrency_list" | jq -r '.[] | @sh')
    echo "Running over max concurrency list $max_concurrency_list"

    # check if there is enough resources to run the test
    tp=$(echo "$server_params" | jq -r '.tensor_parallel_size')
    if [[ "$ON_CPU" == "1" ]]; then
      pp=$(echo "$server_params" | jq -r '.pipeline_parallel_size // 1')
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
      $server_args"

    # run the server
    echo "Running test case $test_name"
    echo "Server command: $server_command"
    # support remote vllm server
    client_remote_args=""
    if [[ -z "${REMOTE_HOST}" ]]; then
      bash -c "$server_command" &
      server_pid=$!
      # wait until the server is alive
      if wait_for_server; then
        echo ""
        echo "vLLM server is up and running."
      else
        echo ""
        echo "vLLM failed to start within the timeout period."
      fi
    else
      server_command="Using Remote Server $REMOTE_HOST $REMOTE_PORT"
      if [[ ${REMOTE_PORT} ]]; then
        client_remote_args=" --host=$REMOTE_HOST --port=$REMOTE_PORT "
      else
        client_remote_args=" --host=$REMOTE_HOST "
      fi
    fi

    # save the compilation mode and optimization level on the serving results
    # whenever they are set
    compilation_mode=$(echo "$server_params" | jq -r '."compilation_config.mode"')
    optimization_level=$(echo "$server_params" | jq -r '.optimization_level')

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
        # pass the tensor parallel size, the compilation mode, and the optimization
        # level to the client so that they can be used on the benchmark dashboard
        client_command="vllm bench serve \
          --save-result \
          --result-dir $RESULTS_FOLDER \
          --result-filename ${new_test_name}.json \
          --request-rate $qps \
          --max-concurrency $max_concurrency \
          --metadata "tensor_parallel_size=$tp" \
          --metadata "compilation_mode=$compilation_mode"
          --metadata "optimization_level=$optimization_level" \
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

    # clean up
    kill -9 $server_pid
    kill_gpu_processes
  done
}

main() {
  local ARCH
  ARCH=''
  if [[ "$ON_CPU" == "1" ]]; then
    check_cpus
    ARCH="-$gpu_type"
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
  ensure_sharegpt_downloaded
  declare -g RESULTS_FOLDER=results/
  mkdir -p $RESULTS_FOLDER
  QUICK_BENCHMARK_ROOT=../.buildkite/performance-benchmarks/

  # dump vllm info via vllm collect-env
  env_output=$(vllm collect-env)

  echo "$env_output" >"$RESULTS_FOLDER/vllm_env.txt"

  # benchmarking
  run_serving_tests $QUICK_BENCHMARK_ROOT/tests/"${SERVING_JSON:-serving-tests$ARCH.json}"
  run_latency_tests $QUICK_BENCHMARK_ROOT/tests/"${LATENCY_JSON:-latency-tests$ARCH.json}"
  run_throughput_tests $QUICK_BENCHMARK_ROOT/tests/"${THROUGHPUT_JSON:-throughput-tests$ARCH.json}"

  # postprocess benchmarking results
  pip install tabulate pandas
  python3 $QUICK_BENCHMARK_ROOT/scripts/convert-results-json-to-markdown.py

  upload_to_buildkite
}

main "$@"
