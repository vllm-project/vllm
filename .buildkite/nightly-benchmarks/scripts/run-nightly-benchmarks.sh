#!/bin/bash

set -o pipefail
set -x

check_gpus() {
  # check the number of GPUs and GPU type.
  declare -g gpu_count=$(nvidia-smi --list-gpus | wc -l)
  if [[ $gpu_count -gt 0 ]]; then
    echo "GPU found."
  else
    echo "Need at least 1 GPU to run benchmarking."
    exit 1
  fi
  declare -g gpu_type="$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '{print $2}')"
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


get_current_llm_serving_engine() {

  if which lmdeploy >/dev/null; then
    echo "Container: lmdeploy"
    export CURRENT_LLM_SERVING_ENGINE=lmdeploy
    return
  fi

  if [ -e /tgi-entrypoint.sh ]; then
    echo "Container: tgi"
    export CURRENT_LLM_SERVING_ENGINE=tgi
    return
  fi

  if which trtllm-build >/dev/null; then
    echo "Container: tensorrt-llm"
    export CURRENT_LLM_SERVING_ENGINE=trt
    return
  fi

  if [ -e /sgl-workspace ]; then
    echo "Container: sglang"
    export CURRENT_LLM_SERVING_ENGINE=sglang
    return
  fi

  if [ -e /vllm-workspace ]; then
    echo "Container: vllm"
    # move to a completely irrelevant directory, to avoid import vllm from current folder
    export CURRENT_LLM_SERVING_ENGINE=vllm

    return
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

kill_gpu_processes() {
  pkill -f '[p]ython'
  pkill -f '[p]ython3'
  pkill -f '[t]ritonserver'
  pkill -f '[p]t_main_thread'
  pkill -f '[t]ext-generation'
  pkill -f '[l]mdeploy'
  # vLLM now names the process with VLLM prefix after https://github.com/vllm-project/vllm/pull/21445
  pkill -f '[V]LLM'

  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -ge 1000 ]; do
    sleep 1
  done
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  timeout 1200 bash -c '
    until curl -s localhost:8000/v1/completions > /dev/null; do
      sleep 1
    done' && return 0 || return 1
}

ensure_installed() {
  # Ensure that the given command is installed by apt-get
  local cmd=$1
  if ! which "$cmd" >/dev/null; then
    apt-get update && apt-get install -y "$cmd"
  fi
}

run_serving_tests() {
  # run serving tests using `vllm bench serve` command
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

    # prepend the current serving engine to the test name
    test_name=${CURRENT_LLM_SERVING_ENGINE}_${test_name}

    # get common parameters
    common_params=$(echo "$params" | jq -r '.common_parameters')
    model=$(echo "$common_params" | jq -r '.model')
    tp=$(echo "$common_params" | jq -r '.tp')
    dataset_name=$(echo "$common_params" | jq -r '.dataset_name')
    dataset_path=$(echo "$common_params" | jq -r '.dataset_path')
    port=$(echo "$common_params" | jq -r '.port')
    num_prompts=$(echo "$common_params" | jq -r '.num_prompts')
    reuse_server=$(echo "$common_params" | jq -r '.reuse_server')

    # get client and server arguments
    server_params=$(echo "$params" | jq -r ".${CURRENT_LLM_SERVING_ENGINE}_server_parameters")
    client_params=$(echo "$params" | jq -r ".${CURRENT_LLM_SERVING_ENGINE}_client_parameters")
    client_args=$(json2args "$client_params")
    qps_list=$(echo "$params" | jq -r '.qps_list')
    qps_list=$(echo "$qps_list" | jq -r '.[] | @sh')
    echo "Running over qps list $qps_list"

    # check if there is enough GPU to run the test
    if [[ $gpu_count -lt $tp ]]; then
      echo "Required num-shard $tp but only $gpu_count GPU found. Skip testcase $test_name."
      continue
    fi

    if [[ $reuse_server == "true" ]]; then
      echo "Reuse previous server for test case $test_name"
    else
      kill_gpu_processes
      bash "$VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/scripts/launch-server.sh" \
        "$server_params" "$common_params"
    fi

    if wait_for_server; then
      echo ""
      echo "$CURRENT_LLM_SERVING_ENGINE server is up and running."
    else
      echo ""
      echo "$CURRENT_LLM_SERVING_ENGINE failed to start within the timeout period."
      break
    fi

    # prepare tokenizer
    # this is required for lmdeploy.
    cd "$VLLM_SOURCE_CODE_LOC/benchmarks"
    rm -rf /tokenizer_cache
    mkdir /tokenizer_cache
    python3 ../.buildkite/nightly-benchmarks/scripts/download-tokenizer.py \
      --model "$model" \
      --cachedir /tokenizer_cache
    cd "$VLLM_SOURCE_CODE_LOC/benchmarks"


    # change model name for lmdeploy (it will not follow standard hf name)
    if [[ "$CURRENT_LLM_SERVING_ENGINE" == "lmdeploy" ]]; then
      model=$(python ../.buildkite/nightly-benchmarks/scripts/get-lmdeploy-modelname.py)
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

      backend=$CURRENT_LLM_SERVING_ENGINE

      if [[ $backend = "trt" ]]; then
        backend="tensorrt-llm"
      fi

      if [[ "$backend" == *"vllm"* ]]; then
        backend="vllm"
      fi

      if [[ "$dataset_name" = "sharegpt" ]]; then

        client_command="vllm bench serve \
          --backend $backend \
          --tokenizer /tokenizer_cache \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --num-prompts $num_prompts \
          --port $port \
          --save-result \
          --result-dir $RESULTS_FOLDER \
          --result-filename ${new_test_name}.json \
          --request-rate $qps \
          --ignore-eos \
          $client_args"

      elif [[ "$dataset_name" = "sonnet" ]]; then

        sonnet_input_len=$(echo "$common_params" | jq -r '.sonnet_input_len')
        sonnet_output_len=$(echo "$common_params" | jq -r '.sonnet_output_len')
        sonnet_prefix_len=$(echo "$common_params" | jq -r '.sonnet_prefix_len')

        client_command="vllm bench serve \
          --backend $backend \
          --tokenizer /tokenizer_cache \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --num-prompts $num_prompts \
          --sonnet-input-len $sonnet_input_len \
          --sonnet-output-len $sonnet_output_len \
          --sonnet-prefix-len $sonnet_prefix_len \
          --port $port \
          --save-result \
          --result-dir $RESULTS_FOLDER \
          --result-filename ${new_test_name}.json \
          --request-rate $qps \
          --ignore-eos \
          $client_args"

      else

        echo "The dataset name must be either 'sharegpt' or 'sonnet'. Got $dataset_name."
        exit 1

      fi



      echo "Running test case $test_name with qps $qps"
      echo "Client command: $client_command"

      eval "$client_command"

      server_command="None"

      # record the benchmarking commands
      jq_output=$(jq -n \
        --arg server "$server_command" \
        --arg client "$client_command" \
        --arg gpu "$gpu_type" \
        --arg engine "$CURRENT_LLM_SERVING_ENGINE" \
        '{
          server_command: $server,
          client_command: $client,
          gpu_type: $gpu,
          engine: $engine
        }')
      echo "$jq_output" >"$RESULTS_FOLDER/${new_test_name}.commands"

    done

  done

  kill_gpu_processes
}

run_genai_perf_tests() {
  # run genai-perf tests

  # $1: a json file specifying genai-perf test cases
  local genai_perf_test_file
  genai_perf_test_file=$1

  # Iterate over genai-perf tests
  jq -c '.[]' "$genai_perf_test_file" | while read -r params; do
    # get the test name, and append the GPU type back to it.
    test_name=$(echo "$params" | jq -r '.test_name')

    # if TEST_SELECTOR is set, only run the test cases that match the selector
    if [[ -n "$TEST_SELECTOR" ]] && [[ ! "$test_name" =~ $TEST_SELECTOR ]]; then
      echo "Skip test case $test_name."
      continue
    fi

    # prepend the current serving engine to the test name
    test_name=${CURRENT_LLM_SERVING_ENGINE}_${test_name}

    # get common parameters
    common_params=$(echo "$params" | jq -r '.common_parameters')
    model=$(echo "$common_params" | jq -r '.model')
    tp=$(echo "$common_params" | jq -r '.tp')
    dataset_name=$(echo "$common_params" | jq -r '.dataset_name')
    dataset_path=$(echo "$common_params" | jq -r '.dataset_path')
    port=$(echo "$common_params" | jq -r '.port')
    num_prompts=$(echo "$common_params" | jq -r '.num_prompts')
    reuse_server=$(echo "$common_params" | jq -r '.reuse_server')

    # get client and server arguments
    server_params=$(echo "$params" | jq -r ".${CURRENT_LLM_SERVING_ENGINE}_server_parameters")
    qps_list=$(echo "$params" | jq -r '.qps_list')
    qps_list=$(echo "$qps_list" | jq -r '.[] | @sh')
    echo "Running over qps list $qps_list"

    # check if there is enough GPU to run the test
    if [[ $gpu_count -lt $tp ]]; then
      echo "Required num-shard $tp but only $gpu_count GPU found. Skip testcase $test_name."
      continue
    fi

    if [[ $reuse_server == "true" ]]; then
      echo "Reuse previous server for test case $test_name"
    else
      kill_gpu_processes
      bash "$VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/scripts/launch-server.sh" \
        "$server_params" "$common_params"
    fi

    if wait_for_server; then
      echo ""
      echo "$CURRENT_LLM_SERVING_ENGINE server is up and running."
    else
      echo ""
      echo "$CURRENT_LLM_SERVING_ENGINE failed to start within the timeout period."
      break
    fi

    # iterate over different QPS
    for qps in $qps_list; do
      # remove the surrounding single quote from qps
      if [[ "$qps" == *"inf"* ]]; then
        echo "qps was $qps"
        qps=$num_prompts
        echo "now qps is $qps"
      fi

      new_test_name=$test_name"_qps_"$qps
      backend=$CURRENT_LLM_SERVING_ENGINE

      if [[ "$backend" == *"vllm"* ]]; then
        backend="vllm"
      fi
      #TODO: add output dir.
      client_command="genai-perf profile \
        -m $model \
        --service-kind openai \
        --backend vllm \
        --endpoint-type chat \
        --streaming \
        --url localhost:$port \
        --request-rate $qps \
        --num-prompts $num_prompts \
      "

    echo "Client command: $client_command"

    eval "$client_command"

    #TODO: process/record outputs
    done
  done

  kill_gpu_processes

}

prepare_dataset() {

  # download sharegpt dataset
  cd "$VLLM_SOURCE_CODE_LOC/benchmarks"
  wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

  # duplicate sonnet by 4x, to allow benchmarking with input length 2048
  cd "$VLLM_SOURCE_CODE_LOC/benchmarks"
  echo "" > sonnet_4x.txt
  for _ in {1..4}
  do
    cat sonnet.txt >> sonnet_4x.txt
  done

}

main() {

  # check if the environment variable is successfully injected from yaml

  check_gpus
  check_hf_token
  get_current_llm_serving_engine

  pip install -U transformers

  pip install -r requirements/dev.txt
  which genai-perf

  # check storage
  df -h

  ensure_installed wget
  ensure_installed curl
  ensure_installed jq
  # genai-perf dependency
  ensure_installed libb64-0d

  prepare_dataset

  cd "$VLLM_SOURCE_CODE_LOC/benchmarks"
  declare -g RESULTS_FOLDER=results/
  mkdir -p $RESULTS_FOLDER
  BENCHMARK_ROOT="$VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/"

  # run the test
  run_serving_tests "$BENCHMARK_ROOT/tests/nightly-tests.json"

  # run genai-perf tests
  run_genai_perf_tests "$BENCHMARK_ROOT/tests/genai-perf-tests.json"
  mv artifacts/ $RESULTS_FOLDER/

  # upload benchmark results to buildkite
  python3 -m pip install tabulate pandas
  python3 "$BENCHMARK_ROOT/scripts/summary-nightly-results.py"
  upload_to_buildkite

}

main "$@"
