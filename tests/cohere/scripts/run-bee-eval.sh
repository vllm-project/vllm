#!/bin/bash

# This script should be run inside the CI process
# This script assumes that we are already inside the vllm/ directory

# Do not set -e, as the mixtral 8x22B model tends to crash occasionally
# and we still want to see other benchmarking results even when mixtral crashes.
set -x
set -o pipefail

# Required environment variables (should be set by the parent script run_tests.sh)
if [[ -z "${VLLM_WORKSPACE}" ]]; then
  echo "Error: VLLM_WORKSPACE is not set"
  exit 1
fi
if [[ -z "${BEE_DIR}" ]]; then
  echo "Error: BEE_DIR is not set"
  exit 1
fi

source tests/cohere/scripts/run-helper.sh

# Apply Cohere hardware_profiles.yaml inside spawned vllm processes.
export VLLM_ENABLE_COHERE_AUTO_CONFIG=1

run_bee_eval() {
  # run bee eval tests using `vllm serve` and `uv run bee` command
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

    # get eval and server arguments
    server_params=$(echo "$params" | jq -r '.server_parameters')
    server_envs=$(echo "$params" | jq -r '.server_environment_variables')
    eval_params=$(echo "$params" | jq -r '.eval_parameters')
    server_args=$(json2args "$server_params")
    server_envs=$(json2envs "$server_envs")

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

    # check if server model and eval model is aligned
    server_model=$(echo "$server_params" | jq -r '.served_model_name')
    eval_model=$(echo "$eval_params" | jq -r '.model')
    if [[ $server_model != "$eval_model" ]]; then
      echo "Server model and eval model must be the same. Skip testcase $test_name."
      continue
    fi

    server_command="$server_envs vllm serve \
      $server_args"

    # run the server
    echo "Running test case $test_name"
    echo "Server command: $server_command"

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

    # Convert tests to "-I test1 -I test2 -I test3"
    tests=$(echo "$eval_params" | jq -r '.tests')
    # Create non-reasoning test args (exclude tests with "reasoning" in name)
    non_reasoning_test_args=$(echo "$tests" | jq -r 'map(select(. | index("reasoning") | not)) | map("-I " + .) | join(" ")')
    # Create reasoning test args (include only tests with "reasoning" in name)
    reasoning_test_args=$(echo "$tests" | jq -r 'map(select(. | index("reasoning") | .)) | map("-I " + .) | join(" ")')
    # Build --thinking_token_budget args from eval config if set
    non_reasoning_thinking_budget_arg=""
    reasoning_thinking_budget_arg=""
    thinking_budget=$(echo "$eval_params" | jq -r '.thinking_token_budget // empty')
    reasoning_thinking_budget=$(echo "$eval_params" | jq -r '.reasoning_thinking_token_budget // empty')
    if [[ -n "$thinking_budget" ]]; then
      non_reasoning_thinking_budget_arg="--thinking_token_budget $thinking_budget"
    fi
    if [[ -n "$reasoning_thinking_budget" ]]; then
      reasoning_thinking_budget_arg="--thinking_token_budget $reasoning_thinking_budget"
    fi

    # Base models are emitted with raw_prompting=true by
    # generate-serving-config.py. Forward it to bee so requests go to
    # /v1/completions instead of /v1/chat/completions.
    raw_prompting_arg=$(echo "$eval_params" | jq -r 'if .raw_prompting then "--raw_prompting True" else "" end')
    settings_toml=$(echo "$eval_params" | jq -r '.settings_toml')


    # Once bee has its packages fixed we use this command instead
    # eval_command="uvx --python 3.11 \
    #   --extra-index-url https://oauth2accesstoken@us-central1-python.pkg.dev/cohere-artifacts/cohere-py/simple \
    #   --index-strategy unsafe-best-match \
    #   --keyring-provider subprocess \
    cd ${BEE_DIR}
    if [[ !  -z "$non_reasoning_test_args" ]]; then
      non_reasoning_eval_command="uv run --no-sync \
        bee \
        --beedb False \
        --enable_local_disk True \
        --skip_completed_tasks False \
        --log_samples_n 1 \
        -I $settings_toml \
        $non_reasoning_test_args \
        --estimator VLLMEstimator \
        --timeout 900 \
        --max_retries 3 \
        --model $eval_model \
        --base_url http://127.0.0.1:8000/v1 \
        $raw_prompting_arg \
        $non_reasoning_thinking_budget_arg"
    fi

    if [[ !  -z "$reasoning_test_args" ]]; then
      reasoning_eval_command="uv run --no-sync \
        bee \
        --beedb False \
        --enable_local_disk True \
        --skip_completed_tasks False \
        --log_samples_n 1 \
        -I $settings_toml \
        $reasoning_test_args \
        --estimator VLLMEstimator \
        --num_workers 16 \
        --timeout 1500 \
        --max_retries 3 \
        --model $eval_model \
        --base_url http://127.0.0.1:8000/v1 \
        $raw_prompting_arg \
        $reasoning_thinking_budget_arg"
    fi

    # Concatenate commands with && if both are non-empty
    if [[ ! -z "$non_reasoning_eval_command" && ! -z "$reasoning_eval_command" ]]; then
      eval_command="$non_reasoning_eval_command && $reasoning_eval_command"
    elif [[ ! -z "$non_reasoning_eval_command" ]]; then
      eval_command="$non_reasoning_eval_command"
    elif [[ ! -z "$reasoning_eval_command" ]]; then
      eval_command="$reasoning_eval_command"
    else
      echo "No eval task found"
      exit 1
    fi

    echo "Eval command: $eval_command"

    bash -c "$eval_command"
    cd ${VLLM_WORKSPACE}

    # record the benchmarking commands
    jq_output=$(jq -n \
      --arg server "$server_command" \
      --arg eval "$eval_command" \
      --arg gpu "$gpu_type" \
      '{
        server_command: $server,
        eval_command: $eval,
        gpu_type: $gpu
      }')
    echo "$jq_output" > "$RESULTS_FOLDER/${test_name}.commands"

    # clean up
    kill -9 $server_pid
    kill_gpu_processes
  done
}

main() {
  local ARCH
  ARCH=''
  if [ "$ON_CPU" == "1" ];then
     check_cpus
     ARCH='-cpu'
  else
     check_gpus
  fi

  # Set to v1 to run v1 benchmark
  if [[ "${ENGINE_VERSION:-v0}" == "v1" ]]; then
    export VLLM_USE_V1=1
  fi

  # get the current IP address, required by `vllm bench serve` command
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  # turn of the reporting of the status of each request, to clean up the terminal output
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

  declare -g RESULTS_FOLDER=results/
  mkdir -p $RESULTS_FOLDER

  # Tell bee to write local-disk output to a fixed, known location so we
  # don't have to guess site-packages paths.
  export APIARY_OUTPUT_PATH="${BEE_DIR}/output"
  mkdir -p "${APIARY_OUTPUT_PATH}"

  # generate configs for serving test sweep
  MODEL_NAME=$1 MODEL_PATH=$2 python3 tests/cohere/scripts/generate-serving-config.py --mode eval

  # Eval
  run_bee_eval tests/cohere/configs/"${SERVING_JSON:-serving-cohere-tests$ARCH.json}"

  # Copy summary_metrics to mounted directory
  if [[ -z "${APIARY_OUTPUT_PATH}" ]]; then
    echo "Error: Could not extract summary_metrics_path from Bee output."
    exit 1
  fi
  # Loop through each subfolder in the source directory
  for subfolder in "${APIARY_OUTPUT_PATH}"/*/; do
    # Get just the subfolder name (without path)
    subfolder_name=$(basename "$subfolder")

    # Path to the source file
    src_file="$subfolder/summary_metrics.jsonl"

    # Check if the file exists
    if [ -f "$src_file" ]; then
      # Destination file name
      dest_file="${RESULTS_FOLDER}$1_${subfolder_name}_summary_metrics.jsonl"

      # Copy the file
      echo "Copying $src_file to $dest_file"
      cp "$src_file" "$dest_file"
    else
      echo "No summary_metrics.jsonl found in $subfolder_name"
    fi

    # Per-task metrics (task_duration, usage/response, etc.) for convert-eval-results-to-json.py
    for mfile in "$subfolder"/metrics_*.jsonl; do
      [[ -f "$mfile" ]] || continue
      echo "Copying $mfile to ${RESULTS_FOLDER}$1_${subfolder_name}_$(basename "$mfile")"
      cp "$mfile" "${RESULTS_FOLDER}$1_${subfolder_name}_$(basename "$mfile")"
    done
  done
  rm -r "${APIARY_OUTPUT_PATH}"
}

main "$@"
