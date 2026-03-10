#!/bin/bash
# This script assumes that we are already inside the vllm/ directory
# Benchmarking results will be available inside vllm/benchmarks/results/

# Do not set -e, as the mixtral 8x22B model tends to crash occasionally
# and we still want to see other benchmarking results even when mixtral crashes.
set -x
set -o pipefail

# Environment-driven debug controls (like ON_CPU=1)
DRY_RUN="${DRY_RUN:-0}"
MODEL_FILTER="${MODEL_FILTER:-}"
DTYPE_FILTER="${DTYPE_FILTER:-}"

# Adaptive search controls
ENABLE_ADAPTIVE_CONCURRENCY="${ENABLE_ADAPTIVE_CONCURRENCY:-0}"
SLA_TTFT_MS="${SLA_TTFT_MS:-3000}"
SLA_TPOT_MS="${SLA_TPOT_MS:-100}"
ADAPTIVE_MAX_PROBES="${ADAPTIVE_MAX_PROBES:-8}"
ADAPTIVE_MAX_CONCURRENCY="${ADAPTIVE_MAX_CONCURRENCY:-1024}"

check_gpus() {
  if command -v nvidia-smi; then
    # check the number of GPUs and GPU type.
    declare -g gpu_count=$(nvidia-smi --list-gpus | grep -c . || true)
  elif command -v amd-smi; then
    declare -g gpu_count=$(amd-smi list | grep -c 'GPU' || true)
  elif command -v hl-smi; then
    declare -g gpu_count=$(hl-smi --list | grep -ci "Module ID" || true)
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
    echo "$numa_count"
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
  local timeout_val="1200"
  timeout "$timeout_val" bash -c '
    until curl -sf http://localhost:8000/v1/models >/dev/null; do
      sleep 1
    done
  '
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

# -------------------------------
# Adaptive concurrency helpers
# -------------------------------
result_json_path_for_serving() {
  local test_name=$1
  local qps=$2
  local max_concurrency=$3
  echo "$RESULTS_FOLDER/${test_name}_qps_${qps}_concurrency_${max_concurrency}.json"
}

extract_metric_ms() {
  local metric_name=$1
  local json_file=$2

  [[ -f "$json_file" ]] || return 0

  if [[ "$metric_name" == "ttft" ]]; then
    jq -r '
      [
        .ttft_ms.p99?,
        .metrics.ttft_ms.p99?,
        .ttft.p99?,
        .metrics.ttft.p99?,
        .p99_ttft_ms?,
        .ttft_ms.mean?,
        .metrics.ttft_ms.mean?,
        .ttft.mean?,
        .metrics.ttft.mean?,
        .mean_ttft_ms?
      ] | map(select(. != null)) | .[0] // empty
    ' "$json_file"
  else
    jq -r '
      [
        .tpot_ms.p99?,
        .metrics.tpot_ms.p99?,
        .tpot.p99?,
        .metrics.tpot.p99?,
        .p99_tpot_ms?,
        .itl_ms.p99?,
        .metrics.itl_ms.p99?,
        .inter_token_latency_ms.p99?,
        .tpot_ms.mean?,
        .metrics.tpot_ms.mean?,
        .tpot.mean?,
        .metrics.tpot.mean?,
        .itl_ms.mean?,
        .metrics.itl_ms.mean?,
        .mean_tpot_ms?,
        .mean_itl_ms?
      ] | map(select(. != null)) | .[0] // empty
    ' "$json_file"
  fi
}

evaluate_sla_from_json() {
  local json_file=$1
  local ttft
  local tpot
  local pass

  [[ -f "$json_file" ]] || return 2

  ttft=$(extract_metric_ms ttft "$json_file")
  tpot=$(extract_metric_ms tpot "$json_file")

  [[ -n "$ttft" && -n "$tpot" ]] || return 2

  pass=$(jq -n \
    --argjson ttft "$ttft" \
    --argjson tpot "$tpot" \
    --argjson sla_ttft "$SLA_TTFT_MS" \
    --argjson sla_tpot "$SLA_TPOT_MS" \
    '($ttft <= $sla_ttft) and ($tpot <= $sla_tpot)')

  [[ "$pass" == "true" ]]
}

write_adaptive_summary_json() {
  local summary_file=$1
  local test_name=$2
  local qps=$3
  local static_last_pass=$4
  local static_first_fail=$5
  local final_last_pass=$6
  local final_first_fail=$7

  jq -n \
    --arg test_name "$test_name" \
    --arg qps "$qps" \
    --argjson sla_ttft "$SLA_TTFT_MS" \
    --argjson sla_tpot "$SLA_TPOT_MS" \
    --arg static_last_pass "${static_last_pass:-}" \
    --arg static_first_fail "${static_first_fail:-}" \
    --arg final_last_pass "${final_last_pass:-}" \
    --arg final_first_fail "${final_first_fail:-}" \
    '{
      test_name: $test_name,
      qps: $qps,
      sla_ttft_ms: $sla_ttft,
      sla_tpot_ms: $sla_tpot,
      static_last_pass: (if $static_last_pass == "" then null else ($static_last_pass | tonumber) end),
      static_first_fail: (if $static_first_fail == "" then null else ($static_first_fail | tonumber) end),
      final_last_pass: (if $final_last_pass == "" then null else ($final_last_pass | tonumber) end),
      final_first_fail: (if $final_first_fail == "" then null else ($final_first_fail | tonumber) end)
    }' > "$summary_file"
}

run_single_serving_probe() {
  local test_name=$1
  local qps=$2
  local max_concurrency=$3
  local tp=$4
  local compilation_config_mode=$5
  local optimization_level=$6
  local client_args_effective=$7
  local client_remote_args=$8
  local server_command=$9

  local new_test_name="${test_name}_qps_${qps}_concurrency_${max_concurrency}"
  local result_json
  local num_prompts_arg=""
  local client_command

  result_json=$(result_json_path_for_serving "$test_name" "$qps" "$max_concurrency")

  if [[ -f "$result_json" ]]; then
    evaluate_sla_from_json "$result_json"
    return $?
  fi

  if [[ -n "${PROMPTS_PER_CONCURRENCY}" ]]; then
    num_prompts=$(( max_concurrency * PROMPTS_PER_CONCURRENCY ))
    if (( num_prompts < MIN_NUM_PROMPTS )); then num_prompts=$MIN_NUM_PROMPTS; fi
    if (( num_prompts > MAX_NUM_PROMPTS )); then num_prompts=$MAX_NUM_PROMPTS; fi
    num_prompts_arg="--num-prompts $num_prompts"
  fi

  client_command="vllm bench serve \
    --save-result \
    --result-dir $RESULTS_FOLDER \
    --result-filename ${new_test_name}.json \
    --request-rate $qps \
    --max-concurrency $max_concurrency \
    $num_prompts_arg \
    --metadata tensor_parallel_size=$tp compilation_config.mode=$compilation_config_mode optimization_level=$optimization_level adaptive_search=1 \
    $client_args_effective $client_remote_args "

  echo "Adaptive probe: $client_command"

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    bash -c "$client_command"
  fi

  jq_output=$(jq -n \
    --arg server "$server_command" \
    --arg client "$client_command" \
    --arg gpu "$gpu_type" \
    '{
      server_command: $server,
      client_command: $client,
      gpu_type: $gpu,
      adaptive_search: true
    }')
  echo "$jq_output" > "$RESULTS_FOLDER/${new_test_name}.commands"

  evaluate_sla_from_json "$result_json"
}

adaptive_refine_from_static_results() {
  local test_name=$1
  local qps=$2
  local max_concurrency_list_raw=$3
  local tp=$4
  local compilation_config_mode=$5
  local optimization_level=$6
  local client_args_effective=$7
  local client_remote_args=$8
  local server_command=$9

  local sorted_points
  local point
  local rc
  local static_last_pass=""
  local static_first_fail=""
  local largest_static=""
  local step_hint=1
  local previous_point=""
  local low
  local high
  local mid
  local probes=0
  local summary_file="$RESULTS_FOLDER/${test_name}_qps_${qps}_sla_summary.json"

  [[ "${ENABLE_ADAPTIVE_CONCURRENCY}" == "1" ]] || return 0
  [[ "${DRY_RUN:-0}" != "1" ]] || return 0

  sorted_points=$(for point in $max_concurrency_list_raw; do printf '%s\n' "$point"; done | tr -d "'" | awk '/^[0-9]+$/' | sort -n | uniq)
  [[ -n "$sorted_points" ]] || return 0

  while read -r point; do
    [[ -z "$point" ]] && continue
    largest_static="$point"
    evaluate_sla_from_json "$(result_json_path_for_serving "$test_name" "$qps" "$point")"
    rc=$?
    if (( rc == 0 )); then
      static_last_pass="$point"
    elif (( rc == 1 )); then
      if [[ -n "$static_last_pass" ]]; then
        static_first_fail="$point"
        break
      fi
    fi

    if [[ -n "$previous_point" ]]; then
      step_hint=$(( point - previous_point ))
      if (( step_hint < 1 )); then step_hint=1; fi
    fi
    previous_point="$point"
  done <<< "$sorted_points"

  if [[ -z "$static_last_pass" ]]; then
    write_adaptive_summary_json "$summary_file" "$test_name" "$qps" "" "$static_first_fail" "" "$static_first_fail"
    return 0
  fi

  if [[ -n "$static_first_fail" ]]; then
    low=$static_last_pass
    high=$static_first_fail
    while (( low + 1 < high )) && (( probes < ADAPTIVE_MAX_PROBES )); do
      mid=$(( (low + high) / 2 ))
      probes=$(( probes + 1 ))
      run_single_serving_probe \
        "$test_name" "$qps" "$mid" "$tp" \
        "$compilation_config_mode" "$optimization_level" \
        "$client_args_effective" "$client_remote_args" "$server_command"
      rc=$?
      if (( rc == 0 )); then
        low=$mid
      elif (( rc == 1 )); then
        high=$mid
      else
        break
      fi
    done
    write_adaptive_summary_json "$summary_file" "$test_name" "$qps" "$static_last_pass" "$static_first_fail" "$low" "$high"
    return 0
  fi

  low=$largest_static
  high=""
  while (( probes < ADAPTIVE_MAX_PROBES )); do
    point=$(( low + step_hint ))
    if (( point > ADAPTIVE_MAX_CONCURRENCY )); then
      point=$ADAPTIVE_MAX_CONCURRENCY
    fi
    (( point > low )) || break
    probes=$(( probes + 1 ))
    run_single_serving_probe \
      "$test_name" "$qps" "$point" "$tp" \
      "$compilation_config_mode" "$optimization_level" \
      "$client_args_effective" "$client_remote_args" "$server_command"
    rc=$?
    if (( rc == 0 )); then
      low=$point
      (( point == ADAPTIVE_MAX_CONCURRENCY )) && break
      step_hint=$(( step_hint * 2 ))
      if (( step_hint < 1 )); then step_hint=1; fi
    elif (( rc == 1 )); then
      high=$point
      break
    else
      break
    fi
  done

  if [[ -n "$high" ]]; then
    while (( low + 1 < high )) && (( probes < ADAPTIVE_MAX_PROBES )); do
      mid=$(( (low + high) / 2 ))
      probes=$(( probes + 1 ))
      run_single_serving_probe \
        "$test_name" "$qps" "$mid" "$tp" \
        "$compilation_config_mode" "$optimization_level" \
        "$client_args_effective" "$client_remote_args" "$server_command"
      rc=$?
      if (( rc == 0 )); then
        low=$mid
      elif (( rc == 1 )); then
        high=$mid
      else
        break
      fi
    done
  fi

  write_adaptive_summary_json "$summary_file" "$test_name" "$qps" "$static_last_pass" "" "$low" "$high"
}

run_benchmark_tests() {
  # run benchmark tests using `vllm bench <test_type>` command
  # $1: test type (latency or throughput)
  # $2: a json file specifying test cases

  local test_type=$1
  local test_file=$2

  # Iterate over tests
  jq -c '.[]' "$test_file" | while read -r params; do
    # get the test name, and append the GPU type back to it.
    test_name=$(echo "$params" | jq -r '.test_name')
    if [[ ! "$test_name" =~ ^${test_type}_ ]]; then
      echo "In ${test_type}-test.json, test_name must start with \"${test_type}_\"."
      exit 1
    fi

    # if TEST_SELECTOR is set, only run the test cases that match the selector
    if [[ -n "$TEST_SELECTOR" ]] && [[ ! "$test_name" =~ $TEST_SELECTOR ]]; then
      echo "Skip test case $test_name."
      continue
    fi

    # get arguments
    bench_params=$(echo "$params" | jq -r '.parameters')
    bench_args=$(json2args "$bench_params")
    bench_environment_variables=$(echo "$params" | jq -r '.environment_variables')
    bench_envs=$(json2envs "$bench_environment_variables")

    # check if there is enough GPU to run the test
    tp=$(echo "$bench_params" | jq -r '.tensor_parallel_size')
    if [[ "$ON_CPU" == "1" ]]; then
      pp=$(echo "$bench_params" | jq -r '.pipeline_parallel_size // 1')
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

    bench_command=" $bench_envs vllm bench $test_type \
      --output-json $RESULTS_FOLDER/${test_name}.json \
      $bench_args"

    echo "Running test case $test_name"
    echo "${test_type^} command: $bench_command"

    # recording benchmarking command and GPU command
    jq_output=$(jq -n \
      --arg command "$bench_command" \
      --arg gpu "$gpu_type" \
      --arg test_type "$test_type" \
      '{
        ($test_type + "_command"): $command,
        gpu_type: $gpu
      }')
    echo "$jq_output" >"$RESULTS_FOLDER/$test_name.commands"

    # run the benchmark
    eval "$bench_command"

    kill_gpu_processes

  done
}

run_latency_tests() { run_benchmark_tests "latency" "$1"; }
run_startup_tests() { run_benchmark_tests "startup" "$1"; }
run_throughput_tests() { run_benchmark_tests "throughput" "$1"; }

merge_serving_tests_stream() {
  # Emit merged serving test objects, optionally filtered by MODEL_FILTER/DTYPE_FILTER in DRY_RUN mode.
  # This helper does NOT modify JSON; it only filters the stream in dry-run mode.
  local serving_test_file="$1"
  # shellcheck disable=SC2016
  local merged='
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
  '

  jq -c "$merged" "$serving_test_file" | \
  if [[ "${DRY_RUN:-0}" == "1" && ( "${MODEL_FILTER}${DTYPE_FILTER}" != "" ) ]]; then
    jq -c --arg model "$MODEL_FILTER" --arg dtype "$DTYPE_FILTER" '
      select((($model|length)==0)
             or ((.server_parameters.model // "") == $model)
             or ((.client_parameters.model // "") == $model))
      | select((($dtype|length)==0) or ((.server_parameters.dtype // "") == $dtype))
    '
  else
    cat
  fi
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

  # In dry-run mode, if filters are provided but no tests match, fail fast.
  if [[ "${DRY_RUN:-0}" == "1" && ( "${MODEL_FILTER}${DTYPE_FILTER}" != "" ) ]]; then
    local count
    count=$(merge_serving_tests_stream "$serving_test_file" | wc -l | tr -d ' ')
    if [[ "$count" -eq 0 ]]; then
      echo "No matching serving tests found in $serving_test_file for model='$MODEL_FILTER' dtype='$DTYPE_FILTER'." >&2
      return 0
    fi
  fi

  # Iterate over serving tests (merged + optional filtered stream)
  merge_serving_tests_stream "$serving_test_file" | while read -r params; do
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

    # vLLM serve CLI: model must be positional (no --model). Convert server_parameters accordingly.
    server_model=$(echo "$server_params" | jq -r '.model // empty')
    if [[ -z "$server_model" || "$server_model" == "null" ]]; then
      echo "Error: serving test '$test_name' is missing server_parameters.model" >&2
      exit 1
    fi
    server_params_no_model=$(echo "$server_params" | jq -c 'del(.model)')
    server_args=$(json2args "$server_params_no_model")

    server_envs=$(json2envs "$server_envs")
    client_args=$(json2args "$client_params")

    # ------------------------------------------------------------
    # Option 1: Dynamic num-prompts scaling based on max_concurrency
    #
    # If PROMPTS_PER_CONCURRENCY is set, override JSON num_prompts with:
    #   num_prompts = max_concurrency * PROMPTS_PER_CONCURRENCY
    #
    # If PROMPTS_PER_CONCURRENCY is NOT set, keep JSON num_prompts behavior
    # unchanged (i.e., whatever is in serving-tests-*.json).
    # ------------------------------------------------------------
    PROMPTS_PER_CONCURRENCY="${PROMPTS_PER_CONCURRENCY-}"  # no default on purpose
    MIN_NUM_PROMPTS="${MIN_NUM_PROMPTS:-1}"
    MAX_NUM_PROMPTS="${MAX_NUM_PROMPTS:-1000000}"

    if [[ -n "${PROMPTS_PER_CONCURRENCY}" ]]; then
      # Remove any fixed --num-prompts from JSON-derived args (avoid duplicates)
      # Remove any fixed --num-prompts from JSON-derived args (avoid duplicates)
      # Handles: --num-prompts 123   and   --num-prompts=123
      client_args_no_np="$(
        printf ' %s ' "$client_args" \
        | sed -E \
          -e 's/[[:space:]]--num-prompts=([^[:space:]]+)([[:space:]]|$)/ /g' \
          -e 's/[[:space:]]--num-prompts[[:space:]]+([^[:space:]]+)([[:space:]]|$)/ /g'
      )"
      # normalize whitespace
      client_args_no_np="$(echo "$client_args_no_np" | tr -s ' ' | sed -E 's/^ //; s/ $//')"
      client_args_no_np="$(echo "$client_args_no_np" | xargs)"
      client_args_effective="$client_args_no_np"
    else
      client_args_effective="$client_args"
    fi
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
    client_model=$(echo "$client_params" | jq -r '.model')
    if [[ $server_model != "$client_model" ]]; then
      echo "Server model and client model must be the same. Skip testcase $test_name."
      continue
    fi

    server_command="$server_envs vllm serve $server_model \
      $server_args"

    # run the server
    echo "Running test case $test_name"
    echo "Server command: $server_command"
    # support remote vllm server
    client_remote_args=""
    if [[ -z "${REMOTE_HOST}" && "${DRY_RUN:-0}" != "1" ]]; then
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
    elif [[ "${DRY_RUN:-0}" == "1" ]]; then
        # dry-run: don't start server
        echo "Dry Run."
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
    compilation_config_mode=$(echo "$server_params" | jq -r '."compilation_config.mode" // empty')
    optimization_level=$(echo "$server_params" | jq -r '.optimization_level // empty')

    # iterate over different QPS
    for qps in $qps_list; do
      # remove the surrounding single quote from qps
      if [[ "$qps" == *"inf"* ]]; then
        qps="inf"
      fi

      # iterate over different max_concurrency
      for max_concurrency in $max_concurrency_list; do
        new_test_name="${test_name}_qps_${qps}_concurrency_${max_concurrency}"
        echo " new test name $new_test_name"
        # If PROMPTS_PER_CONCURRENCY is set, compute per-concurrency --num-prompts.
        num_prompts_arg=""
        if [[ -n "${PROMPTS_PER_CONCURRENCY}" ]]; then
          num_prompts=$(( max_concurrency * PROMPTS_PER_CONCURRENCY ))
          if (( num_prompts < MIN_NUM_PROMPTS )); then num_prompts=$MIN_NUM_PROMPTS; fi
          if (( num_prompts > MAX_NUM_PROMPTS )); then num_prompts=$MAX_NUM_PROMPTS; fi
          num_prompts_arg="--num-prompts $num_prompts"
        fi
        # pass the tensor parallel size, the compilation mode, and the optimization
        # level to the client so that they can be used on the benchmark dashboard
        client_command="vllm bench serve \
          --save-result \
          --result-dir $RESULTS_FOLDER \
          --result-filename ${new_test_name}.json \
          --request-rate $qps \
          --max-concurrency $max_concurrency \
          $num_prompts_arg \
          --metadata tensor_parallel_size=$tp compilation_config.mode=$compilation_config_mode optimization_level=$optimization_level \
          $client_args_effective $client_remote_args "

        echo "Running test case $test_name with qps $qps"
        echo "Client command: $client_command"

        if [[ "${DRY_RUN:-0}" != "1" ]]; then
          bash -c "$client_command"
        fi

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

      adaptive_refine_from_static_results \
        "$test_name" "$qps" "$max_concurrency_list" "$tp" \
        "$compilation_config_mode" "$optimization_level" \
        "$client_args_effective" "$client_remote_args" "$server_command"
    done

    # clean up
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      kill -9 "$server_pid"
      kill_gpu_processes
    fi
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

  # DRY_RUN does not execute vLLM; do not require HF_TOKEN.
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    check_hf_token
  else
    echo "DRY_RUN=1 -> skip HF_TOKEN validation"
  fi

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
  run_serving_tests $QUICK_BENCHMARK_ROOT/tests/"${SERVING_JSON:-serving-tests$ARCH.json}" || exit $?

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1 -> skip latency/startup/throughput suites"
    exit 0
  fi

  run_latency_tests $QUICK_BENCHMARK_ROOT/tests/"${LATENCY_JSON:-latency-tests$ARCH.json}"
  run_startup_tests $QUICK_BENCHMARK_ROOT/tests/"${STARTUP_JSON:-startup-tests$ARCH.json}"
  run_throughput_tests $QUICK_BENCHMARK_ROOT/tests/"${THROUGHPUT_JSON:-throughput-tests$ARCH.json}"

  # postprocess benchmarking results
  pip install tabulate pandas
  python3 $QUICK_BENCHMARK_ROOT/scripts/convert-results-json-to-markdown.py
  python3 $QUICK_BENCHMARK_ROOT/scripts/compare-json-results.py -f $RESULTS_FOLDER/benchmark_results.json

  upload_to_buildkite
}

main "$@"
