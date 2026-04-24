# Helper functions for run-performance-benchmarks.sh and run-bee-eval.sh

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
  declare -g gpu_type="cpu"
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
  timeout 600 bash -c '
    until curl -s -o /dev/null -X POST localhost:8000/v1/completions; do
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
