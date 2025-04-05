#!/bin/bash

function check_hf_token() {
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
    until curl -X POST localhost:8000/v1/completions; do
      sleep 1
    done' && return 0 || return 1
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
