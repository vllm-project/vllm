#!/bin/bash
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-facebook/opt-125m}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-}
DTYPE=${DTYPE:-float32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
MAX_TOKENS=${MAX_TOKENS:-8}
PROMPT=${PROMPT:-The capital of France is}
SERVER_LOG=${SERVER_LOG:-/tmp/vllm-e2e-smoke.log}
PYTHON_BIN=${PYTHON_BIN:-python}
VLLM_CLI=("$PYTHON_BIN" -m vllm.entrypoints.cli.main)

server_pid=""
server_group_pid=""
marker_pid_file=""
marker_pgid_file=""

cleanup() {
  if [[ -n "$server_group_pid" ]] && kill -0 "$server_group_pid" 2>/dev/null; then
    kill -TERM -- "-$server_group_pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! kill -0 "$server_group_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    kill -KILL -- "-$server_group_pid" 2>/dev/null || true
  elif [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
  fi

  if [[ -n "$server_pid" ]]; then
    wait "$server_pid" || true
  fi

  if [[ -n "$marker_pid_file" || -n "$marker_pgid_file" ]]; then
    rm -f "$marker_pid_file" "$marker_pgid_file"
  fi
}

run_runner_npu_preflight_once() {
  "$PYTHON_BIN" - <<'PY'
import importlib.util
import os

import torch

if importlib.util.find_spec("torch_npu") is None:
    raise RuntimeError("torch_npu is not installed in the smoke-test environment")

import torch_npu  # noqa: F401

device = os.environ.get("VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE", "npu:0")
print(f"preflight device={device}")
print("torch_npu import ok=True")
if not torch.npu.is_available():
    raise RuntimeError("torch.npu.is_available() returned False")
torch.npu.set_device(device)
_ = torch.zeros(1, device=device)
print("torch.zeros preflight ok")
PY
}

ensure_runner_npu_ready() {
  local max_attempts=${RUNNER_NPU_PREFLIGHT_ATTEMPTS:-3}
  local delay_seconds=${RUNNER_NPU_PREFLIGHT_DELAY_SECONDS:-10}
  local attempt=1
  local preflight_output=""

  while [[ "$attempt" -le "$max_attempts" ]]; do
    if preflight_output=$(run_runner_npu_preflight_once 2>&1); then
      return 0
    fi

    echo "Runner NPU preflight failed ($attempt/$max_attempts)" >&2
    printf '%s\n' "$preflight_output" >&2

    if [[ "$attempt" -lt "$max_attempts" ]]; then
      sleep "$delay_seconds"
    fi
    attempt=$((attempt + 1))
  done

  echo "Self-hosted runner NPU runtime is unhealthy before vLLM startup." >&2
  if [[ "${GITHUB_EVENT_NAME:-}" == "pull_request" ]]; then
    echo "Skipping smoke test for this PR run because the failure is below vLLM and the runner cannot allocate a basic torch_npu tensor." >&2
    return 2
  fi

  return 1
}

start_server() {
  if command -v setsid >/dev/null 2>&1; then
    setsid "${VLLM_CLI[@]}" serve "$MODEL_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --dtype "$DTYPE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --enforce-eager >"$SERVER_LOG" 2>&1 &
    server_pid=$!
    server_group_pid=$server_pid
  else
    "${VLLM_CLI[@]}" serve "$MODEL_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --dtype "$DTYPE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --enforce-eager >"$SERVER_LOG" 2>&1 &
    server_pid=$!
  fi

  printf '%s\n' "$server_pid" > "$marker_pid_file"
  if [[ -n "$server_group_pid" ]]; then
    printf '%s\n' "$server_group_pid" > "$marker_pgid_file"
  fi
}

allocate_local_port() {
  "$PYTHON_BIN" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

trap cleanup EXIT

if [[ -z "$PORT" ]]; then
  PORT=$(allocate_local_port)
fi

runtime_root=${VLLM_HUST_CI_RUNTIME_ROOT:-${GITHUB_WORKSPACE:-$PWD}/.ci-runtime}
export HOME="$runtime_root/home"
export XDG_CACHE_HOME="$runtime_root/cache"
export XDG_CONFIG_HOME="$runtime_root/config"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" "$PIP_CACHE_DIR"

marker_dir="$runtime_root/process-markers"
marker_pid_file="$marker_dir/vllm-server.pid"
marker_pgid_file="$marker_dir/vllm-server.pgid"
mkdir -p "$marker_dir"

echo "Starting vLLM serve smoke test for $MODEL_NAME"
echo "Using smoke test port $PORT"

if ! ensure_runner_npu_ready; then
  status=$?
  if [[ "$status" -eq 2 ]]; then
    exit 0
  fi
  exit "$status"
fi

start_server

for attempt in $(seq 1 120); do
  if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null; then
    break
  fi

  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM server exited before becoming ready"
    cat "$SERVER_LOG"
    exit 1
  fi

  if [[ "$attempt" -eq 120 ]]; then
    echo "Timed out waiting for vLLM server to become ready"
    cat "$SERVER_LOG"
    exit 1
  fi

  sleep 2
done

curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null

completion_response=$(mktemp)
curl -fsS "http://$HOST:$PORT/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL_NAME\", \"prompt\": \"$PROMPT\", \"max_tokens\": $MAX_TOKENS, \"temperature\": 0}" \
  > "$completion_response"

"$PYTHON_BIN" - "$completion_response" "$MODEL_NAME" <<'PY'
import json
import sys

response_path, expected_model = sys.argv[1:3]

with open(response_path, encoding="utf-8") as handle:
    payload = json.load(handle)

assert payload.get("model") == expected_model, payload
choices = payload.get("choices")
assert isinstance(choices, list) and choices, payload
text = choices[0].get("text")
assert isinstance(text, str) and text.strip(), payload
usage = payload.get("usage")
assert isinstance(usage, dict) and usage.get("total_tokens", 0) > 0, payload
PY

rm -f "$completion_response"
echo "vLLM serve smoke test passed"