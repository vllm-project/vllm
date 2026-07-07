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
CHAT_MESSAGE=${CHAT_MESSAGE:-Tell me one short fact about France.}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-5}
SERVER_LOG=${SERVER_LOG:-/tmp/vllm-e2e-regression.log}
RUNTIME_READY_LOG=${RUNTIME_READY_LOG:-/tmp/vllm-e2e-regression-runtime-ready.log}
PYTHON_BIN=${PYTHON_BIN:-python}
VLLM_ASCEND_HUST_REPO=${VLLM_ASCEND_HUST_REPO:-${GITHUB_WORKSPACE:-$PWD}/vllm-ascend-hust}
SUDO_AUTH_EXIT_CODE=${SUDO_AUTH_EXIT_CODE:-76}
ASCEND_E2E_USE_SUDO=${ASCEND_E2E_USE_SUDO:-0}
DEFAULT_SYSTEM_ASCEND_ROOT_HELPER=${DEFAULT_SYSTEM_ASCEND_ROOT_HELPER:-/usr/local/bin/run_ascend_benchmark_root_helper.sh}
REPO_ASCEND_ROOT_HELPER=${REPO_ASCEND_ROOT_HELPER:-$VLLM_ASCEND_HUST_REPO/.github/workflows/scripts/run_ascend_benchmark_root_helper.sh}
REPO_ASCEND_ROOT_HELPER_INSTALL_SCRIPT=${REPO_ASCEND_ROOT_HELPER_INSTALL_SCRIPT:-$VLLM_ASCEND_HUST_REPO/scripts/install_ascend_benchmark_root_helper.sh}
if [[ -n "${ASCEND_E2E_ROOT_HELPER:-}" ]]; then
  ASCEND_E2E_ROOT_HELPER=${ASCEND_E2E_ROOT_HELPER}
elif [[ -x "$DEFAULT_SYSTEM_ASCEND_ROOT_HELPER" ]]; then
  ASCEND_E2E_ROOT_HELPER=$DEFAULT_SYSTEM_ASCEND_ROOT_HELPER
else
  ASCEND_E2E_ROOT_HELPER=$REPO_ASCEND_ROOT_HELPER
fi
VLLM_CLI=("$PYTHON_BIN" -m vllm.entrypoints.cli.main)
VLLM_OPENAI_SERVER=("$PYTHON_BIN" -m vllm.entrypoints.openai.api_server)

server_pid=""
server_group_pid=""
marker_pid_file=""
marker_pgid_file=""

SUDO_PRESERVE_ENV_VARS=(
  ASCEND_AICPU_PATH
  ASCEND_HOME_PATH
  ASCEND_OPP_PATH
  ASCEND_RT_VISIBLE_DEVICES
  ASCEND_TOOLKIT_HOME
  ASCEND_TOOLKIT_LATEST_HOME
  ASCEND_VISIBLE_DEVICES
  ATB_HOME_PATH
  DTYPE
  HCCL_CONNECT_TIMEOUT
  HCCL_EXEC_TIMEOUT
  HF_ENDPOINT
  HF_HOME
  HF_TOKEN
  HOME
  HOST
  HUST_ATB_SET_ENV
  HUGGINGFACE_HUB_CACHE
  LD_LIBRARY_PATH
  MAX_MODEL_LEN
  MAX_NUM_SEQS
  MODEL_NAME
  PATH
  PIP_CACHE_DIR
  PORT
  PYTHON_BIN
  PYTHONPATH
  TRANSFORMERS_CACHE
  VLLM_CACHE_ROOT
  VLLM_CONFIG_ROOT
  WORKSPACE_ROOT
  XDG_CACHE_HOME
  XDG_CONFIG_HOME
)

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

build_sudo_env_preserve_list() {
  local preserved=()
  local var_name

  for var_name in "${SUDO_PRESERVE_ENV_VARS[@]}"; do
    if [[ -n "${!var_name+x}" ]]; then
      preserved+=("$var_name")
    fi
  done

  if [[ "${#preserved[@]}" -eq 0 ]]; then
    return 0
  fi

  local joined
  joined=$(IFS=,; printf '%s' "${preserved[*]}")
  printf '%s\n' "$joined"
}

export_sudo_preserved_env_vars() {
  local var_name

  for var_name in "${SUDO_PRESERVE_ENV_VARS[@]}"; do
    if [[ -n "${!var_name+x}" ]]; then
      export "$var_name"
    fi
  done
}

prepare_root_helper_python_bin() {
  local helper_runtime_root=${runtime_root:-${VLLM_HUST_CI_RUNTIME_ROOT:-${GITHUB_WORKSPACE:-$PWD}/.ci-runtime}}
  local helper_python_bin="${ROOT_HELPER_PYTHON_BIN_FILE:-$helper_runtime_root/root-helper-python.sh}"
  local resolved_python_bin=${PYTHON_BIN:?PYTHON_BIN must be set}
  local quoted_python_bin

  quoted_python_bin=$(printf '%q' "$resolved_python_bin")
  mkdir -p "$(dirname "$helper_python_bin")"
  cat >"$helper_python_bin" <<EOF
#!/usr/bin/env bash
set -euo pipefail

if [[ -n "\${HUST_ATB_SET_ENV:-}" && -f "\${HUST_ATB_SET_ENV}" ]]; then
  set +u
  source "\${HUST_ATB_SET_ENV}" --cxx_abi=1
  set -u
elif [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  set +u
  source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
  set -u
fi

exec ${quoted_python_bin} "\$@"
EOF
  chmod 755 "$helper_python_bin"
  printf '%s\n' "$helper_python_bin"
}

benchmark_root_helper_fix_command() {
  printf 'sudo RUNNER_USER=grunner bash %s\n' "$REPO_ASCEND_ROOT_HELPER_INSTALL_SCRIPT"
}

report_runner_host_fix() {
  if [[ -f "$REPO_ASCEND_ROOT_HELPER_INSTALL_SCRIPT" ]]; then
    echo "Runner host fix: $(benchmark_root_helper_fix_command)" >&2
  else
    echo "Runner-local install script missing from checkout: $REPO_ASCEND_ROOT_HELPER_INSTALL_SCRIPT" >&2
  fi
}

verify_root_helper_ready() {
  if [[ "$ASCEND_E2E_USE_SUDO" != "1" ]]; then
    return 0
  fi

  if [[ ! -f "$REPO_ASCEND_ROOT_HELPER" ]]; then
    echo "Runner-local root helper source missing from checkout: $REPO_ASCEND_ROOT_HELPER" >&2
    return 1
  fi

  if [[ ! -f "$REPO_ASCEND_ROOT_HELPER_INSTALL_SCRIPT" ]]; then
    echo "Runner-local helper install script missing from checkout: $REPO_ASCEND_ROOT_HELPER_INSTALL_SCRIPT" >&2
    return 1
  fi

  if [[ "$ASCEND_E2E_ROOT_HELPER" != "$DEFAULT_SYSTEM_ASCEND_ROOT_HELPER" ]]; then
    if [[ ! -x "$ASCEND_E2E_ROOT_HELPER" ]]; then
      echo "Configured Ascend E2E root helper is not executable: $ASCEND_E2E_ROOT_HELPER" >&2
      return 1
    fi
    return 0
  fi

  if [[ ! -x "$ASCEND_E2E_ROOT_HELPER" ]]; then
    echo "Installed Ascend E2E root helper is missing or not executable: $ASCEND_E2E_ROOT_HELPER" >&2
    report_runner_host_fix
    return 1
  fi

  if ! cmp -s "$ASCEND_E2E_ROOT_HELPER" "$REPO_ASCEND_ROOT_HELPER"; then
    echo "Installed Ascend E2E root helper is stale: $ASCEND_E2E_ROOT_HELPER does not match $REPO_ASCEND_ROOT_HELPER" >&2
    report_runner_host_fix
    return 1
  fi
}

run_ascend_root_helper() {
  local preserve_list
  local helper_python_bin

  if [[ "$ASCEND_E2E_USE_SUDO" != "1" ]]; then
    echo "run_ascend_root_helper requires ASCEND_E2E_USE_SUDO=1" >&2
    return 2
  fi

  export_sudo_preserved_env_vars
  preserve_list=$(build_sudo_env_preserve_list)
  helper_python_bin=$(prepare_root_helper_python_bin)
  if [[ -n "$preserve_list" ]]; then
    PYTHON_BIN="$helper_python_bin" sudo --preserve-env="$preserve_list" -E -n "$ASCEND_E2E_ROOT_HELPER" "$@"
  else
    PYTHON_BIN="$helper_python_bin" sudo -E -n "$ASCEND_E2E_ROOT_HELPER" "$@"
  fi
}

runtime_ready_log_indicates_sudo_auth_failure() {
  [[ -f "$RUNTIME_READY_LOG" ]] && grep -qE 'sudo: (a password is required|a terminal is required|sorry, you must have a tty|is not allowed to execute)' "$RUNTIME_READY_LOG"
}

wait_for_ascend_runtime_ready() {
  local max_attempts=${RUNNER_NPU_PREFLIGHT_ATTEMPTS:-3}
  local delay_seconds=${RUNNER_NPU_PREFLIGHT_DELAY_SECONDS:-10}
  local attempt=1

  while [[ "$attempt" -le "$max_attempts" ]]; do
    if run_ascend_root_helper runtime-ready >"$RUNTIME_READY_LOG" 2>&1; then
      return 0
    fi

    cat "$RUNTIME_READY_LOG" >&2

    if runtime_ready_log_indicates_sudo_auth_failure; then
      echo "Ascend runtime sudo fallback is not authorized for helper: $ASCEND_E2E_ROOT_HELPER" >&2
      echo "Grant passwordless sudo for this helper script with SETENV support, or disable ASCEND_E2E_USE_SUDO." >&2
      report_runner_host_fix
      return "$SUDO_AUTH_EXIT_CODE"
    fi

    if [[ "$attempt" -eq "$max_attempts" ]]; then
      return 1
    fi

    echo "Ascend runtime not ready yet; retrying in ${delay_seconds}s (${attempt}/${max_attempts})" >&2
    sleep "$delay_seconds"
    attempt=$((attempt + 1))
  done
}

run_runner_npu_preflight_once() {
  # Use hust-ascend-manager runtime check for the NPU probe (controlled env).
  hust-ascend-manager runtime check \
    --repo "${WORKSPACE_ROOT:-${GITHUB_WORKSPACE:-$PWD}}" \
    --python "$PYTHON_BIN" \
    --require-npu --json >/dev/null || return 1

  # Device-specific torch.zeros allocation check
  "$PYTHON_BIN" - <<'PY'
import os

import torch
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

prepend_env_path() {
  local var_name=$1
  local path_value=$2
  local current_value=${!var_name:-}

  [[ -d "$path_value" ]] || return 0

  case ":$current_value:" in
    *":$path_value:"*) ;;
    *)
      if [[ -n "$current_value" ]]; then
        export "$var_name=$path_value:$current_value"
      else
        export "$var_name=$path_value"
      fi
      ;;
  esac
}

configure_ascend_python_runtime_paths() {
  local runtime_root

  for runtime_root in \
    "${ASCEND_HOME_PATH:-}" \
    "${ASCEND_TOOLKIT_HOME:-}" \
    "${ASCEND_TOOLKIT_LATEST_HOME:-}" \
    /usr/local/Ascend/ascend-toolkit/latest; do
    [[ -n "$runtime_root" ]] || continue
    prepend_env_path PYTHONPATH "$runtime_root/python/site-packages"
    prepend_env_path LD_LIBRARY_PATH "$runtime_root/lib64"
  done

  "$PYTHON_BIN" - <<'PY'
import os

try:
    from acl.rt import memcpy  # noqa: F401
except Exception as exc:
    raise RuntimeError(
        "CANN acl Python bindings are not importable. "
        f"ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH')!r}, "
        f"PYTHONPATH={os.environ.get('PYTHONPATH')!r}"
    ) from exc

print("ascend_acl_python_import=ok")
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
    echo "Skipping regression test for this PR run because the failure is below vLLM and the runner cannot allocate a basic torch_npu tensor." >&2
    return 2
  fi

  return 1
}

start_server() {
  if command -v setsid >/dev/null 2>&1; then
    if [[ "$ASCEND_E2E_USE_SUDO" == "1" ]]; then
      local preserve_list
      local helper_python_bin
      export_sudo_preserved_env_vars
      preserve_list=$(build_sudo_env_preserve_list)
      helper_python_bin=$(prepare_root_helper_python_bin)
      if [[ -n "$preserve_list" ]]; then
        PYTHON_BIN="$helper_python_bin" setsid sudo --preserve-env="$preserve_list" -E -n "$ASCEND_E2E_ROOT_HELPER" serve >"$SERVER_LOG" 2>&1 &
      else
        PYTHON_BIN="$helper_python_bin" setsid sudo -E -n "$ASCEND_E2E_ROOT_HELPER" serve >"$SERVER_LOG" 2>&1 &
      fi
    else
      setsid "${VLLM_OPENAI_SERVER[@]}" \
        --model "$MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --enforce-eager >"$SERVER_LOG" 2>&1 &
    fi
    server_pid=$!
    server_group_pid=$server_pid
  else
    if [[ "$ASCEND_E2E_USE_SUDO" == "1" ]]; then
      run_ascend_root_helper serve >"$SERVER_LOG" 2>&1 &
    else
      "${VLLM_OPENAI_SERVER[@]}" \
        --model "$MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --enforce-eager >"$SERVER_LOG" 2>&1 &
    fi
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

print_server_log_tail() {
  if [[ -f "$SERVER_LOG" ]]; then
    echo "---- vLLM server log tail ----" >&2
    tail -n 300 "$SERVER_LOG" >&2
    echo "---- end vLLM server log tail ----" >&2
  else
    echo "vLLM server log not found: $SERVER_LOG" >&2
  fi
}

curl_with_server_log() {
  local description=$1
  shift
  local max_attempts=${E2E_HTTP_REQUEST_ATTEMPTS:-5}
  local delay_seconds=${E2E_HTTP_REQUEST_DELAY_SECONDS:-2}
  local attempt=1
  local rc=0

  while [[ "$attempt" -le "$max_attempts" ]]; do
    if curl -fsS "$@"; then
      return 0
    fi
    rc=$?

    if [[ -n "$server_pid" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
      echo "$description failed because the vLLM server exited (curl exit $rc)" >&2
      print_server_log_tail
      return "$rc"
    fi

    if [[ "$attempt" -eq "$max_attempts" ]]; then
      echo "$description failed after ${max_attempts} attempts (curl exit $rc)" >&2
      print_server_log_tail
      return "$rc"
    fi

    sleep "$delay_seconds"
    attempt=$((attempt + 1))
  done
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

echo "Starting vLLM inference regression test for $MODEL_NAME"
echo "Using regression test port $PORT"
echo "Ascend E2E use sudo: $ASCEND_E2E_USE_SUDO"
if [[ "$ASCEND_E2E_USE_SUDO" == "1" ]]; then
  echo "Ascend E2E root helper: $ASCEND_E2E_ROOT_HELPER"
fi

if [[ "$ASCEND_E2E_USE_SUDO" == "1" ]]; then
  if ! verify_root_helper_ready; then
    exit 1
  fi
  if wait_for_ascend_runtime_ready; then
    :
  else
    exit "$?"
  fi
else
  configure_ascend_python_runtime_paths
  if ensure_runner_npu_ready; then
    :
  else
    status=$?
    if [[ "$status" -eq 2 ]]; then
      exit 0
    fi
    exit "$status"
  fi
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

curl_with_server_log \
  "vLLM models endpoint readiness confirmation" \
  "http://$HOST:$PORT/v1/models" >/dev/null

completion_response=$(mktemp)
curl_with_server_log "vLLM completions request" \
  "http://$HOST:$PORT/v1/completions" \
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

chat_response=$(mktemp)
curl_with_server_log "vLLM chat completions request" \
  "http://$HOST:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"$CHAT_MESSAGE\"}], \"max_tokens\": $MAX_TOKENS, \"temperature\": 0}" \
  > "$chat_response"

"$PYTHON_BIN" - "$chat_response" "$MODEL_NAME" <<'PY'
import json
import sys

response_path, expected_model = sys.argv[1:3]

with open(response_path, encoding="utf-8") as handle:
    payload = json.load(handle)

assert payload.get("model") == expected_model, payload
choices = payload.get("choices")
assert isinstance(choices, list) and choices, payload
message = choices[0].get("message")
assert isinstance(message, dict), payload
content = message.get("content")
assert isinstance(content, str) and content.strip(), payload
usage = payload.get("usage")
assert isinstance(usage, dict) and usage.get("total_tokens", 0) > 0, payload
PY
rm -f "$chat_response"

"${VLLM_CLI[@]}" bench serve \
  --model "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dataset-name random \
  --random-input-len 32 \
  --random-output-len 4 \
  --num-prompts "$BENCH_NUM_PROMPTS" \
  --endpoint /v1/chat/completions \
  --backend openai-chat

echo "vLLM inference regression test passed"
