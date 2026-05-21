#!/bin/bash
set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-${GITHUB_WORKSPACE:-$PWD}}
VLLM_HUST_REPO=${VLLM_HUST_REPO:-$WORKSPACE_ROOT}
VLLM_HUST_BENCHMARK_REPO=${VLLM_HUST_BENCHMARK_REPO:-$WORKSPACE_ROOT/vllm-hust-benchmark}
VLLM_HUST_WEBSITE_REPO=${VLLM_HUST_WEBSITE_REPO:-$WORKSPACE_ROOT/vllm-hust-website}

RUN_ID=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-$(printf '%s' "${GITHUB_SHA:-local}" | cut -c1-8)}
RESULT_ROOT=${RESULT_ROOT:-$VLLM_HUST_REPO/.benchmarks/ci/$RUN_ID}
RAW_RESULT_FILE=${RAW_RESULT_FILE:-$RESULT_ROOT/raw_benchmark.json}
SUBMISSIONS_ROOT=${SUBMISSIONS_ROOT:-$RESULT_ROOT/submissions}
SUBMISSION_DIR=${SUBMISSION_DIR:-$SUBMISSIONS_ROOT/$RUN_ID}
AGGREGATE_OUTPUT_DIR=${AGGREGATE_OUTPUT_DIR:-$RESULT_ROOT/leaderboard-data}
SERVER_LOG=${SERVER_LOG:-$RESULT_ROOT/server.log}
RUNNER_PREFLIGHT_FAILURE_FILE=${RUNNER_PREFLIGHT_FAILURE_FILE:-$RESULT_ROOT/runner_preflight_failure.txt}
DIAGNOSTICS_DIR=${DIAGNOSTICS_DIR:-$RESULT_ROOT/diagnostics}
NODE_ENV_FAILURE_FILE=${NODE_ENV_FAILURE_FILE:-$RESULT_ROOT/node_env_failure.txt}
BENCH_SCENARIO=${BENCH_SCENARIO:-random-online}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-}
BENCH_CONSTRAINTS_FILE=${BENCH_CONSTRAINTS_FILE:-}
ALLOW_RANDOM_HF_PUBLISH=${ALLOW_RANDOM_HF_PUBLISH:-0}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PARAMETERS=${MODEL_PARAMETERS:-0.5B}
MODEL_PRECISION=${MODEL_PRECISION:-BF16}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-}
DTYPE=${DTYPE:-bfloat16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-256}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-8}
BENCH_RANDOM_INPUT_LEN=${BENCH_RANDOM_INPUT_LEN:-64}
BENCH_RANDOM_OUTPUT_LEN=${BENCH_RANDOM_OUTPUT_LEN:-16}
BENCH_RANDOM_BATCH_SIZE=${BENCH_RANDOM_BATCH_SIZE:-1}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-inf}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-4}
BENCH_INPUT_LEN=${BENCH_INPUT_LEN:-}
BENCH_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-}
HARDWARE_VENDOR=${HARDWARE_VENDOR:-Huawei}
HARDWARE_CHIP_MODEL=${HARDWARE_CHIP_MODEL:-910B3}
CHIP_COUNT=${CHIP_COUNT:-1}
NODE_COUNT=${NODE_COUNT:-1}
PUBLISH_TO_HF=${PUBLISH_TO_HF:-0}
HF_REPO_ID=${HF_REPO_ID:-}

PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || command -v python || true)}
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Could not locate a Python interpreter for benchmark CI" >&2
  exit 127
fi

VLLM_CLI=("$PYTHON_BIN" -m vllm.entrypoints.cli.main)
CURL_BIN=${CURL_BIN:-$(command -v curl || true)}
if [[ -z "$CURL_BIN" ]]; then
  echo "curl is required for benchmark CI readiness checks" >&2
  exit 127
fi

server_pid=""
server_group_pid=""
marker_pid_file=""
marker_pgid_file=""
selected_device=""

NODE_ENV_RETRY_EXIT_CODE=86

is_node_env_failure_text() {
  local text=${1:-}
  printf '%s\n' "$text" | grep -Eq "drvRet=87|drvRetCode=87|ErrCode=507899|error code is 507899|rtGetDeviceCount|Can't get ascend_hal device count|driver error:internal error|Resource_Busy\(EL0005\)|The resources are busy"
}

mark_node_env_failure() {
  local reason=${1:-unknown}
  printf '%s\n' "$reason" > "$NODE_ENV_FAILURE_FILE"
}

collect_ascend_diagnostics() {
  local phase=${1:-unknown}
  local phase_dir="$DIAGNOSTICS_DIR/$phase"
  mkdir -p "$phase_dir"

  {
    echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "phase=$phase"
    echo "run_id=$RUN_ID"
    echo "python_bin=$PYTHON_BIN"
    echo "ascend_home_path=${ASCEND_HOME_PATH:-<unset>}"
    echo "ascend_rt_visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
    echo "ld_library_path=${LD_LIBRARY_PATH:-<unset>}"
  } >"$phase_dir/context.txt"

  env | sort >"$phase_dir/env.txt" 2>/dev/null || true

  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info >"$phase_dir/npu-smi-info.txt" 2>&1 || true
    npu-smi info -m >"$phase_dir/npu-smi-map.txt" 2>&1 || true
  fi

  "$PYTHON_BIN" --version >"$phase_dir/python-version.txt" 2>&1 || true
  "$PYTHON_BIN" -m pip show torch torch-npu >"$phase_dir/pip-torch-stack.txt" 2>&1 || true

  "$PYTHON_BIN" - <<'PY' >"$phase_dir/torch-stack.json" 2>&1 || true
import json

payload = {
    "torch_version": None,
    "torch_npu_module": None,
    "torch_npu_import_ok": False,
    "torch_npu_cann_version": None,
    "npu_available": None,
    "device_count": None,
    "error": None,
}

try:
    import torch
    payload["torch_version"] = getattr(torch, "__version__", None)
    import torch_npu
    payload["torch_npu_import_ok"] = True
    payload["torch_npu_module"] = getattr(torch_npu, "__file__", None)
    try:
        from torch_npu import version as torch_npu_version
        payload["torch_npu_cann_version"] = getattr(torch_npu_version, "cann", None)
    except Exception:
        payload["torch_npu_cann_version"] = None
    try:
        payload["npu_available"] = bool(torch.npu.is_available())
        payload["device_count"] = int(torch.npu.device_count())
    except Exception as exc:
        payload["error"] = repr(exc)
except Exception as exc:
    payload["error"] = repr(exc)

print(json.dumps(payload, ensure_ascii=True, indent=2))
PY

  if command -v hust-ascend-manager >/dev/null 2>&1; then
    hust-ascend-manager runtime check --repo "$VLLM_HUST_REPO" --python "$PYTHON_BIN" --require-npu --json >"$phase_dir/runtime-check.json" 2>&1 || true
    hust-ascend-manager doctor --json >"$phase_dir/doctor.json" 2>&1 || true
  elif command -v hust_ascend_manager_run >/dev/null 2>&1; then
    hust_ascend_manager_run runtime check --repo "$VLLM_HUST_REPO" --python "$PYTHON_BIN" --require-npu --json >"$phase_dir/runtime-check.json" 2>&1 || true
    hust_ascend_manager_run doctor --json >"$phase_dir/doctor.json" 2>&1 || true
  fi

  if [[ -f "$RUNNER_PREFLIGHT_FAILURE_FILE" ]]; then
    cp "$RUNNER_PREFLIGHT_FAILURE_FILE" "$phase_dir/runner-preflight-failure.txt" || true
  fi
  if [[ -f "$SERVER_LOG" ]]; then
    cp "$SERVER_LOG" "$phase_dir/server.log" || true
    tail -n 300 "$SERVER_LOG" >"$phase_dir/server.tail.log" || true
  fi
}

enforce_single_runtime_source() {
  "$PYTHON_BIN" - <<'PY'
import os
import pathlib
import site
import sys

import torch
import torch_npu

torch_npu_file = pathlib.Path(torch_npu.__file__).resolve()
site_roots = [pathlib.Path(p).resolve() for p in site.getsitepackages() if p]
user_site = site.getusersitepackages()
if user_site:
    site_roots.append(pathlib.Path(user_site).resolve())

if not any(root in torch_npu_file.parents for root in site_roots):
    raise RuntimeError(
        f"torch_npu module path is outside active site-packages: {torch_npu_file}"
    )

ascend_home = os.environ.get("ASCEND_HOME_PATH")
if not ascend_home:
    raise RuntimeError("ASCEND_HOME_PATH is not set")
ascend_home_path = pathlib.Path(ascend_home).resolve()
if not ascend_home_path.exists():
    raise RuntimeError(f"ASCEND_HOME_PATH does not exist: {ascend_home_path}")

required_lib = ascend_home_path / "lib64" / "libascendcl.so"
if not required_lib.exists():
    raise RuntimeError(
        f"ASCEND_HOME_PATH does not contain libascendcl.so: {required_lib}"
    )

ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
if str(ascend_home_path / "lib64") not in ld_library_path:
    raise RuntimeError(
        "LD_LIBRARY_PATH does not include ASCEND_HOME_PATH/lib64"
    )

device_count = int(torch.npu.device_count())
if device_count <= 0:
    raise RuntimeError("torch.npu.device_count() returned 0 during runtime-source check")

print("runtime_source_check=ok")
print(f"torch_npu_module={torch_npu_file}")
print(f"ascend_home_path={ascend_home_path}")
PY
}

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
import sys

import torch

if importlib.util.find_spec("torch_npu") is None:
    raise RuntimeError("torch_npu is not installed in the benchmark environment")

import torch_npu  # noqa: F401

preferred_device = os.environ.get("VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE", "npu:0")
preferred_index = 0
if ":" in preferred_device:
  try:
    preferred_index = int(preferred_device.rsplit(":", 1)[1])
  except ValueError:
    preferred_index = 0

device_count = int(torch.npu.device_count())
if device_count <= 0:
  raise RuntimeError("torch.npu.device_count() returned 0")

candidate_devices = [
  f"npu:{(preferred_index + offset) % device_count}"
  for offset in range(device_count)
]

print("torch_npu import ok=True")
for device in candidate_devices:
  print(f"preflight device={device}")
  try:
    torch.npu.set_device(device)
    _ = torch.zeros(1, device=device)
  except Exception as exc:  # noqa: BLE001
    print(f"preflight failed on {device}: {exc}", file=sys.stderr)
    continue

  print(f"selected_device={device}")
  print("torch.zeros preflight ok")
  break
else:
  raise RuntimeError(
    "torch.npu basic allocation failed on every visible device"
  )
PY
}

ensure_runner_npu_ready() {
  local max_attempts=${RUNNER_NPU_PREFLIGHT_ATTEMPTS:-3}
  local delay_seconds=${RUNNER_NPU_PREFLIGHT_DELAY_SECONDS:-10}
  local attempt=1
  local preflight_output=""

  while [[ "$attempt" -le "$max_attempts" ]]; do
    if preflight_output=$(run_runner_npu_preflight_once 2>&1); then
      selected_device=$(printf '%s\n' "$preflight_output" | awk -F= '/^selected_device=/{print $2; exit}')
      if [[ -n "$selected_device" ]]; then
        export ASCEND_RT_VISIBLE_DEVICES="${selected_device#npu:}"
        echo "Selected Ascend device: $selected_device"
        echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
      fi
      rm -f "$RUNNER_PREFLIGHT_FAILURE_FILE"
      return 0
    fi

    printf '%s\n' "$preflight_output" > "$RUNNER_PREFLIGHT_FAILURE_FILE"
    echo "Runner NPU preflight failed ($attempt/$max_attempts)" >&2
    cat "$RUNNER_PREFLIGHT_FAILURE_FILE" >&2

    if [[ "$attempt" -lt "$max_attempts" ]]; then
      sleep "$delay_seconds"
    fi
    attempt=$((attempt + 1))
  done

  collect_ascend_diagnostics "preflight-failure"
  if is_node_env_failure_text "$preflight_output"; then
    mark_node_env_failure "runner preflight failed with Ascend driver/runtime node-level error"
    echo "Detected Ascend node-level runtime failure (87/507899)." >&2
    return "$NODE_ENV_RETRY_EXIT_CODE"
  fi

  echo "Self-hosted runner NPU runtime is unhealthy before vLLM startup." >&2
  echo "All visible Ascend devices failed the basic torch_npu allocation check." >&2
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

mkdir -p "$RESULT_ROOT" "$SUBMISSIONS_ROOT" "$AGGREGATE_OUTPUT_DIR"

echo "== Ascend benchmark CI =="
echo "workspace root: $WORKSPACE_ROOT"
echo "run id: $RUN_ID"
echo "result root: $RESULT_ROOT"
echo "benchmark port: $PORT"
echo "benchmark scenario: $BENCH_SCENARIO"
echo "publish to hf: $PUBLISH_TO_HF"

case "$BENCH_SCENARIO" in
  random-online)
    EFFECTIVE_DATASET_NAME="random"
    EFFECTIVE_DATASET_PATH=""
    EFFECTIVE_INPUT_LEN=${BENCH_INPUT_LEN:-$BENCH_RANDOM_INPUT_LEN}
    EFFECTIVE_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-$BENCH_RANDOM_OUTPUT_LEN}
    EFFECTIVE_CONSTRAINTS_FILE=${BENCH_CONSTRAINTS_FILE:-$VLLM_HUST_REPO/.github/workflows/data/random-online-ci-constraints.json}
    bench_args=(
      --backend vllm
      --endpoint /v1/completions
      --dataset-name random
      --random-input-len "$BENCH_RANDOM_INPUT_LEN"
      --random-output-len "$BENCH_RANDOM_OUTPUT_LEN"
      --random-batch-size "$BENCH_RANDOM_BATCH_SIZE"
      --num-prompts "$BENCH_NUM_PROMPTS"
      --request-rate "$BENCH_REQUEST_RATE"
      --max-concurrency "$BENCH_MAX_CONCURRENCY"
    )
    ;;
  sharegpt-online)
    if [[ -z "$BENCH_DATASET_PATH" ]]; then
      echo "BENCH_DATASET_PATH is required for sharegpt-online" >&2
      exit 2
    fi
    if [[ -z "$BENCH_CONSTRAINTS_FILE" ]]; then
      echo "BENCH_CONSTRAINTS_FILE is required for sharegpt-online" >&2
      exit 2
    fi
    EFFECTIVE_DATASET_NAME="sharegpt"
    EFFECTIVE_DATASET_PATH="$BENCH_DATASET_PATH"
    EFFECTIVE_INPUT_LEN=${BENCH_INPUT_LEN:-1024}
    EFFECTIVE_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-256}
    EFFECTIVE_CONSTRAINTS_FILE="$BENCH_CONSTRAINTS_FILE"
    bench_args=(
      --backend vllm
      --endpoint /v1/completions
      --dataset-name sharegpt
      --dataset-path "$BENCH_DATASET_PATH"
      --num-prompts "$BENCH_NUM_PROMPTS"
      --request-rate "$BENCH_REQUEST_RATE"
      --max-concurrency "$BENCH_MAX_CONCURRENCY"
    )
    ;;
  *)
    echo "Unsupported BENCH_SCENARIO: $BENCH_SCENARIO" >&2
    exit 2
    ;;
esac

if [[ "$PUBLISH_TO_HF" == "1" && "$BENCH_SCENARIO" == "random-online" && "$ALLOW_RANDOM_HF_PUBLISH" != "1" ]]; then
  echo "Refusing to publish random-online CI preview to HF without ALLOW_RANDOM_HF_PUBLISH=1" >&2
  exit 2
fi

if [[ ! -f "$EFFECTIVE_CONSTRAINTS_FILE" ]]; then
  echo "constraints file not found: $EFFECTIVE_CONSTRAINTS_FILE" >&2
  exit 2
fi

if ! enforce_single_runtime_source; then
  collect_ascend_diagnostics "runtime-source-check-failure"
  mark_node_env_failure "runtime source mismatch between torch_npu and toolkit"
  echo "Ascend runtime source check failed." >&2
  exit "$NODE_ENV_RETRY_EXIT_CODE"
fi

if ! ensure_runner_npu_ready; then
  status=$?
  if [[ "$status" -eq 2 ]]; then
    exit 0
  fi
  exit "$status"
fi

start_server

for attempt in $(seq 1 120); do
  if "$CURL_BIN" -fsS "http://$HOST:$PORT/v1/models" >/dev/null; then
    break
  fi

  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM server exited before becoming ready"
    cat "$SERVER_LOG"
    collect_ascend_diagnostics "server-exit-before-ready"
    if is_node_env_failure_text "$(cat "$SERVER_LOG" 2>/dev/null || true)"; then
      mark_node_env_failure "vllm server exited with Ascend node-level runtime error before ready"
      exit "$NODE_ENV_RETRY_EXIT_CODE"
    fi
    exit 1
  fi

  if [[ "$attempt" -eq 120 ]]; then
    echo "Timed out waiting for vLLM server to become ready"
    cat "$SERVER_LOG"
    collect_ascend_diagnostics "server-ready-timeout"
    if is_node_env_failure_text "$(cat "$SERVER_LOG" 2>/dev/null || true)"; then
      mark_node_env_failure "vllm server readiness timeout with Ascend node-level runtime errors"
      exit "$NODE_ENV_RETRY_EXIT_CODE"
    fi
    exit 1
  fi

  sleep 2
done

"${VLLM_CLI[@]}" bench serve \
  --model "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  "${bench_args[@]}" \
  --save-result \
  --result-dir "$RESULT_ROOT" \
  --result-filename "$(basename "$RAW_RESULT_FILE")"

if [[ ! -f "$RAW_RESULT_FILE" ]]; then
  collect_ascend_diagnostics "benchmark-result-missing"
  if [[ -f "$SERVER_LOG" ]] && is_node_env_failure_text "$(cat "$SERVER_LOG" 2>/dev/null || true)"; then
    mark_node_env_failure "benchmark failed due to Ascend node-level runtime errors"
    exit "$NODE_ENV_RETRY_EXIT_CODE"
  fi
fi

CORE_VERSION=$("$PYTHON_BIN" - <<'PY'
import vllm
print(vllm.__version__)
PY
)

DISPLAY_VERSION=$(printf '%s' "${GITHUB_SHA:-local}" | cut -c1-8)

"$PYTHON_BIN" -m vllm_hust_benchmark.cli submit \
  "$BENCH_SCENARIO" \
  --benchmark-result-file "$RAW_RESULT_FILE" \
  --constraints-file "$EFFECTIVE_CONSTRAINTS_FILE" \
  --run-id "$RUN_ID" \
  --engine vllm-hust \
  --engine-version "$DISPLAY_VERSION" \
  --model-name "$MODEL_NAME" \
  --model-parameters "$MODEL_PARAMETERS" \
  --model-precision "$MODEL_PRECISION" \
  --hardware-vendor "$HARDWARE_VENDOR" \
  --hardware-chip-model "$HARDWARE_CHIP_MODEL" \
  --chip-count "$CHIP_COUNT" \
  --node-count "$NODE_COUNT" \
  --submitter "${GITHUB_ACTOR:-ci}" \
  --data-source "vllm-hust-ci-$BENCH_SCENARIO" \
  --input-length "$EFFECTIVE_INPUT_LEN" \
  --output-length "$EFFECTIVE_OUTPUT_LEN" \
  --concurrent-requests "$BENCH_MAX_CONCURRENCY" \
  --core-version "$CORE_VERSION" \
  --submissions-dir "$SUBMISSIONS_ROOT"

if [[ "$PUBLISH_TO_HF" == "1" ]]; then
  if [[ -z "$HF_REPO_ID" ]]; then
    echo "HF_REPO_ID must be set when PUBLISH_TO_HF=1" >&2
    exit 2
  fi

  "$PYTHON_BIN" -m vllm_hust_benchmark.cli sync-submission-to-hf \
    --submission-dir "$SUBMISSION_DIR" \
    --aggregate-output-dir "$AGGREGATE_OUTPUT_DIR" \
    --repo-id "$HF_REPO_ID" \
    --submissions-prefix submissions-auto \
    --commit-message "chore: sync vllm-hust benchmark $RUN_ID (${GITHUB_REF_NAME:-detached}@$(printf '%s' "${GITHUB_SHA:-local}" | cut -c1-8))" \
    --execute
else
  "$PYTHON_BIN" -m vllm_hust_benchmark.cli publish-website \
    --source-dir "$SUBMISSIONS_ROOT" \
    --output-dir "$AGGREGATE_OUTPUT_DIR" \
    --execute
fi

echo "RUN_ID=$RUN_ID"
echo "RAW_RESULT_FILE=$RAW_RESULT_FILE"
echo "SUBMISSION_DIR=$SUBMISSION_DIR"
echo "AGGREGATE_OUTPUT_DIR=$AGGREGATE_OUTPUT_DIR"
echo "SERVER_LOG=$SERVER_LOG"
echo "BENCH_SCENARIO=$BENCH_SCENARIO"
echo "EFFECTIVE_CONSTRAINTS_FILE=$EFFECTIVE_CONSTRAINTS_FILE"
echo "EFFECTIVE_CONSTRAINTS_FILE=$EFFECTIVE_CONSTRAINTS_FILE"