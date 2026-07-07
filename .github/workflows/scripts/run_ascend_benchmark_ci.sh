#!/bin/bash
set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-${GITHUB_WORKSPACE:-$PWD}}
VLLM_HUST_REPO=${VLLM_HUST_REPO:-$WORKSPACE_ROOT}
VLLM_ASCEND_HUST_REPO=${VLLM_ASCEND_HUST_REPO:-$WORKSPACE_ROOT/vllm-ascend-hust}
VLLM_HUST_BENCHMARK_REPO=${VLLM_HUST_BENCHMARK_REPO:-$WORKSPACE_ROOT/vllm-hust-benchmark}
VLLM_HUST_WEBSITE_REPO=${VLLM_HUST_WEBSITE_REPO:-$WORKSPACE_ROOT/vllm-hust-website}

RUN_ID=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-$(printf '%s' "${TARGET_REPO_SHA:-${GITHUB_SHA:-local}}" | cut -c1-8)}
RESULT_ROOT=${RESULT_ROOT:-$VLLM_HUST_REPO/.benchmarks/ci/$RUN_ID}
RAW_RESULT_FILE=${RAW_RESULT_FILE:-$RESULT_ROOT/raw_benchmark.json}
SUBMISSIONS_ROOT=${SUBMISSIONS_ROOT:-$RESULT_ROOT/submissions}
SUBMISSION_DIR=${SUBMISSION_DIR:-$SUBMISSIONS_ROOT/$RUN_ID}
AGGREGATE_OUTPUT_DIR=${AGGREGATE_OUTPUT_DIR:-$RESULT_ROOT/leaderboard-data}
BENCHMARK_PUBLICATION_SYNC_SCRIPT=${BENCHMARK_PUBLICATION_SYNC_SCRIPT:-$VLLM_HUST_REPO/.github/workflows/scripts/sync_benchmark_snapshots_to_github.sh}
SERVER_LOG=${SERVER_LOG:-$RESULT_ROOT/server.log}
RUNNER_PREFLIGHT_FAILURE_FILE=${RUNNER_PREFLIGHT_FAILURE_FILE:-$RESULT_ROOT/runner_preflight_failure.txt}
DIAGNOSTICS_DIR=${DIAGNOSTICS_DIR:-$RESULT_ROOT/diagnostics}
NODE_ENV_FAILURE_FILE=${NODE_ENV_FAILURE_FILE:-$RESULT_ROOT/node_env_failure.txt}
BENCH_SCENARIO=${BENCH_SCENARIO:-random-online}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-}
BENCH_CONSTRAINTS_FILE=${BENCH_CONSTRAINTS_FILE:-}
SAME_SPEC_BENCHMARK_ENABLED=${SAME_SPEC_BENCHMARK_ENABLED:-1}
SAME_SPEC_SPEC_FILE=${SAME_SPEC_SPEC_FILE:-$VLLM_HUST_BENCHMARK_REPO/docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json}
SAME_SPEC_CONSTRAINTS_FILE=${SAME_SPEC_CONSTRAINTS_FILE:-$VLLM_HUST_BENCHMARK_REPO/docs/official-baselines/official-ascend-constraints.stub.json}
SAME_SPEC_READY_TIMEOUT_SECONDS=${SAME_SPEC_READY_TIMEOUT_SECONDS:-600}
SAME_SPEC_PR_PREVIEW_COMPAT=${SAME_SPEC_PR_PREVIEW_COMPAT:-1}
ALLOW_RANDOM_HF_PUBLISH=${ALLOW_RANDOM_HF_PUBLISH:-0}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}
MODEL_PARAMETERS=${MODEL_PARAMETERS:-14B}
MODEL_PRECISION=${MODEL_PRECISION:-FP16}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-}
DTYPE=${DTYPE:-float16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-200}
BENCH_RANDOM_INPUT_LEN=${BENCH_RANDOM_INPUT_LEN:-1024}
BENCH_RANDOM_OUTPUT_LEN=${BENCH_RANDOM_OUTPUT_LEN:-256}
BENCH_RANDOM_BATCH_SIZE=${BENCH_RANDOM_BATCH_SIZE:-1}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-1}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-4}
BENCH_INPUT_LEN=${BENCH_INPUT_LEN:-}
BENCH_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-}
HARDWARE_VENDOR=${HARDWARE_VENDOR:-Huawei}
HARDWARE_CHIP_MODEL=${HARDWARE_CHIP_MODEL:-}
CHIP_COUNT=${CHIP_COUNT:-1}
NODE_COUNT=${NODE_COUNT:-1}
PUBLISH_TO_HF=${PUBLISH_TO_HF:-0}
PUBLISH_TO_BENCHMARK_REPO=${PUBLISH_TO_BENCHMARK_REPO:-0}
HF_REPO_ID=${HF_REPO_ID:-}
RUNTIME_READY_LOG=${RUNTIME_READY_LOG:-$RESULT_ROOT/runtime-ready.log}
SUDO_AUTH_EXIT_CODE=${SUDO_AUTH_EXIT_CODE:-76}
ASCEND_BENCHMARK_USE_SUDO=${ASCEND_BENCHMARK_USE_SUDO:-0}
DEFAULT_SYSTEM_ASCEND_BENCHMARK_ROOT_HELPER=${DEFAULT_SYSTEM_ASCEND_BENCHMARK_ROOT_HELPER:-/usr/local/bin/run_ascend_benchmark_root_helper.sh}
REPO_ASCEND_BENCHMARK_ROOT_HELPER=${REPO_ASCEND_BENCHMARK_ROOT_HELPER:-$VLLM_ASCEND_HUST_REPO/.github/workflows/scripts/run_ascend_benchmark_root_helper.sh}
REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT=${REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT:-$VLLM_ASCEND_HUST_REPO/scripts/install_ascend_benchmark_root_helper.sh}
ASCEND_BENCHMARK_ROOT_HELPER=${ASCEND_BENCHMARK_ROOT_HELPER:-$DEFAULT_SYSTEM_ASCEND_BENCHMARK_ROOT_HELPER}

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
NPU_SMI_BIN=${NPU_SMI_BIN:-$(command -v npu-smi || true)}

USER_PROVIDED_ASCEND_VISIBLE_DEVICES=0
if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" || -n "${ASCEND_VISIBLE_DEVICES:-}" ]]; then
  USER_PROVIDED_ASCEND_VISIBLE_DEVICES=1
fi

NODE_ENV_RETRY_EXIT_CODE=86
INVALID_BENCHMARK_RESULT_EXIT_CODE=${INVALID_BENCHMARK_RESULT_EXIT_CODE:-77}

is_node_env_failure_text() {
  local text=${1:-}
  printf '%s\n' "$text" | grep -Eq "DrvMngGetConsoleLogLevel failed|dcmi model initialized failed|ret is -8020|drvRet=87|drvRetCode=87|ErrCode=507899|error code is 507899|rtGetDeviceCount|Can't get ascend_hal device count|driver error:internal error|Resource_Busy\(EL0005\)|The resources are busy"
}

runtime_ready_log_indicates_sudo_auth_failure() {
  [[ -f "$RUNTIME_READY_LOG" ]] && grep -qE 'sudo: (a password is required|a terminal is required|sorry, you must have a tty|is not allowed to execute)' "$RUNTIME_READY_LOG"
}

select_ascend_device() {
  ASCEND_DEVICE_SELECTION_ATTEMPT="${1:-1}" NPU_SMI_BIN="$NPU_SMI_BIN" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def parse_logical_map(mapping_output: str) -> dict[tuple[str, str], int]:
  logical_map = {}
  for line in mapping_output.splitlines():
    parts = line.split()
    if len(parts) < 3:
      continue
    npu_id, chip_id, logical_id = parts[:3]
    if npu_id.isdigit() and chip_id.isdigit() and logical_id.isdigit():
      logical_map[(npu_id, chip_id)] = int(logical_id)
  return logical_map


def list_status_devices(info_output: str) -> list[int]:
  status_devices = set()
  for raw_line in info_output.splitlines():
    line = raw_line.strip()
    if not line.startswith("|"):
      continue

    parts = [part.strip() for part in line.strip("|").split("|")]
    if len(parts) < 2:
      continue

    left_column = parts[0].split()
    if len(left_column) >= 2 and left_column[0].isdigit() and parts[1] and ":" not in parts[1]:
      status_devices.add(int(left_column[0]))

  return sorted(status_devices)


def list_devnode_devices() -> list[int]:
  devnode_devices = set()
  for device_path in Path("/dev").glob("davinci[0-9]*"):
    suffix = device_path.name.removeprefix("davinci")
    if suffix.isdigit():
      devnode_devices.add(int(suffix))
  return sorted(devnode_devices)


def run_npu_smi(*args: str) -> subprocess.CompletedProcess[str] | None:
  npu_smi_bin = os.environ.get("NPU_SMI_BIN")
  if not npu_smi_bin:
    return None

  clean_env = {
    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    "HOME": os.environ.get("HOME", ""),
    "LANG": os.environ.get("LANG", "C.UTF-8"),
    "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
  }

  try:
    result = subprocess.run(
      [npu_smi_bin, *args],
      check=False,
      capture_output=True,
      text=True,
      timeout=5,
      env=clean_env,
    )
    if result.returncode == 0 or os.geteuid() == 0:
      return result

    sudo_bin = shutil.which("sudo", path=clean_env["PATH"])
    if not sudo_bin:
      return result

    sudo_result = subprocess.run(
      [sudo_bin, "-n", npu_smi_bin, *args],
      check=False,
      capture_output=True,
      text=True,
      timeout=5,
      env=clean_env,
    )
    if sudo_result.returncode == 0:
      return sudo_result
    return result
  except Exception as exc:  # noqa: BLE001
    print(f"npu-smi {' '.join(args)} failed: {exc}", file=sys.stderr)
    return None


def select_best_idle_device(info_output: str, logical_map: dict[tuple[str, str], int]) -> tuple[int, str] | None:
  hbm_usage_pattern = re.compile(r"(\d+)\s*/\s*(\d+)\s*$")
  device_stats = []
  current_npu_id = None
  current_health = None

  for raw_line in info_output.splitlines():
    line = raw_line.strip()
    if not line.startswith("|"):
      continue

    parts = [part.strip() for part in line.strip("|").split("|")]
    if len(parts) < 3:
      continue

    left_column = parts[0].split()
    if len(left_column) >= 2 and left_column[0].isdigit() and parts[1] and ":" not in parts[1]:
      current_npu_id = left_column[0]
      current_health = parts[1]
      continue

    if current_npu_id is None or current_health != "OK":
      continue

    if len(left_column) != 1 or not left_column[0].isdigit() or ":" not in parts[1]:
      continue

    chip_id = left_column[0]
    logical_id = logical_map.get((current_npu_id, chip_id))
    device_source = "idle"
    if logical_id is None:
      if chip_id != "0":
        continue
      logical_id = int(current_npu_id)
      device_source = "status-fallback"

    hbm_match = hbm_usage_pattern.search(parts[2])
    if hbm_match is None:
      continue

    used_memory_mb = int(hbm_match.group(1))
    total_memory_mb = int(hbm_match.group(2))
    free_memory_mb = max(0, total_memory_mb - used_memory_mb)
    device_stats.append((logical_id, free_memory_mb, device_source))

  if not device_stats:
    return None

  device_stats.sort(key=lambda item: (-item[1], item[0], item[2]))
  selected_device, _, selected_source = device_stats[0]
  return selected_device, selected_source


mapping_result = run_npu_smi("info", "-m")
logical_map = {}
if mapping_result is not None and mapping_result.returncode == 0:
  logical_map = parse_logical_map(mapping_result.stdout)

selection_attempt = max(1, int(os.environ.get("ASCEND_DEVICE_SELECTION_ATTEMPT", "1")))
info_result = run_npu_smi("info")
if info_result is not None and info_result.returncode == 0:
  selected_device = select_best_idle_device(info_result.stdout, logical_map)
  if selected_device is not None:
    device_id, device_source = selected_device
    print(f"{device_id}\t{device_source}")
    raise SystemExit(0)

  status_devices = list_status_devices(info_result.stdout)
  if status_devices:
    fallback_device = status_devices[(selection_attempt - 1) % len(status_devices)]
    print(f"{fallback_device}\tstatus-round-robin")
    raise SystemExit(0)

devnode_devices = list_devnode_devices()
if devnode_devices:
  fallback_device = devnode_devices[(selection_attempt - 1) % len(devnode_devices)]
  print(f"{fallback_device}\tdevnode-round-robin")
PY
}

configure_single_card_ascend_device() {
  local selection_attempt="${1:-1}"
  local selected_device_info=""
  local visible_devices_display="${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-<unset>}}"

  if [[ "$USER_PROVIDED_ASCEND_VISIBLE_DEVICES" == "1" ]]; then
  export VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE="${VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE:-npu:0}"
  echo "using explicit Ascend visible devices from environment: $visible_devices_display"
  echo "skipping automatic single-card Ascend device selection"
  return 0
  fi

  selected_device_info="$(select_ascend_device "$selection_attempt")"
  if [[ -n "$selected_device_info" ]]; then
  IFS=$'\t' read -r SELECTED_ASCEND_DEVICE SELECTED_ASCEND_DEVICE_SOURCE <<<"$selected_device_info"
  export ASCEND_RT_VISIBLE_DEVICES="$SELECTED_ASCEND_DEVICE"
  export VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE="npu:0"
  echo "selected single-card Ascend device: $ASCEND_RT_VISIBLE_DEVICES (${SELECTED_ASCEND_DEVICE_SOURCE})"
  else
  unset ASCEND_RT_VISIBLE_DEVICES
  unset VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE
  echo "Could not resolve a single-card Ascend device; probing runtime without device scoping"
  fi
}

SUDO_PRESERVE_ENV_VARS=(
  ASCEND_AICPU_PATH
  ASCEND_BENCHMARK_USE_SUDO
  ASCEND_HOME_PATH
  ASCEND_OPP_PATH
  ASCEND_RT_VISIBLE_DEVICES
  ASCEND_TOOLKIT_HOME
  ASCEND_TOOLKIT_LATEST_HOME
  ASCEND_VISIBLE_DEVICES
  ATB_HOME_PATH
  BENCH_CONSTRAINTS_FILE
  BENCH_DATASET_PATH
  BENCH_INPUT_LEN
  BENCH_MAX_CONCURRENCY
  BENCH_NUM_PROMPTS
  BENCH_OUTPUT_LEN
  BENCH_RANDOM_BATCH_SIZE
  BENCH_RANDOM_INPUT_LEN
  BENCH_RANDOM_OUTPUT_LEN
  BENCH_REQUEST_RATE
  BENCH_SCENARIO
  CHIP_COUNT
  CONSTRAINTS_FILE
  CURRENT_CLIENT_PORT
  CURRENT_DATA_SOURCE
  CURRENT_ENGINE_VERSION
  CURRENT_GIT_COMMIT
  CURRENT_GITHUB_REF
  CURRENT_GITHUB_REPOSITORY
  CURRENT_PLUGIN_ENGINE
  CURRENT_PLUGIN_GIT_COMMIT
  CURRENT_PLUGIN_GITHUB_REF
  CURRENT_PLUGIN_GITHUB_REPOSITORY
  CURRENT_RUNTIME_CWD
  CURRENT_RUNTIME_PYTHON
  CURRENT_SERVER_PORT
  CURRENT_SUBMITTER
  CURRENT_VLLM_ASCEND_HUST_REPO
  CURRENT_VLLM_CACHE_ROOT
  CURRENT_VLLM_HUST_REPO
  DTYPE
  GITHUB_ACTOR
  GITHUB_EVENT_NAME
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
  MODEL_PARAMETERS
  MODEL_PRECISION
  NODE_COUNT
  PATH
  PIP_CACHE_DIR
  PORT
  PUBLISH_TO_BENCHMARK_REPO
  PYTHON_BIN
  PYTHONPATH
  READY_TIMEOUT_SECONDS
  RESULT_DIR
  RESULT_ROOT
  RUN_ID
  SAME_SPEC_BENCHMARK_ENABLED
  SAME_SPEC_CONSTRAINTS_FILE
  SAME_SPEC_SPEC_FILE
  TMPDIR
  TRANSFORMERS_CACHE
  VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE
  VLLM_CACHE_ROOT
  VLLM_HUST_BENCHMARK_REPO
  VLLM_CONFIG_ROOT
  VLLM_ASCEND_HUST_REPO
  VLLM_HUST_REPO
  VLLM_HUST_WORKSPACE_ROOT
  WORKSPACE_ROOT
  XDG_CACHE_HOME
  XDG_CONFIG_HOME
)

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
  printf 'sudo RUNNER_USER=grunner bash %s\n' "$REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT"
}

report_runner_host_fix() {
  if [[ -f "$REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT" ]]; then
    echo "Runner host fix: $(benchmark_root_helper_fix_command)" >&2
  else
    echo "Runner-local install script missing from checkout: $REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT" >&2
  fi
}

verify_root_helper_ready() {
  if [[ "$ASCEND_BENCHMARK_USE_SUDO" != "1" ]]; then
    return 0
  fi

  if [[ ! -f "$REPO_ASCEND_BENCHMARK_ROOT_HELPER" ]]; then
    echo "Runner-local benchmark root helper source missing from checkout: $REPO_ASCEND_BENCHMARK_ROOT_HELPER" >&2
    return 1
  fi

  if [[ ! -f "$REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT" ]]; then
    echo "Runner-local benchmark helper install script missing from checkout: $REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT" >&2
    return 1
  fi

  if [[ "$ASCEND_BENCHMARK_ROOT_HELPER" != "$DEFAULT_SYSTEM_ASCEND_BENCHMARK_ROOT_HELPER" ]]; then
    if [[ ! -x "$ASCEND_BENCHMARK_ROOT_HELPER" ]]; then
      echo "Configured Ascend benchmark root helper is not executable: $ASCEND_BENCHMARK_ROOT_HELPER" >&2
      return 1
    fi
    return 0
  fi

  if [[ ! -x "$ASCEND_BENCHMARK_ROOT_HELPER" ]]; then
    echo "Installed Ascend benchmark root helper is missing or not executable: $ASCEND_BENCHMARK_ROOT_HELPER" >&2
    report_runner_host_fix
    return 1
  fi

  if ! cmp -s "$ASCEND_BENCHMARK_ROOT_HELPER" "$REPO_ASCEND_BENCHMARK_ROOT_HELPER"; then
    echo "Installed Ascend benchmark root helper is stale: $ASCEND_BENCHMARK_ROOT_HELPER does not match $REPO_ASCEND_BENCHMARK_ROOT_HELPER" >&2
    report_runner_host_fix
    return 1
  fi
}

run_ascend_root_helper() {
  local preserve_list
  local helper_python_bin

  if [[ "$ASCEND_BENCHMARK_USE_SUDO" != "1" ]]; then
    echo "run_ascend_root_helper requires ASCEND_BENCHMARK_USE_SUDO=1" >&2
    return 2
  fi

  export_sudo_preserved_env_vars
  preserve_list=$(build_sudo_env_preserve_list)
  helper_python_bin=$(prepare_root_helper_python_bin)

  if [[ -n "$preserve_list" ]]; then
    PYTHON_BIN="$helper_python_bin" sudo --preserve-env="$preserve_list" -E -n "$ASCEND_BENCHMARK_ROOT_HELPER" "$@"
  else
    PYTHON_BIN="$helper_python_bin" sudo -E -n "$ASCEND_BENCHMARK_ROOT_HELPER" "$@"
  fi
}

run_with_same_spec_stderr_filter() {
  local summary_interval=${SAME_SPEC_WAIT_LOG_SUMMARY_INTERVAL:-30}

  if (( summary_interval <= 0 )); then
    summary_interval=30
  fi

  "$@" 2> >(
    suppressed=0
    while IFS= read -r line; do
      if [[ "$line" == curl:\ \(7\)\ Failed\ to\ connect\ to\ *"Couldn't connect to server" ]]; then
        suppressed=$((suppressed + 1))
        if (( suppressed == 1 || suppressed % summary_interval == 0 )); then
          echo "[same-spec-current] current same-spec server is still starting (${suppressed} connection-refused readiness probes suppressed)" >&2
        fi
        continue
      fi

      if (( suppressed > 0 )); then
        echo "[same-spec-current] suppressed ${suppressed} connection-refused readiness probes while waiting for current same-spec startup" >&2
        suppressed=0
      fi

      printf '%s\n' "$line" >&2
    done

    if (( suppressed > 0 )); then
      echo "[same-spec-current] suppressed ${suppressed} connection-refused readiness probes while waiting for current same-spec startup" >&2
    fi
  )
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
      echo "Ascend runtime sudo fallback is not authorized for helper: $ASCEND_BENCHMARK_ROOT_HELPER" >&2
      echo "Grant passwordless sudo for this helper script with SETENV support, or disable ASCEND_BENCHMARK_USE_SUDO." >&2
      report_runner_host_fix
      return "$SUDO_AUTH_EXIT_CODE"
    fi

    if [[ "$attempt" -eq "$max_attempts" ]]; then
      if is_node_env_failure_text "$(cat "$RUNTIME_READY_LOG" 2>/dev/null || true)"; then
        mark_node_env_failure "runtime-ready helper failed with Ascend node-level error"
        return "$NODE_ENV_RETRY_EXIT_CODE"
      fi
      return 1
    fi

    echo "Ascend runtime not ready yet; retrying in ${delay_seconds}s (${attempt}/${max_attempts})" >&2
    sleep "$delay_seconds"
    attempt=$((attempt + 1))
  done
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

enforce_single_runtime_source_environment() {
  "$PYTHON_BIN" - <<'PY'
import importlib.util
import os
import pathlib
import site
import sys

torch_npu_spec = importlib.util.find_spec("torch_npu")
if torch_npu_spec is None or torch_npu_spec.origin is None:
    raise RuntimeError("torch_npu is not importable in the benchmark environment")

torch_npu_file = pathlib.Path(torch_npu_spec.origin).resolve()
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

print("runtime_source_environment_check=ok")
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
  # Use hust-ascend-manager runtime check for the NPU probe. This runs the
  # torch_npu probe in a controlled subprocess environment (build_env_dict()
  # exports + PYTHONNOUSERSITE=1) which prevents conda library shadowing and
  # other environment contamination that cause 'path string is NULL' errors.
  if ! command -v hust-ascend-manager >/dev/null 2>&1; then
    echo "[preflight] hust-ascend-manager not found in PATH" >&2
    return 127
  fi

  local manager_output
  local manager_rc
  manager_output="$(hust-ascend-manager runtime check \
    --repo "$VLLM_HUST_REPO" \
    --python "$PYTHON_BIN" \
    --require-npu --json 2>&1)"
  manager_rc=$?
  echo "$manager_output"
  if [[ "$manager_rc" -ne 0 ]]; then
    echo "[preflight] hust-ascend-manager runtime check failed (exit $manager_rc)" >&2
    return "$manager_rc"
  fi

  # Verify device_count and allocation result from the JSON output.
  # The runtime check prints pretty JSON, so parse stdin as a full JSON document.
  local selected_device
  selected_device="$(MANAGER_OUTPUT="$manager_output" "$PYTHON_BIN" - <<'PY'
import json
import os
import sys

try:
    data = json.loads(os.environ["MANAGER_OUTPUT"])
except json.JSONDecodeError as exc:
    print(f"failed to parse runtime check JSON: {exc}", file=sys.stderr)
    raise SystemExit(1)

probe = data.get("torch_npu_probe", {})
device_count = probe.get("device_count")
if not isinstance(device_count, int) or device_count <= 0:
    print(f"torch.npu.device_count() returned {device_count!r}", file=sys.stderr)
    raise SystemExit(1)

if not probe.get("allocation_ok"):
    print(f"torch_npu allocation check failed: {probe.get('error')}", file=sys.stderr)
    raise SystemExit(1)

print(probe.get("selected_device") or "npu:0")
PY
  )" || return 1

  echo "selected_device=${selected_device}"
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
  if printf '%s\n' "$preflight_output" | grep -Eq '"provider_check_ok": false|Conflicting distributions still provide top-level'; then
    echo "vLLM provider validation failed before benchmark startup." >&2
    echo "Remove conflicting distributions so only the checked-out package provides top-level 'vllm'." >&2
    return 1
  fi

  echo "Self-hosted runner NPU runtime is unhealthy before vLLM startup." >&2
  echo "All visible Ascend devices failed the basic torch_npu allocation check." >&2
  return 1
}

validate_benchmark_result_file() {
  local result_file=${1:-}

  if [[ -z "$result_file" ]]; then
    echo "validate_benchmark_result_file requires a result file path" >&2
    return 2
  fi
  if [[ ! -f "$result_file" ]]; then
    echo "benchmark result file not found: $result_file" >&2
    return 2
  fi

  RESULT_FILE="$result_file" INVALID_EXIT_CODE="$INVALID_BENCHMARK_RESULT_EXIT_CODE" "$PYTHON_BIN" - <<'PY'
import json
import os
import sys

result_file = os.environ["RESULT_FILE"]
invalid_exit_code = int(os.environ["INVALID_EXIT_CODE"])

with open(result_file, "r", encoding="utf-8") as handle:
    payload = json.load(handle)

completed = int(payload.get("completed", 0) or 0)
failed = int(payload.get("failed", 0) or 0)
total = int(payload.get("total_input", completed + failed) or (completed + failed))

if completed == 0 and failed > 0:
    print(
        f"invalid-all-failed completed={completed} failed={failed} total_input={total}",
        file=sys.stderr,
    )
    sys.exit(invalid_exit_code)

print(f"validated benchmark result: completed={completed}, failed={failed}, total_input={total}")
PY
}

normalize_hf_cache_repo_id() {
  local repo_id=${1:-}

  repo_id=${repo_id#/}
  repo_id=${repo_id%/}
  repo_id=${repo_id//\//--}
  printf '%s\n' "$repo_id"
}

prepare_hf_publish_cache_for_runner() {
  local normalized_repo_id
  local -a cleanup_targets=()

  if [[ "$PUBLISH_TO_HF" != "1" ]]; then
    return 0
  fi

  if [[ -z "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
    return 0
  fi

  cleanup_targets+=("$HUGGINGFACE_HUB_CACHE/.locks")

  normalized_repo_id=$(normalize_hf_cache_repo_id "${HF_REPO_ID:-}")
  if [[ -n "$normalized_repo_id" ]]; then
    cleanup_targets+=("$HUGGINGFACE_HUB_CACHE/datasets--$normalized_repo_id")
  fi

  echo "Preparing Hugging Face cache for runner-user publish access"
  if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "1" ]]; then
    run_ascend_root_helper cleanup-paths "${cleanup_targets[@]}"
  else
    rm -rf -- "${cleanup_targets[@]}"
  fi
}

sync_benchmark_publication_to_github() {
  local publisher_script=${BENCHMARK_PUBLICATION_SYNC_SCRIPT:-$VLLM_HUST_REPO/.github/workflows/scripts/sync_benchmark_snapshots_to_github.sh}

  if [[ "$PUBLISH_TO_BENCHMARK_REPO" != "1" ]]; then
    return 0
  fi

  if [[ ! -x "$publisher_script" ]]; then
    echo "benchmark publication sync script is missing or not executable: $publisher_script" >&2
    return 2
  fi

  BENCHMARK_REPO_DIR="$VLLM_HUST_BENCHMARK_REPO" \
  WEBSITE_REPO_DIR="$VLLM_HUST_WEBSITE_REPO" \
  CURRENT_SUBMISSION_DIR="$SUBMISSION_DIR" \
  VLLM_HUST_REPO_DIR="$VLLM_HUST_REPO" \
  LOCAL_SNAPSHOT_OUTPUT_DIR="$AGGREGATE_OUTPUT_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  BENCHMARK_REPO_SLUG="${BENCHMARK_REPO_SLUG:-vLLM-HUST/vllm-hust-benchmark}" \
  BENCHMARK_REPO_GH_TOKEN="${BENCHMARK_REPO_GH_TOKEN:-}" \
  BENCHMARK_REPO_SSH_KEY="${BENCHMARK_REPO_SSH_KEY:-}" \
  SNAPSHOT_COMMIT_MESSAGE="chore(data): sync benchmark publication $RUN_ID" \
  RUN_ID="$RUN_ID" \
  "$publisher_script"
}

run_same_spec_current_benchmark() {
  local same_spec_runner=$VLLM_HUST_BENCHMARK_REPO/scripts/run-current-ascend-same-spec.sh
  local same_spec_raw_result=$RESULT_ROOT/raw_benchmark_result.json
  local same_spec_submission_dir=$RESULT_ROOT/submission
  local same_spec_ready_timeout_seconds=${SAME_SPEC_READY_TIMEOUT_SECONDS:-600}
  local effective_same_spec_file=$SAME_SPEC_SPEC_FILE
  local same_spec_server_log=$RESULT_ROOT/server.stdout.log
  local same_spec_status=0
  local current_vllm_hust_commit
  local current_vllm_hust_ref
  local current_plugin_commit
  local current_plugin_ref
  local display_version

  if [[ ! -f "$same_spec_runner" ]]; then
    echo "same-spec benchmark runner not found: $same_spec_runner" >&2
    return 2
  fi
  if [[ ! -f "$SAME_SPEC_SPEC_FILE" ]]; then
    echo "same-spec benchmark spec file not found: $SAME_SPEC_SPEC_FILE" >&2
    return 2
  fi
  if [[ ! -f "$SAME_SPEC_CONSTRAINTS_FILE" ]]; then
    echo "same-spec benchmark constraints file not found: $SAME_SPEC_CONSTRAINTS_FILE" >&2
    return 2
  fi

  current_vllm_hust_commit=$(git -C "$VLLM_HUST_REPO" rev-parse HEAD 2>/dev/null || true)
  current_vllm_hust_ref=${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-$(git -C "$VLLM_HUST_REPO" branch --show-current 2>/dev/null || echo main)}}
  current_plugin_commit=$(git -C "$VLLM_ASCEND_HUST_REPO" rev-parse HEAD 2>/dev/null || true)
  current_plugin_ref=$(git -C "$VLLM_ASCEND_HUST_REPO" branch --show-current 2>/dev/null || echo main)
  display_version=$(printf '%s' "${TARGET_REPO_SHA:-${GITHUB_SHA:-local}}" | cut -c1-8)

  rm -f "$same_spec_raw_result" "$RAW_RESULT_FILE"
  rm -rf "$same_spec_submission_dir" "$SUBMISSION_DIR"

  prepare_same_spec_pr_preview_compat_file() {
    local output_file=$RESULT_ROOT/pr-preview-same-spec.compat.json

    "$PYTHON_BIN" - "$SAME_SPEC_SPEC_FILE" "$output_file" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
payload = json.loads(source.read_text(encoding="utf-8"))

server_parameters = dict(payload.get("server_parameters") or {})
client_parameters = dict(payload.get("client_parameters") or {})

# PR preview runs on self-hosted Ascend runners where the official same-spec
# defaults can trip plugin paths that are not reliable for smoke gating.
server_parameters["no_enable_chunked_prefill"] = True
server_parameters["no_enable_prefix_caching"] = True
client_parameters.setdefault("temperature", 0)

payload["server_parameters"] = server_parameters
payload["client_parameters"] = client_parameters

target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(target)
PY
  }

  if [[ "$SAME_SPEC_PR_PREVIEW_COMPAT" == "1" && ( "${GITHUB_EVENT_NAME:-}" == "pull_request" || "${GITHUB_EVENT_NAME:-}" == "issue_comment" ) ]]; then
    effective_same_spec_file=$(prepare_same_spec_pr_preview_compat_file)
    echo "Using PR preview same-spec compatibility overlay: $effective_same_spec_file"
  fi

  run_same_spec_runner() {
    if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "1" ]]; then
      READY_TIMEOUT_SECONDS="$same_spec_ready_timeout_seconds" \
        VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
        CURRENT_RUNTIME_CWD=/tmp \
        CURRENT_RUNTIME_PYTHON="$PYTHON_BIN" \
        CURRENT_VLLM_HUST_REPO="$VLLM_HUST_REPO" \
        CURRENT_VLLM_ASCEND_HUST_REPO="$VLLM_ASCEND_HUST_REPO" \
        CURRENT_VLLM_CACHE_ROOT="$runtime_root/current-ascend-same-spec-cache" \
        CURRENT_ENGINE_VERSION="$display_version" \
        CURRENT_GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-vLLM-HUST/vllm-hust}" \
        CURRENT_GITHUB_REF="$current_vllm_hust_ref" \
        CURRENT_GIT_COMMIT="$current_vllm_hust_commit" \
        CURRENT_PLUGIN_ENGINE="vllm-ascend-hust" \
        CURRENT_PLUGIN_GITHUB_REPOSITORY="vLLM-HUST/vllm-ascend-hust" \
        CURRENT_PLUGIN_GITHUB_REF="$current_plugin_ref" \
        CURRENT_PLUGIN_GIT_COMMIT="$current_plugin_commit" \
        CURRENT_SUBMITTER="${GITHUB_ACTOR:-ci}" \
        CURRENT_DATA_SOURCE="vllm-hust-ci-same-spec" \
        RESULT_DIR="$RESULT_ROOT" \
        RESULT_ROOT="$RESULT_ROOT" \
        RUN_ID="$RUN_ID" \
        CURRENT_SERVER_PORT="$PORT" \
        CURRENT_CLIENT_PORT="$PORT" \
        CONSTRAINTS_FILE="$SAME_SPEC_CONSTRAINTS_FILE" \
        run_with_same_spec_stderr_filter run_ascend_root_helper same-spec "$same_spec_runner" "$effective_same_spec_file"
    else
      run_with_same_spec_stderr_filter env \
        READY_TIMEOUT_SECONDS="$same_spec_ready_timeout_seconds" \
        VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
        CURRENT_RUNTIME_CWD=/tmp \
        CURRENT_RUNTIME_PYTHON="$PYTHON_BIN" \
        CURRENT_VLLM_HUST_REPO="$VLLM_HUST_REPO" \
        CURRENT_VLLM_ASCEND_HUST_REPO="$VLLM_ASCEND_HUST_REPO" \
        CURRENT_VLLM_CACHE_ROOT="$runtime_root/current-ascend-same-spec-cache" \
        CURRENT_ENGINE_VERSION="$display_version" \
        CURRENT_GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-vLLM-HUST/vllm-hust}" \
        CURRENT_GITHUB_REF="$current_vllm_hust_ref" \
        CURRENT_GIT_COMMIT="$current_vllm_hust_commit" \
        CURRENT_PLUGIN_ENGINE="vllm-ascend-hust" \
        CURRENT_PLUGIN_GITHUB_REPOSITORY="vLLM-HUST/vllm-ascend-hust" \
        CURRENT_PLUGIN_GITHUB_REF="$current_plugin_ref" \
        CURRENT_PLUGIN_GIT_COMMIT="$current_plugin_commit" \
        CURRENT_SUBMITTER="${GITHUB_ACTOR:-ci}" \
        CURRENT_DATA_SOURCE="vllm-hust-ci-same-spec" \
        RESULT_DIR="$RESULT_ROOT" \
        RESULT_ROOT="$RESULT_ROOT" \
        RUN_ID="$RUN_ID" \
        CURRENT_SERVER_PORT="$PORT" \
        CURRENT_CLIENT_PORT="$PORT" \
        CONSTRAINTS_FILE="$SAME_SPEC_CONSTRAINTS_FILE" \
        bash "$same_spec_runner" "$effective_same_spec_file"
    fi
  }

  print_same_spec_server_log_tail() {
    if [[ -f "$same_spec_server_log" ]]; then
      echo "---- current same-spec vLLM server log tail ----" >&2
      tail -n 300 "$same_spec_server_log" >&2
      echo "---- end current same-spec vLLM server log tail ----" >&2
    else
      echo "current same-spec vLLM server log not found: $same_spec_server_log" >&2
    fi
  }

  set +e
  run_same_spec_runner
  same_spec_status=$?
  set -e

  if [[ "$same_spec_status" -ne 0 ]]; then
    print_same_spec_server_log_tail
    collect_ascend_diagnostics "same-spec-current-failure"
    if [[ -f "$same_spec_server_log" ]] && is_node_env_failure_text "$(cat "$same_spec_server_log" 2>/dev/null || true)"; then
      mark_node_env_failure "same-spec benchmark failed due to Ascend node-level runtime errors"
      return "$NODE_ENV_RETRY_EXIT_CODE"
    fi
    return "$same_spec_status"
  fi

  if [[ ! -f "$same_spec_raw_result" ]]; then
    echo "same-spec benchmark did not produce raw result: $same_spec_raw_result" >&2
    print_same_spec_server_log_tail
    return 2
  fi
  if [[ ! -f "$same_spec_submission_dir/leaderboard_manifest.json" || ! -f "$same_spec_submission_dir/run_leaderboard.json" ]]; then
    echo "same-spec benchmark did not produce submission artifacts under: $same_spec_submission_dir" >&2
    return 2
  fi

  validate_benchmark_result_file "$same_spec_raw_result"

  mkdir -p "$SUBMISSION_DIR"
  cp "$same_spec_raw_result" "$RAW_RESULT_FILE"
  cp "$same_spec_submission_dir/leaderboard_manifest.json" "$SUBMISSION_DIR/leaderboard_manifest.json"
  cp "$same_spec_submission_dir/run_leaderboard.json" "$SUBMISSION_DIR/run_leaderboard.json"
}

start_server() {
  local max_model_len_args=()
  if [[ -n "$MAX_MODEL_LEN" ]]; then
    max_model_len_args=(--max-model-len "$MAX_MODEL_LEN")
  fi

  if command -v setsid >/dev/null 2>&1; then
    if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "1" ]]; then
      local preserve_list
      local helper_python_bin
      export_sudo_preserved_env_vars
      preserve_list=$(build_sudo_env_preserve_list)
      helper_python_bin=$(prepare_root_helper_python_bin)
      if [[ -n "$preserve_list" ]]; then
        PYTHON_BIN="$helper_python_bin" setsid sudo --preserve-env="$preserve_list" -E -n "$ASCEND_BENCHMARK_ROOT_HELPER" serve >"$SERVER_LOG" 2>&1 &
      else
        PYTHON_BIN="$helper_python_bin" setsid sudo -E -n "$ASCEND_BENCHMARK_ROOT_HELPER" serve >"$SERVER_LOG" 2>&1 &
      fi
    else
      setsid "${VLLM_CLI[@]}" serve "$MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        "${max_model_len_args[@]}" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --enforce-eager >"$SERVER_LOG" 2>&1 &
    fi
    server_pid=$!
    server_group_pid=$server_pid
  else
    if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "1" ]]; then
      run_ascend_root_helper serve >"$SERVER_LOG" 2>&1 &
    else
      "${VLLM_CLI[@]}" serve "$MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        "${max_model_len_args[@]}" \
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
echo "publish to benchmark repo: $PUBLISH_TO_BENCHMARK_REPO"
echo "publish to hf: $PUBLISH_TO_HF"
echo "same-spec benchmark enabled: $SAME_SPEC_BENCHMARK_ENABLED"
echo "ascend benchmark use sudo: $ASCEND_BENCHMARK_USE_SUDO"
echo "vllm-ascend-hust repo: $VLLM_ASCEND_HUST_REPO"
if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "1" ]]; then
  echo "ascend benchmark root helper: $ASCEND_BENCHMARK_ROOT_HELPER"
  echo "repo benchmark root helper source: $REPO_ASCEND_BENCHMARK_ROOT_HELPER"
fi

case "$BENCH_SCENARIO" in
  random-online)
    EFFECTIVE_DATASET_NAME="random"
    EFFECTIVE_DATASET_PATH=""
    EFFECTIVE_INPUT_LEN=${BENCH_INPUT_LEN:-$BENCH_RANDOM_INPUT_LEN}
    EFFECTIVE_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-$BENCH_RANDOM_OUTPUT_LEN}
    if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then
      EFFECTIVE_CONSTRAINTS_FILE=$SAME_SPEC_CONSTRAINTS_FILE
    else
      EFFECTIVE_CONSTRAINTS_FILE=${BENCH_CONSTRAINTS_FILE:-$VLLM_HUST_REPO/.github/workflows/data/random-online-ci-constraints.json}
    fi
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

if [[ "$SAME_SPEC_BENCHMARK_ENABLED" != "1" && "$PUBLISH_TO_HF" == "1" && "$BENCH_SCENARIO" == "random-online" && "$ALLOW_RANDOM_HF_PUBLISH" != "1" ]]; then
  echo "Refusing to publish random-online CI preview to HF without ALLOW_RANDOM_HF_PUBLISH=1" >&2
  exit 2
fi

if [[ ! -f "$EFFECTIVE_CONSTRAINTS_FILE" ]]; then
  echo "constraints file not found: $EFFECTIVE_CONSTRAINTS_FILE" >&2
  exit 2
fi

if ! enforce_single_runtime_source_environment; then
  collect_ascend_diagnostics "runtime-source-environment-check-failure"
  echo "Ascend runtime source environment check failed." >&2
  exit 1
fi

if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "1" ]]; then
  echo "Skipping runner-user torch.npu.device_count() probe because sudo mode delegates device access checks to the root helper."
  if [[ "$CHIP_COUNT" == "1" ]]; then
    configure_single_card_ascend_device "${ASCEND_DEVICE_SELECTION_ATTEMPT:-1}"
  fi
  if ! verify_root_helper_ready; then
    exit 1
  fi
  if wait_for_ascend_runtime_ready; then
    :
  else
    status=$?
    exit "$status"
  fi
else
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

if [[ "$BENCH_SCENARIO" == "random-online" && "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then
  run_same_spec_current_benchmark
else
  start_server

  for attempt in $(seq 1 120); do
    if "$CURL_BIN" -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
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

  validate_benchmark_result_file "$RAW_RESULT_FILE"

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
fi

if [[ "$PUBLISH_TO_BENCHMARK_REPO" == "1" ]]; then
  sync_benchmark_publication_to_github
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
