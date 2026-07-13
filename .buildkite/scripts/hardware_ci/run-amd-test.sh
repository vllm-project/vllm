#!/bin/bash

# This script runs ROCm tests either directly in a native CI pod or inside the
# corresponding Docker container. Multi-node tests continue to use Docker.
#
# Multi-node detection: Instead of matching on fragile group names, we detect
# multi-node jobs structurally by looking for the bracket command syntax
# "[node0_cmds] && [node1_cmds]" or via the NUM_NODES environment variable.
#
###############################################################################
# QUOTING / COMMAND PASSING
#
# Passing commands as positional arguments ($*) is fragile when the command
# string itself contains double quotes, e.g.:
#
#   bash run-amd-test.sh "export FLAGS="value" && pytest -m "not slow""
#
# The outer shell resolves the nested quotes *before* this script runs, so
# the script receives mangled input it cannot fully recover.
#
# Preferred: pass commands via the VLLM_TEST_COMMANDS environment variable:
#
#   export VLLM_TEST_COMMANDS='export FLAGS="value" && pytest -m "not slow"'
#   bash run-amd-test.sh
#
# Single-quoted assignment preserves all inner double quotes verbatim.
# The $* path is kept for backward compatibility but callers should migrate.
###############################################################################
set -o pipefail

: "${BUILDKIT_PROGRESS:=plain}"
: "${TERM:=xterm-256color}"
: "${FORCE_COLOR:=1}"
: "${CLICOLOR_FORCE:=1}"
: "${PY_COLORS:=1}"
: "${ROCM_DOCKER_TTY:=1}"
: "${PYTHONFAULTHANDLER:=1}"
: "${PYTEST_TIMEOUT:=2100}"
if [[ " ${PYTEST_ADDOPTS:-} " != *" --color"* ]]; then
  PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }--color=yes"
fi
if [[ " ${PYTEST_ADDOPTS:-} " != *" --durations="* ]]; then
  PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }--durations=25"
fi
if [[ " ${PYTEST_ADDOPTS:-} " != *" --durations-min="* ]]; then
  PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }--durations-min=1.0"
fi
# Dump stacks after 15 minutes, then stop an individual test after 35 minutes.
if [[ " ${PYTEST_ADDOPTS:-} " != *" faulthandler_timeout="* ]]; then
  PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }-o faulthandler_timeout=900"
fi
if [[ " ${PYTEST_ADDOPTS:-} " != *" --timeout-method="* &&
  " ${PYTEST_ADDOPTS:-} " != *" --timeout-method "* ]]; then
  PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }--timeout-method=thread"
fi
export BUILDKIT_PROGRESS TERM FORCE_COLOR CLICOLOR_FORCE PY_COLORS PYTEST_ADDOPTS PYTEST_TIMEOUT ROCM_DOCKER_TTY
export PYTHONFAULTHANDLER

# Export Python path for commands that run directly on the host. Containerized
# tests set this to /vllm-workspace below so spawned Python processes do not
# depend on their current working directory.
export PYTHONPATH="${PYTHONPATH:-..}"

###############################################################################
# Helper Functions
###############################################################################

report_docker_usage() {
  echo "--- Docker usage"
  docker system df || true
}

clear_ci_orchestration_env() {
  unset -v \
    VLLM_TEST_GROUP_NAME \
    VLLM_CI_REQUIRE_PERSISTENT_HF_CACHE \
    VLLM_CI_ARTIFACT_STEP \
    VLLM_TEST_CACHE \
    VLLM_CI_EXECUTION_MODE \
    VLLM_CI_WORKSPACE \
    VLLM_CI_REQUIRE_WORKSPACE_MOUNT \
    VLLM_TEST_COMMANDS \
    VLLM_CI_BRANCH \
    VLLM_CI_BASE_IMAGE \
    VLLM_CI_FALLBACK_IMAGE \
    VLLM_CI_DOCKER_DISABLED \
    VLLM_CI_ARTIFACT_GLOB \
    VLLM_CI_ARTIFACT_CHECKSUM_GLOB \
    VLLM_CI_EXPECTED_GPU_COUNT \
    VLLM_CI_USE_ARTIFACTS \
    VLLM_CI_RESULTS_ROOT \
    VLLM_ALLOW_DEPRECATED_BEAM_SEARCH
}

cleanup_network() {
  local max_nodes=${NUM_NODES:-2}
  for node in $(seq 0 $((max_nodes - 1))); do
    if docker ps -a -q -f name="node${node}" | grep -q .; then
      docker stop "node${node}" || true
    fi
  done
  if docker network ls | grep -q docker-net; then
    docker network rm docker-net || true
  fi
}

prepare_artifact_image() {
  if [[ "${VLLM_CI_USE_ARTIFACTS:-0}" != "1" ]]; then
    return 1
  fi
  if ! command -v buildkite-agent >/dev/null 2>&1; then
    echo "buildkite-agent not found; cannot download ROCm wheel artifact"
    return 1
  fi

  local artifact_glob="${VLLM_CI_ARTIFACT_GLOB:-artifacts/vllm-rocm-install/vllm-rocm-install.tar.gz}"
  local archive=""
  local metadata_file=""
  local base_image="${VLLM_CI_BASE_IMAGE:-rocm/vllm-dev:ci_base}"
  local artifact_image=""
  local artifact_key=""
  local base_digest=""
  local wheel_dir=""
  local context_dir=""
  local workspace_dir=""

  artifact_work_dir=$(mktemp -d -t vllm-rocm-artifact.XXXXXX)
  wheel_dir="${artifact_work_dir}/wheels"
  context_dir="${artifact_work_dir}/context"
  workspace_dir="${context_dir}/workspace"
  mkdir -p "${wheel_dir}" "${context_dir}/wheels" "${workspace_dir}"

  echo "--- Downloading ROCm wheel artifact"
  if ! buildkite-agent artifact download "${artifact_glob}" "${artifact_work_dir}"; then
    echo "Failed to download ${artifact_glob}"
    return 1
  fi
  buildkite-agent artifact download \
    "artifacts/vllm-rocm-install/ci-base-image.txt" \
    "${artifact_work_dir}" >/dev/null 2>&1 || true

  archive=$(find "${artifact_work_dir}" -name "vllm-rocm-install.tar.gz" -type f | head -1)
  if [[ -z "${archive}" || ! -f "${archive}" ]]; then
    echo "ROCm wheel artifact archive was not found"
    return 1
  fi

  metadata_file=$(find "${artifact_work_dir}" -name "ci-base-image.txt" -type f | head -1)
  if [[ -n "${metadata_file}" && -s "${metadata_file}" ]]; then
    base_image=$(tr -d '[:space:]' < "${metadata_file}")
  fi

  echo "--- Preparing local ROCm test image"
  echo "Base image: ${base_image}"
  docker pull "${base_image}" || return 1
  base_digest=$(
    docker image inspect \
      --format='{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' \
      "${base_image}" 2>/dev/null || printf '%s' "${base_image}"
  )

  artifact_key=$(
    {
      printf 'base-image:%s\n' "${base_digest}"
      sha256sum "${archive}"
    } | sha256sum | cut -c1-24
  )
  artifact_image="rocm/vllm-ci-artifact:${artifact_key}"

  if docker image inspect "${artifact_image}" >/dev/null 2>&1; then
    echo "Using existing local ROCm artifact image: ${artifact_image}"
    image_name="${artifact_image}"
    return 0
  fi

  tar -xzf "${archive}" -C "${wheel_dir}" || return 1
  if ! ls "${wheel_dir}"/*.whl >/dev/null 2>&1; then
    echo "ROCm wheel artifact did not contain a wheel"
    return 1
  fi
  if [[ ! -d "${wheel_dir}/tests" ]]; then
    echo "ROCm wheel artifact did not contain the test workspace"
    return 1
  fi

  cp "${wheel_dir}"/*.whl "${context_dir}/wheels/" || return 1
  tar -C "${wheel_dir}" \
    --exclude='*.whl' \
    --exclude='.vllm-ci-artifact' \
    --exclude='./.vllm-ci-artifact' \
    -cf - . \
    | tar -C "${workspace_dir}" -xf - || return 1
  cat > "${context_dir}/Dockerfile" <<'EOF'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY wheels/ /tmp/vllm-wheels/
COPY workspace/ /vllm-workspace/
RUN python3 -m pip install --no-deps --force-reinstall /tmp/vllm-wheels/*.whl \
    && rm -rf /tmp/vllm-wheels
WORKDIR /vllm-workspace
EOF

  echo "--- Building local ROCm test image"
  docker build \
    --pull=false \
    --progress "${BUILDKIT_PROGRESS}" \
    --build-arg "BASE_IMAGE=${base_image}" \
    -t "${artifact_image}" \
    "${context_dir}" || return 1
  image_name="${artifact_image}"
  return 0
}

is_native_runtime() {
  [[ "${AMD_CI_RUNTIME:-}" == "native" || "${NATIVE_CI:-}" == "true" ]]
}

validate_native_workspace() {
  local workspace_dir="${VLLM_CI_WORKSPACE:-/vllm-workspace}"
  local workspace_real=""
  local checkout_real=""
  local workspace_mount=""

  mkdir -p "${workspace_dir}" || return 1
  workspace_real=$(readlink -m "${workspace_dir}") || return 1
  if [[ -n "${BUILDKITE_BUILD_CHECKOUT_PATH:-}" ]]; then
    checkout_real=$(readlink -m "${BUILDKITE_BUILD_CHECKOUT_PATH}") || return 1
    if [[ "${checkout_real}" == "${workspace_real}" \
      || "${checkout_real}" == "${workspace_real}/"* \
      || "${workspace_real}" == "${checkout_real}/"* ]]; then
      echo "Refusing to replace ${workspace_real}; it overlaps the Buildkite checkout ${checkout_real}" >&2
      return 1
    fi
  fi
  if [[ "${VLLM_CI_REQUIRE_WORKSPACE_MOUNT:-1}" == "1" ]]; then
    if ! command -v findmnt >/dev/null 2>&1; then
      echo "findmnt is required to verify the native workspace mount" >&2
      return 1
    fi
    workspace_mount=$(findmnt -n -T "${workspace_real}" -o TARGET 2>/dev/null || true)
    if [[ "$(readlink -m "${workspace_mount:-/}")" != "${workspace_real}" ]]; then
      echo "Native CI requires a dedicated volume mounted at ${workspace_real}" >&2
      return 1
    fi
  fi
}

prepare_native_workspace() {
  if [[ "${VLLM_CI_USE_ARTIFACTS:-0}" != "1" ]]; then
    echo "Native CI requires VLLM_CI_USE_ARTIFACTS=1"
    return 1
  fi
  if ! command -v buildkite-agent >/dev/null 2>&1; then
    echo "buildkite-agent not found; cannot download ROCm wheel artifact"
    return 1
  fi
  validate_native_workspace || return 1

  local artifact_glob="${VLLM_CI_ARTIFACT_GLOB:-artifacts/vllm-rocm-install/vllm-rocm-install.tar.gz}"
  local artifact_checksum_glob="${VLLM_CI_ARTIFACT_CHECKSUM_GLOB:-${artifact_glob}.sha256}"
  local artifact_step="${VLLM_CI_ARTIFACT_STEP:-image-build-amd}"
  local archive=""
  local checksum=""
  local download_dir=""
  local metadata_dir=""
  local recorded_base=""
  local recorded_commit=""
  local recorded_wheel=""
  local workspace_dir="${VLLM_CI_WORKSPACE:-/vllm-workspace}"
  local wheel_dir=""
  local attempt=0
  local attempt_dir=""
  local -a archives=()
  local -a checksums=()
  local -a wheels=()

  artifact_work_dir=$(mktemp -d -t vllm-rocm-artifact.XXXXXX) || return 1
  wheel_dir="${artifact_work_dir}/wheels"
  mkdir -p "${wheel_dir}" || return 1

  echo "--- Downloading ROCm wheel artifact from ${artifact_step} (native in-pod)"
  for attempt in 1 2 3; do
    attempt_dir="${artifact_work_dir}/download-${attempt}"
    rm -rf "${attempt_dir}" || return 1
    mkdir -p "${attempt_dir}" || return 1
    if buildkite-agent artifact download \
      "${artifact_glob}" "${attempt_dir}" --step "${artifact_step}" \
      && buildkite-agent artifact download \
        "${artifact_checksum_glob}" "${attempt_dir}" --step "${artifact_step}"; then
      download_dir="${attempt_dir}"
      break
    fi
    echo "Artifact download attempt ${attempt}/3 failed"
    if [[ "${attempt}" -lt 3 ]]; then
      sleep $((attempt * 2))
    fi
  done
  if [[ -z "${download_dir}" ]]; then
    echo "Failed to download ${artifact_glob} and ${artifact_checksum_glob} from ${artifact_step}"
    return 1
  fi

  mapfile -t archives < <(
    find "${download_dir}" -name "vllm-rocm-install.tar.gz" -type f -print
  )
  mapfile -t checksums < <(
    find "${download_dir}" -name "vllm-rocm-install.tar.gz.sha256" -type f -print
  )
  if [[ ${#archives[@]} -ne 1 || ${#checksums[@]} -ne 1 ]]; then
    echo "Expected exactly one ROCm archive and checksum; found ${#archives[@]} archive(s) and ${#checksums[@]} checksum(s)" >&2
    return 1
  fi
  archive="${archives[0]}"
  checksum="${checksums[0]}"
  if [[ "$(dirname "${archive}")" != "$(dirname "${checksum}")" ]]; then
    echo "ROCm archive and checksum were downloaded to different directories" >&2
    return 1
  fi
  (
    cd "$(dirname "${archive}")"
    sha256sum -c "$(basename "${checksum}")"
  ) || return 1

  tar --no-same-owner -xzf "${archive}" -C "${wheel_dir}" || return 1
  mapfile -t wheels < <(
    find "${wheel_dir}" -maxdepth 1 -type f -name '*.whl' -print
  )
  if [[ ${#wheels[@]} -ne 1 ]]; then
    echo "ROCm artifact must contain exactly one top-level wheel; found ${#wheels[@]}" >&2
    return 1
  fi
  metadata_dir="${wheel_dir}/.vllm-ci-artifact"
  for metadata_file in commit.txt native-base-image.txt wheel-filename.txt; do
    if [[ ! -s "${metadata_dir}/${metadata_file}" ]]; then
      echo "ROCm artifact metadata is missing ${metadata_file}" >&2
      return 1
    fi
  done
  for metadata_file in ci-base-image.txt fallback-image.txt; do
    if [[ ! -f "${metadata_dir}/${metadata_file}" ]]; then
      echo "ROCm artifact metadata is missing ${metadata_file}" >&2
      return 1
    fi
  done

  recorded_commit=$(tr -d '\r\n' < "${metadata_dir}/commit.txt")
  recorded_base=$(tr -d '\r\n' < "${metadata_dir}/native-base-image.txt")
  recorded_wheel=$(tr -d '\r\n' < "${metadata_dir}/wheel-filename.txt")
  if [[ -z "${BUILDKITE_COMMIT:-}" || "${recorded_commit}" != "${BUILDKITE_COMMIT}" ]]; then
    echo "ROCm artifact commit ${recorded_commit} does not match ${BUILDKITE_COMMIT:-unset}" >&2
    return 1
  fi
  if [[ -z "${VLLM_CI_BASE_IMAGE:-}" || "${recorded_base}" != "${VLLM_CI_BASE_IMAGE}" ]]; then
    echo "ROCm artifact base ${recorded_base} does not match ${VLLM_CI_BASE_IMAGE:-unset}" >&2
    return 1
  fi
  if [[ "${recorded_wheel}" != "$(basename "${wheels[0]}")" ]]; then
    echo "ROCm artifact wheel manifest ${recorded_wheel} does not match $(basename "${wheels[0]}")" >&2
    return 1
  fi
  for required_dir in tests .buildkite requirements; do
    if [[ ! -d "${wheel_dir}/${required_dir}" ]]; then
      echo "ROCm wheel artifact did not contain ${required_dir}/" >&2
      return 1
    fi
  done

  echo "--- Installing ROCm wheel into pod environment"
  python3 -m pip install --no-deps --force-reinstall "${wheels[0]}" || return 1

  echo "--- Preparing ${workspace_dir} from artifact"
  find "${workspace_dir}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + || return 1
  tar -C "${wheel_dir}" \
    --exclude='*.whl' \
    --exclude='.vllm-ci-artifact' \
    --exclude='./.vllm-ci-artifact' \
    -cf - . | tar --no-same-owner -C "${workspace_dir}" -xf - || return 1
  if [[ ! -d "${workspace_dir}/tests" ]]; then
    echo "Failed to stage the native test workspace" >&2
    return 1
  fi

  return 0
}

initialize_native_environment() {
  local job_id="${BUILDKITE_JOB_ID:-${BUILDKITE_PARALLEL_JOB:-local}}"
  local job_id_suffix=""
  local native_root=""
  local hf_mount=""

  if [[ "$(id -u)" -ne 0 ]]; then
    echo "Native ROCm CI currently requires the ci_base container to run as root" >&2
    return 1
  fi

  job_id="${job_id//[^A-Za-z0-9_.-]/_}"
  job_id_suffix="${job_id##*-}"
  job_id_suffix="${job_id_suffix:0:12}"
  native_root="/tmp/vllm-native-${job_id}"
  TMPDIR="/tmp/vllm-${job_id_suffix}/tmp"
  VLLM_RPC_BASE_PATH="/tmp"
  : "${TORCHINDUCTOR_CACHE_DIR:=${native_root}/cache/torchinductor}"
  : "${TRITON_CACHE_DIR:=${native_root}/cache/triton}"
  : "${VLLM_CACHE_ROOT:=${native_root}/cache/vllm}"
  : "${XDG_CACHE_HOME:=${native_root}/cache/xdg}"
  : "${HF_HOME:=/home/buildkite-agent/huggingface}"
  : "${HF_HUB_DOWNLOAD_TIMEOUT:=300}"
  : "${HF_HUB_ETAG_TIMEOUT:=60}"
  export TMPDIR VLLM_RPC_BASE_PATH
  export TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT XDG_CACHE_HOME
  export HF_HOME HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_ETAG_TIMEOUT
  export PYTORCH_ROCM_ARCH=""

  mkdir -p "${TMPDIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${VLLM_CACHE_ROOT}" \
    "${XDG_CACHE_HOME}" \
    "${HF_HOME}" || return 1

  if [[ "${VLLM_CI_REQUIRE_PERSISTENT_HF_CACHE:-0}" == "1" ]]; then
    if ! command -v findmnt >/dev/null 2>&1; then
      echo "findmnt is required to verify the native Hugging Face cache mount" >&2
      return 1
    fi
    hf_mount=$(findmnt -n -T "${HF_HOME}" -o TARGET 2>/dev/null || true)
    if [[ -z "${hf_mount}" || "${hf_mount}" == "/" ]]; then
      echo "Native CI requires a persistent volume mounted at or above ${HF_HOME}" >&2
      return 1
    fi
  fi
}

run_native_preflight() {
  local expected_gpus="${VLLM_CI_EXPECTED_GPU_COUNT:-1}"

  if [[ ! "${expected_gpus}" =~ ^[0-9]+$ ]]; then
    echo "Invalid VLLM_CI_EXPECTED_GPU_COUNT=${expected_gpus}" >&2
    return 1
  fi

  python3 -c "import encodings, importlib.metadata as im, importlib.util as iu; [im.version(d) for d in ('transformers', 'torch', 'ray', 'sympy', 'markupsafe', 'vllm')]; missing=[m for m in ('torch.utils.model_zoo', 'transformers.models.nomic_bert', 'ray.dag', 'sympy.physics', 'markupsafe._speedups') if iu.find_spec(m) is None]; assert not missing, missing" || return 1

  if [[ "${expected_gpus}" == "0" ]]; then
    echo "Native CPU-only AMD job: skipping ROCm device validation"
    return 0
  fi

  echo "--- ROCm info"
  rocminfo || return 1
  VLLM_CI_EXPECTED_GPU_COUNT="${expected_gpus}" python3 - <<'PY'
import os

import torch

expected = int(os.environ["VLLM_CI_EXPECTED_GPU_COUNT"])
assert torch.version.hip, "PyTorch is not a ROCm build"
assert torch.cuda.is_available(), "ROCm GPU is not available to PyTorch"
actual = torch.cuda.device_count()
assert actual == expected, f"Expected {expected} ROCm GPU(s), found {actual}"
PY
}

is_multi_node() {
  local cmds="$1"
  # Primary signal: NUM_NODES environment variable set by the pipeline
  if [[ "${NUM_NODES:-1}" -gt 1 ]]; then
    return 0
  fi
  # Fallback: detect the bracket syntax structurally
  # Pattern: [...] && [...] (per-node command arrays)
  if [[ "$cmds" =~ \[.*\].*\&\&.*\[.*\] ]]; then
    return 0
  fi
  return 1
}

handle_pytest_exit() {
  local exit_code=$1
  if [ "$exit_code" -eq 5 ]; then
    echo "Pytest exit code 5 (no tests collected) - treating as success."
    exit 0
  fi
  exit "$exit_code"
}

###############################################################################
# Pytest marker/keyword re-quoting
#
# When commands are passed through Buildkite -> shell -> $* -> bash -c,
# quotes around multi-word pytest -m/-k expressions get stripped:
#   pytest -v -s -m 'not cpu_test' v1/core
# becomes:
#   pytest -v -s -m not cpu_test v1/core
#
# pytest then interprets "cpu_test" as a file path, not part of the marker.
#
# This function detects unquoted expressions after -m/-k and re-quotes them
# by collecting tokens until a recognizable boundary is reached:
#   - test path (contains '/')
#   - test file (ends with '.py')
#   - another pytest flag (--xxx or -x single-char flags)
#   - command separator (&& || ; |)
#   - environment variable assignment (FOO=bar)
#
# Single-word markers (e.g. -m cpu_test, -m hybrid_model) pass through
# unquoted since they have no spaces and work fine.
#
# Already-quoted expressions (containing literal single quotes) are passed
# through untouched to avoid double-quoting well-formed shell fragments.
#
# NOTE: This ONLY fixes -m/-k flags. It cannot recover arbitrary inner
# double-quotes stripped by the calling shell (see header comment).
# Use VLLM_TEST_COMMANDS to avoid the problem entirely.
###############################################################################
re_quote_pytest_markers() {
  local input="$1"
  local output=""
  local collecting=false
  local marker_buf=""

  # Strip backslash-newline continuations, then flatten remaining newlines
  local flat="${input//$'\\\n'/ }"
  flat="${flat//$'\n'/ }"

  # Disable globbing to prevent *.py etc. from expanding during read -ra
  local restore_glob
  restore_glob="$(shopt -p -o noglob 2>/dev/null || true)"
  set -o noglob
  local -a words
  read -ra words <<< "$flat"
  eval "$restore_glob"

  for word in "${words[@]}"; do
    if $collecting; then
      # If the token we're about to collect already contains a literal
      # single quote, the expression was already quoted upstream.
      # Flush and stop collecting.
      if [[ "$word" == *"'"* ]]; then
        if [[ -n "$marker_buf" ]]; then
          # Should not normally happen (partial buf + quote), flush raw
          output+="${marker_buf} "
          marker_buf=""
        fi
        output+="${word} "
        collecting=false
        continue
      fi

      local is_boundary=false
      case "$word" in
        # Line-continuation artifact
        "\\")
          is_boundary=true ;;
        # Command separators
        "&&"|"||"|";"|"|")
          is_boundary=true ;;
        # Long flags (--ignore, --shard-id, etc.)
        --*)
          is_boundary=true ;;
        # Short flags (-v, -s, -x, etc.) but NOT negative marker tokens
        # like "not" which don't start with "-". Also skip -k/-m which
        # would start a new marker (handled below).
        -[a-zA-Z])
          is_boundary=true ;;
        # Test path (contains /)
        */*)
          is_boundary=true ;;
        # Test file (ends with .py, possibly with ::method)
        *.py|*.py::*)
          is_boundary=true ;;
        # Environment variable assignment preceding a command (FOO=bar)
        *=*)
          # Only treat as boundary if it looks like VAR=value, not
          # pytest filter expressions like num_gpus=2 inside markers
          if [[ "$word" =~ ^[A-Z_][A-Z0-9_]*= ]]; then
            is_boundary=true
          fi
          ;;
      esac

      if $is_boundary; then
        # Strip surrounding double quotes if present (from upstream
        # single-to-double conversion); without this, wrapping below
        # would produce '"expr"' with literal double-quote characters.
        if [[ "$marker_buf" == '"'*'"' ]]; then
          marker_buf="${marker_buf#\"}"
          marker_buf="${marker_buf%\"}"
        fi
        # Flush the collected marker expression
        if [[ "$marker_buf" == *" "* || "$marker_buf" == *"("* ]]; then
          output+="'${marker_buf}' "
        else
          output+="${marker_buf} "
        fi
        collecting=false
        marker_buf=""
        # Check if this boundary word itself starts a new -m/-k
        if [[ "$word" == "-m" || "$word" == "-k" ]]; then
          output+="${word} "
          collecting=true
        # Drop stray backslash tokens silently
        elif [[ "$word" == "\\" ]]; then
          :
        else
          output+="${word} "
        fi
      else
        # Accumulate into marker buffer
        if [[ -n "$marker_buf" ]]; then
          marker_buf+=" ${word}"
        else
          marker_buf="${word}"
        fi
      fi
    elif [[ "$word" == "-m" || "$word" == "-k" ]]; then
      output+="${word} "
      collecting=true
      marker_buf=""
    else
      output+="${word} "
    fi
  done

  # Flush any trailing marker expression (marker at end of command)
  if $collecting && [[ -n "$marker_buf" ]]; then
    # Strip surrounding double quotes (see mid-stream flush comment)
    if [[ "$marker_buf" == '"'*'"' ]]; then
      marker_buf="${marker_buf#\"}"
      marker_buf="${marker_buf%\"}"
    fi
    if [[ "$marker_buf" == *" "* || "$marker_buf" == *"("* ]]; then
      output+="'${marker_buf}'"
    else
      output+="${marker_buf}"
    fi
  fi

  echo "${output% }"
}

###############################################################################
# Main
###############################################################################

if is_native_runtime; then
  echo "--- Native in-pod ROCm CI (AMD_CI_RUNTIME=${AMD_CI_RUNTIME:-unset}, NATIVE_CI=${NATIVE_CI:-unset})"
  artifact_work_dir=""

  cleanup_native_workspace() {
    if [[ -n "${artifact_work_dir}" ]]; then
      rm -rf "${artifact_work_dir}"
    fi
  }
  trap cleanup_native_workspace EXIT

  if [[ -n "${VLLM_TEST_COMMANDS:-}" ]]; then
    commands="${VLLM_TEST_COMMANDS}"
    commands_source="env"
  else
    commands="$*"
    commands_source="argv"
    if [[ -z "$commands" ]]; then
      echo "Error: No test commands provided for native CI." >&2
      exit 1
    fi
  fi

  if [[ "$commands_source" == "argv" ]]; then
    commands=$(re_quote_pytest_markers "$commands")
  fi

  if is_multi_node "$commands"; then
    echo "Native CI does not support multi-node jobs yet."
    exit 1
  fi

  if ! initialize_native_environment; then
    echo "Failed to initialize the native test environment"
    exit 1
  fi
  if ! prepare_native_workspace; then
    echo "Failed to prepare native test workspace"
    exit 1
  fi

  export PYTHONPATH="${VLLM_CI_WORKSPACE:-/vllm-workspace}"

  echo "Native test commands: $commands"
  run_native_preflight || exit 1
  # Keep AMD CI orchestration variables out of vLLM's runtime environment.
  clear_ci_orchestration_env
  /bin/bash -o pipefail -c "${commands}"
  handle_pytest_exit "$?"
fi

# --- GPU initialization for legacy Docker execution ---
echo "--- ROCm info"
rocminfo

# --- Docker status ---
report_docker_usage

# --- Pull test image ---
echo "--- Pulling container"
image_name="${VLLM_CI_FALLBACK_IMAGE:-rocm/vllm-ci:${BUILDKITE_COMMIT:-local}}"
artifact_work_dir=""
container_name="rocm_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"

remove_docker_container() {
  if docker container inspect "${container_name}" >/dev/null 2>&1; then
    docker rm -f "${container_name}" || true
  fi
  if [[ "${VLLM_CI_REMOVE_TEST_IMAGE:-0}" == "1" ]]; then
    docker image rm -f "${image_name}" || true
  else
    # Keep images by default so later jobs on the same AMD node can reuse layers.
    echo "Keeping ROCm test image locally: ${image_name}"
  fi
  if [[ -n "${artifact_work_dir}" ]]; then
    rm -rf "${artifact_work_dir}"
  fi
}
trap remove_docker_container EXIT

# python_only_compile.sh runs `python setup.py develop` and needs the full repo tree
# under /vllm-workspace (Dockerfile.rocm test stage: mkdir src && mv vllm).
# The ROCm wheel artifact tarball only ships a thin tree (tests, etc.), so
# artifact images cannot satisfy that test — use the full rocm/vllm-ci image.
_cmd_probe="${VLLM_TEST_COMMANDS:-}"
if [[ -z "${_cmd_probe}" ]]; then
  _cmd_probe="$*"
fi
if [[ "${VLLM_CI_USE_ARTIFACTS:-0}" == "1" && "${_cmd_probe}" == *python_only_compile.sh* ]]; then
  echo "INFO: disabling VLLM_CI_USE_ARTIFACTS for python_only_compile (requires full /vllm-workspace tree)"
  export VLLM_CI_USE_ARTIFACTS=0
fi
unset -v _cmd_probe

if ! prepare_artifact_image; then
  echo "Using full ROCm CI image: ${image_name}"
  docker pull "${image_name}" || exit 1
fi

# --- Prepare commands ---
echo "--- Running container"

HF_CACHE="$(realpath ~)/huggingface"
mkdir -p "${HF_CACHE}"
HF_MOUNT="/root/.cache/huggingface"

# Hugging Face Hub defaults to 10s request/download timeouts, while the ROCm
# CI image currently raises downloads to 60s. AMD model-test jobs routinely
# start from a cold or partially-populated shared cache, and the 60s read cap
# has still timed out before pytest reached the vLLM behavior under test.
# Keep the CI default explicit and overridable from the Buildkite environment.
: "${HF_HUB_DOWNLOAD_TIMEOUT:=300}"
: "${HF_HUB_ETAG_TIMEOUT:=60}"

# ---- Command source selection ----
# Prefer VLLM_TEST_COMMANDS (preserves all inner quoting intact).
# Fall back to $* for backward compatibility, but warn that inner
# double-quotes will have been stripped by the calling shell.
if [[ -n "${VLLM_TEST_COMMANDS:-}" ]]; then
  commands="${VLLM_TEST_COMMANDS}"
  commands_source="env"
  echo "Commands sourced from VLLM_TEST_COMMANDS (quoting preserved)"
else
  commands="$*"
  commands_source="argv"
  if [[ -z "$commands" ]]; then
    echo "Error: No test commands provided." >&2
    echo "Usage:" >&2
    echo "  Preferred:  VLLM_TEST_COMMANDS='...' bash $0" >&2
    echo "  Legacy:     bash $0 \"commands here\"" >&2
    exit 1
  fi
  echo "Commands sourced from positional args (legacy mode)"
  echo "WARNING: Inner double-quotes in the command string may have been"
  echo "  stripped by the calling shell. If you see syntax errors, switch to:"
  echo "  export VLLM_TEST_COMMANDS='your commands here'"
  echo "  bash $0"
fi

echo "Raw commands: $commands"

# Only try to repair stripped pytest -m/-k quoting in legacy argv mode.
# VLLM_TEST_COMMANDS preserves inner quoting already, and re-quoting that path
# can corrupt embedded echo strings or otherwise well-formed shell fragments.
if [[ "$commands_source" == "argv" ]]; then
  commands=$(re_quote_pytest_markers "$commands")
  echo "After re-quoting: $commands"
else
  echo "Skipping re-quoting for VLLM_TEST_COMMANDS input"
fi

echo "Final commands: $commands"

standalone_merge_base_env=()
if [[ "$commands" == *python_only_compile.sh* ]]; then
  # The ROCm test image often ships /vllm-workspace without .git. Resolve the
  # wheels.vllm.ai commit from the agent checkout for this test only.
  vllm_standalone_merge_base=""
  checkout="${BUILDKITE_BUILD_CHECKOUT_PATH:-}"
  if [[ -z "${checkout}" || ! -d "${checkout}" ]]; then
    checkout="."
  fi
  # Pass safe.directory per-command because Buildkite uses mixed user IDs.
  if git -c "safe.directory=${checkout}" -C "${checkout}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    vllm_standalone_merge_base="$(
      git -c "safe.directory=${checkout}" -C "${checkout}" merge-base HEAD origin/main 2>/dev/null || true
    )"
  fi
  if [[ -z "${vllm_standalone_merge_base}" ]]; then
    vllm_standalone_merge_base="${BUILDKITE_COMMIT:-}"
  fi
  echo "INFO: passing CI_STANDALONE_MERGE_BASE into container: ${vllm_standalone_merge_base}"
  standalone_merge_base_env=(-e "CI_STANDALONE_MERGE_BASE=${vllm_standalone_merge_base}")
fi

MYPYTHONPATH="/vllm-workspace"

container_job_id="${BUILDKITE_JOB_ID:-${BUILDKITE_PARALLEL_JOB:-0}}"
container_job_id="${container_job_id//[^A-Za-z0-9_.-]/_}"
container_job_id_short="${container_job_id:0:8}"
CONTAINER_TMPDIR="/tmp/vllm-${container_job_id_short}"
CONTAINER_CACHE_ROOT="/tmp/vllm-buildkite-${container_job_id}/cache"
CONTAINER_PREFLIGHT="mkdir -p \"\$TMPDIR\" \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$TRITON_CACHE_DIR\" \"\$VLLM_CACHE_ROOT\" \"\$XDG_CACHE_HOME\" && python -c \"import encodings, importlib.metadata as im, importlib.util as iu; [im.version(d) for d in ('transformers', 'torch', 'ray', 'sympy', 'markupsafe', 'vllm')]; missing=[m for m in ('torch.utils.model_zoo', 'transformers.models.nomic_bert', 'ray.dag', 'sympy.physics', 'markupsafe._speedups') if iu.find_spec(m) is None]; assert not missing, missing\""

# Verify GPU access
render_gid=$(getent group render | cut -d: -f3)
if [[ -z "$render_gid" ]]; then
  echo "Error: 'render' group not found. This is required for GPU access." >&2
  exit 1
fi

# --- RDMA device passthrough (conditional) ---
# If the host has RDMA devices, pass them through so tests like
# test_moriio_connector can access ibverbs. On hosts without RDMA
# hardware the tests will gracefully skip via _rdma_available().
RDMA_FLAGS=""
if [ -d /dev/infiniband ]; then
  echo "RDMA devices detected on host, enabling passthrough"
  RDMA_FLAGS="--device /dev/infiniband --cap-add=IPC_LOCK"
else
  echo "No RDMA devices found on host, RDMA tests will be skipped"
fi

# --- Route: multi-node vs single-node ---
clear_ci_orchestration_env
if is_multi_node "$commands"; then
  echo "--- Multi-node job detected"
  export DCKR_VER=$(docker --version | sed 's/Docker version \(.*\), build .*/\1/')

  # Parse the bracket syntax:  prefix ; [node0_cmds] && [node1_cmds]
  #   BASH_REMATCH[1] = prefix (everything before first bracket)
  #   BASH_REMATCH[2] = comma-separated node0 commands
  #   BASH_REMATCH[3] = comma-separated node1 commands
  if [[ "$commands" =~ ^(.*)\[(.*)"] && ["(.*)\]$ ]]; then
    prefix=$(echo "${BASH_REMATCH[1]}" | sed 's/;//g')
    echo "PREFIX: ${prefix}"

    export composite_command="(command rocm-smi || true)"
    saved_IFS=$IFS
    IFS=','
    read -ra node0 <<< "${BASH_REMATCH[2]}"
    read -ra node1 <<< "${BASH_REMATCH[3]}"
    IFS=$saved_IFS

    if [[ ${#node0[@]} -ne ${#node1[@]} ]]; then
      echo "Warning: node0 has ${#node0[@]} commands, node1 has ${#node1[@]}. They will be paired by index."
    fi

    for i in "${!node0[@]}"; do
      command_node_0=$(echo "${node0[i]}" | sed 's/\"//g')
      command_node_1=$(echo "${node1[i]}" | sed 's/\"//g')

      step_cmd="./.buildkite/scripts/run-multi-node-test.sh /vllm-workspace/tests 2 2 ${image_name} '${command_node_0}' '${command_node_1}'"
      echo "COMMANDS: ${step_cmd}"
      composite_command="${composite_command} && ${step_cmd}"
    done

    /bin/bash -c "${composite_command}"
    exit_code=$?
    cleanup_network
    handle_pytest_exit "$exit_code"
  else
    echo "Multi-node job detected but failed to parse bracket command syntax."
    echo "Expected format: prefix ; [node0_cmd1, node0_cmd2] && [node1_cmd1, node1_cmd2]"
    echo "Got: $commands"
    cleanup_network
    exit 111
  fi
else
  echo "--- Single-node job"
  echo "Render devices: $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES"
  docker_run_terminal_args=(-i)
  if [[ "${ROCM_DOCKER_TTY}" == "1" ]]; then
    docker_run_terminal_args+=(-t)
    echo "Docker interactive stdin: enabled; TTY allocation: enabled"
  else
    echo "Docker interactive stdin: enabled; TTY allocation: disabled"
  fi

  ulimit_core_hard=$(ulimit -H -c)
  if [[ "$ulimit_core_hard" == "unlimited" ]]; then
    # docker run can't pass "unlimited" to --ulimit
    ulimit_core_hard="-1"
  fi
   # Disable core dumps in the ROCm test container unless the ROCm debug agent is enabled
  coredump_flags="--ulimit core=0:$ulimit_core_hard"
  if [[ "$commands" == *"ROCm debug agent enabled"* ]]; then
    # Works around https://github.com/rocm/rocm-systems/issues/6206
    coredump_flags='-e HSA_COREDUMP_PATTERN="/tmp/gpucore.%p"'
  else
    echo "ROCm debug agent not enabled, coredumps are disabled in the test container."
  fi

  docker run \
    "${docker_run_terminal_args[@]}" \
    --device /dev/kfd $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES \
    $RDMA_FLAGS \
    --network=host \
    --shm-size=16gb \
    --group-add "$render_gid" \
    --rm \
    $coredump_flags \
    -e HF_TOKEN \
    -e "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}" \
    -e "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}" \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    -e BUILDKITE_PARALLEL_JOB \
    -e BUILDKITE_PARALLEL_JOB_COUNT \
    -e TERM \
    -e FORCE_COLOR \
    -e CLICOLOR_FORCE \
    -e PY_COLORS \
    -e PYTHONFAULTHANDLER \
    -e PYTEST_ADDOPTS \
    -e PYTEST_TIMEOUT \
    -v "${HF_CACHE}:${HF_MOUNT}" \
    -e "HF_HOME=${HF_MOUNT}" \
    -e "PYTHONPATH=${MYPYTHONPATH}" \
    -e "TMPDIR=${CONTAINER_TMPDIR}/tmp" \
    -e "TORCHINDUCTOR_CACHE_DIR=${CONTAINER_CACHE_ROOT}/torchinductor" \
    -e "TRITON_CACHE_DIR=${CONTAINER_CACHE_ROOT}/triton" \
    -e "VLLM_CACHE_ROOT=${CONTAINER_CACHE_ROOT}/vllm" \
    -e "XDG_CACHE_HOME=${CONTAINER_CACHE_ROOT}/xdg" \
    -e "PYTORCH_ROCM_ARCH=" \
    "${standalone_merge_base_env[@]}" \
    --name "${container_name}" \
    "${image_name}" \
    /bin/bash -c "${CONTAINER_PREFLIGHT} && ${commands}"

  exit_code=$?
  handle_pytest_exit "$exit_code"
fi
