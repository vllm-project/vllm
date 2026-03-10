#!/bin/bash

# This script runs tests inside the Intel XPU docker container.
# It mirrors the structure of run-amd-test.sh while keeping Intel-specific
# container setup and allowing commands to be sourced from YAML or env.
#
# Command sources (in priority order):
#   1) VLLM_TEST_COMMANDS env var (preferred, preserves quoting)
#   2) Positional args (legacy)
#   3) One or more YAML files with a commands list (test-area style)
###############################################################################
set -o pipefail

DRY_RUN=${DRY_RUN:-0}
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

# Export Python path
export PYTHONPATH=".."

###############################################################################
# Helper Functions
###############################################################################

cleanup_docker() {
  docker_root=$(docker info -f '{{.DockerRootDir}}')
  if [ -z "$docker_root" ]; then
    echo "Failed to determine Docker root directory." >&2
    exit 1
  fi
  echo "Docker root directory: $docker_root"

  disk_usage=$(df "$docker_root" | tail -1 | awk '{print $5}' | sed 's/%//')
  threshold=70
  if [ "$disk_usage" -gt "$threshold" ]; then
    echo "Disk usage is above $threshold%. Cleaning up Docker images and volumes..."
    docker image prune -f
    docker volume prune -f && docker system prune --force --filter "until=72h" --all
    echo "Docker images and volumes cleanup completed."
  else
    echo "Disk usage is below $threshold%. No cleanup needed."
  fi
}

re_quote_pytest_markers() {
  local input="$1"
  local output=""
  local collecting=false
  local marker_buf=""

  local flat="${input//$'\n'/ }"
  local restore_glob
  restore_glob="$(shopt -p -o noglob 2>/dev/null || true)"
  set -o noglob
  local -a words
  read -ra words <<< "$flat"
  eval "$restore_glob"

  for word in "${words[@]}"; do
    if $collecting; then
      if [[ "$word" == *"'"* ]]; then
        if [[ -n "$marker_buf" ]]; then
          output+="${marker_buf} "
          marker_buf=""
        fi
        output+="${word} "
        collecting=false
        continue
      fi

      local is_boundary=false
      case "$word" in
        "&&"|"||"|";"|"|")
          is_boundary=true ;;
        --*)
          is_boundary=true ;;
        -[a-zA-Z])
          is_boundary=true ;;
        */*)
          is_boundary=true ;;
        *.py|*.py::*)
          is_boundary=true ;;
        *=*)
          if [[ "$word" =~ ^[A-Z_][A-Z0-9_]*= ]]; then
            is_boundary=true
          fi
          ;;
      esac

      if $is_boundary; then
        if [[ "$marker_buf" == *" "* || "$marker_buf" == *"("* ]]; then
          output+="'${marker_buf}' "
        else
          output+="${marker_buf} "
        fi
        collecting=false
        marker_buf=""
        if [[ "$word" == "-m" || "$word" == "-k" ]]; then
          output+="${word} "
          collecting=true
        else
          output+="${word} "
        fi
      else
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

  if $collecting && [[ -n "$marker_buf" ]]; then
    if [[ "$marker_buf" == *" "* || "$marker_buf" == *"("* ]]; then
      output+="'${marker_buf}'"
    else
      output+="${marker_buf}"
    fi
  fi

  echo "${output% }"
}

apply_intel_test_overrides() {
  local cmds="$1"
  # Placeholder for Intel-specific exclusions/overrides.
  echo "$cmds"
}

is_yaml_file() {
  local p="$1"
  [[ -f "$p" && "$p" == *.yaml ]]
}

extract_yaml_commands() {
  local yaml_path="$1"
  awk '
    $1 == "commands:" { in_cmds=1; next }
    in_cmds && $0 ~ /^[[:space:]]*-[[:space:]]/ {
      sub(/^[[:space:]]*-[[:space:]]/, "");
      print;
      next
    }
    in_cmds && $0 ~ /^[^[:space:]]/ { exit }
  ' "$yaml_path"
}

###############################################################################
# Main
###############################################################################

default_image_name="${REGISTRY}/${REPO}:${BUILDKITE_COMMIT}-xpu"
image_name="${IMAGE_TAG_XPU:-${default_image_name}}"
container_name="xpu_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"

# ---- Command source selection ----
commands=""
if [[ -n "${VLLM_TEST_COMMANDS:-}" ]]; then
  commands="${VLLM_TEST_COMMANDS}"
  echo "Commands sourced from VLLM_TEST_COMMANDS (quoting preserved)"
elif [[ $# -gt 0 ]]; then
  all_yaml=true
  for arg in "$@"; do
    if ! is_yaml_file "$arg"; then
      all_yaml=false
      break
    fi
  done

  if $all_yaml; then
    for yaml in "$@"; do
      mapfile -t COMMANDS < <(extract_yaml_commands "$yaml")
      if [[ ${#COMMANDS[@]} -eq 0 ]]; then
        echo "Error: No commands found in ${yaml}" >&2
        exit 1
      fi
      for cmd in "${COMMANDS[@]}"; do
        if [[ -z "$commands" ]]; then
          commands="${cmd}"
        else
          commands+=" && ${cmd}"
        fi
      done
    done
    echo "Commands sourced from YAML files: $*"
  else
    commands="$*"
    echo "Commands sourced from positional args (legacy mode)"
  fi
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  DEFAULT_YAML="${SCRIPT_DIR}/intel-test.yaml"
  if [[ ! -f "${DEFAULT_YAML}" ]]; then
    echo "Error: YAML file not found: ${DEFAULT_YAML}" >&2
    exit 1
  fi
  mapfile -t COMMANDS < <(extract_yaml_commands "${DEFAULT_YAML}")
  if [[ ${#COMMANDS[@]} -eq 0 ]]; then
    echo "Error: No commands found in ${DEFAULT_YAML}" >&2
    exit 1
  fi
  for cmd in "${COMMANDS[@]}"; do
    if [[ -z "$commands" ]]; then
      commands="${cmd}"
    else
      commands+=" && ${cmd}"
    fi
  done
  echo "Commands sourced from default YAML: ${DEFAULT_YAML}"
fi

if [[ -z "$commands" ]]; then
  echo "Error: No test commands provided." >&2
  exit 1
fi

echo "Raw commands: $commands"
commands=$(re_quote_pytest_markers "$commands")
echo "After re-quoting: $commands"
commands=$(apply_intel_test_overrides "$commands")
echo "Final commands: $commands"

# Dry-run mode prints final commands and exits before Docker.
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1 set, skipping Docker execution."
  exit 0
fi

# --- Docker housekeeping ---
cleanup_docker

# --- Build or pull test image ---
if [[ -n "${IMAGE_TAG_XPU:-}" ]]; then
  echo "Using prebuilt XPU image: ${IMAGE_TAG_XPU}"
  docker pull "${IMAGE_TAG_XPU}"
else
  echo "No IMAGE_TAG_XPU provided, building local XPU image"
  docker build -t "${image_name}" -f docker/Dockerfile.xpu .
fi

remove_docker_container() {
  docker rm -f "${container_name}" || true
  docker image rm -f "${image_name}" || true
  docker system prune -f || true
}
trap remove_docker_container EXIT

# --- Single-node job ---

if [[ -z "${ZE_AFFINITY_MASK:-}" ]]; then
  echo "Warning: ZE_AFFINITY_MASK is not set. Proceeding without device affinity." >&2
fi

docker run \
    --device /dev/dri:/dev/dri \
    --net=host \
    --ipc=host \
    --privileged \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --entrypoint="" \
    -e "HF_TOKEN=${HF_TOKEN:-}" \
    -e "ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-}" \
    -e "CMDS=${commands}" \
    --name "${container_name}" \
    "${image_name}" \
    bash -c 'set -e; echo "${ZE_AFFINITY_MASK:-}"; eval "$CMDS"'
