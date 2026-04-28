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
  # Share the same lock with image pull to avoid cleanup/pull races on one node.
  local docker_lock="/tmp/docker-pull.lock"
  exec 9>"$docker_lock"
  flock 9

  docker_root=$(docker info -f '{{.DockerRootDir}}')
  if [ -z "$docker_root" ]; then
    echo "Failed to determine Docker root directory." >&2
    flock -u 9
    return 1
  fi
  echo "Docker root directory: $docker_root"

  disk_usage=$(df "$docker_root" | tail -1 | awk '{print $5}' | sed 's/%//')
  threshold=70
  if [ "$disk_usage" -gt "$threshold" ]; then
    echo "Disk usage is above $threshold%. Running aggressive CI image cleanup..."
    cleanup_old_ci_images "${REGISTRY}/${REPO}" "${image_name}" "${DOCKER_IMAGE_CLEANUP_HOURS:-72}" 1
  else
    echo "Disk usage is below $threshold%. Checking old CI images anyway."
    cleanup_old_ci_images "${REGISTRY}/${REPO}" "${image_name}" "${DOCKER_IMAGE_CLEANUP_HOURS:-72}" 0
  fi
  echo "Old CI image cleanup completed."

  flock -u 9
}

cleanup_old_ci_images() {
  local repo_prefix="$1"
  local current_image_ref="$2"
  local ttl_hours="$3"
  local aggressive_cleanup="$4"

  if [[ -z "$repo_prefix" || "$repo_prefix" == "/" ]]; then
    echo "Skip old-image cleanup: invalid repo prefix '${repo_prefix}'"
    return 0
  fi

  if ! [[ "$ttl_hours" =~ ^[0-9]+$ ]]; then
    echo "Invalid DOCKER_IMAGE_CLEANUP_HOURS='${ttl_hours}', fallback to 72"
    ttl_hours=72
  fi

  local now_epoch cutoff_epoch
  now_epoch=$(date +%s)
  cutoff_epoch=$((now_epoch - ttl_hours * 3600))

  local -a used_image_ids
  mapfile -t used_image_ids < <(docker ps -aq | xargs -r docker inspect --format '{{.Image}}' | sort -u)

  local removed_count=0
  local examined_count=0
  declare -A seen_ids=()

  while read -r image_ref image_id; do
    [[ -z "$image_ref" || -z "$image_id" ]] && continue
    ((examined_count++))

    # Keep the image this job is going to use.
    if [[ "$image_ref" == "$current_image_ref" ]]; then
      continue
    fi

    # Avoid duplicate deletes when multiple tags point to same image id.
    if [[ -n "${seen_ids[$image_id]:-}" ]]; then
      continue
    fi
    seen_ids[$image_id]=1

    # Never delete images that are used by any container on this node.
    if printf '%s\n' "${used_image_ids[@]}" | grep -qx "$image_id"; then
      continue
    fi

    local created created_epoch
    created=$(docker image inspect -f '{{.Created}}' "$image_id" 2>/dev/null || true)
    [[ -z "$created" ]] && continue
    created_epoch=$(date -d "$created" +%s 2>/dev/null || true)
    [[ -z "$created_epoch" ]] && continue

    if (( created_epoch < cutoff_epoch )) || [[ "$aggressive_cleanup" == "1" ]]; then
      if docker image rm -f "$image_id" >/dev/null 2>&1; then
        ((removed_count++))
      fi
    fi
  done < <(docker image ls --no-trunc "$repo_prefix" --format '{{.Repository}}:{{.Tag}} {{.ID}}')

  # Also trim old dangling layers; this is safe and does not remove referenced images.
  docker image prune -f --filter "until=${ttl_hours}h" >/dev/null 2>&1 || true

  if [[ "$aggressive_cleanup" == "1" ]]; then
    echo "Examined ${examined_count} images under ${repo_prefix}, removed ${removed_count} unused images under disk pressure."
  else
    echo "Examined ${examined_count} images under ${repo_prefix}, removed ${removed_count} old images (>${ttl_hours}h)."
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
#default_image_name="public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:${BUILDKITE_COMMIT}-xpu"
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

aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY"

# --- Build or pull test image ---
IMAGE="${IMAGE_TAG_XPU:-${image_name}}"

echo "Using image: ${IMAGE}"

if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Image already exists locally, skipping pull"
else
  echo "Image not found locally, waiting for lock..."

  flock /tmp/docker-pull.lock bash -c "
    if docker image inspect '${IMAGE}' >/dev/null 2>&1; then
      echo 'Image already pulled by another runner'
    else
      echo 'Pulling image...'
      timeout 900 docker pull '${IMAGE}'
    fi
  "

  echo "Pull step completed"
fi

remove_docker_container() {
  docker rm -f "${container_name}" || true
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
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    --entrypoint="" \
    -e "HF_TOKEN=${HF_TOKEN:-}" \
    -e "ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-}" \
    -e "CMDS=${commands}" \
    --name "${container_name}" \
    "${image_name}" \
    bash -c 'set -e; echo "ZE_AFFINITY_MASK is ${ZE_AFFINITY_MASK:-}"; eval "$CMDS"'
