#!/bin/bash
set -o pipefail

###############################################################################
# Intel XPU Test Runner
# Mirrors run-amd-test.sh but adapted for Intel GPU (Level Zero / SYCL)
###############################################################################

export PYTHONPATH=".."

###############################################################################
# Helper Functions
###############################################################################

cleanup_docker() {
  docker_root=$(docker info -f '{{.DockerRootDir}}')
  if [ -z "$docker_root" ]; then
    echo "Failed to determine Docker root directory."
    exit 1
  fi

  disk_usage=$(df "$docker_root" | tail -1 | awk '{print $5}' | sed 's/%//')
  threshold=70
  if [ "$disk_usage" -gt "$threshold" ]; then
    echo "Disk usage above $threshold%. Cleaning..."
    docker image prune -f
    docker volume prune -f && docker system prune --force --filter "until=72h" --all
  fi
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
# Intel XPU specific pytest overrides
###############################################################################

apply_xpu_test_overrides() {
  local cmds="$1"

  # Disable CUDA specific tests
  if [[ $cmds == *" kernels/attention"* ]]; then
    cmds="${cmds} \
    --ignore=kernels/attention/test_flash_attn.py \
    --ignore=kernels/attention/test_triton_attn.py"
  fi

  # Disable unsupported quant kernels
  if [[ $cmds == *" kernels/quantization"* ]]; then
    cmds="${cmds} \
    --ignore=kernels/quantization/test_marlin_gemm.py \
    --ignore=kernels/quantization/test_cutlass_scaled_mm.py"
  fi

  echo "$cmds"
}

###############################################################################
# Main
###############################################################################

echo "--- Intel GPU Info"
sycl-ls || true
clinfo || true

cleanup_docker

echo "--- Pulling Intel XPU container"
image_name="intel/vllm-ci:${BUILDKITE_COMMIT}"
container_name="xpu_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"

docker pull "${image_name}"

remove_docker_container() {
  docker rm -f "${container_name}" || docker image rm -f "${image_name}" || true
}
trap remove_docker_container EXIT

###############################################################################
# Command sourcing
###############################################################################

if [[ -n "${VLLM_TEST_COMMANDS:-}" ]]; then
  commands="${VLLM_TEST_COMMANDS}"
else
  commands="$*"
  if [[ -z "$commands" ]]; then
    echo "Error: No test commands provided." >&2
    exit 1
  fi
fi

echo "Raw commands: $commands"

commands=$(apply_xpu_test_overrides "$commands")
echo "Final commands: $commands"

###############################################################################
# GPU device check
###############################################################################

if [ ! -d "/dev/dri" ]; then
  echo "Error: /dev/dri not found. Intel GPU device not available."
  exit 1
fi

render_gid=$(getent group render | cut -d: -f3)
if [[ -z "$render_gid" ]]; then
  echo "Error: 'render' group not found."
  exit 1
fi

###############################################################################
# Docker run (Single-node)
###############################################################################

echo "--- Running Intel XPU container"

HF_CACHE="$(realpath ~)/huggingface"
mkdir -p "${HF_CACHE}"
HF_MOUNT="/root/.cache/huggingface"

docker run \
  --device /dev/dri \
  --group-add "$render_gid" \
  --network=host \
  --shm-size=16gb \
  --rm \
  -e HF_TOKEN \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e ZE_AFFINITY_MASK \
  -v "${HF_CACHE}:${HF_MOUNT}" \
  -e "HF_HOME=${HF_MOUNT}" \
  -e "PYTHONPATH=.." \
  --name "${container_name}" \
  "${image_name}" \
  /bin/bash -c "${commands}"

exit_code=$?
handle_pytest_exit "$exit_code"
