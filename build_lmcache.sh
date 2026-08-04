#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
SESSION_NAME="vllm-build"
IMAGE_TAG="nctu6/vllm-lmcache"
CUDA_VERSION="13.0.3"
MAX_JOBS=64
NVCC_THREADS=4
CPU_SET="0-63"

# Automatically find Dockerfile location (./docker/Dockerfile or ./Dockerfile)
DOCKERFILE_PATH="docker/Dockerfile.lmcache"
if [[ ! -f "$DOCKERFILE_PATH"  ]]; then
  echo "Missing: $DOCKERFILE_PATH"
  exit 1
fi

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
run_build() {
  echo "=================================================="
  echo "Starting vLLM Build"
  echo "Dockerfile:    ${DOCKERFILE_PATH}"
  echo "Image Tag:     ${IMAGE_TAG}"
  echo "CUDA Version:  ${CUDA_VERSION}"
  echo "CPU Cores:     ${CPU_SET} (max_jobs=${MAX_JOBS})"
  echo "=================================================="

  if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    echo "ERROR: Could not find Dockerfile at ${DOCKERFILE_PATH}"
    echo "Make sure you are running this script from the root of the vllm repository."
    return 1
  fi

  DOCKER_BUILDKIT=1 docker build \
    --cpuset-cpus="${CPU_SET}" \
    --progress=plain \
    -f "${DOCKERFILE_PATH}" \
    --target vllm-openai \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg max_jobs="${MAX_JOBS}" \
    --build-arg NVCC_THREADS="${NVCC_THREADS}" \
    --build-arg INSTALL_KV_CONNECTORS=true \
    --build-arg VLLM_MAX_SIZE_MB=4000 \
    -t "${IMAGE_TAG}" .
}

# ------------------------------------------------------------------------------
# Execution Logic
# ------------------------------------------------------------------------------
# Option 1: Explicitly pass --no-tmux to skip tmux entirely and print logs directly to stdout
if [[ "${1:-}" == "--no-tmux" ]]; then
  echo "Running build directly (tmux bypassed)..."
  run_build 2>&1 | tee build.log
  exit $?
fi

# Option 2: Internal tmux worker process
if [[ "${1:-}" == "--inside-tmux" ]]; then
  run_build 2>&1 | tee build.log || echo "Build failed with exit code $?"
  echo ""
  echo "Press Enter to drop into shell, or close this window..."
  read -r
  exec bash
fi

# Option 3: Already running inside a tmux session on host
if [[ -n "${TMUX:-}" ]]; then
  run_build 2>&1 | tee build.log
  exit $?
fi

# Option 4: Default launcher (launches tmux if available, unless overridden)
if command -v tmux &> /dev/null; then
  echo "Starting build and attaching to tmux session '${SESSION_NAME}'..."
  echo "(Tip: Pass '--no-tmux' to run directly in this terminal)"
  exec tmux new-session -A -s "${SESSION_NAME}" "$0 --inside-tmux"
else
  echo "Warning: tmux not found on host. Running directly..."
  run_build 2>&1 | tee build.log
fi
