#!/bin/bash
# Test setup. Installs dependencies and sets up vLLM in editable mode
# using precompiled wheels (similar to Dockerfile.cohere pattern).
set -x
set -o pipefail

# Ensure nvidia-smi is discoverable in environments where it lives in /usr/local/nvidia/bin
if ! command -v nvidia-smi >/dev/null 2>&1 && [ -x /usr/local/nvidia/bin/nvidia-smi ]; then
  export PATH="/usr/local/nvidia/bin:$PATH"
fi

# Set platform specific env vars for use in test setup.
if command -v nvidia-smi >/dev/null 2>&1; then
    PLATFORM="nvidia"
    REQUIREMENTS_PATH=requirements/dev.txt
elif command -v rocm-smi >/dev/null 2>&1; then
    PLATFORM="amd"
    REQUIREMENTS_PATH=requirements/test/rocm.txt
    # see `run-amd-test.sh`, this adds the repo root to python path
    # and is needed because rocm defaults to spawn
    export PYTHONPATH='..'
else
    echo "Could not auto-detect GPU platform." >&2
    echo "default to nvidia"
    # temporarily unblock b200
    PLATFORM="nvidia"
    REQUIREMENTS_PATH=requirements/dev.txt
    export PYTHONPATH='..'
fi

# default env vars for testing
UV=uv
export PATH="$HOME/.local/bin:$PATH"
command nvidia-smi || rocm-smi || true
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export VLLM_ALLOW_DEPRECATED_BEAM_SEARCH=${VLLM_ALLOW_DEPRECATED_BEAM_SEARCH:-1}
# automatically apply hardware configs
export VLLM_ENABLE_COHERE_AUTO_CONFIG=${VLLM_ENABLE_COHERE_AUTO_CONFIG:-1}

# install dependencies
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# install system dependencies (only missing ones)
missing_deps=()
which wget >/dev/null 2>&1 || missing_deps+=(wget)
which curl >/dev/null 2>&1 || missing_deps+=(curl)
which jq >/dev/null 2>&1 || missing_deps+=(jq)
which lsof >/dev/null 2>&1 || missing_deps+=(lsof)
which git >/dev/null 2>&1 || missing_deps+=(git)
if command -v apt-get >/dev/null 2>&1 && command -v dpkg >/dev/null 2>&1; then
  dpkg -s zlib1g-dev >/dev/null 2>&1 || missing_deps+=(zlib1g-dev)
  dpkg -s libffi-dev >/dev/null 2>&1 || missing_deps+=(libffi-dev)  # required to compile cffi from source for bee eval
fi
if [ ${#missing_deps[@]} -gt 0 ]; then
  apt-get update && apt-get install -y "${missing_deps[@]}"
fi

# test image setup
export UV_HTTP_TIMEOUT=500
# For gb200 compatibility
# Best-effort requirements install: keep going and summarize failures.
failures=()
global_opts=()
install_req_file() {
  local file="$1"
  local base_dir
  base_dir="$(dirname "$file")"
  set +x
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="$(echo "$line" | xargs)"
    [ -z "$line" ] && continue
    case "$line" in
      -r\ *|--requirement\ *)
        req_file="${line#* }"
        if [[ "$req_file" != /* ]]; then
          req_file="$base_dir/$req_file"
        fi
        install_req_file "$req_file"
        ;;
      --*)
        # Treat requirement-file options as globals for subsequent installs.
        global_opts+=($line)
        ;;
      *)
        if ! $UV pip install --system "${global_opts[@]}" $line; then
          failures+=("$line")
        fi
        ;;
    esac
  done < "$file"
  set -x
}
install_req_file "$REQUIREMENTS_PATH"
if [ ${#failures[@]} -gt 0 ]; then
  echo "WARNING: Failed to install the following requirements:" >&2
  printf '  - %s\n' "${failures[@]}" >&2
fi
$UV pip install --system -e tests/vllm_test_utils
$UV pip install --system hf_transfer
$UV pip install --system junitparser
export HF_HUB_ENABLE_HF_TRANSFER=1

# reinstall cohere-transformers
$UV pip install --system -e /app/cohere/transformers
export TOKENIZERS_PARALLELISM=false # needed for lm-eval, avoid repeated warnings

# TODO(czhu): temporary fixes to get amd tests to run. some of these imports
# are needed for tests to run but not in `rocm-test.txt` requirements file
$UV pip install --system av resampy scipy soundfile "mistral_common[audio]"  # needed for ASR / transcription tests
$UV pip install --system opencv-python-headless==4.11.0.86  # import cv2 error from conftest
$UV pip install --system lm-eval[api,ruler]  # needed for lm-eval
$UV pip install --system ray==2.48.0  # needed for lm-eval
$UV tool install keyring --with keyrings.google-artifactregistry-auth # for bee eval

# lm-eval-harness env vars
export HF_ALLOW_CODE_EVAL=1

# Reinstall vLLM in editable mode using prebuilt wheels for extensions.
# This allows testing local source changes while using precompiled C++/CUDA extensions.
REPO_ROOT="$(pwd)"
cd "$REPO_ROOT"
WHEEL_PATH="$(ls /app/cohere/dist/*.whl 2>/dev/null | head -n1)"
if [ -z "$WHEEL_PATH" ]; then
    echo "Error: No precompiled vLLM wheel found in /app/cohere/dist." >&2
    exit 1
fi
echo "Using precompiled wheel: $WHEEL_PATH"
VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL_PATH" $UV pip install --system -e . --no-deps

# Ensure local workspace source takes precedence over any stale site-packages
# vllm install that may remain in the image.
if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
else
  export PYTHONPATH="$REPO_ROOT"
fi
echo "Using PYTHONPATH: $PYTHONPATH"

cd "$REPO_ROOT/tests"
