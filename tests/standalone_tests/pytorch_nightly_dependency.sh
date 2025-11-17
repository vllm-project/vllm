#!/bin/sh
# This script tests if the nightly torch packages are not overridden by the dependencies

set -e
set -x

cd /vllm-workspace/

# Auto-detect backend if not set
BACKEND="${VLLM_TARGET_DEVICE:-}"

if [ -z "$BACKEND" ]; then
  echo ">>> Auto-detecting backend..."
  
  # Check for ROCm
  if [ -d "/opt/rocm" ] || [ -n "$ROCM_HOME" ] || command -v rocminfo >/dev/null 2>&1; then
    BACKEND="rocm"
    echo ">>> Detected ROCm"
  # Check for CUDA
  elif [ -d "/usr/local/cuda" ] || [ -n "$CUDA_HOME" ] || command -v nvidia-smi >/dev/null 2>&1; then
    BACKEND="cuda"
    echo ">>> Detected CUDA"
  else
    echo "Error: Could not auto-detect backend. Neither ROCm nor CUDA found."
    echo "Please set VLLM_TARGET_DEVICE environment variable (cuda or rocm)"
    exit 1
  fi
else
  # Normalize to lowercase if manually set
  BACKEND=$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')
  echo ">>> Using manually specified backend: ${BACKEND}"
fi

rm -rf .venv

uv venv .venv

source .venv/bin/activate

# check the environment
uv pip freeze

echo ">>> Installing nightly torch packages for ${BACKEND}"

if [ "$BACKEND" = "rocm" ]; then
  if ! command -v rocminfo >/dev/null 2>&1; then
    echo "Error: rocminfo command not found. ROCm installation may be incomplete."
    exit 1
  fi
  
  # # We don't need to extract ROCm version for now, as we only support ROCm 7.0
  # # Extract ROCm version - handle both "ROCm version" and "ROCk module version" formats
  # ROCM_FULL_VERSION=$(rocminfo | grep -iE "(ROCm|ROCk module) version" | head -1 | awk '{print $4}')
  # if [ -z "$ROCM_FULL_VERSION" ]; then
  #   echo "Error: Could not detect ROCm version from rocminfo output."
  #   echo "Debug: rocminfo output:"
  #   rocminfo | head -20
  #   exit 1
  # fi
  
  # # Extract major.minor (e.g., "6.14" from "6.14.14")
  # ROCM_VERSION=$(echo "$ROCM_FULL_VERSION" | cut -d. -f1,2)
  # if [ -z "$ROCM_VERSION" ]; then
  #   echo "Error: Failed to parse ROCm version: ${ROCM_FULL_VERSION}"
  #   exit 1
  # fi
  
  # if [ "$ROCM_MAJOR" -lt 6 ] || { [ "$ROCM_MAJOR" -eq 6 ] && [ "$ROCM_MINOR" -lt 2 ]; }; then
  #   echo "Error: ROCm version ${ROCM_VERSION} is below minimum required version 6.2"
  #   exit 1
  # fi
  
  # Statically set ROCm version to 7.0 for now
  ROCM_VERSION=7.0
  # echo ">>> Using ROCm version: ${ROCM_VERSION}"
  uv pip install --quiet torch torchvision --pre --index-url https://download.pytorch.org/whl/nightly/rocm${ROCM_VERSION}
  TORCH_PACKAGES="torch|torchvision"
elif [ "$BACKEND" = "cuda" ]; then
  uv pip install --quiet torch torchvision torchaudio --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu128
  TORCH_PACKAGES="torch|torchvision|torchaudio"
else
  echo "Error: Unknown backend '${BACKEND}'. Use 'cuda' or 'rocm'"
  exit 1
fi

echo ">>> Capturing torch-related versions before requirements install"
uv pip freeze | grep -E "^(${TORCH_PACKAGES})" | sort > before.txt
echo "Before:"
cat before.txt

echo ">>> Installing requirements/nightly_torch_test.txt"
uv pip install --quiet -r requirements/nightly_torch_test.txt

echo ">>> Capturing torch-related versions after requirements install"
uv pip freeze | grep -E "^(${TORCH_PACKAGES})" | sort > after.txt
echo "After:"
cat after.txt

echo ">>> Comparing versions"
if diff before.txt after.txt; then
  echo "torch version not overridden."
else
  echo "torch version overridden by nightly_torch_test.txt, \
  if the dependency is not triggered by the pytorch nightly test,\
  please add the dependency to the list 'white_list' in tools/pre_commit/generate_nightly_torch_test.py"
  exit 1
fi
