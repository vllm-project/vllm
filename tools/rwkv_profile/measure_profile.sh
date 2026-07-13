#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <full|rwkv> <source-dir> <new-output-dir>" >&2
  exit 2
fi

profile=$1
source_dir=$(realpath "$2")
output_dir=$3

if [[ "$profile" != "full" && "$profile" != "rwkv" ]]; then
  echo "profile must be full or rwkv, got: $profile" >&2
  exit 2
fi
if [[ ! -f "$source_dir/setup.py" ]]; then
  echo "source directory does not contain setup.py: $source_dir" >&2
  exit 2
fi
if [[ -e "$output_dir" ]]; then
  echo "output directory must not exist: $output_dir" >&2
  exit 2
fi

mkdir -p "$output_dir/logs" "$output_dir/fetchcontent" "$output_dir/uv-cache"
output_dir=$(realpath "$output_dir")
venv=$output_dir/venv
python_bin=${PYTHON_BIN:-python3.12}

uv venv --python "$python_bin" "$venv" >"$output_dir/logs/venv.log" 2>&1

install_start=$(date +%s%N)
if [[ "$profile" == "full" ]]; then
  UV_CACHE_DIR="$output_dir/uv-cache" \
    uv pip install --python "$venv/bin/python" \
      -r "$source_dir/requirements/build/cuda.txt" \
      -r "$source_dir/requirements/cuda.txt" \
      >"$output_dir/logs/dependency-install.log" 2>&1
else
  UV_CACHE_DIR="$output_dir/uv-cache" \
    uv pip install --python "$venv/bin/python" \
      -r "$source_dir/requirements/rwkv.txt" \
      >"$output_dir/logs/dependency-install.log" 2>&1
fi
install_end=$(date +%s%N)
install_ms=$(((install_end - install_start) / 1000000))

UV_CACHE_DIR="$output_dir/uv-cache" uv pip check --python "$venv/bin/python" \
  >"$output_dir/logs/pip-check.log" 2>&1
UV_CACHE_DIR="$output_dir/uv-cache" uv pip freeze --python "$venv/bin/python" \
  >"$output_dir/distributions.txt"

build_start=$(date +%s%N)
(
  cd "$source_dir"
  VLLM_BUILD_PROFILE="$profile" \
  VLLM_TARGET_DEVICE=cuda \
  CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" \
  MAX_JOBS="${MAX_JOBS:-16}" \
  NVCC_THREADS="${NVCC_THREADS:-2}" \
  SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0}" \
  FETCHCONTENT_BASE_DIR="$output_dir/fetchcontent" \
  UV_CACHE_DIR="$output_dir/uv-cache" \
    uv pip install --python "$venv/bin/python" --no-deps \
      --no-build-isolation --editable .
) >"$output_dir/logs/native-build.log" 2>&1
build_end=$(date +%s%N)
build_ms=$(((build_end - build_start) / 1000000))

UV_CACHE_DIR="$output_dir/uv-cache" uv pip check --python "$venv/bin/python" \
  >"$output_dir/logs/post-build-pip-check.log" 2>&1
UV_CACHE_DIR="$output_dir/uv-cache" uv pip freeze --python "$venv/bin/python" \
  >"$output_dir/post-build-distributions.txt"

manifest=$source_dir/vllm/_build_profile.json
if [[ ! -f "$manifest" ]]; then
  echo "build did not install immutable profile metadata: $manifest" >&2
  exit 1
fi
cp "$manifest" "$output_dir/build-profile.json"

environment_bytes=$(du -sb "$venv" | awk '{print $1}')
cache_bytes=$(du -sb "$output_dir/uv-cache" | awk '{print $1}')
fetchcontent_bytes=$(du -sb "$output_dir/fetchcontent" | awk '{print $1}')
artifact_bytes=$(find "$source_dir/vllm" -type f \( -name '*.so' -o -name '*.so.*' \) \
  -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')

printf '%s\n' \
  "{" \
  "  \"profile\": \"$profile\"," \
  "  \"dependency_install_ms\": $install_ms," \
  "  \"native_build_ms\": $build_ms," \
  "  \"environment_bytes\": $environment_bytes," \
  "  \"download_cache_bytes\": $cache_bytes," \
  "  \"fetchcontent_bytes\": $fetchcontent_bytes," \
  "  \"native_artifact_bytes\": $artifact_bytes" \
  "}" >"$output_dir/metrics.json"

cat "$output_dir/metrics.json"
