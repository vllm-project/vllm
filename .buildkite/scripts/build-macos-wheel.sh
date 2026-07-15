#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build the macOS arm64 CPU wheel natively on a macOS agent (the `macmini`
# queue) into artifacts/dist/ for upload-nightly-wheels.sh.

set -euo pipefail

# The Rust frontend build needs protoc.
if ! command -v protoc >/dev/null 2>&1; then
  brew install protobuf
fi

# upload-nightly-wheels.sh expects exactly one wheel.
rm -rf artifacts/dist
mkdir -p artifacts/dist

export VLLM_TARGET_DEVICE=cpu
export VLLM_REQUIRE_RUST_FRONTEND=1
export MACOSX_DEPLOYMENT_TARGET=11.0
# uv's CPython is universal2; force an arm64-only build and tag so the wheel
# isn't mislabelled universal2 and installed on Intel Macs where import fails.
export ARCHFLAGS="-arch arm64"
export _PYTHON_HOST_PLATFORM="macosx-11.0-arm64"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

uv venv --python 3.12
uv pip install -r requirements/build/cpu.txt --index-strategy unsafe-best-match
uv build --wheel --no-build-isolation -o artifacts/dist

ls -l artifacts/dist/*.whl
