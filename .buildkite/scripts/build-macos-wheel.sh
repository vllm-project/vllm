#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build the macOS Apple Silicon (arm64) CPU wheel natively on a macOS
# Buildkite agent (the `macmini` queue). macOS wheels cannot be
# cross-compiled from Linux, so this runs on the mac mini itself and leaves
# the wheel in artifacts/dist/ for upload-nightly-wheels.sh to pick up --
# the macOS wheel then flows through the release pipeline like any other.
#
# Host prerequisites (provisioned once on the agent, see vllm-project/ci-infra):
#   - Xcode command line tools, git
#   - uv (https://astral.sh/uv)
#   - protobuf/protoc (Rust frontend build script; `brew install protobuf`)
#   - aws CLI + credentials for the wheel S3 upload
#   - buildkite-agent (bundled with the agent)

set -euo pipefail

# Rust frontend (vllm-server) needs protoc; install on demand if the host is
# missing it so the build fails on a real error rather than a missing tool.
if ! command -v protoc >/dev/null 2>&1; then
  echo "protoc not found; installing protobuf via brew"
  brew install protobuf
fi

# One wheel per run: upload-nightly-wheels.sh asserts exactly one *.whl.
rm -rf artifacts/dist
mkdir -p artifacts/dist

export VLLM_TARGET_DEVICE=cpu
# Fail loudly if the Rust frontend can't build, rather than silently shipping
# a wheel that dropped it.
export VLLM_REQUIRE_RUST_FRONTEND=1
export MACOSX_DEPLOYMENT_TARGET=11.0
# Force an arm64-only build and matching wheel tag. uv's CPython is a
# universal2 build, so without these the wheel is tagged universal2 while the
# binaries are arm64-only, and pip would install it on Intel Macs where import
# fails.
export ARCHFLAGS="-arch arm64"
export _PYTHON_HOST_PLATFORM="macosx-11.0-arm64"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

uv venv --python 3.12
uv pip install -r requirements/build/cpu.txt --index-strategy unsafe-best-match
uv build --wheel --no-build-isolation -o artifacts/dist

echo "Built wheel(s):"
ls -l artifacts/dist/*.whl
