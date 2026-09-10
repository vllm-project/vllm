#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build the AMD Zen CPU image (vLLM + zentorch) as a two-step layered build:
#   1. a CPU base image (vLLM installed) built from docker/Dockerfile.cpu
#   2. docker/Dockerfile.zen --target vllm-zen-test -> zen image on top
#
# The image is (re)built from source every time, mirroring
# .buildkite/scripts/hardware_ci/run-cpu-test.sh. It is not pushed to a registry.
#
# See docker/Dockerfile.zen for the build workflow this mirrors.
set -euo pipefail

ZEN_CPU_DOCKERFILE=
ZEN_CPU_RUST_BUILD_VERSION=

usage() {
  echo "Usage: $0 [<registry> <repo>] <commit>"
  exit 1
}

cleanup() {
  if [[ -n "${ZEN_CPU_DOCKERFILE}" && -f "${ZEN_CPU_DOCKERFILE}" ]]; then
    rm -f "${ZEN_CPU_DOCKERFILE}"
  fi
}

resolve_rust_build_version() {
  ZEN_CPU_RUST_BUILD_VERSION="${ZEN_CPU_RUST_BUILD_VERSION:-${BUILDKITE_COMMIT}}"
  export ZEN_CPU_RUST_BUILD_VERSION
}

prepare_zen_cpu_dockerfile() {
  ZEN_CPU_DOCKERFILE="$(mktemp "${TMPDIR:-/tmp}/Dockerfile.cpu.zen.XXXXXX")"
  export ZEN_CPU_DOCKERFILE

  python3 - <<'PY'
from pathlib import Path
import os
import sys

source = Path("docker/Dockerfile.cpu").read_text()
old = """# build_rust.sh installed rustup here via rustup.rs; put it on PATH so the
# child stage below finds it on disk instead of re-downloading it.
ENV PATH=\"/root/.cargo/bin:${PATH}\"

# Relink with the exact Git-derived package version.
FROM rust-build-cache AS rust-build

RUN --mount=type=cache,target=/root/.cargo/registry,sharing=locked \\
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \\
    --mount=type=bind,source=.git,target=.git \\
    SETUPTOOLS_SCM_PRETEND_METADATA=\"{dirty=false}\" bash build_rust.sh
"""
new = """# build_rust.sh installed rustup here via rustup.rs; put it on PATH so the
# child stage below finds it on disk instead of re-downloading it.
ENV PATH=\"/root/.cargo/bin:${PATH}\"

# Relink with the exact Git-derived package version.
FROM rust-build-cache AS rust-build
ARG ZEN_CPU_RUST_BUILD_VERSION

RUN --mount=type=cache,target=/root/.cargo/registry,sharing=locked \\
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \\
    test -n \"${ZEN_CPU_RUST_BUILD_VERSION}\" && \\
    VLLM_RS_BUILD_VERSION=\"${ZEN_CPU_RUST_BUILD_VERSION}\" bash build_rust.sh
"""

if old not in source:
    print("Failed to locate the CPU Rust relink stage in docker/Dockerfile.cpu", file=sys.stderr)
    sys.exit(1)

Path(os.environ["ZEN_CPU_DOCKERFILE"]).write_text(source.replace(old, new, 1))
PY
}

trap cleanup EXIT

image_repo() {
  if [[ -n "${REGISTRY}" ]]; then
    printf '%s/%s' "${REGISTRY}" "${REPO}"
  else
    printf '%s' "${REPO}"
  fi
}

if [[ $# -eq 1 ]]; then
  REGISTRY=""
  REPO="${ZEN_CPU_IMAGE_REPO:-vllm-zen-ci-local}"
  BUILDKITE_COMMIT=$1
elif [[ $# -eq 3 ]]; then
  REGISTRY=$1
  REPO=$2
  BUILDKITE_COMMIT=$3
else
  usage
fi

IMAGE_REPO="$(image_repo)"

# Local image tags (not pushed).
BASE_IMAGE="$IMAGE_REPO:$BUILDKITE_COMMIT-cpu-base-for-zen"
IMAGE="$IMAGE_REPO:$BUILDKITE_COMMIT-zen-cpu"

# ZENTORCH_VERSION is optional; when unset the Dockerfile falls back to
# installing zentorch via `vllm[zen]`.
ZENTORCH_VERSION=${ZENTORCH_VERSION:-}
resolve_rust_build_version
prepare_zen_cpu_dockerfile

# Step 1: build the CPU base image that Dockerfile.zen layers on.
echo "--- :docker: Building CPU base image"
echo "Using Rust build version: ${ZEN_CPU_RUST_BUILD_VERSION}"
docker build --file "$ZEN_CPU_DOCKERFILE" \
  --platform linux/amd64 \
  --build-arg max_jobs=16 \
  --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
  --build-arg ZEN_CPU_RUST_BUILD_VERSION="$ZEN_CPU_RUST_BUILD_VERSION" \
  --build-arg VLLM_CPU_X86=true \
  --tag "$BASE_IMAGE" \
  --target vllm-openai \
  --progress plain .

# Step 2: build the zen test image on top of the CPU base.
echo "--- :docker: Building Zen test image"
docker build --file docker/Dockerfile.zen \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  ${ZENTORCH_VERSION:+--build-arg ZENTORCH_VERSION="$ZENTORCH_VERSION"} \
  --tag "$IMAGE" \
  --target vllm-zen-test \
  --progress plain .
