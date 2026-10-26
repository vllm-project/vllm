#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Derive everything the release pipeline needs about a ROCm stack from its base
# Dockerfile, so the stack's image is the single input to every ROCm script.
#
# Usage: source .buildkite/scripts/rocm/stack.sh <base-dockerfile>
#   e.g. docker/Dockerfile.rocm_base (ROCm 10.0) or docker/Dockerfile.rocm_72_base
#
# Exports:
#   ROCM_STACK_BASE_DOCKERFILE  the base Dockerfile
#   ROCM_STACK_DOCKERFILE       the matching vLLM Dockerfile (base name minus _base)
#   ROCM_STACK_THEROCK          1 if the ROCm SDK comes from TheRock wheels
#   ROCM_STACK_VERSION          ROCm version, e.g. 10.0.0 or 7.2.3
#   ROCM_STACK_VARIANT          wheel variant and Docker Hub tag flavor from major.minor,
#                               same rule as setup.py (rocm100, rocm72)
#   ROCM_STACK_IMAGE_KEY        hash of the base Dockerfile (base image cache key)
#   ROCM_STACK_WHEEL_KEY        hash of the base Dockerfile plus files it COPYs from
#                               the repo (base wheel cache key)
#   ROCM_STACK_ECR_BASE         ECR tag of the cached base image
#   ROCM_STACK_ECR_IMAGE        ECR tag of this commit's release image
#   ROCM_STACK_BASE_WHEELS_DIR  local dir for base wheels (also the artifact path)
#   ROCM_STACK_VLLM_WHEEL_DIR   local dir for the vLLM wheel (also the artifact path)

rocm_stack_init() {
    local base="$1"
    if [[ ! -f "$base" ]]; then
        echo "ROCm base Dockerfile not found: $base" >&2
        return 2
    fi
    ROCM_STACK_BASE_DOCKERFILE="$base"
    ROCM_STACK_DOCKERFILE="${base%_base}"

    local sdk
    sdk=$(sed -nE 's/^ARG ROCM_SDK_VERSION=([0-9.]+).*/\1/p' "$base" | head -1)
    if [[ -n "$sdk" ]]; then
        ROCM_STACK_THEROCK=1
        ROCM_STACK_VERSION="$sdk"
    else
        ROCM_STACK_THEROCK=0
        # BASE_IMAGE format: rocm/dev-ubuntu-22.04:7.2.3-complete -> 7.2.3
        ROCM_STACK_VERSION=$(sed -nE 's/^ARG BASE_IMAGE=.*:([0-9]+(\.[0-9]+)+).*/\1/p' "$base" | head -1)
    fi
    if [[ -z "$ROCM_STACK_VERSION" ]]; then
        echo "Could not determine the ROCm version from $base" >&2
        return 2
    fi

    local major minor
    IFS=. read -r major minor _ <<< "$ROCM_STACK_VERSION"
    ROCM_STACK_VARIANT="rocm${major}${minor}"

    ROCM_STACK_IMAGE_KEY=$(sha256sum "$base" | cut -c1-16)
    local inputs
    inputs=$(grep -E '^COPY ' "$base" | grep -v -- '--from' | awk '{print $2}' || true)
    if [[ -z "$inputs" ]]; then
        ROCM_STACK_WHEEL_KEY="$ROCM_STACK_IMAGE_KEY"
    else
        # shellcheck disable=SC2086
        ROCM_STACK_WHEEL_KEY=$(sha256sum "$base" $inputs | sha256sum | cut -c1-16)
    fi

    local ecr="public.ecr.aws/q9t5s3a7/vllm-release-repo"
    ROCM_STACK_ECR_BASE="${ecr}:${ROCM_STACK_IMAGE_KEY}-${ROCM_STACK_VARIANT}-base"
    ROCM_STACK_ECR_IMAGE="${ecr}:${BUILDKITE_COMMIT:-unknown}-${ROCM_STACK_VARIANT}"
    ROCM_STACK_BASE_WHEELS_DIR="artifacts/${ROCM_STACK_VARIANT}-base-wheels"
    ROCM_STACK_VLLM_WHEEL_DIR="artifacts/${ROCM_STACK_VARIANT}-vllm-wheel"

    export ROCM_STACK_BASE_DOCKERFILE ROCM_STACK_DOCKERFILE ROCM_STACK_THEROCK \
        ROCM_STACK_VERSION ROCM_STACK_VARIANT ROCM_STACK_IMAGE_KEY \
        ROCM_STACK_WHEEL_KEY ROCM_STACK_ECR_BASE ROCM_STACK_ECR_IMAGE \
        ROCM_STACK_BASE_WHEELS_DIR ROCM_STACK_VLLM_WHEEL_DIR
}

if [[ $# -gt 0 ]]; then
    rocm_stack_init "$1"
fi
