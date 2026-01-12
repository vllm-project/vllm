#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Cache helper for ROCm base wheels
#
# This script manages caching of pre-built ROCm base wheels (torch, triton, etc.)
# to avoid rebuilding them when Dockerfile.rocm_base hasn't changed.
#
# Usage:
#   cache-rocm-base-wheels.sh check    - Check if cache exists, outputs "hit" or "miss"
#   cache-rocm-base-wheels.sh upload   - Upload wheels to cache
#   cache-rocm-base-wheels.sh download - Download wheels from cache
#   cache-rocm-base-wheels.sh key      - Output the cache key
#
# Environment variables:
#   S3_BUCKET          - S3 bucket name (default: vllm-wheels)
#   PYTHON_VERSION     - Python version (affects cache key)
#   PYTORCH_ROCM_ARCH  - GPU architectures (affects cache key)
#
# Note: ROCm version is determined by BASE_IMAGE in Dockerfile.rocm_base,
#       so changes to ROCm version are captured by the Dockerfile hash.

set -euo pipefail

BUCKET="${S3_BUCKET:-vllm-wheels}"
DOCKERFILE="docker/Dockerfile.rocm_base"
CACHE_PREFIX="rocm/cache"

# Generate hash from Dockerfile content + build args
generate_cache_key() {
    # Include Dockerfile content
    if [[ ! -f "$DOCKERFILE" ]]; then
        echo "ERROR: Dockerfile not found: $DOCKERFILE" >&2
        exit 1
    fi
    local dockerfile_hash=$(sha256sum "$DOCKERFILE" | cut -c1-16)

    # Include key build args that affect the output
    # These should match the ARGs in Dockerfile.rocm_base that change the build output
    # Note: ROCm version is determined by BASE_IMAGE in the Dockerfile, so it's captured by dockerfile_hash
    local args_string="${PYTHON_VERSION:-}|${PYTORCH_ROCM_ARCH:-}"
    local args_hash=$(echo "$args_string" | sha256sum | cut -c1-8)

    echo "${dockerfile_hash}-${args_hash}"
}

CACHE_KEY=$(generate_cache_key)
CACHE_PATH="s3://${BUCKET}/${CACHE_PREFIX}/${CACHE_KEY}/"

case "${1:-}" in
    check)
        echo "Checking cache for key: ${CACHE_KEY}" >&2
        echo "Cache path: ${CACHE_PATH}" >&2
        echo "Variables used in cache key:" >&2
        echo "  PYTHON_VERSION: ${PYTHON_VERSION:-<not set>}" >&2
        echo "  PYTORCH_ROCM_ARCH: ${PYTORCH_ROCM_ARCH:-<not set>}" >&2

        # Check if cache exists by listing objects
        # We look for at least one .whl file
        echo "Running: aws s3 ls ${CACHE_PATH}" >&2
        S3_OUTPUT=$(aws s3 ls "${CACHE_PATH}" 2>&1) || true
        echo "S3 ls output:" >&2
        echo "$S3_OUTPUT" | head -5 >&2

        if echo "$S3_OUTPUT" | grep -q "\.whl"; then
            echo "hit"
        else
            echo "miss"
        fi
        ;;

    upload)
        echo "========================================"
        echo "Uploading wheels to cache"
        echo "========================================"
        echo "Cache key: ${CACHE_KEY}"
        echo "Cache path: ${CACHE_PATH}"
        echo ""

        if [[ ! -d "artifacts/rocm-base-wheels" ]]; then
            echo "ERROR: artifacts/rocm-base-wheels directory not found" >&2
            exit 1
        fi

        WHEEL_COUNT=$(ls artifacts/rocm-base-wheels/*.whl 2>/dev/null | wc -l)
        if [[ "$WHEEL_COUNT" -eq 0 ]]; then
            echo "ERROR: No wheels found in artifacts/rocm-base-wheels/" >&2
            exit 1
        fi

        echo "Uploading $WHEEL_COUNT wheels..."
        aws s3 cp --recursive artifacts/rocm-base-wheels/ "${CACHE_PATH}"

        echo ""
        echo "Cache upload complete!"
        echo "========================================"
        ;;

    download)
        echo "========================================"
        echo "Downloading wheels from cache"
        echo "========================================"
        echo "Cache key: ${CACHE_KEY}"
        echo "Cache path: ${CACHE_PATH}"
        echo ""

        mkdir -p artifacts/rocm-base-wheels
        aws s3 cp --recursive "${CACHE_PATH}" artifacts/rocm-base-wheels/

        echo ""
        echo "Downloaded wheels:"
        ls -lh artifacts/rocm-base-wheels/

        WHEEL_COUNT=$(ls artifacts/rocm-base-wheels/*.whl 2>/dev/null | wc -l)
        echo ""
        echo "Total: $WHEEL_COUNT wheels"
        echo "========================================"
        ;;

    key)
        echo "${CACHE_KEY}"
        ;;

    path)
        echo "${CACHE_PATH}"
        ;;

    *)
        echo "Usage: $0 {check|upload|download|key|path}" >&2
        echo "" >&2
        echo "Commands:" >&2
        echo "  check    - Check if cache exists, outputs 'hit' or 'miss'" >&2
        echo "  upload   - Upload wheels from artifacts/rocm-base-wheels/ to cache" >&2
        echo "  download - Download wheels from cache to artifacts/rocm-base-wheels/" >&2
        echo "  key      - Output the cache key" >&2
        echo "  path     - Output the full S3 cache path" >&2
        exit 1
        ;;
esac
