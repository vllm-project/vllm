#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Cache helper for ROCm base wheels
#
# This script manages caching of pre-built ROCm base wheels (torch, triton, etc.)
# to avoid rebuilding them when Dockerfile.rocm_72_base hasn't changed.
#
# Usage:
#   cache-rocm-base-wheels.sh check    - Check if cache exists, outputs "hit" or "miss"
#   cache-rocm-base-wheels.sh upload   - Upload wheels to cache
#   cache-rocm-base-wheels.sh download - Download wheels from cache
#   cache-rocm-base-wheels.sh key      - Output the cache key
#
# Environment variables:
#   S3_BUCKET                 - S3 bucket name (default: vllm-wheels)
#   ROCM_WHEEL_CACHE_DOCKERFILE - base Dockerfile to key on
#                               (default: docker/Dockerfile.rocm_72_base)
#   ROCM_WHEEL_CACHE_INPUTS   - extra files that shape the wheels (space separated)
#   ROCM_BASE_WHEELS_DIR      - local wheel directory (default: artifacts/rocm-base-wheels)
#
# Note: ROCm version is determined by BASE_IMAGE in Dockerfile.rocm_72_base,
#       so changes to ROCm version are captured by the Dockerfile hash.

set -euo pipefail

BUCKET="${S3_BUCKET:-vllm-wheels}"
DOCKERFILE="${ROCM_WHEEL_CACHE_DOCKERFILE:-docker/Dockerfile.rocm_72_base}"
EXTRA_INPUTS="${ROCM_WHEEL_CACHE_INPUTS:-}"
WHEELS_DIR="${ROCM_BASE_WHEELS_DIR:-artifacts/rocm-base-wheels}"
CACHE_PREFIX="rocm/cache"

# Generate hash from Dockerfile content + build args
generate_cache_key() {
    # Include Dockerfile content
    if [[ ! -f "$DOCKERFILE" ]]; then
        echo "ERROR: Dockerfile not found: $DOCKERFILE" >&2
        exit 1
    fi
    if [[ -z "$EXTRA_INPUTS" ]]; then
        sha256sum "$DOCKERFILE" | cut -c1-16
        return
    fi
    # shellcheck disable=SC2086
    sha256sum "$DOCKERFILE" $EXTRA_INPUTS | sha256sum | cut -c1-16
}

CACHE_KEY=$(generate_cache_key)
CACHE_PATH="s3://${BUCKET}/${CACHE_PREFIX}/${CACHE_KEY}/"

case "${1:-}" in
    check)
        echo "Checking cache for key: ${CACHE_KEY}" >&2
        echo "Cache path: ${CACHE_PATH}" >&2

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

        if [[ ! -d "${WHEELS_DIR}" ]]; then
            echo "ERROR: ${WHEELS_DIR} directory not found" >&2
            exit 1
        fi

        WHEEL_COUNT=$(find "${WHEELS_DIR}" -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l)
        if [[ "$WHEEL_COUNT" -eq 0 ]]; then
            echo "ERROR: No wheels found in ${WHEELS_DIR}/" >&2
            exit 1
        fi

        echo "Uploading $WHEEL_COUNT wheels..."
        aws s3 cp --recursive "${WHEELS_DIR}/" "${CACHE_PATH}"

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
        mkdir -p "${WHEELS_DIR}"
        
        # Use sync with include/exclude to only download .whl files
        aws s3 sync "${CACHE_PATH}" "${WHEELS_DIR}/" \
            --exclude "*" \
            --include "*.whl"
        
        echo ""
        echo "Downloaded wheels:"
        find "${WHEELS_DIR}" -maxdepth 1 -name '*.whl' -exec ls -lh {} \;
        WHEEL_COUNT=$(find "${WHEELS_DIR}" -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l)
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
        echo "  upload   - Upload wheels from ${WHEELS_DIR}/ to cache" >&2
        echo "  download - Download wheels from cache to ${WHEELS_DIR}/" >&2
        echo "  key      - Output the cache key" >&2
        echo "  path     - Output the full S3 cache path" >&2
        exit 1
        ;;
esac
