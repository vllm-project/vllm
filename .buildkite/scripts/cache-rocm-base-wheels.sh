#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Cache helper for ROCm base wheels
#
# This script manages caching of pre-built ROCm base wheels (torch, triton, etc.)
# to avoid rebuilding them when the declared ROCm base inputs have not changed.
#
# Usage:
#   cache-rocm-base-wheels.sh check    - Check if cache exists, outputs "hit" or "miss"
#   cache-rocm-base-wheels.sh upload   - Upload wheels to cache
#   cache-rocm-base-wheels.sh download - Download wheels from cache
#   cache-rocm-base-wheels.sh key      - Output the cache key
#   cache-rocm-base-wheels.sh parent   - Output the digest-pinned parent image
#
# Environment variables:
#   S3_BUCKET                    - S3 bucket name (default: vllm-wheels)
#   ROCM_BASE_PARENT_DIGEST      - pre-resolved BASE_IMAGE digest (optional)
#   ROCM_BASE_PLATFORM           - build platform (default: linux/amd64)
#   ROCM_BASE_CONTENT_FILES      - space-separated repository inputs
#   ROCM_BASE_IMAGE_DIGEST       - pushed ECR image digest (upload only)
#   ROCM_BASE_IMAGE_DIGEST_FILE  - where download writes the paired digest
#
# BASE_IMAGE is resolved to a digest so a mutable parent tag cannot reuse an
# incompatible image or wheel cache entry.

set -euo pipefail

BUCKET="${S3_BUCKET:-vllm-wheels}"
DOCKERFILE="${ROCM_BASE_DOCKERFILE:-docker/Dockerfile.rocm_base}"
CACHE_PREFIX="rocm/cache"
DEFAULT_CONTENT_FILES="${DOCKERFILE} .dockerignore requirements/test/rocm.txt"
CACHE_SCHEMA_VERSION="2"

extract_arg_default() {
    local arg_name="$1"

    sed -n -E "s/^[[:space:]]*ARG[[:space:]]+${arg_name}=\"?([^\"[:space:]]+)\"?.*/\\1/p" \
        "${DOCKERFILE}" | head -1
}

parent_image_ref() {
    local image_ref="${ROCM_BASE_PARENT_IMAGE:-}"

    if [[ -z "${image_ref}" ]]; then
        image_ref=$(extract_arg_default BASE_IMAGE)
    fi
    if [[ -z "${image_ref}" ]]; then
        echo "ERROR: BASE_IMAGE has no default in ${DOCKERFILE}" >&2
        exit 1
    fi
    printf '%s\n' "${image_ref}"
}

resolve_parent_digest() {
    local image_ref="$1"
    local digest="${ROCM_BASE_PARENT_DIGEST:-}"
    local inspect_output=""

    if [[ -z "${digest}" && "${image_ref}" =~ @(sha256:[0-9a-f]{64})$ ]]; then
        digest="${BASH_REMATCH[1]}"
    fi
    if [[ -z "${digest}" ]]; then
        if ! inspect_output=$(docker buildx imagetools inspect "${image_ref}"); then
            echo "ERROR: failed to inspect BASE_IMAGE ${image_ref}" >&2
            return 1
        fi
        digest=$(awk '$1 == "Digest:" { print $2; exit }' <<< "${inspect_output}")
    fi
    if [[ ! "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
        echo "ERROR: failed to resolve BASE_IMAGE digest for ${image_ref}" >&2
        exit 1
    fi
    printf '%s\n' "${digest}"
}

pinned_parent_ref() {
    local image_ref=""
    local digest=""

    if ! image_ref=$(parent_image_ref); then
        return 1
    fi
    if ! digest=$(resolve_parent_digest "${image_ref}"); then
        return 1
    fi
    printf '%s@%s\n' "${image_ref%@*}" "${digest}"
}

# Generate a conservative hash from every repository input used by the release
# base-image and wheel builds. In particular, changing the dependency lock must
# invalidate this cache even when the Dockerfile itself is unchanged.
generate_cache_key() {
    local image_ref=""
    local parent_digest=""
    local path=""
    local -a content_files=()
    local platform="${ROCM_BASE_PLATFORM:-linux/amd64}"
    local use_sccache="${ROCM_BASE_USE_SCCACHE:-${USE_SCCACHE:-1}}"

    if ! image_ref=$(parent_image_ref); then
        return 1
    fi
    if ! parent_digest=$(resolve_parent_digest "${image_ref}"); then
        return 1
    fi

    read -r -a content_files <<< \
        "${ROCM_BASE_CONTENT_FILES:-${DEFAULT_CONTENT_FILES}}"
    for path in "${content_files[@]}"; do
        if [[ ! -f "${path}" ]]; then
            echo "ERROR: ROCm base cache input not found: ${path}" >&2
            exit 1
        fi
    done

    {
        printf 'schema:%s\n' "${CACHE_SCHEMA_VERSION}"
        printf 'parent:%s@%s\n' "${image_ref%@*}" "${parent_digest}"
        printf 'platform:%s\n' "${platform}"
        printf 'image-target:%s\n' "${ROCM_BASE_IMAGE_TARGET:-final}"
        printf 'wheel-target:%s\n' "${ROCM_BASE_WHEEL_TARGET:-debs_wheel_release}"
        printf 'use-sccache:%s\n' "${use_sccache}"
        printf 'sccache-url:%s\n' "${SCCACHE_DOWNLOAD_URL:-<default>}"
        printf 'sccache-version:%s\n' "${SCCACHE_VERSION:-$(extract_arg_default SCCACHE_VERSION)}"
        printf 'sccache-sha256:%s\n' "${SCCACHE_DOWNLOAD_SHA256:-$(extract_arg_default SCCACHE_DOWNLOAD_SHA256)}"
        printf 'sccache-bucket:%s\n' "${SCCACHE_BUCKET_NAME:-$(extract_arg_default SCCACHE_BUCKET_NAME)}"
        printf 'sccache-region:%s\n' "${SCCACHE_REGION_NAME:-$(extract_arg_default SCCACHE_REGION_NAME)}"
        printf 'sccache-no-credentials:%s\n' "${SCCACHE_S3_NO_CREDENTIALS:-$(extract_arg_default SCCACHE_S3_NO_CREDENTIALS)}"
        for path in "${content_files[@]}"; do
            printf 'file:%s\n' "${path}"
            sha256sum "${path}"
        done
    } | sha256sum | cut -d' ' -f1
}

if [[ "${1:-}" == "parent" ]]; then
    pinned_parent_ref
    exit 0
fi

CACHE_KEY=$(generate_cache_key)
CACHE_PATH="s3://${BUCKET}/${CACHE_PREFIX}/${CACHE_KEY}/"
MANIFEST_OBJECT="${CACHE_PATH}_MANIFEST"

case "${1:-}" in
    check)
        echo "Checking cache for key: ${CACHE_KEY}" >&2
        echo "Cache path: ${CACHE_PATH}" >&2

        # The manifest pointer is written only after an immutable wheel set is
        # complete. Concurrent writers therefore cannot expose a mixed set.
        MANIFEST_CONTENT=$(aws s3 cp "${MANIFEST_OBJECT}" - 2>/dev/null || true)
        if [[ "$(sed -n '1p' <<< "${MANIFEST_CONTENT}")" == "schema:2" \
            && "$(sed -n '2p' <<< "${MANIFEST_CONTENT}")" == \
                "cache-key:${CACHE_KEY}" \
            && "$(sed -n '3p' <<< "${MANIFEST_CONTENT}")" =~ \
                ^set-id:[0-9a-f]{64}$ \
            && "$(sed -n '4p' <<< "${MANIFEST_CONTENT}")" =~ \
                ^image-digest:sha256:[0-9a-f]{64}$ ]]; then
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

        WHEEL_COUNT=$(find artifacts/rocm-base-wheels -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l)
        if [[ "$WHEEL_COUNT" -eq 0 ]]; then
            echo "ERROR: No wheels found in artifacts/rocm-base-wheels/" >&2
            exit 1
        fi
        IMAGE_DIGEST="${ROCM_BASE_IMAGE_DIGEST:-}"
        if [[ ! "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
            echo "ERROR: ROCM_BASE_IMAGE_DIGEST is required for upload" >&2
            exit 1
        fi

        CHECKSUM_FILE=$(mktemp)
        MANIFEST_FILE=$(mktemp)
        trap 'rm -f "${CHECKSUM_FILE}" "${MANIFEST_FILE}"' EXIT
        (
            cd artifacts/rocm-base-wheels
            sha256sum -- *.whl | LC_ALL=C sort -k2
        ) > "${CHECKSUM_FILE}"
        SET_ID=$(sha256sum "${CHECKSUM_FILE}" | cut -d' ' -f1)
        SET_PATH="${CACHE_PATH}sets/${SET_ID}/"
        {
            printf 'schema:2\n'
            printf 'cache-key:%s\n' "${CACHE_KEY}"
            printf 'set-id:%s\n' "${SET_ID}"
            printf 'image-digest:%s\n' "${IMAGE_DIGEST}"
            cat "${CHECKSUM_FILE}"
        } > "${MANIFEST_FILE}"

        echo "Uploading $WHEEL_COUNT wheels as immutable set ${SET_ID}..."
        aws s3 cp --recursive artifacts/rocm-base-wheels/ "${SET_PATH}" \
            --exclude "*" --include "*.whl"
        aws s3 cp "${MANIFEST_FILE}" "${MANIFEST_OBJECT}"

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
        MANIFEST_FILE=$(mktemp)
        CHECKSUM_FILE=$(mktemp)
        trap 'rm -f "${MANIFEST_FILE}" "${CHECKSUM_FILE}"' EXIT

        aws s3 cp "${MANIFEST_OBJECT}" "${MANIFEST_FILE}"
        if [[ "$(sed -n '1p' "${MANIFEST_FILE}")" != "schema:2" \
            || "$(sed -n '2p' "${MANIFEST_FILE}")" != \
                "cache-key:${CACHE_KEY}" ]]; then
            echo "ERROR: Wheel cache manifest does not match ${CACHE_KEY}" >&2
            exit 1
        fi
        SET_ID=$(sed -n '3s/^set-id://p' "${MANIFEST_FILE}")
        if [[ ! "${SET_ID}" =~ ^[0-9a-f]{64}$ ]]; then
            echo "ERROR: Wheel cache manifest has an invalid set ID" >&2
            exit 1
        fi
        IMAGE_DIGEST=$(sed -n '4s/^image-digest://p' "${MANIFEST_FILE}")
        if [[ ! "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
            echo "ERROR: Wheel cache manifest has an invalid image digest" >&2
            exit 1
        fi
        tail -n +5 "${MANIFEST_FILE}" > "${CHECKSUM_FILE}"
        if [[ ! -s "${CHECKSUM_FILE}" ]] \
            || grep -Ev '^[0-9a-f]{64}  [A-Za-z0-9._+-]+\.whl$' \
                "${CHECKSUM_FILE}" >/dev/null \
            || [[ "$(sha256sum "${CHECKSUM_FILE}" | cut -d' ' -f1)" != \
                "${SET_ID}" ]]; then
            echo "ERROR: Wheel cache manifest is invalid" >&2
            exit 1
        fi

        mkdir -p artifacts/rocm-base-wheels
        find artifacts/rocm-base-wheels -maxdepth 1 -type f \
            -name '*.whl' -delete

        SET_PATH="${CACHE_PATH}sets/${SET_ID}/"
        aws s3 sync "${SET_PATH}" artifacts/rocm-base-wheels/ \
            --exclude "*" \
            --include "*.whl"

        (cd artifacts/rocm-base-wheels && sha256sum --check "${CHECKSUM_FILE}")

        echo ""
        echo "Downloaded wheels:"
        find artifacts/rocm-base-wheels -maxdepth 1 -name '*.whl' -exec ls -lh {} \;
        WHEEL_COUNT=$(find artifacts/rocm-base-wheels -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l)
        EXPECTED_WHEEL_COUNT=$(wc -l < "${CHECKSUM_FILE}")
        if [[ "$WHEEL_COUNT" -ne "$EXPECTED_WHEEL_COUNT" ]]; then
            echo "ERROR: Wheel cache file count does not match its manifest" >&2
            exit 1
        fi
        if [[ -n "${ROCM_BASE_IMAGE_DIGEST_FILE:-}" ]]; then
            printf '%s\n' "${IMAGE_DIGEST}" > "${ROCM_BASE_IMAGE_DIGEST_FILE}"
        fi
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
        echo "Usage: $0 {check|upload|download|key|path|parent}" >&2
        echo "" >&2
        echo "Commands:" >&2
        echo "  check    - Check if cache exists, outputs 'hit' or 'miss'" >&2
        echo "  upload   - Upload wheels from artifacts/rocm-base-wheels/ to cache" >&2
        echo "  download - Download wheels from cache to artifacts/rocm-base-wheels/" >&2
        echo "  key      - Output the cache key" >&2
        echo "  path     - Output the full S3 cache path" >&2
        echo "  parent   - Output BASE_IMAGE pinned to its resolved digest" >&2
        exit 1
        ;;
esac
