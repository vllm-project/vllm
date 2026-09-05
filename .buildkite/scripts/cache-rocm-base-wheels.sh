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
#   ROCM_BASE_WHEEL_CACHE_FORCE  - set to 1 to bypass an otherwise valid cache
#
# BASE_IMAGE is resolved to a digest so a mutable parent tag cannot reuse an
# incompatible image or wheel cache entry.

set -euo pipefail

BUCKET="${S3_BUCKET:-vllm-wheels}"
DOCKERFILE="${ROCM_BASE_DOCKERFILE:-docker/Dockerfile.rocm_base}"
CACHE_PREFIX="rocm/cache"
DEFAULT_CONTENT_FILES="${DOCKERFILE} .dockerignore requirements/build/rocm.txt"
CACHE_SCHEMA_VERSION="3"
MANIFEST_SCHEMA_VERSION="2"

extract_arg_default() {
    local arg_name="$1"

    sed -n -E "s/^[[:space:]]*ARG[[:space:]]+${arg_name}=\"?([^\"[:space:]]+)\"?.*/\\1/p" \
        "${DOCKERFILE}" | head -1
}

effective_use_sccache() {
    local use_sccache="${ROCM_BASE_USE_SCCACHE:-${USE_SCCACHE:-}}"

    if [[ -z "${use_sccache}" ]]; then
        use_sccache=$(extract_arg_default USE_SCCACHE)
    fi
    # A bare Dockerfile ARG has an empty value, which disables sccache.
    use_sccache="${use_sccache:-0}"
    if [[ "${use_sccache}" != "0" && "${use_sccache}" != "1" ]]; then
        echo "ERROR: ROCm base USE_SCCACHE must be 0 or 1: ${use_sccache}" >&2
        return 1
    fi
    printf '%s\n' "${use_sccache}"
}

effective_pytorch_rocm_arch() {
    local arch="${ROCM_BASE_PYTORCH_ROCM_ARCH:-${PYTORCH_ROCM_ARCH:-}}"

    if [[ -z "${arch}" ]]; then
        arch=$(extract_arg_default PYTORCH_ROCM_ARCH)
    fi
    if [[ -z "${arch}" ]]; then
        echo "ERROR: PYTORCH_ROCM_ARCH has no effective value" >&2
        return 1
    fi
    printf '%s\n' "${arch}"
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
    local use_sccache=""
    local pytorch_rocm_arch=""
    local sccache_url="${SCCACHE_DOWNLOAD_URL:-<default>}"
    local sccache_url_hash=""
    local key_material=""

    if ! image_ref=$(parent_image_ref); then
        return 1
    fi
    if ! parent_digest=$(resolve_parent_digest "${image_ref}"); then
        return 1
    fi
    if ! use_sccache=$(effective_use_sccache); then
        return 1
    fi
    if ! pytorch_rocm_arch=$(effective_pytorch_rocm_arch); then
        return 1
    fi
    sccache_url_hash=$(printf '%s' "${sccache_url}" | sha256sum | cut -d' ' -f1)

    read -r -a content_files <<< \
        "${ROCM_BASE_CONTENT_FILES:-${DEFAULT_CONTENT_FILES}}"
    if ((${#content_files[@]} == 0)); then
        echo "ERROR: ROCM_BASE_CONTENT_FILES must name at least one file" >&2
        return 1
    fi
    for path in "${content_files[@]}"; do
        if [[ ! -f "${path}" ]]; then
            echo "ERROR: ROCm base cache input not found: ${path}" >&2
            return 1
        fi
    done

    key_material=$(
        printf 'schema:%s\n' "${CACHE_SCHEMA_VERSION}"
        printf 'parent:%s@%s\n' "${image_ref%@*}" "${parent_digest}"
        printf 'platform:%s\n' "${platform}"
        printf 'image-target:%s\n' "${ROCM_BASE_IMAGE_TARGET:-final}"
        printf 'wheel-target:%s\n' "${ROCM_BASE_WHEEL_TARGET:-debs_wheel_release}"
        printf 'use-sccache:%s\n' "${use_sccache}"
        printf 'pytorch-rocm-arch:%s\n' "${pytorch_rocm_arch}"
        # URLs may contain credentials. Hash them so the identity remains
        # observable without printing sensitive query parameters to CI logs.
        printf 'sccache-url-sha256:%s\n' "${sccache_url_hash}"
        printf 'sccache-version:%s\n' "${SCCACHE_VERSION:-$(extract_arg_default SCCACHE_VERSION)}"
        printf 'sccache-sha256:%s\n' "${SCCACHE_DOWNLOAD_SHA256:-$(extract_arg_default SCCACHE_DOWNLOAD_SHA256)}"
        printf 'sccache-bucket:%s\n' "${SCCACHE_BUCKET_NAME:-$(extract_arg_default SCCACHE_BUCKET_NAME)}"
        printf 'sccache-region:%s\n' "${SCCACHE_REGION_NAME:-$(extract_arg_default SCCACHE_REGION_NAME)}"
        printf 'sccache-no-credentials:%s\n' "${SCCACHE_S3_NO_CREDENTIALS:-$(extract_arg_default SCCACHE_S3_NO_CREDENTIALS)}"
        for path in "${content_files[@]}"; do
            printf 'file:%s\n' "${path}"
            sha256sum "${path}"
        done
    )
    printf 'ROCm base cache key material:\n%s\n' "${key_material}" >&2
    printf '%s\n' "${key_material}" | sha256sum | cut -d' ' -f1
}

validate_manifest_file() {
    local manifest_file="$1"
    local checksum_file="$2"
    local set_id=""
    local image_digest=""
    local checksum_count=0
    local unique_wheel_count=0

    if [[ "$(sed -n '1p' "${manifest_file}")" != \
        "schema:${MANIFEST_SCHEMA_VERSION}" \
        || "$(sed -n '2p' "${manifest_file}")" != \
            "cache-key:${CACHE_KEY}" ]]; then
        echo "ERROR: Wheel cache manifest does not match ${CACHE_KEY}" >&2
        return 1
    fi
    set_id=$(sed -n '3s/^set-id://p' "${manifest_file}")
    if [[ ! "${set_id}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "ERROR: Wheel cache manifest has an invalid set ID" >&2
        return 1
    fi
    image_digest=$(sed -n '4s/^image-digest://p' "${manifest_file}")
    if [[ ! "${image_digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
        echo "ERROR: Wheel cache manifest has an invalid image digest" >&2
        return 1
    fi
    tail -n +5 "${manifest_file}" > "${checksum_file}"
    checksum_count=$(wc -l < "${checksum_file}")
    unique_wheel_count=$(sed -E 's/^[0-9a-f]{64}  //' "${checksum_file}" \
        | LC_ALL=C sort -u | wc -l)
    if [[ ! -s "${checksum_file}" ]] \
        || grep -Ev '^[0-9a-f]{64}  [A-Za-z0-9._+-]+\.whl$' \
            "${checksum_file}" >/dev/null \
        || ((unique_wheel_count != checksum_count)) \
        || [[ "$(sha256sum "${checksum_file}" | cut -d' ' -f1)" != \
            "${set_id}" ]]; then
        echo "ERROR: Wheel cache manifest is invalid" >&2
        return 1
    fi
}

validate_flat_wheel_directory() {
    local wheel_dir="$1"
    local unexpected=""

    unexpected=$(find "${wheel_dir}" -mindepth 1 -maxdepth 1 \
        \( ! -type f -o ! -name '*.whl' \) -print -quit)
    if [[ -n "${unexpected}" ]]; then
        echo "ERROR: Wheel cache directory contains an unmanifested entry: ${unexpected}" >&2
        return 1
    fi
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

        if [[ "${ROCM_BASE_WHEEL_CACHE_FORCE:-0}" == "1" ]]; then
            echo "ROCM_BASE_WHEEL_CACHE_FORCE=1; bypassing cached wheel set" >&2
            echo "miss"
            exit 0
        fi
        if [[ "${ROCM_BASE_WHEEL_CACHE_FORCE:-0}" != "0" ]]; then
            echo "ERROR: ROCM_BASE_WHEEL_CACHE_FORCE must be 0 or 1" >&2
            exit 1
        fi

        # The manifest pointer is written only after an immutable wheel set is
        # complete. Concurrent writers therefore cannot expose a mixed set.
        MANIFEST_FILE=$(mktemp)
        CHECKSUM_FILE=$(mktemp)
        trap 'rm -f "${MANIFEST_FILE}" "${CHECKSUM_FILE}"' EXIT
        AWS_DIAGNOSTIC=""
        if ! AWS_DIAGNOSTIC=$(aws s3 cp "${MANIFEST_OBJECT}" "${MANIFEST_FILE}" 2>&1); then
            echo "Wheel cache manifest lookup failed; treating as a miss:" >&2
            printf '%s\n' "${AWS_DIAGNOSTIC:-<no AWS diagnostic>}" >&2
            echo "miss"
        elif validate_manifest_file "${MANIFEST_FILE}" "${CHECKSUM_FILE}"; then
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

        validate_flat_wheel_directory artifacts/rocm-base-wheels

        WHEEL_COUNT=$(find artifacts/rocm-base-wheels -maxdepth 1 -type f -name '*.whl' 2>/dev/null | wc -l)
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
        VALIDATED_CHECKSUM_FILE=$(mktemp)
        trap 'rm -f "${CHECKSUM_FILE}" "${MANIFEST_FILE}" "${VALIDATED_CHECKSUM_FILE}"' EXIT
        (
            cd artifacts/rocm-base-wheels
            sha256sum -- *.whl | LC_ALL=C sort -k2
        ) > "${CHECKSUM_FILE}"
        SET_ID=$(sha256sum "${CHECKSUM_FILE}" | cut -d' ' -f1)
        SET_PATH="${CACHE_PATH}sets/${SET_ID}/"
        {
            printf 'schema:%s\n' "${MANIFEST_SCHEMA_VERSION}"
            printf 'cache-key:%s\n' "${CACHE_KEY}"
            printf 'set-id:%s\n' "${SET_ID}"
            printf 'image-digest:%s\n' "${IMAGE_DIGEST}"
            cat "${CHECKSUM_FILE}"
        } > "${MANIFEST_FILE}"
        validate_manifest_file "${MANIFEST_FILE}" "${VALIDATED_CHECKSUM_FILE}"

        echo "Uploading $WHEEL_COUNT wheels as immutable set ${SET_ID}..."
        while IFS= read -r wheel_name; do
            aws s3 cp \
                "artifacts/rocm-base-wheels/${wheel_name}" \
                "${SET_PATH}${wheel_name}"
        done < <(sed -E 's/^[0-9a-f]{64}  //' "${CHECKSUM_FILE}")
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
        if [[ "${ROCM_BASE_WHEEL_CACHE_FORCE:-0}" == "1" ]]; then
            echo "ERROR: forced wheel-cache bypass refuses a cached download" >&2
            exit 1
        fi
        if [[ "${ROCM_BASE_WHEEL_CACHE_FORCE:-0}" != "0" ]]; then
            echo "ERROR: ROCM_BASE_WHEEL_CACHE_FORCE must be 0 or 1" >&2
            exit 1
        fi
        MANIFEST_FILE=$(mktemp)
        CHECKSUM_FILE=$(mktemp)
        trap 'rm -f "${MANIFEST_FILE}" "${CHECKSUM_FILE}"' EXIT

        aws s3 cp "${MANIFEST_OBJECT}" "${MANIFEST_FILE}"
        validate_manifest_file "${MANIFEST_FILE}" "${CHECKSUM_FILE}"
        SET_ID=$(sed -n '3s/^set-id://p' "${MANIFEST_FILE}")
        IMAGE_DIGEST=$(sed -n '4s/^image-digest://p' "${MANIFEST_FILE}")

        mkdir -p artifacts/rocm-base-wheels
        find artifacts/rocm-base-wheels -mindepth 1 -depth -delete

        SET_PATH="${CACHE_PATH}sets/${SET_ID}/"
        while IFS= read -r wheel_name; do
            aws s3 cp \
                "${SET_PATH}${wheel_name}" \
                "artifacts/rocm-base-wheels/${wheel_name}"
        done < <(sed -E 's/^[0-9a-f]{64}  //' "${CHECKSUM_FILE}")

        (cd artifacts/rocm-base-wheels && sha256sum --check "${CHECKSUM_FILE}")
        validate_flat_wheel_directory artifacts/rocm-base-wheels

        echo ""
        echo "Downloaded wheels:"
        find artifacts/rocm-base-wheels -maxdepth 1 -name '*.whl' -exec ls -lh {} \;
        WHEEL_COUNT=$(find artifacts/rocm-base-wheels -maxdepth 1 -type f -name '*.whl' 2>/dev/null | wc -l)
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
