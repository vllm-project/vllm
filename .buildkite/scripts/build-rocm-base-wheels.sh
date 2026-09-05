#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Run from the repository root to build or reuse paired ROCm base image/wheels.
set -euo pipefail

if [[ "${SCCACHE_ENDPOINT:-}" == *[@?#]* ||
      "${SCCACHE_ENDPOINT:-}" =~ [[:space:][:cntrl:]] ]]; then
    echo "ERROR: SCCACHE_ENDPOINT must not contain credentials, query parameters, fragments, or whitespace" >&2
    exit 1
fi

TEMP_FILES=()
WHEEL_CONTAINER=""
cleanup() {
    if [[ -n "${WHEEL_CONTAINER}" ]]; then
        docker rm -f "${WHEEL_CONTAINER}" >/dev/null || true
    fi
    if ((${#TEMP_FILES[@]})); then
        rm -f -- "${TEMP_FILES[@]}"
    fi
}
trap cleanup EXIT

# Keep the cache key, image, and exported wheels on the same release inputs.
export ROCM_BASE_DOCKERFILE=docker/Dockerfile.rocm_base
export ROCM_BASE_CONTENT_FILES="docker/Dockerfile.rocm_base .dockerignore requirements/build/rocm.txt"
export ROCM_BASE_IMAGE_TARGET=final
export ROCM_BASE_WHEEL_TARGET=debs_wheel_release
export ROCM_BASE_PLATFORM=linux/amd64
export ROCM_BASE_USE_SCCACHE=1
export SCCACHE_BUCKET_NAME=vllm-build-sccache
export SCCACHE_REGION_NAME=us-west-2
export SCCACHE_S3_NO_CREDENTIALS=0
# Build ARGs are public: release builds use the checksum-pinned download default.
unset ROCM_BASE_PARENT_DIGEST SCCACHE_DOWNLOAD_URL

dockerfile_arg_default() {
    sed -n -E "s/^ARG ${1}=\"?([^\"[:space:]]+)\"?.*/\\1/p" \
        "${ROCM_BASE_DOCKERFILE}" | head -1
}

PYTORCH_ROCM_ARCH="${ROCM_BASE_PYTORCH_ROCM_ARCH:-${PYTORCH_ROCM_ARCH:-$(dockerfile_arg_default PYTORCH_ROCM_ARCH)}}"
SCCACHE_VERSION="${SCCACHE_VERSION:-$(dockerfile_arg_default SCCACHE_VERSION)}"
SCCACHE_DOWNLOAD_SHA256="${SCCACHE_DOWNLOAD_SHA256:-$(dockerfile_arg_default SCCACHE_DOWNLOAD_SHA256)}"
export PYTORCH_ROCM_ARCH SCCACHE_VERSION SCCACHE_DOWNLOAD_SHA256

ROCM_BASE_CACHE_POLICY_ARGS=()
if [[ "${ROCM_BASE_WHEEL_CACHE_FORCE:-0}" == "1" ]]; then
    ROCM_BASE_CACHE_POLICY_ARGS+=(--no-cache)
elif [[ "${ROCM_BASE_WHEEL_CACHE_FORCE:-0}" != "0" ]]; then
    echo "ERROR: ROCM_BASE_WHEEL_CACHE_FORCE must be 0 or 1" >&2
    exit 1
fi

PINNED_BASE_IMAGE=$(.buildkite/scripts/cache-rocm-base-wheels.sh parent)
export ROCM_BASE_PARENT_IMAGE="${PINNED_BASE_IMAGE%@*}"
export ROCM_BASE_PARENT_DIGEST="${PINNED_BASE_IMAGE##*@}"
CACHE_KEY=$(.buildkite/scripts/cache-rocm-base-wheels.sh key)
ECR_REPOSITORY="public.ecr.aws/q9t5s3a7/vllm-release-repo"
ECR_CACHE_TAG="${ECR_REPOSITORY}:${CACHE_KEY}-rocm-base"
ECR_IMAGE_REF=""

echo "ROCm Base Build Configuration"
echo "  CACHE_KEY: ${CACHE_KEY}"
echo "  BASE_IMAGE: ${PINNED_BASE_IMAGE}"
echo "  ECR_CACHE_TAG: ${ECR_CACHE_TAG}"

aws ecr-public get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7

IMAGE_EXISTS=false
WHEELS_EXIST=false
WHEEL_CACHE_STATUS=$(.buildkite/scripts/cache-rocm-base-wheels.sh check)
if [[ "${WHEEL_CACHE_STATUS}" == "hit" ]]; then
    WHEELS_EXIST=true
    echo "S3 wheels cache HIT"

    # Validate the immutable wheel set before trusting its paired image digest.
    CACHED_IMAGE_DIGEST_FILE=$(mktemp)
    TEMP_FILES+=("${CACHED_IMAGE_DIGEST_FILE}")
    export ROCM_BASE_IMAGE_DIGEST_FILE="${CACHED_IMAGE_DIGEST_FILE}"
    if ! .buildkite/scripts/cache-rocm-base-wheels.sh download; then
        WHEELS_EXIST=false
        echo "S3 wheel cache validation failed; rebuilding"
    else
        CACHED_IMAGE_DIGEST=$(cat "${CACHED_IMAGE_DIGEST_FILE}")
        ECR_IMAGE_REF="${ECR_REPOSITORY}@${CACHED_IMAGE_DIGEST}"
        if docker manifest inspect "${ECR_IMAGE_REF}" >/dev/null 2>&1; then
            IMAGE_EXISTS=true
            echo "Paired ECR image cache HIT"
        else
            echo "Paired ECR image digest is unavailable; rebuilding"
        fi
    fi
    unset ROCM_BASE_IMAGE_DIGEST_FILE
fi

if [[ "${IMAGE_EXISTS}" == "true" && "${WHEELS_EXIST}" == "true" ]]; then
    echo "FULL CACHE HIT - Reusing both image and wheels"
else
    echo "CACHE MISS - Building from scratch..."
    ROCM_BASE_BUILD_ARGS=(
        --file "${ROCM_BASE_DOCKERFILE}"
        --platform "${ROCM_BASE_PLATFORM}"
        --build-arg "BASE_IMAGE=${PINNED_BASE_IMAGE}"
        --build-arg "USE_SCCACHE=${ROCM_BASE_USE_SCCACHE}"
        --build-arg "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
        --build-arg "SCCACHE_VERSION=${SCCACHE_VERSION}"
        --build-arg "SCCACHE_DOWNLOAD_SHA256=${SCCACHE_DOWNLOAD_SHA256}"
        --build-arg "SCCACHE_ENDPOINT=${SCCACHE_ENDPOINT:-}"
        --build-arg "SCCACHE_BUCKET_NAME=${SCCACHE_BUCKET_NAME}"
        --build-arg "SCCACHE_REGION_NAME=${SCCACHE_REGION_NAME}"
        --build-arg "SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS}"
    )

    ROCM_BASE_BUILD_METADATA=$(mktemp)
    TEMP_FILES+=("${ROCM_BASE_BUILD_METADATA}")
    DOCKER_BUILDKIT=1 docker buildx build \
        "${ROCM_BASE_BUILD_ARGS[@]}" \
        "${ROCM_BASE_CACHE_POLICY_ARGS[@]}" \
        --tag "${ECR_CACHE_TAG}" \
        --target "${ROCM_BASE_IMAGE_TARGET}" \
        --label "vllm.rocm_base.cache_key=${CACHE_KEY}" \
        --label "vllm.rocm_base.pytorch_rocm_arch=${PYTORCH_ROCM_ARCH}" \
        --label "vllm.rocm_base.sccache_version=${SCCACHE_VERSION}" \
        --label "vllm.rocm_base.sccache_sha256=${SCCACHE_DOWNLOAD_SHA256}" \
        --metadata-file "${ROCM_BASE_BUILD_METADATA}" \
        --push \
        .
    ECR_IMAGE_DIGEST=$(awk -F'"' \
        '$2 == "containerimage.digest" { print $4; exit }' \
        "${ROCM_BASE_BUILD_METADATA}")
    if [[ ! "${ECR_IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
        echo "ERROR: build returned an invalid ROCm base image digest" >&2
        exit 1
    fi
    ECR_IMAGE_REF="${ECR_REPOSITORY}@${ECR_IMAGE_DIGEST}"
    export ROCM_BASE_IMAGE_DIGEST="${ECR_IMAGE_DIGEST}"

    # Reuse the image build's layers even when that build was forced.
    DOCKER_BUILDKIT=1 docker buildx build \
        "${ROCM_BASE_BUILD_ARGS[@]}" \
        --tag "rocm-base-debs:${BUILDKITE_BUILD_NUMBER}" \
        --target "${ROCM_BASE_WHEEL_TARGET}" \
        --load \
        .

    mkdir -p artifacts/rocm-base-wheels
    find artifacts/rocm-base-wheels -mindepth 1 -depth -delete
    WHEEL_CONTAINER=$(docker create "rocm-base-debs:${BUILDKITE_BUILD_NUMBER}")
    docker cp "${WHEEL_CONTAINER}:/app/debs/." artifacts/rocm-base-wheels/
    docker rm "${WHEEL_CONTAINER}"
    WHEEL_CONTAINER=""
    .buildkite/scripts/cache-rocm-base-wheels.sh upload
    echo "Build complete - Image and wheels cached"
fi

if [[ ! "${ECR_IMAGE_REF}" =~ ^public\.ecr\.aws/q9t5s3a7/vllm-release-repo@sha256:[0-9a-f]{64}$ ]]; then
    echo "ERROR: refusing unexpected ROCm base image ref: ${ECR_IMAGE_REF}" >&2
    exit 1
fi
buildkite-agent meta-data set "rocm-base-image-tag" "${ECR_IMAGE_REF}"
