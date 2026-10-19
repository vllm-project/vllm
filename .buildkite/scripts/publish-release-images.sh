#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Publish a release Docker image family from ECR to DockerHub.
# Pulls per-arch images, tags with latest and versioned tags, pushes them,
# then creates and pushes multi-arch manifests.

set -euo pipefail

TARGET="${1:-all}"
case "${TARGET}" in
  cuda-13-0 | cuda-12-9 | cuda-13-0-ubuntu-24-04 | \
    cuda-12-9-ubuntu-24-04 | rocm | rocm72 | xpu | cpu | all) ;;
  *)
    echo "Usage: $0 {cuda-13-0|cuda-12-9|cuda-13-0-ubuntu-24-04|cuda-12-9-ubuntu-24-04|rocm|rocm72|xpu|cpu|all}"
    exit 2
    ;;
esac

target_enabled() {
  [ "${TARGET}" = "all" ] || [ "${TARGET}" = "$1" ]
}

RELEASE_VERSION=$(buildkite-agent meta-data get release-version --default "" | sed 's/^v//')
if [ -z "${RELEASE_VERSION}" ]; then
  echo "ERROR: release-version metadata not set"
  exit 1
fi

COMMIT="$BUILDKITE_COMMIT"

echo "========================================"
echo "Publishing ${TARGET} release images v${RELEASE_VERSION}"
echo "  Commit: ${COMMIT}"
echo "========================================"

# Login to ECR to pull staging images
aws ecr-public get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7

# ---- CUDA (default: 13.0) ----

if target_enabled cuda-13-0; then
  docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64"
  docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-aarch64"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64" vllm/vllm-openai:latest-x86_64
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64" "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64"
  docker push vllm/vllm-openai:latest-x86_64
  docker push "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-aarch64" vllm/vllm-openai:latest-aarch64
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-aarch64" "vllm/vllm-openai:v${RELEASE_VERSION}-aarch64"
  docker push vllm/vllm-openai:latest-aarch64
  docker push "vllm/vllm-openai:v${RELEASE_VERSION}-aarch64"

  docker manifest rm vllm/vllm-openai:latest || true
  docker manifest rm "vllm/vllm-openai:v${RELEASE_VERSION}" || true
  docker manifest create vllm/vllm-openai:latest vllm/vllm-openai:latest-x86_64 vllm/vllm-openai:latest-aarch64
  docker manifest create "vllm/vllm-openai:v${RELEASE_VERSION}" "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64" "vllm/vllm-openai:v${RELEASE_VERSION}-aarch64"
  docker manifest push vllm/vllm-openai:latest
  docker manifest push "vllm/vllm-openai:v${RELEASE_VERSION}"

  ZSTD_DIGEST=$(docker buildx imagetools inspect \
    "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64" \
    --format '{{json .Manifest.Digest}}' | tr -d '"')
  .buildkite/scripts/publish-zstd-image.sh \
    "public.ecr.aws/q9t5s3a7/vllm-release-repo@${ZSTD_DIGEST}" \
    "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-zstd"
fi

# ---- CUDA 12.9 ----

if target_enabled cuda-12-9; then
  docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-cu129"
  docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-aarch64-cu129"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-cu129" vllm/vllm-openai:latest-x86_64-cu129
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-cu129" "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-cu129"
  docker push vllm/vllm-openai:latest-x86_64-cu129
  docker push "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-cu129"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-aarch64-cu129" vllm/vllm-openai:latest-aarch64-cu129
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-aarch64-cu129" "vllm/vllm-openai:v${RELEASE_VERSION}-aarch64-cu129"
  docker push vllm/vllm-openai:latest-aarch64-cu129
  docker push "vllm/vllm-openai:v${RELEASE_VERSION}-aarch64-cu129"

  docker manifest rm vllm/vllm-openai:latest-cu129 || true
  docker manifest rm "vllm/vllm-openai:v${RELEASE_VERSION}-cu129" || true
  docker manifest create vllm/vllm-openai:latest-cu129 vllm/vllm-openai:latest-x86_64-cu129 vllm/vllm-openai:latest-aarch64-cu129
  docker manifest create "vllm/vllm-openai:v${RELEASE_VERSION}-cu129" "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-cu129" "vllm/vllm-openai:v${RELEASE_VERSION}-aarch64-cu129"
  docker manifest push vllm/vllm-openai:latest-cu129
  docker manifest push "vllm/vllm-openai:v${RELEASE_VERSION}-cu129"
fi

# ---- Ubuntu 24.04 (CUDA 13.0) ----

if target_enabled cuda-13-0-ubuntu-24-04; then
  docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-ubuntu2404"
  docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-aarch64-ubuntu2404"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-ubuntu2404" vllm/vllm-openai:latest-x86_64-ubuntu2404
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-ubuntu2404" "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-ubuntu2404"
  docker push vllm/vllm-openai:latest-x86_64-ubuntu2404
  docker push "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-ubuntu2404"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-aarch64-ubuntu2404" vllm/vllm-openai:latest-aarch64-ubuntu2404
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-aarch64-ubuntu2404" "vllm/vllm-openai:v${RELEASE_VERSION}-aarch64-ubuntu2404"
  docker push vllm/vllm-openai:latest-aarch64-ubuntu2404
  docker push "vllm/vllm-openai:v${RELEASE_VERSION}-aarch64-ubuntu2404"

  docker manifest rm vllm/vllm-openai:latest-ubuntu2404 || true
  docker manifest rm "vllm/vllm-openai:v${RELEASE_VERSION}-ubuntu2404" || true
  docker manifest create vllm/vllm-openai:latest-ubuntu2404 vllm/vllm-openai:latest-x86_64-ubuntu2404 vllm/vllm-openai:latest-aarch64-ubuntu2404
  docker manifest create "vllm/vllm-openai:v${RELEASE_VERSION}-ubuntu2404" "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-ubuntu2404" "vllm/vllm-openai:v${RELEASE_VERSION}-aarch64-ubuntu2404"
  docker manifest push vllm/vllm-openai:latest-ubuntu2404
  docker manifest push "vllm/vllm-openai:v${RELEASE_VERSION}-ubuntu2404"
fi

# ---- Ubuntu 24.04 (CUDA 12.9) ----

if target_enabled cuda-12-9-ubuntu-24-04; then
  docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-cu129-ubuntu2404"
  docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-aarch64-cu129-ubuntu2404"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-cu129-ubuntu2404" vllm/vllm-openai:latest-x86_64-cu129-ubuntu2404
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-cu129-ubuntu2404" "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-cu129-ubuntu2404"
  docker push vllm/vllm-openai:latest-x86_64-cu129-ubuntu2404
  docker push "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-cu129-ubuntu2404"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-aarch64-cu129-ubuntu2404" vllm/vllm-openai:latest-aarch64-cu129-ubuntu2404
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-aarch64-cu129-ubuntu2404" "vllm/vllm-openai:v${RELEASE_VERSION}-aarch64-cu129-ubuntu2404"
  docker push vllm/vllm-openai:latest-aarch64-cu129-ubuntu2404
  docker push "vllm/vllm-openai:v${RELEASE_VERSION}-aarch64-cu129-ubuntu2404"

  docker manifest rm vllm/vllm-openai:latest-cu129-ubuntu2404 || true
  docker manifest rm "vllm/vllm-openai:v${RELEASE_VERSION}-cu129-ubuntu2404" || true
  docker manifest create vllm/vllm-openai:latest-cu129-ubuntu2404 vllm/vllm-openai:latest-x86_64-cu129-ubuntu2404 vllm/vllm-openai:latest-aarch64-cu129-ubuntu2404
  docker manifest create "vllm/vllm-openai:v${RELEASE_VERSION}-cu129-ubuntu2404" "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-cu129-ubuntu2404" "vllm/vllm-openai:v${RELEASE_VERSION}-aarch64-cu129-ubuntu2404"
  docker manifest push vllm/vllm-openai:latest-cu129-ubuntu2404
  docker manifest push "vllm/vllm-openai:v${RELEASE_VERSION}-cu129-ubuntu2404"
fi

# ---- ROCm ----
# Each ROCm stack is identified by its base Dockerfile (see rocm/stack.sh). The
# default stack also owns the plain :latest / :v<ver> tags.

publish_rocm_stack() {
  local base_dockerfile="$1" default="$2" suffix
  (
    # shellcheck source=.buildkite/scripts/rocm/stack.sh
    source .buildkite/scripts/rocm/stack.sh "$base_dockerfile"
    local suffixes=("-${ROCM_STACK_VARIANT}")
    [[ "$default" == "1" ]] && suffixes=("" "-${ROCM_STACK_VARIANT}")
    docker pull "$ROCM_STACK_ECR_IMAGE"
    docker pull "$ROCM_STACK_ECR_BASE"
    for suffix in "${suffixes[@]}"; do
      for tag in "latest${suffix}" "v${RELEASE_VERSION}${suffix}"; do
        docker tag "$ROCM_STACK_ECR_IMAGE" "vllm/vllm-openai-rocm:${tag}"
        docker tag "$ROCM_STACK_ECR_BASE" "vllm/vllm-openai-rocm:${tag}-base"
        docker push "vllm/vllm-openai-rocm:${tag}"
        docker push "vllm/vllm-openai-rocm:${tag}-base"
      done
    done
  )
}

if target_enabled rocm; then
  publish_rocm_stack docker/Dockerfile.rocm_base 1
fi

if target_enabled rocm72; then
  publish_rocm_stack docker/Dockerfile.rocm_72_base 0
fi

# ---- XPU ----

if target_enabled xpu; then
  docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-xpu"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-xpu" vllm/vllm-openai-xpu:latest-x86_64
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-xpu" "vllm/vllm-openai-xpu:v${RELEASE_VERSION}-x86_64"
  docker push vllm/vllm-openai-xpu:latest-x86_64
  docker push "vllm/vllm-openai-xpu:v${RELEASE_VERSION}-x86_64"

  docker manifest rm vllm/vllm-openai-xpu:latest || true
  docker manifest rm "vllm/vllm-openai-xpu:v${RELEASE_VERSION}" || true
  docker manifest create vllm/vllm-openai-xpu:latest vllm/vllm-openai-xpu:latest-x86_64 --amend
  docker manifest create "vllm/vllm-openai-xpu:v${RELEASE_VERSION}" "vllm/vllm-openai-xpu:v${RELEASE_VERSION}-x86_64" --amend
  docker manifest push vllm/vllm-openai-xpu:latest
  docker manifest push "vllm/vllm-openai-xpu:v${RELEASE_VERSION}"
fi

# ---- CPU ----
# CPU images are behind separate block steps and may not have been built.
# All-or-nothing: inspect both arches first, then either publish everything
# (per-arch + multi-arch manifest) or skip everything. Publishing only one
# arch would leave `:latest-x86_64` pointing at the new release while the
# `:latest` multi-arch manifest still resolves to the previous release.

if target_enabled cpu; then
  CPU_X86_TAG=public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:${COMMIT}-x86_64
  CPU_ARM_TAG=public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:${COMMIT}-arm64

  CPU_X86_AVAILABLE=false
  CPU_ARM_AVAILABLE=false
  docker manifest inspect "${CPU_X86_TAG}" >/dev/null 2>&1 && CPU_X86_AVAILABLE=true
  docker manifest inspect "${CPU_ARM_TAG}" >/dev/null 2>&1 && CPU_ARM_AVAILABLE=true

  if [ "$CPU_X86_AVAILABLE" = "true" ] && [ "$CPU_ARM_AVAILABLE" = "true" ]; then
    docker pull "${CPU_X86_TAG}"
    docker tag "${CPU_X86_TAG}" vllm/vllm-openai-cpu:latest-x86_64
    docker tag "${CPU_X86_TAG}" "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-x86_64"
    docker push vllm/vllm-openai-cpu:latest-x86_64
    docker push "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-x86_64"

    docker pull "${CPU_ARM_TAG}"
    docker tag "${CPU_ARM_TAG}" vllm/vllm-openai-cpu:latest-arm64
    docker tag "${CPU_ARM_TAG}" "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-arm64"
    docker push vllm/vllm-openai-cpu:latest-arm64
    docker push "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-arm64"

    docker manifest rm vllm/vllm-openai-cpu:latest || true
    docker manifest rm "vllm/vllm-openai-cpu:v${RELEASE_VERSION}" || true
    docker manifest create vllm/vllm-openai-cpu:latest vllm/vllm-openai-cpu:latest-x86_64 vllm/vllm-openai-cpu:latest-arm64
    docker manifest create "vllm/vllm-openai-cpu:v${RELEASE_VERSION}" "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-x86_64" "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-arm64"
    docker manifest push vllm/vllm-openai-cpu:latest
    docker manifest push "vllm/vllm-openai-cpu:v${RELEASE_VERSION}"
  elif [ "$CPU_X86_AVAILABLE" = "false" ] && [ "$CPU_ARM_AVAILABLE" = "false" ]; then
    echo "WARNING: Neither CPU image found in ECR, skipping CPU publish (ensure block-cpu-release-image-build and block-arm64-cpu-release-image-build were unblocked and the builds finished pushing)"
  else
    # Partial state: one arch built, the other did not. Fail loudly rather than
    # ship a Docker Hub state where `:latest-${arch}` and `:latest` (multi-arch)
    # disagree on which release they point at.
    echo "ERROR: Partial CPU build detected (x86_64=${CPU_X86_AVAILABLE}, arm64=${CPU_ARM_AVAILABLE})."
    echo "       Refusing to publish to avoid split-tag drift between per-arch and multi-arch tags."
    echo "       Re-run the missing CPU build and retry, or manually publish if a single-arch release is intended."
    exit 1
  fi
fi

echo ""
echo "Successfully published ${TARGET} release images for v${RELEASE_VERSION}"
