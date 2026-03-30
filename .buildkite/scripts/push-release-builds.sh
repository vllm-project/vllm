#!/bin/bash

set -euo pipefail

# Ensure git tags are up-to-date (Buildkite's default fetch doesn't always include tags)
echo "Fetching latest tags from origin..."
git fetch --tags --force origin

# Derive release version from the git tag on the current commit.
# The pipeline must be triggered on a tagged commit (e.g. v0.18.1).
RELEASE_VERSION=$(git describe --exact-match --tags "${BUILDKITE_COMMIT}" 2>/dev/null || true)
if [ -z "${RELEASE_VERSION}" ]; then
    echo "[FATAL] Commit ${BUILDKITE_COMMIT} has no exact git tag. " \
         "Release images must be published from a tagged commit."
    exit 1
fi

# Strip leading 'v' for use in Docker tags (e.g. v0.18.1 -> 0.18.1)
PURE_VERSION="${RELEASE_VERSION#v}"

echo "========================================"
echo "Publishing release images"
echo "  Commit:          ${BUILDKITE_COMMIT}"
echo "  Release version: ${RELEASE_VERSION}"
echo "========================================"

set -x

# Login to ECR to pull source images
aws ecr-public get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7

# ---- CUDA (default, CUDA 12.9) ----
docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-x86_64"
docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-aarch64"

docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-x86_64" "vllm/vllm-openai:latest-x86_64"
docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-x86_64" "vllm/vllm-openai:v${PURE_VERSION}-x86_64"
docker push "vllm/vllm-openai:latest-x86_64"
docker push "vllm/vllm-openai:v${PURE_VERSION}-x86_64"

docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-aarch64" "vllm/vllm-openai:latest-aarch64"
docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-aarch64" "vllm/vllm-openai:v${PURE_VERSION}-aarch64"
docker push "vllm/vllm-openai:latest-aarch64"
docker push "vllm/vllm-openai:v${PURE_VERSION}-aarch64"

docker manifest rm "vllm/vllm-openai:latest" || true
docker manifest create "vllm/vllm-openai:latest" "vllm/vllm-openai:latest-x86_64" "vllm/vllm-openai:latest-aarch64"
docker manifest push "vllm/vllm-openai:latest"

docker manifest rm "vllm/vllm-openai:v${PURE_VERSION}" || true
docker manifest create "vllm/vllm-openai:v${PURE_VERSION}" "vllm/vllm-openai:v${PURE_VERSION}-x86_64" "vllm/vllm-openai:v${PURE_VERSION}-aarch64"
docker manifest push "vllm/vllm-openai:v${PURE_VERSION}"

# ---- CUDA 13.0 ----
docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-x86_64-cu130"
docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-aarch64-cu130"

docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-x86_64-cu130" "vllm/vllm-openai:latest-x86_64-cu130"
docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-x86_64-cu130" "vllm/vllm-openai:v${PURE_VERSION}-x86_64-cu130"
docker push "vllm/vllm-openai:latest-x86_64-cu130"
docker push "vllm/vllm-openai:v${PURE_VERSION}-x86_64-cu130"

docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-aarch64-cu130" "vllm/vllm-openai:latest-aarch64-cu130"
docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-aarch64-cu130" "vllm/vllm-openai:v${PURE_VERSION}-aarch64-cu130"
docker push "vllm/vllm-openai:latest-aarch64-cu130"
docker push "vllm/vllm-openai:v${PURE_VERSION}-aarch64-cu130"

docker manifest rm "vllm/vllm-openai:latest-cu130" || true
docker manifest create "vllm/vllm-openai:latest-cu130" "vllm/vllm-openai:latest-x86_64-cu130" "vllm/vllm-openai:latest-aarch64-cu130"
docker manifest push "vllm/vllm-openai:latest-cu130"

docker manifest rm "vllm/vllm-openai:v${PURE_VERSION}-cu130" || true
docker manifest create "vllm/vllm-openai:v${PURE_VERSION}-cu130" "vllm/vllm-openai:v${PURE_VERSION}-x86_64-cu130" "vllm/vllm-openai:v${PURE_VERSION}-aarch64-cu130"
docker manifest push "vllm/vllm-openai:v${PURE_VERSION}-cu130"

# ---- ROCm ----
docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm"
docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm-base"

docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm" "vllm/vllm-openai-rocm:latest"
docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm" "vllm/vllm-openai-rocm:v${PURE_VERSION}"
docker push "vllm/vllm-openai-rocm:latest"
docker push "vllm/vllm-openai-rocm:v${PURE_VERSION}"

docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm-base" "vllm/vllm-openai-rocm:latest-base"
docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm-base" "vllm/vllm-openai-rocm:v${PURE_VERSION}-base"
docker push "vllm/vllm-openai-rocm:latest-base"
docker push "vllm/vllm-openai-rocm:v${PURE_VERSION}-base"

# ---- CPU ----
# CPU images in ECR are tagged with the full version including 'v' (e.g. v0.18.1),
# matching the value from the Buildkite release-version metadata input.
docker pull "public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:${RELEASE_VERSION}"
docker pull "public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:${RELEASE_VERSION}"

docker tag "public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:${RELEASE_VERSION}" "vllm/vllm-openai-cpu:latest-x86_64"
docker tag "public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:${RELEASE_VERSION}" "vllm/vllm-openai-cpu:v${PURE_VERSION}-x86_64"
docker push "vllm/vllm-openai-cpu:latest-x86_64"
docker push "vllm/vllm-openai-cpu:v${PURE_VERSION}-x86_64"

docker tag "public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:${RELEASE_VERSION}" "vllm/vllm-openai-cpu:latest-arm64"
docker tag "public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:${RELEASE_VERSION}" "vllm/vllm-openai-cpu:v${PURE_VERSION}-arm64"
docker push "vllm/vllm-openai-cpu:latest-arm64"
docker push "vllm/vllm-openai-cpu:v${PURE_VERSION}-arm64"

docker manifest rm "vllm/vllm-openai-cpu:latest" || true
docker manifest create "vllm/vllm-openai-cpu:latest" "vllm/vllm-openai-cpu:latest-x86_64" "vllm/vllm-openai-cpu:latest-arm64"
docker manifest push "vllm/vllm-openai-cpu:latest"

docker manifest rm "vllm/vllm-openai-cpu:v${PURE_VERSION}" || true
docker manifest create "vllm/vllm-openai-cpu:v${PURE_VERSION}" "vllm/vllm-openai-cpu:v${PURE_VERSION}-x86_64" "vllm/vllm-openai-cpu:v${PURE_VERSION}-arm64"
docker manifest push "vllm/vllm-openai-cpu:v${PURE_VERSION}"

echo "========================================"
echo "Successfully published release images for ${RELEASE_VERSION}"
echo "========================================"
