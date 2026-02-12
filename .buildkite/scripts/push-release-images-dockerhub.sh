#!/bin/bash

set -ex

RELEASE_VERSION=$(buildkite-agent meta-data get release-version 2>/dev/null | sed 's/^v//')
if [ -z "${RELEASE_VERSION}" ]; then
    echo "RELEASE_VERSION is not set"
    exit 1
fi

aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7

# Download images:

docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-x86_64
docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-aarch64
docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-x86_64-cu130
docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-aarch64-cu130
docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm-base
docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm
docker pull public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v${RELEASE_VERSION}
docker pull public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:v${RELEASE_VERSION}

# Tag and push images:

## CUDA

docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-x86_64 vllm/vllm-openai:x86_64
docker tag vllm/vllm-openai:x86_64 vllm/vllm-openai:latest-x86_64
docker tag vllm/vllm-openai:x86_64 vllm/vllm-openai:v${RELEASE_VERSION}-x86_64
docker push vllm/vllm-openai:latest-x86_64
docker push vllm/vllm-openai:v${RELEASE_VERSION}-x86_64

docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-x86_64-cu130 vllm/vllm-openai:x86_64-cu130
docker tag vllm/vllm-openai:x86_64-cu130 vllm/vllm-openai:latest-x86_64-cu130
docker tag vllm/vllm-openai:x86_64-cu130 vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-cu130
docker push vllm/vllm-openai:latest-x86_64-cu130
docker push vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-cu130

docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-aarch64 vllm/vllm-openai:aarch64
docker tag vllm/vllm-openai:aarch64 vllm/vllm-openai:latest-aarch64
docker tag vllm/vllm-openai:aarch64 vllm/vllm-openai:v${RELEASE_VERSION}-aarch64
docker push vllm/vllm-openai:latest-aarch64
docker push vllm/vllm-openai:v${RELEASE_VERSION}-aarch64

docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-aarch64-cu130 vllm/vllm-openai:aarch64-cu130
docker tag vllm/vllm-openai:aarch64-cu130 vllm/vllm-openai:latest-aarch64-cu130
docker tag vllm/vllm-openai:aarch64-cu130 vllm/vllm-openai:v${RELEASE_VERSION}-aarch64-cu130
docker push vllm/vllm-openai:latest-aarch64-cu130
docker push vllm/vllm-openai:v${RELEASE_VERSION}-aarch64-cu130

## ROCm

docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm vllm/vllm-openai-rocm:${BUILDKITE_COMMIT}
docker tag vllm/vllm-openai-rocm:${BUILDKITE_COMMIT} vllm/vllm-openai-rocm:latest
docker tag vllm/vllm-openai-rocm:${BUILDKITE_COMMIT} vllm/vllm-openai-rocm:v${RELEASE_VERSION}
docker push vllm/vllm-openai-rocm:latest
docker push vllm/vllm-openai-rocm:v${RELEASE_VERSION}

docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm-base vllm/vllm-openai-rocm:${BUILDKITE_COMMIT}-base
docker tag vllm/vllm-openai-rocm:${BUILDKITE_COMMIT}-base vllm/vllm-openai-rocm:latest-base
docker tag vllm/vllm-openai-rocm:${BUILDKITE_COMMIT}-base vllm/vllm-openai-rocm:v${RELEASE_VERSION}-base
docker push vllm/vllm-openai-rocm:latest-base
docker push vllm/vllm-openai-rocm:v${RELEASE_VERSION}-base

## CPU

docker tag public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v${RELEASE_VERSION} vllm/vllm-openai-cpu:x86_64
docker tag vllm/vllm-openai-cpu:x86_64 vllm/vllm-openai-cpu:latest-x86_64
docker tag vllm/vllm-openai-cpu:x86_64 vllm/vllm-openai-cpu:v${RELEASE_VERSION}-x86_64
docker push vllm/vllm-openai-cpu:latest-x86_64
docker push vllm/vllm-openai-cpu:v${RELEASE_VERSION}-x86_64

docker tag public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:v${RELEASE_VERSION} vllm/vllm-openai-cpu:arm64
docker tag vllm/vllm-openai-cpu:arm64 vllm/vllm-openai-cpu:latest-arm64
docker tag vllm/vllm-openai-cpu:arm64 vllm/vllm-openai-cpu:v${RELEASE_VERSION}-arm64
docker push vllm/vllm-openai-cpu:latest-arm64
docker push vllm/vllm-openai-cpu:v${RELEASE_VERSION}-arm64

# Create multi-arch manifest:

docker manifest rm vllm/vllm-openai:latest
docker manifest create vllm/vllm-openai:latest vllm/vllm-openai:latest-x86_64 vllm/vllm-openai:latest-aarch64
docker manifest create vllm/vllm-openai:v${RELEASE_VERSION} vllm/vllm-openai:v${RELEASE_VERSION}-x86_64 vllm/vllm-openai:v${RELEASE_VERSION}-aarch64
docker manifest push vllm/vllm-openai:latest
docker manifest push vllm/vllm-openai:v${RELEASE_VERSION}

docker manifest rm vllm/vllm-openai:latest-cu130
docker manifest create vllm/vllm-openai:latest-cu130 vllm/vllm-openai:latest-x86_64-cu130 vllm/vllm-openai:latest-aarch64-cu130
docker manifest create vllm/vllm-openai:v${RELEASE_VERSION}-cu130 vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-cu130 vllm/vllm-openai:v${RELEASE_VERSION}-aarch64-cu130
docker manifest push vllm/vllm-openai:latest-cu130
docker manifest push vllm/vllm-openai:v${RELEASE_VERSION}-cu130

docker manifest rm vllm/vllm-openai-cpu:latest || true
docker manifest create vllm/vllm-openai-cpu:latest vllm/vllm-openai-cpu:latest-x86_64 vllm/vllm-openai-cpu:latest-arm64
docker manifest create vllm/vllm-openai-cpu:v${RELEASE_VERSION} vllm/vllm-openai-cpu:v${RELEASE_VERSION}-x86_64 vllm/vllm-openai-cpu:v${RELEASE_VERSION}-arm64
docker manifest push vllm/vllm-openai-cpu:latest
docker manifest push vllm/vllm-openai-cpu:v${RELEASE_VERSION}
