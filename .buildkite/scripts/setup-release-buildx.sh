#!/bin/bash
# Set up a registry-cache-capable buildx builder for CUDA release image builds.
#
# Release jobs historically used classic `docker build`, which only supports
# inline cache (mode=min) and cannot cache the intermediate csrc-build compile
# layer -- the expensive part. To reuse layers across releases/nightlies we need
# a buildkit-backed builder that can export a mode=max registry cache.
#
# This mirrors .buildkite/image_build/image_build.sh: prefer the warm-AMI
# standalone buildkitd (remote driver) when its socket is present; otherwise
# fall back to a docker-container builder. The default `docker` driver cannot
# export registry cache, so we never fall back to it here.
#
# Also logs in to the private ECR that hosts the release BuildKit cache
# (vllm-release-cache); the release queues are granted read/write by ci-infra.
set -euo pipefail

CACHE_REGISTRY="936637512419.dkr.ecr.us-east-1.amazonaws.com"
BUILDKIT_SOCKET="/run/buildkit/buildkitd.sock"

echo "--- :docker: Logging in to private ECR for release cache"
aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin "${CACHE_REGISTRY}"

echo "--- :buildkite: Selecting buildx builder"
if [[ -S "${BUILDKIT_SOCKET}" ]]; then
    echo "Found warm-AMI buildkitd at ${BUILDKIT_SOCKET}; using remote driver"
    docker buildx use baked-vllm-builder 2>/dev/null \
        || docker buildx create --name baked-vllm-builder --driver remote --use "unix://${BUILDKIT_SOCKET}"
else
    echo "No buildkitd socket; using a docker-container builder (cold cache)"
    docker buildx use release-cache-builder 2>/dev/null \
        || docker buildx create --name release-cache-builder --driver docker-container --use
fi
docker buildx inspect --bootstrap
