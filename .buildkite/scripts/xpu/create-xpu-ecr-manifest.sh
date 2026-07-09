#!/bin/bash

set -euo pipefail

REGISTRY="public.ecr.aws/q9t5s3a7"
REPO="vllm-release-repo"
ARCH_TAG="${BUILDKITE_COMMIT}-$(uname -m)-xpu"
PLATFORM_TAG="${BUILDKITE_COMMIT}-xpu"

aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${REGISTRY}
docker manifest rm ${REGISTRY}/${REPO}:${PLATFORM_TAG} || true
docker manifest create ${REGISTRY}/${REPO}:${PLATFORM_TAG} ${REGISTRY}/${REPO}:${ARCH_TAG} --amend
docker manifest push ${REGISTRY}/${REPO}:${PLATFORM_TAG}