#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

MODE=${1:?Usage: publish-vllm-bench-images.sh nightly|release}
COMMIT=${BUILDKITE_COMMIT:?BUILDKITE_COMMIT is required}
STAGING_REPO=public.ecr.aws/q9t5s3a7/vllm-release-repo
DOCKERHUB_REPO=vllm/vllm-bench

case "$MODE" in
  nightly)
    CURRENT_TAG=nightly
    IMMUTABLE_TAG=nightly-${COMMIT}
    ;;
  release)
    RELEASE_VERSION=$(buildkite-agent meta-data get release-version --default "" | sed 's/^v//')
    if [ -z "$RELEASE_VERSION" ]; then
      echo "ERROR: release-version metadata not set"
      exit 1
    fi
    CURRENT_TAG=latest
    IMMUTABLE_TAG=v${RELEASE_VERSION}
    ;;
  *)
    echo "ERROR: mode must be nightly or release"
    exit 1
    ;;
esac

aws ecr-public get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7

for ARCH in x86_64 aarch64; do
  SOURCE=${STAGING_REPO}:${COMMIT}-vllm-bench-${ARCH}
  docker pull "$SOURCE"
  docker tag "$SOURCE" "${DOCKERHUB_REPO}:${CURRENT_TAG}-${ARCH}"
  docker push "${DOCKERHUB_REPO}:${CURRENT_TAG}-${ARCH}"

  if [ "$MODE" = release ]; then
    docker tag "$SOURCE" "${DOCKERHUB_REPO}:${IMMUTABLE_TAG}-${ARCH}"
    docker push "${DOCKERHUB_REPO}:${IMMUTABLE_TAG}-${ARCH}"
  fi
done

docker manifest rm "${DOCKERHUB_REPO}:${CURRENT_TAG}" || true
docker manifest rm "${DOCKERHUB_REPO}:${IMMUTABLE_TAG}" || true
docker manifest create "${DOCKERHUB_REPO}:${CURRENT_TAG}" \
  "${DOCKERHUB_REPO}:${CURRENT_TAG}-x86_64" \
  "${DOCKERHUB_REPO}:${CURRENT_TAG}-aarch64"

if [ "$MODE" = release ]; then
  docker manifest create "${DOCKERHUB_REPO}:${IMMUTABLE_TAG}" \
    "${DOCKERHUB_REPO}:${IMMUTABLE_TAG}-x86_64" \
    "${DOCKERHUB_REPO}:${IMMUTABLE_TAG}-aarch64"
else
  docker manifest create "${DOCKERHUB_REPO}:${IMMUTABLE_TAG}" \
    "${DOCKERHUB_REPO}:${CURRENT_TAG}-x86_64" \
    "${DOCKERHUB_REPO}:${CURRENT_TAG}-aarch64"
fi

docker manifest push "${DOCKERHUB_REPO}:${CURRENT_TAG}"
docker manifest push "${DOCKERHUB_REPO}:${IMMUTABLE_TAG}"
