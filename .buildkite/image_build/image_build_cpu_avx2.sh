#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Builds an x86_64 CPU image targeting AVX2 only (no AVX-512).
# This image is compatible with GitHub Actions runners, which may
# land on AMD EPYC Zen 1-3 CPUs that lack AVX-512 support.

set -e

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3

# authenticate with AWS ECR
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY"

# skip build if image already exists
if [[ -z $(docker manifest inspect "$REGISTRY"/"$REPO":"$BUILDKITE_COMMIT"-cpu-avx2) ]]; then
  echo "Image not found, proceeding with build..."
else
  echo "Image found"
  exit 0
fi

# build
docker build --file docker/Dockerfile.cpu \
  --build-arg max_jobs=16 \
  --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
  --build-arg VLLM_CPU_DISABLE_AVX512=true \
  --build-arg VLLM_CPU_AVX2=1 \
  --tag "$REGISTRY"/"$REPO":"$BUILDKITE_COMMIT"-cpu-avx2 \
  --target vllm-test \
  --progress plain .

# push
docker push "$REGISTRY"/"$REPO":"$BUILDKITE_COMMIT"-cpu-avx2
