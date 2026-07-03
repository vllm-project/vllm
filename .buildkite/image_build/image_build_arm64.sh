#!/bin/bash
set -e

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3

# authenticate with AWS ECR
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY" || true

# skip build if image already exists
if [[ -z $(docker manifest inspect "$REGISTRY"/"$REPO":"$BUILDKITE_COMMIT"-arm64) ]]; then
  echo "Image not found, proceeding with build..."
else
  echo "Image found"
  exit 0
fi

# build for arm64 GPU targets: Grace/GH200 (sm_90) and DGX Spark/GB10
# (sm_121, family-covered by 12.0 under CUDA 13)
docker build --file docker/Dockerfile \
  --platform linux/arm64 \
  --build-arg max_jobs=16 \
  --build-arg nvcc_threads=4 \
  --build-arg torch_cuda_arch_list="9.0 12.0" \
  --build-arg USE_SCCACHE=1 \
  --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
  --tag "$REGISTRY"/"$REPO":"$BUILDKITE_COMMIT"-arm64 \
  --target test \
  --progress plain .

# push
docker push "$REGISTRY"/"$REPO":"$BUILDKITE_COMMIT"-arm64
