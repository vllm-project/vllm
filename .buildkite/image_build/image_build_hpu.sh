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
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin $REGISTRY

# skip build if image already exists
if [[ -z $(docker manifest inspect $REGISTRY/$REPO:$BUILDKITE_COMMIT-hpu) ]]; then
  echo "Image not found, proceeding with build..."
else
  echo "Image found"
  exit 0
fi

# build
docker build \
  --file tests/pytorch_ci_hud_benchmark/Dockerfile.hpu \
  --build-arg max_jobs=16 \
  --build-arg buildkite_commit=$BUILDKITE_COMMIT \
  --tag $REGISTRY/$REPO:$BUILDKITE_COMMIT-hpu \
  --progress plain \
  https://github.com/vllm-project/vllm-gaudi.git

# push
docker push $REGISTRY/$REPO:$BUILDKITE_COMMIT-hpu
