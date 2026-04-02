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
if [[ -z $(docker manifest inspect "$REGISTRY"/"$REPO":"$BUILDKITE_COMMIT"-cpu) ]]; then
  echo "Image not found, proceeding with build..."
else
  echo "Image found"
  exit 0
fi

# The rust frontend lives in a git submodule under rust/. Buildkite's default
# checkout does not recurse submodules, and the Dockerfile only sees what's in
# the build context, so initialize the submodule here before building.
git submodule sync --recursive
git submodule update --init --recursive

# build
docker build --file docker/Dockerfile.cpu \
  --build-arg max_jobs=16 \
  --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
  --build-arg VLLM_CPU_X86=true \
  --tag "$REGISTRY"/"$REPO":"$BUILDKITE_COMMIT"-cpu \
  --target vllm-test \
  --progress plain .

# push
docker push "$REGISTRY"/"$REPO":"$BUILDKITE_COMMIT"-cpu
