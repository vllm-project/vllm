#!/bin/bash
set -e

if [[ $# -lt 8 ]]; then
  echo "Usage: $0 <registry> <repo> <commit> <branch> <vllm_use_precompiled> <vllm_merge_base_commit> <cache_from> <cache_to>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3
BRANCH=$4
VLLM_USE_PRECOMPILED=$5
VLLM_MERGE_BASE_COMMIT=$6
CACHE_FROM=$7
CACHE_TO=$8

# authenticate with AWS ECR
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin $REGISTRY
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 936637512419.dkr.ecr.us-east-1.amazonaws.com

# docker buildx 
docker buildx create --name vllm-builder --driver docker-container --use
docker buildx inspect --bootstrap
docker buildx ls

# skip build if image already exists
if [[ -z $(docker manifest inspect $REGISTRY/$REPO:$BUILDKITE_COMMIT) ]]; then
  echo "Image not found, proceeding with build..."
else
  echo "Image found"
  exit 0
fi

if [[ "${VLLM_USE_PRECOMPILED:-0}" == "1" ]]; then
  merge_base_commit_build_args="--build-arg VLLM_MERGE_BASE_COMMIT=${VLLM_MERGE_BASE_COMMIT}"
else
  merge_base_commit_build_args=""
fi

# build
docker buildx build --file docker/Dockerfile \
  --build-arg max_jobs=16 \
  --build-arg buildkite_commit=$BUILDKITE_COMMIT \
  --build-arg USE_SCCACHE=1 \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 10.0" \
  --build-arg FI_TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0a 10.0a" \
  --build-arg VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-0}" \
  ${merge_base_commit_build_args} \
  --cache-from type=registry,ref=${CACHE_FROM},mode=max \
  --cache-to type=registry,ref=${CACHE_TO},mode=max \
  --tag ${REGISTRY}/${REPO}:${BUILDKITE_COMMIT} \
  $( [[ "${BRANCH}" == "main" ]] && echo "--tag ${REGISTRY}/${REPO}:latest" ) \
  --push \
  --target test \
  --progress plain .
