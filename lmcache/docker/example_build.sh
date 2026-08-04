#!/usr/bin/env bash
# Example script to build the LMCache integrated with vLLM container image

# Update the following variables accordingly
CUDA_VERSION=13.0
DOCKERFILE_NAME='Dockerfile'
# Set VLLM_VERSION to a specific version before running this script,
# e.g.: VLLM_VERSION=0.9.1 ./example_build.sh
# Defaults to "nightly" if not set.
VLLM_VERSION="${VLLM_VERSION:-nightly}"
DOCKER_BUILD_PATH='../' # This path should point to the LMCache root for access to 'requirements' directory
UBUNTU_VERSION=24.04

# `image-build` target will use the latest LMCache and vLLM code
# Change to 'image-release' target for using release package versions of vLLM and LMCache
BUILD_TARGET=image-build 

IMAGE_TAG='lmcache/vllm-openai:build-latest' # Name of container image to build

docker build \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg UBUNTU_VERSION=$UBUNTU_VERSION \
    --build-arg VLLM_VERSION=$VLLM_VERSION \
    --target $BUILD_TARGET --file $DOCKERFILE_NAME \
    --tag $IMAGE_TAG  $DOCKER_BUILD_PATH
