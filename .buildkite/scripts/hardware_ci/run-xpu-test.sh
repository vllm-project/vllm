#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

image_name="xpu/vllm-ci:${BUILDKITE_COMMIT}"
container_name="xpu_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"

# Try building the docker image
docker build -t ${image_name} -f docker/Dockerfile.xpu .

# Setup cleanup
remove_docker_container() { 
  docker rm -f "${container_name}" || true; 
  docker image rm -f "${image_name}" || true;
  docker system prune -f || true;
}
trap remove_docker_container EXIT

# Run the image and test offline inference/tensor parallel
docker run \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --entrypoint="" \
    --name "${container_name}" \
    "${image_name}" \
    sh -c '
    VLLM_USE_V1=0 python3 examples/offline_inference/basic/generate.py --model facebook/opt-125m
    VLLM_USE_V1=0 python3 examples/offline_inference/basic/generate.py --model facebook/opt-125m -tp 2
    VLLM_USE_V1=1 python3 examples/offline_inference/basic/generate.py --model facebook/opt-125m --block-size 64 --enforce-eager
'
