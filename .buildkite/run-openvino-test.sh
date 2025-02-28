#!/bin/bash

# This script build the OpenVINO docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Try building the docker image
docker build -t openvino-test -f Dockerfile.openvino .

# Setup cleanup
remove_docker_container() { docker rm -f openvino-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image and launch offline inference
docker run --network host --env VLLM_OPENVINO_KVCACHE_SPACE=1 --name openvino-test openvino-test python3 /workspace/examples/offline_inference/basic/generate.py --model facebook/opt-125m
