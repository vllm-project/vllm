#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Try building the docker image
docker build -t hpu-test-env -f Dockerfile.hpu .

# Setup cleanup
# certain versions of HPU software stack have a bug that can
# override the exit code of the script, so we need to use
# separate remove_docker_container and remove_docker_container_and_exit
# functions, while other platforms only need one remove_docker_container
# function.
EXITCODE=1
remove_docker_container() { docker rm -f hpu-test || true; }
remove_docker_container_and_exit() { remove_docker_container; exit $EXITCODE; }
trap remove_docker_container_and_exit EXIT
remove_docker_container

# Run the image and launch offline inference
docker run --runtime=habana --name=hpu-test --network=host -e HABANA_VISIBLE_DEVICES=all -e VLLM_SKIP_WARMUP=true --entrypoint="" hpu-test-env python3 examples/offline_inference/basic.py
EXITCODE=$?
