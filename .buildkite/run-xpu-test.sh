#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Try building the docker image
docker build -t xpu-test -f Dockerfile.xpu .

# Setup cleanup
remove_docker_container() { docker rm -f xpu-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image and test offline inference/tensor parallel
docker run --name xpu-test --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --entrypoint="" xpu-test sh -c '
    python3 examples/offline_inference.py
    python3 examples/offline_inference_cli.py -tp 2
'
