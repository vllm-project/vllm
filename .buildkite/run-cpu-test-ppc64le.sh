#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test || true; docker system prune -f; }
trap remove_docker_container EXIT
remove_docker_container

# Try building the docker image
docker build -t cpu-test -f Dockerfile.ppc64le .

