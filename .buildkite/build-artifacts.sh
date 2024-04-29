#!/usr/env/bin bash

# This script is used to build Python wheels and Docker Images for vLLM.
# We will use container to build wheels to ensure reproducibility and isolation

set -ex

export DOCKER_BUILDKIT=1
docker build --build-arg max_jobs=16 \
    --build-arg python_version=3.8 \
    --target build \
    -f Dockerfile \
    --progress plain \
    -t vllm-build \
    .

# copy the wheels to the host, in the container it is located in /workspace/dist
docker run --rm -v $(pwd):/host vllm-build cp -r /workspace/dist /host

ls dist
