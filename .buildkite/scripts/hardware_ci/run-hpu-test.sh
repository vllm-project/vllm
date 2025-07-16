#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -exuo pipefail

# Try building the docker image
cat <<EOF | docker build -t hpu-plugin-v1-test-env -f - .
FROM gaudi-base-image:latest

COPY ./ /workspace/vllm

WORKDIR /workspace/vllm

ENV no_proxy=localhost,127.0.0.1
ENV PT_HPU_ENABLE_LAZY_COLLECTIVES=true

RUN VLLM_TARGET_DEVICE=empty pip install .
RUN pip install git+https://github.com/vllm-project/vllm-gaudi.git

# install development dependencies (for testing)
RUN python3 -m pip install -e tests/vllm_test_utils

WORKDIR /workspace/

RUN git clone https://github.com/vllm-project/vllm-gaudi.git

RUN ln -s /workspace/vllm/tests && ln -s /workspace/vllm/examples && ln -s /workspace/vllm/benchmarks

EOF

# Setup cleanup
# certain versions of HPU software stack have a bug that can
# override the exit code of the script, so we need to use
# separate remove_docker_containers and remove_docker_containers_and_exit
# functions, while other platforms only need one remove_docker_container
# function.
EXITCODE=1
remove_docker_containers() { docker rm -f hpu-plugin-v1-test || true; }
trap 'remove_docker_containers; exit $EXITCODE;' EXIT
remove_docker_containers

echo "Running HPU plugin v1 test"
docker run --rm --runtime=habana --name=hpu-plugin-v1-test --network=host \
  -e HABANA_VISIBLE_DEVICES=all \
  hpu-plugin-v1-test-env \
  /bin/bash "/workspace/vllm-gaudi/tests/upstream_tests/ci_tests.sh"

EXITCODE=$?
if [ $EXITCODE -eq 0 ]; then
  echo "Test with basic model passed"
else
  echo "Test with basic model FAILED with exit code: $EXITCODE" >&2
fi

# The trap will handle the container removal and final exit.