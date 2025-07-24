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
    VLLM_USE_V1=1 python3 examples/offline_inference/basic/generate.py --model facebook/opt-125m --block-size 64 --enforce-eager
    VLLM_USE_V1=1 python3 examples/offline_inference/basic/generate.py --model facebook/opt-125m --block-size 64 --enforce-eager -tp 2 --distributed-executor-backend ray
    VLLM_USE_V1=1 python3 examples/offline_inference/basic/generate.py --model facebook/opt-125m --block-size 64 --enforce-eager -tp 2 --distributed-executor-backend mp
    cd tests
    pytest -v -s v1/core
    pytest -v -s v1/engine
    pytest -v -s v1/sample --ignore=v1/sample/test_logprobs.py --ignore=v1/sample/test_logprobs_e2e.py
    pytest -v -s v1/worker --ignore=v1/worker/test_gpu_model_runner.py
    pytest -v -s v1/structured_output
    pytest -v -s v1/spec_decode --ignore=v1/spec_decode/test_max_len.py --ignore=v1/spec_decode/test_eagle.py
    pytest -v -s v1/kv_connector/unit --ignore=v1/kv_connector/unit/test_multi_connector.py --ignore=v1/kv_connector/unit/test_nixl_connector.py
    pytest -v -s v1/test_serial_utils.py
    pytest -v -s v1/test_utils.py
    pytest -v -s v1/test_metrics_reader.py
'
