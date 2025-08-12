#!/bin/bash

set -e

# Build the docker image.
docker build -f Dockerfile.tpu -t vllm-tpu .

# Set up cleanup.
remove_docker_container() { docker rm -f tpu-test || true; }
trap remove_docker_container EXIT
# Remove the container that might not be cleaned up in the previous run.
remove_docker_container

# For HF_TOKEN.
source /etc/environment
# Run a simple end-to-end example.
docker run --privileged --net host --shm-size=16G -it \
    -e "HF_TOKEN=$HF_TOKEN" --name tpu-test \
    vllm-tpu /bin/bash -c "python3 -m pip install git+https://github.com/thuml/depyf.git \
    && python3 -m pip install pytest \
    && python3 -m pip install lm_eval[api]==0.4.4 \
    && export VLLM_USE_V1=1 \
    && export VLLM_XLA_CHECK_RECOMPILATION=1 \
    && echo TEST_1 \
    && pytest /workspace/vllm/tests/tpu/test_compilation.py \
    && echo TEST_2 \
    && pytest -v -s /workspace/vllm/tests/v1/tpu/test_basic.py \
    && echo TEST_3 \
    && pytest -v -s /workspace/vllm/tests/entrypoints/llm/test_accuracy.py::test_lm_eval_accuracy_v1_engine \
    && echo TEST_4 \
    && pytest -s -v /workspace/vllm/tests/tpu/test_quantization_accuracy.py \
    && echo TEST_5 \
    && python3 /workspace/vllm/examples/offline_inference/tpu.py" \


# TODO: This test fails because it uses RANDOM_SEED sampling
# && VLLM_USE_V1=1 pytest -v -s /workspace/vllm/tests/tpu/test_custom_dispatcher.py \

