#!/bin/bash

set -xue

# Build the docker image.
docker build -f docker/Dockerfile.tpu -t vllm-tpu .

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
    && python3 -m pip install pytest pytest-asyncio tpu-info \
    && python3 -m pip install lm_eval[api]==0.4.4 \
    && export VLLM_USE_V1=1 \
    && export VLLM_XLA_CHECK_RECOMPILATION=1 \
    && echo HARDWARE \
    && tpu-info \
    && echo TEST_0 \
    && pytest -v -s /workspace/vllm/tests/v1/tpu/test_perf.py \
    && echo TEST_1 \
    && pytest -v -s /workspace/vllm/tests/tpu/test_compilation.py \
    && echo TEST_2 \
    && pytest -v -s /workspace/vllm/tests/v1/tpu/test_basic.py \
    && echo TEST_3 \
    && pytest -v -s /workspace/vllm/tests/entrypoints/llm/test_accuracy.py::test_lm_eval_accuracy_v1_engine \
    && echo TEST_4 \
    && pytest -s -v /workspace/vllm/tests/tpu/test_quantization_accuracy.py \
    && echo TEST_5 \
    && python3 /workspace/vllm/examples/offline_inference/tpu.py \
    && echo TEST_6 \
    && pytest -s -v /workspace/vllm/tests/v1/tpu/worker/test_tpu_model_runner.py \
    && echo TEST_7 \
    && pytest -s -v /workspace/vllm/tests/v1/tpu/test_sampler.py \
    && echo TEST_8 \
    && pytest -s -v /workspace/vllm/tests/v1/tpu/test_topk_topp_sampler.py \
    && echo TEST_9 \
    && pytest -s -v /workspace/vllm/tests/v1/tpu/test_multimodal.py \
    && echo TEST_10 \
    && pytest -s -v /workspace/vllm/tests/v1/tpu/test_pallas.py \
    && echo TEST_11 \
    && pytest -s -v /workspace/vllm/tests/v1/entrypoints/llm/test_struct_output_generate.py" \


# TODO: This test fails because it uses RANDOM_SEED sampling
# && VLLM_USE_V1=1 pytest -v -s /workspace/vllm/tests/tpu/test_custom_dispatcher.py \
