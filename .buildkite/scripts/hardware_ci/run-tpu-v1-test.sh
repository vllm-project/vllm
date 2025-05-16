#!/bin/bash

set -xu

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
    && export VLLM_XLA_CACHE_PATH= \
    && export VLLM_USE_V1=1 \
    && export VLLM_XLA_CHECK_RECOMPILATION=1 \
    && echo HARDWARE \
    && tpu-info \
    && { \
        echo TEST_0: Running test_perf.py; \
        python3 -m pytest -s -v /workspace/vllm/tests/tpu/test_perf.py; \
        echo TEST_0_EXIT_CODE: \$?; \
    } & \
    { \
        echo TEST_1: Running test_compilation.py; \
        python3 -m pytest -s -v /workspace/vllm/tests/tpu/test_compilation.py; \
        echo TEST_1_EXIT_CODE: \$?; \
    } & \
    { \
        echo TEST_2: Running test_basic.py; \
        python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_basic.py; \
        echo TEST_2_EXIT_CODE: \$?; \
    } & \
    { \
        echo TEST_3: Running test_accuracy.py::test_lm_eval_accuracy_v1_engine; \
        python3 -m pytest -s -v /workspace/vllm/tests/entrypoints/llm/test_accuracy.py::test_lm_eval_accuracy_v1_engine; \
        echo TEST_3_EXIT_CODE: \$?; \
    } & \
    { \
        echo TEST_4: Running test_quantization_accuracy.py; \
        python3 -m pytest -s -v /workspace/vllm/tests/tpu/test_quantization_accuracy.py; \
        echo TEST_4_EXIT_CODE: \$?; \
    } & \
    { \
        echo TEST_5: Running examples/offline_inference/tpu.py; \
        python3 /workspace/vllm/examples/offline_inference/tpu.py; \
        echo TEST_5_EXIT_CODE: \$?; \
    } & \
    { \
        echo TEST_6: Running test_tpu_model_runner.py; \
        python3 -m pytest -s -v /workspace/vllm/tests/tpu/worker/test_tpu_model_runner.py; \
        echo TEST_6_EXIT_CODE: \$?; \
    } & \
    { \
        echo TEST_7: Running test_sampler.py; \
        python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_sampler.py; \
        echo TEST_7_EXIT_CODE: \$?; \
    } & \
    { \
        echo TEST_8: Running test_topk_topp_sampler.py; \
        python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_topk_topp_sampler.py; \
        echo TEST_8_EXIT_CODE: \$?; \
    } & \
    { \
        echo TEST_9: Running test_multimodal.py; \
        python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_multimodal.py; \
        echo TEST_9_EXIT_CODE: \$?; \
    } & \
    { \
        echo TEST_10: Running test_pallas.py; \
        python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_pallas.py; \
        echo TEST_10_EXIT_CODE: \$?; \
    } & \
    { \
        echo TEST_11: Running test_struct_output_generate.py; \
        python3 -m pytest -s -v /workspace/vllm/tests/v1/entrypoints/llm/test_struct_output_generate.py; \
        echo TEST_11_EXIT_CODE: \$?; \
    } & \
    { \
        echo TEST_12: Running test_moe_pallas.py; \
        python3 -m pytest -s -v /workspace/vllm/tests/tpu/test_moe_pallas.py; \
        echo TEST_12_EXIT_CODE: \$?; \
    } & \
    # Disable the TPU LoRA tests until the feature is activated
    # & { \
    #     echo TEST_13: Running test_moe_pallas.py; \
    #     python3 -m pytest -s -v /workspace/vllm/tests/tpu/lora/; \
    #     echo TEST_13_EXIT_CODE: \$?; \
    # } & \
    wait \
    && echo 'All tests have attempted to run. Check logs for individual test statuses and exit codes.' \
"

# TODO: This test fails because it uses RANDOM_SEED sampling
# && VLLM_USE_V1=1 pytest -v -s /workspace/vllm/tests/tpu/test_custom_dispatcher.py \
