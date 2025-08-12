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

docker run --privileged --net host --shm-size=16G -it \
    -e "HF_TOKEN=$HF_TOKEN" --name tpu-test \
    vllm-tpu /bin/bash -c '
set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error.

echo "--- Starting script inside Docker container ---"

# Create results directory
RESULTS_DIR=$(mktemp -d)
# If mktemp fails, set -e will cause the script to exit.
echo "Results will be stored in: $RESULTS_DIR"

# Install dependencies
echo "--- Installing Python dependencies ---"
python3 -m pip install --progress-bar off git+https://github.com/thuml/depyf.git \
    && python3 -m pip install --progress-bar off pytest pytest-asyncio tpu-info \
    && python3 -m pip install --progress-bar off lm_eval[api]==0.4.4
echo "--- Python dependencies installed ---"
export VLLM_USE_V1=1
export VLLM_XLA_CHECK_RECOMPILATION=1
export VLLM_XLA_CACHE_PATH=
echo "Using VLLM V1"

echo "--- Hardware Information ---"
tpu-info
echo "--- Starting Tests ---"
set +e
overall_script_exit_code=0

# --- Test Definitions ---
# If a test fails, this function will print logs and will not cause the main script to exit.
run_test() {
    local test_num=$1
    local test_name=$2
    local test_command=$3
    local log_file="$RESULTS_DIR/test_${test_num}.log"
    local actual_exit_code

    echo "--- TEST_$test_num: Running $test_name ---"
    
    # Execute the test command.
    eval "$test_command" > >(tee -a "$log_file") 2> >(tee -a "$log_file" >&2)
    actual_exit_code=$?

    echo "TEST_${test_num}_COMMAND_EXIT_CODE: $actual_exit_code" # This goes to main log
    echo "TEST_${test_num}_COMMAND_EXIT_CODE: $actual_exit_code" >> "$log_file" # Also to per-test log

    if [ "$actual_exit_code" -ne 0 ]; then
        echo "TEST_$test_num ($test_name) FAILED with exit code $actual_exit_code." >&2
        echo "--- Log for failed TEST_$test_num ($test_name) ---" >&2
        if [ -f "$log_file" ]; then
            cat "$log_file" >&2
        else
            echo "Log file $log_file not found for TEST_$test_num ($test_name)." >&2
        fi
        echo "--- End of log for TEST_$test_num ($test_name) ---" >&2
        return "$actual_exit_code" # Return the failure code
    else
        echo "TEST_$test_num ($test_name) PASSED."
        return 0 # Return success
    fi
}

# Helper function to call run_test and update the overall script exit code
run_and_track_test() {
    local test_num_arg="$1"
    local test_name_arg="$2"
    local test_command_arg="$3"

    # Run the test
    run_test "$test_num_arg" "$test_name_arg" "$test_command_arg"
    local test_specific_exit_code=$?

    # If the test failed, set the overall script exit code to 1
    if [ "$test_specific_exit_code" -ne 0 ]; then
        # No need for extra echo here, run_test already logged the failure.
        overall_script_exit_code=1
    fi
}

# --- Actual Test Execution ---
run_and_track_test 0 "test_perf.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_perf.py"
run_and_track_test 1 "test_compilation.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/tpu/test_compilation.py"
run_and_track_test 2 "test_basic.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_basic.py"
run_and_track_test 3 "test_accuracy.py::test_lm_eval_accuracy_v1_engine" \
    "python3 -m pytest -s -v /workspace/vllm/tests/entrypoints/llm/test_accuracy.py::test_lm_eval_accuracy_v1_engine"
run_and_track_test 4 "test_quantization_accuracy.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/tpu/test_quantization_accuracy.py"
run_and_track_test 5 "examples/offline_inference/tpu.py" \
    "python3 /workspace/vllm/examples/offline_inference/tpu.py"
run_and_track_test 6 "test_tpu_model_runner.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/worker/test_tpu_model_runner.py"
run_and_track_test 7 "test_sampler.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_sampler.py"
run_and_track_test 8 "test_topk_topp_sampler.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_topk_topp_sampler.py"
run_and_track_test 9 "test_multimodal.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_multimodal.py"
run_and_track_test 10 "test_pallas.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_pallas.py"
run_and_track_test 11 "test_struct_output_generate.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/entrypoints/llm/test_struct_output_generate.py"
run_and_track_test 12 "test_moe_pallas.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/tpu/test_moe_pallas.py"

# Disable the TPU LoRA tests until the feature is activated
# run_and_track_test 13 "test_lora (directory)" \
#     "python3 -m pytest -s -v /workspace/vllm/tests/tpu/lora/"

# After all tests have been attempted, exit with the overall status.
if [ "$overall_script_exit_code" -ne 0 ]; then
    echo "--- One or more tests FAILED. Overall script exiting with failure code 1. ---"
else
    echo "--- All tests have completed and PASSED. Overall script exiting with success code 0. ---"
fi
exit "$overall_script_exit_code"
' # IMPORTANT: This is the closing single quote for the bash -c "..." command. Ensure it is present and correct.

# Capture the exit code of the docker run command
DOCKER_RUN_EXIT_CODE=$?

# The trap will run for cleanup.
# Exit the main script with the Docker run command's exit code.
if [ "$DOCKER_RUN_EXIT_CODE" -ne 0 ]; then
    echo "Docker run command failed with exit code $DOCKER_RUN_EXIT_CODE."
    exit "$DOCKER_RUN_EXIT_CODE"
else
    echo "Docker run command completed successfully."
    exit 0
fi
# TODO: This test fails because it uses RANDOM_SEED sampling
# pytest -v -s /workspace/vllm/tests/tpu/test_custom_dispatcher.py \
