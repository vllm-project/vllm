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

docker run --privileged --net host --shm-size=16G -i \
    -e "HF_TOKEN=$HF_TOKEN" --name tpu-test \
    vllm-tpu /bin/bash -c '
set -u # Enable error on unset variables within the container script

echo "--- Starting script inside Docker container ---"

# Create and verify results directory
RESULTS_DIR=$(mktemp -d)
if [ -z "$RESULTS_DIR" ] || [ ! -d "$RESULTS_DIR" ]; then
    echo "Critical Error: Failed to create temporary directory for results." >&2
    exit 1
fi
echo "Results will be stored in: $RESULTS_DIR"

# Install dependencies
echo "--- Installing Python dependencies ---"
python3 -m pip install --progress-bar off git+https://github.com/thuml/depyf.git \
    && python3 -m pip install --progress-bar off pytest pytest-asyncio tpu-info \
    && python3 -m pip install --progress-bar off lm_eval[api]==0.4.4
INSTALL_STATUS=$?
if [ $INSTALL_STATUS -ne 0 ]; then
    echo "Critical Error: Failed to install Python dependencies. Exit code: $INSTALL_STATUS" >&2
    exit 1
fi
echo "--- Python dependencies installed ---"

echo "--- Hardware Information ---"
tpu-info
echo "--- Starting Tests ---"

# --- Test Definitions ---
# Helper function to run a test and save its exit code
run_test() {
    local test_num=$1
    local test_name=$2
    local test_command=$3
    local log_file="$RESULTS_DIR/test_${test_num}.log"

    echo "TEST_$test_num: Running $test_name"
    # Execute the command, redirecting stdout/stderr to a log file and also to current stdout/stderr
    # (eval to correctly interpret the command string if it contains redirections or complex constructs, though not strictly needed for simple commands)
    eval "$test_command" > >(tee -a "$log_file") 2> >(tee -a "$log_file" >&2)
    local exit_code=$?
    echo $exit_code > "$RESULTS_DIR/test_${test_num}.exit"
    echo "TEST_${test_num}_COMMAND_EXIT_CODE: $exit_code" # This goes to main log
    # Also log to per-test log for completeness
    echo "TEST_${test_num}_COMMAND_EXIT_CODE: $exit_code" >> "$log_file"
}

# Launch tests in parallel
run_test 0 "test_perf.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/tpu/test_perf.py" &
run_test 1 "test_compilation.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/tpu/test_compilation.py" &
run_test 2 "test_basic.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_basic.py" &
run_test 3 "test_accuracy.py::test_lm_eval_accuracy_v1_engine" \
    "python3 -m pytest -s -v /workspace/vllm/tests/entrypoints/llm/test_accuracy.py::test_lm_eval_accuracy_v1_engine" &
run_test 4 "test_quantization_accuracy.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/tpu/test_quantization_accuracy.py" &
run_test 5 "examples/offline_inference/tpu.py" \
    "python3 /workspace/vllm/examples/offline_inference/tpu.py" &
run_test 6 "test_tpu_model_runner.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/tpu/worker/test_tpu_model_runner.py" &
run_test 7 "test_sampler.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_sampler.py" &
run_test 8 "test_topk_topp_sampler.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_topk_topp_sampler.py" &
run_test 9 "test_multimodal.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_multimodal.py" &
run_test 10 "test_pallas.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/tpu/test_pallas.py" &
run_test 11 "test_struct_output_generate.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/v1/entrypoints/llm/test_struct_output_generate.py" &
run_test 12 "test_moe_pallas.py" \
    "python3 -m pytest -s -v /workspace/vllm/tests/tpu/test_moe_pallas.py" &

# Disable the TPU LoRA tests until the feature is activated
# run_test 13 "test_lora (directory)" \
#     "python3 -m pytest -s -v /workspace/vllm/tests/tpu/lora/" &

# Wait for all background jobs (tests) to complete
wait
echo "--- All tests have completed. Checking exit codes... ---"

OVERALL_STATUS=0 # 0 for success, 1 for failure
EXPECTED_TESTS=13 # Number of tests defined above (0-12)
PROCESSED_FILES=0

for i in $(seq 0 $(($EXPECTED_TESTS - 1))); do
    exit_file="$RESULTS_DIR/test_$i.exit"
    log_file="$RESULTS_DIR/test_$i.log" # For easier debugging

    echo "Processing results for TEST_$i (file: $exit_file)..."
    if [ ! -f "$exit_file" ]; then
        echo "Error: Result file $exit_file not found for TEST_$i!"
        OVERALL_STATUS=1
        # You might want to output content of $log_file here if it exists
        if [ -f "$log_file" ]; then echo "Partial log for TEST_$i:"; cat "$log_file"; fi
        continue
    fi

    PROCESSED_FILES=$((PROCESSED_FILES + 1))
    current_exit_code=$(cat "$exit_file")

    if ! [[ "$current_exit_code" =~ ^-?[0-9]+$ ]]; then
        echo "Error: Result file $exit_file for TEST_$i contains non-numeric value: '\''$current_exit_code'\''"
        OVERALL_STATUS=1
        if [ -f "$log_file" ]; then echo "Log for TEST_$i:"; cat "$log_file"; fi
        continue
    fi

    if [ "$current_exit_code" -ne 0 ]; then
        echo "TEST_$i FAILED with exit code $current_exit_code."
        OVERALL_STATUS=1
        # Output log for failed test
        if [ -f "$log_file" ]; then echo "Log for failed TEST_$i:"; cat "$log_file"; fi
    else
        echo "TEST_$i PASSED."
    fi
done

if [ "$PROCESSED_FILES" -ne "$EXPECTED_TESTS" ]; then
    echo "Error: Expected $EXPECTED_TESTS result files, but only processed $PROCESSED_FILES."
    OVERALL_STATUS=1
fi

echo "--- Final Check ---"
if [ $OVERALL_STATUS -ne 0 ]; then
    echo "One or more tests failed or critical errors occurred."
    exit 1
else
    echo "All tests passed successfully."
    exit 0
fi
' # End of the command string for docker run -c

# Capture the exit code of the docker run command
DOCKER_RUN_EXIT_CODE=$?

# The trap will run for cleanup.
# Exit the main script with the Docker run command's exit code.
if [ $DOCKER_RUN_EXIT_CODE -ne 0 ]; then
    echo "Docker run command failed with exit code $DOCKER_RUN_EXIT_CODE."
    exit $DOCKER_RUN_EXIT_CODE
else
    echo "Docker run command completed successfully."
    exit 0
fi
# TODO: This test fails because it uses RANDOM_SEED sampling
# && VLLM_USE_V1=1 pytest -v -s /workspace/vllm/tests/tpu/test_custom_dispatcher.py \