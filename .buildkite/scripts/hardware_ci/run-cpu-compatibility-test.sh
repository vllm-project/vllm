#!/bin/bash
set -euox pipefail

export VLLM_CPU_KVCACHE_SPACE=1 
export VLLM_CPU_CI_ENV=1
# Reduce sub-processes for acceleration
export TORCH_COMPILE_DISABLE=1 
export VLLM_ENABLE_V1_MULTIPROCESSING=0

SDE_ARCHIVE="sde-external-10.7.0-2026-02-18-lin.tar.xz"
SDE_CHECKSUM="CA3D4086DE4ACB3FAEDF9F57B541C6936B7D5E19AE2BF763B6EA933573A0A217"
wget "https://downloadmirror.intel.com/913594/${SDE_ARCHIVE}"
echo "${SDE_CHECKSUM}  ${SDE_ARCHIVE}" | sha256sum --check
mkdir -p sde
tar -xvf "./${SDE_ARCHIVE}" --strip-components=1 -C ./sde/

wait_for_pid_and_check_log() {
    local pid="$1"
    local log_file="$2"
    local exit_status

    if [ -z "$pid" ] || [ -z "$log_file" ]; then
        echo "Usage: wait_for_pid_and_check_log <PID> <LOG_FILE>"
        return 1
    fi

    echo "Waiting for process $pid to finish..."
    
    # Use the 'wait' command to pause the script until the specific PID exits.
    # The 'wait' command's own exit status will be that of the waited-for process.
    if wait "$pid"; then
        exit_status=$?
        echo "Process $pid finished with exit status $exit_status (Success)."
    else
        exit_status=$?
        echo "Process $pid finished with exit status $exit_status (Failure)."
    fi

    if [ "$exit_status" -ne 0 ]; then
        echo "Process exited with a non-zero status."
        echo "--- Last few lines of log file: $log_file ---"
        tail -n 50 "$log_file"
        echo "---------------------------------------------"
        return 1 # Indicate failure based on exit status
    fi

    echo "No errors detected in log file and process exited successfully."
    return 0
}

# Test Sky Lake (AVX512F)
./sde/sde64 -skl -- python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --dtype bfloat16 > test_0.log 2>&1 &
PID_TEST_0=$!

# Test Cascade Lake (AVX512F + VNNI)
./sde/sde64 -clx -- python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --dtype bfloat16 > test_1.log 2>&1 &
PID_TEST_1=$!

# Test Cooper Lake (AVX512F + VNNI + BF16)
./sde/sde64 -cpx -- python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --dtype bfloat16 > test_2.log 2>&1 &
PID_TEST_2=$!

wait_for_pid_and_check_log $PID_TEST_0 test_0.log
wait_for_pid_and_check_log $PID_TEST_1 test_1.log
wait_for_pid_and_check_log $PID_TEST_2 test_2.log
