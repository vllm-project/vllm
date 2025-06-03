# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys

import pytest
import torch


def run_python_script(script_name, timeout):
    script_name = f'kv_transfer/{script_name}'
    try:
        # Start both processes asynchronously using Popen
        process0 = subprocess.Popen(
            [sys.executable, script_name],
            env={"RANK":
                 "0"},  # Set the RANK environment variable for process 0
            stdout=sys.stdout,  # Pipe stdout to current stdout
            stderr=sys.stderr,  # Pipe stderr to current stderr
        )

        process1 = subprocess.Popen(
            [sys.executable, script_name],
            env={"RANK":
                 "1"},  # Set the RANK environment variable for process 1
            stdout=sys.stdout,  # Pipe stdout to current stdout
            stderr=sys.stderr,  # Pipe stderr to current stderr
        )

        # Wait for both processes to complete, with a timeout
        process0.wait(timeout=timeout)
        process1.wait(timeout=timeout)

        # Check the return status of both processes
        if process0.returncode != 0:
            pytest.fail(
                f"Test {script_name} failed for RANK=0, {process0.returncode}")
        if process1.returncode != 0:
            pytest.fail(
                f"Test {script_name} failed for RANK=1, {process1.returncode}")

    except subprocess.TimeoutExpired:
        # If either process times out, terminate both and fail the test
        process0.terminate()
        process1.terminate()
        pytest.fail(f"Test {script_name} timed out")
    except Exception as e:
        pytest.fail(f"Test {script_name} failed with error: {str(e)}")


# Define the test cases using pytest's parametrize
@pytest.mark.parametrize(
    "script_name,timeout",
    [
        ("test_lookup_buffer.py",
         60),  # Second test case with a 60-second timeout
        ("test_send_recv.py", 120)  # First test case with a 120-second timeout
    ])
def test_run_python_script(script_name, timeout):
    # Check the number of GPUs
    if torch.cuda.device_count() < 2:
        pytest.skip(
            f"Skipping test {script_name} because <2 GPUs are available")

    # Run the test if there are at least 2 GPUs
    run_python_script(script_name, timeout)
