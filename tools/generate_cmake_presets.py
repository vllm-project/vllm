# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import multiprocessing
import os
import sys
from shutil import which

try:
    # Try to get CUDA_HOME from PyTorch installation, which is the
    # most reliable source of truth for vLLM's build.
    from torch.utils.cpp_extension import CUDA_HOME
except ImportError:
    print("Warning: PyTorch not found. Falling back to CUDA_HOME environment variable.")
    CUDA_HOME = os.environ.get("CUDA_HOME")


def get_python_executable():
    """Get the current Python executable, which is used to run this script."""
    return sys.executable


def get_cpu_cores():
    """Get the number of CPU cores."""
    return multiprocessing.cpu_count()


def generate_presets(output_path="CMakeUserPresets.json", force_overwrite=False):
    """Generates the CMakeUserPresets.json file."""

    print("Attempting to detect your system configuration...")

    # Detect NVCC
    nvcc_path = None
    if CUDA_HOME:
        prospective_path = os.path.join(CUDA_HOME, "bin", "nvcc")
        if os.path.exists(prospective_path):
            nvcc_path = prospective_path
            print(f"Found nvcc via torch.utils.cpp_extension.CUDA_HOME: {nvcc_path}")

    if not nvcc_path:
        nvcc_path = which("nvcc")
        if nvcc_path:
            print(f"Found nvcc in PATH: {nvcc_path}")

    if not nvcc_path:
        nvcc_path_input = input(
            "Could not automatically find 'nvcc'. Please provide the full "
            "path to nvcc (e.g., /usr/local/cuda/bin/nvcc): "
        )
        nvcc_path = nvcc_path_input.strip()
    print(f"Using NVCC path: {nvcc_path}")

    # Detect Python executable
    python_executable = get_python_executable()
    if python_executable:
        print(f"Found Python via sys.executable: {python_executable}")
    else:
        python_executable_prompt = (
            "Could not automatically find Python executable. Please provide "
            "the full path to your Python executable for vLLM development "
            "(typically from your virtual environment, e.g., "
            "/home/user/venvs/vllm/bin/python): "
        )
        python_executable = input(python_executable_prompt).strip()
        if not python_executable:
            raise ValueError(
                "Could not determine Python executable. Please provide it manually."
            )

    print(f"Using Python executable: {python_executable}")

    # Get CPU cores
    cpu_cores = get_cpu_cores()
    nvcc_threads = min(4, cpu_cores)
    cmake_jobs = max(1, cpu_cores // nvcc_threads)
    print(
        f"Detected {cpu_cores} CPU cores. "
        f"Setting NVCC_THREADS={nvcc_threads} and CMake jobs={cmake_jobs}."
    )

    # Get vLLM project root (assuming this script is in vllm/tools/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"VLLM project root detected as: {project_root}")

    # Ensure python_executable path is absolute or resolvable
    if not os.path.isabs(python_executable) and which(python_executable):
        python_executable = os.path.abspath(which(python_executable))
    elif not os.path.isabs(python_executable):
        print(
            f"Warning: Python executable '{python_executable}' is not an "
            "absolute path and not found in PATH. CMake might not find it."
        )

    cache_variables = {
        "CMAKE_CUDA_COMPILER": nvcc_path,
        "CMAKE_BUILD_TYPE": "Release",
        "VLLM_PYTHON_EXECUTABLE": python_executable,
        "CMAKE_INSTALL_PREFIX": "${sourceDir}",
        "CMAKE_CUDA_FLAGS": "",
        "NVCC_THREADS": str(nvcc_threads),
    }

    # Detect compiler cache
    if which("sccache"):
        print("Using sccache for compiler caching.")
        for launcher in ("C", "CXX", "CUDA", "HIP"):
            cache_variables[f"CMAKE_{launcher}_COMPILER_LAUNCHER"] = "sccache"
    elif which("ccache"):
        print("Using ccache for compiler caching.")
        for launcher in ("C", "CXX", "CUDA", "HIP"):
            cache_variables[f"CMAKE_{launcher}_COMPILER_LAUNCHER"] = "ccache"
    else:
        print("No compiler cache ('ccache' or 'sccache') found.")

    configure_preset = {
        "name": "release",
        "binaryDir": "${sourceDir}/cmake-build-release",
        "cacheVariables": cache_variables,
    }
    if which("ninja"):
        print("Using Ninja generator.")
        configure_preset["generator"] = "Ninja"
        cache_variables["CMAKE_JOB_POOLS"] = f"compile={cmake_jobs}"
    else:
        print("Ninja not found, using default generator. Build may be slower.")

    presets = {
        "version": 6,
        # Keep in sync with CMakeLists.txt and requirements/build.txt
        "cmakeMinimumRequired": {"major": 3, "minor": 26, "patch": 1},
        "configurePresets": [configure_preset],
        "buildPresets": [
            {
                "name": "release",
                "configurePreset": "release",
                "jobs": cmake_jobs,
            }
        ],
    }

    output_file_path = os.path.join(project_root, output_path)

    if os.path.exists(output_file_path):
        if force_overwrite:
            print(f"Overwriting existing file '{output_file_path}'")
        else:
            overwrite = (
                input(f"'{output_file_path}' already exists. Overwrite? (y/N): ")
                .strip()
                .lower()
            )
            if overwrite != "y":
                print("Generation cancelled.")
                return

    try:
        with open(output_file_path, "w") as f:
            json.dump(presets, f, indent=4)
        print(f"Successfully generated '{output_file_path}'")
        print("\nTo use this preset:")
        print(f"1. Ensure you are in the vLLM root directory: cd {project_root}")
        print("2. Initialize CMake: cmake --preset release")
        print("3. Build+install: cmake --build --preset release --target install")

    except OSError as e:
        print(f"Error writing file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Force overwrite existing CMakeUserPresets.json without prompting",
    )

    args = parser.parse_args()
    generate_presets(force_overwrite=args.force_overwrite)
