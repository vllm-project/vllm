# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import multiprocessing
import os
import shutil


def find_executable(executable_name, common_paths=None):
    """Find an executable in PATH or common paths."""
    if path := shutil.which(executable_name):
        return path

    if common_paths:
        for p in common_paths:
            full_path = os.path.join(p, executable_name)
            if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                return full_path
    return None


def get_python_executable():
    """Get the current Python executable."""
    return shutil.which("python") or shutil.which("python3")


def get_cpu_cores():
    """Get the number of CPU cores."""
    return multiprocessing.cpu_count()


def generate_presets(output_path="CMakeUserPresets.json"):
    """Generates the CMakeUserPresets.json file."""

    print("Attempting to detect your system configuration...")

    # Detect NVCC
    nvcc_path = find_executable("nvcc")
    if not nvcc_path:
        nvcc_path_input = input(
            "Could not automatically find 'nvcc'. Please provide the full "
            "path to nvcc (e.g., /usr/local/cuda/bin/nvcc): ")
        nvcc_path = nvcc_path_input.strip()
    print(f"Using NVCC path: {nvcc_path}")

    # Detect Python executable
    default_python_executable = get_python_executable()
    python_executable_prompt = (
        "Enter the path to your Python executable for vLLM development "
        "(typically from your virtual environment, e.g., "
        "/home/user/venvs/vllm/bin/python).\n"
        "Press Enter to use the current detected Python: "
        f"'{default_python_executable}': ")
    python_executable = input(python_executable_prompt).strip()
    if not python_executable:
        if not default_python_executable:
            raise ValueError(
                "Could not determine Python executable. "
                "Please ensure it's in your PATH or provide it manually.")
        python_executable = default_python_executable

    print(f"Using Python executable: {python_executable}")

    # Get CPU cores
    cpu_cores = get_cpu_cores()
    nvcc_threads = min(4, cpu_cores)
    cmake_jobs = max(1, cpu_cores // nvcc_threads)
    print(f"Detected {cpu_cores} CPU cores. "
          f"Setting NVCC_THREADS={nvcc_threads} and CMake jobs={cmake_jobs}.")

    # Get vLLM project root (assuming this script is in vllm/tools/)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    print(f"VLLM project root detected as: {project_root}")

    # Ensure python_executable path is absolute or resolvable
    if not os.path.isabs(python_executable) and shutil.which(
            python_executable):
        python_executable = os.path.abspath(shutil.which(python_executable))
    elif not os.path.isabs(python_executable):
        print(f"Warning: Python executable '{python_executable}' is not an "
              "absolute path and not found in PATH. CMake might not find it.")

    presets = {
        "version":
        6,
        # Keep in sync with CMakeLists.txt and requirements/build.txt
        "cmakeMinimumRequired": {
            "major": 3,
            "minor": 26,
            "patch": 1
        },
        "configurePresets": [{
            "name": "release",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/cmake-build-release",
            "cacheVariables": {
                "CMAKE_CUDA_COMPILER": nvcc_path,
                "CMAKE_C_COMPILER_LAUNCHER": "ccache",
                "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
                "CMAKE_CUDA_COMPILER_LAUNCHER": "ccache",
                "CMAKE_BUILD_TYPE": "Release",
                "VLLM_PYTHON_EXECUTABLE": python_executable,
                "CMAKE_INSTALL_PREFIX": "${sourceDir}",
                "CMAKE_CUDA_FLAGS": "",
                "NVCC_THREADS": str(nvcc_threads),
                "CMAKE_JOB_POOLS": f"compile={cmake_jobs}",
            },
        }],
        "buildPresets": [{
            "name": "release",
            "configurePreset": "release",
            "jobs": cmake_jobs,
        }],
    }

    output_file_path = os.path.join(project_root, output_path)

    if os.path.exists(output_file_path):
        overwrite = input(
            f"'{output_file_path}' already exists. Overwrite? (y/N): ").strip(
            ).lower()
        if overwrite != 'y':
            print("Generation cancelled.")
            return

    try:
        with open(output_file_path, "w") as f:
            json.dump(presets, f, indent=4)
        print(f"Successfully generated '{output_file_path}'")
        print("\nTo use this preset:")
        print(
            f"1. Ensure you are in the vLLM root directory: cd {project_root}")
        print("2. Initialize CMake: cmake --preset release")
        print(
            "3. Build+install: cmake --build --preset release --target install"
        )

    except OSError as e:
        print(f"Error writing file: {e}")


if __name__ == "__main__":
    generate_presets()
