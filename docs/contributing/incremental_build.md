# Incremental Compilation Workflow

When working on vLLM's C++/CUDA kernels located in the `csrc/` directory, recompiling the entire project with `uv pip install -e .` for every change can be time-consuming. An incremental compilation workflow using CMake allows for faster iteration by only recompiling the necessary components after an initial setup. This guide details how to set up and use such a workflow, which complements your editable Python installation.

## Prerequisites

Before setting up the incremental build:

1. **vLLM Editable Install:** Ensure you have vLLM installed from source in an editable mode. Using pre-compiled wheels for the initial editable setup can be faster, as the CMake workflow will handle subsequent kernel recompilations.

    ```console
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
    ```

2. **CUDA Toolkit:** Verify that the NVIDIA CUDA Toolkit is correctly installed and `nvcc` is accessible in your `PATH`. CMake relies on `nvcc` to compile CUDA code. You can typically find `nvcc` in `$CUDA_HOME/bin/nvcc` or by running `which nvcc`. If you encounter issues, refer to the [official CUDA Toolkit installation guides](https://developer.nvidia.com/cuda-toolkit-archive) and vLLM's main [GPU installation documentation](../getting_started/installation/gpu.md#troubleshooting) for troubleshooting. The `CMAKE_CUDA_COMPILER` variable in your `CMakeUserPresets.json` should also point to your `nvcc` binary.

3. **Build Tools:** It is highly recommended to install `ccache` for fast rebuilds by caching compilation results (e.g., `sudo apt install ccache` or `conda install ccache`). Also, ensure the core build dependencies like `cmake` and `ninja` are installed. These are installable through `requirements/build.txt` or your system's package manager.

    ```console
    uv pip install -r requirements/build.txt --torch-backend=auto
    ```

## Setting up the CMake Build Environment

The incremental build process is managed through CMake. You can configure your build settings using a `CMakeUserPresets.json` file at the root of the vLLM repository.

### Generate `CMakeUserPresets.json` using the helper script

To simplify the setup, vLLM provides a helper script that attempts to auto-detect your system's configuration (like CUDA path, Python environment, and CPU cores) and generates the `CMakeUserPresets.json` file for you.

**Run the script:**

Navigate to the root of your vLLM clone and execute the following command:

```console
python tools/generate_cmake_presets.py
```

The script will prompt you if it cannot automatically determine certain paths (e.g., `nvcc` or a specific Python executable for your vLLM development environment). Follow the on-screen prompts. If an existing `CMakeUserPresets.json` is found, the script will ask for confirmation before overwriting it.

After running the script, a `CMakeUserPresets.json` file will be created in the root of your vLLM repository.

### Example `CMakeUserPresets.json`

Below is an example of what the generated `CMakeUserPresets.json` might look like. The script will tailor these values based on your system and any input you provide.

```json
{
    "version": 6,
    "cmakeMinimumRequired": {
        "major": 3,
        "minor": 26,
        "patch": 1
    },
    "configurePresets": [
        {
            "name": "release",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/cmake-build-release",
            "cacheVariables": {
                "CMAKE_CUDA_COMPILER": "/usr/local/cuda/bin/nvcc",
                "CMAKE_C_COMPILER_LAUNCHER": "ccache",
                "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
                "CMAKE_CUDA_COMPILER_LAUNCHER": "ccache",
                "CMAKE_BUILD_TYPE": "Release",
                "VLLM_PYTHON_EXECUTABLE": "/home/user/venvs/vllm/bin/python",
                "CMAKE_INSTALL_PREFIX": "${sourceDir}",
                "CMAKE_CUDA_FLAGS": "",
                "NVCC_THREADS": "4",
                "CMAKE_JOB_POOLS": "compile=32"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "release",
            "configurePreset": "release",
            "jobs": 32
        }
    ]
}
```

**What do the various configurations mean?**

- `CMAKE_CUDA_COMPILER`: Path to your `nvcc` binary. The script attempts to find this automatically.
- `CMAKE_C_COMPILER_LAUNCHER`, `CMAKE_CXX_COMPILER_LAUNCHER`, `CMAKE_CUDA_COMPILER_LAUNCHER`: Setting these to `ccache` (or `sccache`) significantly speeds up rebuilds by caching compilation results. Ensure `ccache` is installed (e.g., `sudo apt install ccache` or `conda install ccache`). The script sets these by default.
- `VLLM_PYTHON_EXECUTABLE`: Path to the Python executable in your vLLM development environment. The script will prompt for this, defaulting to the current Python environment if suitable.
- `CMAKE_INSTALL_PREFIX: "${sourceDir}"`: Specifies that the compiled components should be installed back into your vLLM source directory. This is crucial for the editable install, as it makes the newly built kernels immediately available to your Python environment.
- `CMAKE_JOB_POOLS` and `jobs` in build presets: Control the parallelism of the build. The script sets these based on the number of CPU cores detected on your system.
- `binaryDir`: Specifies where the build artifacts will be stored (e.g., `cmake-build-release`).

## Building and Installing with CMake

Once your `CMakeUserPresets.json` is configured:

1. **Initialize the CMake build environment:**
   This step configures the build system according to your chosen preset (e.g., `release`) and creates the build directory at `binaryDir`

    ```console
    cmake --preset release
    ```

2. **Build and install the vLLM components:**
   This command compiles the code and installs the resulting binaries into your vLLM source directory, making them available to your editable Python installation.

    ```console
    cmake --build --preset release --target install
    ```

3. **Make changes and repeat!**
    Now you start using your editable install of vLLM, testing and making changes as needed. If you need to build again to update based on changes, simply run the CMake command again to build only the affected files.

    ```console
    cmake --build --preset release --target install
    ```

## Verifying the Build

After a successful build, you will find a populated build directory (e.g., `cmake-build-release/` if you used the `release` preset and the example configuration).

```console
> ls cmake-build-release/
bin             cmake_install.cmake      _deps                                machete_generation.log
build.ninja     CPackConfig.cmake        detect_cuda_compute_capabilities.cu  marlin_generation.log
_C.abi3.so      CPackSourceConfig.cmake  detect_cuda_version.cc               _moe_C.abi3.so
CMakeCache.txt  ctest                    _flashmla_C.abi3.so                  moe_marlin_generation.log
CMakeFiles      cumem_allocator.abi3.so  install_local_manifest.txt           vllm-flash-attn
```

The `cmake --build ... --target install` command copies the compiled shared libraries (like `_C.abi3.so`, `_moe_C.abi3.so`, etc.) into the appropriate `vllm` package directory within your source tree. This updates your editable installation with the newly compiled kernels.

## Additional Tips

- **Adjust Parallelism:** Fine-tune the `CMAKE_JOB_POOLS` in `configurePresets` and `jobs` in `buildPresets` in your `CMakeUserPresets.json`. Too many jobs can overload systems with limited RAM or CPU cores, leading to slower builds or system instability. Too few won't fully utilize available resources.
- **Clean Builds When Necessary:** If you encounter persistent or strange build errors, especially after significant changes or switching branches, consider removing the CMake build directory (e.g., `rm -rf cmake-build-release`) and re-running the `cmake --preset` and `cmake --build` commands.
- **Specific Target Builds:** For even faster iterations when working on a specific module, you can sometimes build a specific target instead of the full `install` target, though `install` ensures all necessary components are updated in your Python environment. Refer to CMake documentation for more advanced target management.
