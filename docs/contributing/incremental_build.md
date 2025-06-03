# Incremental Compilation Workflow for vLLM Development

When working on vLLM's C++/CUDA kernels located in the `csrc/` directory, recompiling the entire project with `pip install -e .` for every change can be time-consuming. An incremental compilation workflow using CMake allows for faster iteration by only recompiling the necessary components after an initial setup. This guide details how to set up and use such a workflow, which complements your editable Python installation.

## Prerequisites

Before setting up the incremental build:

1. **vLLM Editable Install:** Ensure you have vLLM installed from source in an editable mode. Using pre-compiled wheels for the initial editable setup can be faster, as the CMake workflow will handle subsequent kernel recompilations.

   ```bash
   VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
   ```

2. **CUDA Toolkit:** Verify that the NVIDIA CUDA Toolkit is correctly installed and `nvcc` is accessible in your `PATH`. CMake relies on `nvcc` to compile CUDA code. If you encounter issues, refer to the [official CUDA Toolkit installation guides](https://developer.nvidia.com/cuda-toolkit-archive) and vLLM's main [GPU installation documentation](../getting_started/installation/gpu/cuda.inc.md#troubleshooting) for troubleshooting. The `CMAKE_CUDA_COMPILER` variable in your `CMakeUserPresets.json` should also point to your `nvcc` binary.

3. **Build Tools:** Ensure the dependencies for building are installed and available, like `cmake` and `ninja`. These are installable through the `requirements/build.txt`, or can be installed by your package manager.

    ```bash
    uv pip install -r requirements/build.txt --torch-backend=auto
    ```

## Setting up the CMake Build Environment

The incremental build process is managed through CMake. You can configure your build settings using a `CMakeUserPresets.json` file at the root of the vLLM repository.

### Create `CMakeUserPresets.json`

Here is an example configuration. Create a file named `CMakeUserPresets.json` in the root directory of your vLLM clone and paste the following content. Adjust paths (like `VLLM_PYTHON_EXECUTABLE` and `CMAKE_CUDA_COMPILER`) and settings as per your system configuration.

```json
{
    "version": 6,
    "cmakeMinimumRequired": {
        "major": 3,
        "minor": 20,
        "patch": 0
    },
    "configurePresets": [
        {
            "name": "debug",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/cmake-build-debug",
            "cacheVariables": {
                "CMAKE_CUDA_COMPILER": "/usr/local/cuda/bin/nvcc", // Adjust if your nvcc is elsewhere
                "CMAKE_C_COMPILER_LAUNCHER": "ccache",
                "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
                "CMAKE_CUDA_COMPILER_LAUNCHER": "ccache",
                "CMAKE_BUILD_TYPE": "Debug",
                "VLLM_PYTHON_EXECUTABLE": "/home/user/venvs/vllm/bin/python", // Adjust to your Python executable from the editable install's virtual environment
                "CMAKE_INSTALL_PREFIX": "${sourceDir}",
                "CMAKE_CUDA_FLAGS": "",
                "NVCC_THREADS": "8", // Adjust based on your CPU cores
                "CMAKE_JOB_POOL_COMPILE": "compile",
                "CMAKE_JOB_POOLS": "compile=64" // Adjust based on your CPU cores
            }
        },
        {
            "name": "release",
            "inherits": "debug",
            "binaryDir": "${sourceDir}/cmake-build-release",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "debug",
            "configurePreset": "debug",
            "jobs": 64 // Corresponds to CMAKE_JOB_POOLS
        },
        {
            "name": "release",
            "configurePreset": "release",
            "jobs": 64 // Corresponds to CMAKE_JOB_POOLS
        }
    ]
}
```

**Key configurations in `CMakeUserPresets.json`:**
- `CMAKE_C_COMPILER_LAUNCHER`, `CMAKE_CXX_COMPILER_LAUNCHER`, `CMAKE_CUDA_COMPILER_LAUNCHER`: Setting these to `ccache` (or `sccache`) significantly speeds up rebuilds by caching compilation results. Ensure `ccache` is installed (e.g., `sudo apt install ccache` or `conda install ccache`).
- `VLLM_PYTHON_EXECUTABLE`: Path to the Python executable in your vLLM development environment.
- `CMAKE_JOB_POOLS` and `jobs` in build presets: Control the parallelism of the build. Adjust `compile=64` and `jobs: 64` based on your system's resources to optimize build times without overloading your machine.
- `binaryDir`: Specifies where the build artifacts will be stored (e.g., `cmake-build-release`).

## Building and Installing with CMake

Once your `CMakeUserPresets.json` is configured:

1. **Prime the CMake build environment:**
   This step configures the build system according to your chosen preset (e.g., `release`).

   ```bash
   cmake --preset release
   ```

2. **Build and install the vLLM components:**
   This command compiles the code and installs the resulting binaries into your vLLM source directory, making them available to your editable Python installation.

   ```bash
   cmake --build --preset release --target install
   ```

3. **Make changes and repeat!**
    Now you start using your editable install of vLLM, testing and making changes as needed. If you need to build again to update based on changes, simply run the CMake command again to build only the affected files.

    ```bash
    cmake --build --preset release --target install
    ```

## Verifying the Build

After a successful build, you will find a populated build directory (e.g., `cmake-build-release/` if you used the `release` preset and the example configuration).

```bash
> ls cmake-build-release/
bin             cmake_install.cmake      _deps                                machete_generation.log
build.ninja     CPackConfig.cmake        detect_cuda_compute_capabilities.cu  marlin_generation.log
_C.abi3.so      CPackSourceConfig.cmake  detect_cuda_version.cc               _moe_C.abi3.so
CMakeCache.txt  ctest                    _flashmla_C.abi3.so                  moe_marlin_generation.log
CMakeFiles      cumem_allocator.abi3.so  install_local_manifest.txt           vllm-flash-attn
```

The `cmake --build ... --target install` command copies the compiled shared libraries (like `_C.abi3.so`, `_moe_C.abi3.so`, etc.) into the appropriate `vllm` package directory within your source tree. This updates your editable installation with the newly compiled kernels.

## Tips for an Efficient Workflow

- **Leverage `ccache`:** As mentioned, using `ccache` (or `sccache`) via the `CMAKE_..._COMPILER_LAUNCHER` variables in your preset is crucial for fast incremental builds. Ensure it's installed and configured correctly. This is also a recommended practice for standard `pip install -e .` builds, as noted in the general installation guide.
- **Adjust Parallelism:** Fine-tune the `CMAKE_JOB_POOLS` in `configurePresets` and `jobs` in `buildPresets` in your `CMakeUserPresets.json`. Too many jobs can overload systems with limited RAM or CPU cores, leading to slower builds or system instability. Too few won't fully utilize available resources.
- **Clean Builds When Necessary:** If you encounter persistent or strange build errors, especially after significant changes or switching branches, consider removing the CMake build directory (e.g., `rm -rf cmake-build-release`) and re-running the `cmake --preset` and `cmake --build` commands.
- **Specific Target Builds:** For even faster iterations when working on a specific module, you can sometimes build a specific target instead of the full `install` target, though `install` ensures all necessary components are updated in your Python environment. Refer to CMake documentation for more advanced target management.

This incremental workflow should significantly reduce compilation times when developing vLLM's core C++/CUDA components.
