<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

vLLM has experimental CPU support on macOS with Apple Silicon.

Currently the CPU implementation for macOS supports FP32 and FP16 datatypes.

!!! tip "GPU-Accelerated Inference with vLLM-Metal"
    For GPU-accelerated inference on Apple Silicon using Metal, check out [vllm-metal](https://github.com/vllm-project/vllm-metal), a community-maintained hardware plugin that uses MLX as the compute backend.

--8<-- [end:installation]
--8<-- [start:requirements]

- OS: `macOS Sonoma` or later
- SDK: `XCode 15.4` or later with Command Line Tools
- Compiler: `Apple Clang >= 15.0.0`

!!! note
    The macOS CPU build is smoke-tested in CI on the latest GA Apple Silicon
    runner; other macOS or Apple Clang versions are best-effort.

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

Pre-built CPU wheels for Apple Silicon are available in the [GitHub release assets](https://github.com/vllm-project/vllm/releases).
For the release you want, choose the wheel with `+cpu` in the filename and a `macosx` platform tag ending in `_arm64`.

For example, to install vLLM v0.34.0 in your activated Python environment:

```bash
uv pip install \
    'https://github.com/vllm-project/vllm/releases/download/v0.34.0/vllm-0.34.0%2Bcpu-cp38-abi3-macosx_11_0_arm64.whl'
vllm --version
```

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

After installation of XCode and the Command Line Tools, which include Apple Clang, execute the following commands to build and install vLLM from source.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -r requirements/cpu.txt
uv pip install -e .
```

!!! note
    On macOS the `VLLM_TARGET_DEVICE` is automatically set to `cpu`, which is currently the only supported device.

!!! example "Troubleshooting"
    If the build fails with errors like the following where standard C++ headers cannot be found, try to remove and reinstall your
    [Command Line Tools for Xcode](https://developer.apple.com/download/all/).

    ```text
    [...] fatal error: 'map' file not found
            1 | #include <map>
                |          ^~~~~
        1 error generated.
        [2/8] Building CXX object CMakeFiles/_C.dir/csrc/cpu/pos_encoding.cpp.o

    [...] fatal error: 'cstddef' file not found
            10 | #include <cstddef>
                |          ^~~~~~~~~
        1 error generated.
    ```

    ---

    The vLLM source build requires C++20. If the compiler errors mention a C++20
    feature such as `std::bit_cast`, check the active Apple Clang toolchain:

    ```bash
    xcrun clang++ -std=c++20 -dM -E -x c++ /dev/null | grep '^#define __cplusplus '
    ```

    The command should print `#define __cplusplus 202002L`. If it fails, update
    the active Xcode Command Line Tools and check that any `CC` or `CXX`
    overrides select a compiler with C++20 support.

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

Currently, there are no pre-built Arm silicon CPU images.

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

--8<-- [end:build-image-from-source]
--8<-- [start:extra-information]
--8<-- [end:extra-information]
