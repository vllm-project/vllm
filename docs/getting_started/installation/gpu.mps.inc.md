# --8<-- [start:installation]

vLLM has experimental support for Apple Silicon Macs using Metal Performance Shaders (MPS). MPS provides GPU acceleration on Apple Silicon, offering significantly better performance than CPU-only execution.

# --8<-- [end:installation]
# --8<-- [start:requirements]

- OS: `macOS Sonoma` or later
- Hardware: Apple Silicon (M1/M2/M3/M4 series)
- SDK: `XCode 15.4` or later with Command Line Tools
- Compiler: `Apple Clang >= 15.0.0`
- Supported datatypes: FP16, FP32, BF16 (on M2+ with BF16 support)

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

!!! note
    MPS is automatically selected as the default backend on Apple Silicon Macs when available. No additional configuration is required.

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

Currently, there are no pre-built MPS wheels. You must build from source.

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

After installation of XCode and the Command Line Tools, which include Apple Clang, execute the following commands to build and install vLLM from source.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install -e .
```

!!! tip
    The `--index-strategy unsafe-best-match` flag is needed to resolve dependencies across multiple package indexes (PyTorch CPU index and PyPI). Without this flag, you may encounter `typing-extensions` version conflicts.

    The term "unsafe" refers to the package resolution strategy, not security. By default, `uv` only searches the first index where a package is found to prevent dependency confusion attacks. This flag allows `uv` to search all configured indexes to find the best compatible versions. Since both PyTorch and PyPI are trusted package sources, using this strategy is safe and appropriate for vLLM installation.

!!! note
    vLLM automatically detects MPS availability at runtime. If MPS is available on your system, it will be used by default. To force CPU-only execution instead, you can build with `VLLM_TARGET_DEVICE=cpu`.

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

    If the build fails with C++11/C++17 compatibility errors like the following, the issue is that the build system is defaulting to an older C++ standard:

    ```text
    [...] error: 'constexpr' is not a type
    [...] error: expected ';' before 'constexpr'
    [...] error: 'constexpr' does not name a type
    ```

    **Solution**: Your compiler might be using an older C++ standard. Edit `cmake/cpu_extension.cmake` and add `set(CMAKE_CXX_STANDARD 17)` before `set(CMAKE_CXX_STANDARD_REQUIRED ON)`.

    To check your compiler's C++ standard support:
    ```bash
    clang++ -std=c++17 -pedantic -dM -E -x c++ /dev/null | grep __cplusplus
    ```
    On Apple Clang 16 you should see: `#define __cplusplus 201703L`

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

Docker is not commonly used on macOS for MPS workloads as MPS requires native macOS execution.

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

# --8<-- [end:build-image-from-source]
# --8<-- [start:supported-features]

MPS backend has the following limitations compared to CUDA:

- Single device only (no tensor parallelism)
- No CUDA graph support
- No custom all-reduce operations
- No KV cache quantization
- Limited torch.compile support (eager mode used)

Supported features:

- FP16, FP32, and BF16 (on supported hardware) inference
- Structured output
- LoRA adapters

# --8<-- [end:supported-features]
# --8<-- [start:extra-information]

## MPS-specific Configuration

- `VLLM_CPU_KVCACHE_SPACE`: Specify the KV Cache size in GiB (e.g., `VLLM_CPU_KVCACHE_SPACE=8` means 8 GiB for KV cache). If not set, vLLM uses 50% of system memory by default. Apple Silicon uses unified memory, so this space is shared between CPU and GPU.

## Choosing Between MPS and CPU

On Apple Silicon Macs, vLLM automatically selects MPS when available. MPS provides GPU acceleration and is generally faster than CPU execution. However, if you encounter compatibility issues or need CPU-only execution for any reason, you can force CPU mode by building with `VLLM_TARGET_DEVICE=cpu`.

# --8<-- [end:extra-information]
