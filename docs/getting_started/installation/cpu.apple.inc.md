# --8<-- [start:installation]

vLLM has experimental support for macOS with Apple Silicon. For now, users must build from source to natively run on macOS.

Currently the CPU implementation for macOS supports FP32 and FP16 datatypes.

# --8<-- [end:installation]
# --8<-- [start:requirements]

- OS: `macOS Sonoma` or later
- SDK: `XCode 15.4` or later with Command Line Tools
- Compiler: `Apple Clang >= 15.0.0`

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

Currently, there are no pre-built Apple silicon CPU wheels.

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

Currently, there are no pre-built Arm silicon CPU images.

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]
# --8<-- [end:extra-information]
