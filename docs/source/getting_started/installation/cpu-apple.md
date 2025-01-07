(installation-apple)=

# Installation for macOS

vLLM has experimental support for macOS with Apple Silicon. For now, users shall build from the source vLLM to natively run on macOS. For more details, like running on vLLM in a docker container, see [ARM CPU Documentation](installation-arm)

Currently the CPU implementation for macOS supports FP32 and FP16 datatypes.

## Requirements

- **Operating System**: `macOS Sonoma` or later
- **SDK** `XCode 15.4` or later with Command Line Tools
- **Compilers**: `Apple Clang >= 15.0.0`

<!-- (arm-backend-quick-start-dockerfile)= -->

## Build and installation

After installation of XCode and the Command Line Tools, which include Apple Clang, execute the following commands to build and install vLLM from the source.

```
$ git clone https://github.com/vllm-project/vllm.git
$ cd vllm
$ pip install -r requirements-cpu.txt
$ pip install -e . 
```

```{note}
On macOS the `VLLM_TARGET_DEVICE` is automatically set to `cpu`, which currently is the only supported device.
```



## Troubleshooting

If the build has error like the following snippet where standard C++ headers cannot be found, try to remove and reinstall your 
[Command Line Tools for Xcode](https://developer.apple.com/download/all/).

```
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

