# GPU

vLLM is a Python library that supports the following GPU variants. Select your GPU type to see vendor specific instructions:

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:installation"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:installation"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:installation"

## Requirements

- OS: Linux
- Python: 3.9 -- 3.12

!!! note
    vLLM does not support Windows natively. To run vLLM on Windows, you can use the Windows Subsystem for Linux (WSL) with a compatible Linux distribution, or use some community-maintained forks, e.g. [https://github.com/SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows).

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:requirements"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:requirements"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:requirements"

## Set up using Python

### Create a new Python environment

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:create-a-new-python-environment"

=== "AMD ROCm"

    There is no extra information on creating a new Python environment for this device.

=== "Intel XPU"

    There is no extra information on creating a new Python environment for this device.

### Pre-built wheels

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:pre-built-wheels"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:pre-built-wheels"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:pre-built-wheels"

[](){ #build-from-source }

### Build wheel from source

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:build-wheel-from-source"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:build-wheel-from-source"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:build-wheel-from-source"

## Set up using Docker

### Pre-built images

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:pre-built-images"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:pre-built-images"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:pre-built-images"

### Build image from source

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:build-image-from-source"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:build-image-from-source"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:build-image-from-source"

## Supported features

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:supported-features"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:supported-features"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:supported-features"
