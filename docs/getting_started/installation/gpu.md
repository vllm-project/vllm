---
toc_depth: 3
---

# GPU

vLLM is a Python library that supports the following GPU variants. Select your GPU type to see vendor specific instructions:

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:installation"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:installation"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:installation"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:installation"

## Requirements

- OS: Linux
- Python: 3.10 -- 3.13

!!! note
    vLLM does not support Windows natively. To run vLLM on Windows, you can use the Windows Subsystem for Linux (WSL) with a compatible Linux distribution, or use some community-maintained forks, e.g. [https://github.com/SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows).

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:requirements"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:requirements"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:requirements"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:requirements"

## NVIDIA GPU Architecture Support

The official `vllm/vllm-openai` Docker images and pip wheels are compiled for a specific set of CUDA compute capabilities. If your GPU's architecture is **not** in the compiled set, you will get a runtime error (`cudaErrorNoKernelImageForDevice`) rather than a clear startup-time failure.

### Compiled architecture matrix

| CUDA version | Image / wheel variant | sm_70 Volta | sm_75 Turing | sm_80 Ampere | sm_86 Ampere | sm_89 Ada | sm_90 Hopper | sm_100 Blackwell |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 12.8 | `vllm/vllm-openai:latest` (default) | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| 12.6 | `--extra-index-url .../cu126` | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| 12.1 | `--extra-index-url .../cu121` | Yes | Yes | Yes | Yes | Yes | Yes | No |
| 11.8 | `--extra-index-url .../cu118` | Yes | Yes | Yes | Yes | No | No | No |

**GPU generations by compute capability:**

| Generation | Compute capability | Representative GPUs |
|---|---|---|
| Volta | sm_70 | V100, Titan V |
| Turing | sm_75 | T4, RTX 20xx, Quadro RTX |
| Ampere | sm_80 | A100, A30 |
| Ampere | sm_86 | A10, A40, RTX 30xx |
| Ada Lovelace | sm_89 | L4, L40, RTX 40xx |
| Hopper | sm_90 | H100, H200 |
| Blackwell | sm_100 | B200, GB200 |

!!! warning "Vendor-specific images may support fewer architectures"
    Images distributed by platform vendors (e.g., Red Hat AI/RHOAI, NGC, cloud-provider registries) may be compiled with a different CUDA version or a narrower architecture list than the upstream `vllm/vllm-openai` image. A GPU that works with the upstream image may silently fail with a vendor image at runtime. Always verify the compiled architectures for third-party images using the diagnostic below.

### Verify compiled architectures at runtime

To check which compute capabilities are compiled into the installed vLLM binary:

```console
python -c "
import torch, vllm
print('vLLM version:', vllm.__version__)
print('CUDA version:', torch.version.cuda)
props = torch.cuda.get_device_properties(0)
print(f'GPU: {props.name}  (sm_{props.major}{props.minor})')
"
```

If you receive `cudaErrorNoKernelImageForDevice` at runtime, your GPU's compute capability is not included in the current build. Options:

1. Switch to the official `vllm/vllm-openai` image which includes sm_70+.
2. [Build vLLM from source](#build-from-source) targeting your GPU:
   ```console
   TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6" pip install -e .
   ```

## Set up using Python

### Create a new Python environment

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:set-up-using-python"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:set-up-using-python"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:set-up-using-python"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:set-up-using-python"

### Pre-built wheels {#pre-built-wheels}

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:pre-built-wheels"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:pre-built-wheels"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:pre-built-wheels"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:pre-built-wheels"

### Build wheel from source

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:build-wheel-from-source"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:build-wheel-from-source"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:build-wheel-from-source"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:build-wheel-from-source"

## Set up using Docker

### Pre-built images

--8<-- [start:pre-built-images]

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:pre-built-images"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:pre-built-images"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:pre-built-images"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:pre-built-images"

--8<-- [end:pre-built-images]

### Build image from source

--8<-- [start:build-image-from-source]

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:build-image-from-source"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:build-image-from-source"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:build-image-from-source"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:build-image-from-source"

--8<-- [end:build-image-from-source]

## Supported features

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:supported-features"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:supported-features"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:supported-features"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:supported-features"
