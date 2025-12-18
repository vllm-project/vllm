# Installation

vLLM supports the following hardware platforms:

- [GPU](gpu.md)
    - [NVIDIA CUDA](gpu.md#nvidia-cuda)
    - [AMD ROCm](gpu.md#amd-rocm)
    - [Intel XPU](gpu.md#intel-xpu)
- [CPU](cpu.md)
    - [Intel/AMD x86](cpu.md#intelamd-x86)
    - [ARM AArch64](cpu.md#arm-aarch64)
    - [Apple silicon](cpu.md#apple-silicon)
    - [IBM Z (S390X)](cpu.md#ibm-z-s390x)

## Hardware Plugins

The backends below live **outside** the main `vllm` repository and follow the
[Hardware-Pluggable RFC](../../design/plugin_system.md).

| Accelerator | PyPI / package | Repository |
|-------------|----------------|------------|
| Google TPU | `tpu-inference` | <https://github.com/vllm-project/tpu-inference> |
| Ascend NPU | `vllm-ascend` | <https://github.com/vllm-project/vllm-ascend> |
| Intel Gaudi (HPU) | N/A, install from source | <https://github.com/vllm-project/vllm-gaudi> |
| MetaX MACA GPU | N/A, install from source | <https://github.com/MetaX-MACA/vLLM-metax> |
| Rebellions ATOM / REBEL NPU | `vllm-rbln` | <https://github.com/rebellions-sw/vllm-rbln> |
| IBM Spyre AIU | `vllm-spyre` | <https://github.com/vllm-project/vllm-spyre> |
| Cambricon MLU | `vllm-mlu` | <https://github.com/Cambricon/vllm-mlu> |
| Baidu Kunlun XPU | N/A, install from source | <https://github.com/baidu/vLLM-Kunlun> |
| Sophgo TPU | N/A, install from source | <https://github.com/sophgo/vllm-tpu> |
