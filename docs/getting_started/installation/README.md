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

All supported hardware can be found in the [vllm.ai](https://vllm.ai/#hardware) website. If you want to add new hardware, please contact us on [Slack](https://slack.vllm.ai/) or [Email](mailto:collaboration@vllm.ai).
