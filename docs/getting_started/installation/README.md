# Installation

vLLM supports the following hardware platforms:

- [GPU](gpu.md)
    - [NVIDIA CUDA](gpu.md)
    - [AMD ROCm](gpu.md)
    - [Intel XPU](gpu.md)
    - [Apple Silicon](gpu.md) (via [vLLM-Metal](https://github.com/vllm-project/vllm-metal))
- [CPU](cpu.md)
    - [Intel/AMD x86](cpu.md#intelamd-x86)
    - [ARM AArch64](cpu.md#arm-aarch64)
    - [Apple silicon](cpu.md#apple-silicon)
    - [IBM Z (S390X)](cpu.md#ibm-z-s390x)

## Hardware Plugins

vLLM supports third-party hardware plugins that live **outside** the main `vllm` repository. These follow the [Hardware-Pluggable RFC](../../design/plugin_system.md).

A list of all supported hardware can be found on the vLLM website, see [Universal Compatibility - Hardware](https://vllm.ai/#compatibility).

If you want to add new hardware, please contact us on [Slack](https://slack.vllm.ai/) or [Email](mailto:collaboration@vllm.ai).
