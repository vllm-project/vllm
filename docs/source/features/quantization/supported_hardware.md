(quantization-supported-hardware)=

# Supported Hardware

The table below shows the compatibility of various quantization implementations with different hardware platforms in vLLM:

```{list-table}
:header-rows: 1
:widths: 20 8 8 8 8 8 8 8 8 8 8

* - Implementation
  - Volta
  - Turing
  - Ampere
  - Ada
  - Hopper
  - AMD GPU
  - Intel GPU
  - x86 CPU
  - AWS Inferentia
  - Google TPU
* - AWQ
  - ✗
  - ✅︎
  - ✅︎
  - ✅︎
  - ✅︎
  - ✗
  - ✅︎
  - ✅︎
  - ✗
  - ✗
* - GPTQ
  - ✅︎
  - ✅︎
  - ✅︎
  - ✅︎
  - ✅︎
  - ✗
  - ✅︎
  - ✅︎
  - ✗
  - ✗
* - Marlin (GPTQ/AWQ/FP8)
  - ✗
  - ✗
  - ✅︎
  - ✅︎
  - ✅︎
  - ✗
  - ✗
  - ✗
  - ✗
  - ✗
* - INT8 (W8A8)
  - ✗
  - ✅︎
  - ✅︎
  - ✅︎
  - ✅︎
  - ✗
  - ✗
  - ✅︎
  - ✗
  - ✗
* - FP8 (W8A8)
  - ✗
  - ✗
  - ✗
  - ✅︎
  - ✅︎
  - ✅︎
  - ✗
  - ✗
  - ✗
  - ✗
* - AQLM
  - ✅︎
  - ✅︎
  - ✅︎
  - ✅︎
  - ✅︎
  - ✗
  - ✗
  - ✗
  - ✗
  - ✗
* - bitsandbytes
  - ✅︎
  - ✅︎
  - ✅︎
  - ✅︎
  - ✅︎
  - ✗
  - ✗
  - ✗
  - ✗
  - ✗
* - DeepSpeedFP
  - ✅︎
  - ✅︎
  - ✅︎
  - ✅︎
  - ✅︎
  - ✗
  - ✗
  - ✗
  - ✗
  - ✗
* - GGUF
  - ✅︎
  - ✅︎
  - ✅︎
  - ✅︎
  - ✅︎
  - ✅︎
  - ✗
  - ✗
  - ✗
  - ✗
```

- Volta refers to SM 7.0, Turing to SM 7.5, Ampere to SM 8.0/8.6, Ada to SM 8.9, and Hopper to SM 9.0.
- "✅︎" indicates that the quantization method is supported on the specified hardware.
- "✗" indicates that the quantization method is not supported on the specified hardware.

```{note}
This compatibility chart is subject to change as vLLM continues to evolve and expand its support for different hardware platforms and quantization methods.

For the most up-to-date information on hardware support and quantization methods, please refer to <gh-dir:vllm/model_executor/layers/quantization> or consult with the vLLM development team.
```
