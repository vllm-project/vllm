# Quantization

Quantization trades off model precision for smaller memory footprint, allowing large models to be run on a wider range of devices.

Contents:

- [AutoAWQ](auto_awq.md)
- [AutoRound](auto_round.md)
- [BitsAndBytes](bnb.md)
- [BitBLAS](bitblas.md)
- [GGUF](gguf.md)
- [GPTQModel](gptqmodel.md)
- [INC](inc.md)
- [INT4 W4A16](int4.md)
- [INT8 W8A8](int8.md)
- [FP8 W8A8](fp8.md)
- [NVIDIA TensorRT Model Optimizer](modelopt.md)
- [AMD Quark](quark.md)
- [Quantized KV Cache](quantized_kvcache.md)
- [TorchAO](torchao.md)

## Supported Hardware

The table below shows the compatibility of various quantization implementations with different hardware platforms in vLLM:

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| Implementation        | Volta   | Turing   | Ampere   | Ada   | Hopper   | AMD GPU   | Intel GPU   | Intel Gaudi | x86 CPU   |
|-----------------------|---------|----------|----------|-------|----------|-----------|-------------|-------------|-----------|
| AWQ                   | ❌      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ✅︎          | ❌         | ✅︎        |
| GPTQ                  | ✅︎      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ✅︎          | ❌         | ✅︎        |
| Marlin (GPTQ/AWQ/FP8) | ❌      | ❌       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌         | ❌        |
| INT8 (W8A8)           | ❌      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌         | ✅︎        |
| FP8 (W8A8)            | ❌      | ❌       | ❌       | ✅︎    | ✅︎       | ✅︎         | ❌          | ❌         | ❌        |
| BitBLAS               | ✅︎      | ✅       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌         | ❌        |
| BitBLAS (GPTQ)        | ❌      | ❌       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌         | ❌        |
| bitsandbytes          | ✅︎      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌         | ❌        |
| DeepSpeedFP           | ✅︎      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌         | ❌        |
| GGUF                  | ✅︎      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ✅︎         | ❌          | ❌         | ❌        |
| INC (W8A8)            | ❌      | ❌       | ❌       | ❌    | ❌       | ❌         | ❌          | ✅︎         | ❌        |

- Volta refers to SM 7.0, Turing to SM 7.5, Ampere to SM 8.0/8.6, Ada to SM 8.9, and Hopper to SM 9.0.
- ✅︎ indicates that the quantization method is supported on the specified hardware.
- ❌ indicates that the quantization method is not supported on the specified hardware.

!!! note
    For information on quantization support on Google TPU, please refer to the [TPU-Inference Recommended Models and Features](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/) documentation.

!!! note
    This compatibility chart is subject to change as vLLM continues to evolve and expand its support for different hardware platforms and quantization methods.

    For the most up-to-date information on hardware support and quantization methods, please refer to [vllm/model_executor/layers/quantization](../../../vllm/model_executor/layers/quantization) or consult with the vLLM development team.
