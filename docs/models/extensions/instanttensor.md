# Loading Model Weights with InstantTensor

InstantTensor accelerates loading Safetensors weights on CUDA devices through distributed loading, pipelined prefetching, and direct I/O. InstantTensor also supports GDS (GPUDirect Storage) when available.
For more details, see the [InstantTensor GitHub repository](https://github.com/scitix/InstantTensor).

## Installation

```bash
pip install instanttensor
```

## Use InstantTensor in vLLM

Add `--load-format instanttensor` as a command-line argument.

For example:

```bash
vllm serve Qwen/Qwen2.5-0.5B --load-format instanttensor
```
