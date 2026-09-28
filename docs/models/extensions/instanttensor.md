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

## Benchmarks

| Model | GPU | Backend | Load Time (s) | Throughput (GB/s) | Speedup |
| --- | ---: | --- | ---: | ---: | --- |
| Qwen3-30B-A3B | 1*H200 | Safetensors | 57.4 | 1.1 | 1x |
| Qwen3-30B-A3B | 1*H200 | InstantTensor | 1.77 | 35 | <span style="color: green">**32.4x**</span> |
| DeepSeek-R1 | 8*H200 | Safetensors | 160 | 4.3 | 1x |
| DeepSeek-R1 | 8*H200 | InstantTensor | 15.3 | 45 | <span style="color: green">**10.5x**</span> |

For the full benchmark results, see <https://github.com/scitix/InstantTensor/blob/main/docs/benchmark.md>.
