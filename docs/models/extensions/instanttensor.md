# Loading Model Weights with InstantTensor

InstantTensor accelerates loading Safetensors weights on CUDA devices through distributed loading, pipelined prefetching, and direct I/O. InstantTensor also supports GDS (GPUDirect Storage) when available.
For more details, see the [InstantTensor GitHub repository](https://github.com/scitix/InstantTensor).

## Installation

```bash
pip install 'instanttensor>=0.2.2'
```

## Use InstantTensor in vLLM

Add `--load-format instanttensor` as a command-line argument.

For example:

```bash
vllm serve Qwen/Qwen2.5-0.5B --load-format instanttensor
```

## Selective loading

For models that provide `is_unused_checkpoint_weight`, including supported MTP
draft models, vLLM passes the required checkpoint tensor names to InstantTensor.
Unused tensors are skipped before payload I/O, including those in shards that also
contain required weights. Names are matched exactly, before vLLM applies any source
prefix or model-specific name mapping. Progress and buffer sizing reflect only the
selected tensors; reads can still include neighboring bytes for alignment.

InstantTensor 0.2.2 requires filtered loads to read independently on each GPU,
without an NCCL process group. A filtered load therefore trades coordinated
multi-GPU I/O for reading fewer tensors; the performance benefit depends on the
checkpoint layout and storage. Without this hook, dense and fused-expert
checkpoints retain coordinated distributed loading; expert-parallel filtering of
individually indexed expert weights uses independent reads. When both filters
apply, only tensors selected by both are read. This does not automatically select
pipeline-stage weights or tensor-parallel slices.

## Benchmarks

| Model | GPU | Backend | Load Time (s) | Throughput (GB/s) | Speedup |
| --- | ---: | --- | ---: | ---: | --- |
| Qwen3-30B-A3B | 1*H200 | Safetensors | 57.4 | 1.1 | 1x |
| Qwen3-30B-A3B | 1*H200 | InstantTensor | 1.77 | 35 | <span style="color: green">**32.4x**</span> |
| DeepSeek-R1 | 8*H200 | Safetensors | 160 | 4.3 | 1x |
| DeepSeek-R1 | 8*H200 | InstantTensor | 15.3 | 45 | <span style="color: green">**10.5x**</span> |

For the full benchmark results, see <https://github.com/scitix/InstantTensor/blob/main/docs/benchmark.md>.
