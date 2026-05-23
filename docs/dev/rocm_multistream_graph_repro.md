# ROCm Multi-Stream Graph Replay Reproducer

`tools/rocm_multistream_graph_repro.py` isolates the ROCm graph replay hang
found while enabling DeepSeek-V4 CSA decode multi-stream on AMD GPUs.

## Reproduce the hang

```bash
HIP_VISIBLE_DEVICES=0 timeout 90s \
  .venv/bin/python tools/rocm_multistream_graph_repro.py --mode allocating
```

Observed on MI355X: warmup and graph capture complete, then the first
`CUDAGraph.replay()` does not return. `rocm-smi` shows the GPU at 100% busy with
0% memory bandwidth.

## Verify the workaround

```bash
HIP_VISIBLE_DEVICES=0 timeout 90s \
  .venv/bin/python tools/rocm_multistream_graph_repro.py --mode preallocated
```

The preallocated mode creates side-stream GEMM output buffers before capture and
uses `torch.mm(..., out=...)`. This replays successfully on the same system.

`torch.cuda.Event(external=True)` is not available as a ROCm workaround; it
raises `RuntimeError: External events are disallowed in rocm`.

## DeepSeek-V4 implication

The vLLM ROCm path must not capture side-stream work that allocates new tensors
inside the graph. The current branch uses explicit stream/event ordering and
preallocated `out=` buffers for the overlapped CSA GEMM. SGLang's DeepSeek-V4
ROCm implementation goes further: it gates multi-stream to graph-capture decode,
uses pre-created streams, fuses KV cache writes into the K path, and runs the
indexer/compressor through fused kernels with prebuilt metadata. That design
avoids the side-stream allocation pattern reproduced here.

## Benchmark notes

Benchmark: DeepSeek-V4-Pro, TP=8, `random` 1k input / 1k output,
`--max-concurrency 4`, 40 prompts, 8 warmups.

| Path | Output tok/s | Total tok/s | Mean TPOT |
|---|---:|---:|---:|
| ROCm graph-safe workaround, aux disabled | 60.14 | 120.86 | 63.90 ms |
| One preallocated aux CSA GEMM | 57.89 | 116.34 | 66.51 ms |
| One aux CSA GEMM, threshold 16 | 57.42 | 115.39 | 66.88 ms |

The narrowed vLLM overlap path avoids the hang, but it does not improve this
low-workload benchmark. The likely missing piece versus SGLang is a deeper
fused implementation that removes intermediate tensors and cache-write
allocation from the side-stream path.
