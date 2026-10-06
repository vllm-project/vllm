# DeepEP v2 expanded-prefill follow-up

The first experiment separates layout from host count synchronization. It uses
fresh dispatch handles and leaves the production always-off policy unchanged.

| Case | Expanded expert groups | CPU count wait | Receive allocation |
| --- | --- | --- | --- |
| `nonexpanded-nosync` | No | No | Deduplicated-token capacity |
| `nonexpanded-sync` | No | Yes | Actual deduplicated tokens |
| `expanded-nosync` | Yes | No | Expanded capacity plus alignment padding |
| `expanded-sync` | Yes | Yes | Expert rows including alignment padding |

CPU count polling must stay outside CUDA graph capture. Cached dispatch does
not permit expansion or CPU sync in the pinned DeepEP implementation.

`benchmark_deepep_v2_layout.py` reconstructs global expert IDs and real expert
counts from GPU prefixes for the expanded cases. Alignment gaps, empty experts
and unused tail capacity get ID `-1`. CPU expert counts are alignment-padded;
compare them with rounded GPU counts, not the real counts directly.

The harness checks dispatch/combine against an expert-dependent scalar reference
before timing. It masks uninitialized receive rows and weights before arithmetic.
All ranks verify the same CLI configuration before dispatch. Each invocation
runs one case in a fresh process, with no graph capture or cached dispatch.

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmarks/kernels/benchmark_deepep_v2_layout.py \
  --case nonexpanded-sync --tokens-per-rank 8192,8192 \
  --output /tmp/nonexpanded-sync.json
```

For same-node NVLink-only runs, set `EP_DISABLE_GIN=1`, `NCCL_GIN_ENABLE=0`
and `NCCL_IB_DISABLE=1`. The harness requires the entire process group to be
one physical NVLink domain when GIN is disabled. No RDMA resource is requested.

Repeat for all four cases, with matching dimensions, first at `512,512`, then
`8192,8192`, `8192,512`, and `8192,0`. Run `16384,16384` only after those pass.
Use the same allocated GPUs for the matrix; repeat in reverse case order before
interpreting small differences. A worker that hangs or fails is a correctness
failure, not a timing result. Record GPU topology, immutable runtime identity,
source hash and all rank reports.

The measurement includes allocation, dispatch, GPU metadata, a synthetic expert
operation, combine, and per-step CUDA completion synchronization. The reported
latency is the median of per-step maximum rank wall times. Memory reports are
PyTorch allocator peaks, not total device footprints. Warmup/compilation is
excluded. Per-step synchronization changes execution cadence, so these numbers
are neither serving latency nor steady-state asynchronous throughput.

There is no expert GEMM in this first harness. It cannot establish that expansion
saves expert preparation or improves model performance. GPU-prefix reconstruction
uses Torch operations as a correctness prototype; tune/fuse it only after GPU
validation. No production threshold or automatic rank-local policy is introduced.

Next integrate an expert kernel consuming the grouped rows directly. Match
DeepEP expert alignment and activation-scale layout to the grouped GEMM contract,
and bypass the existing input permutation. Then measure preparation, GEMMs,
combine and the complete forward with all four allocation/layout cases. Follow
with accuracy and non-overloaded serving evaluation. Coordinate integration with
existing PR #51589, which already reconstructs sync-less expanded metadata.

Reference implementation:
<https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/elastic/buffer.hpp>
