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
runs one case in a fresh process. Dispatch always computes fresh routing rather
than using a cached handle.

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmarks/kernels/benchmark_deepep_v2_layout.py \
  --case nonexpanded-sync --tokens-per-rank 8192,8192 \
  --output /tmp/nonexpanded-sync.json
```

For same-node NVLink-only runs, set `EP_DISABLE_GIN=1`, `NCCL_GIN_ENABLE=0`
and `NCCL_IB_DISABLE=1`. The harness requires the entire process group to be
one physical NVLink domain when GIN is disabled. No RDMA resource is requested.

Add `--cuda-graph` to either `nonexpanded-nosync` or `expanded-nosync` to capture
the entire dispatch/expert/combine step. CPU-sync cases are rejected before GPU
initialization. Warmup/JIT compilation happens before capture. The captured
graph is replayed with changed activations and two different routing shifts at
fixed input addresses, then with restored inputs; each output is checked against
its reference outside capture. Measured iterations replay this same graph, and
the final output is checked again. Reports include `cuda_graph` and the replay
correctness checks. Stage debug synchronization is incompatible with capture.

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmarks/kernels/benchmark_deepep_v2_layout.py \
  --expert-kernel grouped-fp8 --cuda-graph \
  --case expanded-nosync --tokens-per-rank 8192,8192 \
  --output /tmp/expanded-nosync-graph.json
```

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
excluded. Graph capture and correctness replays are also excluded from timing.
Per-step synchronization changes execution cadence, so these numbers
are neither serving latency nor steady-state asynchronous throughput.

The default `scalar` expert has no GEMM and cannot establish preparation savings.
Select `--expert-kernel grouped-fp8` for the first direct grouped-GEMM prototype.
That mode dispatches FP8 tokens and block scales, and runs one grouped DeepGEMM
with expert-dependent diagonal weights. Inputs and weights are exactly
representable in FP8, so the same scalar reference checks the actual GEMM path.

For non-expanded inputs it invokes the existing `deepgemm_moe_permute` and
`deepgemm_unpermute_and_reduce`. For expanded inputs it uses DeepEP's received
activation tensor directly, builds GPU expert IDs, and weights output rows
before combine. It skips both input permutation and output unpermutation.
DeepEP expert alignment is set to the grouped GEMM alignment. DeepEP leaves
padding scales uninitialized, while DeepGEMM's FP32-to-UE8M0 conversion reads
them and requires nonnegative exponent-only values. The prototype replaces
invalid-row scales with `1` before conversion. Direct activations therefore
still require scale preparation. GPU-prefix reconstruction uses Torch as a correctness
prototype; tune/fuse it only after validation. No production threshold or
rank-local automatic policy is introduced.

The grouped experiment is one dense GEMM with synthetic diagonal weights, not
a full MoE layer or a serving workload. After correctness and paired timing,
add a second GEMM and activation with realistic weights, then integrate the
layout contract into the expert backend and validate full forwards, accuracy
and non-overloaded serving. Coordinate with existing PR #51589, which already
reconstructs sync-less expanded metadata.

Reference implementation:
<https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/elastic/buffer.hpp>
