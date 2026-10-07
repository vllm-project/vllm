# DeepEP v2 expanded-prefill follow-up

The first experiment separates layout from host count synchronization. It uses
fresh dispatch handles and leaves the production always-off policy unchanged.

The follow-up also implements all four combinations in vLLM's
`DeepEPV2PrepareAndFinalize`. Select them independently with `KernelConfig`
fields `deepep_v2_do_expand` and `deepep_v2_do_cpu_sync`; both default to `False`.
For example, a DeepEP v2 server can select expanded eager dispatch with:

```bash
vllm serve MODEL --data-parallel-size 2 --enable-expert-parallel \
  --all2all-backend deepep_v2 \
  --enforce-eager \
  --kernel-config '{"deepep_v2_do_expand":true,"deepep_v2_do_cpu_sync":true}'
```

Use the same flags on every EP rank. CPU-sync configurations require eager
execution; attempting CPU count polling during capture raises an error before
dispatch. With CPU sync disabled, either layout supports capture. No adaptive
threshold or rank-local layout selection is introduced.

The integrated receiver reconstructs expanded routing IDs and real expert
counts from GPU prefixes, including alignment gaps and unused capacity. It
clears padding before quantization, while already quantized activations retain
the dispatch allocation. Padding scales and weights are sanitized separately.

For block FP8, select `"moe_backend":"deep_gemm"` in the kernel config to consume
expanded inputs directly. The FP8 oracle requests DeepGEMM's expert alignment
from DeepEP, and receive metadata identifies the aligned, already grouped rows.
DeepGEMM skips input permutation and output unpermutation for both expanded
configurations. It prepares scales, masks unused rows inside fused activation/quantization,
and weights output rows directly before DeepEP combine.
Neither choice requires CPU receive counts for input preparation.

```bash
vllm serve FP8_BLOCK_MODEL --data-parallel-size 2 --enable-expert-parallel \
  --all2all-backend deepep_v2 \
  --kernel-config '{"moe_backend":"deep_gemm","deepep_v2_do_expand":true,"deepep_v2_do_cpu_sync":false}'
```

The integration also supports direct expanded input for DeepGEMM FP4 experts.
It includes the padding-aware schedulers from upstream PRs #59044 and #59128.
The FP4 path prepares local IDs, safe scales, and live expert endpoints in one
GPU kernel; row count does not specialize the kernel. Both GEMMs and activation
quantization use expert endpoints to skip padded computation. This preparation
does not copy activations or read CPU receive counts.

Non-expanded DeepGEMM inputs retain the permutation path. Other expert backends
retain their compatibility paths.
The standalone benchmark below separately demonstrates one grouped GEMM.

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
correctness checks.

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmarks/kernels/benchmark_deepep_v2_layout.py \
  --cuda-graph \
  --case expanded-nosync --tokens-per-rank 8192,8192 \
  --output /tmp/expanded-nosync-graph.json
```

Repeat for all four cases, with matching dimensions, first at `512,512`, then
`8192,8192`, `8192,512`, and `8192,0`. Run `16384,16384` only after those pass.
Use the same allocated GPUs for the matrix; repeat in reverse case order before
interpreting small differences. A worker that hangs or fails is a correctness
failure, not a timing result. Record GPU topology, immutable runtime identity,
source hash and all rank reports.

The measurement includes allocation, dispatch, GPU metadata, grouped FP8
GEMM, combine, and per-step CUDA completion synchronization. The reported
latency is the median of per-step maximum rank wall times. Memory reports are
PyTorch allocator peaks, not total device footprints. Warmup/compilation is
excluded. Graph capture and correctness replays are also excluded from timing.
Per-step synchronization changes execution cadence, so these numbers
are neither serving latency nor steady-state asynchronous throughput.

The benchmark dispatches FP8 tokens and block scales, and runs one grouped DeepGEMM
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
a full MoE layer or a serving workload. The vLLM integration is validated
separately with full two-GEMM reference comparisons and graph replay tests in
`tests/kernels/moe/test_deepep_v2_moe.py`. Direct block-FP8 DeepGEMM forwards bypass both permutations in the integrated
path. Other expert backends and choosing a production threshold remain
follow-up work.
Coordinate with existing PR #51589, which already reconstructs sync-less
expanded metadata.

Eight-B200 NVLink serving validation with DeepSeek-V4.1-Flash compared
non-expanded, expanded direct with unfused preparation, and expanded direct
with fused preparation on the same padding-aware stack. At fixed 8192 input /
1024 output tokens and concurrency eight, median TPOT across three rounds was
10.744 / 13.425 / 12.613 ms. Fusion improved expanded TPOT by about 6%, but
non-expanded remained faster. TTFT was 524.59 / 576.19 / 588.45 ms; sampled
total GPU memory was 547.90 / 572.80 / 574.24 GiB. There is no TTFT or memory
improvement demonstrated by this experiment. Full GSM8K scores were
1271 / 1277 / 1278 correct out of 1319, with truncations counted as incorrect.
These scores do not establish output equivalence. All configurations used
no CPU sync and graph decode; cases ran sequentially rather than interleaved.
The subsequent runtime-row-count and 64-bit preparation-offset fix passed
GPU correctness and profiling checks, but did not receive another full serving
rerun. Neither expansion nor CPU sync is enabled by default.

Reference implementation:
<https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/elastic/buffer.hpp>
