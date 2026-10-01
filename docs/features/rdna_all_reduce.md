# RDNA HIP all-reduce

vLLM provides an opt-in HIP all-reduce backend for single-node tensor parallel
groups on homogeneous `gfx1100` (RDNA3) or `gfx1201` (RDNA4) GPUs. It is disabled
by default. All participating GPUs must be visible to every worker and support
peer access along the kernel's communication topology.

```bash
VLLM_ROCM_USE_RDNA_ALL_REDUCE=1 vllm serve /path/to/model \
    --tensor-parallel-size 4
```

Use a ROCm build containing the RDNA kernels. Building vLLM from source includes
them in `_rocm_C` when `PYTORCH_ROCM_ARCH` includes a supported architecture.
There is no separate runtime compilation or shared-library path to configure.
The startup log lists `RDNA_HIP` when initialization succeeds.

## Selection and fallback

The backend is used only for single-token, uniform decode batches during GPU
graph capture; replay executes the captured HIP kernels. Prefill, mixed batches,
eager execution, and calls without a forward context continue through the
existing all-reduce fallback. The fail-closed no-context rule also covers model
warmup work that runs before the forward context is installed.
`--disable-custom-all-reduce` takes precedence over the environment variable.
Batch-invariant mode and microbatch overlap (including DBO) disable this backend.

| Property | Supported range |
| --- | --- |
| Tensor parallel size | 2 or 4, single node, distinct GPUs of the same architecture |
| Input dtype | BF16 or FP16 |
| Batch semantics | Single-token uniform decode with `num_reqs == num_tokens` |
| Input shape | Two dimensions: `[batch, hidden]` |
| Dimensions | `1 <= batch <= 128`, `1 <= hidden <= 8192` |
| Element count | Multiple of 8 |
| RDNA3 selection | Fewer than 131072 elements |
| RDNA4 selection | Up to 1048576 elements |

The tensor's first dimension must equal the decode batch size in the forward
context. Selection applies two independent limits. The shape contract is common
to both architectures and includes the maximum
`[128, 8192]` shape, which has 1048576 elements. RDNA4 can therefore select the
custom kernel at exactly 1048576 elements; shapes with either dimension beyond
the limits continue through the existing fallback. The examples below assume a
matching single-token decode context; the same tensors fall back outside decode.

RDNA3 has an additional performance-routing threshold. Inputs with 131072 or
more elements use the fallback even when their dimensions satisfy the common
shape contract. This is not a kernel capacity or correctness restriction: the
RDNA3 kernels can execute larger inputs, but their tagged/bulk ring region did
not consistently outperform RCCL in graph benchmarks. The conservative route
keeps the custom kernel for the measured smaller-message region until the
large-message implementation is improved and revalidated.

| Example shape | Elements | RDNA3 | RDNA4 |
| --- | ---: | --- | --- |
| `[16, 8184]` | 130944 | Custom | Custom |
| `[16, 8192]` | 131072 | Fallback | Custom |
| `[128, 8192]` | 1048576 | Fallback | Custom |
| `[129, 8192]` | Outside shape contract | Fallback | Fallback |

Returning to the fallback means continuing through vLLM's existing all-reduce
dispatch. The final backend depends on the enabled configuration; in a standard
ROCm configuration it is normally PyNCCL/RCCL rather than necessarily being a
direct RCCL call at this decision point.

Noncontiguous or misaligned inputs are copied to a contiguous, aligned tensor
before the kernel call. The input is preserved and the result is returned in a
new tensor. The copy is part of graph execution and can reduce the performance
benefit for these layouts.

All ranks agree on eligibility and initialization success before enabling the
backend. Missing bindings, unsupported hardware, inaccessible peers, or an IPC
initialization failure disable it for the whole group. The standalone kernel's
`VLLM_RDNA3_*` and `VLLM_RDNA4_*` tuning overrides are not supported by this
integration. Runtime kernel errors are propagated, not retried with another
collective after communication has started.

## Numerical behavior and lifetime

Reduction order can differ from RCCL and between kernel algorithms. Results are
not guaranteed to match RCCL bit for bit. BF16/FP16 intermediate rounding and
overflow remain possible, including for inputs whose mathematical sum is finite.

Graphs using the same communicator must execute serially. Concurrent replay on
multiple streams is unsupported. IPC storage is allocated before capture and
remains alive until communicator destruction. All ranks must finish their graphs
and destroy the communicator together before destroying its process groups.

## Validation

The distributed tests exercise the public vLLM all-reduce entry point, graph
replay, changed inputs, rank-dependent alignment, decode admission, prefill and
no-context fallback, eager fallback, and unsupported shape fallback:

```bash
.venv/bin/python -m pytest tests/distributed/test_custom_all_reduce.py \
    -k rdna -v
```

Use `HIP_VISIBLE_DEVICES` to select a homogeneous group before starting the
tests. The standalone kernel benchmark is
`benchmarks/kernels/benchmark_rdna_custom_all_reduce.py`; its measurements do
not include Python dispatch, model execution, or extra layout-staging copies.
