<!-- markdownlint-disable -->

# Kernel and Operator Fusions

vLLM applies a set of kernel/operator fusions at compile time (via custom [`torch.compile`](torch_compile.md) Inductor passes)
to separate optimizations from model definitions and avoid breaking layer abstractions in model code. 
These fusions are controlled by fields in [`PassConfig`](../configuration/engine_args.md) and are automatically enabled
at appropriate [optimization levels](optimization_levels.md).

## Quick Reference

The table below maps each fusion to its controlling flag/config knob, the
operations it fuses, what level enables it by default, and an indicative speedup.
The last column indicates whether the fusion requires the entire model graph to be
visible (either via Inductor partition or `splitting_ops=[]`).

> Note that speedup depends heavily on the exact model, batch size, and hardware. 
> If tuning performance by hand, always benchmark your exact use-case with and without the fusion to verify the impact.

| Fusion                                                                    | `PassConfig` flag            | Fused operations                               | Default at                     | E2E Speedup        | Requires fullgraph |
|---------------------------------------------------------------------------|------------------------------|------------------------------------------------|--------------------------------|--------------------|--------------------|
| [Attention + Quant](#attention--quantization-fuse_attn_quant)             | `fuse_attn_quant`            | Attention output → FP8/NVfp4 quant             | Off by default                 | 3-7%               | Yes                |
| [AllReduce + RMSNorm](#allreduce--rmsnorm-fuse_allreduce_rms)             | `fuse_allreduce_rms`         | All-reduce → RMSNorm (+residual_add) (→ quant) | O2 (Hopper/Blackwell + TP > 1) | 5-20%              | No                 |
| [QK Norm + RoPE](#qk-norm--rope-enable_qk_norm_rope_fusion)               | `enable_qk_norm_rope_fusion` | Q/K RMSNorm → rotary embedding                 | Off by default                 | 2-3%               | No                 |
| [Sequence Parallelism](#sequence-parallelism-enable_sp)                   | `enable_sp`                  | AllReduce → ReduceScatter + AllGather          | Off by default                 | Prereq for AsyncTP | Yes                |
| [AsyncTP GEMM + collective](#asynctp-gemm--collective-overlap-fuse_gemm_comms) | `fuse_gemm_comms`       | GEMM → reduce-scatter / all-gather → GEMM      | Off by default                 | 7-10%              | Yes                |
| [RMSNorm + Quant](#rmsnorm--quantization-fuse_norm_quant)                 | `fuse_norm_quant`            | RMSNorm (+residual add) → FP8/FP4 quant        | O1 (conditional)               | 1-4%               | No                 |
| [SiLU+Mul + Quant](#silumul--quantization-fuse_act_quant)                 | `fuse_act_quant`             | SiLU+Mul activation → FP8/FP4 quant            | O1 (conditional)               | 1-4%               | No                 |
| [RoPE + KV-Cache Update](#rope--kv-cache-update-fuse_rope_kvcache)        | `fuse_rope_kvcache`          | Rotary embedding → KV cache write              | O1 (ROCm/AITER only)           | TBD                | No                 |
| [RMSNorm + Padding](#rmsnorm--padding-fuse_act_padding)                   | `fuse_act_padding`           | Residual add + RMSNorm → padding               | O1 (ROCm/AITER only)           | TBD                | No                 |

---

## Enabling / Disabling Fusions

Fusions are exposed through `PassConfig`, which is nested inside `CompilationConfig`:

```python
from vllm import LLM
from vllm.config import CompilationConfig, PassConfig

llm = LLM(
    model="...",
    optimization_level=2, # Default optimization level
    compilation_config=CompilationConfig(
        pass_config=PassConfig(
            fuse_norm_quant=True,
            fuse_act_quant=True,
            fuse_allreduce_rms=False,  # disable a specific fusion
        )
    ),
)
```

Fusions can also be enabled using command-line flags with any `vllm ...` command:

```bash
# Enable O2 defaults, but turn off allreduce fusion
vllm serve meta-llama/Llama-3.1-8B-Instruct -O2 -cc.pass_config.fuse_allreduce_rms=False

# The above is equivalent to the more verbose:
vllm serve meta-llama/Llama-3.1-8B-Instruct -O2 --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'

# Same syntax in other commands, e.g. vllm bench:
vllm bench latency --model=meta-llama/Llama-3.1-8B-Instruct -O2 -cc.pass_config.fuse_allreduce_rms=False
```

Fields set explicitly by the user always take precedence over optimization-level defaults.

---

## Fusion Details

### RMSNorm + Quantization (`fuse_norm_quant`)
> [!WARNING]
> On NVIDIA, Inductor actually generates a faster fused kernel than our custom CUDA kernel.
> Hence this fusion is only enabled when either rms_norm or quant_fp8 is using a custom kernel.

**What it fuses.** Combines the custom `rms_norm` / `fused_add_rms_norm`
operations with subsequent FP8 quantization into a single fused kernel,
eliminating an intermediate read/write of the full-precision activation tensor.
Two variants are fused:

- *Plain RMSNorm + quant*: `rms_norm(x) → quant_fp8(y)`
- *Fused-add RMSNorm + quant*: `fused_add_rms_norm(x, residual) → quant_fp8(y)` — also updates the residual in-place.

Note that AITER fusions are currently in a separate pass in `vllm.compilation.passes.fusion.rocm_aiter_fusion`.

Supported quantization scheme/hardware combinations:
- FP8 static per-tensor: CUDA & HIP kernel
- FP8 dynamic per-token: CUDA & HIP kernel, AITER
- FP8 dynamic per-token-group (128/64): CUDA & HIP kernel, AITER 

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/rms_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rms_quant_fusion.py)
- ROCm AITER pass: [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)
- CUDA/HIP kernels: [`csrc/layernorm_quant_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_quant_kernels.cu)

---

### SiLU+Mul + Quantization (`fuse_act_quant`)
> [!WARNING]
> Same as `fuse_norm_quant`: on NVIDIA, Inductor generates a faster fused kernel than our custom ops.
> This fusion is only enabled when either `silu_and_mul` or `quant_fp8` custom ops are active,
> or for NVfp4-quantized models (where FP4 quant is always a custom op).

**What it fuses.** Fuses the `silu_and_mul` gate-up projection activation (used in LLaMA/Mistral FFN
layers) with a subsequent quantization step into a single kernel, avoiding materialization of the
full-precision post-activation tensor.

Note that AITER fusions are in a separate pass in `vllm.compilation.passes.fusion.rocm_aiter_fusion`.

Supported quantization scheme/hardware combinations:
- FP8 static per-tensor: CUDA & HIP kernel
- NVfp4 dynamic: CUDA only (requires `scaled_fp4_quant`)
- FP8 per-token-group (128): ROCm AITER only

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/act_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/act_quant_fusion.py)
- ROCm AITER pass: [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)
- CUDA/HIP kernels: [`csrc/quantization/`](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/)

---

### Attention + Quantization (`fuse_attn_quant`)
> [!WARNING]
> `fuse_attn_quant` is currently not enabled at any optimization level by default and must be set
> explicitly. It requires the full model graph to be visible (fullgraph compilation or Inductor
> partition) and `CompilationConfig.static_forward_context` to contain attention layer metadata.

**What it fuses.** Fuses the attention output quantization directly after the attention computation,
removing an extra full pass over the output tensor. Patterns covered:

- `Attention → FP8 static quant`
- `Attention → NVfp4 dynamic quant` (CUDA only, requires `scaled_fp4_quant`)

FlexAttentionImpl does not yet support fused output quantization.

Supported hardware: CUDA (SM70+ for FP8; SM90+ recommended for NVfp4).

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/attn_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/attn_quant_fusion.py)

---

### AllReduce + RMSNorm (`fuse_allreduce_rms`)
> [!WARNING]
> TP+DP and TP+PP combinations are currently broken
> ([#34458](https://github.com/vllm-project/vllm/issues/34458) and
> [#35426](https://github.com/vllm-project/vllm/issues/35426)).
> Only supported on NVIDIA Hopper (SM90) and Blackwell (SM100) with FlashInfer installed.

**What it fuses.** Fuses the tensor-parallel all-reduce collective with the subsequent residual add,
RMSNorm, and optionally a quantization step into a single FlashInfer / TRT-LLM communication kernel,
reducing the number of synchronisation barriers per transformer layer.

Patterns covered:
- `AllReduce → RMSNorm`
- `AllReduce → residual add → RMSNorm`
- Both with optional suffix: `→ FP8 static quant` or `→ NVfp4 dynamic quant`

The maximum tensor size below which the fused kernel is used is hardware-dependent (64 MB for TP=2
on SM90/SM100) and configurable via `PassConfig.fi_allreduce_fusion_max_size_mb`.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/allreduce_rms_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/allreduce_rms_fusion.py)
- FlashInfer all-reduce: [`vllm/distributed/device_communicators/flashinfer_all_reduce.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/flashinfer_all_reduce.py)
- Benchmark: [`benchmarks/kernels/benchmark_fused_collective.py`](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_fused_collective.py)

---

### Sequence Parallelism (`enable_sp`)
> [!NOTE]
> Sequence Parallelism itself does not directly improve performance; it is a prerequisite for the
> AsyncTP pass (`fuse_gemm_comms`). SP is only applied above a minimum token threshold that is
> auto-configured based on device capability and model `hidden_size`. Currently only active on
> H100/SM90 for models with `hidden_size >= 8192`. The threshold is configurable via
> `PassConfig.sp_min_token_num`.

**What it fuses.** Replaces all-reduce collectives with reduce-scatter + local RMSNorm + all-gather,
splitting the sequence dimension across TP ranks. This restructures the graph so the subsequent AsyncTP
pass can fuse the reduce-scatter / all-gather with the surrounding GEMMs.

The general transformation:

```
Input → AllReduce → RMSNorm → Output
becomes:
Input → ReduceScatter → local RMSNorm → AllGather → Output
```

Patterns covered:
- First block: `AllReduce → RMSNorm` → `ReduceScatter → RMSNorm → AllGather`
- Middle blocks: `AllReduce → fused_add_RMSNorm` → `ReduceScatter → fused_add_RMSNorm → AllGather`
- Both with optional `→ FP8 static quant` suffix

Requires: `use_inductor_graph_partition=True` **or** piecewise compilation with batch sizes
divisible by `tensor_parallel_size`.

Supported hardware: NVIDIA CUDA only (SM90 enabled by default; other capabilities require explicit
`sp_min_token_num` configuration).

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/sequence_parallelism.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/sequence_parallelism.py)

---

### AsyncTP GEMM + Collective Overlap (`fuse_gemm_comms`)
> [!WARNING]
> Requires `enable_sp=True`. This pass is a no-op if Sequence Parallelism has not been applied.
> Requires symmetric-memory support (`torch.distributed._symmetric_memory`) on CUDA.

**What it fuses.** After Sequence Parallelism transforms the graph, fuses GEMM kernels with the
surrounding reduce-scatter (output projection) and all-gather (input projection) using
`torch.ops.symm_mem` symmetric-memory primitives, overlapping communication and computation.

Patterns covered:
- `GEMM → reduce-scatter` → `fused_matmul_reduce_scatter`
- `all-gather → GEMM` → `all_gather_matmul`
- FP8 scaled variants of both patterns (CUTLASS)

Supported hardware: NVIDIA CUDA (requires `tensor_parallel_size > 1`).

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/collective_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/collective_fusion.py)
- Sequence parallelism pass: [`vllm/compilation/passes/fusion/sequence_parallelism.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/sequence_parallelism.py)

---

### QK Norm + RoPE (`enable_qk_norm_rope_fusion`)
> [!NOTE]
> Only applicable to models that apply per-head RMSNorm to Q and K before rotary positional
> embedding (e.g. some Gemma or Cohere variants). Not enabled by default at any optimization level.

**What it fuses.** Fuses the sequence: split QKV → reshape → Q/K RMSNorm → reshape → rotary
embedding into a single `fused_qk_norm_rope` CUDA kernel.

```
# Unfused:
q, k, v = split(qkv)
q_norm = rms_norm(q.view(heads))
k_norm = rms_norm(k.view(kv_heads))
q_rope, k_rope = rotary_embedding(q_norm, k_norm, ...)

# Fused:
fused_qk_norm_rope(qkv, ...)
```

Supported hardware: CUDA (SM70+) only.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/qk_norm_rope_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py)
- CUDA kernel: [`csrc/ops.h`](https://github.com/vllm-project/vllm/blob/main/csrc/ops.h) (`fused_qk_norm_rope`)

---

### RoPE + KV-Cache Update (`fuse_rope_kvcache`)
> [!NOTE]
> ROCm/AITER-only. Not available on NVIDIA CUDA or CPU.
> The fusion only fires when `num_tokens ≤ rope_kvcache_fusion_max_token_num` (default: 256);
> larger batches fall back to unfused kernels.

**What it fuses.** Fuses the rotary positional embedding kernel with the KV-cache scatter/write into
a single kernel, avoiding separate reads and writes of the key and value tensors.

Requires: AMD ROCm with AITER enabled, the `rotary_embedding` custom op active, and
`use_inductor_graph_partition=True`. The token threshold is configurable via
`PassConfig.rope_kvcache_fusion_max_token_num`.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/rope_kvcache_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rope_kvcache_fusion.py)

---

### RMSNorm + Padding (`fuse_act_padding`)
> [!NOTE]
> ROCm/AITER-only. Targeted at models with `hidden_size=2880` (GPT-OSS variants).
> Not available on NVIDIA CUDA or CPU.

**What it fuses.** Fuses a residual add + RMSNorm with a subsequent padding operation that pads
the hidden dimension to a multiple required by downstream AITER Triton GEMM kernels.

Requires: AMD ROCm with AITER RMSNorm enabled and AITER Triton GEMMs *not* enabled
(the padding is only needed without them), and `model_config.hidden_size == 2880`.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py) (`RocmAiterTritonAddRMSNormPadFusionPass`)

---

### Fused Add RMSNorm (kernel-level)
> [!NOTE]
> This is a kernel-level fusion (not a compilation pass) that is always active when
> `RMSNorm.forward_cuda` is called with a non-`None` residual argument. Not user-configurable.

**What it fuses.** Performs residual addition and RMSNorm normalization in a single kernel pass,
avoiding an intermediate materialization of the residual-added tensor.

```python
# Equivalent unfused:
residual = residual + x
x = rms_norm(residual, weight)

# Fused (in-place, single kernel):
ops.fused_add_rms_norm(x, residual, weight, epsilon)
# x → normalised output; residual → updated residual
```

On SM100 (Blackwell) with the Oink library, an alternative `oink.fused_add_rms_norm` path is taken
when the tensor layout is compatible.

Supported hardware/kernels:
- CUDA (SM70+): `torch.ops._C.fused_add_rms_norm` (`csrc/layernorm_kernels.cu`)
- CUDA SM100 (Blackwell): `torch.ops.oink.fused_add_rms_norm` (Oink fast path)
- AMD ROCm: equivalent HIP kernel

Alignment requirements: input, residual, and weight pointers must be 16-byte aligned and
`hidden_size` a multiple of 8 for the vectorised (width-8) path; otherwise a scalar fallback is used.

**Code locations.**

- Layer: [`vllm/model_executor/layers/layernorm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py)
- Python op: [`vllm/_custom_ops.py`](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) (`fused_add_rms_norm`)
- CUDA kernel: [`csrc/layernorm_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu)
- Benchmark: [`benchmarks/kernels/benchmark_rmsnorm.py`](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_rmsnorm.py)

---

### Fused Mixture-of-Experts (`fused_moe`)
> [!NOTE]
> Always active for MoE models. Controlled by the `CustomOp` system
> (see [`VLLM_CUSTOM_OPS`](../configuration/env_vars.md)).

**What it fuses.** The `FusedMoE` custom op implements the entire MoE forward pass — token routing,
batched expert GEMMs (gate+up projections), activation (SiLU/GELU), down projection, and weighted
expert reduction — as a single fused unit. Multiple backend implementations are selected
automatically based on hardware and quantization.

| Backend | Quant schemes | Hardware |
|---|---|---|
| Triton (default) | FP16, BF16, FP8 (dynamic / static) | NVIDIA CUDA SM70+, AMD ROCm |
| CUTLASS | FP8 (block / tensor scale) | NVIDIA SM90+ |
| FlashInfer CuteDSL | FP8, NVfp4 | NVIDIA SM90+ |
| FlashInfer TRT-LLM | FP8, NVfp4 | NVIDIA SM90+ |
| ROCm AITER | FP16, BF16, FP8 (group / token), INT8 | AMD CDNA (gfx940+) |
| GGUF | GGUF quantized types | NVIDIA CUDA |

ROCm AITER backend requires `rocm_aiter_ops` to be enabled; falls back to Triton otherwise.
FlashInfer-based backends require FlashInfer to be installed.
CUTLASS FP8 path requires SM90+ and the compiled CUTLASS extension.

**Code locations.**

- Layer: [`vllm/model_executor/layers/fused_moe/layer.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/layer.py)
- Triton kernel: [`vllm/model_executor/layers/fused_moe/fused_moe.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)
- ROCm AITER: [`vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py)
- FlashInfer CuteDSL: [`vllm/model_executor/layers/fused_moe/flashinfer_cutedsl_moe.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/flashinfer_cutedsl_moe.py)
- FlashInfer TRT-LLM: [`vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py)
- GGUF: [`vllm/model_executor/layers/quantization/gguf.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/gguf.py)
- Design doc: [Fused MoE Modular Kernel](fused_moe_modular_kernel.md)

---

## Support Matrix

The matrix below shows which fusions are supported on each hardware/software
combination for common numerical precisions.  **Yes** = fully supported;
**Partial** = supported with limitations noted; **No** = not supported;
**—** = not applicable.

### Compilation-Time Fusions

| Fusion | NVIDIA SM70–89 (Volta–Ada) FP16/BF16 | NVIDIA SM90 (Hopper) FP8 | NVIDIA SM100 (Blackwell) FP8/NVfp4 | AMD ROCm / CDNA FP16/BF16 | AMD ROCm / CDNA FP8 | CPU |
|---|---|---|---|---|---|---|
| `fuse_norm_quant` (RMSNorm+quant) | Partial (no quant) | Yes | Yes | Yes | Yes | No |
| `fuse_act_quant` (SiLU+Mul+quant) | Partial (no quant) | Yes | Yes | Yes | Yes | No |
| `fuse_attn_quant` (Attn+quant) | No | Yes | Yes (NVfp4) | No | No | No |
| `fuse_allreduce_rms` | No | Yes | Yes | No | No | No |
| `enable_sp` (Sequence Parallelism) | No | Yes | No† | No | No | No |
| `fuse_gemm_comms` (AsyncTP) | No | Yes | No† | No | No | No |
| `enable_qk_norm_rope_fusion` | Yes | Yes | Yes | No | No | No |
| `fuse_rope_kvcache` (ROCm/AITER) | No | No | No | Yes | Yes | No |
| `fuse_act_padding` (ROCm/AITER) | No | No | No | Yes | Yes | No |

† `enable_sp` and `fuse_gemm_comms` are only auto-configured for SM90 today;
SM100 support requires setting `PassConfig.sp_min_token_num` explicitly.

### Kernel-Level Fusions

| Fusion | NVIDIA SM70–89 | NVIDIA SM90 | NVIDIA SM100 | AMD ROCm | CPU |
|---|---|---|---|---|---|
| Fused Add RMSNorm (kernel) | Yes | Yes | Yes (+ Oink fast path) | Yes | No |
| Fused MoE — Triton | Yes (FP16/BF16) | Yes (FP8) | Yes (FP8) | Yes (FP16/BF16/FP8) | No |
| Fused MoE — CUTLASS | No | Yes (FP8) | Yes (FP8) | No | No |
| Fused MoE — FlashInfer CuteDSL | No | Yes (FP8/NVfp4) | Yes (FP8/NVfp4) | No | No |
| Fused MoE — ROCm AITER | No | No | No | Yes (FP8/INT8) | No |

### Quantization-Scheme Columns

For quick reference, the FP16 and BF16 columns in the compilation-time fusion
table above indicate that the *activation* dtype is FP16/BF16; those fusions
produce quantised *outputs* (FP8 or FP4).  Where labelled "Partial (no quant)"
the RMSNorm or SiLU+Mul kernel fires but without the output quantization step.

---

## See Also

- [Optimization Levels](optimization_levels.md) — high-level presets that set
  fusion defaults.
- [torch.compile in vLLM](torch_compile.md) — how the Inductor pass pipeline
  works.
- [Attention Backends](attention_backends.md) — attention-specific kernel
  selection.
- [Fused MoE Modular Kernel design](fused_moe_modular_kernel.md) — deep-dive
  into the MoE fusion architecture.
- [`PassConfig` API reference](../configuration/engine_args.md)
