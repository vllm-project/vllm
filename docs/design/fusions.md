<!-- markdownlint-disable -->

# Kernel and Operator Fusions

vLLM applies a set of kernel/operator fusions at compile time (via
[`torch.compile`](torch_compile.md) Inductor passes) and at the kernel level
(custom CUDA/Triton/ROCm kernels) to reduce memory traffic and kernel launch
overhead.  Most fusions are controlled by fields in
[`PassConfig`](../configuration/engine_args.md) and are automatically enabled
at appropriate [optimization levels](optimization_levels.md).

## Quick Reference

The table below maps each fusion to its controlling flag/config knob, the
operations it fuses, when it is enabled by default, and an indicative speedup.
Because speedup depends heavily on model architecture, sequence length, batch
size, and hardware, numbers marked **TBD** are placeholders pending systematic
benchmarking.

| Fusion | `PassConfig` flag | Fused operations | Default at | Speedup (decode) |
|---|---|---|---|---|
| [RMSNorm + Quant](#rmsnorm--quantization-fuse_norm_quant) | `fuse_norm_quant` | RMSNorm (+residual add) → FP8/FP4 quant | O1 (conditional) | TBD |
| [SiLU+Mul + Quant](#silumul--quantization-fuse_act_quant) | `fuse_act_quant` | SiLU+Mul gate activation → FP8/FP4 quant | O1 (conditional) | TBD |
| [Attention + Quant](#attention--quantization-fuse_attn_quant) | `fuse_attn_quant` | Attention output → FP8/NVfp4 quant | O2 (disabled†) | TBD |
| [AllReduce + RMSNorm](#allreduce--rmsnorm-fuse_allreduce_rms) | `fuse_allreduce_rms` | All-reduce → residual add → RMSNorm (→ quant) | O2 (Hopper/Blackwell + TP > 1) | TBD |
| [GEMM + Communication Overlap](#gemm--communication-overlap-fuse_gemm_comms) | `fuse_gemm_comms` | GEMM → reduce-scatter / all-gather → GEMM | O2 (disabled†) | TBD |
| [QK Norm + RoPE](#qk-norm--rope-enable_qk_norm_rope_fusion) | `enable_qk_norm_rope_fusion` | Per-head Q/K RMSNorm → rotary embedding | Off by default | TBD |
| [RoPE + KV-Cache Update](#rope--kv-cache-update-fuse_rope_kvcache) | `fuse_rope_kvcache` | Rotary embedding → KV cache write | O1 (ROCm/AITER only) | TBD |
| [RMSNorm + Padding](#rmsnorm--padding-fuse_act_padding) | `fuse_act_padding` | Residual add + RMSNorm → padding | O1 (ROCm/AITER, hidden=2880 only) | TBD |
| [Fused Add RMSNorm](#fused-add-rmsnorm-kernel) | *(kernel-level)* | Residual add + RMSNorm | Always (CUDA/ROCm) | TBD |
| [Fused MoE](#fused-mixture-of-experts-fused_moe) | *(custom op)* | Routing + expert GEMMs + activation + reduce | Always (MoE models) | TBD |

† `IS_QUANTIZED` and `IS_DENSE` flags are currently hard-coded to `False` in
the optimization-level defaults; see
[`vllm/config/vllm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py).

> **Note on speedup numbers.** Actual speedups depend on model size, batch
> size, sequence length, GPU generation, and quantization scheme.  Benchmarks
> can be found under [`benchmarks/kernels/`](https://github.com/vllm-project/vllm/tree/main/benchmarks/kernels).
> The TBD entries will be filled as systematic numbers become available.

---

## Enabling / Disabling Fusions

Fusions are exposed through `PassConfig`, which is nested inside
`CompilationConfig`:

```python
from vllm import LLM
from vllm.config import CompilationConfig, PassConfig

llm = LLM(
    model="...",
    compilation_config=CompilationConfig(
        pass_config=PassConfig(
            fuse_norm_quant=True,
            fuse_act_quant=True,
            fuse_allreduce_rms=False,  # disable a specific fusion
        )
    ),
)
```

Alternatively, set an [optimization level](optimization_levels.md) and override
individual flags:

```bash
# Enable O2 defaults, but turn off allreduce fusion
vllm serve meta-llama/Llama-3.1-8B-Instruct -O2 \
  --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'
```

Fields set explicitly by the user always take precedence over optimization-level
defaults.

---

## Fusion Details

### RMSNorm + Quantization (`fuse_norm_quant`)

**What it fuses.** Combines the custom `rms_norm` / `fused_add_rms_norm`
operations with subsequent FP8 or NVfp4 quantization into a single fused kernel,
eliminating an intermediate read/write of the full-precision activation tensor.
Two variants are fused:

- *Plain RMSNorm + quant*: `rms_norm(x) → quantize(y)`
- *Fused-add RMSNorm + quant*: `fused_add_rms_norm(x, residual) → quantize(y)` — also updates the residual in-place.

Supported quantization schemes: FP8 (static tensor scale, dynamic per-token,
dynamic per-token-group 128 / 64), NVfp4 (dynamic).

**How to enable/disable.**

```python
PassConfig(fuse_norm_quant=True)   # explicit enable
PassConfig(fuse_norm_quant=False)  # explicit disable
```

Enabled automatically at **O1 and above** when either the `rms_norm` or the
`quant_fp8` custom op is active (i.e., not replaced by a Torch built-in).  The
`enable_norm_fusion` helper in `vllm/config/vllm.py` determines this
automatically.

**Supported backends / kernels.**

- CUDA (SM70+): custom fused kernels in `csrc/layernorm_quant_kernels.cu`
- AMD ROCm: standard path; AITER variant available via
  `RocmAiterRMSNormQuantFusionPass` when AITER is enabled.

**Known limitations / fallbacks.**

- Falls back to unfused ops if the custom ops are disabled (e.g., under
  `torch.compile` with Inductor handling the fusion natively).
- The ROCm AITER variant is registered as an additional pass alongside the
  standard `RMSNormQuantFusionPass`.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/rms_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rms_quant_fusion.py)
- ROCm AITER pass: [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)
- CUDA kernels: [`csrc/layernorm_quant_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_quant_kernels.cu)

---

### SiLU+Mul + Quantization (`fuse_act_quant`)

**What it fuses.** Fuses the `silu_and_mul` gate-up projection activation (used
in LLaMA/Mistral FFN layers) with a subsequent FP8 or NVfp4 quantization step.
Avoids materialising the full-precision post-activation tensor.

Supported quantization schemes: FP8 static tensor scale, NVfp4 dynamic.
On ROCm/AITER, a group-quantized FP8 variant (`silu_and_mul_group_fp8_quant`)
is also fused.

**How to enable/disable.**

```python
PassConfig(fuse_act_quant=True)
PassConfig(fuse_act_quant=False)
```

Enabled automatically at **O1 and above** when the `silu_and_mul` or `quant_fp8`
custom op is active, or for NVfp4-quantised models.

**Supported backends / kernels.**

- CUDA (SM70+): `torch.ops._C.silu_and_mul_quant`, `silu_and_mul_nvfp4_quant`
- AMD ROCm/AITER: `AiterSiluMulFp8GroupQuantPattern` via
  `RocmAiterSiluMulFp8GroupQuantFusionPass`

**Known limitations / fallbacks.**

- `silu_and_mul_nvfp4_quant` requires CUDA and the CUDA `scaled_fp4_quant`
  kernel to be present; falls back to unfused path otherwise.
- ROCm AITER group-quant variant is only registered when AITER is enabled.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/act_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/act_quant_fusion.py)
- ROCm AITER pass: [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)
- CUDA kernels: [`csrc/quantization/`](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/)

---

### Attention + Quantization (`fuse_attn_quant`)

**What it fuses.** Fuses the attention output quantization (FP8 static / NVfp4
dynamic) directly after the attention computation, removing an additional
pass over the output tensor.  Patterns covered:

- `Attention → static FP8 quant`
- `Attention → NVfp4 dynamic quant` (CUDA only, requires `scaled_fp4_quant`)

**How to enable/disable.**

```python
PassConfig(fuse_attn_quant=True)
PassConfig(fuse_attn_quant=False)
```

Included in **O2 and above** optimization levels, but currently the
`IS_QUANTIZED` guard is `False` (hard-coded), so it must be enabled explicitly
for now.

**Supported backends / kernels.**

- CUDA (SM70+ for FP8; SM90+ recommended for NVfp4)

**Known limitations / fallbacks.**

- Requires `CompilationConfig.static_forward_context` to contain attention
  layer metadata; logs a warning if no attention layers are found.
- FlexAttentionImpl does not yet support fused output quantization.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/attn_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/attn_quant_fusion.py)

---

### AllReduce + RMSNorm (`fuse_allreduce_rms`)

**What it fuses.** Fuses the tensor-parallel all-reduce collective with the
subsequent residual add and RMSNorm (and optionally an FP8/NVfp4 quantization)
into a single FlashInfer / TRT-LLM collective-communication kernel.  This
reduces the number of synchronisation barriers per transformer layer by
combining communication and computation.

Patterns covered:

- `AllReduce → RMSNorm`
- `AllReduce → residual add → RMSNorm`
- `AllReduce → RMSNorm → FP8 static quant`
- `AllReduce → residual add → RMSNorm → FP8 static quant`
- `AllReduce → RMSNorm → NVfp4 dynamic quant`
- `AllReduce → residual add → RMSNorm → NVfp4 dynamic quant`

**How to enable/disable.**

```python
PassConfig(fuse_allreduce_rms=True)
PassConfig(fuse_allreduce_rms=False)
```

Enabled automatically at **O2 and above** when all of the following hold:
- `tensor_parallel_size > 1`
- Running on CUDA (Hopper SM90 or Blackwell SM100)
- FlashInfer is installed
- `data_parallel_size == 1` (TP+DP combination has a known issue)
- `pipeline_parallel_size == 1` (TP+PP combination has a known issue)

The maximum tensor size below which the fused kernel is used is
hardware-dependent (64 MB for TP=2 on SM90/SM100; see `PassConfig.fi_allreduce_fusion_max_size_mb`).

**Supported backends / kernels.**

- NVIDIA Hopper (SM90) and Blackwell (SM100) only
- Requires [FlashInfer](https://flashinfer.ai/) with TRT-LLM all-reduce backend

**Known limitations / fallbacks.**

- TP+DP and TP+PP combinations are currently broken (tracked in
  [#34458](https://github.com/vllm-project/vllm/issues/34458) and
  [#35426](https://github.com/vllm-project/vllm/issues/35426)).
- Not supported on AMD ROCm or CPU.
- Falls back to separate all-reduce + RMSNorm if FlashInfer is unavailable.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/allreduce_rms_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/allreduce_rms_fusion.py)
- FlashInfer all-reduce: [`vllm/distributed/device_communicators/flashinfer_all_reduce.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/flashinfer_all_reduce.py)
- Benchmark: [`benchmarks/kernels/benchmark_fused_collective.py`](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_fused_collective.py)

---

### GEMM + Communication Overlap (`fuse_gemm_comms`)

**What it fuses.** When [sequence parallelism](torch_compile.md) (`enable_sp`)
is active, this pass further overlaps the GEMM kernel with the subsequent
reduce-scatter (output projection) or precedes it with an all-gather (input
projection), using `torch.ops.symm_mem` symmetric-memory primitives.

Patterns covered:

- `GEMM (matmul) → reduce-scatter` → replaced by `fused_matmul_reduce_scatter`
- `all-gather → GEMM (matmul)` → replaced by `all_gather_matmul`

**How to enable/disable.**

```python
# enable_sp must also be True
PassConfig(enable_sp=True, fuse_gemm_comms=True)
```

Included at **O2 and above** for dense (non-MoE) models, but currently the
`IS_DENSE` guard is `False`; enable explicitly if needed.

**Supported backends / kernels.**

- CUDA; requires symmetric-memory support (`torch.distributed._symmetric_memory`)

**Known limitations / fallbacks.**

- Only active when `enable_sp=True`; the pass is a no-op otherwise.
- Requires `tensor_parallel_size > 1`.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/collective_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/collective_fusion.py)
- Sequence parallelism pass: [`vllm/compilation/passes/fusion/sequence_parallelism.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/sequence_parallelism.py)

---

### QK Norm + RoPE (`enable_qk_norm_rope_fusion`)

**What it fuses.** For models that apply per-head RMSNorm to Q and K before
rotary positional embedding (e.g. some newer models), this pass fuses the whole
sequence — split QKV → reshape → Q/K RMSNorm → reshape → rotary embedding —
into a single `fused_qk_norm_rope` CUDA kernel.

```
# Unfused:
q, k, v = split(qkv)
q_norm = rms_norm(q.view(heads))
k_norm = rms_norm(k.view(kv_heads))
q_rope, k_rope = rotary_embedding(q_norm, k_norm, ...)

# Fused:
fused_qk_norm_rope(qkv, ...)
```

**How to enable/disable.**

```python
PassConfig(enable_qk_norm_rope_fusion=True)
```

Not enabled by any optimization level by default; must be set explicitly.

**Supported backends / kernels.**

- CUDA: `torch.ops._C.fused_qk_norm_rope`

**Known limitations / fallbacks.**

- Only applicable to models that have per-head QK norm (e.g. those using
  `RotaryEmbedding` together with per-head `RMSNorm` on Q and K).

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/qk_norm_rope_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py)
- CUDA kernel: [`csrc/ops.h`](https://github.com/vllm-project/vllm/blob/main/csrc/ops.h) (`fused_qk_norm_rope`)

---

### RoPE + KV-Cache Update (`fuse_rope_kvcache`)

**What it fuses.** On AMD ROCm with AITER, fuses the rotary positional embedding
kernel with the KV-cache scatter/write.  Without this fusion the two operations
require separate reads and writes of the key and value tensors.

**How to enable/disable.**

```python
PassConfig(fuse_rope_kvcache=True)
# rope_kvcache_fusion_max_token_num controls the batch-size threshold (default 256)
PassConfig(fuse_rope_kvcache=True, rope_kvcache_fusion_max_token_num=512)
```

Enabled automatically at **O1 and above** when:
- AMD ROCm with AITER is active
- The `rotary_embedding` custom op is enabled
- `use_inductor_graph_partition` is enabled

The fusion only fires when `num_tokens ≤ rope_kvcache_fusion_max_token_num`;
larger batches (e.g. prefill) fall back to the unfused path.

**Supported backends / kernels.**

- AMD ROCm / CDNA (AITER only)

**Known limitations / fallbacks.**

- Not supported on NVIDIA CUDA or CPU.
- Automatically falls back to separate RoPE + KV-cache kernels when the token
  count exceeds the threshold.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/rope_kvcache_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rope_kvcache_fusion.py)

---

### RMSNorm + Padding (`fuse_act_padding`)

**What it fuses.** On AMD ROCm with AITER, fuses a residual add + RMSNorm with
a subsequent padding operation that pads the hidden dimension to a multiple
required by downstream AITER Triton GEMM kernels.  Targeted at models with
`hidden_size=2880` (e.g. GPT-OSS variants).

**How to enable/disable.**

```python
PassConfig(fuse_act_padding=True)
```

Enabled automatically at **O1 and above** when:
- AMD ROCm with AITER RMSNorm is enabled
- AITER Triton GEMMs are *not* enabled (the padding is only needed without them)
- `model_config.hidden_size == 2880`

**Supported backends / kernels.**

- AMD ROCm / CDNA (AITER only)

**Known limitations / fallbacks.**

- Narrow applicability: only for specific hidden sizes.
- Not supported on NVIDIA CUDA or CPU.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py) (`RocmAiterTritonAddRMSNormPadFusionPass`)

---

### Fused Add RMSNorm (kernel-level)

**What it fuses.** A CUDA kernel that performs residual addition and RMSNorm in
a single pass over the data.  This is a *kernel-level* fusion (not a compilation
pass) and is always used when `RMSNorm.forward_cuda` is called with a non-`None`
`residual` argument.

```python
# Equivalent unfused:
residual = residual + x
x = rms_norm(residual, weight)

# Fused (in-place, single kernel):
ops.fused_add_rms_norm(x, residual, weight, epsilon)
# x → normalised output; residual → updated residual
```

On SM100 (Blackwell) with the Oink library, an alternative
`oink.fused_add_rms_norm` path is taken when the tensor layout is compatible.

**How to enable/disable.** Always enabled; not user-configurable.

**Supported backends / kernels.**

- CUDA (SM70+): `torch.ops._C.fused_add_rms_norm`, `csrc/layernorm_kernels.cu`
- CUDA SM100 (Blackwell): `torch.ops.oink.fused_add_rms_norm` (Oink fast path)
- AMD ROCm: equivalent HIP kernel

**Known limitations / fallbacks.**

- Alignment requirements: input, residual, and weight pointers must be 16-byte
  aligned and `hidden_size` must be a multiple of 8 for the vectorised (width-8)
  path; otherwise a scalar fallback is used.

**Code locations.**

- Layer: [`vllm/model_executor/layers/layernorm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py)
- Python op: [`vllm/_custom_ops.py`](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) (`fused_add_rms_norm`)
- CUDA kernel: [`csrc/layernorm_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu)
- Benchmark: [`benchmarks/kernels/benchmark_rmsnorm.py`](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_rmsnorm.py)

---

### Fused Mixture-of-Experts (`fused_moe`)

**What it fuses.** The `FusedMoE` custom op (`@CustomOp.register("fused_moe")`)
implements the entire MoE forward pass — token routing, batched expert GEMMs
(gate+up projections), activation (SiLU/GELU), down projection, and weighted
expert reduction — as a single fused unit.  Multiple backend implementations are
available and selected automatically based on hardware and quantisation.

**How to enable/disable.** Always active for MoE models; controlled by the
`CustomOp` system (see [`VLLM_CUSTOM_OPS`](../configuration/env_vars.md)).

**Supported backends / kernels.**

| Backend | Quant schemes | Hardware |
|---|---|---|
| Triton (default) | FP16, BF16, FP8 (dynamic / static) | NVIDIA CUDA SM70+, AMD ROCm |
| CUTLASS | FP8 (block / tensor scale) | NVIDIA SM90+ |
| FlashInfer CuteDSL | FP8, NVfp4 | NVIDIA SM90+ |
| FlashInfer TRT-LLM | FP8, NVfp4 | NVIDIA SM90+ |
| ROCm AITER | FP16, BF16, FP8 (group / token), INT8 | AMD CDNA (gfx940+) |
| GGUF | GGUF quantized types | NVIDIA CUDA |

**Known limitations / fallbacks.**

- ROCm AITER backend requires `rocm_aiter_ops` to be enabled; logs a warning
  and falls back to Triton otherwise.
- FlashInfer-based backends require FlashInfer to be installed.
- CUTLASS FP8 path requires SM90+ and the compiled CUTLASS extension.

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
| `fuse_gemm_comms` (async TP) | Yes | Yes | Yes | No | No | No |
| `enable_qk_norm_rope_fusion` | Yes | Yes | Yes | No | No | No |
| `fuse_rope_kvcache` (ROCm/AITER) | No | No | No | Yes | Yes | No |
| `fuse_act_padding` (ROCm/AITER) | No | No | No | Yes | Yes | No |

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
