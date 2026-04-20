# Fusion torch.compile passes

vLLM applies a set of kernel/operator fusions at compile time (via custom [`torch.compile`](torch_compile.md) Inductor passes)
to separate optimizations from model definitions and avoid breaking layer abstractions in model code.
These fusions are controlled by fields in [`PassConfig`][vllm.config.compilation.PassConfig] and are automatically enabled
at appropriate [optimization levels](optimization_levels.md).

## Quick Reference

The table below maps each fusion to its controlling flag/config knob, the
operations it fuses, what level enables it by default, and an indicative speedup.
The Fullgraph column indicates whether the fusion requires the entire model graph to be
visible (either via Inductor partition or `splitting_ops=[]`),
and the last column indicates whether the fusion activates for all `num_tokens`
or just on the low or high end.

!!! info
    Speedup depends heavily on the exact model, batch size, and hardware.
    If tuning performance by hand, always benchmark your exact use-case with and without the fusion to verify the impact.

| Fusion                                                                         | `PassConfig` flag            | Fused operations                               | Default at                     | E2E Speedup        | Fullgraph | `num_tokens` |
| ------------------------------------------------------------------------------ | ---------------------------- | ---------------------------------------------- | ------------------------------ | ------------------ | --------- | ------------ |
| [AllReduce + RMSNorm](#allreduce--rmsnorm-fuse_allreduce_rms)                  | `fuse_allreduce_rms`         | All-reduce → RMSNorm (+residual_add) (→ quant) | O2 (Hopper/Blackwell + TP > 1) | 5-20%              | No        | Low          |
| [MiniMax QK Norm](#minimax-qk-norm-fuse_minimax_qk_norm)                       | `fuse_minimax_qk_norm`       | Q/K variance all-reduce → Q/K RMSNorm          | Off by default                 | 2-3%               | No        | Low          |
| [Attention + Quant](#attention--quantization-fuse_attn_quant)                  | `fuse_attn_quant`            | Attention output → FP8/NVFP4 quant             | Off by default                 | 3-7%               | Yes       | Always       |
| [MLA Attention + Quant](#attention--quantization-fuse_attn_quant)              | `fuse_attn_quant`            | MLA Attention output → FP8/NVFP4 quant         | Off by default                 | TBD                | Yes       | Always       |
| [RoPE + KV-Cache Update](#rope--kv-cache-update-fuse_rope_kvcache)             | `fuse_rope_kvcache`          | Rotary embedding → KV cache write              | O2 (ROCm/AITER only)           | 2-4%               | No        | Low          |
| [QK Norm + RoPE](#qk-norm--rope-enable_qk_norm_rope_fusion)                    | `enable_qk_norm_rope_fusion` | Q/K RMSNorm → rotary embedding                 | Off by default                 | 2-3%               | No        | Low          |
| [Sequence Parallelism](#sequence-parallelism-enable_sp)                        | `enable_sp`                  | AllReduce → ReduceScatter + AllGather          | Off by default                 | Prereq for AsyncTP | Yes       | High         |
| [AsyncTP GEMM + collective](#asynctp-gemm--collective-overlap-fuse_gemm_comms) | `fuse_gemm_comms`            | GEMM → reduce-scatter / all-gather → GEMM      | Off by default                 | 7-10%              | Yes       | High         |
| [RMSNorm + Quant](#rmsnorm--quantization-fuse_norm_quant)                      | `fuse_norm_quant`            | RMSNorm (+residual add) → FP8/FP4 quant        | O1 (conditional)               | 1-4%               | No        | Always       |
| [SiLU+Mul + Quant](#silumul--quantization-fuse_act_quant)                      | `fuse_act_quant`             | SiLU+Mul activation → FP8/FP4 quant            | O1 (conditional)               | 1-4%               | No        | Always       |
| [RMSNorm + Padding](#rmsnorm--padding-fuse_act_padding)                        | `fuse_act_padding`           | Residual add + RMSNorm → padding               | O1 (ROCm/AITER only)           | TBD                | No        | Always       |

## Support Matrix

The table below lists the quantization schemes supported by each fusion on each platform.
**—** means the fusion is not available on that platform. The latest and in-progress work is available in the tracking issue:
[#36066](https://github.com/vllm-project/vllm/issues/36066)

| Fusion                       | SM100 (Blackwell)                        | SM90 (Hopper)                            | SM89 (Ada)                               | SM80 (Ampere) | ROCm                                     |
| ---------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ------------- | ---------------------------------------- |
| `fuse_allreduce_rms`         | FP16/BF16, FP8 static, NVFP4             | FP16/BF16, FP8 static                    | —                                        | —             | —                                        |
| `fuse_minimax_qk_norm`\*     | FP16/BF16                                | FP16/BF16                                | FP16/BF16                                | FP16/BF16     | —                                        |
| `fuse_attn_quant`\*          | FP8 static\*, NVFP4\*                    | FP8 static\*                             | FP8 static\*                             | —             | FP8 static\*                             |
| `fuse_attn_quant` (MLA)\*    | FP8 static\*, NVFP4\*                    | FP8 static\*                             | FP8 static\*                             | —             | FP8 static(untested)\*                   |
| `fuse_rope_kvcache`          | —                                        | —                                        | —                                        | —             | FP16/BF16                                |
| `enable_qk_norm_rope_fusion` | FP16/BF16                                | FP16/BF16                                | FP16/BF16†                               | FP16/BF16†    | —                                        |
| `enable_sp`                  | FP16/BF16, FP8 static†                   | FP16/BF16, FP8 static                    | FP16/BF16†                               | FP16/BF16†    | —                                        |
| `fuse_gemm_comms`            | FP16/BF16, FP8 static†                   | FP16/BF16, FP8 static                    | FP16/BF16†                               | FP16/BF16†    | —                                        |
| `fuse_norm_quant`            | FP8 static, FP8 per-token, FP8 per-group | FP8 static, FP8 per-token, FP8 per-group | FP8 static, FP8 per-token, FP8 per-group | —             | FP8 static, FP8 per-token, FP8 per-group |
| `fuse_act_quant`             | FP8 static, NVFP4                        | FP8 static, FP8 per-group (128/64)       | FP8 static, FP8 per-group (128/64)       | —             | FP8 per-group                            |
| `fuse_act_padding`           | —                                        | —                                        | —                                        | —             | FP16/BF16                                |

\* `fuse_attn_quant` support depends on the attention backend in use; not all backends support
fused quantization output. See the [`fuse_attn_quant` section](#attention--quantization-fuse_attn_quant)
for per-backend details.

\* `fuse_minimax_qk_norm` is a model-specific pass for `MiniMaxM2ForCausalLM`. It also requires
tensor parallelism (`tp_size > 1`) and the CUDA custom op `minimax_allreduce_rms_qk`.

† `enable_sp` and `fuse_gemm_comms` are only autoconfigured for SM90 today;
other architectures support requires setting `PassConfig.sp_min_token_num` explicitly.
SM100 support also requires setting `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`.

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

## Fusion Details

### AllReduce + RMSNorm (`fuse_allreduce_rms`)

!!! warning
    TP+DP and TP+PP combinations are currently broken
    ([#34458](https://github.com/vllm-project/vllm/issues/34458) and
    [#35426](https://github.com/vllm-project/vllm/issues/35426)).
    Only supported on NVIDIA Hopper (SM90) and Blackwell (SM100) with FlashInfer installed.

**What it fuses.** Fuses the tensor-parallel all-reduce collective with the subsequent residual add,
RMSNorm, and optionally a quantization step into a single FlashInfer / TRT-LLM communication kernel.
This fusion is only profitable for small `num_tokens`,
so the fusion is only performed in the lower compiled range.

Patterns covered:

- `AllReduce → RMSNorm(+residual_add)`: CUDA sm90+ with FlashInfer
- `AllReduce → RMSNorm(+residual_add) → FP8 static quant`: CUDA sm90+ with FlashInfer
- `AllReduce → RMSNorm(+residual_add) → NVFP4 dynamic quant`: CUDA sm100+ with FlashInfer

The maximum tensor size below which the fused kernel is used is hardware-dependent (64 MB for TP=2
on SM90/SM100) and configurable via `PassConfig.fi_allreduce_fusion_max_size_mb`.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/allreduce_rms_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/allreduce_rms_fusion.py)
- FlashInfer all-reduce: [`vllm/distributed/device_communicators/flashinfer_all_reduce.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/flashinfer_all_reduce.py)
- Benchmark: [`benchmarks/kernels/benchmark_fused_collective.py`](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_fused_collective.py)

### Attention + Quantization (`fuse_attn_quant`)

!!! info
    `fuse_attn_quant` is currently not enabled at any optimization level by default and must be set
    explicitly. It requires the full model graph to be visible (Inductor partition or `splitting_ops=[]`).

**What it fuses.** Fuses the attention output quantization directly after the attention computation,
eliminating a full-precision memory round-trip of the attention output. This fusion supports both
standard `Attention` and `MLAAttention` (used by DeepSeek-V2/V3/R1 models). Patterns covered:

`Attention → FP8 static quant`:

- `TRITON_ATTN`: CUDA, ROCm
- `FLASHINFER`: CUDA sm100+ with FlashInfer installed
- `ROCM_ATTN`: ROCm
- `ROCM_AITER_UNIFIED_ATTN`: ROCm with AITER

`Attention → NVFP4 dynamic quant`:

- `FLASHINFER`: CUDA sm100+ with FlashInfer installed

`MLAAttention → FP8 static quant` / `MLAAttention → NVFP4 dynamic quant`:

The MLA fusion operates at the graph level on the `unified_mla_attention_with_output` op and works
with all MLA decode and prefill backend combinations. Unlike standard `Attention` backends (where
the kernel writes FP8 output directly), no MLA prefill or decode backend currently supports direct
FP8/FP4 output. The fusion writes to an intermediate buffer and quantizes in a separate step, so
there is no memory round-trip elimination yet.

!!! info
    The MLA attention fusion is not expected to yield a measurable speedup yet.
    This will improve once MLA prefill/decode kernels support direct FP8/FP4 output.

Other attention backends do not support fused output quantization yet.

**Code locations.**

- Pass (Attention): [`vllm/compilation/passes/fusion/attn_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/attn_quant_fusion.py)
- Pass (MLAAttention): [`vllm/compilation/passes/fusion/mla_attn_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/mla_attn_quant_fusion.py)
- Attention backends: [`vllm/v1/attention/backends/`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/)

### RoPE + KV-Cache Update (`fuse_rope_kvcache`)

!!! info
    ROCm/AITER-only. Not available on NVIDIA CUDA or CPU. The fusion is only enabled for
    `num_tokens ≤ 256` by default due to AITER fused kernel performance issues.
    This threshold is configurable via `PassConfig.rope_kvcache_fusion_max_token_num`.

**What it fuses.** Fuses the rotary positional embedding kernel with the KV-cache scatter/write into
a single kernel, avoiding separate reads and writes of the key and value tensors.

Requires: AMD ROCm with AITER enabled, the `rotary_embedding` custom op active (automatic),
and the `kv_cache` update op visible in the graph: either by using Inductor graph partition
or removed from `splitting_ops`.
If these conditions are set, the fusion is enabled automatically for optimization level O1 and above.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/rope_kvcache_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rope_kvcache_fusion.py)

### MiniMax QK Norm (`fuse_minimax_qk_norm`)

!!! info
    This is a MiniMax-specific compile pass. It is currently only enabled when all of the following hold:
    the model architecture is `MiniMaxM2ForCausalLM`, tensor parallelism is enabled (`tp_size > 1`),
    and the CUDA custom op `minimax_allreduce_rms_qk` is available. It is not enabled by default at any
    optimization level.

**What it fuses.** Fuses the MiniMax M2 Q/K normalization path that performs an all-reduce over the
per-token Q/K variances before applying RMS normalization to Q and K.

This pass is distinct from [`enable_qk_norm_rope_fusion`](#qk-norm--rope-enable_qk_norm_rope_fusion):
`fuse_minimax_qk_norm` targets MiniMax M2's tensor-parallel all-reduce + RMSNorm sequence, while
`enable_qk_norm_rope_fusion` targets the later Q/K RMSNorm + RoPE sequence used by several other models.

Example:

```bash
vllm serve MiniMaxAI/MiniMax-M2.5 \
  --tensor-parallel-size 4 \
  --compilation-config '{"mode": 3, "pass_config": {"fuse_minimax_qk_norm": true}}'
```

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py)
- CUDA op: [`csrc/minimax_reduce_rms_kernel.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/minimax_reduce_rms_kernel.cu) (`minimax_allreduce_rms_qk`)
- Workspace helper: [`vllm/model_executor/layers/mamba/lamport_workspace.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/lamport_workspace.py)

### Sequence Parallelism (`enable_sp`)

**What it fuses.** Replaces all-reduce collectives with reduce-scatter + local RMSNorm + all-gather,
splitting the sequence dimension across TP ranks. This restructures the graph so the subsequent AsyncTP
pass can fuse the reduce-scatter / all-gather with the surrounding GEMMs.

Sequence Parallelism itself does not directly improve performance; it is a prerequisite for the
AsyncTP pass (`fuse_gemm_comms`). SP is only applied above a minimum token threshold that is
autoconfigured based on device capability and model `hidden_size`. Currently only active on
H100/SM90 for models with `hidden_size >= 8192`. The threshold is configurable via
`PassConfig.sp_min_token_num`.

The general transformation:

```text
Input → AllReduce → RMSNorm → Output
becomes:
Input → ReduceScatter → local RMSNorm → AllGather → Output
```

Patterns covered:

- First block: `AllReduce → RMSNorm` → `ReduceScatter → RMSNorm → AllGather`
- Middle blocks: `AllReduce → fused_add_RMSNorm` → `ReduceScatter → fused_add_RMSNorm → AllGather`
- Both with optional `→ FP8 static quant` suffix

Requires: `use_inductor_graph_partition=True` **or** piecewise compilation with static sizes
divisible by `tensor_parallel_size`.

Supported hardware: Only tested on NVIDIA CUDA, possibly works on ROCm. FP8 all-gather requires sm90+.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/sequence_parallelism.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/sequence_parallelism.py)

### AsyncTP GEMM + Collective Overlap (`fuse_gemm_comms`)

!!! info
    Requires `enable_sp=True` (enabled automatically). This pass is a no-op if Sequence Parallelism has not been applied.

**What it fuses.** After Sequence Parallelism transforms the graph, fuses GEMM kernels with the
surrounding reduce-scatter (output projection) and all-gather (input projection) using
`torch.ops.symm_mem` symmetric-memory primitives, overlapping communication and computation.
This overlap is only profitable for large `num_tokens`, so the fusion (and preceding SP)
is only performed in the higher compiled range above `PassConfig.sp_min_token_num`.

Patterns covered:

- `GEMM → reduce-scatter` → `fused_matmul_reduce_scatter`
- `all-gather → GEMM` → `all_gather_matmul`
- FP8 scaled variants of both patterns

Supported hardware: NVIDIA CUDA with symmetric-memory (`torch.distributed._symmetric_memory`) support.

On B200, pattern-matching fp8 FlashInfer scaled MM is not supported, so it must be disabled
([#27893](https://github.com/vllm-project/vllm/issues/27893))

```shell
VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel ...
```

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/collective_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/collective_fusion.py)
- Sequence parallelism pass: [`vllm/compilation/passes/fusion/sequence_parallelism.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/sequence_parallelism.py)

### QK Norm + RoPE (`enable_qk_norm_rope_fusion`)

!!! info
    Only applicable to models that apply per-head RMSNorm to Q and K before rotary positional
    embedding (e.g. Qwen). Not enabled by default at any optimization level due to perf issues on H100:
    [#34391](https://github.com/vllm-project/vllm/issues/34391)

**What it fuses.** Fuses the sequence: split QKV → reshape → Q/K RMSNorm → reshape → rotary
embedding into a single `fused_qk_norm_rope` CUDA kernel.

```text
# Unfused:
q, k, v = split(qkv)
q_norm = rms_norm(q.view(heads))
k_norm = rms_norm(k.view(kv_heads))
q_rope, k_rope = rotary_embedding(q_norm, k_norm, ...)

# Fused:
fused_qk_norm_rope(qkv, ...)
```

Supported hardware: CUDA (sm80+) only, tested only on sm90 and sm100.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/qk_norm_rope_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py)
- CUDA kernel: [`csrc/ops.h`](https://github.com/vllm-project/vllm/blob/main/csrc/ops.h) (`fused_qk_norm_rope`)

### RMSNorm + Quantization (`fuse_norm_quant`)

!!! warning
    On NVIDIA, Inductor actually generates a faster fused kernel than our custom CUDA kernel.
    Hence, this fusion is only enabled when either `rms_norm` or `quant_fp8` is using a custom kernel.

**What it fuses.** Combines the custom `rms_norm` / `fused_add_rms_norm`
operations with subsequent quantization into a single fused kernel,
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

### SiLU+Mul + Quantization (`fuse_act_quant`)

!!! warning
    Same as `fuse_norm_quant`: on NVIDIA, Inductor generates a faster fused kernel than our custom ops.
    This fusion is only enabled when either `silu_and_mul` or `quant_fp8` are using a custom kernel,
    or for NVFP4-quantized models (where FP4 quant is always a custom op).

**What it fuses.** Fuses the `silu_and_mul` gate-up projection activation with subsequent quantization into a single kernel,
avoiding materialization of the full-precision post-activation tensor.

Note that AITER fusions are in a separate pass in `vllm.compilation.passes.fusion.rocm_aiter_fusion`.

Supported quantization scheme/hardware combinations:

- FP8 static per-tensor: CUDA & HIP kernel
- FP8 dynamic per-group (128/64): CUDA kernel (sm89+, not active when DeepGemm is used on sm100+)
- NVFP4 dynamic: CUDA sm100+ only with FlashInfer
- FP8 per-token-group (128): ROCm AITER only

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/act_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/act_quant_fusion.py)
- ROCm AITER pass: [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)
- CUDA/HIP kernels: [`csrc/quantization/`](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/)
- Fused SiLU+Mul+BlockQuant kernel: [`csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu)

### RMSNorm + Padding (`fuse_act_padding`)

!!! info
    ROCm/AITER-only. Targeted at GPT-OSS models.

**What it fuses.** Fuses a residual add + RMSNorm with a subsequent padding operation that pads
the hidden dimension to a multiple required by downstream AITER Triton GEMM kernels.

Requires: AMD ROCm with AITER RMSNorm enabled. Enabled by default in optimization level O1 and above
when the hidden size is 2880 and AITER Triton GEMMs *not* enabled.

**Code locations.**

- Pass: [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py) (`RocmAiterTritonAddRMSNormPadFusionPass`)

## See Also

- [Optimization Levels](optimization_levels.md) — high-level presets that set
  fusion defaults.
- [torch.compile in vLLM](torch_compile.md) — how the Inductor pass pipeline
  works.
- [Attention Backends](attention_backends.md) — attention-specific kernel
  selection.
