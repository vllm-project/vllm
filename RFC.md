# Motivation

[Helion](https://github.com/pytorch/helion) is a Python-embedded DSL that compiles PyTorch-like kernel code to lower-level DSLs, such as Triton.  As demonstrated in the PyTorch blog post [Portable vLLM Model Inference Kernels in Helion](https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/), kernels under `vllm/kernels/helion/` are competitive with—and often faster than—their CUDA counterparts in vLLM, delivering end-to-end gains on Qwen3 FP8 models running on H100 and B200 GPUs. We have also observed end-to-end gains on AMD MI350X GPUs.

This RFC proposes an integration model for **using Helion kernels in vLLM CustomOps when Cudagraph is enabled.** The initial rollout will cover the following operations:

1. per_token_group_fp8_quant
2. silu_and_mul_per_block_quant
3. rms_norm_per_block_quant

The earlier kernel-authoring RFC, [[RFC]: Add Helion integration in vLLM (#32219)](https://github.com/vllm-project/vllm/issues/32219), remains the design reference for implementing and registering kernels. This RFC proposes the following additional integration changes:

1. **Make Helion a pinned vLLM dependency.**
2. **Represent each eligible choice as an opaque, CUDA-graph-aware routed custom op.**
3. **Call the custom op from respective CustomOp’s forward_cuda.**
4. **For kernels used in compiled path, swap in using an inductor pass**

**Comparison with Helion Integration in SGLang.** Helion is integrated into [SGLang as an optional backend for Kimi Delta Attention](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/ops/attention/helion). Like the integration proposed here, it uses pre-tuned Helion configs and requires no runtime autotuning. There two main differences:

1. In SGLang, users explicitly opt into the Helion backend, whereas here we propose using Helion by default for selected CustomOps.
2. SGLang provides separate kernel implementations and fixed configurations for decode and prefill workloads. Our integration uses a single kernel implementation for each CustomOp and selects among pre-tuned configurations based on the input shape range.

In SGLang, for Kimi-Linear-48B-A3B-Instruct (TP=2, 2xGB200), we observe geomean 5.04%-5.27% improvement in tok/s compared to default kernels.

# Proposed Change

The proposed integration is prototyped in <https://github.com/vllm-project/vllm/pull/48995>. If this RFC is accepted, we will submit the changes as a series of polished PRs.

- [existing] `@register_kernel` exposes a kernel as `torch.ops.vllm_helion.<kernel_name>`.
- [existing] `ConfigManager` loads checked-in Helion configurations from `configs/<kernel>/<platform>.json`.
- [existing] Each kernel's deterministic `pick_config()` chooses one pre-tuned config. There is no runtime autotuning.
- **[new]** Helion kernels being used will JIT-compile a selected variant during CUDA-graph capture.
- **[new]** Gate CustomOp routing to Helion kernels with `VLLM_USE_HELION_KERNELS`, defaulting to `1`. Setting it to `0` falls back to the torch.ops._C alternative cuda kernel.

The runtime behavior is:

| Condition | Behavior |
| --- | --- |
| Supported op during CUDA-graph capture | Helion |
| VLLM_USE_HELION_KERNELS Feature disabled | Same as current |
| No CUDA graph | Same as current; compiled routing pass is not installed |
| Unsupported platform (no Helion config file for it) | Same as current |
| Op executed outside CUDA-graph capture | Same as current |

### Routed custom op

vLLM defines `torch.ops.vllm_helion.routed_<kernel_name>`. Its schema and mutation contract match the corresponding Helion op. Its implementation is conceptually:

```python
def routed_impl(*args):
    if torch.cuda.is_current_stream_capturing():
        return helion_op(*args)  # torch.ops.vllm_helion.<name>
    return native_op(*args)  # torch.ops._C.<name>
```

During CUDA-graph capture the Helion launch is recorded; replay executes the recorded kernel without re-entering the Python dispatcher. Outside capture the same routed op calls the native implementation.

Registration is idempotent. If either the helion op or the native op is unavailable, for example because the native build does not contain the CUDA op or the platform has no Helion configs, the pair is omitted and the graph retains the native target.

### Post-grad Inductor Pass

Some fused ops, such as `silu_and_mul_per_block_quant` and `rms_norm_per_block_quant` are inserted into torch.compile’d graph using inductor passes.

We add an additional post-grad FX pass, `HelionFusionRoutingPass`. It retargets direct calls to these native ops and remaining `auto_functionalized` wrappers whose inner target is a native op. Running it last therefore preserves vLLM's existing fusion, lowering, functionalization, and copy-elimination behavior.

The pass is installed only when `VLLM_USE_HELION_KERNELS=1`and `cudagraph_mode != NONE`. Without CUDA graphs, the pass is not created and the graph is unchanged.

### Direct Call

`per_token_group_fp8_quant` has a direct eager call site in `QuantFP8.forward_cuda`. Conceptually, we write:

```python
if (
    VLLM_USE_HELION_KERNELS
    and not torch.compiler.is_compiling()
    and x.is_contiguous()
    and torch.cuda.is_current_stream_capturing()
):
    return helion_per_token_group_fp8_quant(...)
return native_per_token_group_fp8_quant(...)
```

The `not torch.compiler.is_compiling()` condition short-circuits before the capture check while Dynamo is tracing. The compiled graph therefore still contains the native quant op, allowing RMSNorm/quant fusion to match normally; the post-grad routing pass handles any supported op that remains afterward. Only genuinely eager execution being captured into a CUDA graph takes the direct Helion path.


### AOT compile-cache correctness

On a `torch.compile` AOT cache miss, `HelionFusionRoutingPass` defines the routed custom ops before producing the cached graph. On a cache hit, the post-grad pass pipeline is skipped, but the loaded graph may still reference those op names.

The AOT artifact load path therefore calls `register_compiled_routed_helion_ops()` under the same feature and CUDA-graph gates (in vllm/compilation/decorators.py). This recreates the definitions before the cached graph is used and makes cache-hit and cache-miss behavior equivalent.

The pass UUID includes its routed-op set so that routing changes participate in compile-cache invalidation.

### Perf Verification

We have verified the following for the three kernels.

1. Its Helion and native schemas have compatible argument order, defaults, and mutation semantics.
2. The target GPU (sm90, sm100) has checked-in configs and a deterministic config picker.
3. Both in-config and representative out-of-config shapes matches eager numerically
4. Both in-config and representative out-of-config shapes out-performs current alternative baseline in vLLM

### Add Helion as a required dependency

Enabling Helion routing by default requires Helion to be installed with supported CUDA distributions, along with packaging changes such as <https://github.com/vllm-project/vllm/pull/50631>.

### Number of Configs used by Helion Kernel

The vLLM repository currently contains many shape-specific configurations for Helion kernels. Multi-shape autotuning allows us to use one default configuration per kernel on sm90 and at most five configurations per kernel on sm100 while preserving most kernel-level and end-to-end performance gains (see results below). These consolidated configurations substantially reduce maintenance overhead.

### Extending support to AMD

Although this RFC initially targets NVIDIA sm90 and sm100, the same integration can support AMD GPUs by adding platform-specific tuned configurations. On ROCm, `torch.cuda.is_current_stream_capturing()` reports the HIP stream-capture state, allowing the routed custom op to work without platform-specific logic.

The AMD results below demonstrate the resulting performance gains.

## Performance

When measuring the performance numbers, we used the single checked-in default configuration for each Helion kernel on H100 and up to five checked-in configurations per kernel on B200.

### Standalone Kernel Performance with Extensive Shape Sweep on H100 and B200

A sweep uses physical dimensions from public model configurations and TP-local dimensions for TP sizes 1, 2, 4, and 8. The matrix includes RNJ1, Gemma 3 (270M through 27B), dense and MoE Qwen3 variants, MiniMax-M2, DeepSeek-V3, and Kimi-K2. Every physical shape is evaluated over the 51 default vLLM CUDA-graph capture sizes through 512 tokens.

H100 artifact:  https://gist.github.com/yushangdi/592fc18a79459198504737f019811b4b
B200 artifact:

H100:

| Kernel | Cases | Geomean | Faster | Within 2% | >5% slower | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `per_token_group_fp8_quant` | 1,173 | 1.481x | 1,173 | 1,173 | 0 | 1.028x |
| `rms_norm_per_block_quant` | 255 | 1.785x | 255 | 255 | 0 | 1.309x |
| `silu_and_mul_per_block_quant` | 918 | 1.382x | 873 | 904 | 2 | 0.941x |
| **Total** | **2,346** | — | **2,301** | **2,332** | **2** | — |


B200:

| Kernel | Shapes | Geomean | Faster | Within 2% | >5% slower | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `per_token_group_fp8_quant` | 1,173 | 1.779x | 1,173 | 0 | 0 | 1.272x |
| `rms_norm_per_block_quant` | 255 | 1.750x | 255 | 0 | 0 | 1.021x |
| `silu_and_mul_per_block_quant` | 918 | 1.617x | 911 | 20 | 0 | 0.976x |
| **Overall** | **2,346** | — | **2,339** | **20** | **0** | — |

### End-to-End Model Performance on Qwen/Qwen3-8B-FP8

#### H100

We use the following configs
```text
VLLM_USE_DEEP_GEMM=1
VLLM_USE_DEEP_GEMM_E8M0=1
--linear-backend deep_gemm
```

| Concurrency | Native tok/s | Single-config tok/s | Single vs. native |
| ---: | ---: | ---: | ---: |
| 8 | 1,273.46 | 1,317.89 | +3.489% |
| 16 | 2,352.54 | 2,432.04 | +3.379% |
| 32 | 4,008.54 | 4,129.85 | +3.026% |
| 64 | 6,292.87 | 6,455.52 | +2.585% |
| **Aggregate** | — | — | **+3.119%** |

#### B200

This benchmark uses:

```text
USE_DEEP_GEMM=0
USE_DEEP_GEMM_E8M0=0
```

| Concurrency | Native median tok/s | Helion median tok/s | Paired median gain |
| ---: | ---: | ---: | ---: |
| 8 | 1,798.5 | 1,938.9 | +7.81% |
| 16 | 3,142.2 | 3,270.3 | +4.08% |
| 32 | 5,599.6 | 5,949.0 | +6.30% |
| 64 | 9,012.5 | 9,422.7 | +4.58% |

### ROCm Performance

For ROCm, we also use up-to-five configs for each Helion kernel.

#### Kernel Performance on MI350X

The following results compare Helion with the corresponding AITER operations.

| Kernel | Geomean speedup | Token mapping |
| --- | ---: | ---: |
| `per_token_group_fp8_quant` | **1.094x** | **1; 2–8; 16–128** |
| `rms_norm_per_block_quant` | **1.100x** | **1; 2–128** |
| `silu_and_mul_per_block_quant` | **1.223x** | **1–128** |

#### End-to-End Performance on MI350X

These results use up to three configurations per kernel.

| Max concurrency | Prompts | AITER output tok/s | Helion + AITER GEMM output tok/s | Throughput change | AITER TPOT (ms) | Helion TPOT (ms) | TPOT change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 64 | 1,365.83 | 1,453.45 | +6.41% | 5.504 | 5.191 | -5.70% |
| 16 | 128 | 2,613.53 | 2,798.44 | +7.08% | 5.784 | 5.414 | -6.40% |
| 32 | 256 | 4,396.18 | 4,622.99 | +5.16% | 6.900 | 6.509 | -5.65% |
| 64 | 256 | 8,488.90 | 8,937.57 | +5.29% | 7.429 | 7.051 | -5.08% |

Geometric mean: **+5.98% output throughput** and **-5.71% mean TPOT**.
