# Experimental Kimi-K3 TP MegaMoE on ROCm

This opt-in path targets Kimi-K3 MXFP4 routed experts on eight gfx950 GPUs.
It uses FP8 activations, SiTUv2 (beta 4, linear beta 25), and the existing router.
It requires an AITER build with MegaMoeTP and explicit FP8 SiTUv2 fused-MoE support.
The default is `off`.

Select eager execution with
`--enforce-eager --additional-config '{"k3_megamoe_tp":"mega"}'`.
`split` selects the same A8W4 split-MoE implementation used for large prefill,
for a precision-matched eager comparison. The default MXFP4 backend may use a
different activation precision. Select
`VLLM_ROCM_USE_AITER_MOE_SITUV2=a8w4` explicitly for a native A8W4
comparison with `off`; the default `auto` selects A4W4 and synchronizes
AITER's activation environment during import.

Supported deployment: TP8, PP1, DP1, BF16 model tensors, no EP, context/sequence
parallelism, ubatching, DBO, LoRA, or specialized decode. Unsupported configurations
raise before expert execution. Shared experts execute serially on the main stream;
the persistent kernel must have exclusive GPU use.

The model owns one communication runtime shared across its MoE layers. Weight
loading converts the original separated MXFP4 checkpoint bytes once and releases
the original raw storage. MegaMoE stores standard separated packing; split mode
stores the FP8 split kernels' interleaved packing. Global token counts up to 256
use MegaMoE, padded to a multiple of eight. Larger counts temporarily unpack and
repack the same quantized bytes for split A8W4 plus TP all-reduce. This conversion
adds prefill work and transient memory, without retaining both formats per layer.
A reduced latent output is normalized without another reduction; its
up-projection shard is merged with the shared partial before the final reduction.

## Decode graphs

Graph support is restricted to MegaMoE with Model Runner V2, no Inductor,
`FULL_DECODE_ONLY`, a 4096-token maximum context, at most 32 requests and 1024
batched tokens, prefix caching disabled, auto KV dtype, and an explicit 8 GiB KV
cache budget per GPU. Set `VLLM_ROCM_AITER_MLA_ASM_PADDING=gluon`.
The actual target MLA metadata must select Gluon decode or verify; a backend or
cache-size fallback raises before capture/execution. A requested environment
setting alone does not establish the actual kernel route.

For example, add these settings to the normal Kimi-K3 TP8 launch:

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2=a8w4
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ROCM_AITER_MLA_ASM_PADDING=gluon
# Omit --enforce-eager for graph execution.
vllm serve moonshotai/Kimi-K3 \
    --tensor-parallel-size 8 --language-model-only \
    --dtype bfloat16 --max-model-len 4096 \
    --max-num-seqs 32 --max-num-batched-tokens 1024 \
    --kv-cache-memory-bytes 8589934592 --no-enable-prefix-caching \
    --disable-custom-all-reduce \
    --additional-config '{"k3_megamoe_tp":"mega"}' \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}'
```

Padded ASM and default compilation failed whole-model quality checks in the tested
environment. This limited Gluon configuration avoids that failing route; it does
not fix the ASM kernel or establish support for other graph configurations.
Split mode remains eager only. Isolated operator graph tests do not establish
whole-model graph correctness.

## DSpark

The supported speculative configuration is the K3 DSpark model on TP8, three
draft tokens, and `TRITON_MLA` draft attention. Reuse the existing K3 DSpark
implementation; the target routed experts continue to use A8W4 MegaMoE.
Other speculative methods are rejected. The context, batching, KV budget and
prefix-caching restrictions above also apply to eager DSpark.

For graph + DSpark, replace the graph capture sizes above with
`[4,8,16,32,64,128]` and add:

```bash
--speculative-config '{"method":"dspark","model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":3,"attention_backend":"TRITON_MLA"}'
```

Capture sizes count global target verification tokens, including the bonus
token: 32 requests with three draft tokens require a capacity of 128. They are
not per-rank EP batch sizes. Larger capture capacities are rejected; the temporary
TP-to-split packing path cannot run during graph capture.

## Validation scope

A matched graph + DSpark comparison used the real Kimi-K3 checkpoint on eight
MI355X GPUs. Both arms explicitly selected A8W4; `off` used the native
`Mxfp4MoEMethod` with no MegaMoE pool. Other model, graph, draft and scheduler
settings were identical to the supported recipe above.

| Graph + DSpark, A8W4 | GSM8K correct | Truncations | Median seconds |
| --- | ---: | ---: | ---: |
| Native MoE (`off`) | 1255/1319 | 0 | 1.643770 |
| MegaMoE (`mega`) | 1261/1319 | 0 | 2.385155 |

Quality used all 1319 GSM8K test examples, the first five train examples as shots,
greedy sampling, seed 1234, a 512-token output limit and 32-request batches.
MegaMoE changed accuracy by +0.455 percentage points; paired lost/gained
counts were 16/22. This is not statistical,
per-question or logits equivalence. Both arms recorded actual graph replay on
every TP rank and nonzero DSpark acceptance.

Timing used one workload only: ISL256, OSL64, eight concurrent requests, one
warmup and three trials. These are offline end-to-end generation medians, not
online latency distributions. MegaMoE took 45.1% longer
(0.689x speedup), so this configuration has a performance
regression. Three trials do not establish statistical stability. Temporary
large-prefill packing is extra work, but this run did not isolate its contribution.

The restored default-worker MegaMoE graph + DSpark API also passed real
completions, nonzero draft-acceptance deltas and token/context boundary checks:
4095 input tokens plus one output succeeded; 4097 inputs were rejected.

Earlier eager custom-split/Mega teacher-prefill checks retained unequal logprobs (maximum absolute
difference 0.3423 on 128 target tokens), and independent latent diagnostics
retained unexplained differences (maximum first-call relative L2 0.0197).
The experiment reused image-native libraries with a Python source overlay;
a fresh native build and checkpoint CI coverage remain unverified.
These results do not establish deployment readiness or other workloads.
Weight reload is unsupported; construct a new model instead.
