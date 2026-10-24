# Experimental Kimi-K3 MegaMoE on ROCm

The TP path below targets Kimi-K3 MXFP4 routed experts on eight gfx950 GPUs.
It uses FP8 activations, SiTUv2 (beta 4, linear beta 25), and the existing router.
It requires an AITER build with MegaMoeTP and explicit FP8 SiTUv2 fused-MoE support.
For the separate EP backend, see [MegaMoEV2](#megamoev2-expert-parallelism).
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

## MegaMoEV2 expert parallelism

The separate `aiter_mega_moe` backend uses AITER MegaMoEV2 with TP8 attention,
EP8 routed experts and DP1. Each GPU owns 112 complete experts. It requires
an AITER build supporting SiTUv2 A8W4 MegaMoEV2 and MoRI SHMEM.
The TP-specific `k3_megamoe_tp` option must remain `off`.

The router selects experts on the replicated global token batch before tokens
are padded and divided across TP ranks. MegaMoEV2 dispatches each rank's token
slice to the expert owners and combines the latent results. A TP all-gather
reconstructs the global latent batch before the original RMSNorm/up projection.
The original shared MLP retains its precision and a separate TP reduction.
No shared-expert requantization or fusion is performed. EP tokens per rank
are distinct from the global token count used by MegaMoeTP.

The current AITER implementation uses its intra-node FlyDSL dispatch/combine
kernel with MoRI symmetric-memory allocations and peer pointers. It does not
call MORI-IO transfer APIs. TCP bootstrap initializes the SHMEM environment;
it does not imply TCP token transfer. On a single node, `MORI_SOCKET_IFNAME=lo`
avoids selecting an unreachable bootstrap interface. P2P data is intended to
use the node's GPU interconnect; hardware link counters have not been profiled.

For the same bounded graph and DSpark settings documented above, replace
the TP-specific backend selection with:

```bash
export MORI_SHMEM_HEAP_SIZE=8G
export MORI_SOCKET_IFNAME=lo
# Keep AITER, A8W4, MRV2, Gluon and shared-stream settings above.
# Add these arguments to the bounded graph + DSpark launch:
--enable-expert-parallel --moe-backend aiter_mega_moe \
    --all2all-backend mori_high_throughput \
    --additional-config '{"k3_megamoe_tp":"off"}'
```

Only TP8/EP8/DP1/PP1 on gfx950 with the exact K3 latent/SiTUv2 shape is
supported. The same 4K context, C32, batched-token1024, 8GiB KV, MRV2,
Gluon FULL_DECODE_ONLY and TP8 DSpark3 restrictions apply. Prefix caching,
EPLB, DBO, ubatching, sequence/context parallelism, LoRA and specialized decode
are rejected. Eight GPUs must be available exclusively to the persistent kernel.
DP8/TP1 with this checkpoint and draft does not leave enough memory for the
MoRI heap and KV cache in the tested memory budget.

### V2 validation

A real-checkpoint layer passed all 66 eager/changed-input graph checks for
global token counts 1/7/8/9/32/128 against the existing FP32 EP reference
threshold (relative L2 <=0.10); the observed maximum was 0.007812875.
This operator threshold is separate from whole-model quality.

Both full-model arms used TP8 attention, EP8 experts, DP1, explicit A8W4,
FULL_DECODE_ONLY, the same K3 DSpark draft and the bounded recipe above.
The native arm had 92 Mxfp4MoEMethod modules and 112 experts per rank; V2 had
92 MegaMoEV2 expert modules and one shared runtime per rank. Every rank
recorded actual FULL replay, and both arms recorded nonzero DSpark acceptance.

| TP8 attention / EP8 experts, graph + DSpark, A8W4 | GSM8K correct | Truncations | Median seconds |
| --- | ---: | ---: | ---: |
| Native EP (`aiter`) | 1247/1319 | 0 | 1.905887 |
| MegaMoEV2 (`aiter_mega_moe`) | 1251/1319 | 0 | 2.425798 |

Quality used all 1319 GSM8K test questions, the first five train examples as
shots, greedy sampling, seed1234, max512 output tokens and batches of32.
V2 changed accuracy by +0.303 percentage points,
with 17 questions lost and 21 gained.
The preselected maximum1pp drop gate passed; this is not statistical,
per-question or logits equivalence.

Timing used only ISL256/OSL64/concurrency8, one warmup and three trials.
These offline end-to-end medians give 0.786x native/V2 speedup:
V2 took 27.3% longer. No tuning or profiler attribution was performed,
and three trials do not establish stable or general performance.

The actual default-worker V2 graph+DSpark API additionally passed four
completions with positive accepted/drafted counter deltas, finite generated
logprobs, token-count boundaries1/7/8/9/255/256/257/1024, 4095+1 success,
4097 HTTP400 rejection and a nonempty chat smoke in ordinary runtime mode.
A separate VLLM_GPU_SYNC_CHECK=error attempt failed in unchanged base Gluon
MLA metadata at per_req_len.min().item(), before model forward. Strict sync
validation has not passed and that synchronization remains unfixed.
Full quality used the frozen EP-v2 source;
the final API used EP-v4 with typing/docstrings and stricter attention guards.
Runtime ASTs match across all four K3 numerical modules after excluding
configuration validation and typing/docstrings. The new attention rejection
probe failed on EP-v3 and passed on EP-v4. A fresh native build and
eight-GPU checkpoint CI remain unverified. Weight reload is unsupported;
construct a new model instead.

The TP results below are separate measurements and cannot be claimed for V2.

## TP validation scope

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
