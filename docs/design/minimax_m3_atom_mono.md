# MiniMax-M3 decode through the ATOM kernel library

This experimental path calls ATOM's fused sparse-layer kernels from native
vLLM. It is disabled by default. vLLM retains scheduling, cache allocation,
EAGLE3, the model loop, and CUDA graph ownership. ATOM supplies kernels, without
loading its engine, runner, or platform plugin.

## Library interface and ownership

The paired ATOM change exposes `AtomM3Mono(layer_specs, cache_specs, tp_context)`
with `prepare_step`, `forward_layer`, and `close`. vLLM supplies native tensors,
cache views, the existing TP group, and each batch's metadata. ATOM owns
metadata expansion, compilation, scratch, IPC, graph operations,
and kernel launch details. Its native runner shares the same sparse execution
implementation. The model keeps dispatch, auxiliary-state capture, and its
library lifetime. It uses the ordinary vLLM worker and model runner.

Weights are shared with native execution. QKV/O must already use an AITER
preshuffled per-channel E4M3 backend, with FP32 scales; supported Hipb transposes
are reversed through views. Router weights must be BF16 with FP32 logits, and
experts must use the supported shuffled MXFP4 backend. ATOM does not quantize,
cast, shuffle, or copy model weights. If any sparse layer or TP rank is
incompatible, the model logs the reason once and uses native execution.

The first eager forward with bound main/index caches initializes the library,
including graph warmup. Initial memory profiling without caches stays native.
Temporary graph-profiling caches get their own library instance; the normal
cache detach closes it before the final caches are bound. The model attention's
main-cache setter handles this explicit detach after graph teardown, clears
native K/V views, and rejects replacement of a live main cache. All TP ranks
must detach together after destroying their graphs. No GC callback performs
collectives, and no engine hook or worker selection is added.

The port uses ATOM's shared mono runtime and retains its store-publication,
per-step mailbox reset/fence, and padded-row expert-reduction fixes. It requires
the paired ATOM library revision; the earlier low-level cache ABI is insufficient.
The current interface supports eager execution and CUDA graphs. Direct
`torch.compile` of the mono library is rejected because it cannot safely follow
functionalization's replacement storage.

This is an alternative ownership boundary to vLLM
[59705](https://github.com/vllm-project/vllm/pull/59705), which vendors the
kernels, and [59653](https://github.com/vllm-project/vllm/pull/59653), which
uses their AITER port. This path imports ATOM directly and shares the loaded
weights between native execution and mono.

## Supported experiment

- gfx950 with 256 CUs, TP4/PP1/DP1, V2 GPU model runner.
- Reserve the GPUs for this runtime. The persistent kernel's forward-progress
  contract requires 256 CTAs to reside together; concurrent external GPU work
  is not qualified by this integration.
- MiniMax-M3-MXFP4 with 57 sparse layers (indices 3 through 59).
- EAGLE3 with three speculative tokens; the draft remains native.
- FP8 main and index caches, block size 128, maximum model length 16384,
  prefix caching disabled. Each mono batch must contain at most four padded
  request rows; larger batches run natively, including in CUDA graphs.
- Shared-expert fusion enabled; no context/expert parallelism, microbatching,
  LoRA, KV transfer, sleep, or online weight updates. Unsupported configurations
  disable mono and retain native execution.
- Explicit `num_gpu_blocks_override` is required to reserve room for the lazy
  library allocation; the recorded command uses 1024 blocks. The graph-memory
  profiling warmup also includes that allocation in its measured footprint.
- Decode/verification buckets 1, 4, 8, and 16. Prefill, mixed batches, dense
  layers, and unsupported step shapes use native execution.

Unsupported weights/configurations fall back collectively before allocation.
Initialization errors propagate collectively. A kernel execution failure is
an error, rather than a successful native retry. Graph teardown precedes peer
memory release. KV and index allocation remains entirely native; the adapter
uses static scalar K/V scales and independent index addressing, without a
shadow cache or dynamic scale sidecar.

Weights must remain fixed after loading. Reset, raw/kernel-format reload, and
direct tensor writes can bypass model methods and are unsupported; this model
integration does not claim to intercept them before mutation. A failed reload
or external weight mutation requires destroying the execution state and
reconstructing the model, including its graphs and runtime.

Direct main/index cache rebinding, tensor-storage mutation, and partial cache
reinitialization outside the normal lifecycle are also unsupported. The main
cache setter is a lifecycle hook, not a comprehensive mutation guard: index
cache writes can bypass it. Such changes require destroying and rebuilding the
execution state; existing graphs must not be reused.

## Launch the pinned experiment

Install the paired patched ATOM revision in the same environment first. Set
`TARGET_MODEL` and `DRAFT_MODEL` to the local MiniMax-M3-MXFP4 and
MiniMax-M3-EAGLE3-GQA checkpoints. The recorded draft revision is
`Inferact/MiniMax-M3-EAGLE3-GQA@96692486b5fd38ebf8fd2a5f6bb53427d30819a8`.

```bash
export VLLM_PLUGINS=''
export ATOM_DISABLE_VLLM_PLUGIN=1
export AITER_LOG_LEVEL=WARNING
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCR_VISIBLE_DEVICES=0,1,2,3
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_LINEAR_HIPBMM=0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ROCM_USE_ATOM_M3_MONO=1

vllm serve "$TARGET_MODEL" \
    --served-model-name minimax-m3 --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --language-model-only --max-model-len 16384 --max-num-seqs 8 \
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.70 \
    --num-gpu-blocks-override 1024 --block-size 128 --kv-cache-dtype fp8 \
    --attention-config '{"indexer_kv_dtype":"fp8"}' \
    --hf-overrides '{"text_config":{"router_dtype":"bfloat16"}}' \
    --quantization-config '{"targets":{"re:.*\\.layers\\.(?:[3-9]|[1-5][0-9])\\.self_attn\\.(?:qkv_proj|o_proj)$":"fp8_per_channel"}}' \
    --no-enable-prefix-caching \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":3,\"rejection_sample_method\":\"standard\"}"
```

For the native baseline set `VLLM_ROCM_USE_ATOM_M3_MONO=0` and keep the remaining
settings unchanged. Accuracy requests must use the checkpoint's chat template
with `thinking_mode=enabled`; the recorded evaluation rendered it client-side.
Never use synthetic acceptance for accuracy evaluation.

`router_dtype` defaults to `float32`; enabling mono never changes it. Explicit
`bfloat16` loads the checkpoint router into the model's only BF16 router weight
tensor, while keeping logits and correction bias in FP32. This is a
lossy model configuration and needs its own accuracy evaluation. The ATOM
library keeps FP32 logits before sigmoid; its native runner retains its existing
BF16-logit behavior through a separate kernel build option.

The example checkpoint has a nested `text_config`; a text-only checkpoint uses
`--hf-overrides '{"router_dtype":"bfloat16"}'` instead. The native preshuffled
backend also requires AITER tuning entries for the QKV/O shapes. Hipb storage is
supported by the adapter, but requires a hipBLASLt installation that supplies
rowwise FP8 algorithms for both decode and prefill shapes.

## Validation

The shared-weight interface is validated separately from earlier converted-weight
experiments. Fourteen model tests cover borrowed projection storage, incompatible
weights, large/small batch transitions, pipeline-partition fallback, TP cache
readiness, cache generations, explicit detach, and capture-time refusal of first
initialization. They are selected by the AMD basic-model CI job. The paired ATOM
runtime/layout/dispatch/library/build-key suites pass 182 checks, including the
router precision option's distinct binary key.

A fixed-activation CPU probe of layers 3 and 20 compared FP32 router weights
against their BF16-rounded values, with FP32 matrix multiplication in both arms.
Two of 112 token rows changed their top-4 expert set. This isolates weight
rounding; it is not an end-to-end accuracy or GPU-kernel equivalence result.
The paired Draft PRs record current GPU results and remaining evaluation limits.
Earlier saved-layer replay, poisoned-padding, and shutdown results used private
converted weights and do not qualify the shared-weight revision.

The current shared-weight version completed ordinary-worker startup, CUDA graph
capture and a 64-question GSM8K pilot: FP32 router/native fallback scored 62/64,
BF16 router/native scored 63/64, and mono with the same BF16/FP8 weights scored
62/64. None had API errors, truncations or parse failures; these small samples do
not establish accuracy parity. A 12-request GPU trace exercised
8-to-4-to-8-to-4 request transitions while long requests retained their caches.
Profiler summary export later stalled, so that diagnostic service was stopped.
A subsequent restart, with other GPU jobs present and a lower memory reservation,
stopped progressing during final warmup; native BF16 execution under the same
conditions completed its pilot. The mono restart cause remains unresolved.
Long-input smoke and normal mono shutdown need a dedicated-GPU rerun.

The historical full GSM8K chat-template experiment scored 1277/1319 native and
1276/1319 mono. The earlier raw-completion protocol scored 1142 versus 1108.
Those results and the historical synthetic-acceptance throughput measurements
are evidence for the old implementation only; they do not qualify this refactor.
Compare native FP32-router and BF16-router configurations to measure the router
choice, then compare native and mono using identical FP8/BF16 weights to measure
kernel differences. Historical converted-weight evaluations do not qualify the
shared-weight interface. Arithmetic order can still differ between kernels, so
shared weights alone do not imply bitwise-identical layer outputs.
