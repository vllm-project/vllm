# MiniMax-M3 decode through the ATOM kernel library

This experimental path calls ATOM's fused sparse-layer kernels from native
vLLM. It is disabled by default. vLLM retains scheduling, cache allocation,
EAGLE3, the model loop, and CUDA graph ownership. ATOM supplies kernels, without
loading its engine, runner, or platform plugin.

## Library interface and ownership

The paired ATOM change exposes `AtomM3Mono(layer_specs, cache_specs, tp_context)`
with `prepare_step`, `forward_layer`, and `close`. vLLM supplies native tensors,
cache views, the existing TP group, and each batch's metadata. ATOM owns weight
conversion, metadata expansion, compilation, scratch, IPC, graph operations,
and kernel launch details. Its native runner shares the same sparse execution
implementation. The model keeps dispatch, auxiliary-state capture, and its
library lifetime. It uses the ordinary vLLM worker and model runner.

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
uses their AITER port. This path imports ATOM directly and retains the native
BF16 attention weights for fallback, using separate converted copies for mono.

## Supported experiment

- gfx950 with 256 CUs, TP4/PP1/DP1, V2 GPU model runner.
- MiniMax-M3-MXFP4 with 57 sparse layers (indices 3 through 59).
- EAGLE3 with three speculative tokens; the draft remains native.
- FP8 main and index caches, block size 128, maximum model length 16384,
  at most four sequences, prefix caching disabled.
- Shared-expert fusion enabled; no context/expert parallelism, microbatching,
  LoRA, KV transfer, sleep, or online weight updates. Configured weight transfer
  is rejected during model construction.
- Explicit `num_gpu_blocks_override` is required to reserve room for the lazy
  library allocation; the recorded command uses 1024 blocks. The graph-memory
  profiling warmup also includes that allocation in its measured footprint.
- Decode/verification buckets 1, 4, 8, and 16. Prefill, mixed batches, dense
  layers, and unsupported step shapes use native execution.

Initialization errors propagate collectively. A kernel execution failure is
an error, rather than a successful native retry. Graph teardown precedes peer
memory release. KV and index allocation remains entirely native; the adapter
uses static scalar K/V scales and independent index addressing, without a
shadow cache or dynamic scale sidecar.

Weights must remain fixed after loading. Reset, raw/kernel-format reload, and
direct tensor writes can bypass model methods and are unsupported; this model
integration does not claim to intercept them before mutation. A failed reload
or external weight mutation requires destroying the execution state and
reconstructing the model, including its graphs and converted weights.

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
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ROCM_USE_ATOM_M3_MONO=1

vllm serve "$TARGET_MODEL" \
    --served-model-name minimax-m3 --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --language-model-only --max-model-len 16384 --max-num-seqs 4 \
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.70 \
    --num-gpu-blocks-override 1024 --block-size 128 --kv-cache-dtype fp8 \
    --attention-config '{"indexer_kv_dtype":"fp8"}' \
    --no-enable-prefix-caching \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":3,\"rejection_sample_method\":\"standard\"}"
```

For the native baseline set `VLLM_ROCM_USE_ATOM_M3_MONO=0` and keep the remaining
settings unchanged. Accuracy requests must use the checkpoint's chat template
with `thinking_mode=enabled`; the recorded evaluation rendered it client-side.
Never use synthetic acceptance for accuracy evaluation.

## Validation

The refactor is validated separately from the earlier October 2 experiment.
The current checks cover kernel compilation for both cache modes and every
supported width, the public tensor interface on saved native layers, graph
replay with changed cache addresses, poisoned padding, collective startup
failures, and shutdown. Model tests cover unbound caches, cache generations,
explicit detach, refusal of live main-cache replacement, and capture-time refusal
of first initialization. They are selected by the AMD basic-model CI job.
The paired Draft PRs record commands, exact results, and remaining evaluation
limits for the current revisions.

The historical full GSM8K chat-template experiment scored 1277/1319 native and
1276/1319 mono. The earlier raw-completion protocol scored 1142 versus 1108.
Those results and the historical synthetic-acceptance throughput measurements
are evidence for the old implementation only; they do not qualify this refactor.
FP8 projection copies, the BF16 router, and MoE arithmetic differ from native,
so layer outputs are not expected to be bitwise identical to the native model.
