# SenseNova Loop with mHC

The `SenseNovaLoopForConditionalGeneration` implementation serves text from
the public [SenseNova-Loop-9B-Preview](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview)
checkpoint and compatible `sensenova_loop` checkpoints. It uses a fixed
`loop_settings.eval_count`, shares repeated layer parameters, and keeps separate
attention and recurrent caches for every execution of a layer. Optional mHC
routers retain FP32 weights and use log-space Sinkhorn normalization.

The implementation requires unquantized weights. On NVIDIA GPUs, it defaults to
full decode CUDA Graphs without Torch compilation; prefill executes eagerly.
Explicit compilation settings and `--enforce-eager` take precedence.
It supports tensor parallelism, but not pipeline parallelism, vision, audio,
LoRA, speculative decoding, learned loop gates, or layer scaling. CPU weight
offloading is unsupported with shared loop weights: nonzero `--cpu-offload-gb`
or `--offload-group-size` is rejected before workers start.
The `sharded_state` and `runai_streamer_sharded` load formats are also rejected
before workers start. Sharded-state exports deduplicate shared parameters, but
this model restores their aliases only after weight processing. Load the original
checkpoint with `auto` or `safetensors` instead of a `save_sharded_state` export.

SSM cache precision follows the checkpoint's `mamba_ssm_dtype` when
`--mamba-ssm-cache-dtype` is `auto`. The public checkpoint uses FP32 SSM state
even with BF16 weights and convolution state. An explicit
`--mamba-ssm-cache-dtype` setting takes precedence.

```bash
vllm serve sensenova/SenseNova-Loop-9B-Preview \
  --served-model-name sensenova_loop \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --host 127.0.0.1 --port 8001
```

The public checkpoint loads natively without `--trust-remote-code` or
architecture overrides. Its [reference implementation](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview/blob/main/modeling_sensenova_loop.py)
and weights are published under Apache 2.0. The checkpoint includes vision and
MTP weights; this implementation loads the text backbone and mHC routers and
ignores those additional weights. Image inputs and MTP remain unsupported.

Native loading expects the following top-level fields in `config.json`, alongside
the checkpoint's text, loop and mHC settings:

```json
{
  "model_type": "sensenova_loop",
  "architectures": ["SenseNovaLoopForConditionalGeneration"]
}
```

To override the loop count or mHC settings at startup, pass their complete
objects through `--hf-overrides`. The execution plan and mHC boundary checks
are recomputed after overrides; physical checkpoint layers remain unchanged.
Setting `loop_settings` to `null` runs each physical layer once and retains
the mHC settings. The mHC `pre_scale` must be a positive finite number.

```bash
--hf-overrides '{"loop_settings":{"start":4,"end":28,"eval_count":3}}'
```

## Execution plan and parameter sharing

The checkpoint describes physical layers. vLLM needs one logical layer for each
execution of a physical layer, including repetitions, so it can allocate separate
attention and recurrent cache state. `SenseNovaLoopConfig` expands the physical
layer indices and attention types before model initialization. It retains the
physical metadata and rebuilds the plan after `--hf-overrides`; applying an
override therefore does not expand an already expanded plan.

Each logical layer has its own module and cache identity. A model post-load hook
ties repeated layers' parameters after loading and weight processing, including
dummy initialization. Reusing the entire decoder module would also reuse its
cache identity, which would mix states from different loop iterations. Parameter
sharing reduces steady-state weight memory, but does not reduce the number of
attention or recurrent cache entries. At load time, repeated layers temporarily
allocate separate weights before tying parameters, so allow additional memory
during initialization. Vision weights are ignored for text inference.

The native implementation reuses Qwen3.5 attention, recurrent-state helpers,
normalization, MLP and weight-loading components. It supplies the loop execution
plan and mHC residual-stream operations while keeping the scheduler and model
runner unchanged. The custom config adapts checkpoint metadata to vLLM's logical
layer layout; it is not a replacement for the checkpoint's reference model.
The HF reference maps each physical-layer visit to a separate `DynamicCache`
slot. The native model expresses these execution slots through vLLM's hybrid
model interfaces, so both attention and recurrent state follow the loop plan.

## mHC numerical behavior

BF16 CUDA inference reuses vLLM's
`hc_collapse_triton` mHC stream reduction. Coefficients are rounded to BF16
before reduction, as in the checkpoint implementation. A fused Triton kernel
performs log-space Sinkhorn normalization. Residual merging preserves the
checkpoint's matrix orientation and BF16 intermediate rounding; the generic
`MHCPreOp` and `MHCPostOp` use different normalization and rounding conventions.

CUDA routing uses a fused FP32 RMS normalization and, for up to 32 tokens, a
split-K FP32 projection. Larger batches retain the cuBLAS projection. Two-stream
BF16 residual merging is fused into one kernel; wider streams retain the
original matrix multiplication to preserve its accumulation order. These paths
support CUDA Graph replay without reducing router precision.

Use `--enforce-eager` instead of the compilation configuration for an eager
baseline. Add `--cudagraph-metrics` to inspect actual graph dispatch statistics.
