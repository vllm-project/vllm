# Meta-Device Operator Capture

`vllm.profiler.op_capture` reports **which operators a model runs, in what order, with what shapes**, including vLLM's custom kernels such as `_C::fused_add_rms_norm` and `_vllm_fa2_C::varlen_fwd`. It needs no weights (only the Hugging Face config is read), no free accelerator memory, and no engine, scheduler or worker, and it executes no kernel.

Typical uses: finding out which kernels a new model needs on your hardware, feeding an operator trace into a performance simulator, and checking whether a model fell off the fast attention path.

## How it works

A tensor on `torch.device("meta")` has a shape and a dtype but no memory; operators on it only propagate shapes. So the model is built on `meta`, and the normal vLLM forward path runs over it while every dispatched operator is recorded.

`ForwardHarness` stands in for the worker and model runner, redoing what they settle before a forward pass:

1. **Build the config.** `EngineArgs(load_format="meta")` produces an ordinary `VllmConfig`.
2. **Register meta kernels** for vLLM's custom ops (see below).
3. **Settle kernel choices as a worker does**: the current vLLM config, `kernel_config.ir_op_priority` and the vLLM IR wrap flag. These decide whether a layer calls `_C::rms_norm` or a wrapper around it, so they are set before the model is built.
4. **Create a single-rank Gloo process group**, which model code expects. At world size 1 no collective reaches the backend.
5. **Build the model** through `MetaModelLoader`, under `torch.device("meta")` and without loading weights.
6. **Plan the KV cache for real.** Layers are grouped by attention backend, the layout is resolved, and the cache is sized for one request at `max_model_len`, allocated (shaped but empty) and bound to the layers.
7. **Build attention metadata** with the backend's own metadata builder for the requested `BatchSpec`.

The forward pass then runs inside `set_forward_context(...)` as in a real step. A `TorchDispatchMode` records every dispatched operator, and module hooks attribute each one to the innermost module that issued it. Recording covers the model call and `compute_logits`. Input preparation is the harness's own, and sampling is not recorded because it reads logit values that do not exist on `meta`.

## The real platform stays in charge

There is no "meta platform": `current_platform` is the actual one (CUDA, ROCm, XPU, ...), and only the tensors are fake. The platform decides most of what the trace contains:

| Platform decision | Shows up in the trace as |
| --- | --- |
| Attention backend | `_vllm_fa2_C::varlen_fwd` vs `_vllm_fa3_C::*` vs a Triton kernel vs `_flashmla_C::*` |
| KV cache layout and block size | cache tensor shapes, and which `reshape_and_cache_*` variant runs |
| Fused custom ops enabled | `_C::fused_add_rms_norm` vs the same math as `aten::` ops |
| Kernel priority and IR wrapping | `_C::rms_norm` directly vs a `vllm_ir::` wrapper |
| Quantization support | which MoE/GEMM kernel namespace appears at all |

This keeps the recorded code the production path, and is what lets a meta capture be compared operator for operator against a real-device one. The price is that you can only capture the platform you are on, so every capture carries its selection metadata, and a model whose kernels need another platform refuses to build.

## Custom ops on the meta device

PyTorch knows the output shape of every `aten::` op but nothing about vLLM's compiled kernels. `meta_ops` fills the gap in two tiers:

- **Glue ops**, registered from Python through `direct_register_custom_op` (e.g. `vllm::unified_attention_with_output`), are device-agnostic, so their bodies are reused as meta kernels. The compiled kernel inside each one still shows up.
- **Leaf ops**, the compiled extensions (`_C`, `_C_cache_ops`, `_moe_C`, `_vllm_fa2_C`, `_xpu_C`, ...), are never executed. Their meta kernel returns nothing if the schema returns nothing, or hands back the input the return aliases. Failing that it comes from the `OVERRIDES` table, and failing that the op raises `UnsupportedMetaOpError` naming itself. The capture never guesses a shape.

Registration only adds the `Meta` dispatch key and skips ops that already have one, so none of this is reachable on a real device.

**Raw Triton kernels** launched from Python (`kernel[grid](...)`) bypass the dispatcher, so no meta kernel can stand in for them, and Triton would run them on the accelerator with null pointers. During a meta capture such a launch is instead recorded as `triton::<kernel qualified name>`, with its argument shapes, and dropped before it is compiled. Triton kernels write into buffers their caller allocated, so no later shape changes. A Triton launch that reaches the device any other way raises `UnsupportedMetaOpError`.

## Running it

With a source checkout, install vLLM editable or prefix the commands with `PYTHONPATH=.` so an older installed vLLM does not shadow it.

```bash
python examples/features/profiling/capture_model_ops.py \
    --model Qwen/Qwen2.5-0.5B-Instruct
```

The output starts with the selection metadata the capture is only valid under:

```text
Selection metadata
  platform           xpu
  device             meta
  dtype              torch.bfloat16
  quantization       none
  attention backend  vllm.v1.attention.backends.flash_attn.FlashAttentionBackend
  kv cache layout    LBNHC
  kv cache dtype     auto
  block size         16 (kernel (16,))
  attention layers   24
  heads              14 query, 2 kv, head size 64
```

then the module tree, with identical sibling layers collapsed and `*` marking custom ops (extra indentation marks an op issued from inside another op):

```text
      0 [Qwen2DecoderLayer]
        input_layernorm [RMSNorm]
            aten::empty.memory_format
          * _C::rms_norm
        self_attn [Qwen2Attention]
          qkv_proj [QKVParallelLinear]
              aten::linear
          rotary_emb [RotaryEmbedding]
            * _C::rotary_embedding
          attn [Attention]
            * vllm::unified_kv_cache_update
              * _C_cache_ops::reshape_and_cache_flash
            * vllm::unified_attention_with_output
              * _vllm_fa2_C::varlen_fwd
```

and finally operator counts, total and per attention layer.

Other flags:

- `--shapes` prints operand shapes and dtypes next to every operator.
- `--num-reqs 4 --num-tokens 32 --num-computed-tokens 128` changes the batch, here to a decode-like step with 128 tokens already cached per request.
- `--trace /tmp/qwen05.et.json` also writes a Chakra execution trace through PyTorch's `ExecutionTraceObserver`.
- `--verify-against xpu` captures the same model and batch on `meta` and on the given device, normalizes both execution traces and diffs them, exiting non-zero on a mismatch. It needs enough device memory for the model.

```text
Execution trace comparison (meta vs real device)
  operators: 1045 meta, 1045 real, 1045 aligned
MATCH
```

Normalization keeps operator names, operand shapes and types, and drops run-specific ids. It also drops everything issued from inside a native or compiled operator, since meta shape inference and real kernels do different work there.

### From Python

```python
from vllm.engine.arg_utils import EngineArgs
from vllm.profiler.op_capture import BatchSpec, capture_model_ops, format_report

model = "Qwen/Qwen2.5-0.5B-Instruct"
capture = capture_model_ops(
    model,
    batch=BatchSpec(num_reqs=1, num_tokens=8),
    trace_path="/tmp/qwen05.et.json",
    engine_args=EngineArgs(model=model, max_model_len=1024),
)
print(format_report(capture))
print([op.name for op in capture.custom_ops])
```

`engine_args` reaches anything else a config can express, such as `quantization` or `kv_cache_dtype`; the harness overrides only `model`, `load_format` and `enforce_eager`. `compare_traces(meta_path, real_path)` diffs two traces already on disk, and `load_trace(path)` reads one as normalized operators.

## Limitations

- **The eager path is captured.** The harness sets `enforce_eager`, which also makes vLLM enable all custom ops. A default-config step instead runs under `torch.compile` with `custom_ops=["none"]`, so `CustomOp` layers such as `SiluAndMul` take their native forward, Inductor and the compilation passes fuse the result, and graph capture pads the batch. Ops dispatched through vLLM IR follow `kernel_config.ir_op_priority` either way. A capture therefore matches an `--enforce-eager` step; passing `compilation_config={"custom_ops": ["none"]}` in `engine_args` shows the pre-Inductor sequence instead.
- **Branches on a tensor's device see `meta`.** Code that checks `tensor.is_cuda` or `tensor.device.type` rather than `current_platform` takes its non-CUDA branch, e.g. the SM100 skinny-GEMM dispatch in `vllm/model_executor/layers/utils.py` and the Triton slot-mapping path in `vllm/v1/attention/backends/mla/indexer.py`. On CUDA a capture differs from the real step at such sites.
- **Values are undefined.** Nothing data-dependent is real, including MoE routing; only shapes and the operator sequence are.
- **Single process only.** The harness needs `tp=pp=dp=1`. A capture still matches one rank of a tensor-parallel run, minus the collectives and with per-rank shapes.
- **The batch must fit one scheduler step**, i.e. within `max_num_batched_tokens` and `max_num_seqs`, because metadata builders size their buffers by those.
- **Raw Triton kernels appear in the operator list, not the execution trace**, since they never go through the dispatcher.
- **A capture is platform-specific.** Keep the selection metadata with the operator list.
- **`load_format="meta"` is not a serving mode**; it builds unmaterialized weights for shape-only use.

## Troubleshooting

`UnsupportedMetaOpError: <op> returns (...), whose shape is not derivable from its schema`
: Add an entry for the op to `OVERRIDES` in `vllm/profiler/op_capture/meta_ops.py`, deriving the output shape from its call site in vLLM.

`UnsupportedMetaOpError: Triton kernel '<name>' reached the device without going through kernel[grid](...)`
: The kernel was launched some other way (`kernel.run(...)`, say), so it could not be dropped. Launch it with `kernel[grid](...)`, or wrap it in a custom op with a fake impl.

`AssertionError: ... requires a CUDA device` (or similar)
: The model's code is restricted to another platform; capture it there.

`ValueError: ForwardHarness runs in a single process ...` or `... exceeds the scheduler budget ...`
: Set the parallel sizes to 1, or raise `max_num_batched_tokens` / `max_num_seqs` in `engine_args`.
