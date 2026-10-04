# Meta-Device Operator Capture

`vllm.profiler.op_capture` reports **which operators a model runs, in what order, with what shapes**, including vLLM's custom kernels such as `_C::fused_add_rms_norm` and `_vllm_fa2_C::varlen_fwd`. It needs no weights (only the Hugging Face config is read), no free accelerator memory, and no engine, scheduler or worker, and it executes no kernel.

Typical uses: finding out which kernels a new model needs on your hardware, feeding an operator trace into a performance simulator, and checking whether a model fell off the fast attention path.

## How it works

A tensor on `torch.device("meta")` has a shape and a dtype but no memory; operators on it only propagate shapes. So the model is built on `meta`, and the normal vLLM forward path runs over it while every dispatched operator is recorded.

`ForwardHarness` stands in for the worker and model runner, redoing what they settle before a forward pass:

1. **Build the config.** `EngineArgs(load_format="meta")` produces an ordinary `VllmConfig`.
2. **Register meta kernels** for vLLM's custom ops (see below).
3. **Settle kernel choices as a worker does**: the current vLLM config, `kernel_config.ir_op_priority`, the vLLM IR wrap flag and, on XPU, the model runner's `torch.cuda` aliases for `torch.xpu`. The first three decide whether a layer calls `_C::rms_norm` or a wrapper around it, so they are set before the model is built.
4. **Create a Gloo process group**, which model code expects: one rank, or one per process under `capture_ranks`. Gloo's collectives have meta kernels, so they are recorded but exchange nothing.
5. **Build the model** through `MetaModelLoader`, under `torch.device("meta")` and without loading weights.
6. **Plan the KV cache for real.** Layers are grouped by attention backend, the layout is resolved, and the cache is sized for one request at `max_model_len`, allocated (shaped but empty) and bound to the layers. Encoder-only layers, which keep no cache, get their own metadata group as in the model runner.
7. **Build attention metadata** with the backend's own metadata builder for the requested `BatchSpec`, rebuilt per batch when several are captured on one model.

The forward pass then runs inside `set_forward_context(...)` as in a real step. A `TorchDispatchMode` records every dispatched operator, and module hooks attribute each one to the innermost module that issued it. Recording covers the model call and `compute_logits`, or the pooler for a pooling model. For a multimodal model it also covers what the model runner does first: `embed_multimodal` on the batch's multimodal items, if any, then `embed_input_ids` to build the embeddings the model is called with. M-RoPE models get the runner's three rows of positions. Input preparation is the harness's own, and sampling is not recorded because it reads logit values that do not exist on `meta`.

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

A model with sliding-window layers or MoE layers adds rows such as `sliding window  128 (18 layers)` and `moe experts  XPUExpertsMxFp4`.

`heads` describes the first layer that reports a head shape, not the whole model: a hybrid model's Mamba mixers and sparse attention's indexer report no head counts, or their own, and a model may implement attention in a class that reports none either -- `heads` then reads `unknown (no layer reports head counts)`. The per-layer breakdown is `attention backend`, which lists every backend in play.

`attention layers` counts every `AttentionLayerBase`, layers that own a KV cache without running attention included, so on a sparse-attention model it exceeds the model's depth and names the kinds it is made of:

```text
  attention layers   243 (91 CompressorStateCache, 61 DeepseekV4SWACache, 61 DeepseekV4XPUAttention, 30 DeepseekV4IndexerCache)
```

Operator counts are per decoder layer -- 61 of them above, not 243 -- so an operator issued once per layer reads `1.00`.

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

Every engine argument is accepted, e.g. `--max-model-len`, `--trust-remote-code` or `--kv-cache-dtype`. Other flags:

- `--shapes` prints operand shapes and dtypes next to every operator.
- `--batch 4,32,128` changes the batch to 4 requests and 32 tokens in total, with 128 tokens already cached per request; a fourth field, as in `--batch 1,2048,0,1`, also runs a multimodal model's encoder on one item. Repeat `--batch` to capture several batches on one built model; a **Batches** section then lists the operators only some of them reach. Layers pick kernels by batch, so one batch rarely reaches every path: DeepSeek-V4's sparse-attention indexer, for instance, only runs once the context outgrows its top-k (`--batch 2,16 --batch 1,16,4096 --max-model-len 8192`).
- `--output-dir DIR` also writes `ops.txt` (distinct operators), `ops.sequence.txt` (every operator in order with shapes and module, attention ops tagged `[sliding_window=N]` or `[full]`), `capture.json` (batch, selection metadata and gaps) and `report.txt`, one subdirectory per batch when there are several.
- `--hf-overrides '{"quantization_config": null}'` captures a quantized checkpoint's unquantized path, e.g. when this platform lacks its quantization kernels. The result is not what the checkpoint would run.
- `--trace /tmp/qwen05.et.json` also writes a Chakra execution trace through PyTorch's `ExecutionTraceObserver`.
- `--verify-against xpu` captures the same model and batch on `meta` and on the given device, normalizes both execution traces and diffs them, exiting non-zero on a mismatch. It needs enough device memory for the model.

```text
Execution trace comparison (meta vs real device)
  operators: 1045 meta, 1045 real, 1045 aligned
MATCH
```

Normalization keeps operator names, operand shapes and types, and drops run-specific ids. It also drops everything issued from inside a native or compiled operator, since meta shape inference and real kernels do different work there.

### Finding what blocks a model on this platform

A capture normally stops at the first problem. `--keep-going` (`keep_going=True` from Python) collects them in one run and adds a **Platform gaps** section to the report. Here DeepSeek-V4-Pro was captured with an older `vllm_xpu_kernels` than `requirements/xpu.txt` pins:

```text
Platform gaps
  Forward pass stopped in model.layers.0.ffn.experts [MoERunner], after 162 ops:
    RuntimeError: _moe_C::topk_softplus_sqrt() expected at most 10 argument(s) but received 12 argument(s). ...
    at vllm/_custom_ops.py:2356 in topk_hash_softplus_sqrt
  Placed on xpu despite the meta device (explicit `device=` in model code), 91 tensors, e.g.:
    model.layers.0.attn.compressor.ape
```

It lists five kinds of gap:

- **Where the forward pass stopped.** Model code failed outside any kernel, for example on an op the platform's extension does not define or a call with the wrong arguments. The report gives the module, the error and the vLLM source line. The ops recorded up to that point are kept.
- **Ops with no kernel for this platform.** A meta capture never runs a real kernel, so an op registered only for another backend would otherwise pass unnoticed. Every captured op is checked against the platform's dispatch key (`capture.missing_kernels`).
- **Custom ops whose kernel failed on `meta`**, typically by reading a tensor's value. The op is finished with its fake kernel's outputs, or placeholders when it has none, so the forward pass carries on. The kernels it would have dispatched after the error are missing (`op.body_error`).
- **Ops with unknown output shapes.** These get placeholder outputs shaped like their first tensor argument and are marked `placeholder`. Shapes after the first placeholder are guesses, so a shape error that follows one may be a knock-on effect, and the report says so. Add the op to `OVERRIDES` and rerun.
- **Tensors model code placed on the real device**, by passing `device=` explicitly instead of following the default device (`capture.materialized`).

A model that fails to build, for example because its implementation needs another platform, still raises, since nothing is recorded before that point.

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

`capture_batches(model, [BatchSpec(...), ...])` captures several batches on one built model and returns one `OpCapture` per batch; `format_batches(captures)` renders the operators only some of them reach, and `write_capture_files(capture, directory)` writes the files `--output-dir` does. `capture_ranks(model, batches, engine_args=...)` does the same on every tensor-parallel rank, one process each, returning one list per rank.

`engine_args` reaches anything else a config can express, such as `quantization` or `kv_cache_dtype`; the harness overrides only `model`, `load_format` and `enforce_eager`. `compare_traces(meta_path, real_path)` diffs two traces already on disk, and `load_trace(path)` reads one as normalized operators.

## Limitations

- **The eager path is captured.** The harness sets `enforce_eager`, which also makes vLLM enable all custom ops. A default-config step instead runs under `torch.compile` with `custom_ops=["none"]`, so `CustomOp` layers such as `SiluAndMul` take their native forward, Inductor and the compilation passes fuse the result, and graph capture pads the batch. Ops dispatched through vLLM IR follow `kernel_config.ir_op_priority` either way. A capture therefore matches an `--enforce-eager` step; passing `compilation_config={"custom_ops": ["none"]}` in `engine_args` shows the pre-Inductor sequence instead.
- **Branches on a tensor's device see `meta`.** Code that checks `tensor.is_cuda` or `tensor.device.type` rather than `current_platform` takes its non-CUDA branch, e.g. the SM100 skinny-GEMM dispatch in `vllm/model_executor/layers/utils.py` and the Triton slot-mapping path in `vllm/v1/attention/backends/mla/indexer.py`. On CUDA a capture differs from the real step at such sites.
- **Kernels called outside the dispatcher are not covered**, apart from raw Triton launches. Libraries such as FlashInfer, DeepGEMM, CuTe DSL and aiter's Python API take tensors directly unless vLLM wraps the call in a custom op with a fake impl. Those that go through DLPack refuse meta tensors (`BufferError: Cannot pack tensors on meta`), which `--keep-going` reports as where the forward pass stopped. An extension that reads `data_ptr()` instead gets a null pointer, so on a real accelerator it may launch a kernel on it.
- **Values are undefined.** Nothing data-dependent is real, including MoE routing; only shapes and the operator sequence are. Model code that reads a value on the host, e.g. `.item()`, fails on `meta`. With `--keep-going`, a custom op whose kernel does so is finished with fake outputs; outside one, the forward pass stops there.
- **One kind of multimodal item.** `BatchSpec(num_mm_items=N)` encodes `N` items of the modality with the most tokens per item, at maximum size, from the processor's dummy inputs, as the model runner's profiling run does; their embeddings fill the batch's first tokens. Mixed modalities and smaller items are not generated. Models that take multimodal inputs raw in `forward` are not supported, and encoder-decoder models, such as Whisper, are refused.
- **Tensor parallelism only.** `capture_ranks` runs one process per tensor-parallel rank, so collectives appear with per-rank shapes; on `meta` they rendezvous over gloo and exchange nothing. Pipeline, context and data parallelism are refused.
- **The batch must fit one scheduler step**, i.e. within `max_num_batched_tokens` and `max_num_seqs`, because metadata builders size their buffers by those, and within `max_model_len`.
- **Raw Triton kernels appear in the operator list, not the execution trace**, since they never go through the dispatcher.
- **A capture is platform-specific.** Keep the selection metadata with the operator list.
- **`load_format="meta"` is not a serving mode**; it builds unmaterialized weights for shape-only use.

## Troubleshooting

`UnsupportedMetaOpError: <op> returns (...), whose shape is not derivable from its schema`
: Add an entry for the op to `OVERRIDES` in `vllm/profiler/op_capture/meta_ops.py`, deriving the output shape from its call site in vLLM.

`UnsupportedMetaOpError: Triton kernel '<name>' reached the device without going through kernel[grid](...)`
: The kernel was launched some other way (`kernel.run(...)`, say), so it could not be dropped. Launch it with `kernel[grid](...)`, or wrap it in a custom op with a fake impl.

`BufferError: Cannot pack tensors on meta`
: A library kernel was called directly with meta tensors. Wrap the call in a custom op with a fake impl, as vLLM does for its other library kernels.

`AssertionError: ... requires a CUDA device` (or similar)
: The model's code is restricted to another platform; capture it there.

`ValueError: ForwardHarness captures tensor-parallel ranks only ...`, `... needs one harness per rank ...` or `... exceeds the scheduler budget ...`
: Set the pipeline, context and data parallel sizes to 1, use `capture_ranks` for tensor parallelism, or raise `max_num_batched_tokens` / `max_num_seqs` in `engine_args`.

`AssertionError: Tensors left the meta device: [...]`
: Model code created these parameters or buffers with an explicit `device=`, so they were allocated on the accelerator. Construct them on the default device instead. `--keep-going` reports them without failing.
