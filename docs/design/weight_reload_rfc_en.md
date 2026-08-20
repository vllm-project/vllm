# vLLM Weight Reload Refactoring RFC (Review Draft)

Chinese version: [weight_reload_rfc_zh.md](weight_reload_rfc_zh.md)

## 1. Motivation

A checkpoint tensor and its serving tensor may differ in dtype, shape, stride, layout, and even parameter count:

```text
checkpoint -> name mapping -> TP/EP sharding -> fusion/padding
           -> transpose/layout conversion -> quantization/repack
           -> PWAL/derived state -> serving tensor
```

A reload therefore cannot copy every received tensor directly into the current runtime parameter by name. Runtime parameters may already be fused, packed for Marlin, transformed by PWAL, or accompanied by MLA-derived tensors. At the same time, captured CUDA Graphs require stable parameter objects and storage addresses.

## 2. Refactoring principles and lifecycle

This design does not use layerwise reload and does not restore the entire model to a meta/checkpoint schema for rematerialization. Reload always runs after requests have drained: `drain -> update -> resume`. Hot-swapping weights while requests are executing is out of scope. `START` and explicit `FINISH` define an update; layer boundaries, transport chunks, and `numel` do not.

The design has four invariants:

1. `data_ptr()` remains stable and CUDA Graphs are not recaptured. Writes use `copy_`; shape-changing paths use `resize_ + copy_`.
2. Reload always calls the model's native `model.load_weights()` and reuses AutoWeightsLoader, model-specific name routing, TP/EP sharding, and stacked/expert mappings.
3. Reload does not use layerwise meta restoration, rematerialization, kernel tensors, or a generic online-process loader.
4. A dtype mismatch without an explicit online quantization loader fails closed. An incomplete manifest also fails closed.

```text
START
  drain requests
  call restore_weights_before_loading for shape-changing quant methods
  resize parameters to checkpoint shapes and reset conversion guards
DATA*
  receive checkpoint-format tensors
  call model.load_weights([(name, tensor)])
  let param.weight_loader perform copy/transpose/repack/online quantization
FINISH
  run selective PWAL for restored shape-changing layers
  run refresh_derived_state on applicable modules
  run manifest_check(expected, loaded)
  resume requests
```

The design introduces two extension points. Extension A is the permanent per-parameter `weight_loader`, which performs eager parameter processing. Extension B is per-module `refresh_derived_state()`, which rebuilds derived values at finish. The transport layer supplies checkpoint-format tensors and does not implement model transformations.

## 3. Scope overview: two reload paths

Weight reload is divided into two connected but clearly bounded scopes. **V1** covers cases where the sender already provides the dtype and checkpoint format required by serving. Its primary concern is to reuse the native loader, PWAL, and derived-state logic while keeping runtime storage stable. **V2** covers cases where trainer and serving dtypes differ. It keeps the same reload lifecycle but adds online quantization loaders so conversion happens while weights are received, rather than after an entire layer or model has been staged.

In short, V1 answers how to safely update format-matched weights, while V2 answers how to update weights that require online conversion. Pre-quantized checkpoints remain in V1. BF16 trainer to FP8/FP4/INT8 serving belongs to V2 and requires an explicit online quantization loader. The two scopes are detailed below.

## 4. V1 scope: matching dtype and checkpoint format

V1 assumes the sender already provides the dtype and checkpoint format required by the serving configuration. It reuses existing loaders, PWAL, and derived-state logic, but does not implicitly convert BF16 weights into a low-precision serving format.

| Scenario | Scope | Start | Receive (`weight_loader`) | Finish |
|---|---|---|---|---|
| BF16 dense/MoE with a BF16 trainer | V1 | None | `copy_` | None |
| Block FP8 with an FP8 trainer | V1 | None | `copy_` | None |
| Per-tensor FP8 dense with an FP8 trainer | V1 | None | transpose + `copy_` | None |
| Per-tensor FP8 MoE with an FP8 trainer | V1 + PWAL | Restore checkpoint shape | Copy checkpoint-format values | Scale requantization, kernel shuffle, and `_g1_alphas` refresh |
| MLA models such as DeepSeek V2/V3/R1 and Kimi K3 | V1 + derived state | None | Copy `kv_b_proj` | Refresh `W_UK_T/W_UV`; MXFP4/FP8 backends use their corresponding re-quantization path |
| Pre-quantized INT4/Marlin checkpoint | V1 + PWAL | Restore checkpoint shape | Copy GPTQ/AWQ-format values | Marlin repack |
| Block FP8 serving with a BF16 trainer | V2 | None | Online loader performs per-block quantization + `copy_` | None |
| Per-tensor FP8 serving with a BF16 trainer | V2 | None | Online loader plus fused-parameter unit buffer | Finalize only unified-scale units not completed eagerly |
| NVFP4 serving with a BF16 trainer | V2 | Backend dependent | Online loader plus the smallest required closure buffer | Layout shuffle or derived-state refresh when required |
| BF16 trainer with low-precision serving but no online loader | Reject | None | Raise `ValueError` | None |

The main V1 prerequisite is replacing shape-changing `replace_parameter` calls in PWAL with `resize_ + copy_`. Quantization methods record original shapes during `create_weights`, implement `restore_weights_before_loading()`, and use an idempotent conversion guard. Non-shape-changing PWAL does not need to run again. Derived-only work moves into `refresh_derived_state()`.

## 5. V2 scope: online quantization

V2 preserves the V1 lifecycle and permanently attaches online quantization to parameter loaders during `create_weights`. When `model.load_weights()` receives a BF16 or FP16 checkpoint tensor, the loader converts it into the serving format and writes it into stable runtime storage. Cold loading and reload therefore share the same conversion implementation.

| Input -> serving | V2 path | Buffer requirement |
|---|---|---|
| BF16 -> block FP8 | Per-block cast + `copy_` | No layer-level buffer; each shard is independent |
| BF16 -> per-tensor FP8 | Retain the shards sharing a scale, then quantize with the unified scale | Fused parameters require a unit buffer |
| BF16 -> NVFP4 | Collect the smallest required closure, such as w1+w3, then quantize and convert layout | One buffer per incomplete unit |
| BF16 -> MXFP8/INT8 | Quantization-method-specific online loader | Determined by scale granularity |

A serving quantization configuration without a matching online loader is rejected. Pre-quantized checkpoints such as GPTQ and compressed-tensors remain V1 checkpoint-format/PWAL paths.

## 6. Shared V1/V2 responsibilities

* **Start:** drain requests and restore only shape-changing layers, not the whole model schema.
* **Manifest:** declare tensor names, checkpoint dtypes and shapes, logical shards, quantization format/version, and expected coverage.
* **Loader:** perform model-specific mapping and sharding; a V2 loader additionally performs online quantization.
* **Selective PWAL:** process only restored layers requiring shape or layout conversion. A transport chunk must never trigger whole-layer PWAL prematurely.
* **Derived state:** refresh MLA and similar values after all dependencies arrive.
* **Finish:** complete selective PWAL and derived-state refresh before manifest validation. Serving does not resume after any failure.
* **Storage:** perform conversions in the original parameter storage, preserving Python identity, `data_ptr`, and CUDA Graph validity.

## 7. Smallest quantizable unit

Automatic detection operates on a `QuantizationUnit`, not on an arbitrary parameter or network chunk. A unit is the smallest input set satisfying all of these conditions:

1. All required logical shards have arrived.
2. The complete scale domain is available. A per-block domain is local, while a per-tensor domain may span shards.
3. Quantization, transpose/layout conversion, and fusion depend only on this set.
4. The result and its scale/metadata can be written into the target serving slice as one operation.

The quantization method and parameter loader jointly discover units. The loader derives logical keys from expert ids, TP shards, stacked parameters, and scale parameters. A tracker determines completion with key-set coverage, never tensor counts or `numel`.

In the reference implementation, a quantization method exposes `reload_units(layer)` to declare unit keys, staged parameters, and a finalize callback. A shard-aware wrapper around the normal loader maps global expert ids to local expert ids and reports `(parameter, local expert, shard id)` keys. Quantization semantics remain owned by the quantization method, while actual shard arrivals remain owned by the model loader. The framework does not hard-code Q/K/V, w1/w2/w3, or individual quantization formats.

## 8. Eager processing and buffer lifetime

```text
receive one shard
  -> validate manifest/key/checksum
  -> does this shard contain a complete quantization domain?
       yes: quantize + copy_, then release the received tensor
       no: let the quant tracker retain the shard and update coverage
           -> is the unit complete?
                no: wait for the remaining shards
                yes: quantize/requantize + convert layout
                     write the serving slice and scale
                     release retained shards after the CUDA event completes
```

Only units that require multiple shards need staging. The relevant quantization tracker creates and owns this storage lazily; the protocol is not tied to `reload_arena` or any global allocator. Per-block, per-group, per-channel, and non-fused per-tensor paths consume the current shard directly. Fused per-tensor paths retain only shards that are still waiting for their peers.

The per-expert reference implementation lazily creates one checkpoint-format slab for each staged `(parameter, local expert)` slot. The quantization method explicitly declares its shape and dtype because they cannot be inferred from the runtime parameter after PWAL. For example, a checkpoint scale can have shape `[E, 2]` while its runtime form has shape `[E]`. The tracker exposes the slab through an expert-dimension broadcast proxy so the original loader can continue to perform TP narrowing, expert indexing, and fusion. Finalizing the unit removes both slab and proxy. The allocator remains an implementation detail.

Asynchronous quantization must protect input lifetime with a CUDA event. Retained inputs are released as soon as the kernel no longer reads them. Peak additional memory is proportional to unmatched shards, not the checkpoint size of the complete layer or model.

## 9. Unit granularity matrix

| Granularity/parameter | Smallest processing unit | Cross-shard buffer | Processing point |
|---|---|---|---|
| Per-block: Block FP8, MXFP4, MXFP8 | One logical shard, processing its internal blocks independently | No | Quantize when the shard arrives |
| Per-group: GPTQ/AWQ group | A logical shard containing complete groups | No | Convert or quantize when the shard arrives |
| Per-channel: INT8/SmoothQuant | A logical shard containing the complete row/column scale domain | No | Quantize when the shard arrives |
| Per-tensor FP8, non-fused parameter | Complete local tensor shard | No | Compute the local tensor scale and quantize on arrival |
| Per-tensor FP8 fused QKV | Q/K/V shards sharing one serving scale | Yes | Quantize after all paired shards arrive |
| Per-tensor FP8 fused MoE `w13` | One expert's w1, w3, and corresponding scales | Yes | Requantize with a unified scale after both halves arrive |
| MoE w2 | One expert's w2 and scale | No | Process the serving slice on arrival |
| Layer-wide activation scale | All expert/layer contributions covered by the scale | Yes | After the dependency set completes, usually at `FINISH` |
| MLA-derived `W_UK_T/W_UV` | One MLA layer with all base tensors updated | Not a transport buffer | `refresh_derived_state()` |

If a quantization backend has scale or layout dependencies across an entire layer, it must not be forced into per-expert units merely to reduce memory. It remains deferred until `FINISH`.

## 10. Sharded transport

Because one transfer operation carries one sharding, each transport record needs a complete logical identity rather than only a parameter name:

```text
Record {
  update_id, sequence_no,
  tensor_name, logical_unit_id,
  tp/ep/expert/shard coordinates,
  byte_range, shape, stride, dtype,
  quant_format/version, checksum
}
```

The sender may pack records into buckets constrained by `max_chunk_bytes` and memory budgets. A bucket may cross parameters, experts, and layers. The receiver routes every record to its logical unit and never treats a bucket as a completion boundary. Out-of-order and duplicate records are deduplicated. At `FINISH`, missing shards are reported by unit and key.

The design adopts the memory-reduction objective discussed in SGLang issue 32335 but integrates it with vLLM's native reload path. A chunk is a communication and scheduling object; a quantization unit is the processing object:

```text
transport bucket -> model.load_weights(record)
                 -> unit tracker
                 -> unit complete: quantize + write + release retained inputs
                 -> FINISH: verify all units + deferred PWAL/derived work
```

The receiver applies backpressure when bytes retained by quantization trackers reach the configured budget. It pauses sending or reduces bucket size and prioritizes shards that complete nearly covered units. NCCL, CUDA IPC, and filesystem transfer share the same record schema. `(update_id, sequence_no)` identifies retransmissions.

## 11. Consistency and failure semantics

Requests have drained before reload begins, so eager writes cannot be observed by an executing forward pass. Immediate quantization and input release are memory optimizations inside the update window, not hot-swap or per-unit publication. Requests resume only after `FINISH` completes coverage validation, selective PWAL, derived-state refresh, and manifest validation.

An in-place update cannot cheaply roll back units already written. If `FINISH` detects a missing shard, conversion error, or OOM, the worker must not resume serving. The control plane must complete the same update or rebuild/reload the worker. Continuing to serve the old model after a failed update would require a complete shadow copy or double buffer and is outside this RFC's V1/V2 scope.

## 12. Verification requirements

Every enabled quantization backend must test out-of-order shards, units spanning buckets, duplicates and retransmission, missing keys, dtype mismatch, release of unit buffers, cold-load versus reload tensor/output equivalence, stable parameter identity, and stable `data_ptr`. A row in the V1/V2 matrix is considered supported only after its corresponding tests pass.
