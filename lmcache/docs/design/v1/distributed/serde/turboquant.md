# `lmcache.v1.distributed.serde.turboquant` — TurboQuant Serde Backend

## Scope

This document describes the TurboQuant serde backend for LMCache L2 adapters.
TurboQuant serde is a storage-layer transform. It compresses LMCache KV tensors
before they are written to L2 storage and reconstructs KV tensors after L2
prefetch. It does not implement an attention backend and does not change the
StorageManager or L2 adapter public APIs.

The serde is intended to be used through the generic serde framework described
in [`docs/design/v1/distributed/serde/README.md`](README.md) and the L2 adapter
wrapper described in
[`docs/design/v1/distributed/l2_adapters/serde_wrapper.md`](../l2_adapters/serde_wrapper.md).

## Motivation

KV cache tensors can be large, especially for long-context workloads and
multi-layer models. Storing raw fp16/bf16 KV tensors in L2 increases storage
capacity requirements and L2 transfer volume.

TurboQuant serde reduces the serialized KV size by applying low-bit KV
compression before L2 store. During L2 load / prefetch, the compressed bytes are
decompressed back into the original LMCache KV tensor layout.

## Data Path

### Store Path

```text
MemoryObj(KV tensor)
  -> TurboQuantSerializer.serialize(src, dst, key)
  -> TurboQuant store Triton kernel
  -> MemoryObj(uint8 compressed bytes)
  -> inner L2 adapter store
```

### Load / Prefetch Path

```text
inner L2 adapter load
  -> MemoryObj(uint8 compressed bytes)
  -> TurboQuantDeserializer.deserialize(src, dst, key)
  -> TurboQuant decode Triton kernel
  -> MemoryObj(restored KV tensor)
```

The caller only observes the normal LMCache L2 store / load behavior. Temporary
byte buffers, serde task scheduling, eventfd signaling, and lock lifecycle are
handled by SerdeL2AdapterWrapper and AsyncSerdeProcessor.

## Public Interfaces

TurboQuant serde provides the synchronous serde interfaces required by the
generic serde framework:

* `TurboQuantSerializer.serialize(src, dst, key) -> int`
* `TurboQuantSerializer.estimate_serialized_size(layout_desc) -> int`
* `TurboQuantDeserializer.deserialize(src, dst, key) -> None`

It is registered under the serde type name:

```json
{
  "type": "turboquant"
}
```

The factory accepts TurboQuant-specific kwargs and constructs an
AsyncSerdeProcessor wrapping the serializer and deserializer.

## Configuration

`TurboQuantSerdeConfig` controls the compression preset and layout parameters.

Supported presets:

| Preset | Key path | Value path | Norm correction |
| --- | --- | --- | --- |
| `turboquant_k8v4` | FP8 key | 4-bit value quantization | No |
| `turboquant_4bit_nc` | 4-bit MSE key | 4-bit value quantization | Yes |
| `turboquant_k3v4_nc` | 3-bit MSE key | 4-bit value quantization | Yes |
| `turboquant_3bit_nc` | 3-bit MSE key | 3-bit value quantization | Yes |

Other config fields:

* `head_dim`: per-head hidden dimension.
* `block_size`: token block size used by the compressed layout.
* `skip_first_layers`: number of leading layers stored in the original raw KV
  format instead of the TurboQuant compressed format. The default is `2`.
* `skip_last_layers`: number of trailing layers stored in the original raw KV
  format instead of the TurboQuant compressed format. The default is `2`.

The skipped layer settings implement boundary-layer protection. By default,
TurboQuant serde stores the first two and last two layers without quantization
and only compresses the middle layers. This matches the default vLLM
TurboQuant behavior, where the first and last two attention layers are skipped
from TurboQuant KV-cache compression because they are more sensitive to
quantization error.

For a model with `num_layers` layers, the compressed range is:

```text
quant_start = min(skip_first_layers, num_layers)
quant_end = max(quant_start, num_layers - skip_last_layers)
```

Layers `[0, quant_start)` and `[quant_end, num_layers)` are serialized as raw
KV bytes. Layers `[quant_start, quant_end)` are serialized with the TurboQuant
packed layout.

Invalid presets are rejected with `ValueError`.

## Tensor Layout

TurboQuant serde expects LMCache KV tensors in this layout:

```text
[2, num_layers, num_tokens, hidden_dim]
```

The first dimension separates key and value tensors:

* `src[0]`: key cache
* `src[1]`: value cache

The serialized byte layout is:

```text
raw first layers | compressed middle layers | raw last layers
```

where:

* raw layer groups use the original `[2, num_layers, num_tokens, hidden_dim]`
  KV byte layout.

* compressed middle layers use
  `[num_compressed_layers, num_blocks, block_size, num_heads, slot_size]`.

* `num_blocks = ceil(num_tokens / block_size)`
* `num_heads = hidden_dim / head_dim`
* `slot_size = key_packed_size + value_packed_size`
* `slot_size_aligned` is used for the serialized byte layout

`estimate_serialized_size()` computes the number of bytes required for this
compressed layout from `MemoryLayoutDesc`.

## Compression Path

`TurboQuantSerializer` launches Triton store kernels to compress each layer.

The store path performs:

1. KV layout validation.
2. Temporary CUDA staging when StorageManager provides CPU / pinned-memory
   `MemoryObj` tensors.
3. Key compression:
   * FP8 key path for `turboquant_k8v4`.
   * MSE / centroid low-bit key path for low-bit presets.
4. Value uniform quantization.
5. 3-bit or 4-bit bit-packing into a uint8 byte buffer.
6. Metadata storage, including scale / zero for values and norm metadata when
   required by the preset.

The compressed output is written into the destination uint8 `MemoryObj`.

## Decompression Path

`TurboQuantDeserializer` launches Triton decode kernels to reconstruct KV
tensors from compressed bytes.

The load path performs:

1. Serialized byte buffer validation.
2. Temporary CUDA staging when StorageManager provides CPU / pinned-memory
   `MemoryObj` tensors.
3. Key unpacking and dequantization:
   * FP8 key decode for FP8 presets.
   * MSE / centroid decode for low-bit presets.
4. Value unpacking and scale / zero dequantization.
5. Restoration of the LMCache KV tensor layout:
   `[2, num_layers, num_tokens, hidden_dim]`.

The deserializer reconstructs KV tensors for storage reuse. It does not compute
attention outputs.

## Device Handling

TurboQuant uses Triton kernels, so tensors participating in one kernel launch
must be on the same CUDA device.

Device selection follows these rules:

1. If any source or destination tensor is already on CUDA, all CUDA tensors in
   the same serde operation must be on the same device. Otherwise, TurboQuant
   serde raises `ValueError`.
2. If `cuda_device` is configured, that device is used as the staging device.
   If CUDA tensors already exist, the configured device must match them.
3. If all source and destination tensors are CPU tensors and `cuda_device` is
   not configured, TurboQuant serde selects a CUDA device with sufficient free
   memory and the lowest GPU utilization.
4. CPU / pinned-memory tensors are staged to the selected CUDA working device
   before Triton kernel execution and copied back afterward.

This backend does not change LMCache runtime placement policy; the automatic
selection only applies to CPU-only serde staging.

## Relationship to vLLM TurboQuant

The TurboQuant store/decode Triton kernel logic follows the implementation
approach used by the vLLM TurboQuant PR. The integration target is different:

* vLLM integrates TurboQuant into the attention backend.
* LMCache integrates TurboQuant into the L2 serde path.

In LMCache, TurboQuant is a storage transform only: it compresses objects before
L2 store and reconstructs objects after L2 load / prefetch.

## Performance Snapshot

The following numbers are from a local H20 serde microbenchmark. They are
intended as a sanity check for compression ratio, latency, and reconstruction
error rather than full serving benchmark results.

### Serde microbenchmark

| Serde | Preset | Raw MB | Serialized MB | Compression ratio | Encode ms | Decode ms | Corr | Mean abs err | Max abs err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp8 | `float8_e4m3fn` | 8.00 | 4.00 | 2.00 | 0.024 | 0.040 | 0.999645 | 0.017948 | 0.250000 |
| turboquant | `turboquant_k8v4` | 8.00 | 3.06 | 2.61 | 0.375 | 0.523 | 0.997342 | 0.051538 | 0.273438 |
| turboquant | `turboquant_4bit_nc` | 8.00 | 2.09 | 3.82 | 0.554 | 0.642 | 0.995225 | 0.080693 | 0.505249 |
| turboquant | `turboquant_k3v4_nc` | 8.00 | 1.84 | 4.34 | 0.555 | 0.642 | 0.989075 | 0.115782 | 0.970703 |
| turboquant | `turboquant_3bit_nc` | 8.00 | 1.59 | 5.02 | 0.557 | 0.643 | 0.980405 | 0.164546 | 0.970703 |

### Generation validation

I also ran generation-based sanity checks to compare Base, vLLM Native
TurboQuant, and the LMCache TurboQuant serde-only path.

On a RULER-style 8K replay task, LMCache serde round2 is close to Base /
Native TurboQuant across the tested presets:

| Group | Preset | Accuracy |
| --- | --- | ---: |
| Base | none | 0.998 |
| Native TQ | k8v4 | 0.998 |
| Native TQ | 4bit_nc | 0.996 |
| Native TQ | k3v4_nc | 0.982 |
| Native TQ | 3bit_nc | 0.972 |
| LMCache serde round2 | k8v4 | 0.992 |
| LMCache serde round2 | 4bit_nc | 0.994 |
| LMCache serde round2 | k3v4_nc | 0.982 |
| LMCache serde round2 | 3bit_nc | 0.988 |

For the follow-up correctness check, I used a smaller but more diagnostic
CLBench subset instead of the full 163-case table.

Subset construction:

- First, I ran the bf16/fp16 baseline on the long-context CLBench samples.
- Then I selected 64 samples where the baseline model scored highly. This avoids
  cases where the base model itself fails, because those samples are not useful
  for comparing KV-cache correctness.
- Therefore, this subset is not meant to report absolute CLBench capability. It
  is meant to compare relative correctness degradation between no-serde, native
  TurboQuant, and LMCache TurboQuant serde on cases where the model can normally
  answer.

Metric:

- I used the official CLBench-style rubric judge.
- For each task:

      task_score = passed_rubrics / len(rubrics[:50])

- For each context:

      context_score = mean(task_score over tasks in that context)

- Final score:

      avg_context_score = mean(context_score over contexts)

This gives each context equal weight, so contexts with more tasks or more rubrics
do not dominate the result.

Results on the selected 64-sample subset:

| Group | Preset | Avg context score |
| --- | --- | ---: |
| LMCache | no serde | 0.8343 |
| Native TQ | 4bit_nc | 0.8073 |
| Native TQ | k3v4_nc | 0.7669 |
| Native TQ | 3bit_nc | 0.7647 |
| LMCache serde | 4bit_nc | 0.7725 |
| LMCache serde | k3v4_nc | 0.7628 |
| LMCache serde | 3bit_nc | 0.8409 |

The main point is that, after the serde fix, LMCache TurboQuant serde is in the
same range as native TurboQuant and no-serde on this controlled subset.



## Limitations

This backend currently focuses on LMCache L2 serde integration. The main
limitation is that the current path restores TurboQuant-compressed KV back to
ordinary bf16 KV and then relies on the normal attention backend.

This differs from vLLM Native TurboQuant, which computes attention directly in
the rotated quantized space with its backend-specific layout and metadata. As
shown in the generation validation above, this serde-only restore path can be
reasonable for shorter replay-style tasks, but it is not yet robust for
longer-context generation.

A follow-up direction is to align LMCache TurboQuant more deeply with the
serving backend semantics, including compressed KV layout, block/slot mapping,
K/V-specific metadata, and the attention path, rather than treating TurboQuant
only as an independent external serde format.

## Tests

The TurboQuant serde tests cover:

* serde factory / processor creation
* preset config parsing
* invalid preset rejection
* serialized size estimation
* direct CUDA serialize / deserialize roundtrip
* StorageManager roundtrip through serde-wrapped L2
* filesystem-backed L2 roundtrip
* reconstruction quality checks using correlation and error thresholds

These tests validate the core LMCache L2 serde path. They do not claim
serving-level performance speedups or task-level quality preservation.
