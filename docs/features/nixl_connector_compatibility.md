# NixlConnector Compatibility Matrix

This page documents the feature compatibility of **disaggregated prefilling with the NixlConnector**. For general usage instructions, see the [NixlConnector Usage Guide](nixl_connector_usage.md). For an overview of disaggregated prefilling, see [Disaggregated Prefilling](disagg_prefill.md).

!!! note
    This page reflects the current state of the codebase and is subject to change as features evolve. Entries marked 🟠 or ❌ may link to tracking issues. See the [NIXL connector roadmap](https://github.com/vllm-project/vllm/issues/33702) for upcoming feature development.

**Legend:**

- ✅ = Fully supported
- 🟠 = Partial support (see footnotes)
- ❌ = Not supported
- ❔ = Unknown / not yet validated
- 🚧 = Work in progress

!!! info "Universally supported features"
    The following features work with **all** model architectures when using NixlConnector PD disaggregated serving:

    [Chunked Prefill](../configuration/optimization.md#chunked-prefill) |
    [APC (Prefix Caching)](automatic_prefix_caching.md) |
    [Data Parallel](../serving/data_parallel_deployment.md) |
    CUDA graph |
    Logprobs |
    Prompt Logprobs |
    [Prompt Embeds](prompt_embeds.md) |
    Multiple NIXL backends (UCX, GDS, LIBFABRIC, etc.)

## Model Architecture x Capability

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| Model type | <abbr title="Basic Prefill/Decode disaggregation">Basic PD</abbr> | <abbr title="Speculative Decoding">Spec Decode</abbr> | <abbr title="Heterogeneous Tensor Parallelism (P TP != D TP)">Hetero TP</abbr> | <abbr title="Cross-layer blocks optimization">Cross-layer blocks</abbr> | <abbr title="Sliding Window Attention">SWA</abbr> | <abbr title="CPU host buffer offload (e.g. TPU)">Host buffer</abbr> | <abbr title="Different block sizes on P and D">Hetero block size</abbr> |
| - | - | - | - | - | - | - | - |
| Dense Transformers | ✅ | ✅<sup>1</sup> | ✅ | ✅<sup>2</sup> | ✅ | ✅ | 🟠<sup>3</sup> |
| MLA (e.g. DeepSeek-V2/V3) | ✅ | ✅<sup>1</sup> | 🟠<sup>4</sup> | ✅<sup>2</sup> | ✅ | ✅ | 🟠<sup>3</sup> |
| Sparse MLA (e.g. DeepSeek-V3.2) | ✅ | ✅<sup>1</sup> | 🟠<sup>4</sup> | ✅<sup>2</sup> | ✅ | ✅ | 🟠<sup>3</sup> |
| Hybrid SSM / Mamba | ✅ | ❔ | 🚧<sup>5</sup> | ❌ | ✅ | ✅ | ❌<sup>6</sup> |
| MoE | ✅ | ✅<sup>1</sup> | ✅ | ✅<sup>2</sup> | ✅ | ✅ | 🟠<sup>3</sup> |
| Multimodal | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ |
| Encoder-Decoder | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

<sup>1</sup> P and D instances must use the same speculation configuration.

<sup>2</sup> Requires `FLASH_ATTN` or `FLASHINFER` backend **and** `HND` KV cache layout. Enable via `--kv-transfer-config '{"kv_connector_extra_config": {"enable_cross_layers_blocks": "True"}}'`.

<sup>3</sup> Supported only when HMA is **not** required (i.e., non-hybrid models). Block IDs are remapped automatically. Only P block size < D block size is supported.

<sup>4</sup> MLA KV cache is replicated across TP workers, so heterogeneous TP works but there is no head-splitting. When P TP > D TP, only a single read is executed (redundant ranks are skipped). D TP > P TP also works.

<sup>5</sup> Hybrid SSM (Mamba) models require **homogeneous TP** (`P TP == D TP`). Heterogeneous TP is not yet supported for Mamba layers.

<sup>6</sup> HMA (required by hybrid models) does not support different remote block sizes.

## Configuration Notes

### What must match between P and D

By default, a **compatibility hash** is checked during handshake. P and D instances must agree on:

- vLLM version and NIXL connector version
- Model (architecture, dtype, number of KV heads, head size, number of hidden layers)
- Attention backend
- KV cache dtype (`cache_dtype`)

!!! warning
    Disable the hash check with `--kv-transfer-config '{"kv_connector_extra_config": {"enforce_handshake_compat": false}}'` at your own risk.

### What can safely differ between P and D

- `tensor-parallel-size` (heterogeneous TP, subject to model restrictions above)
- `block-size` (heterogeneous block size, subject to restrictions above)
- Number of KV cache blocks (determined by available memory on each instance)

### KV cache layout

- NixlConnector defaults to **`HND`** layout for optimal transfer performance (non-MLA models).
- `NHD` layout is supported but does **not** allow heterogeneous TP head splitting.
- Experimental `HND` ↔ `NHD` permute: enable via `--kv-transfer-config '{"enable_permute_local_kv": true}'`. Not supported with HMA.

### Quantized KV cache

[Quantized KV cache](quantization/quantized_kvcache.md) (e.g., FP8) requires both P and D instances to use the **same** `cache_dtype`. Mismatched cache dtypes will fail the compatibility hash check during handshake.

- **Static quantization** (scales loaded from checkpoint): ✅ Supported. Scales are loaded independently by each instance from the model checkpoint.
- **Dynamic quantization** (scales computed at runtime): ❌ Not supported. Per-block scales are not transferred alongside KV cache data.
- **Packed-layout scales** (scales stored inline with weights): ✅ Supported. Scales are transferred together with the KV cache blocks.
