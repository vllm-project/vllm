# Shared Encoder output Store

The shared Encoder output Store reuses embeddings across Encoder instances in
EPD serving. An Encoder publishes newly computed outputs while delivering them
to Prefill; another Encoder can load those outputs instead of running the
vision encoder again. Mooncake Store provides the RAM cache, and
`ECMooncakeConnector` P2P is the currently supported delivery integration.

## Execution

The scheduler retains its existing local-cache, compute-budget and allocation
rules. On a local miss, the producer worker attempts a Store read before
encoding. Loaded items enter the local encoder cache and are submitted to the
existing P2P reservation. Only unresolved items run the encoder. Newly computed
outputs are published asynchronously, independently of delivery completion.

Publication is best-effort: serving request completion does not acknowledge
Store readiness or pin an object against eviction. A request arriving before
publication completes may encode the input again.

Store reads are synchronous on the worker path. They do not bypass preprocessing
or refund the scheduler's reserved encoder budget. Concurrent cold requests may
compute the same image independently.

## Configuration

Start from a working Mooncake EPD deployment. The producer requires Model Runner
V2, Encoder TP=1, and no dynamic LoRA. Shared lookup currently covers images;
other modalities retain the normal encoder path. Supported embedding dtypes are
FP16, BF16 and FP32. Only contiguous outputs are published; non-contiguous
outputs are skipped without allocating a staging copy.

Keep `--mm-processor-cache-gb` greater than zero when shared reuse is enabled.
Encoder-only serving disables prefix caching; also disabling processor caching
replaces content identifiers with process-local request counters that can collide
across Encoders or restarts. This combination is rejected at startup.

On each Encoder, enable `cross_encoder_cache` in the normal Mooncake P2P
configuration. For example:

```json
{
  "ec_connector": "ECMooncakeConnector",
  "ec_role": "ec_producer",
  "ec_connector_extra_config": {
    "mooncake_protocol": "tcp",
    "cross_encoder_cache": true,
    "embedding_cache_prefix": "my-deployment-v1",
    "embedding_model_identity": "my-model-snapshot",
    "store_max_pending_items": 32,
    "store_max_pending_bytes": 2147483648
  }
}
```

Keep Prefill configured with `ECMooncakeConnector`, `ec_consumer`, and its normal
P2P addresses and buffer settings. It does not need a Store client. Omitting
`cross_encoder_cache` (or setting it to false) disables shared reuse and does
not create a Store client or publisher.

Set `MOONCAKE_CONFIG_PATH` on Encoder workers to a JSON file describing the
running Store. A client using an independently managed RAM pool can use:

```json
{
  "metadata_server": "http://STORE_HOST:2379/metadata",
  "master_server_address": "STORE_HOST:50051",
  "protocol": "tcp",
  "device_name": "",
  "mode": "standalone-store",
  "global_segment_size": 0,
  "local_buffer_size": "4GB"
}
```

Replace the addresses with the actual deployment endpoints. This file connects
to existing Mooncake services; it does not start the master, metadata service
or storage pool. The bindings must provide `batch_is_exist`,
`batch_put_from_multi_buffers`, `get_into_ranges`, `register_buffer`, and
`unregister_buffer`. Use Mooncake 0.3.12 or later, including its `close()` and
`ObjectDataType.TENSOR` APIs. EC and KV use the same Store configuration parser
and setup. Non-default `tenant_id` is passed to Mooncake; unsupported tenant
configuration fails at setup rather than falling back to the default tenant.
`standalone-store` requires an explicit zero `global_segment_size`. This
feature uses RAM storage and rejects `enable_offload: true`.

An independent Store process allows embeddings to survive Encoder worker
restarts. RAM contents do not survive loss of the storage process unless the
Store deployment provides the necessary replicas. Capacity and eviction belong
to Mooncake Store.

## Cache identity and memory

Keys contain the deployment prefix, model identity and configured revision,
encoder configuration hash, dtype, format version, and vLLM
multimodal identifier. They exclude the request and Encoder instance IDs.
The current key namespace uses `protocol:v2`.

All Encoders sharing a namespace must use the same immutable model weights and
compatible preprocessing. `embedding_model_identity` overrides only the key's
model field; reuse also requires identical vLLM multimodal identifiers.
Automatically generated identifiers include the configured model path.
Caller-provided UUIDs are also hashed with that path when processor or media
kwargs are present. Use the same configured model path across Encoders when
relying on automatic identifiers: an equal `embedding_model_identity` alone
does not enable reuse across different mount paths.

The model identity must not identify different weights as equivalent. Change
the namespace when replacing weights in place or changing an output-affecting
setting. Caller-supplied multimodal identifiers must identify the same input
consistently. Shape validation alone cannot detect semantically different
outputs with the same dimensions.

The default publisher admits at most 32 pending items and 2 GiB of retained
tensor storage per Encoder. Both limits require positive integers and are
validated before native initialization. Views charge their full backing storage;
aliases may be charged more than once. Each admitted item also needs a 304-byte header.
Store client buffers and the existing P2P pool are separate allocations.
Exceeding the publication budget skips that cache write without rejecting the
request.

## Failure semantics and limitations

A missing object, rejected lookup, completed failed read, or incompatible tensor
falls back to encoding the affected item. A buffer-registration rejection occurs
before I/O submission: after undoing any earlier registrations for that operation,
reads fall back to encoding and publications are skipped. Completed publication
failures do not change P2P request completion. Unexpected binding exceptions and
invalid lookup result counts remain errors. If native I/O completion or buffer
unregistration cannot be established, the worker fails while retaining the
affected buffers: it cannot safely release memory
that a transfer may still access. Shared Store is therefore an optional
optimization, not isolation from every backend failure.

The feature requires compatible model weights and preprocessing across
Encoders. It does not coordinate concurrent cold computation, change request
routing, or make the scheduler aware of remote hits. Benefits depend on the
frequency of reusable local misses and the cost of Store access relative to
encoding.

Shutdown stops publication admission, waits for queued writes, reaps their results,
and closes the Store client. Buffers with unconfirmed I/O or unregistration remain
owned until process exit; shutdown does not reinterpret them as recoverable misses.
