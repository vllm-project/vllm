# Cross-Encoder Output Reuse

Cross-Encoder output reuse lets an Encoder load image embeddings computed by
another Encoder from a shared Mooncake Store. Local cache misses check the Store
before encoding; newly computed outputs are published asynchronously. Both paths
use the existing `ECMooncakeConnector` P2P delivery to Prefill.

## Requirements

- A working [disaggregated Encoder](disagg_encoder.md) deployment using
  `ECMooncakeConnector`. See the [Mooncake integration example](../../tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh)
  for the base E/PD configuration.
- Encoder TP=1 and no dynamic LoRA.
- `--mm-processor-cache-gb` greater than zero to preserve content identifiers.
- Mooncake 0.3.12 or later with a running Store. SSD offloading uses an
  independently managed `standalone-store` owner.

Shared reuse supports images with contiguous 2D FP16, BF16 or FP32 encoder outputs.
Other modalities use normal encoding; non-contiguous outputs are not published.

## Usage

On each Encoder, add `cross_encoder_cache: true` to the existing producer
`--ec-transfer-config`, preserving its P2P settings:

```json
{
  "ec_connector": "ECMooncakeConnector",
  "ec_role": "ec_producer",
  "ec_connector_extra_config": {
    "cross_encoder_cache": true
  }
}
```

Prefill keeps its existing `ECMooncakeConnector` consumer configuration and does
not need a Store client.

Create a JSON file pointing to an existing Store. For an independently managed
RAM pool, use `standalone-store` with `global_segment_size` set to zero:

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

Replace the addresses with your Store endpoints and set the path on each Encoder
before starting vLLM:

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```

This connects to an existing Store; it does not start the storage services.
For Store setup and tenant configuration, see the
[Mooncake Store guide](mooncake_store_connector_usage.md#prerequisites).

## Configuration

These options belong in the producer's `ec_connector_extra_config`:

| Option | Default | Description |
| --- | --- | --- |
| `cross_encoder_cache` | `false` | Enable shared output reuse. |
| `embedding_cache_prefix` | `""` | Namespace prefix for shared embeddings. |
| `embedding_model_identity` | Configured model path | Override the model field in Store keys. Matching multimodal identifiers are still required. |
| `store_max_pending_items` | `32` | Maximum pending publications per Encoder. Must be positive. |
| `store_max_pending_bytes` | `2147483648` (2 GiB) | Maximum retained tensor storage for pending publications per Encoder. Must be positive. |
| `store_read_buffer_bytes` | `134217728` (128 MiB) | Maximum reusable CPU staging buffer per Encoder; pinned for CUDA outputs. Must be positive. |
| `store_disk_staging_budget_bytes` | unset | Required when Store JSON enables offload. Positive per-GET allowance for the SSD owner's temporary read buffer, including DirectIO overhead. Ignored for RAM-only Store configurations. |
| `preferred_segment` | unset | Owner segment used for PUT placement. Falls back to `MOONCAKE_PREFERRED_SEGMENT`. Use the actual SSD owner segment name. |

Encoders sharing outputs must use the same immutable weights, compatible
preprocessing and matching multimodal identifiers. Use the same configured model
path for automatic identifiers: `embedding_model_identity` alone does not enable
reuse across different paths. Change `embedding_cache_prefix` when replacing
weights in place or changing output-affecting settings. Caller-provided UUIDs
must consistently identify the same input; see [cached inputs](multimodal_inputs.md#cached-inputs).

## Synchronous SSD offloading

Start the master and an external Store owner with SSD offload enabled as described
in [Disk Offloading](mooncake_store_connector_usage.md#disk-offloading).
In the Encoder's Store JSON, set `mode` to `standalone-store`,
`global_segment_size` to `0`, and `enable_offload` to `true`.
Embedded SSD ownership is not supported by this EC integration.

For a low-volume correctness check with the bucket backend, set
`MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT=1` on the external owner. The examined
Mooncake backend can retain a partial batch in its ungrouped pool until it
meets the bucket grouping thresholds; a successful PUT does not establish SSD
persistence. Confirm a `LOCAL_DISK` replica before deliberately creating DRAM
pressure. This one-object bucket setting is for validation, not a performance
recommendation.

Add the read allowance to the producer's existing EC settings:

```json
{
  "ec_connector": "ECMooncakeConnector",
  "ec_role": "ec_producer",
  "ec_connector_extra_config": {
    "cross_encoder_cache": true,
    "store_disk_staging_budget_bytes": 67108864
  }
}
```

The example allows 64 MiB of owner staging per GET. Choose an allowance below
the owner's configured available staging capacity and leave room for other
requesters. Each object's allowance is estimated as its full byte size (including
the 24-byte header) rounded up to 4096 bytes, plus 8192 bytes of padding.
Batches obey both this allowance and `store_read_buffer_bytes`; individual
objects exceeding either bound fall back to encoding. This is a per-call
limit, not a reservation or global admission controller for the owner.

PUT still publishes completed outputs asynchronously to the Store. The Store
manages background SSD persistence and eviction; PUT completion does not
acknowledge SSD persistence. On a Store hit, the Encoder synchronously reads
the object into its registered CPU buffer, validates it, and copies the payload
to the output device before continuing. The existing P2P delivery is unchanged.
There is no request-arrival prefetch in this implementation.

SSD GET RPC failures/timeouts and unclassified native failures retain the
registered receive buffer and fail the worker because I/O completion cannot
be confirmed. They are not retried as cache misses. Confirmed rejections and
missing objects retain the normal encoding fallback. Physical disk faults,
owner recovery and concurrent staging exhaustion require deployment validation.

## Limitations

- Store reads are synchronous. Reuse skips Encoder computation, but not
  preprocessing, P2P delivery or the scheduler's Encoder budget reservation.
- Hits are read in batches through one lazily allocated, registered CPU staging
  buffer. Each chunk is copied to independent output tensors before the buffer
  is reused. Reads are split by `store_read_buffer_bytes`; an individual object
  larger than this capacity (including its 24-byte header) falls back to encoding.
  Staging memory is additional to the Store's `local_buffer_size` and is released
  on healthy shutdown. The first load includes allocation and registration costs.
- Publication is best-effort. A request can finish before its outputs reach the
  Store, so concurrent cold requests may still encode the same image.
- Publication budgets count each retained backing storage once across pending
  views. Its charge is released after the last view is safely reclaimed.
  Exceeding a budget skips the write. Store client buffers and the P2P pool
  consume additional memory.
- Cache misses and recoverable read errors fall back to encoding. Recoverable
  publication errors skip the write. Unexpected native errors or unconfirmed
  I/O completion or buffer release can fail the worker.
- Incompatible objects are rejected without replacement; they can trigger
  repeated fallback until evicted.
- An independent Store can retain embeddings across Encoder restarts. Capacity,
  eviction and resilience to Store failures depend on the Store deployment.
- The `protocol:v3` namespace isolates the compact embedding format from older
  cache objects. Encoders using different format versions do not share hits.
