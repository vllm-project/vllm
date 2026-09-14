# CPU EC Connector Usage Guide

`ECCPUConnector` extends the GPU-based encoder cache with a CPU tier: it offloads encoder outputs (`encoder_cache[mm_hash]`) to a shared `/dev/shm` mmap region so later steps and later requests reuse them instead of recomputing.
GPU↔CPU copies run on pooled CUDA streams via `swap_blocks_batch`, asynchronously with model compute.

Setting `ec_enable_nixl: true` in `ec_connector_extra_config` additionally enables peer-to-peer (P2P) transfer: a consumer instance pulls an encoding directly out of a producer instance's CPU tier over NIXL instead of recomputing it locally — for E/PD disaggregation or encoder/decoder-instance sharing.

## Prerequisites

- `ECCPUConnector` requires the V2 model runner: `VLLM_USE_V2_MODEL_RUNNER=1`. It raises `ValueError` at construction otherwise.
- Local CPU-tier offload (`ec_enable_nixl` unset or `false`) needs no extra packages — the gate-off code path (`cpu/connector.py`, `cpu/scheduler/`, `cpu/worker/`, `cpu/common.py`) imports no `nixl`/`zmq`/`msgspec`, enforced by a repo test (`tests/v1/ec_connector/unit/test_no_nixl_imports.py`).
- P2P NIXL mode (`ec_enable_nixl: true`) requires the `nixl` package: `uv pip install nixl` (pinned to `nixl==1.3.2` in `requirements/kv_connectors.txt`, shared with `NixlConnector`). Refer to the [NIXL repository](https://github.com/ai-dynamo/nixl) for platform-specific installation. If `nixl` isn't importable, the connector raises `RuntimeError: ec_enable_nixl requires NIXL; install the nixl package or remove ec_enable_nixl from ec_connector_extra_config.`

## Basic Usage

Local CPU-tier offload only, within a single engine instance:

```bash
vllm serve <model> --ec-transfer-config '{
  "ec_connector": "ECCPUConnector",
  "ec_role": "ec_both",
  "ec_connector_extra_config": {"ec_cpu_bytes": 1073741824}
}'
```

- `ec_role="ec_both"`: the same process offloads to and reloads from the CPU tier.
- The tier is one mmap region (`/dev/shm/vllm_ec_{instance_id}_dp{dp_rank}.mmap`) shared by every TP/PCP worker of the instance; only TP rank 0 / PCP rank 0 writes on save, since all ranks hold identical encoder output.
- Entries are keyed by `mm_hash`; `EmbeddingCache` evicts ready+unpinned entries FIFO when space is needed.
- Each batched save/load runs on a pooled CUDA stream; completion is reported to the scheduler once the transfer's end event fires (`ECCPUWorker.build_connector_worker_meta` → `ECCPUScheduler.update_connector_output`), which marks saved entries ready and unpins loaded ones.
- The region is unlinked from `/dev/shm` in `shutdown()`.

## Usage With P2P NIXL

Producer — offloads to its CPU tier and serves reads from consumers:

```bash
vllm serve <model> --ec-transfer-config '{
  "ec_connector": "ECCPUConnector",
  "ec_role": "ec_producer",
  "ec_connector_extra_config": {"ec_enable_nixl": true, "ec_cpu_bytes": 1073741824}
}'
```

`ec_role="ec_producer"` alone already enables mm_encoder_only on the multimodal config, so vllm_config.is_mm_encoder_only is True (skips the language model, sampler, and pooler); add `--mm-encoder-only` only if you need encoder-only execution independent of `ec_transfer_config`.

Consumer — pulls encodings named in a request's `ec_transfer_params` before falling back to local encoding:

```bash
vllm serve <model> --ec-transfer-config '{
  "ec_connector": "ECCPUConnector",
  "ec_role": "ec_consumer",
  "ec_connector_extra_config": {"ec_enable_nixl": true, "ec_cpu_bytes": 1073741824}
}'
```

### Orchestration flow

1. A request finishes on the producer. `ECCPUConnector.request_finished()` returns, for each `mm_hash` still resident in its CPU tier:

   ```python
   {mm_hash: {"metadata": {...}, "peer_host": str, "peer_port": int, "size_bytes": int}}
   ```

   surfaced to the caller as `ec_transfer_params` (`RequestOutput.ec_transfer_params` / `EngineCoreOutput.ec_transfer_params`). `metadata` carries the placeholder fields the model declares for the modality, for an orchestrator that rewrites the media into a metadata-only reference; the remaining keys are the connector's own handle on the published encoding.

   The two halves are published together or not at all. An `mm_hash` the producer cannot serve — never saved because the region was full, or evicted since — is reported with empty `metadata` and no address, so an orchestrator leaves the media on the request and the consumer encodes it locally.
2. The orchestrator issues a follow-up request with the same `mm_hash` to a consumer instance, passing the producer's `ec_transfer_params` through `SamplingParams.extra_args["ec_transfer_params"]`.
3. On the consumer, `ECCPUScheduler.ensure_cache_available()` reads `request.ec_transfer_params`. For each `mm_hash` not already cached locally, it opens a ZMQ session to `(peer_host, peer_port)`, sends an `XferReq`, and on an `OK` `XferAck` issues a NIXL READ that pulls the blocks straight from the producer's mmap into its own. The request is deferred until the READ completes.
4. A `NACK_NOT_READY` means the producer announced the encoding but its GPU→mmap save has not landed yet. The consumer releases the in-flight entry without recording a failure and re-requests the read on a later step, so a save that lands a few steps late costs latency rather than a recompute.
5. On any other NACK (`NACK_MISSING`, `NACK_INCOMPAT`, `NACK_VERSION`, `NACK_INTERNAL`), ack timeout, read timeout, or peer disconnect, the consumer discards the in-flight entry and falls back to local encoding for that `mm_hash` — a P2P failure never blocks the request indefinitely.

### Protocol

- **Control plane**: ZMQ. The producer binds a `ROUTER` socket on `VLLM_EC_SIDE_CHANNEL_HOST:VLLM_EC_SIDE_CHANNEL_PORT`; each consumer opens one `DEALER` connection per producer peer, with ZMQ heartbeating (2s interval, 4s timeout, 8s TTL) to detect a dead peer. `XferReq`/`XferAck` are `msgspec` msgpack structs, versioned by `EC_CONNECTOR_VERSION` (currently `1`) — a version mismatch is NACKed.
- **Compatibility check**: every `XferReq` carries a SHA-256 hash over `(vllm_version, model, dtype, block_size_bytes)`; the producer NACKs (`NACK_INCOMPAT`) any peer whose hash differs.
- **Ack statuses**: `OK`, `NACK_MISSING` (the producer no longer holds the encoding), `NACK_NOT_READY` (held, but its save has not landed), `NACK_INCOMPAT`, `NACK_VERSION`, `NACK_INTERNAL`. Only `NACK_NOT_READY` is retryable — `RETRYABLE_NACKS` in `cpu/protocol.py` is what both ends consult, so the classification lives with the wire vocabulary rather than at each call site.
- **Data plane**: NIXL, `UCX` backend, consumer-initiated `READ` — the consumer pulls bytes directly out of the producer's registered mmap region; the producer never pushes.
- **Producer restart recovery**: the `XferAck` carries the producer's NIXL agent metadata, so a consumer can recover a READ against a restarted producer without a fresh handshake round-trip.
- **Timeouts**: consumer XferAck wait 2s; NIXL read 20s (then quarantined — not evicted — for up to 60s to let an unabortable DMA settle); the producer releases an unclaimed pinned grant after a 30s pin lease.

## Configuration

EC transfer is configured via `--ec-transfer-config` (CLI) or the `ec_transfer_config` field of `VllmConfig` (`ECTransferConfig`, `vllm/config/ec_transfer.py`):

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `ec_connector` | `str \| None` | `None` | Connector class name. Use `"ECCPUConnector"`. |
| `ec_role` | `"ec_producer" \| "ec_consumer" \| "ec_both" \| None` | `None` | Required whenever `ec_connector` is set. `ec_producer` offloads GPU→CPU only, `ec_consumer` reloads CPU→GPU only, `ec_both` does both in the same process. |
| `ec_connector_extra_config` | `dict[str, Any]` | `{}` | Connector-specific settings, including `ec_enable_nixl` — see [`ec_connector_extra_config` Reference](#ec_connector_extra_config-reference). |
| `engine_id` | `str \| None` | random UUID4 | Names the NIXL agent when `ec_enable_nixl=True`. |
| `ec_connector_module_path` | `str \| None` | `None` | Python module path to load an out-of-tree connector from, when `ec_connector` isn't in the built-in registry (`ECExampleConnector`, `ECCPUConnector`). |

### `ec_connector_extra_config` Reference

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `ec_enable_nixl` | `bool` | No (default `false`) | Enables NIXL P2P transfer in addition to local CPU offload. Omitted or `false` imports no NIXL/ZMQ. Extra config is not type coerced, so a string value is parsed: `"true"`, `"1"`, `"yes"` enable it, anything else does not. |
| `consumer_ack_timeout_s` | `float` | No (default `2.0`) | How long a consumer waits for an `XferAck` before giving up on a read. The producer answers `XferReq`s from its own scheduler step, so its reply latency scales with the encoder's `--max-num-batched-tokens`: a loaded encoder whose steps run longer than this makes consumers abandon reads the producer is about to grant. Raise it for large encoder batches. |
| `ec_cpu_bytes` | `int` | Yes | Total size, in bytes, of the shared CPU mmap region. `ECCPUConnector` raises `ValueError` if unset. Block count = `ec_cpu_bytes // block_size_bytes`, where `block_size_bytes = hidden_dim * dtype.element_size()` (`hidden_dim` accounts for Qwen3-VL deepstack: `out_hidden_size * (1 + num_deepstack_layers)`). |

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_EC_SIDE_CHANNEL_HOST` | `localhost` | Host the producer's ZMQ `ROUTER` socket binds to. Set to a routable address (e.g. the pod IP) for multi-instance/multi-node P2P — the default only works when producer and consumer share a host. |
| `VLLM_EC_SIDE_CHANNEL_PORT` | `5601` | Port for the same ZMQ `ROUTER` socket. |

Both are read only when `ec_enable_nixl=True` on a producer (`ec_role="ec_producer"` or `"ec_both"`).

## Limitations

- No mechanism to notify an orchestrator or peer instance when an encoding is evicted from the CPU tier before it's consumed. A consumer only discovers a miss when its `XferReq` is NACKed (`NACK_MISSING`) and falls back to local recompute.
- At the moment, Mmap cleanup on process shutdown is best-effort: if the creating process is `SIGKILL`ed before `ECSharedRegion.cleanup()` runs, the `/dev/shm/vllm_ec_*.mmap` file leaks and must be removed manually.
- `NixlDataTransport` hardcodes the `UCX` backend.
- A retried read releases its destination blocks and re-allocates them on the next step, so a producer whose save is slow to land makes the consumer evict ready entries to win the same blocks back, once per engine step.
- Falling back to local encoding requires the media to still be on the request. Announcing an encoding only when it can be served keeps an orchestrator from rewriting the media away for one that cannot, but an encoding lost *after* it was announced — evicted between the announcement and the consumer's read — leaves a rewritten request with nothing to embed, and it fails in the worker's `sanity_check_mm_encoder_outputs`. `ensure_cache_available()` can only defer a request, not fail it, so reporting this cleanly needs a scheduler-side failure path.
