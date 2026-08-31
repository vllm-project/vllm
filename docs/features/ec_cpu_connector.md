# CPU EC Connector Usage Guide

`ECCPUConnector` extends the GPU-based encoder cache with a CPU tier: it offloads encoder outputs (`encoder_cache[mm_hash]`) to a shared `/dev/shm` mmap region so later steps and later requests reuse them instead of recomputing.
GPU↔CPU copies run on pooled CUDA streams via `swap_blocks_batch`, asynchronously with model compute.

Setting `ec_enable_nixl: true` additionally enables peer-to-peer (P2P) transfer: a consumer instance pulls an encoding directly out of a producer instance's CPU tier over NIXL instead of recomputing it locally — for E/PD disaggregation or encoder/decoder-instance sharing.

## Prerequisites

- `ECCPUConnector` requires the V2 model runner: `VLLM_USE_V2_MODEL_RUNNER=1`. It raises `ValueError` at construction otherwise.
- Local CPU-tier offload (`ec_enable_nixl` unset or `false`) needs no extra packages — the gate-off code path (`cpu/connector.py`, `cpu/scheduler/`, `cpu/worker/`, `cpu/common.py`) imports no `nixl`/`zmq`/`msgspec`, enforced by a repo test (`test_no_nixl_imports.py`).
- P2P NIXL mode (`ec_enable_nixl: true`) requires the `nixl` package: `uv pip install nixl` (pinned to `nixl==1.3.1` in `requirements/kv_connectors.txt`, shared with `NixlConnector`). Refer to the [NIXL repository](https://github.com/ai-dynamo/nixl) for platform-specific installation. If `nixl` isn't importable, the connector raises `RuntimeError: ec_enable_nixl=True requires NIXL; install the nixl package or set ec_enable_nixl=False.`

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
- The tier is one mmap region (`/dev/shm/vllm_ec_{engine_id}_dp{dp_rank}.mmap`) shared by every TP/PCP worker of the instance; only TP rank 0 / PCP rank 0 writes on save, since all ranks hold identical encoder output.
- Entries are keyed by `mm_hash`; `EmbeddingCache` evicts ready+unpinned entries FIFO when space is needed.
- Each batched save/load runs on a pooled CUDA stream; completion is reported to the scheduler once the transfer's end event fires (`ECCPUWorker.build_connector_worker_meta` → `ECCPUScheduler.update_connector_output`), which marks saved entries ready and unpins loaded ones.
- The region is unlinked from `/dev/shm` in `shutdown()`.

## Usage With P2P NIXL

Producer — offloads to its CPU tier and serves reads from consumers:

```bash
vllm serve <model> --ec-transfer-config '{
  "ec_connector": "ECCPUConnector",
  "ec_role": "ec_producer",
  "ec_enable_nixl": true,
  "ec_connector_extra_config": {"ec_cpu_bytes": 1073741824}
}'
```

`ec_role="ec_producer"` alone already sets `vllm_config.is_encoder_only=True` (skips the language model, sampler, and pooler); add `--mm-encoder-only` only if you need encoder-only execution independent of `ec_transfer_config`.

Consumer — pulls encodings named in a request's `ec_transfer_params` before falling back to local encoding:

```bash
vllm serve <model> --ec-transfer-config '{
  "ec_connector": "ECCPUConnector",
  "ec_role": "ec_consumer",
  "ec_enable_nixl": true,
  "ec_connector_extra_config": {"ec_cpu_bytes": 1073741824}
}'
```

### Orchestration flow

1. A request finishes on the producer. `ECCPUConnector.request_finished()` returns, for each `mm_hash` still resident in its CPU tier:

   ```python
   {mm_hash: {"peer_host": str, "peer_port": int, "size_bytes": int}}
   ```

   surfaced to the caller as `ec_transfer_params` (`RequestOutput.ec_transfer_params` / `EngineCoreOutput.ec_transfer_params`).
2. The orchestrator issues a follow-up request with the same `mm_hash` to a consumer instance, passing the producer's `ec_transfer_params` through `SamplingParams.extra_args["ec_transfer_params"]`.
3. On the consumer, `ECCPUScheduler.ensure_cache_available()` reads `request.ec_transfer_params`. For each `mm_hash` not already cached locally, it opens a ZMQ session to `(peer_host, peer_port)`, sends an `XferReq`, and on an `OK` `XferAck` issues a NIXL READ that pulls the blocks straight from the producer's mmap into its own. The request is deferred until the READ completes.
4. On any NACK (`NACK_MISSING`, `NACK_INCOMPAT`, `NACK_VERSION`), ack timeout, read timeout, or peer disconnect, the consumer discards the in-flight entry and falls back to local encoding for that `mm_hash` — a P2P failure never blocks the request indefinitely.

### Protocol

- **Control plane**: ZMQ. The producer binds a `ROUTER` socket on `VLLM_EC_SIDE_CHANNEL_HOST:VLLM_EC_SIDE_CHANNEL_PORT`; each consumer opens one `DEALER` connection per producer peer, with ZMQ heartbeating (2s interval, 4s timeout, 8s TTL) to detect a dead peer. `XferReq`/`XferAck` are `msgspec` msgpack structs, versioned by `EC_CONNECTOR_VERSION` (currently `1`) — a version mismatch is NACKed.
- **Compatibility check**: every `XferReq` carries a SHA-256 hash over `(vllm_version, model, dtype, block_size_bytes)`; the producer NACKs (`NACK_INCOMPAT`) any peer whose hash differs.
- **Data plane**: NIXL, `UCX` backend, consumer-initiated `READ` — the consumer pulls bytes directly out of the producer's registered mmap region; the producer never pushes.
- **Producer restart recovery**: the `XferAck` carries the producer's NIXL agent metadata, so a consumer can recover a READ against a restarted producer without a fresh handshake round-trip.
- **Timeouts**: consumer XferAck wait 2s; NIXL read 20s (then quarantined — not evicted — for up to 60s to let an unabortable DMA settle); the producer releases an unclaimed pinned grant after a 30s pin lease.

## Configuration

EC transfer is configured via `--ec-transfer-config` (CLI) or the `ec_transfer_config` field of `VllmConfig` (`ECTransferConfig`, `vllm/config/ec_transfer.py`):

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `ec_connector` | `str \| None` | `None` | Connector class name. Use `"ECCPUConnector"`. |
| `ec_role` | `"ec_producer" \| "ec_consumer" \| "ec_both" \| None` | `None` | Required whenever `ec_connector` is set. `ec_producer` offloads GPU→CPU only, `ec_consumer` reloads CPU→GPU only, `ec_both` does both in the same process. |
| `ec_enable_nixl` | `bool` | `False` | Enables NIXL P2P transfer in addition to local CPU offload. `False` imports no NIXL/ZMQ. |
| `ec_connector_extra_config` | `dict[str, Any]` | `{}` | Connector-specific settings — see [`ec_connector_extra_config` Reference](#ec_connector_extra_config-reference). |
| `engine_id` | `str \| None` | random UUID4 | Names the shared mmap file: `/dev/shm/vllm_ec_{engine_id}_dp{dp_rank}.mmap`. Also seeds the NIXL agent name when `ec_enable_nixl=True`. |
| `ec_connector_module_path` | `str \| None` | `None` | Python module path to load an out-of-tree connector from, when `ec_connector` isn't in the built-in registry (`ECExampleConnector`, `ECCPUConnector`). |

### `ec_connector_extra_config` Reference

| Key | Type | Required | Description |
| --- | --- | --- | --- |
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
