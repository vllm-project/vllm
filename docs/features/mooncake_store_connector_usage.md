# MooncakeStoreConnector Usage Guide

MooncakeStoreConnector is a KV cache connector that uses [MooncakeDistributedStore](https://github.com/kvcache-ai/Mooncake) as a shared KV cache pool. Unlike `MooncakeConnector` which does direct point-to-point KV transfer between prefiller and decoder, MooncakeStoreConnector enables KV cache offloading to an external distributed store, supporting:

- **CPU/disk offloading**: Extend effective KV cache capacity by offloading to CPU memory or disk via Mooncake's transfer engine.
- **Prefix caching across instances**: Hash-based deduplication allows multiple vLLM instances to share cached KV blocks through the store.
- **Single-node and multi-node deployment**: Works both as a standalone KV cache extension and in disaggregated prefill-decode setups.
- **EGM shared KV cache**: Opt in to an external Extended GPU Memory provider over NVLink; see [Standalone EGM over NVLink](#standalone-egm-over-nvlink).

## Prerequisites

### Install Mooncake

Install mooncake through pip:

```bash
uv pip install mooncake-transfer-engine
```

Refer to the [Mooncake official repository](https://github.com/kvcache-ai/Mooncake) for more installation instructions and building from source.

### Start the Mooncake Master Server

The Mooncake master manages metadata and coordinates the distributed store. Start it before launching vLLM:

```bash
mooncake_master --port 50051
```

Default ports:

- RPC: 50051

Multiple vLLM instances can share the same master server.

### Configure Mooncake

Create a JSON configuration file (e.g., `mooncake_config.json`):

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "80GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
```

- `mode`: Topology selection. `"embedded"` (default, PR-40900 baseline) has each
  vLLM rank contribute `global_segment_size` to the pool in-process.
  `"standalone-store"` makes ranks pure requesters — an external
  Store provider owns the memory pool and (optionally) the SSD tier.
- `protocol`: Use `"rdma"` for best performance. `"tcp"` works as a fallback.
  `"nvlink"` is used by the opt-in [EGM deployment](#standalone-egm-over-nvlink).
- `global_segment_size`: CPU memory contributed to the distributed pool (per
  GPU). Must be `> 0` in `embedded` mode and `0` in `standalone-store` mode.
- `local_buffer_size`: Private buffer for this node's own operations (per GPU).
  Must be greater than zero in the vLLM config. The default is 4 GiB.
- `enable_offload`: When `true`, vLLM allocates a DirectIO staging buffer so
  large prefills do not exceed the owner's SSD-write budget. Set this together
  with the matching `--enable_offload=true` flag on `mooncake_master` and on
  the external `mooncake_client` (if any).
- `tenant_id`: Optional Mooncake tenant namespace. Producers and consumers
  that should share store data must use the same tenant id. Default:
  `"default"`.

Set the config path via environment variable:

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```

## Usage

### Single-Node KV Cache Offloading

Use MooncakeStoreConnector to offload KV cache to CPU memory, extending the effective cache size:

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
```

### Disaggregated Prefill-Decode (XpYd)

In disaggregated prefill-decode mode, use `MultiConnector` to combine `MooncakeConnector` (point-to-point KV transfer) with `MooncakeStoreConnector` (shared KV cache pool). This enables both direct P2P transfer between prefiller and decoder, and cross-instance prefix cache sharing via the distributed store.
**Prefiller Node:**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50052 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8100 \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_producer"
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both"
                }
            ]
        }
    }'
```

**Decoder Node:**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50053 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8200 \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_consumer"
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_consumer"
                }
            ]
        }
    }'
```

To also offload newly completed decode KV blocks, add the following extra
configuration to the decoder's `MooncakeStoreConnector` entry.

When decode processing starts, the consumer checks the block-aligned prompt
prefix and fills any blocks missing from the Store. Subsequent saves append
newly completed decode blocks. This keeps a complete, reusable prefix in the
Store. It also covers prompt KV delivered directly by `MooncakeConnector`.

```json
{
    "kv_connector_extra_config": {
        "save_decode_cache": true
    }
}
```

#### Sharing one Store across multiple Prefill TP sizes

Heterogeneous-TP sharing normally uses a fixed `store_tp_size`. When several
prefillers use different TP sizes, opt in to a common Store TP derived from
their least common multiple:

```json
{
    "kv_connector_extra_config": {
        "enable_store_tp_lcm": true,
        "prefill_tp_sizes": [4, 2]
    }
}
```

Every prefiller and decoder that shares these entries must use the same list.
The example selects Store TP 4: a TP4 endpoint maps each rank to one Store
shard, while TP2 endpoints map each rank to two Store shards. Runtime TP sizes
remain unchanged. A decoder configured with `"save_decode_cache": true` uses
the same Store TP for decode KV from every prefiller.

The list may contain positive integer TP sizes. Sharing requires a Store TP that
is at least the local TP and divisible by it, an LBHNC or LBNHC local KV cache,
and the existing topology and KV-head constraints. The Store namespace includes
the attention backend's selected layout. Different layouts use separate Store
entries. Malformed lists and unsupported endpoints use an isolated rank-local
key layout. When
`enable_store_tp_lcm` is absent or false, `prefill_tp_sizes` has no effect and
the existing `store_tp_size` behavior is unchanged.

**Proxy:**

A disaggregation proxy routes requests between prefiller and decoder nodes.
When `MooncakeConnector` is also used for direct P2P transfer, refer to its
[usage guide](mooncake_connector_usage.md) for proxy setup details.

### Disk Offloading

Disk offloading is most commonly run in `standalone-store` mode: an external
`mooncake_client` process owns the CPU pool and the SSD tier, and each vLLM
rank is a pure requester. This avoids per-rank duplication of the SSD pool
and keeps DirectIO budget tracking on a single process.

Three things need to be aligned for end-to-end disk offloading:

1. **`mooncake_master`** is started with `--enable_offload=true`.
2. **`mooncake_client`** (the owner) is started with `--enable_offload=true`
   plus an SSD path via `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`.
3. **vLLM-side** sets `"enable_offload": true` in the JSON config file (this is
   read by the connector and is **not** an environment variable).

Example `mooncake_config.json` for the vLLM side:

```json
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": 0,
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "mlx5_0",
  "enable_offload": true
}
```

Steer this rank to the local owner segment with:

```bash
export MOONCAKE_PREFERRED_SEGMENT=127.0.0.1:50053
```

The owner's SSD directory, on-disk eviction policy, and the DirectIO staging
buffer size are controlled on the `mooncake_client` side via the standard
Mooncake environment variables (`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`,
`MOONCAKE_BUCKET_EVICTION_POLICY`, `MOONCAKE_USE_URING`,
`MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`,
`MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES`, etc.). Those are independent of
the vLLM JSON config.

### Tenant Isolation

Set `tenant_id` in the Mooncake JSON config when different vLLM deployments should use separate Mooncake tenant namespaces:

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "80GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false,
  "tenant_id": "tenant-a"
}
```

Strict isolation requires a Mooncake master started with `--enable_multi_tenants=true` and a tenant quota policy that registers each tenant. Non-default `tenant_id` also requires a Mooncake version whose `MooncakeDistributedStore.setup()` accepts the `tenant_id` parameter. In `standalone-store` mode, start the external `mooncake_client` with the matching tenant id because that process owns the real store client.

### Standalone EGM over NVLink

Extended GPU Memory (EGM) lets a separate Mooncake provider contribute
Grace host memory to a shared KV Store over the NVLink fabric. This recipe
combines that provider with the existing `MooncakeStoreConnector` and NVLink
allocator. vLLM saves and loads KV blocks through the Store; attention still
uses the local GPU KV cache. Only the provider opts into EGM pool ownership.

This recipe uses a provider on one OS node and vLLM requesters on another node
in the same supported NVLink fabric and IMEX domain. A same-node run does not
validate the cross-node EGM path.

#### Requirements and scope

- Use EGM-capable hardware, such as a supported GB200/GB300 NVLink deployment,
  with working CUDA fabric-handle import/export and IMEX on both nodes. Follow
  Mooncake's [NVLink prerequisites](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/getting_started/supported-protocols.md)
  and your platform's IMEX setup instructions before starting the Store.
- Install a Mooncake SDK built with Store support, `USE_CUDA=ON` and
  `USE_MNNVL=ON`, including the
  [EGM Store API](https://github.com/kvcache-ai/Mooncake/blob/8c19b728/docs/source/api-reference/python/mooncake-store.md)
  and `NVLinkAllocator`. Follow Mooncake's
  [build instructions](https://github.com/kvcache-ai/Mooncake/blob/8c19b728/docs/source/getting_started/build.md);
  installing an arbitrary wheel does not establish EGM support.
- This recipe uses Mooncake's classic C++ backend and manual NVLink transport
  selection. In every provider and requester shell, set:

    ```bash
    export MOONCAKE_STORE_BACKEND=cpp
    export MC_MS_AUTO_DISC=0
    export PYTHONHASHSEED=0
    unset MC_USE_TENT MC_USE_TEV1 MC_USE_NVLINK_IPC
    ```

- Keep `enable_offload: false` for this memory-only example. SSD/DFS tiering,
  cross-encoder caching and embedded EGM ownership inside vLLM are outside its
  scope. The example selects the existing `custom_mem_pool: "NVLINK"` option
  for GPU KV allocation; that option is incompatible with sleep mode and the
  CuMem allocator.
- Use the same model revision, tokenizer, dtype, cache layout, block size,
  hash algorithm, tensor-parallel configuration, tenant and `cache_prefix` on
  the instances sharing KV. Start with a dense model and TP=1. Validate other
  model and parallelism combinations separately.

#### Start the external provider

On the provider node, start the Mooncake master as described above. Use its
reachable address, rather than loopback, in both nodes' configuration. For
example, replace `192.0.2.10` with the provider node's address:

Use a dedicated master containing only this EGM provider for the validation
below.

```bash
export PROVIDER_IP=192.0.2.10
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    examples/disaggregated/mooncake_store/egm_provider.py \
    --hostname "${PROVIDER_IP}" \
    --master "${PROVIDER_IP}:50051" \
    --size-gib 64 \
    --numa-nodes auto
```

The [provider example](../../examples/disaggregated/mooncake_store/egm_provider.py)
owns the EGM allocation and stays alive until SIGINT or SIGTERM. Its dictionary
`setup()` explicitly enables `enable_egm_store_pool`, contributes nonzero
`global_segment_size`, and selects `egm_numa_nodes="auto"`. The example uses
`local_buffer_size=0` because this storage-only provider does not need an SDK
client staging buffer. The separate requester retains its own local buffer.
Adjust `--size-gib` to the host-memory budget. It prints
`EGM provider ready` only after successful setup; this confirms initialization,
not successful transfers from a remote requester.

#### Configure and start the requesters

On the requester node, create `mooncake_egm_requester.json`:

```json
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "192.0.2.10:50051",
  "global_segment_size": 0,
  "local_buffer_size": "4GB",
  "protocol": "nvlink",
  "device_name": "",
  "enable_offload": false
}
```

Replace the master address and export the file path in each requester shell:

```bash
export MOONCAKE_CONFIG_PATH="${PWD}/mooncake_egm_requester.json"
```

The requester uses the existing positional Store `setup()` API. Do not put
`enable_egm_store_pool` or `egm_numa_nodes` in the vLLM JSON: only the external
provider owns and configures EGM capacity. Keep the requester's ordinary
positive `local_buffer_size`: vLLM requires it, and it is independent of the
provider's EGM capacity and the GPU KV cache. The existing default is 4 GiB per
rank. Retaining this SDK client buffer does not add CPU staging to the
connector's registered GPU multi-buffer transfers to memory replicas.

Start instance A on requester GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --port 8100 --max-model-len 4096 --enforce-eager \
    --enable-prompt-tokens-details \
    --kv-transfer-config '{
      "kv_connector": "MooncakeStoreConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "custom_mem_pool": "NVLINK",
        "cache_prefix": "egm-demo",
        "lookup_rpc_port": 51001
      }
    }'
```

Start instance B in another shell with the same command and environment,
changing `CUDA_VISIBLE_DEVICES` to `1`, `--port` to `8200`, and
`lookup_rpc_port` to `51002`. The lookup value distinguishes their local IPC
sockets; it must differ for instances on the same host. Keep `cache_prefix`
identical. Use an unused prefix for each validation run and do not send requests
to B before the sharing check.

#### Verify cross-instance reuse and shut down

Keep the provider alive throughout this check. With `jq` installed, create a
prompt longer than one KV block and submit it to A:

```bash
jq -n '{
  model: "Qwen/Qwen2.5-0.5B-Instruct",
  prompt: (("Reusable EGM prefix. " * 256) + "Summarize the prefix."),
  temperature: 0,
  max_tokens: 32,
  logprobs: 1,
  return_token_ids: true
}' > egm-request.json
curl --fail-with-body http://127.0.0.1:8100/v1/completions \
    -H 'Content-Type: application/json' \
    --data-binary @egm-request.json > egm-a.json
curl --fail http://127.0.0.1:8100/metrics > egm-a-metrics.txt
```

Before using B, inspect A's metrics or connector logs for completed
`save_put` operations with `status="ok"`, positive transferred bytes and no
failed keys. Saves are asynchronous; an HTTP response alone does not confirm
the Store PUT has finished. Recheck metrics until those operations finish.
Then send the identical request as B's **first** request:

```bash
curl --fail-with-body http://127.0.0.1:8200/v1/completions \
    -H 'Content-Type: application/json' \
    --data-binary @egm-request.json > egm-b.json
curl --fail http://127.0.0.1:8200/metrics > egm-b-metrics.txt
jq '{usage, token_ids: .choices[0].token_ids}' egm-a.json egm-b.json
```

Require successful `load_get` operations on B as well as positive
`usage.prompt_tokens_details.cached_tokens`. The relevant metrics are
`vllm:mooncake_store_operation_bytes_total` and
`vllm:mooncake_store_operation_failed_keys_total`, labelled by `operation` and
`status`. A fresh B has no local prefix to reuse. Confirm that the provider is
on the other OS node and that SDK logs select NVLink; cache hits alone do not
identify the transport. Compare generated token IDs and log probabilities with
an otherwise identical cold run. Prefix reuse can change floating-point
rounding, so investigate output differences with the existing vLLM correctness
checks rather than treating every greedy-token difference as data corruption.

These steps describe a validation procedure, not hardware or throughput
results. Record the vLLM and Mooncake revisions, hardware, CUDA/driver/IMEX
configuration, output comparisons and PUT/GET evidence for the deployed setup.

For ordinary shutdown, stop sending requests, let outstanding requests and
Store transfers finish, and stop both vLLM instances before stopping the
provider. Send SIGINT or SIGTERM to the provider only after the requesters have
exited. The example calls Store `close()` once on exit and reports a nonzero
return as a cleanup error. Close is terminal: do not retry it on the same
handle after an error. Successful close releases the EGM pool; failed cleanup
requires a process restart. Restarting creates a new pool and requires cold
prefill. This example does not implement eviction-safe provider removal,
restart recovery or forced teardown during transfers.

## Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `MOONCAKE_CONFIG_PATH` | Path to Mooncake JSON config file | (required) |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | Bootstrap port for MooncakeConnector P2P transfer (disagg mode only) | 8998 |
| `MOONCAKE_PREFERRED_SEGMENT` | Pin this rank's replicas to a specific owner segment (`host:port`); used in `standalone-store` mode | — |
| `MOONCAKE_REQUESTER_LOCAL_HOSTNAME` | Override the hostname the vLLM rank registers with Mooncake as a requester. Defaults to the rank's resolved IP. | — |
| `VLLM_MOONCAKE_STORE_TIER_LOG` | When `1`, logs a per-batch tier summary (memory vs disk hits) for observability | disabled |
| `VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO` | Fraction of the owner's DirectIO staging buffer that the requester will fill in a single `batch_get_into_multi_buffers` call. Lower → more conservative pre-split, more round trips. | 0.9 |

## KV Transfer Config

### KV Role Options

- **kv_producer**: For instances that store KV caches to the pool.
- **kv_consumer**: For instances that load KV caches from the pool.
- **kv_both**: The instance both stores and loads KV caches. Use this for single-node CPU offloading or prefiller instances.

### kv_connector_extra_config

- `load_async` (bool): Enable asynchronous loading for better compute-I/O overlap. Default: `true`.
- `lookup_async` (bool): Run the external prefix-cache lookup on a background thread so it never blocks the scheduler step. The request is held until the in-flight lookup completes, then resumed on a later step. Default: `false`.
- `lookup_rpc_port` (int): Custom port for the ZMQ lookup RPC socket. Default: `0`.
- `custom_mem_pool` (str): Optional Mooncake allocator for the GPU KV cache.
  `"NVLINK"` is used in the [EGM deployment example](#standalone-egm-over-nvlink);
  `"BAREX"` selects the BAR-exposed allocator. Unset by default. Both are
  incompatible with sleep mode and the CuMem allocator.
- `cache_prefix` (str): Namespace prepended to every store key. Lets separate deployments share one Mooncake master without polluting each other — instances configured with different prefixes never see each other's cached blocks, even for identical prompts. All instances that should share a prefix cache must use the same value. Default: `""` (no prefix; keys are byte-identical to the unprefixed format).
- `save_decode_cache` (bool): Enable offloading decode tokens' KV cache. A `kv_consumer` does not save during prefill; when decode starts, it fills any missing block-aligned prompt prefix before appending completed decode blocks. Default: `false`.
- `store_tp_size` (int): Common Store TP for endpoints with different local TP sizes. It supports LBHNC and LBNHC local KV caches, with `store_tp_size >= local_tp_size` and `store_tp_size % local_tp_size == 0`. The current topology is one full-attention cache group, PCP/DCP disabled, and cross-layer blocks disabled. For GQA and MHA, the total KV-head count must be divisible by `store_tp_size`. Store shards contain fixed global KV-head ranges in the local layout. Shared endpoints use the same KV cache layout, pipeline-parallel size, and Store TP. The Store namespace includes the layout and PP size. Unsupported configurations use a topology-specific rank-local namespace.

LBHNC/HND is strongly recommended for TP-sharded Store when supported. LBNHC/NHD
creates many transfer segments and may significantly reduce PUT/GET performance.

For example, with prefill TP 4, decode TP 2, and eight KV heads, set
`store_tp_size` to 4 on both instances. Each decode rank reads and writes two
of the four Store shards.

MQA with one total KV head uses a replicated-head layout. For the supported
prefill TP 4 to decode TP 2 case, every rank uses the
same rank-0 key namespace. The four prefill replicas stripe block PUTs so each
object is stored once, while both decode ranks GET every block into their local
KV replica. `store_tp_size` does not appear in MQA keys, so identical MQA
objects written at different store TP sizes share the same pool entry when PP
sizes match.

Tensor-parallel collectives and low-precision arithmetic are not bitwise
invariant across TP sizes, so heterogeneous-TP reuse does not guarantee the
same greedy output as recomputing the prefix at the decode TP size.

## Notes

### Reproducible Block Hashes Across Processes

The `MooncakeStoreConnector` relies on consistent block hashes across all vLLM processes sharing the distributed store. Block hashes chain from `NONE_HASH`, which is derived from a fixed default seed, so identical prompts produce identical block hashes across processes by default — enabling cross-process prefix cache hits without extra configuration.

The exception is the non-cryptographic `xxhash`/`xxhash_cbor` values of `--prefix-caching-hash-algo`, which seed `NONE_HASH` randomly per process; sharing a store with those requires `PYTHONHASHSEED`.

To use a custom shared seed, set the same `PYTHONHASHSEED` on every instance that shares the store (DP ranks, separate prefiller/decoder nodes, and any other vLLM process pointed at the same Mooncake store):

```bash
PYTHONHASHSEED=<shared-value> vllm serve ...
```
