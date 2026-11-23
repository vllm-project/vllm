# Preload

vLLM's preload feature keeps model artifacts resident in GPU memory across
engine restarts, so a restarting engine reuses them instead of rebuilding them
from scratch. The design is extensible to different kinds of artifacts; it
currently supports **model weights** via the weight cache daemon.

With weight preloading, a daemon process per GPU holds its rank's
post-quantized, TP-sharded weights and serves local sharing handles to vLLM
engines over a Unix domain socket, so a restarting engine maps the weights via
zero-copy sharing instead of reloading them from disk.

Key benefits:

- **Fast engine restarts**: weight loading reduces to mapping existing GPU
  memory, cutting restart time from minutes (disk load + quantization
  processing) to seconds.
- **Zero-copy sharing**: the engine directly shares the daemon's GPU memory,
  so a restart adds no extra GPU memory footprint.
- **Safe fallback**: if the daemon is unavailable or its cached weights don't
  match the engine's configuration, the engine falls back to loading from
  disk.

!!! note
    Weight preloading is only supported on CUDA and ROCm platforms. Tensor,
    expert and data parallelism are supported, including across nodes;
    pipeline parallelism is rejected when launching the daemon.

## Quick start

Launch one weight cache daemon per GPU with a single command:

```bash
vllm preload --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4
```

The daemon process loads, post-processes and warms up the weights once, then
publishes their final state and waits for engines. Each rank binds its Unix
socket only after that state is ready, so engines connecting before then simply
fall back to disk loading.

Then start (or restart) engines with the `ipc_cache` load format:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 4 \
    --load-format ipc_cache
```

The daemon itself must load from disk (or seed from a peer replica, see
below); passing `--load-format ipc_cache` or `peer_cache` to `vllm preload`
is an error.

## Local sharing backends

The preload daemon's `--weight-cache-backend` option selects how it holds and
exports GPU weights:

| Backend | Storage and sharing | Platforms |
| --- | --- | --- |
| `torch_ipc` (default) | PyTorch CUDA IPC handles over the local Unix socket | CUDA and ROCm |
| `vmm_ipc` | CUDA/HIP VMM allocations exported as POSIX file descriptors | CUDA and ROCm on Linux |

For example:

```bash
vllm preload --model /path/to/model --weight-cache-backend vmm_ipc
vllm serve /path/to/model --load-format ipc_cache
```

The `ipc_cache` loader does not configure the backend again through
`--model-loader-extra-config`. It negotiates with the daemon and accepts the
backend named in its response.

A backend mismatch follows the ordinary cache-mismatch policy: disk fallback
when enabled, an error otherwise. New engines can connect to existing CUDA IPC
daemons; VMM daemons reject engines that do not advertise VMM support.

After loading, post-processing and warmup, the daemon calls `build_manifest`
to record the final canonical parameters and buffers as logical views over
distinct storages, including any tensors created or replaced during warmup. It
then materializes the storages, drops the model and releases startup runtime
resources. VMM moves non-empty GPU storages into exportable `cuMemCreate` or
`hipMemCreate` allocations. By default, consecutive storages are packed in
input order toward a 1 GiB logical cap, with 512-byte alignment between logical
storages. An
oversized storage gets its own allocation, and every allocation is rounded up
to the driver's granularity. Set `--weight-cache-vmm-max-allocation-bytes` to
change the cap, or to `0` to use one allocation per storage. Each pack is copied
and synchronized separately, avoiding retention of a second complete model.
VMM handles only allocation and FD export/import; the manifest preserves the
logical layout, including tied parameters, shared storages, offsets and
strides. Each `StorageRecord` holds `owner`, `storage` and `byte_offset`. The
manifest records each logical storage's `nbytes`, while
`record.storage.nbytes()` reports the physical size of its potentially shared
allocation. Granularity padding counts once per unique owner in the
external-weight memory budget. CPU and empty storages continue to use the
existing by-value transfer path.

`WeightCacheState` carries `manifest`, `storages`, `aliases` and `attrs` on both
sides; the producer and consumer differ only in where they use the same state
type. The daemon captures its state before starting listeners or reporting
readiness. For each `get_state` consumer, the daemon's backend exports every
storage in the state once. CUDA IPC still creates fresh per-consumer handle
reductions. VMM entries hold `SharedFileDescriptor` objects: the existing
pickle persistent-ID protocol encodes their references, while `SCM_RIGHTS`
passes the actual FDs over the owner-verified Unix socket, never as serialized
process-local integers. The consumer calls `rebuild_entries` to obtain
`StorageRecord`s, deduplicating imports for entries that reference the same
descriptor so each packed allocation is imported only once. It then constructs
`WeightCacheState` and rebuilds all tensor views from the manifest. The daemon's
state owns the producer allocations; the returned tensors and records keep the
consumer's imported mappings alive. A connected engine does not need the daemon
to stay alive to retain those mappings, but future engines need a live daemon
to fetch them. Zero-copy loading works with either backend. Same-host peer
seeding detects VMM storage and transfers its descriptors; remote seeding
continues to use its configured copy mover. VMM uses the existing FD transport
and D2D protocols; this PR does not rewrite either one.

Both backends can be deployed independently of the serving Pod, provided GPU
access, socket mounts, ownership and container settings allow it. VMM does not
introduce that deployment capability. The daemon still loads the model and
initializes the GPU; this backend does not implement a separate GPU-free FD
holder. It also does not add EPLB or sleep-mode support.

The built-in `torch_ipc` and `vmm_ipc` backends are the only statically supported
backends. They implement `IPCBackendBase`, whose `materialize(storages)` returns
`StorageRecord`s and whose `export_storage(record)` returns a rebuildable
`StorageEntry`. Tensor reconstruction remains separate from the model's
processed-weight contract and daemon-to-daemon copy movers.

## Preloading the FlashInfer autotune cache

FlashInfer has several implementations of each operation and chooses between
them by benchmarking. vLLM runs that pass during kernel warmup and persists the
result per rank in the on-disk FlashInfer autotune cache (under
`VLLM_CACHE_ROOT`, or `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`), so a later engine
with the same configuration loads the tuned configs instead of profiling again.
Normally the first serving engine pays for the pass, and on a large MoE model it
dominates what is left of the startup time once the weights come from the
daemon.

Unless FlashInfer autotune is disabled (`--disable-flashinfer-autotune`),
`vllm preload` pays it at preload time: each daemon runs the autotune pass in
place — profiling dummy runs against the shard it already holds — after
loading and before binding its socket, then writes the cache:

```bash
vllm preload --model /path/to/model --tensor-parallel-size 4
```

After warmup and state capture, the daemon releases its communication resources
and, on CUDA, cached cuBLAS workspaces before accepting engine connections. The
state's storages remain resident for sharing through the selected backend.
This reduces the daemon's GPU memory overhead; the amount reclaimed depends on
the model and allocator state.

### Seeding warmup caches between daemons (D2D)

A fresh daemon replica — a new node joining a fleet, a restarted preload
set — would normally pay the autotune pass itself. With a seed source
configured, each daemon rank fetches its rank-matched warmup caches from the
source replica over an authenticated TCP control plane before tuning,
validates them, and installs them into its own on-disk cache, so its warmup
skips the profiling sweep. Any failure (unreachable, missing, incompatible)
falls back to local tuning.

Source replica:

```bash
MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json \
vllm preload --model /path/to/model --tensor-parallel-size 4 \
    --seed-listen-host 0.0.0.0 --seed-listen-port 29600 \
    --seed-token "$TOKEN" --warmup-cache-stores mooncake
```

Mirror replica (possibly on another node):

```bash
MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json \
vllm preload --model /path/to/model --tensor-parallel-size 4 \
    --seed-source-host <source-host> --seed-source-port 29600 \
    --seed-token "$TOKEN" --warmup-cache-stores mooncake
```

Each rank binds and dials `base_port + group offset + its global rank`, so one
`host:base_port` pair covers a whole single-node replica, including the draft
daemon group when speculative decoding is configured. The standalone
`--seed-source-host` option names one source host, so it cannot by itself reach
every node of a multi-node source. When mirroring weights from a multi-node
source, the per-node `--weight-cache-seed` addresses below also provide
per-node warmup cache sources. The Store options in these examples enable
compiled-cache reuse; autotune-only D2D seeding does not require them.

Compiled-cache reuse requires a live D2D source and the same Store
configuration on source and target. After warmup, the source publishes a
content-addressed compiled bundle as
`vllm-warmup-cache/compile_cache/<bundle-sha256>/bundle.tar`. Its authenticated
D2D response carries the cache manifest, bundle key, byte length and SHA256
digest, not the bundle itself. The target fetches the bundle from the Store,
verifies its length and digest, and installs it before model loading. A Store
miss or error, an invalid or incompatible bundle, or a D2D failure falls back
to local compilation. A Store without a source cannot provide a compiled-cache
cold start.

The source scans compiled caches only when a shared Store is configured. If
collecting or publishing the bundle fails, for example because the cache exceeds
the 8 GiB limit, the source logs a warning and continues serving model weights.
It advertises no compiled bundle, so targets use local compilation. Invalid
Store configuration and Store initialization errors remain startup errors.

For autotune tables, the exchange is keyed by the on-disk cache layout: a
request names the cache by kind, configuration hash, DP rank and world rank,
and the source resolves it to files under its cache root. Before installing,
the target rejects a payload whose FlashInfer environment metadata (GPU,
CUDA/cuDNN stack) differs from its own — such a table would never load anyway,
and a hard conflict is never overwritten, which would poison the cache file
into re-tuning on every start. Tables are shareable across vLLM versions as
long as the configuration hash matches.

The channel is a generic seeding point: warmup caches are the first kind, and
further daemon state (e.g. Triton warmup caches or weight manifests) plugs in
as additional kinds. Small JSON payloads and bundle descriptors travel over
the authenticated control plane; compiled bundles are fetched separately from
the configured shared Store.

## Seeding a replica from another

A second replica's daemons can fill their shards from a replica that is
already loaded, instead of reading the checkpoint again. A mirror daemon is
an ordinary weight cache daemon: it forms its own world group and rendezvous
like any preload (the multi-node `--weight-cache-master-port` requirements
apply unchanged) and builds the model itself, so the model's config files
must be present locally — only the safetensors are never read. What changes
is where the weights come from: the daemon's loader pulls the
post-processed, TP-sharded weights directly from its rank-matched source
daemon. The FlashInfer autotune table does not travel with the weights;
instead each mirror daemon adopts it from the same source through the
ordinary [warmup-cache seeding](#seeding-warmup-caches-between-daemons-d2d)
path — with `--weight-cache-seed` set and no explicit `--seed-source-*`,
each daemon automatically seeds its warmup caches from its weights source,
so the new replica's first engine also skips the autotune pass. If the
source has no cache to send (autotune disabled, or the single-GPU managed
store), the mirror tunes in place and fills its local cache instead, like
any other daemon.

The address form selects the mover: Unix socket paths copy through
CUDA IPC on one host (`peer_ipc`):

```bash
# source replica on GPUs 0-3, already serving
vllm preload --model /path/to/model --tensor-parallel-size 4 \
    --weight-cache-socket-dir /run/vllm-a

# mirror replica, filled from the source (it lands on the free GPUs 4-7)
vllm preload --model /path/to/model --tensor-parallel-size 4 \
    --weight-cache-socket-dir /run/vllm-b \
    --weight-cache-seed /run/vllm-a/vllm_weight_cache_<hash0>.sock,/run/vllm-a/vllm_weight_cache_<hash1>.sock,/run/vllm-a/vllm_weight_cache_<hash2>.sock,/run/vllm-a/vllm_weight_cache_<hash3>.sock
```

`--weight-cache-seed` takes one source target socket per local GPU, in device
order, and rank *i* of the mirror is filled from rank *i* of the source — the
cache fingerprints are compared first, so a mismatched pairing is rejected
rather than silently mirroring the wrong shard. With speculative decoding,
always pass the target sockets; each draft daemon automatically uses the
corresponding `*_draft.sock` path instead. The socket paths live in the
source's `--weight-cache-socket-dir` and are named from each GPU's UUID, as
described under [Loader configuration](#loader-configuration). The mirror
fetches each source rank's manifest over that same owner-verified Unix
socket, so no token or network listener is involved. A mirror on a host
without the checkpoint's safetensors cannot fingerprint them, so it matches
the source by `--model` path instead, as an engine without the files matches
its daemon; give both sides the same `--model` in that case.

D2D seeding copies each source storage once, transferring only the logical
`nbytes` prefix recorded in the manifest rather than the full physical
allocation. The mirror then calls `rebuild_tensors` to rebuild every view at
its recorded offset and strides, so views that post-processing leaves on one
storage (such as MLA's `W_UV` and `W_UK_T`) keep their layout and stay shared.

A same-host mirror keeps every GPU visible — a CUDA IPC handle names its
device by index, so the source's indices must resolve — and the launcher
places the mirror's ranks on the GPUs the source does not use: it learns the
source's GPU UUIDs from its sockets and takes the remaining indices in order,
so no placement flag is needed and an arbitrary source placement (not just a
leading block) works. Do not remap devices with `CUDA_VISIBLE_DEVICES`; the
mirror refuses a handle that would land on the wrong physical GPU.

`rdma` pulls through the Mooncake TransferEngine and is the mover for a
source on another host. The source has to expose its daemon-to-daemon
listener, and both sides need the same shared token:

```bash
# single-node source host
vllm preload --model /path/to/model --tensor-parallel-size 8 \
    --seed-listen-host 0.0.0.0 --seed-listen-port 29700 \
    --seed-token "$TOKEN"

# single-node mirror host
vllm preload --model /path/to/model --tensor-parallel-size 8 \
    --weight-cache-seed 10.0.0.1:29700 --seed-token "$TOKEN"
```

For a multi-node source, pass one `host:base_port` per source node, comma
separated in `node_rank` order (for example,
`10.0.0.1:29700,10.0.0.2:29700`). A single `host:base_port` remains the
single-node form. The source and mirror must be homogeneous: the same node
count, parallel layout and rank placement, so every mirror rank has a
rank-matched source. Each mirror node selects the address at its own
`node_rank`, then each rank dials `base_port + group offset + its global
rank`. The draft daemon group's listeners sit past the target group's port
block on every source node, so the selected address seeds the draft group
too. An IPv6 source is written `[address]:base_port`. Do not mix remote
addresses with Unix socket paths in one list. The remote form pulls over
Mooncake RDMA and requires `--seed-token`.

!!! warning
    The remote control plane accepts JSON seed requests after authenticating
    the shared token; it never uses pickle. The connection is unencrypted, so
    use it only on a trusted network and keep the token secret. Exporting IPC
    handles remains a local operation over the owner-verified Unix socket.

## Sharing warmup caches through Mooncake (optional)

For autotune tables, pluggable shared stores are an alternative to D2D seeding
and let a new node fetch them even when no source replica is online.
`--warmup-cache-stores` takes a comma-separated list of store backends; the
`mooncake` backend uses a
[Mooncake](https://github.com/kvcache-ai/Mooncake) distributed store. Point
`MOONCAKE_CONFIG_PATH` at the same JSON configuration used by
[MooncakeStoreConnector](mooncake_store_connector_usage.md#configure-mooncake):

```bash
MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json \
vllm preload --model /path/to/model --tensor-parallel-size 4 \
    --warmup-cache-stores mooncake
```

After tuning, each daemon publishes its per-rank autotune table to every
configured store under keys of the form
`vllm-warmup-cache/<kind>/<config-hash>/<filename>`, in addition to writing
its local disk cache. All tiers share one interface: the local disk is always
tier zero, and the stores listed in `--warmup-cache-stores` follow it in list
order — reads take the first tier that hits, writes fan out to all of them.
During warmup a daemon pulls its table from the shared stores first, in list
order; when no store has it — or none are configured — it falls back to D2D
seeding from a live source daemon, and finally tunes in place. Stores come
first because they are built for read fan-out, while a D2D fetch makes a busy
peer proxy the same bytes. For autotune tables, this covers a cold start with
no live source daemon and nodes whose cache root is ephemeral. When serving
peers, a daemon answers from the same chain: disk first, then the stores.
Whatever it fetches is validated and installed to the local disk cache as
usual. A misspelled store name or a store that cannot be configured fails
daemon startup immediately; at runtime, store operations are best-effort and
never block warmup.

Prefer the `standalone-store` topology for this use case. In `embedded` mode
every daemon contributes `global_segment_size` of host memory to the pool,
which is hard to justify for kilobyte-sized JSON artifacts. Note that the
store is a trust boundary: any writer in the Mooncake cluster can influence
the autotune tables and compiled bundles other nodes load, so only point
daemons at a trusted deployment.

## How it works

1. `vllm preload` spawns one daemon process per GPU. Each daemon loads and
   post-processes its shard, runs warmup, then publishes a `WeightCacheState`
   containing a manifest of the final canonical parameters and buffers and
   their storage records through the selected sharing backend (CUDA IPC by
   default). It drops the model and startup runtime resources before accepting
   engine connections.
2. An engine started with `--load-format ipc_cache` builds its model on the
   meta device (no storage) and requests the storage entries and state metadata
   from the daemon on its GPU. It rebuilds each entry into a `StorageRecord`,
   constructs `WeightCacheState`, then uses the manifest to recreate all tensor
   views.
3. Before serving anything, the engine and daemon compare a fingerprint of the
   cached weights: checkpoint content (hashed from safetensors metadata, so
   identical weights in different directories still match), model
   architecture, TP/DP size/rank, dtype, quantization method and config, text
   RoPE parameters, model revision, KV cache dtype and quantization skip layers,
   the multimodal options that skip model components, and vLLM version. On any
   mismatch the engine falls back to disk loading (unless `fallback` is
   disabled).
4. The engine also verifies the daemon's GPU UUID matches its own device, so a
   stale socket cannot serve weights for the wrong GPU.

Tied weights (e.g. `lm_head.weight` sharing storage with
`embed_tokens.weight`) share a storage export; the manifest and aliases
re-establish their views and parameter identity in the engine.

Shared weights count toward the engine's `gpu_memory_utilization` budget.
If IPC loading falls back to disk, the engine removes the corresponding external
weight charge before profiling its locally loaded weights for KV cache sizing.
The updated budget must still fit the available GPU memory. Engines never
actively release the daemon's cache, and fallback does not stop the daemon:
its cached weights remain resident until the daemon shuts down.

## Multi-node and data parallelism

Local backend handles are node-local, so each node serves only its local GPUs'
shards. For multi-node tensor parallelism, run one `vllm preload` launcher per
node with a shared rendezvous: reuse the `--nnodes` / `--node-rank` /
`--master-addr` flags you pass the engine, plus a `--weight-cache-master-port`
distinct from the engine's `--master-port`:

```bash
# node 0 (8 local GPUs)
vllm preload --model /path/to/model --tensor-parallel-size 16 \
    --nnodes 2 --node-rank 0 --master-addr 10.0.0.1 \
    --weight-cache-master-port 29600
# node 1 (8 local GPUs)
vllm preload --model /path/to/model --tensor-parallel-size 16 \
    --nnodes 2 --node-rank 1 --master-addr 10.0.0.1 \
    --weight-cache-master-port 29600
```

For data parallelism (e.g. a TP1 x DP16 x EP decode fleet), run one launcher
per node with the engine's DP placement flags. Local GPU `i` serves DP rank
`start_rank + i // tp_size` and TP rank `i % tp_size`, and all
`dp_size * tp_size` daemons form one world group on `--data-parallel-address`
/ `--weight-cache-master-port` so the expert shards are laid out exactly as in
the engine:

```bash
# node r (4 local GPUs)
vllm preload --model /path/to/model --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 16 --data-parallel-size-local 4 \
    --data-parallel-start-rank 4r --data-parallel-address 10.0.0.1 \
    --weight-cache-master-port 29600
```

Data parallelism also combines with multi-node tensor parallelism: pass both
flag sets. Each node then serves a contiguous block of the
`dp_size * tp_size` global ranks.

## Speculative decoding

With MTP, EAGLE or EAGLE3 speculative decoding, `vllm preload` additionally
starts a draft daemon group that caches the draft model. It uses its own cache
key, Unix sockets (`*_draft.sock`) and rendezvous port
(`--weight-cache-draft-master-port`, default `--weight-cache-master-port + 1`),
so each daemon process serves exactly one model role. Other draft types are
not cached and keep loading from disk in the engine. With pipeline
parallelism, only the GPUs of the last stage serve the draft model, since the
engine runs the draft model there; draft daemons on the other stages exit
after startup.

## Loader configuration

The `ipc_cache` loader supports only zero-copy loading of new model instances;
standalone in-place weight reload is not supported. The engine maps the
daemon's shared allocations directly through the selected backend, so each GPU
carries only one copy of the weights. Already-mapped engines keep serving if
the daemon exits, but a restarted daemon holds a second copy and new engines
cannot attach until it is ready.

The loader accepts extra keys via `--model-loader-extra-config`:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --load-format ipc_cache \
    --model-loader-extra-config '{"fallback": false}'
```

| Key | Default | Description |
| --- | ------- | ----------- |
| `socket_path` | per-GPU path derived from the GPU UUID | Explicit daemon socket path. |
| `socket_dir` | per-user private dir under the temp dir | Directory containing the daemon sockets. |
| `fallback` | `true` | Fall back to disk loading when the daemon is unavailable or the fingerprints mismatch. |
| `connect_timeout_s` | `5.0` | Socket connect timeout in seconds. |
| `state_timeout_s` | `300.0` | Timeout for the weight-transfer request in seconds. |

The daemon's `--weight-cache-socket-dir` and the loader's `socket_dir` /
`socket_path` must agree when the default per-user directory is not used.
Socket paths are derived from the GPU UUID, so they are stable regardless of
`CUDA_VISIBLE_DEVICES` index remapping.

!!! warning
    The weights live in externally held backend allocations, so
    [sleep mode](sleep_mode.md) (CuMemAllocator weight offloading) must not be
    used with this loader.

## Limitations

- **Platform**: CUDA and ROCm only; other platforms raise
  `UnsupportedPlatformForIPCError` even when `fallback` is enabled, since it
  is a permanent misconfiguration rather than a transient daemon outage.
- **Parallelism**: tensor, expert, pipeline and data parallelism are
  supported, including across nodes.
- **Autotune preloading**: tuning runs across the daemon's world group,
  including across nodes and DP ranks. With model-based speculative decoding
  both daemon groups skip it (the target cannot run the drafter's dummy
  passes, and a table tuned on the draft model would key on the draft's
  config, which no engine looks up), so those deployments' first engine
  tunes and fills the cache instead.
- **Cache seeding**: autotune seeding supports only the v1 per-rank FlashInfer
  cache; single-rank daemons use FlashInfer's managed v2 store and always tune
  in place. Compiled-cache seeding additionally requires a live D2D source and
  matching Store configuration on source and target; a Store alone cannot
  cold-start compiled caches. Seeding requires a trusted network — the TCP
  channel is authenticated by a shared token but unencrypted. Shared stores
  can serve autotune tables without a live source. Cache misses and errors
  fall back to local tuning or compilation.
- **Weight seeding**: `peer_ipc` cannot cross hosts, because local sharing
  handles are node-local; `rdma` requires Mooncake on both sides. A mirror
  copies the weights into its own memory, so it does not depend on the
  source staying alive afterwards, and a mirror replica forms its own world
  group like a normal preload. Each outstanding copy holds a lease until the
  mirror acknowledges it. During graceful shutdown, the source waits for these
  acknowledgements before weight memory is freed, subject to the timeout under
  [Daemon lifecycle](#daemon-lifecycle).
- **Autotuning cannot overlap the load**: autotuning picks a tactic by timing
  candidates, so running it while a daemon is still copying weights onto the
  same GPU would measure the wrong thing and cache the wrong choice. To pay
  the download once per machine instead, prefetch FlashInfer's kernels
  separately with `python -m flashinfer download-cubin` (it fetches the whole
  cubin set, so do it when building an image, not on a serving node).
- **Quantization**: every quantization method in the model must declare
  support for pre-processed weights (the daemon transfers weights *after*
  quantization post-processing). Unsupported methods raise
  `UnsupportedQuantForIPCError`.
  Online `fp8_per_tensor`, `fp8_per_channel`, and `fp8_per_block` dense
  linear layers support CUDA IPC with the CUTLASS linear backend. Set
  `--kernel-config '{"linear_backend": "cutlass"}'` on both the daemon and
  engine; the block variant additionally requires a GPU supported by the
  CUTLASS block-FP8 kernel. Online `mxfp8` linear layers can reuse weights
  with an MXFP8 kernel that supports pre-processed weights, including
  FlashInfer CUTLASS on Blackwell.
  The engine reuses the quantized weights and scales without running online
  quantization again. This does not enable online MoE quantization or
  conversion from an already quantized checkpoint.
  Serialized `compressed-tensors` W8A8 FP8 dense linear layers support
  per-tensor and per-channel weights with static per-tensor or dynamic
  per-token activations when both processes select the CUTLASS linear
  backend. Engines retain the processed tensor layout and scales and rebuild
  attention scale metadata from the cached scale buffers. Other
  compressed-tensors schemes, transform wrappers, block FP8 and MoE are
  not enabled by this support.
  Serialized ModelOpt `FP8` (per-tensor) and `FP8_PER_CHANNEL_PER_TOKEN`
  linear layers also support CUDA IPC with the CUTLASS linear backend.
  Set the same `--kernel-config '{"linear_backend": "cutlass"}'` on the daemon
  and engine. Engines reuse the processed weights and scales, including
  fused-projection scales and output padding. This does not enable ModelOpt
  block FP8, FP8 MoE, or other FP8 linear backends.
  Serialized ModelOpt `INT8_PB_WO` linear layers can also reuse pre-processed
  weights with the Marlin backend and FP16/BF16 activations. Humming and
  ModelOpt INT8 MoE layers are not covered.
  Serialized `compressed-tensors` W8A8 FP8 linear layers with 128×128
  weight blocks and dynamic per-token group-128 activations support CUDA IPC
  when both processes select the CUTLASS block-FP8 backend on a supported GPU.
  Set `--kernel-config '{"linear_backend": "cutlass"}'` on the daemon and
  engine. Attached weights retain their untransposed layout. Other
  compressed-tensors
  schemes, block sizes, transform wrappers, linear backends and MoE are not
  enabled by this support.
  Serialized `compressed-tensors` WNA16 linear layers can reuse Marlin-packed
  weights with FP16/BF16 activations. Humming, other WNA16 backends and MoE
  are not enabled by this support.
  ModelOpt NVFP4 MoE supports the TRT-LLM and vLLM CUTLASS backends. With
  CUTLASS, the daemon fuses activation scales once; engines reuse the fused
  scales without modifying the daemon's shared tensors.
- **Consistency**: the daemon and engine must run the same vLLM version and
  agree on model, dtype, quantization, and TP/DP layout, otherwise the
  fingerprint mismatch triggers the disk fallback. For multimodal models, also
  pass the same `--language-model-only`, `--mm-encoder-only`, and zero
  `--limit-mm-per-prompt` entries to both: they decide which components (such
  as the vision tower) are built. Nonzero limits may differ.
- **Localhost only**: the daemon serves over a Unix domain socket; both the
  daemon and the engines must run on the same node as the same user.

## Security

The socket protocol uses pickle and is intended only for trusted local
processes owned by the same user:

- Daemon sockets live in a per-user private directory (mode `0700`) and the
  socket files are restricted to the owner (`0600`).
- Both sides reject symlinked or non-owned socket paths; the auto-derived
  directory is additionally rejected if it is group/world accessible.
- On Linux the daemon verifies the connecting peer's UID via `SO_PEERCRED`.

The optional daemon-to-daemon listener (`--seed-listen-*`) is a
separate TCP channel: requests are JSON-only and size-capped, authenticate
with the shared `--seed-token` (constant-time comparison), and
an unauthenticated peer that stops sending is cut off by a whole-message
deadline instead of blocking the accept loop. The channel is unencrypted —
bind it to a trusted network.

The optional shared stores (`--warmup-cache-stores`) trust the store
they are pointed at: a writer in the store's cluster can influence the
autotune tables and compiled bundles a daemon installs, so use them only with
a trusted deployment.

See the [security documentation](../usage/security.md) for vLLM's general
threat model.

## Daemon lifecycle

- Each GPU's socket path is guarded by an exclusive lock file, so a second
  daemon for the same GPU fails fast instead of clobbering the live socket.
- `vllm preload` reports readiness once every rank is serving. If any rank
  exits during startup or serving, the launcher terminates and reaps the
  remaining ranks, stops its health endpoint, and exits with a nonzero status.
  The failing rank's nonzero exit code is preserved. Signal failures use
  `128 + signal number` (for example, 137 for `SIGKILL`); an unexpected clean
  rank exit is reported as status 1. Restart the entire preload group before
  attaching new engines.
- `SIGINT`/`SIGTERM` to the `vllm preload` process terminates all daemon
  ranks and exits with status 130/143.
- During graceful shutdown, a daemon serving as a D2D weight source stops
  accepting new copy leases and waits up to five minutes for outstanding
  copies to acknowledge before releasing its weight memory. The launcher
  allows 30 seconds beyond this lease timeout before killing and reaping ranks
  that have not stopped. When shutdown follows a rank failure, the launcher
  instead allows ten seconds. A forcibly terminated rank may leave a stale
  socket; the next daemon replaces it after acquiring the per-GPU lock.

See [vllm preload](../cli/preload.md) for the full CLI reference.
