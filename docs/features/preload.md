# Preload

vLLM's preload feature keeps model artifacts resident in GPU memory across
engine restarts, so a restarting engine reuses them instead of rebuilding them
from scratch. The design is extensible to different kinds of artifacts; it
currently supports **model weights** via the weight cache daemon.

With weight preloading, a daemon process per GPU holds its rank's
post-quantized, TP-sharded weights and serves CUDA IPC handles to vLLM engines
over a Unix domain socket, so a restarting engine maps the weights via
zero-copy IPC instead of reloading them from disk.

Key benefits:

- **Fast engine restarts**: weight loading reduces to mapping existing GPU
  memory, cutting restart time from minutes (disk load + quantization
  processing) to seconds.
- **Zero-copy sharing**: in the default `zero_copy` mode the engine directly
  shares the daemon's GPU memory, so a restart adds no extra GPU memory
  footprint.
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

The daemon process loads and post-processes the weights once, then waits for
engines. Each rank binds its Unix socket only after its shard is fully cached,
so engines connecting before that simply fall back to disk loading.

Then start (or restart) engines with the `ipc_cache` load format:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 4 \
    --load-format ipc_cache
```

The daemon itself must load from disk (or seed from a peer replica, see
below); passing `--load-format ipc_cache` or `peer_cache` to `vllm preload`
is an error.

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

After warmup, the daemon releases its communication resources and, on CUDA,
cached cuBLAS workspaces before accepting engine connections. The model weights
remain resident for CUDA IPC sharing. This reduces the daemon's GPU memory
overhead; the amount reclaimed depends on the model and allocator state.

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
vllm preload --model /path/to/model --tensor-parallel-size 4 \
    --seed-listen-host 0.0.0.0 --seed-listen-port 29600 \
    --seed-token "$TOKEN"
```

Mirror replica (possibly on another node):

```bash
vllm preload --model /path/to/model --tensor-parallel-size 4 \
    --seed-source-host <source-host> --seed-source-port 29600 \
    --seed-token "$TOKEN"
```

Each rank binds and dials `base_port + group offset + its global rank`. The
standalone `--seed-source-host` option names one source host, so it addresses a
single-node source replica; it cannot by itself reach every node of a
multi-node source. When mirroring weights from a multi-node source, the
per-node `--weight-cache-seed` addresses below also provide per-node warmup
cache sources.

The exchange is keyed by the on-disk cache layout: a request names the cache
by kind, configuration hash, DP rank and world rank, and the source resolves
it to files under its cache root. Before installing, the target rejects a
payload whose FlashInfer environment metadata (GPU, CUDA/cuDNN stack) differs
from its own — such a table would never load anyway, and a hard conflict is
never overwritten, which would poison the cache file into re-tuning on every
start. Tables are shareable across vLLM versions as long as the configuration
hash matches.

The channel is a generic seeding point: warmup caches are the first kind, and
further daemon state (e.g. Triton warmup caches or weight manifests) plugs in
as additional kinds — small JSON payloads travel inline, while large binary
state rides the data plane.

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

The mirror copies each of the source's allocations once and rebuilds every
tensor on it at the source's offset and strides, so views that post-processing
leaves on one allocation (such as MLA's `W_UV` and `W_UK_T`) keep their layout
and stay shared, and the mirror uses as much memory as the source.

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
    The seed listener is a network service. It is JSON only and
    authenticates before it decodes a payload, so an unauthenticated peer
    cannot reach the daemon's pickle protocol, but it is **unencrypted and
    unauthenticated beyond the shared token**. Only read-only seed requests
    are served remotely; exporting IPC handles and releasing weights stay on
    the local, owner-verified Unix socket. Seed peers are served on their own
    thread and get 10 seconds to send their request, so a stalled or
    unauthenticated peer cannot hold up the engines loading from the daemon.
    Keep the listener on a trusted network and treat the token as a secret.

## Sharing warmup caches through Mooncake (optional)

D2D seeding needs a live source daemon. As an alternative, the daemon can
share its warmup caches through pluggable shared stores, so a new node can
fetch them even when no source replica is online. `--warmup-cache-stores`
takes a comma-separated list of store backends; the `mooncake` backend uses
a [Mooncake](https://github.com/kvcache-ai/Mooncake) distributed store.
Point `MOONCAKE_CONFIG_PATH` at the same JSON configuration used by
[MooncakeStoreConnector](mooncake_store_connector_usage.md#configure-mooncake):

```bash
MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json \
vllm preload --model /path/to/model --tensor-parallel-size 4 \
    --warmup-cache-stores mooncake
```

After tuning, each daemon publishes its per-rank autotune table to every
configured store under keys of the form
`vllm-warmup-cache/<kind>/<config-hash>/<filename>`, in addition to writing
its local disk cache. All tiers share one interface: the local disk is
always tier zero, and the stores listed in `--warmup-cache-stores`
follow it in list order — reads take the first tier that hits, writes fan
out to all of them. During warmup a daemon pulls its table from the shared stores first, in
list order; when no store has it — or none are configured — it falls back
to D2D seeding from a live source daemon, and finally tunes in place.
Stores come first because they are built for read fan-out, while a D2D
fetch makes a busy peer proxy the same bytes. This covers a cold start
with no live source daemon and nodes whose cache root is ephemeral. When
serving peers, a daemon answers from the same chain: disk first, then the
stores. Whatever it fetches is validated and installed to the local disk
cache as usual. A misspelled store name or a store that cannot be
configured fails daemon startup immediately; at runtime, store operations
are best-effort and never block warmup.

Prefer the `standalone-store` topology for this use case. In `embedded` mode
every daemon contributes `global_segment_size` of host memory to the pool,
which is hard to justify for kilobyte-sized JSON artifacts. Note that the
store is a trust boundary: any writer in the Mooncake cluster can influence
the autotune tables other nodes load, so only point daemons at a trusted
deployment.

## How it works

1. `vllm preload` spawns one daemon process per GPU. Each daemon loads its
   shard from disk using the configured loader, runs quantization
   post-processing, and exports every parameter/buffer as a CUDA IPC handle.
2. An engine started with `--load-format ipc_cache` builds its model on the
   meta device (no storage) and requests the tensors from the daemon on its
   GPU.
3. Before serving anything, the engine and daemon compare a fingerprint of the
   cached weights: checkpoint content (hashed from safetensors metadata, so
   identical weights in different directories still match), model
   architecture, TP/DP size/rank, dtype, quantization method and config, text
   RoPE parameters, model revision, KV cache dtype and quantization skip layers,
   and vLLM version. On any mismatch the engine falls back to disk loading
   (unless `fallback` is disabled).
4. The engine also verifies the daemon's GPU UUID matches its own device, so a
   stale socket cannot serve weights for the wrong GPU.

Tied weights (e.g. `lm_head.weight` sharing storage with
`embed_tokens.weight`) are exported once and re-established as aliases in the
engine, preserving parameter identity.

Shared weights count toward the engine's `gpu_memory_utilization` budget.
If IPC loading falls back to disk, the engine removes the corresponding external
weight charge before profiling its locally loaded weights for KV cache sizing.
The updated budget must still fit the available GPU memory; fallback does not
stop the daemon or release its cached weights in zero-copy mode.

## Multi-node and data parallelism

CUDA IPC handles are node-local, so each node serves only its local GPUs'
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
not cached and keep loading from disk in the engine.

## Cache modes

The loader supports two modes, selected via `--model-loader-extra-config`:

- `zero_copy` (default): the engine maps the daemon's CUDA allocations
  directly. The daemon must stay alive for the engine's lifetime, and each
  GPU carries only one copy of the weights.
- `copy`: the engine clones every tensor into its own GPU memory and then
  asks the daemon to release its cache. Use this when the daemon should free
  GPU memory after handing off, at the cost of a full copy per restart.

!!! warning
    In `zero_copy` mode the weights live in the daemon's CUDA IPC allocations,
    so [sleep mode](sleep_mode.md) (CuMemAllocator weight offloading) must not
    be used with this loader.

## Loader configuration

The `ipc_cache` loader accepts extra keys via `--model-loader-extra-config`:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --load-format ipc_cache \
    --model-loader-extra-config '{"mode": "copy", "fallback": false}'
```

| Key | Default | Description |
| --- | ------- | ----------- |
| `socket_path` | per-GPU path derived from the GPU UUID | Explicit daemon socket path. |
| `socket_dir` | per-user private dir under the temp dir | Directory containing the daemon sockets. |
| `mode` | `zero_copy` | `zero_copy` or `copy` (see above). |
| `fallback` | `true` | Fall back to disk loading when the daemon is unavailable or the fingerprints mismatch. |
| `connect_timeout_s` | `5.0` | Socket connect timeout in seconds. |
| `state_timeout_s` | `300.0` | Timeout for the weight-transfer request in seconds. |

The daemon's `--weight-cache-socket-dir` and the loader's `socket_dir` /
`socket_path` must agree when the default per-user directory is not used.
Socket paths are derived from the GPU UUID, so they are stable regardless of
`CUDA_VISIBLE_DEVICES` index remapping.

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
- **Warmup cache seeding**: only the v1 per-rank FlashInfer autotune cache is
  seeded; single-rank daemons use FlashInfer's managed v2 store and always
  tune in place. Seeding requires a trusted network — the TCP channel is
  authenticated by a shared token but unencrypted. The optional shared
  stores (`--warmup-cache-stores`) serve the same tables without a live
  source daemon.
- **Weight seeding**: `peer_ipc` cannot cross hosts, because a CUDA IPC
  handle is node-local; `rdma` requires Mooncake on both sides. A mirror
  copies the weights into its own memory, so it does not depend on the
  source staying alive afterwards, and a mirror replica forms its own world
  group like a normal preload. Source release and graceful shutdown wait for
  outstanding copies to acknowledge before weight memory is freed.
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
  Serialized ModelOpt `FP8` (per-tensor) and `FP8_PER_CHANNEL_PER_TOKEN`
  linear layers also support CUDA IPC with the CUTLASS linear backend.
  Set the same `--kernel-config '{"linear_backend": "cutlass"}'` on the daemon
  and engine. Engines reuse the processed weights and scales, including
  fused-projection scales and output padding. This does not enable ModelOpt
  block FP8, FP8 MoE, or other FP8 linear backends.
  ModelOpt NVFP4 MoE supports the TRT-LLM and vLLM CUTLASS backends. With
  CUTLASS, the daemon fuses activation scales once; engines reuse the fused
  scales without modifying the daemon's shared tensors.
- **Consistency**: the daemon and engine must run the same vLLM version and
  agree on model, dtype, quantization, and TP/DP layout, otherwise the
  fingerprint mismatch triggers the disk fallback.
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
autotune tables a daemon installs, so use them only with a trusted
deployment.

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
- In either shutdown path, ranks that do not stop within ten seconds are
  killed and reaped, releasing their GPU memory. A forcibly
  terminated rank may leave a stale socket; the next daemon replaces it
  after acquiring the per-GPU lock.

See [vllm preload](../cli/preload.md) for the full CLI reference.
