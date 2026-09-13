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
    Weight preloading is only supported on CUDA and ROCm platforms, and only
    with tensor/expert parallelism. Pipeline and data parallelism are rejected
    when launching the daemon.

## Quick start

Launch one weight cache daemon per TP rank with a single command:

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

The daemon itself must load from disk; passing `--load-format ipc_cache` to
`vllm preload` is an error.

## How it works

1. `vllm preload` spawns one daemon process per TP rank. Each daemon loads its
   shard from disk using the configured loader, runs quantization
   post-processing, and exports every parameter/buffer as a CUDA IPC handle.
2. An engine started with `--load-format ipc_cache` builds its model on the
   meta device (no storage) and requests the tensors from the daemon on its
   GPU.
3. Before serving anything, the engine and daemon compare a fingerprint of the
   cached weights: checkpoint content (hashed from safetensors metadata, so
   identical weights in different directories still match), model
   architecture, TP size/rank, dtype, quantization method and config, model
   revision, and vLLM version. On any mismatch the engine falls back to disk
   loading (unless `fallback` is disabled).
4. The engine also verifies the daemon's GPU UUID matches its own device, so a
   stale socket cannot serve weights for the wrong GPU.

Tied weights (e.g. `lm_head.weight` sharing storage with
`embed_tokens.weight`) are exported once and re-established as aliases in the
engine, preserving parameter identity.

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
| `socket_path` | per-GPU path derived from the physical GPU id | Explicit daemon socket path. |
| `socket_dir` | per-user private dir under the temp dir | Directory containing the daemon sockets. |
| `mode` | `zero_copy` | `zero_copy` or `copy` (see above). |
| `fallback` | `true` | Fall back to disk loading when the daemon is unavailable or the fingerprints mismatch. |
| `connect_timeout_s` | `5.0` | Socket connect timeout in seconds. |
| `state_timeout_s` | `300.0` | Timeout for the weight-transfer request in seconds. |

The daemon's `--weight-cache-socket-dir` and the loader's `socket_dir` /
`socket_path` must agree when the default per-user directory is not used.

!!! note
    When `CUDA_VISIBLE_DEVICES` contains GPU UUIDs instead of integer indices,
    the engine cannot infer the physical GPU id; pass `socket_path` (or
    `socket_dir`) explicitly via `--model-loader-extra-config`.

## Limitations

- **Platform**: CUDA and ROCm only; other platforms raise
  `UnsupportedPlatformForIPCError` even when `fallback` is enabled, since it
  is a permanent misconfiguration rather than a transient daemon outage.
- **Parallelism**: tensor and expert parallelism only; launching the daemon
  with pipeline or data parallelism is rejected.
- **Quantization**: every quantization method in the model must declare
  support for pre-processed weights (the daemon transfers weights *after*
  quantization post-processing). Unsupported methods raise
  `UnsupportedQuantForIPCError`.
- **Consistency**: the daemon and engine must run the same vLLM version and
  agree on model, dtype, quantization, and TP layout, otherwise the
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

See the [security documentation](../usage/security.md) for vLLM's general
threat model.

## Daemon lifecycle

- Each GPU's socket path is guarded by an exclusive lock file, so a second
  daemon for the same GPU fails fast instead of clobbering the live socket.
- `vllm preload` reports readiness once every rank is serving; if any rank
  dies during startup, the remaining ranks are terminated and the command
  exits with that rank's exit code.
- `SIGINT`/`SIGTERM` to the `vllm preload` process terminates all daemon
  ranks, which unlinks their sockets and releases the GPU memory.

See [vllm preload](../cli/preload.md) for the full CLI reference.
