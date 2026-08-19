# Initialized engine snapshots

Initialized engine snapshots are an experimental way to trade local disk space
and host privileges for a faster vLLM activation. Snapshot creation initializes
the engine and records deterministic generation output before CRIU captures the
process tree and CUDA state. Restore validates the saved environment, restores
the engine, binds the HTTP server, and reproduces the recorded token and sampled
token log probability before returning.

This path is intended for repeatedly activating the same model and engine
configuration on the same machine. It is not a portable model artifact.

## Requirements

Snapshots currently require all of the following:

- Linux on x86-64 with one NVIDIA GPU.
- Tensor, pipeline, data, and prefill-context parallel size 1.
- One plaintext TCP HTTP API server without built-in API authentication,
  custom middleware, TLS, or a Unix domain socket.
- No established TCP connections to a peer outside the captured process tree.
  Snapshot creation inspects tree-owned connections and rejects external peers.
- [CRIU](https://github.com/checkpoint-restore/criu), its matching CUDA plugin,
  and a `cuda-checkpoint`-compatible helper installed in the runtime.
- `criu`, `cuda-checkpoint`, and `nvidia-smi` available on `PATH`.
- Either root execution or passwordless `sudo` for CRIU.
- `io_uring` disabled for the vLLM process before snapshot creation because
  CRIU cannot dump active `io_uring` descriptors. For an unprivileged vLLM
  process without an `io_uring_group` exemption, set
  `kernel.io_uring_disabled=1`. Use `kernel.io_uring_disabled=2` to disable it
  host-wide, including for privileged processes.
- An installed vLLM build or a clean editable Git checkout. Snapshot creation
  and restore reject tracked or untracked source changes because a commit SHA
  alone cannot identify their contents.
- Enough local disk for the captured process and CUDA state. Compact capture is
  the default, so the recorded model files must remain available for weight
  reload after restore. `--include-model-state` captures the initialized model
  and KV state instead.
- Compact capture requires a remote model ID and an immutable 40-character
  `--revision`. Use `--include-model-state` for local model directories or
  mutable revisions.
- The container filesystem and generated-cache files referenced by the captured
  process must remain available at the same paths. CRIU restores held file
  locks, but the snapshot does not copy every regular file or file-backed
  mapping into the artifact. Rotating a referenced JIT cache invalidates the
  snapshot.

The official `vllm/vllm-openai` Linux x86-64 image includes pinned CRIU, its
matching CUDA plugin, and a source-built helper based on NVIDIA Dynamo's
Apache-2.0 implementation of the public CUDA Driver checkpoint APIs. The
container still requires a compatible host driver and kernel plus the CRIU
privileges described above. Arm64 images do not include this experimental
runtime.

Run snapshot commands with `docker exec` inside a long-lived container. Restore
hands the API server off as a detached process, so a one-shot container would
stop that server when its PID 1 exits. This example also keeps the container
filesystem and `/dev/shm` namespace stable for the lifetime of the artifact:

```bash
snapshot_root="$(pwd)/vllm-snapshots"
install -d -m 0700 "${snapshot_root}"

docker run --detach --name vllm-snapshot \
  --gpus all \
  --privileged \
  --pid=host \
  --ipc=host \
  --network=host \
  --mount "type=bind,source=${snapshot_root},target=/snapshots" \
  --entrypoint sleep \
  vllm/vllm-openai:latest infinity

docker exec vllm-snapshot vllm snapshot create Qwen/Qwen3-0.6B \
  --snapshot-dir /snapshots/qwen3-0.6b \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --dtype float16 \
  --max-model-len 512

docker exec vllm-snapshot vllm snapshot restore \
  /snapshots/qwen3-0.6b --host 0.0.0.0 --port 8000
```

Keep that container and its mounts available while the snapshot is in use.
Stop and remove it only after the restored API server is no longer needed.

For wheel or source installations, install those dependencies separately and
set `CRIU_CUDA_PLUGIN_DIR` to the directory containing `cuda_plugin.so`:

```bash
export CRIU_CUDA_PLUGIN_DIR=/path/to/criu/plugins
```

## Create a snapshot

Pass normal `vllm serve` engine arguments after the model name:

```bash
vllm snapshot create Qwen/Qwen3-0.6B \
  --snapshot-dir /var/lib/vllm/snapshots/qwen3-0.6b \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --dtype float16 \
  --max-model-len 512
```

The command initializes the engine and records a restore-fidelity canary. In the
default compact mode, it then releases the model-weight and KV-cache
allocations, reloads them, reruns the canary, and requires the result to match
before releasing those allocations again for capture. The artifact is
published by an atomic manifest write only after the CRIU dump completes, and
the source process tree is then stopped. Snapshot creation is an offline
preparation step and is not part of the restore latency.

The artifact contains process memory, CUDA state, engine arguments, a recorded
software and hardware compatibility identity, and the canary output. This is a
strict compatibility check over the recorded fields, not a cryptographic proof
of every installed binary. Treat the artifact as sensitive data. vLLM creates
the snapshot directory with mode `0700` and its manifest with mode `0600`.

### Include full model state

Compact mode is the default. It reloads weights from the recorded model files
and recreates the KV cache after restore, before binding the HTTP server. Those
model files must remain available for every restore.

Use the full-state escape hatch `--include-model-state` when restore should avoid
reloading model weights from those files, the model source is local or mutable,
or compact rehearsal fails:

```bash
vllm snapshot create Qwen/Qwen3-0.6B \
  --snapshot-dir /var/lib/vllm/snapshots/qwen3-0.6b-full \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --dtype float16 \
  --max-model-len 512 \
  --include-model-state
```

This mode captures the initialized model and KV state. Compact mode currently
rejects speculative decoding; `--include-model-state` permits it.

The tradeoff depends on storage state. Warm model-file pages can make the
compact artifact both cheaper to store and faster to activate, while cold or
slow storage can make weight reload slower than restoring the full-state
process image. Measure both modes with the page-cache state that represents the
target deployment.

Compact mode has been validated on dense float16 TP1 models. Other model
formats depend on their existing sleep level 2 weight-reload support and have
not been validated by this snapshot path. The restore command checks the first
generated token and sampled-token log probability against the snapshot canary
and tears down the restored process tree if the result differs.

## Inspect a snapshot

Inspection does not restore or execute the saved process:

```bash
vllm snapshot inspect /var/lib/vllm/snapshots/qwen3-0.6b
```

The JSON output includes the source and binary revisions, model revision,
Python and CUDA environment, GPU identity, snapshot size, process inventory,
and recorded canary token and log probability.

## Restore a snapshot

```bash
vllm snapshot restore /var/lib/vllm/snapshots/qwen3-0.6b \
  --host 127.0.0.1 \
  --port 8000
```

Restore fails before CRIU runs if the saved identity does not match the current
host. It does not silently fall back to ordinary startup. After CRIU restores
the process tree, vLLM releases the saved engine to bind the requested HTTP
address and checks the first generated token and sampled-token log probability
against the snapshot canary. The command returns only after that check passes.
The restored API server continues to run as a detached process.

The same artifact may be restored again after the previous restored process
tree has stopped. Only one initialized-snapshot create or restore operation may
use a shared `/dev/shm` mount at a time; do not overlap an external CRIU
workflow that owns, creates, or consumes `link_remap.*` entries there.

## Tradeoffs and limitations

- Snapshot creation has its own latency and temporarily requires the full
  initialized engine. It is useful only when that cost can be amortized.
- Artifact size can be comparable to the process and GPU memory captured. Disk
  capacity and disk bandwidth directly affect the result.
- Restore currently requires the same host, GPU, driver, kernel, Python,
  PyTorch, vLLM source, installed vLLM binary, model revision, engine arguments,
  selected environment variables, and CRIU plugin binaries.
- The current boundary supports TP1 only. It does not restore NCCL or other
  multi-GPU communicator state.
- CRIU and CUDA checkpoint support varies by kernel and driver. A successful
  artifact on one host is not evidence that another host is compatible.
- Package files, shared libraries, and generated cache entries opened or mapped
  by the process are part of the same-host compatibility boundary. Preserve the
  container filesystem and cache roots for the lifetime of the artifact.
- A snapshot can include application secrets or request state present in the
  process. Build it before serving user traffic and protect it like model
  weights and process memory.
- Any feature that opens an external connection must close it before capture.

For a lower-complexity option that retains a live process, see
[Sleep mode](sleep_mode.md). Sleep mode and initialized snapshots retain
different amounts of state and have different idle resource costs.
