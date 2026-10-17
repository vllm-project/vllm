# Initialized engine snapshots

Initialized engine snapshots are an experimental way to trade local disk space
and host privileges for a faster vLLM activation. Snapshot creation initializes
the engine and records deterministic generation output before CRIU captures the
process tree and CUDA state. Restore validates the saved environment, restores
the engine, binds the HTTP server, and reproduces the recorded token and sampled
token log probability before returning.

This path is intended for repeatedly activating the same model and engine
configuration on the same machine. It is not a portable model artifact.

A snapshot is in place by default: the artifact stays where it was created, the
host keeps the generated caches the engine depends on, and restore refuses to
run once one of them changes. `vllm snapshot create --portable` instead marks
the artifact as one to restore in another container or on another host, and
`vllm snapshot restore` requires `--portable` to match what the manifest
recorded, so neither mode can be mistaken for the other.

A portable create also keeps the run's control files and logs outside the
artifact, in a work directory: one named by `--work-dir`, or otherwise a fresh
random temporary directory. That path is recorded in the manifest and frozen
into the captured process arguments, so every host restoring the artifact
sees the same path while each one holds its own private copy. A work
directory belongs to one running instance, not to the artifact: instances
restoring the same artifact concurrently would collide in a directory on
shared storage such as NFS, and the artifact may be mounted read-only by
the containers that restore it, so create refuses a `--work-dir` inside
the snapshot directory and the artifact itself is never written into; it
can sit on shared, read-only storage. A directory named with `--work-dir`
is kept after create and restore finish, so its logs and control files
remain available for inspection, while a temporary one is removed when the
run that owns it ends, on the create host and on every restore host alike
— the restored tree keeps writing through its open log file, whose bytes
then live on as unlinked inodes. Restore takes the path from the manifest
alone, recreates the directory when it is missing, and clears the control
files an earlier run left behind, so a named work directory is reusable
once its previous restored tree stops. An in-place checkpoint instead
keeps the same state laid out flat inside the artifact, and its restore
reuses and resets those files.

## Requirements

Snapshots currently require:

- Linux on x86-64 with one NVIDIA GPU.
- TP1 with one unauthenticated plaintext HTTP server. Other parallel sizes,
  TLS, middleware, Unix sockets, and speculative decoding are unsupported.
- Because snapshot mode requires an unauthenticated plaintext HTTP server, use
  a trusted host or network boundary, or external controls. See the vLLM
  [Security guide](../usage/security.md).
- [CRIU](https://github.com/checkpoint-restore/criu), its CUDA plugin, a
  `cuda-checkpoint`-compatible helper, and `nvidia-smi` on `PATH`.
- Root or passwordless `sudo` for CRIU.
- `io_uring` disabled before launch because CRIU cannot dump it. Use
  `kernel.io_uring_disabled=1` for an unprivileged process. A process running
  as root, including the `docker exec` flow below, bypasses `=1`, so use `=2`
  host-wide.
- No established TCP connection to a peer outside the captured process tree.
  Current Hugging Face hub clients hold their connections for the process
  lifetime, so download the model in a separate step and run create with
  `HF_HUB_OFFLINE=1`, as the quickstart below does.
- Snapshot creation defaults `NCCL_IB_DISABLE=1` for its singleton donor and
  inherited workers because CRIU cannot capture live InfiniBand/RDMA state.
  An explicit caller value is retained, but creation rejects an open
  `/dev/infiniband/` descriptor before CRIU. Close non-NCCL RDMA clients before
  capture.
- A remote model ID and an immutable 40-character `--revision`. Local model
  directories and mutable revisions are not supported.
- Enough disk for the artifact, with the same installed vLLM package, model
  files, container filesystem, and generated-cache paths available at restore.
  The manifest fingerprints the generated-cache files the captured tree keeps
  open, so restore fails early and names the file when one is removed or
  replaced.
- With `--portable` the artifact also carries its own copy of them: create
  stages every file the tree reaches through a descriptor or a file-backed
  mapping under a resolved generated-cache root, plus every other regular
  file those roots hold as warm state, and restore reproduces the paths that
  are missing. A path holding different content is refused rather than
  overwritten, so a cache root shared with another engine still needs the
  same state. A lock file the tree holds open is carried like any other
  referenced file, and CRIU replays its advisory lock on restore. The
  per-file and total carry size limits default to 1 GiB each; set
  `VLLM_SNAPSHOT_ATTACHMENT_MAX_FILE_BYTES` or
  `VLLM_SNAPSHOT_ATTACHMENT_MAX_TOTAL_BYTES` to change them. A create that
  exceeds a limit fails while the inventory is assembled, before CRIU runs.
- `/var/lib/libuuid/clock.txt` is the same kind of dependency with different
  semantics: libuuid keeps it open for the process lifetime (a time-based-UUID
  watermark), so CRIU reopens its path, but its content is machine state. A
  portable artifact carries it as warm state — written back only when
  missing, never fingerprinted — so a target host keeps its own watermark.
- The child log is carried with the same semantics: CRIU reopens its path,
  but the work directory holding it does not travel with the artifact, so a
  portable artifact carries the log as warm state and restore writes it back
  when the directory has no log to reset.

The official CUDA 13 `vllm/vllm-openai` Linux x86-64 images include the
snapshot runtime. CUDA 12.x and Arm64 images omit it. A compatible host
driver, kernel, and privileges are still required. Source installs must set
`CRIU_CUDA_PLUGIN_DIR` to the directory containing `cuda_plugin.so`.

Run snapshot commands with `docker exec` inside a long-lived container. Restore
hands the API server off as a detached process, so a one-shot container would
stop that server when its PID 1 exits. Snapshot preflight requires every
component of the artifact path to be owned by the invoking user or root, with
the directory itself at mode 0700, and the commands below run as root inside
the official image, so the bind-mounted host directory is created with `sudo`
and root ownership. The model downloads in its own step so the captured tree
holds no hub connection, and create runs offline. This example also keeps the
container filesystem and `/dev/shm` namespace stable for the lifetime of the
artifact:

```bash
sudo sysctl kernel.io_uring_disabled=2

snapshot_root="$(pwd)/vllm-snapshots"
sudo install -d -m 0700 -o root -g root "${snapshot_root}"

docker run --detach --name vllm-snapshot \
  --gpus all \
  --privileged \
  --pid=host \
  --ipc=host \
  --network=host \
  --mount "type=bind,source=${snapshot_root},target=/snapshots" \
  --entrypoint sleep \
  vllm/vllm-openai:latest infinity

docker exec vllm-snapshot hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca

docker exec -e HF_HUB_OFFLINE=1 vllm-snapshot vllm snapshot create Qwen/Qwen3-0.6B \
  --snapshot-dir /snapshots/qwen3-0.6b \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --dtype float16 \
  --max-model-len 512

docker exec vllm-snapshot vllm snapshot inspect /snapshots/qwen3-0.6b

docker exec vllm-snapshot vllm snapshot restore \
  /snapshots/qwen3-0.6b --host 0.0.0.0 --port 8000
```

Keep that container and its mounts available while the snapshot is in use.
Stop and remove it only after the restored API server is no longer needed.

Create initializes the engine, records a one-token canary, releases and reloads
weights and KV cache to rehearse restore, then releases them again for capture.
The manifest is published only after CRIU completes and the
source tree stops. Once the images are written create stages the generated
files the image still reopens by path; the process that compiled them no longer
exists, so nothing can rewrite those bytes while they are copied.
Creation is offline preparation and is not part of restore latency.

The private `0700` artifact contains process memory, CUDA state, engine
arguments, compatibility identity, canary output, and the generated files
the image reopens by path; its manifest is `0600`. Literal API keys and
Hugging Face tokens are redacted from the manifest's engine arguments. The
manifest records selected environment names plus deterministic name-bound
fingerprints, not their values. The protected CRIU artifact can still
contain process secrets, so treat it as sensitive data. Restore reloads
model files and KV cache before binding HTTP, then reproduces the canary or
tears down the restored tree. The inspect command prints that identity and
canary without executing the saved process.

## Restore behavior

Restore fails before CRIU runs if the saved identity does not match the current
host. It does not silently fall back to ordinary startup. Before CRIU runs, the
paths the artifact carries are reproduced from it: a missing file is written
back, a file that exists with the captured content is left alone, and a file
whose content differs fails the restore instead of being overwritten. Anything
that step wrote is removed again when the restore later fails. After CRIU
restores the process tree, vLLM releases the saved engine to bind the requested
HTTP address and checks the first generated token and sampled-token log
probability against the snapshot canary. The command returns only after that
check passes. The restored API server continues to run as a detached process.

The pre-release port probe is best-effort only. It neither reserves the port
nor authenticates the listener that appears afterward.

Rollback terminates and waits for the restored tree after process identity
verification succeeds. If verification fails, vLLM writes an abort marker for
the snapshot server instead of signaling unverified PIDs. If that server has
already exited, surviving engine processes can require operator cleanup.
Verify that the previous tree has stopped before retrying; do not kill a process
solely because its PID appears in the manifest, since PIDs can be reused.

The artifact is reusable after its previous restored tree stops. Only one
snapshot or external CRIU operation may use a shared `/dev/shm` mount at a time.

## Tradeoffs and limitations

- Creation has its own latency and briefly requires the full engine. Artifact
  size can approach the captured process and GPU memory.
- Artifacts are not guaranteed to survive a power loss. The manifest is
  fsynced, but publication does not explicitly flush all captured payload files
  and directories. Recreate the artifact after an unclean host shutdown.
- Restore currently requires the same host, GPU, driver, kernel, Python,
  PyTorch, installed vLLM version, model revision, engine arguments,
  selected effective environment variables, and CRIU plugin binaries. An unset
  `NCCL_IB_DISABLE` therefore matches the snapshot donor default of `1`, while
  an explicit different value does not.
- Only dense float16 TP1 has been validated. Other model formats depend on their
  existing sleep level 2 reload support; distributed and RDMA snapshots are not
  supported. The snapshot-only NCCL default does not change ordinary serve or
  multi-GPU defaults.
- CRIU support varies by kernel and driver. Preserve the package, library, and
  model paths for the artifact lifetime. The generated files the image reopens
  travel in the artifact, so clearing that cache no longer invalidates it, but
  restore still needs to write them back where it finds nothing, and a path it
  finds holding other content stops the restore.
- A snapshot can include application secrets or request state present in the
  process. Build it before traffic and protect it like process memory.
- Any feature that opens an external connection must close it before capture.

For a lower-complexity option that retains a live process, see
[Sleep mode](sleep_mode.md). Sleep mode and initialized snapshots retain
different amounts of state and have different idle resource costs.
