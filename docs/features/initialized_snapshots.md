# Initialized engine snapshots

Initialized engine snapshots are an experimental way to trade local disk space
and host privileges for a faster vLLM activation. Snapshot creation initializes
the engine and records deterministic generation output before CRIU captures the
process tree and CUDA state. Restore validates the saved environment, restores
the engine, and checks the recorded token and sampled-token log probability
before binding HTTP. The controller repeats the check through the public API
before returning.

This path is intended for repeatedly activating the same model and engine
configuration on the same machine. It is not a portable model artifact.

## Requirements

The `vllm snapshot create/restore` CLI currently requires:

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
  Generated-cache files are not copied into the artifact. The manifest
  fingerprints the ones the captured tree holds open, so restore fails early
  and names the file when one is removed or replaced. Only open descriptors
  are recorded. A library the tree mapped and then closed is not fingerprinted,
  and CRIU still reopens it by path, so replacing that file permanently
  invalidates the artifact without an early error.

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
Preparation and recovery also await the existing communicator checkpoint hooks.
The manifest is published only after CRIU completes and the source tree stops.
Creation is offline preparation and is not part of restore latency.

The private `0700` artifact contains process memory, CUDA state, engine
arguments, compatibility identity, and canary output; its manifest is `0600`.
Literal API keys and Hugging Face tokens are redacted from the manifest's engine
arguments. The manifest records selected environment names plus deterministic
name-bound fingerprints, not their values. The protected CRIU artifact can
still contain process secrets, so treat it as sensitive data. Restore reloads
model files and KV cache, then validates the canary before binding HTTP.
The inspect command prints that identity and canary without executing the saved
process.

## Restore behavior

Restore fails before CRIU runs if the saved identity does not match the current
host. It does not silently fall back to ordinary startup. After CRIU restores
the process tree, vLLM completes communicator and memory recovery and checks
the snapshot canary on the private engine. Only then does it bind the requested
HTTP address. The command returns after a second canary check through HTTP.
The restored API server continues to run as a detached process.

A failed engine preparation or recovery terminates the attempt without opening
HTTP. The artifact's `error.json` records the phase and error before worker
cleanup, since the restored process's launch-log streams are detached. Each
activation clears the previous recovery error before attempting recovery.

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
- CRIU support varies by kernel and driver. Preserve package, library, model,
  and generated-cache paths for the artifact lifetime.
- A snapshot can include application secrets or request state present in the
  process. Build it before traffic and protect it like process memory.
- Any feature that opens an external connection must close it before capture.

For a lower-complexity option that retains a live process, see
[Sleep mode](sleep_mode.md). Sleep mode and initialized snapshots retain
different amounts of state and have different idle resource costs.

## External startup capture with ordinary serve

The ordinary Python launcher also has an opt-in startup path for an external
capturer. It prepares the engine before exposing it to request producers,
waits for native capture and release, and completes recovery and a private
canary before binding HTTP. It then uses the normal application, including
API-key authentication, routes and serving options. Ordinary startup without
`--snapshot-config` retains its existing behavior.

This is a startup extension to the `--snapshot-config` proposal in
[#54940](https://github.com/vllm-project/vllm/issues/54940). It does not expose
live-server suspend/resume routes. The configuration and file protocol are
experimental. The bounded profile is Linux x86-64 CUDA, a single Python HTTP
frontend, TP=PP=DP=1, and a dense unquantized generation model loaded with
`auto` or `safetensors`. It uses compact level-2 sleep and weight reload.
Ray, other launchers, Unix sockets, offloading, hybrid/multimodal/MoE models,
LoRA, speculation and KV/EC transfer are unsupported. The launcher sets the
default worker method to `spawn` and `NCCL_IB_DISABLE=1` before worker creation.
An explicit conflicting native configuration still requires capturer validation.

The capturer creates an empty `0700` directory, owned by the serving user, at a
stable absolute path. Its ancestors must satisfy the private-path requirements
above. For example, after preparing the immutable model files in a separate
process:

```bash
install -d -m 0700 /run/vllm-startup
HF_HUB_OFFLINE=1 vllm serve Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --dtype float16 --max-model-len 512 --api-key example-key \
  --snapshot-config '{"mode":"startup","control_dir":"/run/vllm-startup","timeout_s":300}'
```

Use the ordinary launch command instead of a custom `AsyncLLM` application.
The external capturer remains responsible for native dump/restore and must
implement the following file exchange, using atomic replacement for its writes:

1. Wait for `capture-ready.json`. It contains protocol `version: 1`, a
   `capture_id`, the frontend `pid` and the private `oracle`. vLLM writes this
   only after preparation and recovery rehearsal complete. No public socket
   has been bound. Capture the complete process tree at this barrier.
2. After native CUDA restore/unlock completes, write `activation.json`. Use
   the recorded capture ID, a new activation ID and an explicit `kind` of
   `donor` or `copy`. Do not infer source disposition from a PID or captured
   environment variable. Supply the public listener address:

    ```json
    {
      "version": 1,
      "capture_id": "the-id-from-capture-ready",
      "activation_id": "replica-002",
      "kind": "copy",
      "native_complete": true,
      "host": "0.0.0.0",
      "port": 8000
    }
    ```

3. Read `challenge.json`. vLLM generates its nonce after reading the activation
   request, so the challenge is not part of the captured image. Verify the
   capture/activation IDs, then copy the complete challenge object into
   `release.json`. Leave the activation request unchanged. Only a matching
   response permits engine recovery.
4. Probe normal HTTP readiness after recovery. `status.json` records
   `recovering` and then `validated`; neither value means the listener is
   active. A failed phase writes `error.json` before terminating the owned
   engine. Treat that attempt as terminal and preserve the native logs.

Provide fresh control contents for each copy at the same path. Do not carry
`challenge.json`, `release.json`, `status.json` or `error.json` from another
activation. Those files are rejected, and a release from a previous copy does
not match the fresh challenge. Repeated completed waits return the recorded
activation; overlapping or failed waits cannot restart recovery.

`timeout_s` bounds each engine preparation/recovery phase and each file polling
budget. Frozen process time is governed by the capturer's independent deadline.
A timeout or cancellation keeps serving unavailable; it cannot roll back a
partially completed worker RPC. The launcher terminates its owned engine
without draining. The controller must also bound native operations and clean up
its resources. There is no fallback from partial recovery.

The capturer must preserve all required model, library and generated-cache
files, shared-memory files and internal endpoint addresses. With CRIU 4.2.1
`--leave-running`, preserve required filesystem state in the `post-dump` hook:
CRIU removes temporary link remaps before returning to its caller. Collecting
them after dump returns can leave an image that cannot reopen semaphore
mappings. Restore those saved files into the activation's private mounts
before native restore, preserving hard-link relationships so reopened named
semaphores share the captured state. For same-host TP1 reuse, disposable
PID/network/IPC namespaces with loopback internal addresses avoid collisions
with a continuing or recently stopped donor. Configure and inventory the
actual endpoints before capture; merely setting a new namespace or public
HTTP port does not reconstruct internal transports. A stopped donor can leave
TCP reservations behind. This adapter neither sleeps through those reservations
nor retries native restoration.

Standard streams remain attached. The capturer must arrange per-activation
logs, for example with CRIU external-file/inherited descriptors. CRIU 4.2.1
identifies a regular external file as `file[mount_id:inode]` using hexadecimal
numbers. Record and check namespace firewall rules around native operations.
Restore placeholders must remain inert: no model, engine or GPU initialization
before the captured process tree is restored.

The file protocol trusts the capturer's assertion of native completion; it does
not implement artifact compatibility, storage or placement. This path is not
qualification of the Kubernetes Snapshot provider, concurrent clones, TP2 or
worker relocation. Authentication follows ordinary serving semantics; see the
[Security guide](../usage/security.md) for its endpoint coverage.
