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

Snapshots currently require:

- Linux on x86-64 with one NVIDIA GPU.
- TP1 with one unauthenticated plaintext HTTP server. Other parallel sizes,
  TLS, middleware, Unix sockets, and speculative decoding are unsupported.
- [CRIU](https://github.com/checkpoint-restore/criu), its CUDA plugin, a
  `cuda-checkpoint`-compatible helper, and `nvidia-smi` on `PATH`.
- Root or passwordless `sudo` for CRIU.
- `io_uring` disabled before launch because CRIU cannot dump it. Use
  `kernel.io_uring_disabled=1` for an unprivileged process or `=2` host-wide.
- No established TCP connection to a peer outside the captured process tree.
- A remote model ID and an immutable 40-character `--revision`. Local model
  directories and mutable revisions are not supported.
- Enough disk for the artifact, with the same installed vLLM package, model
  files, container filesystem, and generated-cache paths available at restore.

The official `vllm/vllm-openai` Linux x86-64 image includes the snapshot
runtime. It still requires a compatible host driver, kernel, and privileges.
Arm64 images omit it. Source installs must set `CRIU_CUDA_PLUGIN_DIR` to the
directory containing `cuda_plugin.so`.

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

docker exec vllm-snapshot vllm snapshot inspect /snapshots/qwen3-0.6b

docker exec vllm-snapshot vllm snapshot restore \
  /snapshots/qwen3-0.6b --host 0.0.0.0 --port 8000
```

Keep that container and its mounts available while the snapshot is in use.
Stop and remove it only after the restored API server is no longer needed.

Create initializes the engine, records a one-token canary, releases and reloads
weights and KV cache to rehearse restore, then releases them again for capture.
The manifest is published only after CRIU completes and the source tree stops.
Creation is offline preparation and is not part of restore latency.

The private `0700` artifact contains process memory, CUDA state, engine
arguments, compatibility identity, and canary output; its manifest is `0600`.
Treat it as sensitive data. Restore reloads model files and KV cache before
binding HTTP, then reproduces the canary or tears down the restored tree.
The inspect command prints that identity and canary without executing the saved
process.

## Restore behavior

Restore fails before CRIU runs if the saved identity does not match the current
host. It does not silently fall back to ordinary startup. After CRIU restores
the process tree, vLLM releases the saved engine to bind the requested HTTP
address and checks the first generated token and sampled-token log probability
against the snapshot canary. The command returns only after that check passes.
The restored API server continues to run as a detached process.

The artifact is reusable after its previous restored tree stops. Only one
snapshot or external CRIU operation may use a shared `/dev/shm` mount at a time.

## Tradeoffs and limitations

- Creation has its own latency and briefly requires the full engine. Artifact
  size can approach the captured process and GPU memory.
- Restore currently requires the same host, GPU, driver, kernel, Python,
  PyTorch, installed vLLM version, model revision, engine arguments,
  selected environment variables, and CRIU plugin binaries.
- Only dense float16 TP1 has been validated. Other model formats depend on their
  existing sleep level 2 reload support; NCCL state is not restored.
- CRIU support varies by kernel and driver. Preserve package, library, model,
  and generated-cache paths for the artifact lifetime.
- A snapshot can include application secrets or request state present in the
  process. Build it before traffic and protect it like process memory.
- Any feature that opens an external connection must close it before capture.

For a lower-complexity option that retains a live process, see
[Sleep mode](sleep_mode.md). Sleep mode and initialized snapshots retain
different amounts of state and have different idle resource costs.
