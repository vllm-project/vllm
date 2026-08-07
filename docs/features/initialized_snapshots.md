# Initialized engine snapshots

Initialized engine snapshots are an experimental way to trade local disk space
and host privileges for a faster vLLM activation. Snapshot creation initializes
the engine and verifies one generated token before CRIU captures the process
tree and CUDA state. Restore validates the saved environment, restores the
engine, binds the HTTP server, and verifies the same token before returning.

This path is intended for repeatedly activating the same model and engine
configuration on the same machine. It is not a portable model artifact.

## Requirements

Snapshots currently require all of the following:

- Linux on x86-64 with one NVIDIA GPU.
- Tensor, pipeline, and data parallel size 1.
- One plaintext TCP HTTP API server without built-in API authentication,
  custom middleware, TLS, or a Unix domain socket.
- [CRIU](https://github.com/checkpoint-restore/criu),
  [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint), and NVIDIA's
  CRIU CUDA plugin installed on the host.
- `criu`, `cuda-checkpoint`, and `nvidia-smi` available on `PATH`.
- Either root execution or passwordless `sudo` for CRIU.
- Enough local disk for the complete process and CUDA snapshot.

Set `CRIU_CUDA_PLUGIN_DIR` to the directory containing `cuda_plugin.so`:

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

The command initializes the engine, runs a correctness canary, writes the
snapshot, and stops the source process tree. The artifact is published by an
atomic manifest write only after the CRIU dump completes. Snapshot creation is
an offline preparation step and is not part of the restore latency.

The artifact contains process memory, CUDA state, engine arguments, a recorded
software and hardware compatibility identity, and the canary output. This is a
strict compatibility check over the recorded fields, not a cryptographic proof
of every installed binary. Treat the artifact as sensitive data. vLLM creates
the snapshot directory with mode `0700` and its manifest with mode `0600`.

## Inspect a snapshot

Inspection does not restore or execute the saved process:

```bash
vllm snapshot inspect /var/lib/vllm/snapshots/qwen3-0.6b
```

The JSON output includes the source and binary revisions, model revision,
Python and CUDA environment, GPU identity, snapshot size, process inventory,
and expected canary token.

## Restore a snapshot

```bash
vllm snapshot restore /var/lib/vllm/snapshots/qwen3-0.6b \
  --host 127.0.0.1 \
  --port 8000
```

Restore fails before CRIU runs if the saved identity does not match the current
host. It does not silently fall back to ordinary startup. After CRIU restores
the process tree, vLLM releases the saved engine to bind the requested HTTP
address and checks the first generated token against the snapshot canary. The
command returns only after that check passes. The restored API server continues
to run as a detached process.

The same artifact may be restored again after the previous restored process
tree has stopped. Concurrent restores of one artifact are not supported.

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
- A snapshot can include application secrets or request state present in the
  process. Build it before serving user traffic and protect it like model
  weights and process memory.

For a lower-complexity option that retains a live process, see
[Sleep mode](sleep_mode.md). Sleep mode and initialized snapshots retain
different amounts of state and have different idle resource costs.
