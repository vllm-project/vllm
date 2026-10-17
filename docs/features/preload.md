# Preload

vLLM's preload feature keeps model weights in GPU memory across engine restarts. A weight cache daemon per GPU loads and post-processes the weights once and serves them to engines over a Unix domain socket. An engine started with `--load-format ipc_cache` maps them through CUDA IPC instead of loading them from disk, which cuts the weight-loading part of a restart from minutes to seconds. If no daemon is reachable, or its weights do not match the engine's configuration, the engine loads from disk as usual.

## Requirements

- **Platform**: CUDA or ROCm. Other platforms raise `UnsupportedPlatformForIPCError`, even when fallback is enabled.
- **Same node and user**: the daemon serves over a Unix domain socket, and CUDA IPC handles work only on the node that created them. The daemon and the engine must run as the same user.
- **Same vLLM version** on the daemon and the engine.
- **Quantization**: every quantization method in the model must support loading pre-processed weights. Unquantized models, except MoE models with `--all2all-backend moonep`, and `fp8`, ModelOpt NVFP4 and MXFP4 checkpoints, except GPT-OSS, are supported. Other methods, such as compressed-tensors, GPTQ and AWQ, raise `UnsupportedQuantForIPCError`, even when fallback is enabled.
- **Sleep mode**: do not use [sleep mode](sleep_mode.md) in `zero_copy` mode. The weights live in the daemon's CUDA IPC allocations, so they cannot be offloaded.
- **Parallelism**: tensor, pipeline, expert and data parallelism are supported, including across nodes.

## Quick start

1. Start one daemon per GPU. Pass the engine arguments, such as model, dtype, quantization and parallelism, that you will pass to `vllm serve`:

    ```bash
    vllm preload --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4
    ```

    Each daemon loads and post-processes its shard and, on Hopper and newer GPUs, runs the [FlashInfer autotune pass](#preloading-the-flashinfer-autotune-cache). It binds its socket only after that. `vllm preload` logs `Weight cache daemon READY` once every rank is serving.

2. Start the engine with the `ipc_cache` load format:

    ```bash
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --tensor-parallel-size 4 \
        --load-format ipc_cache
    ```

    On success the engine logs `Mapped <N> tensors from the weight cache daemon (zero_copy mode)`. If the daemon is not ready, it logs `Weight cache unusable (...); falling back to disk loading` and loads from disk.

3. After an engine restart, run the same `vllm serve` command. The weights are mapped from the daemon instead of loaded from disk.

The daemon must keep running as long as engines use it. `SIGINT` or `SIGTERM` stops all ranks and frees the GPU memory.

!!! tip "Start order"
    An engine that starts before the daemon is ready falls back to disk and keeps its own copy of the weights next to the daemon's. Wait for the [health endpoint](#health-endpoint) or the `READY` log line before starting the engine, or disable the fallback so the engine waits for the daemon:

    ```bash
    vllm serve ... --load-format ipc_cache \
        --model-loader-extra-config '{"fallback": false}'
    ```

    With `fallback` disabled the engine retries for up to `state_timeout_s`, 300 seconds by default, and fails if the daemon is still not ready. Set it above the daemon's load and tuning time.

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

Each rank logs `tuned FlashInfer; the tuned configs are in the on-disk autotune cache` when it finishes.

The pass runs where the engine's own autotune does: FlashInfer must be installed and the GPU must be Hopper (compute capability 9.0) or newer. It finishes before the daemon reports ready.

An engine reuses the tuned configs only if it computes the same cache key. The key is a hash of the engine configuration, and `vllm preload` builds its configuration with the same batch size defaults as `vllm serve`, so pass both the same arguments. An engine with different arguments, or one created through the offline `LLM` class, which uses different batch size defaults, tunes again. The load format is not part of the key. The engine logs `[Autotuner]: Loaded N configs from <path>` when it finds the daemon's file.

With MTP, EAGLE or EAGLE3 speculative decoding both daemon groups skip the pass: the target cannot run the drafter's dummy passes, and a table tuned on the draft model would key on the draft's config, which no engine looks up. The first engine tunes and fills the cache instead. Otherwise, tuning runs across the daemons' world group, including across nodes and DP ranks.

A failure in the pass is fatal: the daemon rank exits and `vllm preload` shuts down the other ranks. Pass `--no-enable-flashinfer-autotune` to skip the pass.

The engine reads the tuned configs from the cache directory, so the daemon and the engine must share it. In Docker, mount one volume at the cache directory in both containers, as shown in [Docker](#docker).

## How it works

1. Each daemon loads its shard with the regular loader, runs quantization post-processing and, where applicable, the autotune pass, and binds a Unix domain socket named after its GPU's UUID. An engine worker connects to the socket of the GPU it runs on, so the order of `CUDA_VISIBLE_DEVICES` does not matter.
2. An engine started with `--load-format ipc_cache` builds its model on the meta device and requests the tensors from the daemon. Before anything is transferred, both sides compare a fingerprint of the cached weights: checkpoint content, hashed from safetensors metadata so that identical weights in different directories still match, model architecture, TP, PP and DP size and rank, dtype, quantization method and config, model revision, and vLLM version. On a mismatch the engine falls back to disk loading, or fails if `fallback` is disabled. The engine also checks that the daemon's GPU UUID matches its own device.
3. The daemon exports each GPU tensor as a CUDA IPC handle and sends the few CPU tensors by value. The engine maps the handles, re-establishes tied weights, such as `lm_head.weight` sharing storage with `embed_tokens.weight`, as aliases, and re-runs only the Python-side part of post-processing, such as MoE kernel selection.

### GPU memory accounting

In `zero_copy` mode the daemon allocates the weights. The engine asks the daemon how much memory it holds and counts it toward its own budget, so `--gpu-memory-utilization` keeps its usual meaning: the fraction of the GPU used for weights and KV cache together, including the daemon's weights. The engine logs the split as `Weights are held outside this process: ...`. If the daemon's weights alone exceed the budget, startup fails with `Externally held weights ... exceed the desired GPU memory utilization`, and the fix is a higher `--gpu-memory-utilization`.

After loading, the daemon releases the transient allocations left over from loading and post-processing. The autotune pass runs afterwards and needs transient activation memory on top of the weights.

## Cache modes

Select the mode with `--model-loader-extra-config '{"mode": "..."}'`:

- `zero_copy` (default): the engine maps the daemon's allocations directly. Each GPU holds one copy of the weights, and the daemon serves any number of engine restarts. The daemon must stay alive for as long as the engine runs.
- `copy`: the engine copies every tensor into its own GPU memory and then asks the daemon to release its cache. Use this when the daemon should free its GPU memory after the handoff. The GPU holds two copies during the copy. Afterwards the daemon has nothing left to serve, so the next restart falls back to disk, or fails after `state_timeout_s` if `fallback` is disabled.

## Health endpoint

`--weight-cache-health-port` makes `vllm preload` serve an HTTP endpoint for orchestrators:

```bash
vllm preload --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4 \
    --weight-cache-health-port 8001
```

`GET /health` returns `200` once every local daemon rank, including the draft group with speculative decoding, is serving, and `503` before that or after any rank exits. The endpoint binds to `0.0.0.0`, or to `--weight-cache-health-host` if set. Without a port it is disabled.

## Docker

Run the daemon and the engine as separate containers. The daemon container is long-lived. The engine container can restart without losing the cached weights. The containers need the following:

| Requirement | Reason |
| --- | --- |
| `--ipc=host` on both containers | PyTorch keeps the reference counters of shared CUDA tensors in `/dev/shm`, which both containers must see. |
| `--pid=host` on both containers | CUDA IPC handles identify the exporting process by PID, so both processes must be in the same PID namespace. |
| A shared volume for the socket directory | The engine connects to the daemon's Unix sockets. Pass the mounted path as `--weight-cache-socket-dir` to the daemon and as `socket_dir` to the engine. |
| A shared volume for the vLLM cache | The daemon writes its [FlashInfer autotune results](#preloading-the-flashinfer-autotune-cache) under `VLLM_CACHE_ROOT` and the engine reads them from there. Mount one volume at `/root/.cache/vllm`, or `/home/vllm/.cache/vllm` in the non-root image, in both containers, or point `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` at a shared path. Without it the engine tunes again. |
| The same GPUs | Pass the same `--gpus` selection to both containers. |
| The same user | The socket directory and the sockets must be owned by the user that connects. Run both containers as root, the default, or with the same `--user`. |
| The same image tag | The fingerprint includes the vLLM version. |

The `vllm/vllm-openai` image uses `vllm serve` as its entrypoint. Start the daemon with `--entrypoint vllm` and pass `preload` as the first argument.

!!! note "Socket directory ownership"
    An explicitly configured socket directory must be owned by the user running the daemon. A named volume is created root-owned, and a bind mount is owned by whoever created it. Point the daemon at a subdirectory of the mount, such as `/run/vllm-weight-cache/sockets`, which it creates with the right owner. A non-root container also needs a mount that is writable by its UID.

### docker run

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct
SOCKET_DIR=/run/vllm-weight-cache/sockets

# 1. The daemon: long-lived, holds the weights.
docker run -d --name vllm-weight-cache \
    --gpus all --ipc=host --pid=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v vllm-weight-cache:/run/vllm-weight-cache \
    -v vllm-cache:/root/.cache/vllm \
    --health-cmd "curl -sf http://localhost:8001/health || exit 1" \
    --health-interval 10s --health-start-period 10m \
    --entrypoint vllm \
    vllm/vllm-openai:latest \
    preload --model $MODEL --tensor-parallel-size 4 \
        --weight-cache-socket-dir $SOCKET_DIR \
        --weight-cache-health-port 8001

# 2. The engine: restarted by Docker whenever it exits.
docker run -d --name vllm \
    --gpus all --ipc=host --pid=host \
    --restart always \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v vllm-weight-cache:/run/vllm-weight-cache \
    -v vllm-cache:/root/.cache/vllm \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    $MODEL --tensor-parallel-size 4 \
        --load-format ipc_cache \
        --model-loader-extra-config "{\"socket_dir\": \"$SOCKET_DIR\", \"fallback\": false}"
```

With `"fallback": false` the engine waits for the daemon on a cold start instead of loading from disk next to it. Without it, start the engine only after `docker inspect` reports the daemon as `healthy`. Raise `--health-start-period` for models that take longer than 10 minutes to load and tune.

Restarting the engine container, by hand or through `--restart always`, reuses the cached weights. Do not restart the daemon container while an engine runs against it in `zero_copy` mode.

### Docker Compose

The same deployment as a Compose file. `condition: service_healthy` starts the engine only after the daemon reports ready. Raise `start_period` for models that take longer than 10 minutes to load and tune.

```yaml
services:
  weight-cache:
    image: vllm/vllm-openai:latest
    entrypoint: ["vllm", "preload"]
    command:
      - --model=meta-llama/Llama-3.1-8B-Instruct
      - --tensor-parallel-size=4
      - --weight-cache-socket-dir=/run/vllm-weight-cache/sockets
      - --weight-cache-health-port=8001
    ipc: host
    pid: host
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - weight-cache:/run/vllm-weight-cache
      - vllm-cache:/root/.cache/vllm
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8001/health || exit 1"]
      interval: 10s
      start_period: 10m
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  vllm:
    image: vllm/vllm-openai:latest
    command:
      - meta-llama/Llama-3.1-8B-Instruct
      - --tensor-parallel-size=4
      - --load-format=ipc_cache
      - '--model-loader-extra-config={"socket_dir": "/run/vllm-weight-cache/sockets", "fallback": false}'
    ipc: host
    pid: host
    restart: always
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - weight-cache:/run/vllm-weight-cache
      - vllm-cache:/root/.cache/vllm
    depends_on:
      weight-cache:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  weight-cache:
  vllm-cache:
```

`docker compose restart vllm` restarts only the engine.

### Scoping the shared namespaces

To share the IPC and PID namespaces between the two containers only, not with the host, start the daemon with `--ipc=shareable` instead of `--ipc=host`, since Docker's default IPC mode is private, and without `--pid=host`. Start the engine with `--ipc=container:vllm-weight-cache --pid=container:vllm-weight-cache`. The engine then uses the daemon container's `/dev/shm`, so size the daemon's `--shm-size` for both.

## Kubernetes

With the standard GPU device plugins a GPU is allocated to one container, so the daemon and the engine cannot be separate containers that request the same GPUs. Run both in one container: the daemon in the background and the engine in a restart loop, so an engine crash restarts only the engine.

```yaml
spec:
  volumes:
    - name: shm
      emptyDir:
        medium: Memory
        sizeLimit: "2Gi"
  containers:
    - name: vllm
      image: vllm/vllm-openai:latest
      command: ["/bin/bash", "-c"]
      args:
        - |
          vllm preload --model meta-llama/Llama-3.1-8B-Instruct \
              --tensor-parallel-size 4 \
              --weight-cache-health-port 8001 &
          while true; do
            vllm serve meta-llama/Llama-3.1-8B-Instruct \
                --tensor-parallel-size 4 \
                --load-format ipc_cache \
                --model-loader-extra-config '{"fallback": false}'
            sleep 1
          done
      ports:
        - containerPort: 8000
        - containerPort: 8001
      resources:
        limits:
          nvidia.com/gpu: "4"
      volumeMounts:
        - name: shm
          mountPath: /dev/shm
      # Give the daemon up to 30 minutes to load and tune before liveness applies.
      startupProbe:
        httpGet:
          path: /health
          port: 8001
        periodSeconds: 10
        failureThreshold: 180
      # Restart the container only when the daemon is unhealthy.
      livenessProbe:
        httpGet:
          path: /health
          port: 8001
        periodSeconds: 10
      # Serve traffic only while the engine is up.
      readinessProbe:
        httpGet:
          path: /health
          port: 8000
        periodSeconds: 5
```

In this spec:

- The liveness probe targets the daemon. A liveness probe on the engine would restart the container, and the daemon with it, on every engine crash.
- Both processes share the container's IPC and PID namespaces, the default socket directory and the vLLM cache directory. Mount a volume at `/root/.cache/vllm` to keep the tuned FlashInfer configs across container restarts.
- The engine waits for the daemon because `fallback` is disabled. If the daemon's load and tuning take longer than `state_timeout_s`, the engine exits and the loop restarts it. Raise the timeout to avoid this.

See [Using Kubernetes](../deployment/k8s.md) for a complete deployment manifest to add this container spec to.

## Loader configuration

The `ipc_cache` loader accepts these keys through `--model-loader-extra-config`:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --load-format ipc_cache \
    --model-loader-extra-config '{"fallback": false, "state_timeout_s": 900}'
```

| Key | Default | Description |
| --- | ------- | ----------- |
| `mode` | `zero_copy` | `zero_copy` or `copy`. See [Cache modes](#cache-modes). |
| `fallback` | `true` | Load from disk when the daemon is unavailable or the fingerprints mismatch. When `false`, the engine waits up to `state_timeout_s` for the daemon and fails if it never becomes usable. |
| `socket_dir` | per-user private directory under the temp dir | Directory containing the daemon sockets. Must match the daemon's `--weight-cache-socket-dir`. |
| `socket_path` | derived from the GPU UUID | Explicit socket path. Cannot be combined with a cached speculative draft (MTP, EAGLE or EAGLE3), which needs separate target and draft sockets. |
| `connect_timeout_s` | `5.0` | Socket connect timeout in seconds. |
| `state_timeout_s` | `300.0` | Timeout for the weight-transfer request in seconds. With `fallback` disabled, also the total time to wait for the daemon. |

## Daemon configuration

`vllm preload` takes the full set of engine arguments plus:

| Flag | Default | Description |
| --- | ------- | ----------- |
| `--weight-cache-socket-dir` | per-user private directory under the temp dir | Directory for the per-GPU Unix sockets. Must match the engine's `socket_dir`. |
| `--weight-cache-health-port` | disabled | Port for the `/health` readiness endpoint. |
| `--weight-cache-health-host` | `0.0.0.0` | Host for the `/health` endpoint. |
| `--weight-cache-master-port` | a free port | Rendezvous port for the daemons' own process group. Required and identical on every node for multi-node setups. Must differ from the engine's `--master-port`. |
| `--weight-cache-draft-master-port` | `--weight-cache-master-port + 1` for multi-node, otherwise a free port | Rendezvous port for the draft daemon group with speculative decoding. |

The daemon itself must load from disk. Passing `--load-format ipc_cache` to `vllm preload` is an error.

The default socket directory is `$TMPDIR/vllm_weight_cache_<uid>`, created with mode `0700`. An explicit directory must be owned by the user running the daemon, but its permission bits are not enforced.

See [vllm preload](../cli/preload.md) for the full CLI reference.

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

Pipeline parallelism needs no extra flags. There is still one daemon per local GPU, and the daemons' global ranks enumerate data-parallel, then pipeline, then tensor ranks, matching the engine's placement.

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

Pass the same `--speculative-config` to `vllm preload` and `vllm serve`. With MTP, EAGLE or EAGLE3, `vllm preload` also starts a draft daemon group that caches the draft model under its own cache key, Unix sockets (`*_draft.sock`) and rendezvous port (`--weight-cache-draft-master-port`). The engine routes the draft load to that group automatically. Other draft types are not cached and load from disk. Both groups skip the [FlashInfer autotune pass](#preloading-the-flashinfer-autotune-cache).

## Security

The socket protocol uses pickle and is intended only for trusted local
processes owned by the same user:

- Daemon sockets live in a per-user private directory (mode `0700`) and the
  socket files are restricted to the owner (`0600`).
- Both sides reject symlinked or non-owned socket paths; the auto-derived
  directory is additionally rejected if it is group/world accessible.
- On Linux the daemon verifies the connecting peer's UID via `SO_PEERCRED`.

The `--ipc=host` and `--pid=host` flags used in [Docker](#docker) let a container see host processes and shared memory. To limit this, share the namespaces between the two containers only, as described in [Scoping the shared namespaces](#scoping-the-shared-namespaces). See the [security documentation](../usage/security.md) for vLLM's general threat model.

## Daemon lifecycle

- Each GPU's socket path is guarded by an exclusive lock file, so a second
  daemon for the same GPU fails fast instead of clobbering the live socket.
- `vllm preload` reports readiness (log line and health endpoint) once every rank is serving. If any rank dies during startup, the remaining ranks are terminated and the command exits with that rank's exit code.
- `SIGINT`/`SIGTERM` to the `vllm preload` process terminates all daemon ranks, which unlinks their sockets and releases the GPU memory. Engines mapped to those weights in `zero_copy` mode will fail.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Weight cache socket ... is unavailable` or `Cannot connect to weight cache daemon at ...` | No daemon is serving at that path. Check that the daemon logged `READY`, that both sides use the same socket directory, which in Docker means the same mounted volume, and that the engine runs on the same GPUs. |
| `Weight cache daemon did not become ready within ...` | `fallback` is disabled and the daemon was not ready within `state_timeout_s`. Check the daemon logs, or raise the timeout if loading and tuning are slow. |
| `Another weight cache daemon already owns ...` | A daemon for this GPU is already running. Stop it first. The lock is released when a daemon exits or crashes. |
| `WeightCacheKey mismatch on fields: [...]` | The engine's configuration differs from the daemon's in the listed fields, such as `dtype`, `quantization`, `tp_size` or `vllm_version`. Start the daemon with the same arguments and vLLM version as the engine. |
| `Socket directory ... is not owned by the current user` | The daemon and the engine run as different users, or someone else owns the directory. Point `--weight-cache-socket-dir` at a subdirectory the daemon can create, and run both as the same user. |
| `Daemon GPU ... != engine GPU ...` | The socket belongs to a daemon on another GPU, usually because of an explicit `socket_path`. Use `socket_dir` and let the path derive from the GPU UUID. |
| `Weights were released` | The daemon already handed its weights off in `copy` mode. Restart the daemon. |
| `Externally held weights ... exceed the desired GPU memory utilization` | The daemon's weights alone exceed the `--gpu-memory-utilization` budget. Raise the utilization. |
| `UnsupportedQuantForIPCError` | The model uses a quantization method that cannot load pre-processed weights. Load this model from disk. |
| `Weight cache daemon rank(s) exited during startup` | A daemon rank crashed while loading or tuning. The traceback above the message names the cause. If it comes from the FlashInfer autotune pass, add `--no-enable-flashinfer-autotune`. The first engine then tunes instead. |
| The engine tunes FlashInfer again although the daemon tuned | The engine did not find the daemon's cache file. It logs `[Autotuner]: Loaded N configs from <path>` when it does. Check that both see the same cache directory, which in Docker means a shared volume, that both use the same engine arguments, and that the engine runs through `vllm serve` rather than the offline `LLM` class. |
