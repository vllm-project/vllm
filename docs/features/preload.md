# Preload: Fast Engine Restarts

Loading a large model takes minutes: the weights are read from disk, sharded,
and post-processed for the quantization kernels before the engine can serve.
When the engine crashes or is restarted, all of that work is repeated.

Preload avoids the repeat. A long-lived **weight cache daemon** per GPU loads
the weights once and keeps them resident in GPU memory. An engine started with
`--load-format ipc_cache` maps those weights through CUDA IPC instead of
loading from disk, so the weight-loading part of a restart takes seconds
instead of minutes.

- **Zero copy**: in the default mode the engine shares the daemon's GPU memory,
  so the daemon adds no extra GPU footprint while the engine runs.
- **Safe fallback**: when no daemon is reachable or its cached weights do not
  match the engine's configuration, the engine loads from disk as usual.
- **Same outputs**: the daemon hands over the exact post-processed tensors the
  engine would have produced itself.

The design is extensible to other artifacts; today it caches **model weights**.

## Requirements

- **CUDA or ROCm**. Other platforms raise `UnsupportedPlatformForIPCError`.
- **Same node, same user**. The daemon serves over a Unix domain socket and
  CUDA IPC handles only work on the node that created them. The daemon and the
  engine must run as the same user; connections from other users are rejected.
- **Same vLLM version** on the daemon and the engine. In Docker, run both from
  the same image tag.
- **Supported quantization**. The daemon serves weights that already went
  through quantization post-processing, so every quantization method in the
  model must support loading pre-processed weights. Unquantized models
  (except MoE models with `--all2all-backend moonep`) and checkpoints
  quantized with `fp8`, ModelOpt NVFP4 or MXFP4 (except GPT-OSS) are
  supported. Other methods, such as compressed-tensors, GPTQ and AWQ, raise
  `UnsupportedQuantForIPCError`, even when fallback is enabled.
- **No [sleep mode](sleep_mode.md)** in zero-copy mode: the weights live in
  the daemon's allocations, so the engine cannot offload them.

Tensor, pipeline, expert and data parallelism are supported, including across
nodes.

## Quick start

1. Start one daemon per GPU. `vllm preload` accepts the same engine arguments
   as `vllm serve` (model, dtype, quantization, parallelism, ...), and these
   must match the engine you start later:

    ```bash
    vllm preload --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4
    ```

    The daemons load and post-process the weights, then log
    `Weight cache daemon READY`. Each daemon binds its socket only after its
    shard is fully cached.

2. Start the engine with the `ipc_cache` load format:

    ```bash
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --tensor-parallel-size 4 \
        --load-format ipc_cache
    ```

    On success the engine logs
    `Mapped <N> tensors from the weight cache daemon (zero_copy mode)`.
    If the daemon is not ready yet, the engine logs
    `Weight cache unusable (...); falling back to disk loading` and loads from
    disk instead.

3. Kill the engine (or let it crash) and run the same `vllm serve` command
   again. Weight loading now takes seconds.

The daemon must stay running for as long as engines use it. Stop it with
`SIGINT`/`SIGTERM`, which shuts down all ranks and frees the GPU memory.

!!! tip "Cold start ordering"
    Starting the engine while the daemon is still loading makes both load the
    same weights from disk at the same time. The engine then keeps its own
    copy of the weights next to the daemon's for its whole lifetime, and the
    two loads compete for GPU memory, so memory profiling can fail or leave
    too little room for the KV cache. Either wait for the daemon's readiness
    signal (the [health endpoint](#health-endpoint) or the `READY` log line)
    before starting the engine, or disable the fallback so the engine waits
    for the daemon instead:

    ```bash
    vllm serve ... --load-format ipc_cache \
        --model-loader-extra-config '{"fallback": false}'
    ```

    With `fallback` disabled the engine retries the daemon for up to
    `state_timeout_s` (default 300 s) and fails if it never becomes ready.
    Raise the timeout for models whose cold load takes longer.

## How it works

1. `vllm preload` spawns one daemon process per local GPU. Each daemon loads
   its shard with the regular loader, runs quantization post-processing, and
   keeps the resulting tensors on its GPU.
2. The daemon binds a Unix domain socket named after its GPU's UUID. Engine
   workers connect to the socket of the GPU they run on, so the mapping is
   stable even when `CUDA_VISIBLE_DEVICES` orders the GPUs differently in
   the two processes.
3. An engine started with `--load-format ipc_cache` builds its model on the
   meta device (no storage) and requests the tensors from the daemon. Before
   anything is transferred, both sides compare a fingerprint of the cached
   weights: checkpoint content (hashed from safetensors metadata, so identical
   weights in different directories still match), model architecture,
   TP/PP/DP size and rank, dtype, quantization method and config, model
   revision, and vLLM version. On any mismatch the engine falls back to disk
   loading (unless `fallback` is disabled, in which case it fails).
4. The daemon exports every GPU-resident parameter and buffer as a CUDA IPC
   handle (the few CPU tensors are sent by value), and the engine maps them
   into its address space. The engine also checks that
   the daemon's GPU UUID matches its own device, so a stale socket can never
   serve weights for the wrong GPU.
5. The engine re-runs only the Python-side part of post-processing (for
   example, selecting MoE kernels), skipping the expensive weight
   transformations.

Tied weights (e.g. `lm_head.weight` sharing storage with
`embed_tokens.weight`) are exported once and re-established as aliases in the
engine, preserving parameter identity.

### GPU memory accounting

In zero-copy mode the weights are allocated by the daemon, not the engine.
The engine queries the daemon for the amount of memory it holds and counts it
as its own weight memory, so `--gpu-memory-utilization` keeps its usual
meaning: the fraction of the GPU available to the engine for weights **and**
KV cache, with the daemon-held weights included. You do not need to lower it
to leave room for the daemon. The engine logs the split at startup
(`Weights are held outside this process: ...`). If the daemon's weights alone
exceed the budget, startup fails with
`Externally held weights ... exceed the desired GPU memory utilization`;
raise `--gpu-memory-utilization` in that case.

After loading, the daemon releases the transient allocations left over from
loading and post-processing, so only the weights themselves stay resident.

## Cache modes

Select the mode with `--model-loader-extra-config '{"mode": "..."}'`:

- `zero_copy` (default): the engine maps the daemon's allocations directly.
  Each GPU holds a single copy of the weights, and the daemon serves any
  number of engine restarts. The daemon must stay alive for the lifetime of
  every engine using it: if it exits, the engine's weights go with it.
- `copy`: the engine clones every tensor into its own GPU memory and then asks
  the daemon to release its cache. This is a one-shot handoff: while the copy
  runs the GPU holds two copies of the weights, and after the release the
  daemon has nothing left to serve: the next engine restart falls back to
  disk, or fails once `state_timeout_s` elapses if `fallback` is disabled.
  Use it to pre-stage weights for a single engine start and stop the daemon
  afterwards.

## Health endpoint

`vllm preload` can expose an HTTP readiness endpoint for orchestrators:

```bash
vllm preload --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4 \
    --weight-cache-health-port 8001
```

`GET /health` returns `200` once every local daemon rank (and the draft group,
with speculative decoding) is serving, and `503` before that or after any rank
exits. The endpoint binds to `0.0.0.0` by default; change it with
`--weight-cache-health-host`. It is disabled unless a port is given.

Use it as a Docker health check or Kubernetes probe, and to gate the engine's
start on the daemon being ready (see [Docker](#docker) below).

## Docker

The daemon and the engine run in separate containers: a long-lived daemon
container that is never restarted while engines depend on it, and an engine
container that Docker restarts whenever it exits. Both must be able to share
GPU memory and talk over a Unix socket, which needs the following:

| Requirement | Why |
| --- | --- |
| `--ipc=host` on both containers | PyTorch's CUDA IPC keeps reference counters for shared tensors in `/dev/shm`; both containers must see the same one. (vLLM already needs this for its own shared memory.) |
| `--pid=host` on both containers | The legacy CUDA IPC handles PyTorch uses resolve the exporting process by PID, so the two processes must be in the same PID namespace. |
| A shared volume for the socket directory | The engine connects to the daemon's Unix sockets. Pass the mounted path as `--weight-cache-socket-dir` to the daemon and as `socket_dir` to the engine. |
| The same GPUs | Pass the same `--gpus` selection to both. Socket names derive from GPU UUIDs, so the order does not matter. |
| The same user | The socket directory and sockets must be owned by the connecting user. Run both containers as root (the default) or both with the same `--user`. |
| The same image tag | The fingerprint includes the vLLM version. |

The `vllm/vllm-openai` image's entrypoint is `vllm serve`, so the daemon
container overrides it with `--entrypoint vllm` and passes `preload` as its
first argument.

!!! note "Socket directory ownership"
    An explicitly configured socket directory must be owned by the user the
    daemon runs as, and a mount is owned by whoever created it (root for a
    named volume, your host user for a bind mount). Point the daemon at a
    *subdirectory* of the mount, such as `/run/vllm-weight-cache/sockets`: the
    daemon creates it with the right owner. A non-root container additionally
    needs the mount itself to be writable by its UID, for example a
    bind-mounted host directory that you `chown` to that UID.

### docker run

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct
SOCKET_DIR=/run/vllm-weight-cache/sockets

# 1. The daemon: long-lived, holds the weights.
docker run -d --name vllm-weight-cache \
    --gpus all --ipc=host --pid=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v vllm-weight-cache:/run/vllm-weight-cache \
    -p 8001:8001 \
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
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    $MODEL --tensor-parallel-size 4 \
        --load-format ipc_cache \
        --model-loader-extra-config "{\"socket_dir\": \"$SOCKET_DIR\", \"fallback\": false}"
```

`"fallback": false` makes the engine wait for the daemon on a cold start
instead of loading from disk alongside it. Alternatively, keep the default
fallback and start the engine only once `docker inspect` reports the daemon
container as `healthy`.

Restarting the engine container (`docker restart vllm`, or a crash with
`--restart always`) now reuses the cached weights. Do **not** restart the
daemon container while an engine is running against it in zero-copy mode.

### Docker Compose

The same deployment as a Compose file. `depends_on` with
`condition: service_healthy` makes Compose start the engine only after the
daemon reports ready:

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
```

Use `docker compose restart vllm` to restart only the engine.

### Scoping the shared namespaces

If sharing the host's IPC and PID namespaces is not acceptable, the engine
container can join the daemon container's namespaces instead. A container
can only be joined if its IPC namespace is shareable, so start the daemon
with `--ipc=shareable` (Docker's default is private) instead of `--ipc=host`,
drop `--pid=host` from both, and point the engine at the daemon container:

```bash
# daemon
docker run -d --name vllm-weight-cache \
    --gpus all --ipc=shareable --shm-size 16g \
    ... \
    --entrypoint vllm vllm/vllm-openai:latest preload ...

# engine
docker run -d --name vllm \
    --gpus all --ipc=container:vllm-weight-cache --pid=container:vllm-weight-cache \
    ... \
    vllm/vllm-openai:latest ...
```

The engine inherits the daemon's `/dev/shm`, so size the daemon's
`--shm-size` for the engine's own shared memory use.

## Kubernetes

With the standard GPU device plugins a GPU is allocated to exactly one
container, so the daemon and the engine cannot be two containers of a Pod that
both request the same `nvidia.com/gpu`. Run them in **one container** instead:
the daemon as the container's long-lived process and the engine in a restart
loop, so an engine crash restarts only the engine, not the container.

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
      # Give the daemon up to 30 minutes to load before liveness applies.
      startupProbe:
        httpGet:
          path: /health
          port: 8001
        periodSeconds: 10
        failureThreshold: 180
      # The daemon is the process worth restarting the container for.
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

Three things matter here:

- The **liveness probe targets the daemon**, not the engine. A liveness probe
  on port 8000 would restart the whole container (and with it the daemon)
  every time the engine crashes, defeating the purpose.
- Both processes share the container's IPC and PID namespaces and the default
  socket directory, so no extra configuration is needed.
- The engine waits for the daemon because `fallback` is disabled. If the
  daemon's first load takes longer than `state_timeout_s` (default 300 s),
  the engine exits and the loop starts it again; raise the timeout to avoid
  the churn.

See [Using Kubernetes](../deployment/k8s.md) for a complete deployment
manifest to add this container spec to.

## Loader configuration

The `ipc_cache` loader accepts extra keys via `--model-loader-extra-config`:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --load-format ipc_cache \
    --model-loader-extra-config '{"fallback": false, "state_timeout_s": 900}'
```

| Key | Default | Description |
| --- | ------- | ----------- |
| `mode` | `zero_copy` | `zero_copy` or `copy` (see [Cache modes](#cache-modes)). |
| `fallback` | `true` | Fall back to disk loading when the daemon is unavailable or the fingerprints mismatch. When `false`, the engine waits up to `state_timeout_s` for the daemon and fails if it never becomes usable. |
| `socket_dir` | per-user private directory under the temp dir | Directory containing the daemon sockets. Must match the daemon's `--weight-cache-socket-dir`. |
| `socket_path` | derived from the GPU UUID | Explicit socket path. Cannot be combined with a cached speculative draft (MTP/EAGLE), which needs separate target and draft sockets. |
| `connect_timeout_s` | `5.0` | Socket connect timeout in seconds. |
| `state_timeout_s` | `300.0` | Timeout for the weight-transfer request, and the total wait for the daemon when `fallback` is `false`. |

## Daemon configuration

`vllm preload` takes the full set of engine arguments plus:

| Flag | Default | Description |
| --- | ------- | ----------- |
| `--weight-cache-socket-dir` | per-user private directory under the temp dir | Directory for the per-GPU Unix sockets. Must match the engine's `socket_dir`. |
| `--weight-cache-health-port` | disabled | Port for the `/health` readiness endpoint. |
| `--weight-cache-health-host` | `0.0.0.0` | Host for the `/health` endpoint. |
| `--weight-cache-master-port` | a free port | Rendezvous port for the daemons' own process group. Required and identical on every node for multi-node setups; must differ from the engine's `--master-port`. |
| `--weight-cache-draft-master-port` | `--weight-cache-master-port + 1` (multi-node) or a free port | Rendezvous port for the draft daemon group with speculative decoding. |

The daemon itself must load from disk; passing `--load-format ipc_cache` to
`vllm preload` is an error.

The default socket directory is `$TMPDIR/vllm_weight_cache_<uid>`, created
with mode `0700`. When you set an explicit directory, it must be owned by the
user running the daemon, but its group/world permission bits are not enforced.

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

Pipeline parallelism needs no extra flags: there is still one daemon per
local GPU, and the daemons' global ranks enumerate data-parallel, then
pipeline, then tensor ranks, matching the engine's placement.

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

Pass the same `--speculative-config` to `vllm preload` and `vllm serve`. With
MTP, EAGLE or EAGLE3 drafts, `vllm preload` additionally starts a draft daemon
group that caches the draft model. It uses its own cache key, Unix sockets
(`*_draft.sock`) and rendezvous port (`--weight-cache-draft-master-port`), so
each daemon process serves exactly one model role. The engine routes the
draft load to that group automatically. Other draft types are not cached and
keep loading from disk in the engine.

## Security

The socket protocol uses pickle and is intended only for trusted local
processes owned by the same user:

- Daemon sockets live in a per-user private directory (mode `0700`) and the
  socket files are restricted to the owner (`0600`).
- Both sides reject symlinked or non-owned socket paths; the auto-derived
  directory is additionally rejected if it is group/world accessible.
- On Linux the daemon verifies the connecting peer's UID via `SO_PEERCRED`.

Sharing the host IPC and PID namespaces, as the [Docker](#docker) setup
requires, widens what a compromised container can observe on the host. Scope
the namespaces to the daemon container where possible. See the
[security documentation](../usage/security.md) for vLLM's general threat
model.

## Daemon lifecycle

- Each GPU's socket path is guarded by an exclusive lock file, so a second
  daemon for the same GPU fails fast instead of clobbering the live socket.
- `vllm preload` reports readiness (log line and health endpoint) once every
  rank is serving; if any rank dies during startup, the remaining ranks are
  terminated and the command exits with that rank's exit code.
- `SIGINT`/`SIGTERM` to the `vllm preload` process terminates all daemon
  ranks, which unlinks their sockets and releases the GPU memory. Engines
  mapped to those weights in zero-copy mode will fail.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Weight cache socket ... is unavailable` or `Cannot connect to weight cache daemon at ...` | No daemon is serving at that path. Check that the daemon is `READY`, that both sides use the same socket directory (in Docker: the same mounted volume) and that the engine runs on the same GPUs. |
| `Weight cache daemon did not become ready within ...` | `fallback` is disabled and the daemon did not come up within `state_timeout_s`. Check the daemon logs, or raise the timeout if the cold load is simply slow. |
| `Another weight cache daemon already owns ...` | A daemon for this GPU is already running, possibly from an earlier launch. Stop it first; the lock is released automatically when a daemon exits or crashes. |
| `WeightCacheKey mismatch on fields: [...]` | The engine's configuration differs from the daemon's in the listed fields (e.g. `dtype`, `quantization`, `tp_size`, `vllm_version`). Start the daemon with the same arguments and vLLM version as the engine. |
| `Socket directory ... is not owned by the current user` | The daemon or engine runs as a different user than the owner of the socket directory. Point `--weight-cache-socket-dir` at a subdirectory the daemon can create, and run both with the same UID. |
| `Daemon GPU ... != engine GPU ...` | The socket belongs to a daemon on another GPU. Usually caused by an explicit `socket_path`; prefer `socket_dir` and let the path derive from the GPU UUID. |
| `Weights were released` | The daemon already handed its weights off in `copy` mode. Restart the daemon. |
| Engine OOMs or profiles too little KV cache on a cold start | The engine and daemon loaded from disk concurrently. Wait for the daemon to be ready, or set `"fallback": false`. |
| `Externally held weights ... exceed the desired GPU memory utilization` | The daemon's weights alone are larger than the `--gpu-memory-utilization` budget. Raise the utilization. |
| `UnsupportedQuantForIPCError` | The model uses a quantization method that cannot load pre-processed weights. Load this model from disk. |
