# Dynamo + LMCache MP integration

Recipes for running [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) vLLM
serving with KV cache offloaded to an **LMCache multiprocess (mp) server**, in
both **aggregated** and **disaggregated** modes.

In mp mode LMCache runs as an out-of-process cache engine. vLLM workers attach
to it through the `LMCacheMPConnector` and share KV tensors over CUDA IPC, so
`sharedMemory` is disabled (LMCache owns `/dev/shm`) and `hostIPC` is enabled.

## Layout

| Path | What it is |
|------|------------|
| [`deploy/lmcache_engine.yaml`](deploy/lmcache_engine.yaml) | `LMCacheEngine` CR for the shared MP server. Apply **before** the workers. |
| [`deploy/agg_lmcache_mp.yaml`](deploy/agg_lmcache_mp.yaml) | Kubernetes `DynamoGraphDeployment`, aggregated (single worker). |
| [`deploy/disagg_lmcache_mp.yaml`](deploy/disagg_lmcache_mp.yaml) | Kubernetes `DynamoGraphDeployment`, disaggregated (prefill + decode workers). |
| [`launch/agg_lmcache_mp.sh`](launch/agg_lmcache_mp.sh) | Local single-node launch script, aggregated (1 GPU). |
| [`launch/disagg_lmcache_mp.sh`](launch/disagg_lmcache_mp.sh) | Local single-node launch script, disaggregated (2 GPUs). |

## The LMCacheEngine (shared prerequisite)

[`deploy/lmcache_engine.yaml`](deploy/lmcache_engine.yaml) declares the MP
server that both the aggregated and disaggregated workers attach to. The
LMCache operator reconciles the CR into:

- a per-node **manager DaemonSet** pod (the LMCache server that holds the CPU
  KV pool), and
- a **ConfigMap** named `<metadata.name>-connection` (i.e. `lmcache-mp-connection`),
  which the vLLM workers mount at `/etc/lmcache` and read as
  `--kv-transfer-config`.

It must live in the **same namespace** as the workers (`default`) so the
connection ConfigMap is created where the workers can mount it.

**Version pinning**: the server's bundled lmcache must be wire-compatible with
the worker's. The worker image `vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3`
ships lmcache `0.4.4` (vLLM `0.20.1`), and the recipe pairs it with the
guide-validated server build `nightly-2026-04-25` (lmcache `0.4.5.dev31`, a
pre-stable build wire-compatible with that worker). Replace `my-tag` in each
manifest accordingly.

## Usage

The `deploy/` manifests are applied with `kubectl` against a cluster that
already has the Dynamo platform and the LMCache operator installed. Apply the
`LMCacheEngine` first, then one of the worker manifests:

```bash
kubectl apply -n default -f deploy/lmcache_engine.yaml
kubectl apply -n default -f deploy/disagg_lmcache_mp.yaml   # or agg_lmcache_mp.yaml
```

See the [Kubernetes deployment guide](../../docs/source/mp/deployment.rst) for
the full step-by-step (platform install, operator, namespace, HF token Secret,
then `kubectl apply`).

The `launch/` scripts are the single-node local equivalents. They run **inside
the Dynamo `vllm-runtime` container** and source Dynamo's
`examples/common/{gpu_utils,launch_utils}.sh` helpers, so run them from a Dynamo
checkout (or copy them into `examples/backends/vllm/launch/`).

> The deploy manifests pin the worker image to `my-tag` and the
> `LMCacheEngine` image separately; keep the two wire-compatible (see the
> validated pairings in the deployment guide).
