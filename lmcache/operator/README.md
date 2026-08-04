# LMCache Kubernetes Operator

A Kubernetes operator that automates the deployment and lifecycle management of [LMCache](https://github.com/LMCache/LMCache) multiprocess cache servers. It manages a single CRD (`LMCacheEngine`) and reconciles it into a DaemonSet, ConfigMap, Service, and optional ServiceMonitor.

See [DESIGN.md](DESIGN.md) for architecture details, reconciliation logic, and CRD spec reference.

## Prerequisites

- Kubernetes 1.20+
- `kubectl` configured to access your cluster
- For NVIDIA GPUs (default): NVIDIA GPU Operator with the `nvidia` RuntimeClass available on GPU nodes
- For AMD GPUs: set `spec.gpuVendor: amd` in your `LMCacheEngine` (see [AMD GPUs (ROCm)](#amd-gpus-rocm) below)
- (Optional) [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator) for ServiceMonitor support
- (CacheBlend only) [cert-manager](https://cert-manager.io) for the injection webhook's serving cert — see [CacheBlend](#cacheblend) below

> [!IMPORTANT]
> By default the operator runs LMCache pods with `runtimeClassName: nvidia` and `NVIDIA_VISIBLE_DEVICES=all` to gain GPU visibility without consuming GPU resources via the device plugin. This allows the serving engine (e.g., vLLM) to claim all GPUs on the node. On most clusters that is enough; on some, the engine cannot see the GPUs unless the pod is also privileged. Set `spec.privileged: true` to run the engine container in privileged mode (default `false`). When it is enabled, clusters using Pod Security Standards must allow the `privileged` profile for the LMCache namespace.
>
> On AMD ROCm clusters, `spec.gpuVendor: amd` omits `runtimeClassName` and skips NVIDIA-specific env vars.

> [!WARNING]
> **Upgrade note:** earlier operator versions always ran the engine container privileged. `spec.privileged` now defaults to `false`. Upgrading rewrites the DaemonSet pod template (forcing a rolling pod replacement), and on any cluster where privileged was load-bearing for GPU visibility the engine pods will come back up **without** GPU access. If your cluster relied on privileged mode (always the case for `gpuVendor: amd`), set `spec.privileged: true` on existing CRs before upgrading.

## Quick Start

### 1. Install the Operator

**Option A: One-line install from release (recommended)**

Install the latest stable release:

```bash
kubectl apply -f https://github.com/LMCache/LMCache/releases/download/operator-latest/install.yaml
```

Or use the nightly build from the `dev` branch:

```bash
kubectl apply -f https://github.com/LMCache/LMCache/releases/download/operator-nightly-latest/install.yaml
```

**Option B: Build from source**

```bash
cd operator
make build
make install
make deploy IMG=<your-registry>/lmcache-operator:latest
```

### 2. Deploy an LMCacheEngine

The minimal CR just needs `l1.sizeGB`. Apply the sample (a fully-commented field reference covering every option):

```bash
kubectl apply -f config/samples/lmcache_v1alpha1_lmcacheengine.yaml
```

The operator automatically handles `hostIPC`, GPU visibility (`runtimeClassName: nvidia`, `NVIDIA_VISIBLE_DEVICES=all`; set `spec.privileged: true` if your cluster also needs privileged mode), node-local service routing, resource sizing, and Prometheus metrics — see [DESIGN.md](DESIGN.md) for details.

### 3. Connect vLLM to LMCache

The operator creates a ConfigMap named `<engine-name>-connection` with the `kv-transfer-config` JSON vLLM needs. A **mutating webhook** injects it into an opted-in vLLM pod automatically — see [Connection injection](#connection-injection) and the sample [`config/samples/vllm_lmcache_deployment.yaml`](config/samples/vllm_lmcache_deployment.yaml).

Key points for vLLM pods:

- **`hostIPC: true` is required** — CUDA IPC (`cudaIpcOpenMemHandle`) needs a shared IPC namespace between vLLM and LMCache. Without this, GPU memory mapping fails.
- **ConfigMap mount** — the `$(cat ...)` pattern reads the connection JSON and passes it inline to `--kv-transfer-config`. The ConfigMap name is always `<LMCacheEngine name>-connection`.
- **External LMCache connector required** — the operator-generated config now sets `kv_connector_module_path=lmcache.integration.vllm.lmcache_mp_connector` so vLLM loads the external LMCache MP connector instead of silently resolving a vendored builtin path.
- **No `hostNetwork` needed** — the operator creates a ClusterIP Service with `internalTrafficPolicy=Local`. kube-proxy routes traffic to the LMCache pod on the same node automatically. The ConfigMap points to the service DNS name, so neither LMCache nor vLLM pods need `hostNetwork`.

> [!IMPORTANT]
> Use a pinned vLLM image that is new enough to honor `kv_connector_module_path` for KV connector loading. In practice, that means a build that includes the external-module selection fix from vLLM PR #38301 (merged April 7, 2026). Builds that also include vLLM PR #42596 (merged May 15, 2026) are preferred because they default LMCache MP to the external connector with builtin fallback. If your existing vLLM build predates those changes or you are unsure, upgrade it before enabling this operator path.

> [!WARNING]
> **Do NOT mount an emptyDir at `/dev/shm`** on either LMCache or vLLM pods. With `hostIPC: true`, both pods share the host's `/dev/shm`. Mounting an emptyDir (even with `medium: Memory`) shadows it with a private tmpfs, breaking CUDA IPC — `cudaIpcOpenMemHandle` fails because IPC handles from one pod become invisible to the other.

### 4. Verify the Deployment

```bash
# Check LMCacheEngine status
kubectl get lmc
```

```
NAME       PHASE     READY   DESIRED   AGE
my-cache   Running   3       3         5m
```

```bash
# Check the connection ConfigMap
kubectl get configmap my-cache-connection -o yaml

# Check LMCache pods
kubectl get pods -l app.kubernetes.io/managed-by=lmcache-operator

# Check detailed status with endpoints
kubectl describe lmc my-cache
```

## Examples

Every scenario has a ready-to-edit manifest under [`config/samples/`](config/samples/) (`kubectl apply -f config/samples/<file>`):

| Scenario | Sample |
|---|---|
| Minimal + **full commented field reference** (GPU `nodeSelector`, custom `server.port`, L2 `raw`/`raw_block`, `resourceOverrides`, …) | [`lmcache_v1alpha1_lmcacheengine.yaml`](config/samples/lmcache_v1alpha1_lmcacheengine.yaml) |
| Production: Prometheus `ServiceMonitor`, custom port, `priorityClassName` | [`lmcache_v1alpha1_lmcacheengine_production.yaml`](config/samples/lmcache_v1alpha1_lmcacheengine_production.yaml) |
| L2 storage: Redis/Valkey (optional Secret auth) | [`lmcache_v1alpha1_lmcacheengine_l2_redis.yaml`](config/samples/lmcache_v1alpha1_lmcacheengine_l2_redis.yaml) |
| AMD GPUs (ROCm) | [`lmcache_v1alpha1_lmcacheengine_amd.yaml`](config/samples/lmcache_v1alpha1_lmcacheengine_amd.yaml) |
| vLLM Deployment wired to an LMCacheEngine — webhook-injected (see [Connection injection](#connection-injection)) | [`vllm_lmcache_deployment.yaml`](config/samples/vllm_lmcache_deployment.yaml) |
| CacheBlend engine + opted-in vLLM (see [CacheBlend](#cacheblend)) | [`lmcache_v1alpha1_cacheblendengine.yaml`](config/samples/lmcache_v1alpha1_cacheblendengine.yaml), [`vllm_cacheblend_deployment.yaml`](config/samples/vllm_cacheblend_deployment.yaml) |
| MP coordinator (fleet-wide registry, L2 quota eviction, global CacheBlend directory) + **commented field reference** | [`lmcache_v1alpha1_lmcachecoordinator.yaml`](config/samples/lmcache_v1alpha1_lmcachecoordinator.yaml) |

Notes:

- **GPU targeting** — `nodeSelector: {nvidia.com/gpu.present: "true"}` runs LMCache only on GPU nodes; new GPU nodes auto-get a pod.
- **AMD (ROCm)** — `spec.gpuVendor: amd` omits `runtimeClassName` and the NVIDIA env vars; vLLM connects via HIP IPC over `hostIPC` the same way (`PYTHONHASHSEED=0` still required). Supply a `nodeSelector` matching your platform's AMD label and a ROCm-built `spec.image`. AMD has no RuntimeClass-based device injection, so set `spec.privileged: true` to let the engine reach `/dev/kfd`/`/dev/dri` (see the [AMD sample](config/samples/lmcache_v1alpha1_lmcacheengine_amd.yaml)).
- **Custom port** — set `server.port`; the connection ConfigMap updates automatically and vLLM picks it up on restart.
- **L2 adapters** — only one at a time today. Redis/Valkey is natively typed; cross-namespace auth Secrets are copied automatically and injected via env (never in args or `kubectl describe`). Other types (`nixl_store`, `fs`, `mock`, `raw_block`) use the `raw` escape hatch — see the commented blocks in the minimal sample. For `raw_block` with `use_odirect: true`, `--l1-align-bytes` must be ≥ `block_align`.
- **Resources** auto-compute from `l1.sizeGB`; override with `resourceOverrides`.

## Connection injection

Wiring a vLLM Deployment to an LMCacheEngine by hand means mounting the
`<engine>-connection` ConfigMap and passing
`--kv-transfer-config "$(cat /etc/lmcache/kv-transfer-config.json)"`. A **mutating
webhook** can do this for you so the vLLM manifest stays clean.

Opt a vLLM pod in with the label `lmcache.ai/lmcache-inject: "true"` and the
annotation `lmcache.ai/lmcache-engine: "<engine>"` on its pod template, launching
vLLM via the image ENTRYPOINT (args-only — a `sh -c` wrapper is skipped). At pod
CREATE the webhook injects, reading the engine's `<engine>-connection` ConfigMap:

- `--kv-transfer-config <JSON>` — the `LMCacheMPConnector` config, inlined onto
  the vLLM container args (no volume mount needed);
- `hostIPC: true` — CUDA IPC with the node-local LMCache server;
- `PYTHONHASHSEED=0` — deterministic prefix hashing (only if you didn't set it).

### Optional: code-payload staging

By default the webhook only wires the connection — vLLM runs whatever `lmcache`
is baked into its image. To instead pin the vLLM pod to a specific `lmcache`
build (so the vLLM client and the engine server share one version), set
`spec.injection.payloadImage` on the `LMCacheEngine`. The webhook then also
stages that image's `lmcache` tree into the vLLM container, mirroring CacheBlend:

- a shared `emptyDir` (`lmcache-payload`) + a payload **init container** that
  copies the image's `/payload` tree into it (busybox `cp -a`, no command
  override);
- a read-only mount of that volume on the vLLM container;
- `PYTHONPATH=/lmcache-payload` prepended so vLLM imports the staged `lmcache`;
- the engine's `injection.imagePullSecrets` merged onto the pod (override per-pod
  with the `lmcache.ai/lmcache-image-pull-secrets` annotation) for private images.

`payloadImage` is a **separate, purpose-built** image: it must ship the unpacked
`lmcache` tree under `/payload` and copy it to `$SHARED_DIR` on start (the same
contract as the CacheBlend payload image). Its `repository` has no valid default,
so you must set it explicitly; leave `injection` unset for connection-only wiring.

Editable sample: [`config/samples/vllm_lmcache_deployment.yaml`](config/samples/vllm_lmcache_deployment.yaml).

> [!IMPORTANT]
> The webhook needs `make deploy` (not `make run`) + cert-manager, and the vLLM
> pod's namespace labeled `pod-security.kubernetes.io/enforce=privileged` (the
> injected `hostIPC` is rejected by `baseline`/`restricted`). The engine must
> already be reconciled in the same namespace (its `<engine>-connection`
> ConfigMap must exist — the webhook reads it).

The webhook mutates **Pods**, not the Deployment, so verify on a pod
(`kubectl get pod -l app=vllm-lmcache -o yaml | grep -E "hostIPC|kv-transfer-config|lmcache-injected"`).
If nothing was injected, check the pod's `lmcache.ai/lmcache-skip-reason`
annotation (`command-override`, `kv-transfer-config-present`, `engine-not-found`,
or `target-container-not-found`).

## CacheBlend

CacheBlend reuses cached KV at shifted positions. The operator manages it as a
second CRD (`CacheBlendEngine`) plus a **mutating webhook** that injects the
`lmcache-cacheblend` plugin into your vLLM pods — no vLLM image rebuild. See
[DESIGN.md](DESIGN.md#cacheblend-cacheblendengine-crd--injection-webhook) for the
architecture and the full field reference.

Quick start: deploy an engine, then opt a vLLM pod in with the label
`lmcache.ai/cacheblend-inject: "true"` and the annotation
`lmcache.ai/cacheblend-engine: "<engine>"` on its pod template (launch vLLM via the
image ENTRYPOINT — a `sh -c` wrapper is skipped). Editable samples:

- [`config/samples/lmcache_v1alpha1_cacheblendengine.yaml`](config/samples/lmcache_v1alpha1_cacheblendengine.yaml) — the `CacheBlendEngine`
- [`config/samples/vllm_cacheblend_deployment.yaml`](config/samples/vllm_cacheblend_deployment.yaml) — an opted-in vLLM Deployment

> [!IMPORTANT]
> CacheBlend needs the **webhook**, so deploy with `make deploy` (not `make run`,
> which is controller-only) and install **cert-manager** first
> (`kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml`).
> If Pod Security Standards are enforced, label the engine's and the vLLM pod's
> namespaces `pod-security.kubernetes.io/enforce=privileged` — the webhook injects
> `hostIPC`/`privileged`, which `baseline`/`restricted` reject.

> [!IMPORTANT]
> CacheBlend is still in early stage development and under heavy testing. Its
> docker image will not be publicly released until we are confident that it is
> ready to be shipped for general use cases. If you would like to try it first,
> please contact us in Slack Channel.

The webhook mutates **Pods**, not the Deployment, so verify on a pod
(`kubectl get pod -l app=vllm-cacheblend -o yaml | grep -E "cb-plugin|cacheblend-injected|skip-reason"`).
If nothing was injected, check the pod's `lmcache.ai/cacheblend-skip-reason`
annotation (`command-override`, `kv-transfer-config-present`, `engine-not-found`,
`payload-image-unset`, or `target-container-not-found`).

## Development

```bash
make generate     # Generate DeepCopy methods
make manifests    # Generate CRD YAML + RBAC
make build        # Compile operator binary
make fmt          # go fmt
make vet          # go vet
make test         # Run unit tests (envtest, CPU-only)
make lint         # Run golangci-lint
```

### End-to-End Tests

Four `make` targets cover the e2e tiers. The `-kind` variants create
a throwaway Kind cluster and tear it down on exit; the `-cluster`
variants run against whatever your current `KUBECONFIG` points at.
M2 (GPU) targets additionally run a runtime HTTP check against the
LMCache server and a vLLM round-trip that proves the KV cache
stores on the first request and retrieves on the second.

```bash
make test-e2e-kind                                   # local Kind, no GPU, ~5 min
make test-e2e-cluster        IMG=<registry/image:tag>  # existing cluster, no GPU
make test-e2e-gpu-kind                               # local Kind, GPU, ~30 min
make test-e2e-gpu-cluster    IMG=<registry/image:tag>  # existing GPU cluster
```

#### Tool prerequisites

| Target | Tools to install |
|---|---|
| `test-e2e-kind` | `kind`, `kubectl`, `docker` |
| `test-e2e-cluster` | `kubectl` (cluster access via `KUBECONFIG`) |
| `test-e2e-gpu-kind` | `kind`, `kubectl`, `docker`, `helm` (v3) |
| `test-e2e-gpu-cluster` | `kubectl`, `helm` (cluster access via `KUBECONFIG`) |

```bash
# Install kind (the other tools are distro-specific)
go install sigs.k8s.io/kind/cmd/kind@latest
```

#### One-time host setup for `test-e2e-gpu-kind`

Beyond the tools above, your host needs the NVIDIA driver and
`nvidia-container-toolkit` installed (distro-specific), plus two
`nvidia-ctk` commands that flip docker's default runtime to `nvidia`
and toggle the volume-mount-based GPU injection mechanism the
target's inline Kind config relies on:

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled
sudo nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place
sudo systemctl restart docker
```

The target fails fast with a copy-pasteable fix command if either
piece of host config is missing. Note that flipping docker's default
runtime is a **host-wide** change — every container on the host then
starts through `nvidia-container-runtime`. Non-GPU workloads still
work but go through one extra hook.

The target installs the [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator)
inside the Kind cluster to handle the toolkit / containerd reconfig
that lets pods scheduled by Kind's inner containerd see the driver
libraries. That install takes 5-10 min, which is the bulk of the
`test-e2e-gpu-kind` runtime. (`nvkind` is **not** required — we
tried it and the current target uses a hand-rolled Kind config
instead.)

#### Cluster prerequisites for `test-e2e-gpu-cluster`

The existing cluster must have:

- at least one node labeled `nvidia.com/gpu.present=true`
- the `nvidia` `RuntimeClass` installed (GPU Operator or equivalent)
- the operator image pushed to a registry the cluster's image puller can reach (pass as `IMG=…`)

#### Knobs (env vars)

| Variable | Default | Used by |
|---|---|---|
| `KIND_CLUSTER` | `operator-test-e2e-<id>` | `test-e2e-kind` — set to target an existing Kind cluster |
| `GPU_KIND_CLUSTER` | `operator-test-e2e-gpu-<id>` | `test-e2e-gpu-kind` — same idea |
| `KEEP_CLUSTER_ON_FAILURE` | unset | `test-e2e-gpu-kind` — set to `1` to keep the cluster alive after a failure for live debugging |
| `VLLM_MODEL` | `Qwen/Qwen2.5-0.5B` | vLLM integration spec — Hugging Face model id |
| `VLLM_IMAGE` | `lmcache/vllm-openai:latest` | vLLM integration spec |
| `SKIP_VLLM_INTEGRATION` | unset | set to `true` to skip the heavyweight vLLM spec but still run the runtime HTTP check |

If a GPU run fails during setup, use the diagnostic target:

```bash
make diagnose-test-e2e-gpu-kind GPU_KIND_CLUSTER=<name>
```

It dumps `ClusterPolicy` status, GPU-Operator pod state, events,
toolkit/device-plugin daemonset logs, `/dev/nvidia*` inside the Kind
worker, and the `nvidia` stanza of the worker's containerd config.

For deeper design notes (why GPU Operator over the bare device plugin,
how the volume-mount marker propagates GPUs into the Kind worker, the
test specs themselves), see [`AGENTS.md`](./AGENTS.md).

### Pushing a Custom Operator Image

```bash
# Docker Hub
docker login
make docker-build docker-push IMG=docker.io/<your-user>/lmcache-operator:latest
make deploy IMG=docker.io/<your-user>/lmcache-operator:latest

# Private registry
docker login <your-registry>
make docker-build docker-push IMG=<your-registry>/lmcache-operator:latest
make deploy IMG=<your-registry>/lmcache-operator:latest

# Multi-platform (amd64 + arm64)
make docker-buildx IMG=<your-registry>/lmcache-operator:latest
```

If your cluster needs pull credentials, create a secret and reference it in `config/manager/manager.yaml`:

```bash
kubectl create secret docker-registry regcred \
  --docker-server=<your-registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n lmcache-operator-system
```

## License

Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
