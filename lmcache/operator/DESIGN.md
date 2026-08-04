# LMCache Kubernetes Operator Design

## Overview

LMCache multiprocess mode runs as a separate server process that vLLM instances connect to for KV cache offloading. Today this is deployed manually via raw K8s manifests (a DaemonSet for LMCache + a Deployment for vLLM). A K8s operator would automate lifecycle management, enforce best practices (`hostIPC`, resource sizing), and expose connection info for vLLM discovery.

### Current Pain Points the Operator Solves

- Manual `hostIPC` setup and node-local service discovery
- No automated connection info propagation to vLLM
- No Prometheus ServiceMonitor integration
- No validation of configuration parameters

### How the Operator Addresses These

The operator introduces a single CRD (`LMCacheEngine`) that declaratively captures the full LMCache server configuration. The controller reconciles this CR into the underlying K8s resources (DaemonSet, Service, ConfigMap, ServiceMonitor), automatically injecting the required pod-level settings that are easy to forget in hand-written manifests.

**Auto-injected pod settings eliminate manual boilerplate.** The controller always sets `hostIPC: true` and `--host 0.0.0.0` — settings that the current `lmcache-daemonset.yaml` requires users to specify by hand. Getting any of these wrong (e.g., forgetting `hostIPC`) causes silent connectivity failures or CUDA IPC errors that are hard to debug.

**A node-local Service with connection ConfigMap provides a stable discovery contract for vLLM.** The operator creates a ClusterIP Service with `internalTrafficPolicy=Local`, which ensures kube-proxy routes traffic only to the LMCache pod on the same node. A ConfigMap (`<name>-connection`) contains the `kv-transfer-config` JSON pointing to this service's cluster DNS name. vLLM deployments simply mount the ConfigMap — no downward API or shell substitution needed. When the LMCache CR changes, the ConfigMap updates automatically and vLLM pods pick up the new values on restart.

**Prometheus integration is declarative.** When `prometheus.serviceMonitor.enabled` is set, the operator creates a ServiceMonitor CR that the Prometheus Operator discovers automatically. Without the operator, users must manually create ServiceMonitor resources and keep labels/ports in sync with the DaemonSet.

**CRD validation catches misconfigurations at apply time.** OpenAPI schema validation and a validating webhook enforce constraints (e.g., `l1.sizeGB > 0`, `eviction.triggerWatermark` in `(0.0, 1.0]`) before any pods are created. Today, invalid CLI flags only surface as runtime crashes inside the container.

**Resource sizing is auto-computed from the L1 cache size.** The operator derives `memoryRequest` (`l1.sizeGB + 5 GiB`) and `memoryLimit` (`1.5x request`) automatically, eliminating the mental math that currently leads to either OOM kills (under-provisioned) or wasted node capacity (over-provisioned). Users can override with explicit values.

---

## API Group & CRD

```yaml
apiVersion: lmcache.ai/v1alpha1
kind: LMCacheEngine
```

- **Group** `lmcache.ai`
- **v1alpha1** — alpha maturity; shape will evolve as L2 backends stabilize
- **LMCacheEngine** — represents the per-node cache engine. Future CRDs can
cover other concerns (e.g., `LMCacheKeyManager` for global key management,
`LMCacheMonitor` for engine state monitoring)

---

## CRD Spec (Complete)

```yaml
apiVersion: lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: string
  namespace: string
spec:
  # -- Container image --
  image:
    repository: string        # default: "lmcache/standalone"
    tag: string               # default: "nightly"
    pullPolicy: string        # default: IfNotPresent
  imagePullSecrets: []LocalObjectReference

  # -- Server config (maps to server.py argparse) --
  server:
    port: int                 # default: 5555
    chunkSize: int            # default: 256 tokens
    maxWorkers: int           # default: 1
    hashAlgorithm: string     # default: blake3 (builtin | sha256_cbor | blake3)

  # -- L1 cache (maps to L1MemoryManagerConfig + L1ManagerConfig) --
  # Internal tuning knobs (useLazy, alignBytes, writeTTLSeconds,
  # readTTLSeconds) use server defaults and can be overridden via the
  # env escape hatch if needed.
  l1:
    sizeGB: float             # REQUIRED

  # -- Eviction (maps to EvictionConfig) --
  eviction:
    policy: string            # default: "LRU" (only supported value)
    triggerWatermark: float   # default: 0.8 (range 0.0-1.0)
    evictionRatio: float      # default: 0.2 (range 0.0-1.0)

  # -- Monitoring (maps to PrometheusConfig from mp_observability/config.py) --
  # Note: the CRD uses `enabled: true` by default; the CLI equivalent is
  # the absence of the `--disable-prometheus` flag.
  prometheus:
    enabled: bool             # default: true  (CLI: omit --disable-prometheus)
    port: int                 # default: 9090
    serviceMonitor:
      enabled: bool           # default: false
      interval: string        # default: "30s"
      labels: map[string]string
  # ServiceMonitor is a CRD from the Prometheus Operator (kube-prometheus-stack).
  # When enabled, the operator creates a ServiceMonitor resource that tells
  # Prometheus to automatically discover and scrape LMCache metrics endpoints.
  # Without it, you'd need to manually configure Prometheus scrape targets.
  # If you don't use the Prometheus Operator, leave serviceMonitor.enabled=false
  # and use the pod annotations (prometheus.io/scrape, prometheus.io/port) instead.

  # -- L2 storage backend (single adapter) --
  # Currently only one L2 adapter is supported at a time.
  # LMCache MP mode supports multiple adapters, but this is not yet
  # fully tested. Once validated, the operator will support multiple.
  # Exactly one of resp or raw must be set.
  l2Backend:
    # Option A: Native RESP (Redis/Valkey) adapter
    resp:
      host: string              # REQUIRED
      port: int                 # REQUIRED, 1-65535
      numWorkers: int           # default: 8
      maxCapacityGB: float      # default: 0 (disabled)
      authSecretRef:            # optional, Secret with "username"/"password" keys
        name: string
    # Option B: Raw escape hatch for other adapter types
    raw:
      type: string              # adapter type name (nixl_store, fs, mock, raw_block, etc.)
      config: map[string]any    # type-specific config as free-form map
    # Optional: serde transform on KV bytes to/from the L2 adapter. Renders
    # a "serde" sub-dict into the --l2-adapter JSON; applies to whichever
    # adapter is configured. Exactly one serde type must be set (only aesgcm
    # today; other server-side serdes like fp8/turboquant can gain typed
    # fields here later, or be set via raw.config.serde).
    serde:
      # aesgcm: at-rest encryption keyed per cache_salt. The referenced
      # Secret is mounted directly into the engine pods, read-only at
      # /etc/lmcache/keys/master (only the "master" data key is projected).
      # The operator never reads or writes the Secret and never generates
      # key material; provenance/rotation stay with the user. The reference
      # is same-namespace only, so CR-create permission cannot be leveraged
      # to read Secrets from other namespaces (confused deputy). See
      # docs/design/v1/distributed/serde/aesgcm.md for the threat/key models.
      aesgcm:
        masterKeySecretRef:     # REQUIRED, user-created Secret with a "master" data key,
          name: string          # in the engine's namespace (same-namespace only)
        keyProvider: string     # default: hkdf (only implemented provider)
        aesBits: int            # default: 128 (or 256)

  # -- Connection-injection webhook defaults (optional) --
  # Read by the LMCache mutating webhook for pods bound to this engine. When
  # unset, the webhook wires only the connection (--kv-transfer-config, hostIPC,
  # PYTHONHASHSEED). Set injection.payloadImage to ALSO stage an lmcache code
  # tree into the vLLM container (emptyDir + init container + readOnly mount +
  # PYTHONPATH=/lmcache-payload), reusing the CacheBlend payload-staging
  # mechanism so the vLLM client and the engine server run one lmcache build.
  injection:
    payloadImage:               # SEPARATE image: ships lmcache under /payload
      repository: string        # REQUIRED (no valid default)
      tag: string               # default: latest
      pullPolicy: string        # default: IfNotPresent
    imagePullSecrets: []LocalObjectReference  # for a private payload image
    targetContainer: string     # default: first container

  # -- Resources (auto-computed, no user input needed) --
  # The operator derives resource requests/limits from l1.sizeGB:
  #   memoryRequest = ceil(l1.sizeGB + 5) Gi
  #   memoryLimit   = ceil(memoryRequest * 1.5) Gi
  #   cpuRequest    = "4"
  # To override, use the resourceOverrides escape hatch below.
  resourceOverrides: ResourceRequirements  # optional, raw K8s resources override

  # -- Logging --
  logLevel: string            # default: INFO (DEBUG|INFO|WARNING|ERROR)

  # -- Scheduling --
  nodeSelector: map[string]string
  affinity: Affinity
  tolerations: []Toleration
  # nodeSelector determines which nodes get an LMCache instance.
  # Use nodeSelector: {nvidia.com/gpu.present: "true"} to target all GPU
  # nodes. When new GPU nodes join the cluster, the DaemonSet controller
  # automatically schedules an LMCache pod on them.

  # -- Overrides --
  env: []EnvVar               # additional environment variables
  volumes: []Volume           # additional volumes (e.g. for L2 disk backend)
  volumeMounts: []VolumeMount
  podAnnotations: map[string]string
  podLabels: map[string]string
  serviceAccountName: string
  priorityClassName: string

  # -- Security --
  privileged: bool            # default: false; run the engine container in
                              # privileged mode. Needed only on clusters where
                              # the engine cannot see the node GPUs otherwise
                              # (see "Auto-managed Pod Settings" below).

  # -- Extra CLI flags --
  extraArgs: []string         # appended last, can override any auto-generated flag
```

---

## Auto-managed Pod Settings (not in CRD spec)

The operator always injects these into the pod spec:

- **`hostIPC: true`** — **required for CUDA IPC between LMCache and vLLM.** LMCache uses `CudaIPCWrapper` which calls PyTorch's `_share_cuda_()` to get a GPU driver-level IPC handle. The handle is serialized and sent over ZMQ TCP. The receiving process reconstructs the tensor via `cudaIpcOpenMemHandle` at the driver level. This call requires both processes to share the same IPC namespace — without `hostIPC: true`, `cudaIpcOpenMemHandle` fails with `cudaErrorMapBufferObjectFailed`. **Both the LMCache pods and vLLM pods must have `hostIPC: true`.**
- **`runtimeClassName: nvidia`** — uses the NVIDIA container runtime, which injects the host's NVIDIA driver libraries and device files into the container. This is required for CUDA to function inside the pod.
- **`NVIDIA_VISIBLE_DEVICES=all`** — gives the engine access to all GPUs on the node for CUDA IPC and custom data transfer kernels without claiming any via `nvidia.com/gpu` resource requests (which would make those GPUs unavailable to the serving engine). The combination of `runtimeClassName: nvidia` + `NVIDIA_VISIBLE_DEVICES=all` lets the container see all GPUs without consuming device-plugin resources, so the serving engine (e.g., vLLM) can still request all GPUs on the node. On most clusters this is sufficient; on some, the engine cannot see the GPUs unless the pod is also privileged — set `spec.privileged: true` to run the engine container privileged (default `false`).
- **`NVIDIA_VISIBLE_DEVICES=all`** and **`NVIDIA_DRIVER_CAPABILITIES=all`** — env vars that instruct the NVIDIA container runtime to expose all GPUs and all driver capabilities to the container.
- **`--host 0.0.0.0`** — always passed as a container arg. The server defaults to `--host localhost` which only binds to loopback; the server must bind to all interfaces so the node-local Service can route traffic to it.
- **No `hostNetwork`** — the operator does **not** use `hostNetwork`. Instead, it creates a ClusterIP Service with `internalTrafficPolicy=Local`. kube-proxy ensures that traffic to the service is routed only to the LMCache pod on the same node. This avoids occupying host ports and reduces the privileged surface area.
- **No `/dev/shm` emptyDir mount** — the operator intentionally does *not* mount an emptyDir at `/dev/shm`. With `hostIPC: true`, the container already sees the host's `/dev/shm`. Mounting an emptyDir would shadow the host's `/dev/shm` with a private tmpfs, breaking CUDA IPC (`cudaIpcOpenMemHandle` fails because IPC handles written by one pod are invisible to others). If your workload needs a larger `/dev/shm` for non-IPC purposes, add it via `spec.volumes` / `spec.volumeMounts`.

> **Security implications:** The LMCache pods run with `hostIPC: true` (required for CUDA IPC), which exposes the host's IPC namespace. When `spec.privileged: true` is set they additionally run privileged, granting full device access to the container. Only deploy in trusted environments. Clusters using Pod Security Standards must allow the `privileged` profile for the LMCache namespace whenever `hostIPC`/`privileged` are in use — the `baseline` and `restricted` profiles reject these settings.

---

## Validation Rules

| Field | Rule |
|---|---|
| `l1.sizeGB` | Required, must be `> 0` |
| `eviction.policy` | Must be `"LRU"` (if set) |
| `eviction.triggerWatermark` | Must be in `(0.0, 1.0]` |
| `eviction.evictionRatio` | Must be in `(0.0, 1.0]` |
| `server.port` | Must be in `[1024, 65535]` |

---

## CRD Status

```yaml
status:
  phase: Pending | Running | Degraded | Failed
  observedGeneration: int64

  desiredInstances: int
  readyInstances: int

  # Per-node connection info (for kubectl visibility)
  endpoints:
    - nodeName: string
      hostIP: string
      podName: string
      port: int
      metricsPort: int
      ready: bool

  # Standard conditions
  conditions:
    - type: Available          # at least one instance ready
    - type: AllInstancesReady  # all desired instances ready
    - type: ConfigValid        # spec validation passed
  # Connection ConfigMap is always named <metadata.name>-connection;
  # no need to store the ref in status.
```

---

## Examples

### Minimal Deployment

```yaml
apiVersion: lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: my-cache
spec:
  l1:
    sizeGB: 60
```

This deploys a DaemonSet with 60GB L1 cache, LRU eviction, blake3 hashing, port 5555, auto-computed 65Gi memory request / 98Gi limit, Prometheus on 9090, and a connection ConfigMap for vLLM. The operator auto-injects `hostIPC` and `--host 0.0.0.0`, and creates a node-local Service for vLLM discovery.

### Production Deployment (all GPU nodes)

```yaml
apiVersion: lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: production-cache
  namespace: llm-serving
spec:
  # Target all GPU nodes — new GPU nodes automatically get an LMCache pod
  nodeSelector:
    nvidia.com/gpu.present: "true"

  image:
    repository: lmcache/standalone
    tag: v0.1.0
    pullPolicy: IfNotPresent

  server:
    port: 6555
    chunkSize: 256
    maxWorkers: 4
    hashAlgorithm: blake3

  l1:
    sizeGB: 60

  eviction:
    triggerWatermark: 0.8
    evictionRatio: 0.2

  prometheus:
    enabled: true
    port: 9090
    serviceMonitor:
      enabled: true
      labels:
        release: kube-prometheus-stack

  logLevel: INFO
  podAnnotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
  priorityClassName: system-node-critical
```

---

## Resources Created by the Operator

For `LMCacheEngine` named `production-cache`:

| Resource | Name | Purpose |
|---|---|---|
| DaemonSet | `production-cache` | Runs LMCache server pods |
| Service (ClusterIP, `internalTrafficPolicy=Local`) | `production-cache` | Node-local service discovery for vLLM |
| ConfigMap | `production-cache-connection` | kv-transfer-config JSON pointing to the lookup Service |
| Service (headless) | `production-cache-metrics` | Prometheus scrape target |
| ServiceMonitor (optional) | `production-cache` | Prometheus Operator integration |

### ConfigMap Content (for vLLM discovery)

```json
{
  "kv_connector": "LMCacheMPConnector",
  "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "tcp://<name>.<namespace>.svc.cluster.local",
    "lmcache.mp.port": "<spec.server.port, default 5555>"
  }
}
```

The ConfigMap uses the lookup Service's cluster DNS name. Because the Service has `internalTrafficPolicy=Local`, kube-proxy routes traffic only to the LMCache pod on the same node as the vLLM pod. vLLM pods mount this ConfigMap and pass the JSON to `--kv-transfer-config` — no downward API or shell variable substitution required. The explicit `kv_connector_module_path` makes the external LMCache MP connector path load-bearing and avoids silent fallback to an older vendored builtin connector path.

---

## Auto-injected Pod Template Details

In addition to the auto-managed pod settings above (`hostIPC`, `--host 0.0.0.0`),
the operator injects:

- Container command: `/opt/venv/bin/lmcache server`
- Container args: serialized from spec fields
- Env: `LMCACHE_LOG_LEVEL` from `spec.logLevel`
- Probes:
  - **Startup:** TCP on server port, `initialDelay=5s`, `period=5s`, `failureThreshold=30`
  - **Liveness:** TCP on server port, `period=10s`
  - **Readiness:** TCP on server port, `period=5s`

---

## Reconciliation Logic

```
OnEvent(LMCacheEngine create/update/delete):

1. VALIDATE spec -> set condition ConfigValid
2. COMPUTE derived values (unless overridden):
   - memoryRequest = ceil(l1.sizeGB + 5) Gi
   - memoryLimit = ceil(memoryRequest * 1.5) Gi
   - containerArgs from all spec fields
3. RECONCILE DaemonSet (CreateOrUpdate, ownerRef)
   - Always inject: hostIPC, runtimeClassName: nvidia, NVIDIA_VISIBLE_DEVICES=all, --host 0.0.0.0 (privileged only when spec.privileged=true)
4. RECONCILE node-local lookup Service (internalTrafficPolicy=Local)
5. RECONCILE headless Service for metrics
6. RECONCILE connection ConfigMap
7. RECONCILE ServiceMonitor (if enabled)
8. UPDATE status:
   - Query workload for ready/desired counts
   - Enumerate pods -> build endpoints list
   - Set phase: Running | Degraded | Pending | Failed
   - Set conditions, observedGeneration
```

**Secondary watches:** DaemonSet, Pods (readiness changes → update endpoints), Nodes (new GPU node → DaemonSet auto-schedules)

**Deletion / cleanup**: every child resource the operator creates
(DaemonSet, lookup Service, metrics Service, connection ConfigMap,
managed RESP auth Secret, optional ServiceMonitor) carries an
`ownerReference` to the LMCacheEngine, so Kubernetes garbage
collection cascade-deletes them when the CR goes away. **No finalizer
is used.** An earlier design added a `lmcache.ai/cleanup` finalizer
to mirror that GC behavior, but it was a no-op that only created
deadlocks when the controller pod was not running (e.g. during
cluster issues or a single-step `kubectl delete -k config/default`).
The reconciler now actively strips that legacy finalizer from any CR
it sees, so migration from older operator versions is automatic.
Finalizers will return when we need to clean up state K8s GC cannot
reach (Redis L2 keys, federation deregistration, etc.).

---

## CacheBlend: `CacheBlendEngine` CRD + Injection Webhook

CacheBlend reuses cached KV at shifted positions. It has two halves the operator
manages together: a **GPU-resident blend engine** (server side) and a
**vLLM-side plugin** that must be loaded into the serving container. The operator
ships both as a second CRD plus a mutating admission webhook.

> This implements what was previously deferred as a future `blend.enabled` field
> on `LMCacheEngine`. It is instead a **separate `CacheBlendEngine` kind** (with
> its own controller) plus an injection webhook — cleaner separation, and no
> behavior change to `LMCacheEngine`.

### `CacheBlendEngine` CRD

Group `lmcache.lmcache.ai`, `v1alpha1`, kind `CacheBlendEngine` (shortName `cbe`).
The spec **mirrors `LMCacheEngineSpec`** (image, server, l1, eviction, prometheus,
l2Backend, scheduling, overrides, imagePullSecrets) and adds:

- `blend.checkLayer` (default 1) and `blend.recompRatio` (default 0.15) — CB
  tunables fed to the vLLM connector.
- `injection` — what the webhook injects into vLLM pods: `payloadImage` (an
  `ImageSpec` — `repository`/`tag`/`pullPolicy`, like `spec.image` — for the
  private `lmcache-cacheblend` init-container image; set `repository` explicitly,
  the inherited engine-image default is not a valid payload), `imagePullSecrets`
  (appended to the vLLM pod so the private payload image can pull — the Secret
  must exist in the vLLM pod's namespace), `targetContainer` (default: first
  container), and `cudagraph` (`eager`|`piecewise`|`full_decode_only`, default
  `eager`).
- `server.chunkSize` defaults to **256** and is validated to equal 256 (the blend
  matcher requires `chunk_size == vLLM --block-size * 4`).

### The blend engine (controller)

`CacheBlendEngineReconciler` mirrors `LMCacheEngineReconciler` and reconciles a
DaemonSet running `lmcache server --engine-type blend` (plus
`--l1-align-bytes 16777216`), a node-local lookup Service, a metrics Service, and
a `<name>-connection` ConfigMap. **GPU model is identical to `LMCacheEngine`**:
`runtimeClassName: nvidia` + `NVIDIA_VISIBLE_DEVICES=all` + `hostIPC: true` (and
`privileged` when `spec.privileged=true`), with **no `nvidia.com/gpu`
device-plugin claim** — the engine
*shares* the vLLM GPU rather than reserving one, because the blend server scatters
re-RoPE'd KV directly into vLLM's paged KV over **same-device CUDA IPC**. The
engine resource builders are the same name/spec-keyed cores used by
`LMCacheEngine`.

The `<name>-connection` ConfigMap carries the **`CBKVConnector`**
`kv-transfer-config` (vs `LMCacheMPConnector` for `LMCacheEngine`) — same node-local
`tcp://` host/port shape, plus the `cb.*` tunables:

```json
{
  "kv_connector": "CBKVConnector",
  "kv_connector_module_path": "lmcache_cacheblend.connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "tcp://<name>.<namespace>.svc.cluster.local",
    "lmcache.mp.port": "<server.port>",
    "cb.check_layer": <blend.checkLayer>,
    "cb.recomp_ratio": <blend.recompRatio>
  }
}
```

Co-location works exactly like `LMCacheEngine`: one engine per GPU node
(DaemonSet), and the node-local Service (`internalTrafficPolicy: Local`) routes a
vLLM pod to the same-node engine. The control-plane RPC is TCP via that Service;
the data-plane KV write is CUDA IPC on the shared GPU.

### The injection webhook

A mutating admission webhook (`/mutate--v1-pod`, `CREATE`, `failurePolicy: Ignore`)
injects the `lmcache-cacheblend` plugin into opted-in pods so a **stock vLLM image
needs no rebuild**. A pod opts in with label `lmcache.ai/cacheblend-inject: "true"`
and binds to an engine with annotation `lmcache.ai/cacheblend-engine: <name>`. The
webhook then applies:

| Mutation | What |
|---|---|
| pod `hostIPC: true` | required for CUDA IPC with the node-local engine |
| `cb-plugin` emptyDir + payload init container | the busybox payload `cp -a`'s the pure-Python plugin tree onto the shared volume |
| readOnly mount + `PYTHONPATH=/cb-plugin` on the vLLM container | vLLM discovers the plugin via its `vllm.general_plugins` entry point |
| append required vLLM args | `--attention-backend CUSTOM`, `--kv-transfer-config <from the connection ConfigMap>`, `--block-size 64`, `--pipeline-parallel-size 1`, `--no-enable-chunked-prefill`, `--no-async-scheduling`, `--enforce-eager` (or the configured cudagraph) |
| append `injection.imagePullSecrets` | so the private payload image can pull |
| stamp `lmcache.ai/cacheblend-injected: "true"` | idempotency guard |

The webhook **skips** (stamping `lmcache.ai/cacheblend-skip-reason`) when: the
target container overrides `command` (a `sh -c` wrapper — appended args wouldn't
reach `vllm serve`); the user already supplies `--kv-transfer-config` (not
clobbered); the named engine's connection ConfigMap doesn't exist; the engine's
`injection.payloadImage` resolves to an empty reference (`payload-image-unset`);
or the requested `targetContainer`/`cacheblend-container` annotation names a
container that does not exist on the pod (`target-container-not-found`). It does
**not** gate on engine readiness — like `LMCacheEngine`, the connector connects
when the engine comes up. Args are emitted in two-token form
(`--attention-backend CUSTOM`); the replace-not-duplicate dedup still recognizes a
user-supplied `--flag=value`.

> **Shared with the LMCache injector.** The emptyDir + payload init container +
> readOnly mount + `PYTHONPATH` staging (rows 2–3 above) is generic — the standard
> `LMCacheEngine` injector (`/mutate-lmcache--v1-pod`) reuses it (under
> `internal/webhook/payload_staging.go`) to optionally stage an `lmcache` build
> when `LMCacheEngine.spec.injection.payloadImage` is set. There the staging is
> **additive and optional**: the connection wiring always runs, and an unset (or
> absent) `injection.payloadImage` simply stages nothing — it never skips the
> whole injection. The LMCache staging uses its own names (`lmcache-payload`
> volume, `/lmcache-payload` mount) so both injectors can fire on one pod.

### Prerequisites

- **cert-manager** — the webhook's serving cert is a cert-manager `Issuer` +
  `Certificate` (caBundle injected via `inject-ca-from`); install it before
  `make deploy`.
- **`make deploy`, not `make run`** — `make run` sets `ENABLE_WEBHOOKS=false` and
  installs no `MutatingWebhookConfiguration`; it is controller-only. The webhook
  needs the operator running as an in-cluster pod.
- **Pod Security Standards** — the injected `hostIPC`/`privileged` is rejected by
  the `baseline`/`restricted` profiles, so the engine's and the vLLM pod's
  namespaces must be labeled `pod-security.kubernetes.io/enforce=privileged`.

### Resources created (for a `CacheBlendEngine` named `cb`)

| Resource | Name | Purpose |
|---|---|---|
| DaemonSet | `cb` | `lmcache server --engine-type blend` on GPU nodes |
| Service (node-local) | `cb` | same-node discovery for vLLM (`CBKVConnector`) |
| Service (headless) | `cb-metrics` | Prometheus scrape target |
| ConfigMap | `cb-connection` | `CBKVConnector` kv-transfer-config |
| MutatingWebhookConfiguration | (operator-wide) | injects the plugin into opted-in vLLM pods |

---

## Coordinator: `LMCacheCoordinator` CRD

The **coordinator** (`lmcache/v1/mp_coordinator/`) is the fleet-level HTTP service
that engine servers register/heartbeat against, and which drives L2 quota
eviction and global CacheBlend lookups. Unlike the engines (one DaemonSet pod per
GPU node), the coordinator is a single fleet-wide service, so `LMCacheCoordinator`
reconciles a **Deployment + ClusterIP Service** instead of a DaemonSet. The
controller carries no finalizer — owner-reference GC cascade-deletes the children.

### `LMCacheCoordinatorSpec`

The spec mirrors `MPCoordinatorConfig` (`lmcache/v1/mp_coordinator/config.py`); the
controller renders each field into the matching `lmcache coordinator` CLI flag:
`host`, `port` (9300), `instanceTimeout` (30), `healthCheckInterval` (10),
`evictionCheckInterval` (5), `evictionRatio` (0.2), `triggerWatermark` (1.0),
`blendChunkSize` (256), `blendProbeStride` (1). It also carries `replicas`,
`image`, `prometheus`, and the usual pod-shaping fields (resources, scheduling,
env, etc.).

The global-CacheBlend knobs (`blendChunkSize`, `blendProbeStride`) render into the
`--blend-chunk-size` / `--blend-probe-stride` flags and default to 256 / 1. Note
`blendChunkSize` **must equal** the blend servers' chunk size.

> **Metrics caveat:** the coordinator process exposes only `/healthz`, not a
> Prometheus `/metrics` endpoint. ServiceMonitor support is wired for parity but
> defaults **disabled**; enabling it is only useful once a metrics endpoint is
> added to the coordinator app.

### Connecting engines to a coordinator

`LMCacheEngineSpec` and `CacheBlendEngineSpec` gained a `coordinator` block
(`CoordinatorConnectionSpec`) that maps to the server's coordinator-client flags
(`lmcache/v1/multiprocess/config.py`: `add_coordinator_args`):

- `ref` — name of an `LMCacheCoordinator` in the same namespace. The controller
  resolves it to the coordinator's Service URL (`http://<name>.<ns>.svc:<port>`)
  before building the DaemonSet, so `BuildContainerArgs` stays a pure function.
- `url` — explicit escape hatch for a coordinator the operator does not manage.
  Exactly one of `ref`/`url` is required (enforced in validation).
- `advertiseIP` — **do not set this in almost every case.** It defaults to the
  pod IP via the downward API env `LMCACHE_COORDINATOR_ADVERTISE_IP`, which is
  correct for normal in-cluster deployments. Only override it if you know exactly
  what you are doing (e.g. the coordinator runs outside the cluster and must
  reach the server through a specific externally-routable address); an incorrect
  value silently breaks coordinator-to-server connectivity.
- `heartbeatInterval`, `l2EventReporting`, `l2EventFlushInterval`.

When `coordinator` is unset, the server emits no `--coordinator-url` and does not
register (unchanged behavior).

### Resources created (for an `LMCacheCoordinator` named `coord`)

| Resource | Name | Purpose |
|---|---|---|
| Deployment | `coord` | `lmcache coordinator` HTTP server (port 9300) |
| Service (ClusterIP) | `coord` | fleet-wide discovery endpoint for engines |
| Service (headless) | `coord-metrics` | Prometheus scrape target (only if `prometheus.serviceMonitor.enabled`) |
| ServiceMonitor | `coord` | Prometheus scrape config (gated, disabled by default) |

---

## Future Extensibility

- **L2 backends:** The RESP (Redis/Valkey) adapter is natively supported with typed CRD fields and Secret-based auth injection. Other adapter types (nixl_store, fs, mock, mooncake_store, raw_block) can be configured via the `raw` escape hatch. Currently only a single L2 adapter is supported at a time. LMCache MP mode is designed to support multiple adapters in cascade, but this is not yet fully tested — once validated, the operator will support multiple adapters.
- **Blend mode:** Implemented as the separate `CacheBlendEngine` CRD + injection webhook — see [CacheBlend](#cacheblend-cacheblendengine-crd--injection-webhook) above. (This supersedes the earlier idea of a `blend.enabled` field on `LMCacheEngine`.)
- **Update strategy:** Future `spec.updateStrategy` field for `RollingUpdate`/`OnDelete` control on the DaemonSet.
- **Coordinator:** Implemented as the separate `LMCacheCoordinator` CRD (Deployment + Service) with engine-side `coordinator` connection wiring — see [Coordinator](#coordinator-lmcachecoordinator-crd) above.
- **Additional CRDs:** `LMCacheKeyManager` (global key management), `LMCacheMonitor` (engine state monitoring), `LMCacheFederation` (cross-cluster P2P topology).

---

## Key Source Files Referenced

| File | What it defines |
|---|---|
| `lmcache/v1/distributed/config.py` | `L1MemoryManagerConfig`, `L1ManagerConfig`, `EvictionConfig`, `StorageManagerConfig`, argparse |
| `lmcache/v1/mp_observability/config.py` | `PrometheusConfig`, `add_prometheus_args`, `parse_args_to_prometheus_config` |
| `lmcache/v1/multiprocess/server.py` | `MPCacheServer`, server CLI entry point, argparse (lines 629–653) |
| `lmcache/v1/multiprocess/http_server.py` | HTTP server with `/healthcheck` endpoint (FastAPI + ZMQ) |
| `lmcache/v1/mp_coordinator/config.py` | `MPCoordinatorConfig` (coordinator knobs mapped by `LMCacheCoordinator`) |
| `lmcache/cli/commands/coordinator.py` | `lmcache coordinator` CLI (flags rendered into the coordinator Deployment) |
| `lmcache/v1/multiprocess/config.py` | `add_coordinator_args` (engine `coordinator` connection flags) |
| `lmcache/v1/distributed/l2_adapters/config.py` | L2 adapter registry pattern, `L2AdapterConfigBase`, `L2AdaptersConfig` |
| `examples/multi_process/lmcache-daemonset.yaml` | Reference DaemonSet manifest |
| `examples/multi_process/vllm-deployment.yaml` | Reference vLLM deployment with kv-transfer-config |
