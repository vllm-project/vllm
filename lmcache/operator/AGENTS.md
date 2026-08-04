# operator - AI Agent Guide

## Smoke Test Suite

The smoke suite under `test/e2e/` validates the operator end-to-end against
a real Kubernetes cluster. It is all-Go, built on Ginkgo/Gomega, and gated
by Go build tags so unit tests run without it.

### Naming convention

Targets follow a consistent `-kind` / `-cluster` suffix:

| Suffix | What it means |
|---|---|
| `-kind` | Self-contained: the target creates a fresh Kind cluster, runs the suite, and deletes the cluster on exit. |
| `-cluster` | Uses whatever cluster the current `KUBECONFIG` / context points at (OpenShift, EKS, k3s, an existing Kind cluster, …). Requires `IMG=` pointing at a registry the cluster can pull from. |

### Targets (M1, no-GPU)

```bash
make test-e2e-kind                                                                  # local Kind, ~5 min
make test-e2e-cluster IMG=<registry>/<repo>:<tag>                                   # existing cluster
```

### Targets (M2, GPU)

```bash
make test-e2e-gpu-kind                                                              # local Kind, ~30 min
make test-e2e-gpu-cluster IMG=<registry>/<repo>:<tag>                               # existing GPU cluster
```

Both run the M1 + M2 specs (M2 = runtime `/conf` round-trip + vLLM
integration, under the `e2e_gpu` build tag). Pick `test-e2e-gpu-kind`
when you have GPUs on the dev box and want a self-contained Kind
cluster; pick `test-e2e-gpu-cluster` when you're targeting an
existing OpenShift / EKS / GKE GPU cluster. See *GPU tier* below for
details.

#### `test-e2e-kind` — local Kind run

Builds the manager image, loads it into a dedicated Kind cluster
(`operator-test-e2e-<id>` by default), installs CRDs, deploys the controller,
runs every `//go:build e2e` spec under `test/e2e/`, then tears the
cluster down. No prereqs beyond Kind + Docker on `$PATH` (plus network
egress to GitHub).

The suite installs **cert-manager** into the cluster before deploying the
controller — it issues the mutating webhook's serving cert and injects the
CA bundle (see `config/certmanager`), so the controller-manager Deployment
never reaches `Available` without it. Install is skipped when the cluster
already ships cert-manager (e.g. an existing OpenShift/EKS cluster), and
the suite only uninstalls what it installed. Override the release with
`CERT_MANAGER_VERSION` (default `v1.16.3`).

#### `test-e2e-cluster` — existing cluster (OpenShift, EKS, k3s, …)

For when you already have a cluster running and just want the suite to
install/deploy → test → undeploy against it. Prereqs:

1. **Image pushed to a reachable registry.** The cluster nodes must be
   able to `docker pull` `$IMG`. On OpenShift the simplest path is the
   internal registry — see *Pushing to OpenShift's internal registry*
   below.
2. **`KUBECONFIG` / current-context** points at the target cluster.
3. Pass the in-cluster pull URL via `IMG=`. The target fails fast if
   `IMG` is missing or still the default `controller:latest`.

```bash
oc config use-context admin                                                          # or kubectl config use-context <name>
export IMG=image-registry.openshift-image-registry.svc:5000/lmcache-operator-system/operator:v0.0.1
make test-e2e-cluster IMG=$IMG
```

Under the hood this sets `SMOKE_SKIP_IMAGE_LOAD=true` so the suite
neither rebuilds nor `kind load`s. The cluster is left intact after
the run.

#### Pushing to OpenShift's internal registry

```bash
# One-time per cluster: expose the registry on a default route
oc patch configs.imageregistry.operator.openshift.io/cluster \
  --type merge -p '{"spec":{"defaultRoute":true}}'

# Log Docker into the registry using your oc session
oc registry login --to=$HOME/.docker/config.json --skip-check

# Build, tag, push
oc create namespace lmcache-operator-system --dry-run=client -o yaml | oc apply -f -
export OCP_REGISTRY=$(oc get route default-route -n openshift-image-registry -o jsonpath='{.spec.host}')
make docker-build IMG=controller:latest
docker tag controller:latest $OCP_REGISTRY/lmcache-operator-system/operator:v0.0.1
docker push                  $OCP_REGISTRY/lmcache-operator-system/operator:v0.0.1
```

The push uses the external route; pods inside the cluster pull via the
in-cluster service name (`image-registry.openshift-image-registry.svc:5000/...`).
Both names point at the same image; only the hostnames differ.

#### OpenShift caveats

- **PodSecurity admission**: test namespaces are pre-labeled
  `pod-security.kubernetes.io/enforce=privileged` so the operator's
  DaemonSet (which always sets `hostIPC=true`, and `privileged=true` only
  when `spec.privileged` is enabled) is accepted at admission time.
  `hostIPC=true` alone is rejected by the `baseline`/`restricted` profiles,
  so the label is required regardless of `privileged`. Harmless on clusters
  that don't enforce PodSecurity.
- **SCC (Security Context Constraints)**: M1 smokes never wait for
  DaemonSet pods to schedule, so SCC isn't a blocker. If you need pods
  to actually run later (M2/M3 GPU tier), grant the LMCache
  ServiceAccount the `privileged` SCC explicitly.

### Specs included in M1

| Spec file | Coverage |
|---|---|
| `crd_smoke_test.go` | harness sanity check + minimal-CR shape + custom-port propagation |
| `lifecycle_smoke_test.go` | port update propagation, delete + ownerRef GC, invalid `sizeGB` rejection |
| `field_coverage_smoke_test.go` | ServiceMonitor (auto-skipped if CRD absent), `extraArgs` override, `resourceOverrides` |
| `auth_smoke_test.go` | cross-namespace `authSecretRef` mirroring + env-var injection |

### Specs included in M2 (GPU, build tag `e2e_gpu`)

| Spec file | Coverage |
|---|---|
| `runtime_smoke_test.go` | HTTP `/conf` round-trip — proves CR field values reach the live LMCache server (not just the K8s objects) by asserting `mp.port` / `mp.chunk_size` / `mp.max_workers` / `mp.hash_algorithm` / `http.http_port` against the running pod's `/conf` payload. |
| `vllm_integration_smoke_test.go` | vLLM + LMCache round-trip — spins up a vLLM `Deployment` configured against the operator's `<engine>-connection` ConfigMap with `--no-enable-prefix-caching`, sends the same long prompt twice, and asserts `lmcache:num_hit_tokens` on the LMCache `/metrics` endpoint increments on the second call. |
| `cacheblend_integration_smoke_test.go` | vLLM + CacheBlendEngine round-trip — reconciles a `CacheBlendEngine` (blend server DaemonSet), creates an args-only vLLM `Deployment` that opts into CacheBlend injection (label `lmcache.ai/cacheblend-inject` + engine annotation), and asserts: the mutating webhook stamped `cacheblend-injected=true` and injected `--attention-backend CUSTOM`; vLLM logs the CUSTOM backend banner and serves `/v1/models`; the engine logs `Registered CB rope state for instance N` after a completion; the completion returns HTTP 200. Pulls the PRIVATE payload image via a `dockerconfigjson` Secret built from `CACHEBLEND_REGISTRY_USER`/`CACHEBLEND_REGISTRY_TOKEN` — **Skips** if those are unset. |

### Prerequisites

What you need to install / configure before each target. The "Host
config" column is **only needed once per host** — subsequent runs
reuse it. Detailed rationale for each item lives in the per-target
sections below; this table is the quick-reference.

| Target | Tools | Host config | Cluster reqs |
|---|---|---|---|
| `test-e2e-kind` | `kind`, `kubectl`, `docker` | — | (cluster created fresh) |
| `test-e2e-cluster` | `kubectl` | — | `KUBECONFIG` → target cluster; pass `IMG=` (registry the cluster can pull from) |
| `test-e2e-gpu-kind` | `kind`, `kubectl`, `docker`, `helm` | NVIDIA driver + `nvidia-container-toolkit` installed, **plus** the two `nvidia-ctk` commands below | (cluster created fresh) |
| `test-e2e-gpu-cluster` | `kubectl`, `helm` | — | `KUBECONFIG` → GPU cluster; GPU node labeled `nvidia.com/gpu.present=true`; `nvidia` `RuntimeClass` installed; pass `IMG=` |

#### Tool install hints

```bash
# kind
go install sigs.k8s.io/kind/cmd/kind@latest

# kubectl   — distro-specific; e.g. `curl -LO https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl`
# helm v3   — https://helm.sh/docs/intro/install/
# docker    — distro-specific
```

#### GPU host one-time setup (only for `test-e2e-gpu-kind`)

```bash
# Configure docker to default to the nvidia runtime, then toggle the
# volume-mount-marker mechanism the inline Kind config uses to inject
# GPUs into the worker container. Restart docker once at the end.
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled
sudo nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place
sudo systemctl restart docker
```

The target fails fast with a copy-pasteable fix command if either
piece of host config is missing.

**Not needed**: `nvkind`. An earlier iteration used it; the current
target uses the NVIDIA GPU Operator instead and gets GPU passthrough
via an inline Kind config. See `make/e2e-gpu.mk` for the rationale.

### Helper library (`test/utils/`)

| File | Purpose |
|---|---|
| `lmc.go` | `ApplyLMC`, `WaitLMCReconciled`, `WaitLMCPhase`, `WaitLMCReady`, `GetConnectionConfig` (typed parser), `PatchLMCSpec`, `DeleteLMCAndWaitGC` |
| `portforward.go` | `PortForward(spec, ports...)` — wraps `kubectl port-forward`, waits for the local port to accept TCP |
| `fixtures.go` | `go:embed`-backed fixture/golden loader |
| `runner.go` | `RunMake` / `RunFromOperator` — runs commands from the operator/ root **without** `os.Chdir` |
| `utils.go` | Legacy exec helpers retained for the kubebuilder-template Manager spec |

### Adding a new smoke spec

1. Drop the YAML under `test/utils/fixtures/<name>.yaml`.
2. Use `utils.NewLMCFromFixture(...)` to load and override `name`/`namespace`.
3. Call the helpers above; do **not** call `os.Chdir` or read paths
   relative to the working directory — fixtures resolve via `go:embed`,
   and shell commands accept their working directory through `cmd.Dir`.
4. Wrap the spec body with `recordOnFailure(nsName)` in `AfterEach` so
   failures dump controller logs, events, pod descriptions, and the CR yaml.

### GPU tier (M2)

Two entry points share the same `e2e_gpu`-tagged specs:

#### `test-e2e-gpu-kind` — self-contained Kind cluster

Best when you have GPUs on the dev box and don't want to wire up an
external cluster. The target hand-rolls a Kind cluster config that
mounts `/dev/null` at `/var/run/nvidia-container-devices/all` in the
worker — combined with the host setup below,
nvidia-container-runtime sees that marker and injects all GPU
devices + driver libraries into the Kind worker container. Then the
target installs the
[NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator), which
runs a toolkit daemonset that installs nvidia-container-toolkit
*inside the Kind node* and registers a `nvidia` containerd runtime
handler. After that, pods with `runtimeClassName: nvidia` (which the
LMCache DaemonSet and the test-side vLLM Deployment already use)
get NVML / libcuda injected. The cluster is auto-deleted on exit —
same trap pattern as `test-e2e-kind`. All in-cluster manifests
(including the Kind config) are inlined into the Makefile.

We explored several alternatives before settling on GPU Operator:
the bare device plugin alone fails with `Failed to initialize NVML:
ERROR_LIBRARY_NOT_FOUND` because pods scheduled by Kind's inner
containerd don't inherit the worker's library mounts; manually
apt-installing the toolkit via `docker exec` works but breaks if
the kindest/node image changes; nvkind didn't reliably produce
GPU-passthrough markers on the target host. GPU Operator does the
toolkit install as a Kubernetes-native DaemonSet, which is the most
robust path. Trade-off: helm install takes ~10 min (operator +
ClusterPolicy + 5 daemonsets), versus ~30 s for the bare plugin
when it works.

Host one-time setup (NOT automated):

```bash
# 1. NVIDIA driver + nvidia-container-toolkit installed (distro-specific).

# 2. Configure docker + nvidia-container-runtime:
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled
sudo nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place
sudo systemctl restart docker

# 3. helm + kubectl + kind on PATH.
```

Then:

```bash
make test-e2e-gpu-kind
```

The Makefile target fails fast with a copy-pasteable fix command if
either `Default Runtime: nvidia` is missing from `docker info` or
`accept-nvidia-visible-devices-as-volume-mounts=true` is missing
from `/etc/nvidia-container-runtime/config.toml`.

Single-node is intentional: the LMCache DaemonSet and the test-side
vLLM Deployment both schedule onto the same (only) worker, which is
what the kv-cache transfer needs anyway (hostIPC + cudaIPC require
colocation).

**Side effect of step 2 to be aware of**: after the flip, every
docker container on the host — not just Kind workers — starts
through nvidia-container-runtime. Non-GPU workloads still work but
go through one extra hook on startup.

The `nvidia` RuntimeClass is registered by nvkind. Pods that need
GPU access (the LMCache DaemonSet, the test-side vLLM Deployment)
already reference `runtimeClassName: nvidia`.

#### `test-e2e-gpu-cluster` — existing GPU cluster

Use when targeting OpenShift / EKS / GKE GPU clusters. Prerequisites:

1. At least one node has `nvidia.com/gpu.present=true` and the
   `nvidia` RuntimeClass installed.
2. `KUBECONFIG` points at that cluster, the operator image is pushed
   to a registry the cluster can pull from, and `IMG=` references it.
3. The cluster can pull `lmcache/vllm-openai:latest` (default for
   both the LMCache DaemonSet and the test-side vLLM workload).
   Override with `VLLM_IMAGE=` if you mirror it elsewhere.
4. Internet egress for Hugging Face model downloads, OR the model is
   already on the node. Default model is `Qwen/Qwen2.5-0.5B`; override
   with `VLLM_MODEL=<org>/<model>`.
5. To run only the `/conf` spec and skip the heavyweight vLLM
   round-trip, set `SKIP_VLLM_INTEGRATION=true`.

Timeout is 60 min — cold image pulls + model download routinely eat
20+ min before the first inference.

#### CacheBlend integration knobs

`cacheblend_integration_smoke_test.go` pulls a **private** payload image
(`tensormesh/cacheblend-plugin:latest-nightly`). It builds a
`dockerconfigjson` pull Secret in the test namespace from env credentials, so
set these (in Buildkite: pipeline secrets, same mechanism as `HF_TOKEN`):

- `CACHEBLEND_REGISTRY_USER` / `CACHEBLEND_REGISTRY_TOKEN` — registry username +
  read-only PAT. **Unset ⇒ the spec Skips** (keeps credential-less clusters green).
- `CACHEBLEND_REGISTRY_SERVER` — registry server (default
  `https://index.docker.io/v1/` for Docker Hub).
- `CACHEBLEND_PAYLOAD_IMAGE` — override the private plugin image (default
  `tensormesh/cacheblend-plugin:latest-nightly`).
- `CACHEBLEND_ENGINE_IMAGE` — blend server image (default
  `lmcache/vllm-openai:latest-nightly`). `VLLM_IMAGE` (also defaults to
  `latest-nightly` for this spec) / `VLLM_MODEL` apply as above. Engine, vLLM,
  and the latest-nightly payload plugin must share a compatibility window.
- `CACHEBLEND_BACKEND_LOG_PATTERN` — regex proving vLLM loaded the CUSTOM
  attention backend (default `Using AttentionBackendEnum\.CUSTOM backend`).
- `SKIP_CACHEBLEND_INTEGRATION=true` — skip this spec.

---

## Project Structure

**Single-group layout (default):**
```
cmd/main.go                    Manager entry (registers controllers/webhooks)
api/<version>/*_types.go       CRD schemas (+kubebuilder markers)
api/<version>/zz_generated.*   Auto-generated (DO NOT EDIT)
internal/controller/*          Reconciliation logic
internal/webhook/*             Validation/defaulting (if present)
config/crd/bases/*             Generated CRDs (DO NOT EDIT)
config/rbac/role.yaml          Generated RBAC (DO NOT EDIT)
config/samples/*               Example CRs (edit these)
Makefile                       Top-level orchestrator: vars + `include make/*.mk`
make/*.mk                      Targets by concern (dev / e2e / e2e-gpu / lint / build / deploy / tools)
PROJECT                        Kubebuilder metadata Auto-generated (DO NOT EDIT)
```

**Multi-group layout** (for projects with multiple API groups):
```
api/<group>/<version>/*_types.go       CRD schemas by group
internal/controller/<group>/*          Controllers by group
internal/webhook/<group>/<version>/*   Webhooks by group and version (if present)
```

Multi-group layout organizes APIs by group name (e.g., `batch`, `apps`). Check the `PROJECT` file for `multigroup: true`.

**To convert to multi-group layout:**
1. Run: `kubebuilder edit --multigroup=true`
2. Move APIs: `mkdir -p api/<group> && mv api/<version> api/<group>/`
3. Move controllers: `mkdir -p internal/controller/<group> && mv internal/controller/*.go internal/controller/<group>/`
4. Move webhooks (if present): `mkdir -p internal/webhook/<group> && mv internal/webhook/<version> internal/webhook/<group>/`
5. Update import paths in all files
6. Fix `path` in `PROJECT` file for each resource
7. Update test suite CRD paths (add one more `..` to relative paths)

## Critical Rules

### Never Edit These (Auto-Generated)
- `config/crd/bases/*.yaml` - from `make manifests`
- `config/rbac/role.yaml` - from `make manifests`
- `config/webhook/manifests.yaml` - from `make manifests`
- `**/zz_generated.*.go` - from `make generate`
- `PROJECT` - from `kubebuilder [OPTIONS]`

### Never Remove Scaffold Markers
Do NOT delete `// +kubebuilder:scaffold:*` comments. CLI injects code at these markers.

### Keep Project Structure
Do not move files around. The CLI expects files in specific locations.

### Always Use CLI Commands
Always use `kubebuilder create api` and `kubebuilder create webhook` to scaffold. Do NOT create files manually.

### E2E Tests Require an Isolated Kind Cluster
The e2e tests are designed to validate the solution in an isolated environment (similar to GitHub Actions CI).
Ensure you run them against a dedicated [Kind](https://kind.sigs.k8s.io/) cluster (not your “real” dev/prod cluster).

## After Making Changes

**After editing `*_types.go` or markers:**
```
make manifests  # Regenerate CRDs/RBAC from markers
make generate   # Regenerate DeepCopy methods
```

**After editing `*.go` files:**
```
make lint-fix   # Auto-fix code style
make test       # Run unit tests
```

## CLI Commands Cheat Sheet

### Create API (your own types)
```bash
kubebuilder create api --group <group> --version <version> --kind <Kind>
```


### Deploy Image Plugin (scaffold to deploy/manage ANY container image)

Generate a controller that deploys and manages a container image (nginx, redis, memcached, your app, etc.):

```bash
# Example: deploying memcached
kubebuilder create api --group example.com --version v1alpha1 --kind Memcached \
  --image=memcached:alpine \
  --plugins=deploy-image.go.kubebuilder.io/v1-alpha
```

Scaffolds good-practice code: reconciliation logic, status conditions, finalizers, RBAC. Use as a reference implementation.


### Create Webhooks
```bash
# Validation + defaulting
kubebuilder create webhook --group <group> --version <version> --kind <Kind> \
  --defaulting --programmatic-validation

# Conversion webhook (for multi-version APIs)
kubebuilder create webhook --group <group> --version v1 --kind <Kind> \
  --conversion --spoke v2
```

### Controller for Core Kubernetes Types
```bash
# Watch Pods
kubebuilder create api --group core --version v1 --kind Pod \
  --controller=true --resource=false

# Watch Deployments
kubebuilder create api --group apps --version v1 --kind Deployment \
  --controller=true --resource=false
```

### Controller for External Types (e.g., from other operators)

Watch resources from external APIs (cert-manager, Argo CD, Istio, etc.):

```bash
# Example: watching cert-manager Certificate resources
kubebuilder create api \
  --group cert-manager --version v1 --kind Certificate \
  --controller=true --resource=false \
  --external-api-path=github.com/cert-manager/cert-manager/pkg/apis/certmanager/v1 \
  --external-api-domain=io \
  --external-api-module=github.com/cert-manager/cert-manager
```

**Note:** Use `--external-api-module=<module>@<version>` only if you need a specific version. Otherwise, omit `@<version>` to use what's in go.mod.

### Webhook for External Types

```bash
# Example: validating external resources
kubebuilder create webhook \
  --group cert-manager --version v1 --kind Issuer \
  --defaulting \
  --external-api-path=github.com/cert-manager/cert-manager/pkg/apis/certmanager/v1 \
  --external-api-domain=io \
  --external-api-module=github.com/cert-manager/cert-manager
```

## Testing & Development

```bash
make test              # Run unit tests (uses envtest: real K8s API + etcd)
make run               # Run locally (uses current kubeconfig context)
```

Tests use **Ginkgo + Gomega** (BDD style). Check `suite_test.go` for setup.

## Deployment Workflow

```bash
# 1. Regenerate manifests
make manifests generate

# 2. Build & deploy
export IMG=<registry>/<project>:tag
make docker-build docker-push IMG=$IMG  # Or: kind load docker-image $IMG --name <cluster>
make deploy IMG=$IMG

# 3. Test
kubectl apply -k config/samples/

# 4. Debug
kubectl logs -n <project>-system deployment/<project>-controller-manager -c manager -f
```

### API Design

**Key markers for** `api/<version>/*_types.go`:

```go
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced
// +kubebuilder:printcolumn:name="Status",type=string,JSONPath=".status.conditions[?(@.type=='Ready')].status"

// On fields:
// +kubebuilder:validation:Required
// +kubebuilder:validation:Minimum=1
// +kubebuilder:validation:MaxLength=100
// +kubebuilder:validation:Pattern="^[a-z]+$"
// +kubebuilder:default="value"
```

- **Use** `metav1.Condition` for status (not custom string fields)
- **Use predefined types**: `metav1.Time` instead of `string` for dates
- **Follow K8s API conventions**: Standard field names (`spec`, `status`, `metadata`)

### Controller Design

**RBAC markers in** `internal/controller/*_controller.go`:

```go
// +kubebuilder:rbac:groups=mygroup.example.com,resources=mykinds,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=mygroup.example.com,resources=mykinds/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=mygroup.example.com,resources=mykinds/finalizers,verbs=update
// +kubebuilder:rbac:groups=events.k8s.io,resources=events,verbs=create;patch
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
```

**Implementation rules:**
- **Idempotent reconciliation**: Safe to run multiple times
- **Re-fetch before updates**: `r.Get(ctx, req.NamespacedName, obj)` before `r.Update` to avoid conflicts
- **Structured logging**: `log := log.FromContext(ctx); log.Info("msg", "key", val)`
- **Owner references**: Enable automatic garbage collection (`SetControllerReference`)
- **Watch secondary resources**: Use `.Owns()` or `.Watches()`, not just `RequeueAfter`
- **Finalizers**: Clean up external resources (buckets, VMs, DNS entries)

### Logging

**Follow Kubernetes logging message style guidelines:**

- Start from a capital letter
- Do not end the message with a period
- Active voice: subject present (`"Deployment could not create Pod"`) or omitted (`"Could not create Pod"`)
- Past tense: `"Could not delete Pod"` not `"Cannot delete Pod"`
- Specify object type: `"Deleted Pod"` not `"Deleted"`
- Balanced key-value pairs

```go
log.Info("Starting reconciliation")
log.Info("Created Deployment", "name", deploy.Name)
log.Error(err, "Failed to create Pod", "name", name)
```

**Reference:** https://github.com/kubernetes/community/blob/master/contributors/devel/sig-instrumentation/logging.md#message-style-guidelines

### Webhooks
- **Create all types together**: `--defaulting --programmatic-validation --conversion`
- **When`--force`is used**: Backup custom logic first, then restore after scaffolding
- **For multi-version APIs**: Use hub-and-spoke pattern (`--conversion --spoke v2`)
  - Hub version: Usually oldest stable version (v1)
  - Spoke versions: Newer versions that convert to/from hub (v2, v3)
  - Example: `--group crew --version v1 --kind Captain --conversion --spoke v2` (v1 is hub, v2 is spoke)

### Learning from Examples

The **deploy-image plugin** scaffolds a complete controller following good practices. Use it as a reference implementation:

```bash
kubebuilder create api --group example --version v1alpha1 --kind MyApp \
  --image=<your-image> --plugins=deploy-image.go.kubebuilder.io/v1-alpha
```

Generated code includes: status conditions (`metav1.Condition`), finalizers, owner references, events, idempotent reconciliation.

## Distribution Options

### Option 1: YAML Bundle (Kustomize)

```bash
# Generate dist/install.yaml from Kustomize manifests
make build-installer IMG=<registry>/<project>:tag
```

**Key points:**
- The `dist/install.yaml` is generated from Kustomize manifests (CRDs, RBAC, Deployment)
- Commit this file to your repository for easy distribution
- Users only need `kubectl` to install (no additional tools required)

**Example:** Users install with a single command:
```bash
kubectl apply -f https://raw.githubusercontent.com/<org>/<repo>/<tag>/dist/install.yaml
```

### Option 2: Helm Chart

```bash
kubebuilder edit --plugins=helm/v2-alpha                      # Generates dist/chart/ (default)
kubebuilder edit --plugins=helm/v2-alpha --output-dir=charts  # Generates charts/chart/
```

**For development:**
```bash
make helm-deploy IMG=<registry>/<project>:<tag>          # Deploy manager via Helm
make helm-deploy IMG=$IMG HELM_EXTRA_ARGS="--set ..."    # Deploy with custom values
make helm-status                                         # Show release status
make helm-uninstall                                      # Remove release
make helm-history                                        # View release history
make helm-rollback                                       # Rollback to previous version
```

**For end users/production:**
```bash
helm install my-release ./<output-dir>/chart/ --namespace <ns> --create-namespace
```

**Important:** If you add webhooks or modify manifests after initial chart generation:
1. Backup any customizations in `<output-dir>/chart/values.yaml` and `<output-dir>/chart/manager/manager.yaml`
2. Re-run: `kubebuilder edit --plugins=helm/v2-alpha --force` (use same `--output-dir` if customized)
3. Manually restore your custom values from the backup

### Publish Container Image

```bash
export IMG=<registry>/<project>:<version>
make docker-build docker-push IMG=$IMG
```

## References

### Essential Reading
- **Kubebuilder Book**: https://book.kubebuilder.io (comprehensive guide)
- **controller-runtime FAQ**: https://github.com/kubernetes-sigs/controller-runtime/blob/main/FAQ.md (common patterns and questions)
- **Good Practices**: https://book.kubebuilder.io/reference/good-practices.html (why reconciliation is idempotent, status conditions, etc.)
- **Logging Conventions**: https://github.com/kubernetes/community/blob/master/contributors/devel/sig-instrumentation/logging.md#message-style-guidelines (message style, verbosity levels)

### API Design & Implementation
- **API Conventions**: https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md
- **Operator Pattern**: https://kubernetes.io/docs/concepts/extend-kubernetes/operator/
- **Markers Reference**: https://book.kubebuilder.io/reference/markers.html

### Tools & Libraries
- **controller-runtime**: https://github.com/kubernetes-sigs/controller-runtime
- **controller-tools**: https://github.com/kubernetes-sigs/controller-tools
- **Kubebuilder Repo**: https://github.com/kubernetes-sigs/kubebuilder
