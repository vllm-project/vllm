# K3s CI Harness Architecture

This document outlines the main motivations, design decisions, and architecture of the K3s-based Buildkite CI Harness for LMCache.

## Motivations

1. **Ease of Setup**: Bring up a fully functional CI node on any raw Linux GPU machine with a single script (`setup-cluster.sh`), avoiding manual configuration of dependencies or persistent Buildkite agents.
2. **Machine Agnostic**: Remove reliance on machine-specific scripts (e.g., custom `pick-free-gpu.sh` scripts). The CI runs identically on any machine with an NVIDIA GPU and Docker.
3. **Clean Environments**: Eliminate state bleeding between test runs (no shared pip/uv caches or leftover files). Every job gets a pristine environment.
4. **Automated Resource Management**: Leverage standard Kubernetes primitives for GPU allocation, volume mounting, and cleanup.

## Core Design Decisions

### 1. K3s + GPU Operator
We use **K3s** as a lightweight, single-node Kubernetes cluster. The **NVIDIA GPU Operator** automatically configures the container runtime and exposes GPUs to K8s. This abstracts away the underlying hardware and handles GPU discovery automatically.

### 2. Ephemeral Pod-Based Execution (`agent-stack-k8s`)
Instead of running persistent `buildkite-agent` binaries, we use Buildkite's `agent-stack-k8s`. A controller pod polls the Buildkite queue and dynamically provisions an ephemeral K8s Pod to run each job. When the job finishes, the pod is destroyed.

### 3. Declarative GPU Allocation & Parallelizability
Each pipeline step explicitly requests the number of GPUs it needs in its Kubernetes pod specification:

```yaml
resources:
  limits:
    nvidia.com/gpu: "2"  # Requests 2 GPUs
```

Because Kubernetes handles atomic resource scheduling, **jobs automatically run in parallel on multi-GPU nodes**. If a node has 4 GPUs and three jobs arrive (requesting 1, 2, and 1 GPU respectively), K8s schedules them concurrently. If another job requests 2 GPUs, it waits in the queue until resources free up. This completely eliminates the need for manual GPU locking mechanisms.

### 4. Local Base Image & Ephemeral Setup
To avoid Docker registry dependencies, `setup-cluster.sh` automatically detects the host GPU's compute capability, builds the `lmcache/ci-base:latest` image locally, and imports it directly into K3s. Each job pod uses this base image and dynamically installs vLLM and LMCache (`setup-env.sh`) at runtime.

### 5. Shared Host Volumes
Large, read-heavy directories (like HuggingFace models and datasets) are mounted into pods via `hostPath` to speed up tests without duplicating downloads, while keeping the job environment itself stateless.

---

## Architecture Flow

### Node Initialization (One-Time Setup)

```text
[ Raw Linux Host + NVIDIA GPU ]
          |
          | (run setup-cluster.sh)
          v
+-----------------------------------------------------+
|                     K3s Cluster                     |
|                                                     |
|  1. Install K3s                                     |
|  2. Install Helm -> Deploy NVIDIA GPU Operator      |
|  3. Build CI Base Image (lmcache/ci-base:latest)    |
|  4. Import Base Image to K3s containerd             |
+-----------------------------------------------------+
          |
          | (run install-agent-stack.sh)
          v
[ agent-stack-k8s Controller Pod (Watches queue) ]
```

### CI Execution Flow

```text
Buildkite Web UI
       |
       | 1. Job pushed to 'k8s' queue
       v
agent-stack-k8s Controller (in K3s)
       |
       | 2. Reads job, creates ephemeral Pod requesting GPUs
       v
+-------------------------------------------------------------+
| Job Pod (e.g., limits: nvidia.com/gpu: "1")                 |
|                                                             |
|  - Mounts /data/huggingface from host                       |
|  - Runs setup-env.sh (Installs vLLM, LMCache)               |
|  - Executes test script (e.g., pytest)                      |
+-------------------------------------------------------------+
       |
       | 3. Test finishes (Pass/Fail)
       v
agent-stack-k8s Controller
       |
       | 4. Reports result to Buildkite
       | 5. Destroys the Pod (wipes environment)
       v
Buildkite Web UI
```