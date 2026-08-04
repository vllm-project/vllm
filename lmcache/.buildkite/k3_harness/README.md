# K3s CI Harness

Single-node K3s + NVIDIA GPU Operator for running LMCache CI on any GPU machine.

## Prerequisites

- Linux host with NVIDIA GPU(s) and driver installed
- Docker (for building the CI base image)
- Root access
- A GitHub token (PAT or fine-grained) with read/write access to the repo

## Setup (one-time)

```bash
# 1. Install K3s + GPU Operator + build CI base image
.buildkite/k3_harness/setup-cluster.sh

# 2. Verify GPUs work in pods
.buildkite/k3_harness/smoke-test.sh

# 3. Connect to Buildkite (needs agent token + GitHub token for HTTPS checkout)
.buildkite/k3_harness/install-agent-stack.sh <BUILDKITE_AGENT_TOKEN> <GITHUB_TOKEN>
```

`setup-cluster.sh` does everything: installs K3s, Helm, GPU Operator, builds the CI base image from `ci-base.Dockerfile`, imports it into K3s containerd locally, and creates host volume directories. Safe to re-run.

## How Buildkite integration works

This is different from the bare-metal agent setup where you install a Buildkite agent binary on your own machine, register it with a queue created on the Web UI, and manage it as a systemd service. With agent-stack-k8s, there are **no persistent agents on the machine**.

Instead, agent-stack-k8s runs a controller pod in K8s that polls Buildkite for jobs matching the configured queue. When a job appears, it creates a K8s pod to run it. When the job finishes, the pod is deleted.

**What you need from the Buildkite web UI**:

1. **Create a queue** — Go to **Organization Settings → Default cluster → Queues → New Queue** and create a queue named `k8s` (or your chosen name). The queue needs no configuration — just a name. You do **not** register any agents on it.
2. **Get an agent token** — From the cluster settings page, copy the agent token (or create a new one).
3. **Get a GitHub token** — Create a PAT (or fine-grained token) with read/write access to the repo. This is used for HTTPS checkout and for pushing baselines to `benchmarks-main`.
4. **Run `install-agent-stack.sh`** with the agent token and GitHub token. The script creates a K8s secret and installs agent-stack-k8s.

To use a queue name other than `k8s`, set `BUILDKITE_QUEUE` when running install:
```bash
BUILDKITE_QUEUE=my-queue .buildkite/k3_harness/install-agent-stack.sh <AGENT_TOKEN> <GITHUB_TOKEN>
```

Pipeline steps target the queue with:
```yaml
agents:
  queue: "k8s"   # must match the queue name
```

## Per-job environment

Every CI job sources `setup-env.sh` to install vLLM + LMCache:

```yaml
command: |
  source .buildkite/k3_harness/setup-env.sh
  bash .buildkite/scripts/my-test.sh
```

This installs:
1. **vLLM** from nightly wheel
2. **LMCache** from PR source

Each pod has its own ephemeral filesystem that is cleanly erased after being shut down — no shared pip/uv cache, no cross-pod contention.

## Shared volumes

Mounted into every pod via `hostPath`:

| Host | Container | What |
|------|-----------|------|
| `/data/huggingface` | `/root/.cache/huggingface` | Model weights (read-heavy, write-once) |
| `/data/datasets` | `/root/correctness` | Test datasets (read-only after download) |

## GPU allocation

Request GPUs per pipeline step:

```yaml
plugins:
  - kubernetes:
      podSpec:
        containers:
          - resources:
              limits:
                nvidia.com/gpu: "2"  # 1 or 2
```

K8s device plugin handles atomic allocation. No `pick-free-gpu.sh`.

## CI base image

`ci-base.Dockerfile` builds an image with CUDA + Python 3.12 + uv + build deps (no vLLM/LMCache). `setup-cluster.sh` builds it automatically, auto-detects your GPU's compute capability for `TORCH_CUDA_ARCH_LIST`, and imports it into K3s containerd — no registry needed.

To rebuild after changing `requirements/*.txt`:
```bash
REBUILD_IMAGE=1 .buildkite/k3_harness/setup-cluster.sh
```

## Files

```
k3_harness/
├── ci-base.Dockerfile      # CI base image definition
├── setup-cluster.sh        # One-time: K3s + GPU Operator + base image
├── install-agent-stack.sh  # One-time: Buildkite agent (needs token + GitHub token)
├── values.yaml             # Reference Helm values (documentation only)
├── setup-env.sh            # Per-job: install vLLM + LMCache (includes GPU health check)
├── smoke-test.sh           # Verify: GPU pod runs in K3s
├── gpu-monitor.sh          # Host-level: detect stale non-K8s GPU processes
├── setup-gpu-monitor.sh    # Install gpu-monitor.sh as a cron job
├── teardown.sh             # Remove everything (preserves /data/*)
└── README.md
```

## GPU monitoring

CI jobs automatically check GPU health at startup (via `check_gpu_health` in `setup-env.sh`). If a GPU assigned to a pod has less than 80% free memory — usually due to a stale host process — the job fails immediately with a clear message instead of wasting time on setup and then getting a cryptic CUDA OOM.

For **host-level monitoring**, install the GPU monitor cron job:

```bash
# Install cron job that checks every 10 minutes for stale non-K8s GPU processes
sudo bash .buildkite/k3_harness/setup-gpu-monitor.sh
```

This runs `gpu-monitor.sh` every 10 minutes. It distinguishes K8s pod processes (in `kubepods` cgroups) from stale host processes and logs warnings to `/var/log/gpu-monitor.log` with PID, command, GPU, memory usage, and process age.

```bash
# View monitor logs
tail -f /var/log/gpu-monitor.log

# Remove the cron job
crontab -l 2>/dev/null | grep -v gpu-monitor.sh | crontab -
```

## Teardown

```bash
.buildkite/k3_harness/teardown.sh
# Removes K3s, GPU Operator, agent-stack. Preserves /data/*
```
