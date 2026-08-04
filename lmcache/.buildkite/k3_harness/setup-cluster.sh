#!/usr/bin/env bash
# Idempotent K3s + GPU Operator + CI base image setup.
# Safe to re-run — skips components already installed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

CI_BASE_IMAGE="lmcache/ci-base:latest"
DATA_DIR="/data"

# ── K3s ──────────────────────────────────────────────────────
if command -v k3s &>/dev/null && systemctl is-active --quiet k3s; then
    echo "✓ K3s already running"
else
    echo "→ Installing K3s..."
    curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server \
        --disable=traefik \
        --write-kubeconfig-mode=644" sh -
    echo "✓ K3s installed"
fi

# Persist KUBECONFIG for interactive shells
grep -q 'KUBECONFIG=/etc/rancher/k3s/k3s.yaml' ~/.bashrc 2>/dev/null || \
    echo 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml' >> ~/.bashrc

kubectl get nodes
echo ""

# ── Helm ─────────────────────────────────────────────────────
if ! command -v helm &>/dev/null; then
    echo "→ Installing Helm..."
    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi
echo "✓ Helm $(helm version --short)"

# ── NVIDIA GPU Operator ──────────────────────────────────────
if helm status gpu-operator -n gpu-operator &>/dev/null; then
    echo "✓ GPU Operator already installed"
else
    echo "→ Installing GPU Operator..."
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
    helm repo update nvidia
    helm install gpu-operator nvidia/gpu-operator \
        --namespace gpu-operator --create-namespace \
        --set driver.enabled=false \
        --set toolkit.enabled=true \
        --wait --timeout 5m
    echo "✓ GPU Operator installed"
fi

echo "→ Waiting for device plugin..."
kubectl wait --for=condition=ready pod \
    -l app=nvidia-device-plugin-daemonset \
    -n gpu-operator --timeout=120s

GPU_COUNT=$(kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}')
echo "✓ GPUs available: ${GPU_COUNT}"

# ── CI base image ────────────────────────────────────────────
# Auto-detect GPU compute capability for TORCH_CUDA_ARCH_LIST
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
ARCH_LIST="${COMPUTE_CAP}+PTX"

# Check if image already exists in K3s containerd
if k3s ctr images check | grep -q "${CI_BASE_IMAGE}"; then
    echo "✓ CI base image already imported (${CI_BASE_IMAGE})"
    echo "  To rebuild: run with REBUILD_IMAGE=1"
    if [[ "${REBUILD_IMAGE:-}" != "1" ]]; then
        SKIP_IMAGE=1
    fi
fi

if [[ "${SKIP_IMAGE:-}" != "1" ]]; then
    echo "→ Building CI base image (TORCH_CUDA_ARCH_LIST=${ARCH_LIST})..."
    docker build \
        -f "${SCRIPT_DIR}/ci-base.Dockerfile" \
        --build-arg TORCH_CUDA_ARCH_LIST="${ARCH_LIST}" \
        -t "${CI_BASE_IMAGE}" \
        "${REPO_ROOT}"

    echo "→ Importing into K3s containerd..."
    docker save "${CI_BASE_IMAGE}" | k3s ctr images import -
    echo "✓ CI base image ready"
fi

# ── Shared host volumes ──────────────────────────────────────
mkdir -p "${DATA_DIR}/huggingface" "${DATA_DIR}/datasets"
echo "✓ Host volumes: ${DATA_DIR}/{huggingface,datasets}"

echo ""
echo "=== Cluster ready ==="
echo "  K3s:          $(k3s --version | head -1)"
echo "  GPU Operator: $(helm list -n gpu-operator -o json | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["app_version"])' 2>/dev/null || echo 'unknown')"
echo "  GPUs:         ${GPU_COUNT}x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Arch list:    ${ARCH_LIST}"
echo "  Base image:   ${CI_BASE_IMAGE}"
echo "  Data dir:     ${DATA_DIR}"
echo ""
echo "Next: run ./install-agent-stack.sh <BUILDKITE_AGENT_TOKEN>"
