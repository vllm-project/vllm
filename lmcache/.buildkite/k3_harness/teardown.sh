#!/usr/bin/env bash
# Tear down K3s stack. Removes: agent-stack-k8s, GPU Operator, K3s.
# Host data volumes (/data/*) are NOT deleted.
set -euo pipefail

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

echo "=== Teardown ==="

# agent-stack-k8s
if helm status agent-stack-k8s -n buildkite &>/dev/null; then
    echo "→ Removing agent-stack-k8s..."
    helm uninstall agent-stack-k8s -n buildkite --wait
fi

# GPU Operator
if helm status gpu-operator -n gpu-operator &>/dev/null; then
    echo "→ Removing GPU Operator..."
    helm uninstall gpu-operator -n gpu-operator --wait
    kubectl delete namespace gpu-operator --ignore-not-found
fi

# K3s
if command -v k3s-uninstall.sh &>/dev/null; then
    echo "→ Removing K3s..."
    /usr/local/bin/k3s-uninstall.sh
fi

echo "✓ Teardown complete"
echo "  Note: /data/{huggingface,datasets} preserved"
