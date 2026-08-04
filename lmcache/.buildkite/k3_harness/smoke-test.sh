#!/usr/bin/env bash
# Verify K3s + GPU Operator works: schedule a pod that runs nvidia-smi.
set -euo pipefail

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

echo "=== Smoke Test ==="

# 1. Check cluster
echo "→ Cluster"
kubectl get nodes -o wide
echo ""

# 2. Check GPU resources
GPU_COUNT=$(kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}')
echo "→ GPUs allocatable: ${GPU_COUNT}"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
    echo "✗ No GPUs found on node"
    exit 1
fi

# 3. Run GPU pod
echo "→ Launching GPU test pod..."
kubectl delete pod gpu-smoke-test --ignore-not-found --wait=false &>/dev/null

cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-smoke-test
spec:
  restartPolicy: Never
  containers:
    - name: test
      image: nvidia/cuda:12.8.0-base-ubuntu24.04
      command: ["nvidia-smi"]
      resources:
        limits:
          nvidia.com/gpu: "1"
EOF

kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/gpu-smoke-test --timeout=120s
echo ""
echo "→ Pod logs:"
kubectl logs gpu-smoke-test
kubectl delete pod gpu-smoke-test

echo ""
echo "=== Smoke test passed ==="
