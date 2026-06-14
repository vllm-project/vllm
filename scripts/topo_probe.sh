#!/usr/bin/env bash
set -euo pipefail

echo "=== vLLM 3080 Ti PCIe topology probe ==="
echo

echo "[1/5] nvidia-smi topo -m"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi topo -m || true
else
  echo "nvidia-smi not found"
fi

echo
echo "[2/5] GPU PCIe summary"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,pci.bus_id,pstate,pcie.link.gen.current,pcie.link.width.current,memory.total --format=csv || true
fi

echo
echo "[3/5] CUDA_VISIBLE_DEVICES"
echo "${CUDA_VISIBLE_DEVICES:-<unset>}"

echo
echo "[4/5] NUMA topology"
if command -v numactl >/dev/null 2>&1; then
  numactl --hardware || true
else
  echo "numactl not found"
fi

echo
echo "[5/5] NVIDIA PCI devices"
if command -v lspci >/dev/null 2>&1; then
  lspci | grep -i nvidia || true
else
  echo "lspci not found"
fi

echo
echo "Probe complete. Save this output with benchmark logs for TP/KV route comparisons."
