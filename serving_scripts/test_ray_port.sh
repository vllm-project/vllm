#!/usr/bin/env bash
#SBATCH --nodelist=htc-g[059-060]
#SBATCH --job-name=test_ray_port
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err
#SBATCH --mail-user=jason.miller@eng.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=engs-glass
#SBATCH --qos=priority

set -euo pipefail

SCRIPT_VERSION="test-ray-dynamic-port-pp2-tp1-v1"

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
RAY_BIN="${VENV_DIR}/bin/ray"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Error: missing venv python at ${PYTHON_BIN}" >&2
  exit 1
fi

if [ ! -x "${RAY_BIN}" ]; then
  echo "Error: missing ray binary at ${RAY_BIN}" >&2
  exit 1
fi

HEAD_NODE="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"
WORKER_NODES="$(scontrol show hostnames "${SLURM_NODELIST}" | tail -n+2)"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}"

# Smoke-test labels for the target layout.
TP="${TP:-1}"
PP="${PP:-2}"

resolve_host_ip() {
  local nodename="$1"
  local ip=""

  ip="$(dig +short "${nodename}" 2>/dev/null | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}' || true)"
  if [ -z "${ip}" ]; then
    ip="$(getent hosts "${nodename}" 2>/dev/null | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/ {print $1; exit}' || true)"
  fi
  if [ -z "${ip}" ]; then
    ip="$(
      srun --nodelist="${nodename}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${CPUS_PER_TASK}" \
        bash -lc "hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'" \
        2>/dev/null || true
    )"
  fi

  printf '%s' "${ip}"
}

HEAD_NODE_IP="$(resolve_host_ip "${HEAD_NODE}")"
if [ -z "${HEAD_NODE_IP}" ]; then
  echo "Error: could not resolve head node IP for ${HEAD_NODE}" >&2
  exit 1
fi

export RAY_PORT="${RAY_PORT:-$((6300 + (${SLURM_JOB_ID:-0} % 1000)))}"
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

RAY_TMP_BASE="${RAY_TMP_BASE:-/tmp/vray-port-test-${SLURM_JOB_ID:-manual}}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-20000000000}"

echo "=== Ray dynamic-port smoke test ==="
echo "SCRIPT_VERSION=${SCRIPT_VERSION}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "HEAD_NODE=${HEAD_NODE}"
echo "WORKER_NODES=${WORKER_NODES}"
echo "HEAD_NODE_IP=${HEAD_NODE_IP}"
echo "RAY_PORT=${RAY_PORT}"
echo "RAY_ADDRESS=${RAY_ADDRESS}"
echo "TP=${TP} PP=${PP} GPUS_PER_NODE=${GPUS_PER_NODE} CPUS_PER_TASK=${CPUS_PER_TASK}"

HEAD_RAY_PID=""
WORKER_RAY_PIDS=""

cleanup() {
  for pid in ${WORKER_RAY_PIDS}; do
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
  if [ -n "${HEAD_RAY_PID}" ]; then
    kill "${HEAD_RAY_PID}" 2>/dev/null || true
    wait "${HEAD_RAY_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "=== starting Ray head ==="
srun \
  --nodelist "${HEAD_NODE}" \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-task="${GPUS_PER_NODE}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  bash -lc "\
    '${RAY_BIN}' start --block \
      --head \
      --node-ip-address='${HEAD_NODE_IP}' \
      --port='${RAY_PORT}' \
      --temp-dir='${RAY_TMP_BASE}-${HEAD_NODE}' \
      --plasma-directory=/dev/shm \
      --object-store-memory='${RAY_OBJECT_STORE_MEMORY}' \
      --num-gpus='${GPUS_PER_NODE}' \
      --num-cpus='${CPUS_PER_TASK}'" &
HEAD_RAY_PID="$!"

sleep 10

echo "=== starting Ray workers ==="
for WORKER in ${WORKER_NODES}; do
  WORKER_IP="$(resolve_host_ip "${WORKER}")"
  if [ -z "${WORKER_IP}" ]; then
    echo "Error: could not resolve worker IP for ${WORKER}" >&2
    exit 1
  fi

  echo "Starting worker ${WORKER} at ${WORKER_IP}"
  srun \
    --nodelist "${WORKER}" \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --gpus-per-task="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    bash -lc "\
      '${RAY_BIN}' start --block \
        --address='${RAY_ADDRESS}' \
        --node-ip-address='${WORKER_IP}' \
        --temp-dir='${RAY_TMP_BASE}-${WORKER}' \
        --plasma-directory=/dev/shm \
        --object-store-memory='${RAY_OBJECT_STORE_MEMORY}' \
        --num-gpus='${GPUS_PER_NODE}' \
        --num-cpus='${CPUS_PER_TASK}'" &
  WORKER_RAY_PIDS="${WORKER_RAY_PIDS} $!"
done

echo "=== waiting for Ray nodes ==="
"${PYTHON_BIN}" - <<'PY'
import os
import sys
import time

import ray

address = os.environ["RAY_ADDRESS"]
expected_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES") or os.environ.get("SLURM_NNODES") or "2")
deadline = time.time() + int(os.environ.get("RAY_WAIT_TIMEOUT_S", "180"))
last_error = None

while time.time() < deadline:
    try:
        if not ray.is_initialized():
            ray.init(address=address)
        nodes = ray.nodes()
        alive = [n for n in nodes if n.get("Alive")]
        print(f"alive_nodes={len(alive)}/{expected_nodes}")
        for node in alive:
            print(
                "  "
                f"{node.get('NodeManagerAddress')} "
                f"resources={node.get('Resources')}"
            )
        if len(alive) >= expected_nodes:
            print("Ray dynamic-port smoke test passed.")
            sys.exit(0)
    except Exception as exc:  # noqa: BLE001 - printed for batch diagnostics.
        last_error = exc
        print(f"Ray not ready yet: {type(exc).__name__}: {exc}", flush=True)
    time.sleep(5)

print(f"Timed out waiting for {expected_nodes} Ray nodes at {address}. Last error: {last_error}", file=sys.stderr)
sys.exit(1)
PY

echo "=== ray status ==="
"${RAY_BIN}" status --address="${RAY_ADDRESS}" || true

echo "Done."
