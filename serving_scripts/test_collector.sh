#!/usr/bin/env bash
#SBATCH --nodelist=htc-g[059-060]
#SBATCH --job-name=nccl-inspector-smoke
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err
#SBATCH --mail-user=jason.miller@eng.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=engs-glass
#SBATCH --qos=priority
#SBATCH --reservation=engs-glass

set -euo pipefail

SCRIPT_VERSION="arc-nccl-inspector-smoke-v1-pytorch-nccl-cu130"

echo "=== NCCL Inspector smoke test job ==="
echo "SCRIPT_VERSION=${SCRIPT_VERSION}"
echo "Date: $(date -Is 2>/dev/null || date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

REPO_ROOT="${REPO_ROOT:-/data/engs-glass/catz0932/inference-traces/vllm}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

TRACE_BASE="${TRACE_BASE:-/data/engs-glass/catz0932/inference-traces/vllm/results}"
TRACE_RUN_DIR="${TRACE_BASE}/${SLURM_JOB_ID}"

NCCL_INSPECTOR_PLUGIN="${NCCL_INSPECTOR_PLUGIN:-/data/engs-glass/catz0932/inference-traces/nccl/plugins/profiler/inspector/libnccl-profiler-inspector.so}"
NCCL_INSPECTOR_BASE_DIR="${NCCL_INSPECTOR_BASE_DIR:-${TRACE_RUN_DIR}/nccl_inspector}"
NCCL_INSPECTOR_SMOKE_DIR="${NCCL_INSPECTOR_SMOKE_DIR:-${NCCL_INSPECTOR_BASE_DIR}/smoke}"

mkdir -p "${TRACE_RUN_DIR}/nccl_logs"
mkdir -p "${NCCL_INSPECTOR_SMOKE_DIR}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "VENV_DIR=${VENV_DIR}"
echo "TRACE_RUN_DIR=${TRACE_RUN_DIR}"
echo "NCCL_INSPECTOR_PLUGIN=${NCCL_INSPECTOR_PLUGIN}"
echo "NCCL_INSPECTOR_SMOKE_DIR=${NCCL_INSPECTOR_SMOKE_DIR}"
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
echo "nvcc=$(command -v nvcc || echo '<not found>')"
nvcc --version || true

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "ERROR: venv python not found: ${VENV_DIR}/bin/python" >&2
  exit 1
fi

if [ ! -f "${NCCL_INSPECTOR_PLUGIN}" ]; then
  echo "ERROR: NCCL Inspector plugin missing: ${NCCL_INSPECTOR_PLUGIN}" >&2
  exit 1
fi

export HEAD_NODE
HEAD_NODE="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"

export WORKER_NODES
WORKER_NODES="$(scontrol show hostnames "${SLURM_NODELIST}" | tail -n+2)"

resolve_host_ip() {
  local nodename="$1"
  local ip=""

  pick_ipv4() {
    awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'
  }

  ip="$(dig +short "${nodename}" 2>/dev/null | pick_ipv4 || true)"
  if [ -z "${ip}" ]; then
    ip="$(getent hosts "${nodename}" 2>/dev/null | awk '{print $1}' | pick_ipv4 || true)"
  fi
  if [ -z "${ip}" ]; then
    ip="$(
      srun --nodelist="${nodename}" --nodes=1 --ntasks=1 --cpus-per-task=1 \
        bash -lc "hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'" 2>/dev/null || true
    )"
  fi

  printf '%s' "${ip}"
}

export HEAD_NODE_IP
HEAD_NODE_IP="$(resolve_host_ip "${HEAD_NODE}")"

if [ -z "${HEAD_NODE_IP}" ]; then
  echo "ERROR: could not resolve IPv4 address for head node ${HEAD_NODE}" >&2
  exit 1
fi

echo "HEAD_NODE=${HEAD_NODE}"
echo "WORKER_NODES=${WORKER_NODES}"
echo "HEAD_NODE_IP=${HEAD_NODE_IP}"

echo "=== PyTorch/NCCL provenance on batch host ==="
source "${VENV_DIR}/bin/activate"
python - <<'PY'
import glob
import os
import site
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch nccl:", torch.cuda.nccl.version())
print("torch path:", torch.__file__)

for base in site.getsitepackages():
    for p in glob.glob(base + "/**/libnccl.so*", recursive=True):
        print("libnccl:", p)
    for p in glob.glob(base + "/**/nccl.h", recursive=True):
        print("nccl.h:", p)
PY

echo "=== Plugin dependency check ==="
ls -lh "${NCCL_INSPECTOR_PLUGIN}"
ldd "${NCCL_INSPECTOR_PLUGIN}" | egrep 'cuda|cudart|nccl|stdc|gcc|pthread|dl|not found' || true

echo "=== Running 2-node PyTorch NCCL all_reduce with Inspector enabled ==="

export MASTER_ADDR="${HEAD_NODE_IP}"
export MASTER_PORT="${NCCL_INSPECTOR_SMOKE_MASTER_PORT:-29631}"

# Keep these aligned with your vLLM script defaults.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"

# Inspector env. Use -1 first: dump at communicator teardown/finalization, not every 500 us.
export NCCL_PROFILER_PLUGIN="${NCCL_INSPECTOR_PLUGIN}"
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_DIR="${NCCL_INSPECTOR_SMOKE_DIR}"
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS="${NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS:--1}"
export NCCL_INSPECTOR_DUMP_VERBOSE="${NCCL_INSPECTOR_DUMP_VERBOSE:-1}"
export NCCL_INSPECTOR_ENABLE_P2P="${NCCL_INSPECTOR_ENABLE_P2P:-1}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,PROFILE}"
export NCCL_DEBUG_FILE="${TRACE_RUN_DIR}/nccl_logs/inspector_smoke_%h_%p.log"

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "NCCL_NET=${NCCL_NET}"
echo "NCCL_IB_HCA=${NCCL_IB_HCA}"
echo "NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY}"
echo "NCCL_PROFILER_PLUGIN=${NCCL_PROFILER_PLUGIN}"
echo "NCCL_INSPECTOR_ENABLE=${NCCL_INSPECTOR_ENABLE}"
echo "NCCL_INSPECTOR_DUMP_DIR=${NCCL_INSPECTOR_DUMP_DIR}"
echo "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=${NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS}"
echo "NCCL_INSPECTOR_DUMP_VERBOSE=${NCCL_INSPECTOR_DUMP_VERBOSE}"
echo "NCCL_DEBUG=${NCCL_DEBUG}"
echo "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
echo "NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}"

srun \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
  --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail

    source "'"${VENV_DIR}"'/bin/activate"

    # Choose the local interface that owns this node's selected IPv4 address.
    NODE_IP="$(hostname -I 2>/dev/null | tr " " "\n" | awk "/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}")"
    if [ -n "${NODE_IP}" ]; then
      IFACE="$(ip -o -4 addr show 2>/dev/null | awk -v target="${NODE_IP}" "{split(\$4, addr, \"/\"); if (addr[1] == target) {print \$2; exit}}")"
      if [ -n "${IFACE}" ]; then
        export NCCL_SOCKET_IFNAME="${IFACE}"
        export GLOO_SOCKET_IFNAME="${IFACE}"
      fi
    fi

    echo "[smoke] host=$(hostname) rank=${SLURM_PROCID}/${SLURM_NTASKS} local=${SLURM_LOCALID:-0} NODE_IP=${NODE_IP:-} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}"

    python -u <<PY
import os
import socket
import torch
import torch.distributed as dist

rank = int(os.environ["SLURM_PROCID"])
world = int(os.environ["SLURM_NTASKS"])
local_rank = int(os.environ.get("SLURM_LOCALID", "0"))

torch.cuda.set_device(local_rank)

print(f"[rank {rank}] host={socket.gethostname()} world={world} local_rank={local_rank}", flush=True)
print(f"[rank {rank}] torch={torch.__version__} cuda={torch.version.cuda} nccl={torch.cuda.nccl.version()}", flush=True)
print(f"[rank {rank}] gpu={torch.cuda.get_device_name(local_rank)}", flush=True)
print(f"[rank {rank}] MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')}", flush=True)
print(f"[rank {rank}] NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME')}", flush=True)
print(f"[rank {rank}] NCCL_PROFILER_PLUGIN={os.environ.get('NCCL_PROFILER_PLUGIN')}", flush=True)
print(f"[rank {rank}] NCCL_INSPECTOR_DUMP_DIR={os.environ.get('NCCL_INSPECTOR_DUMP_DIR')}", flush=True)

dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world)

for nbytes in [1048576, 16777216, 67108864]:
    nelem = nbytes // 4
    x = torch.ones(nelem, device="cuda", dtype=torch.float32) * (rank + 1)
    dist.all_reduce(x)
    torch.cuda.synchronize()
    expected = sum(range(1, world + 1))
    got = float(x[0].item())
    print(f"[rank {rank}] all_reduce nbytes={nbytes} got={got} expected={expected}", flush=True)
    if abs(got - expected) > 1e-3:
        raise RuntimeError(f"bad all_reduce result: got={got} expected={expected}")

dist.barrier()
dist.destroy_process_group()

print(f"[rank {rank}] loaded libnccl mappings:", flush=True)
seen = set()
with open("/proc/self/maps") as f:
    for line in f:
        if "libnccl" in line:
            p = line.split()[-1]
            if p not in seen:
                seen.add(p)
                print(f"[rank {rank}] {p}", flush=True)

print(f"[rank {rank}] done", flush=True)
PY
  '

echo "=== NCCL Inspector smoke outputs ==="
find "${NCCL_INSPECTOR_SMOKE_DIR}" -maxdepth 5 -type f -printf "%p %s bytes\n" 2>/dev/null | sort || true

echo "=== NCCL Inspector smoke NCCL logs ==="
find "${TRACE_RUN_DIR}/nccl_logs" -maxdepth 2 -type f -name "inspector_smoke_*" -printf "%p %s bytes\n" 2>/dev/null | sort || true

echo "=== PROFILE lines from NCCL logs ==="
grep -R "PROFILE\|Inspector\|profiler\|NCCL_PROFILER_PLUGIN\|Plugin" "${TRACE_RUN_DIR}/nccl_logs" 2>/dev/null | tail -200 || true

if find "${NCCL_INSPECTOR_SMOKE_DIR}" -type f -size +0c -print -quit 2>/dev/null | grep -q .; then
  echo "SUCCESS: NCCL Inspector smoke test produced non-empty output."
else
  echo "WARNING: NCCL Inspector smoke test completed but produced no non-empty Inspector files." >&2
  echo "Check logs under: ${TRACE_RUN_DIR}/nccl_logs" >&2
fi

echo "=== final artifact summary ==="
echo "TRACE_RUN_DIR=${TRACE_RUN_DIR}"
find "${TRACE_RUN_DIR}" -maxdepth 5 -type f -printf "%p %s bytes\n" 2>/dev/null | sort || true

echo "=== done ==="
