#!/usr/bin/env bash
#SBATCH --job-name=vllm-bench-sharegpt
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

set -euo pipefail
SCRIPT_VERSION="arc-bench-ipv4-netif-direct-2026-05-08-v2"

resolve_host_ipv4() {
  local nodename="$1"
  local ip=""

  pick_ipv4() {
    awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'
  }

  ip=$(dig +short "${nodename}" 2>/dev/null | pick_ipv4 || true)
  if [ -z "${ip}" ]; then
    ip=$(getent hosts "${nodename}" 2>/dev/null | awk '{print $1}' | pick_ipv4 || true)
  fi
  if [ -z "${ip}" ]; then
    ip=$(
      srun --nodelist="${nodename}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
        bash -c "hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'" 2>/dev/null || true
    )
  fi
  printf '%s' "${ip}"
}

interface_for_ip() {
  local target_ip="$1"
  ip -o -4 addr show 2>/dev/null | awk -v target="${target_ip}" '
    {
      split($4, addr, "/")
      if (addr[1] == target) {
        print $2
        exit
      }
    }'
}

interface_has_ip() {
  local iface="$1"
  local target_ip="$2"
  ip -o -4 addr show dev "${iface}" 2>/dev/null | awk -v target="${target_ip}" '
    {
      split($4, addr, "/")
      if (addr[1] == target) {
        found = 1
        exit
      }
    }
    END { exit(found ? 0 : 1) }'
}

configure_gloo_ifname() {
  local target_ip="$1"
  local iface="${GLOO_SOCKET_IFNAME:-}"

  if [ -n "${iface}" ] && ! interface_has_ip "${iface}" "${target_ip}"; then
    echo "Ignoring GLOO_SOCKET_IFNAME=${iface}; it does not own ${target_ip} on $(hostname)." >&2
    iface=""
  fi
  if [ -z "${iface}" ]; then
    iface="$(interface_for_ip "${target_ip}")"
  fi
  if [ -n "${iface}" ]; then
    export GLOO_SOCKET_IFNAME="${iface}"
  fi
}

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
VENV_DIR="${REPO_ROOT}/.venv"
echo "REPO_ROOT=${REPO_ROOT}"

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

PYTHON_PATH="$(command -v python)"
EXPECTED_PYTHON="${VENV_DIR}/bin/python"
echo "Using python: ${PYTHON_PATH}"
if [ "${PYTHON_PATH}" != "${EXPECTED_PYTHON}" ]; then
  echo "Error: python did not resolve to venv interpreter." >&2
  echo "Expected: ${EXPECTED_PYTHON}" >&2
  echo "Got:      ${PYTHON_PATH}" >&2
  exit 1
fi

python -m pip install -U pip
VLLM_BIN="${VENV_DIR}/bin/vllm"
if [ ! -x "${VLLM_BIN}" ]; then
  echo "Error: vllm binary not found at ${VLLM_BIN}. Run the host setup/install first." >&2
  exit 1
fi

DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/sharegpt_buckets/Qwen_Qwen3-30B-A3B-Instruct-2507/sharegpt_sp128.jsonl}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
REQUEST_RATE="${REQUEST_RATE:-1.0}"
BURSTINESS="${BURSTINESS:-1.0}"
SEED="${SEED:-100}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-1}}"
RAY_PORT="${RAY_PORT:-6378}"

# Usage modes:
# 1) standalone: bash serve_ShareGPT_multi_node.sh <ray_jobid> <head_node>
# 2) from host script: env provides SLURM_JOB_ID and head node details
RAY_JOBID="${1:-${SLURM_JOB_ID:-}}"
HEAD_NODE="${2:-${HEAD_NODE:-}}"
if [ -z "${HEAD_NODE}" ] && [ -n "${SLURM_NODELIST:-}" ]; then
  HEAD_NODE="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"
fi
if [ -z "${RAY_JOBID}" ] || [ -z "${HEAD_NODE}" ]; then
  echo "Usage: $0 <ray_jobid> <head_node>" >&2
  echo "Or run inside an sbatch allocation with SLURM_JOB_ID/SLURM_NODELIST set." >&2
  exit 1
fi
if [ -n "${HEAD_NODE_IP:-}" ]; then
  case "${HEAD_NODE_IP}" in
    *.*.*.*) ;;
    *)
      echo "Error: HEAD_NODE_IP must be IPv4, got ${HEAD_NODE_IP}." >&2
      exit 1
      ;;
  esac
else
  HEAD_NODE_IP="$(resolve_host_ipv4 "${HEAD_NODE}")"
fi
if [ -z "${HEAD_NODE_IP}" ]; then
  echo "Failed to resolve an IPv4 HEAD_NODE_IP for ${HEAD_NODE}" >&2
  exit 1
fi
configure_gloo_ifname "${HEAD_NODE_IP}"
RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

echo "STARTING VLLM BENCH SERVE ON RAY CLUSTER"
echo "SCRIPT_VERSION=${SCRIPT_VERSION}"
echo "RAY_JOBID=${RAY_JOBID}"
echo "HEAD_NODE=${HEAD_NODE}"
echo "HEAD_NODE_IP=${HEAD_NODE_IP}"
echo "RAY_ADDRESS=${RAY_ADDRESS}"
echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-<unset>}"

if [ ! -f "${DATASET_PATH}" ]; then
  echo "Dataset file not found: ${DATASET_PATH}" >&2
  echo "Generate buckets first with datasets/build_sharegpt_length_buckets.py" >&2
  exit 1
fi

cd "${REPO_ROOT}"

"${VLLM_BIN}" bench serve \
  --backend vllm \
  --host "${HOST:-${HEAD_NODE_IP}}" \
  --port "${PORT}" \
  --endpoint "${ENDPOINT}" \
  --model "${MODEL_ID}" \
  --dataset-name custom \
  --dataset-path "${DATASET_PATH}" \
  --num-prompts "${NUM_PROMPTS}" \
  --request-rate "${REQUEST_RATE}" \
  --burstiness "${BURSTINESS}" \
  --seed "${SEED}"
