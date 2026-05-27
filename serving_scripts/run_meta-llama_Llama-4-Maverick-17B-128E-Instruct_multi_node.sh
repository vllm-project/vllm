#!/usr/bin/env bash
#SBATCH --nodelist=htc-g[059-060]
#SBATCH --job-name=vllm-host-llama-4-maverick-17b-128e
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err
#SBATCH --mail-user=jason.miller@eng.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRIPT_VERSION="arc-ray-llama4-maverick-tp8-2x4h100-2026-05-10-v1"

DEBUG_SLURM_SCRIPT="${DEBUG_SLURM_SCRIPT:-0}"
slurm_debug() {
  if [ "${DEBUG_SLURM_SCRIPT}" = "1" ]; then
    echo "[slurm-debug] $*" >&2
  fi
}


# SP = prompt / prefill token bucket
# SD = decode / output tokens per request
SP="${SP:-128}"
SD="${SD:-128}"


export HEAD_NODE
export WORKER_NODES
HEAD_NODE=$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)
WORKER_NODES=$(scontrol show hostnames "${SLURM_NODELIST}" | tail -n+2)

echo "=== vLLM multi-node host job ==="
echo "SCRIPT_VERSION=${SCRIPT_VERSION}"
echo "Date: $(date -Is 2>/dev/null || date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "HEAD_NODE=${HEAD_NODE}"
echo "WORKER_NODES=${WORKER_NODES}"
slurm_debug "SLURM_NTASKS=${SLURM_NTASKS:-} SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-}"
slurm_debug "Full nodelist: $(scontrol show hostnames "${SLURM_NODELIST}" 2>/dev/null | tr '\n' ' ')"

resolve_host_ip() {
  local nodename="$1"
  local ip=""
  local method=""

  pick_ipv4() {
    awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'
  }

  ip=$(dig +short "${nodename}" 2>/dev/null | pick_ipv4 || true)
  if [ -n "${ip}" ]; then
    method="dig_ipv4"
  fi

  if [ -z "${ip}" ]; then
    ip=$(getent hosts "${nodename}" 2>/dev/null | awk '{print $1}' | pick_ipv4 || true)
    [ -n "${ip}" ] && method="getent_ipv4"
  fi

  if [ -z "${ip}" ]; then
    ip=$(
      srun --nodelist="${nodename}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
        bash -c "hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'" 2>/dev/null || true
    )
    [ -n "${ip}" ] && method="srun_hostname_I_ipv4"
  fi

  slurm_debug "resolve_host_ip(${nodename}) -> ${ip:-<empty>} [${method:-failed}]"
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

configure_socket_ifnames() {
  local target_ip="$1"
  local set_nccl="${2:-1}"
  local iface="${GLOO_SOCKET_IFNAME:-}"

  if [ -n "${iface}" ] && ! interface_has_ip "${iface}" "${target_ip}"; then
    echo "Ignoring GLOO_SOCKET_IFNAME=${iface}; it does not own ${target_ip} on $(hostname)." >&2
    iface=""
  fi

  if [ -z "${iface}" ]; then
    iface="$(interface_for_ip "${target_ip}")"
  fi

  if [ -z "${iface}" ]; then
    echo "Error: could not find a network interface for ${target_ip} on $(hostname)." >&2
    ip -o -4 addr show >&2 || true
    exit 1
  fi

  export GLOO_SOCKET_IFNAME="${iface}"

  if [ "${set_nccl}" = "1" ]; then
    local nccl_iface="${NCCL_SOCKET_IFNAME:-}"
    if [ -n "${nccl_iface}" ] && ! interface_has_ip "${nccl_iface}" "${target_ip}"; then
      echo "Ignoring NCCL_SOCKET_IFNAME=${nccl_iface}; it does not own ${target_ip} on $(hostname)." >&2
      nccl_iface=""
    fi
    export NCCL_SOCKET_IFNAME="${nccl_iface:-${iface}}"
  fi

  echo "Socket interface for ${target_ip} on $(hostname): GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}"
}

export HEAD_NODE_IP
HEAD_NODE_IP="$(resolve_host_ip "${HEAD_NODE}")"
if [ -z "${HEAD_NODE_IP}" ]; then
  echo "Error: could not resolve an IPv4 address for head node ${HEAD_NODE}." >&2
  exit 1
fi

echo "HEAD_NODE_IP=${HEAD_NODE_IP}"
export VLLM_HOST_IP="${HEAD_NODE_IP}"
echo "VLLM_HOST_IP=${VLLM_HOST_IP}"
configure_socket_ifnames "${HEAD_NODE_IP}" 0

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno,ens,enp,eth,ib}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"

export RAY_PORT="${RAY_PORT:-$((6300 + (${SLURM_JOB_ID:-0} % 1000)))}"
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
echo "RAY_PORT=${RAY_PORT} RAY_ADDRESS=${RAY_ADDRESS}"

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
echo "VENV_DIR=${VENV_DIR}"

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

echo "After venv: python=$(command -v python) ray=$(command -v ray 2>/dev/null || echo '<not on PATH>')"
slurm_debug "PATH=${PATH}"
slurm_debug "pip install starting..."

python -m pip install -U pip
python -m pip install -r "${REPO_ROOT}/requirements/cuda.txt"
python -m pip install -r "${REPO_ROOT}/requirements/build/cuda.txt"

RAY_REQUIREMENT="${RAY_REQUIREMENT:-ray[cgraph]>=2.48.0}"
echo "Installing Ray requirement: ${RAY_REQUIREMENT}"
python -m pip install "${RAY_REQUIREMENT}"

(
  cd "${REPO_ROOT}" || exit 1
  export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
  python -m pip install -e . ${VLLM_PIP_INSTALL_EXTRA_ARGS:-}
)

RAY_BIN="${VENV_DIR}/bin/ray"
if [ ! -x "${RAY_BIN}" ]; then
  echo "Error: ray binary not found at ${RAY_BIN}." >&2
  exit 1
fi
echo "Using RAY_BIN=${RAY_BIN}"

export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8}"
HOST="${HOST:-${HEAD_NODE_IP}}"
PORT="${PORT:-8000}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-2}}"
TOTAL_GPUS="$((GPUS_PER_NODE * NUM_NODES))"

TP="${TP:-${TOTAL_GPUS}}"
PP="${PP:-1}"
EP="${EP:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-1}}"
SERVE_SCRIPT="${REPO_ROOT}/serving_scripts/serve_ShareGPT_multi_node.sh"

echo "=== runtime knobs ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT} TP=${TP} PP=${PP} EP=${EP}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE} NUM_NODES=${NUM_NODES} TOTAL_GPUS=${TOTAL_GPUS} CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_NUM_SEQS=${MAX_NUM_SEQS} MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION} KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE} NCCL_NET=${NCCL_NET} NCCL_IB_HCA=${NCCL_IB_HCA}"
echo "NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "NCCL_DEBUG=${NCCL_DEBUG} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
echo "SERVE_SCRIPT=${SERVE_SCRIPT}"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set"
else
  echo "HF_TOKEN is not set"
fi

SERVER_STEP_PID=""
HEAD_RAY_PID=""
WORKER_RAY_PIDS=""

cleanup() {
  if [ -n "${SERVER_STEP_PID}" ] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    kill "${SERVER_STEP_PID}" 2>/dev/null || true
    wait "${SERVER_STEP_PID}" 2>/dev/null || true
  fi

  if [ -n "${HEAD_RAY_PID}" ] && kill -0 "${HEAD_RAY_PID}" 2>/dev/null; then
    kill "${HEAD_RAY_PID}" 2>/dev/null || true
    wait "${HEAD_RAY_PID}" 2>/dev/null || true
  fi

  for pid in ${WORKER_RAY_PIDS}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

echo "=== Ray head (background srun) ==="
RAY_HEAD_CMD="$(
  declare -f interface_for_ip
  declare -f interface_has_ip
  declare -f configure_socket_ifnames
)
source \"${VENV_DIR}/bin/activate\"
unset GLOO_SOCKET_IFNAME
export VLLM_HOST_IP=${HEAD_NODE_IP}
configure_socket_ifnames \"${HEAD_NODE_IP}\" 0
\"${RAY_BIN}\" start --block \\
  --head \\
  --node-ip-address=${HEAD_NODE_IP} \\
  --port=${RAY_PORT} \\
  --num-gpus=${GPUS_PER_NODE} \\
  --num-cpus=${CPUS_PER_TASK}"

srun \
  --nodelist "${HEAD_NODE}" \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-task="${GPUS_PER_NODE}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  bash -lc "${RAY_HEAD_CMD}" &

HEAD_RAY_PID=$!
echo "HEAD_RAY_PID=${HEAD_RAY_PID}"
sleep 20

if [ -n "${WORKER_NODES}" ]; then
  echo "=== Ray workers ==="
  for WORKER in ${WORKER_NODES}; do
    WORKER_IP="$(resolve_host_ip "${WORKER}")"
    if [ -z "${WORKER_IP}" ]; then
      echo "Error: could not resolve IP for worker ${WORKER}." >&2
      exit 1
    fi

    echo "Starting worker node: ${WORKER} with IP ${WORKER_IP}"

    RAY_WORKER_CMD="$(
      declare -f interface_for_ip
      declare -f interface_has_ip
      declare -f configure_socket_ifnames
)
source \"${VENV_DIR}/bin/activate\"
unset GLOO_SOCKET_IFNAME
export VLLM_HOST_IP=${WORKER_IP}
configure_socket_ifnames \"${WORKER_IP}\" 0
\"${RAY_BIN}\" start --block \\
  --address=${HEAD_NODE_IP}:${RAY_PORT} \\
  --node-ip-address=${WORKER_IP} \\
  --num-gpus=${GPUS_PER_NODE} \\
  --num-cpus=${CPUS_PER_TASK}"

    srun \
      --nodelist "${WORKER}" \
      --nodes=1 \
      --ntasks=1 \
      --ntasks-per-node=1 \
      --gpus-per-task="${GPUS_PER_NODE}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      bash -lc "${RAY_WORKER_CMD}" &

    WORKER_RAY_PIDS="${WORKER_RAY_PIDS} $!"
    echo "Worker Ray step pid: $!"
  done
  sleep 20
fi

echo "=== ray status ==="
"${RAY_BIN}" status || echo "Warning: ray status failed; continuing with Python Ray node check."

python - <<'PY'
import ray
ray.init(address="auto")
nodes = ray.nodes()
print("Ray nodes:")
for node in nodes:
    print(
        f"  {node.get('NodeManagerAddress')} "
        f"alive={node.get('Alive')} "
        f"resources={node.get('Resources')}"
    )
PY

echo "=== vLLM api_server (background process) ==="
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --distributed-executor-backend ray \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size "${PP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --enforce-eager \
  --disable-custom-all-reduce &

SERVER_STEP_PID=$!
echo "Started vLLM server process pid=${SERVER_STEP_PID}. Waiting for /health ..."

_health_wait_n=0
until curl -fsS "http://${HEAD_NODE_IP}:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    echo "Server process exited before becoming ready." >&2
    wait "${SERVER_STEP_PID}" || true
    exit 1
  fi

  _health_wait_n=$((_health_wait_n + 1))
  if [ "${DEBUG_SLURM_SCRIPT}" = "1" ] || [ "$((_health_wait_n % 12))" -eq 0 ]; then
    echo "Still waiting for http://${HEAD_NODE_IP}:${PORT}/health attempt ${_health_wait_n} ..."
  fi
  sleep 5
done

echo "Server is healthy. Running ${SERVE_SCRIPT} ..."
echo "SP=${SP} SD=${SD}"
HOST="${HEAD_NODE_IP}" PORT="${PORT}" MODEL_ID="${MODEL_ID}" \
  SP="${SP}" SD="${SD}" \
  HEAD_NODE_IP="${HEAD_NODE_IP}" \
  GPUS_PER_NODE="${GPUS_PER_NODE}" CPUS_PER_TASK="${CPUS_PER_TASK}" \
  RAY_PORT="${RAY_PORT}" bash "${SERVE_SCRIPT}" "${SLURM_JOB_ID}" "${HEAD_NODE}"