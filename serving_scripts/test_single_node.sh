#!/usr/bin/env bash
#SBATCH --job-name=vllm-single-qwen3-30b-tp2
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=00:45:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err
#SBATCH --mail-user=jason.miller@eng.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=engs-glass
#SBATCH --qos=priority

set -euo pipefail

SCRIPT_VERSION="arc-ray-qwen3-30b-a3b-instruct-single-node-tp2-nsys-v2"

DEBUG_SLURM_SCRIPT="${DEBUG_SLURM_SCRIPT:-0}"
slurm_debug() {
  if [ "${DEBUG_SLURM_SCRIPT}" = "1" ]; then
    echo "[slurm-debug] $*" >&2
  fi
}

SP="${SP:-128}"
SD="${SD:-128}"
export NSYS_ENABLE="${NSYS_ENABLE:-1}"

export NODE="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"

echo "=== vLLM single-node job (TP=2) ==="
echo "SCRIPT_VERSION=${SCRIPT_VERSION}"
echo "Date: $(date -Is 2>/dev/null || date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "NODE=${NODE}"
slurm_debug "SLURM_NTASKS=${SLURM_NTASKS:-} SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-}"

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
    ip=$(hostname -I 2>/dev/null | tr ' ' '\n' | pick_ipv4 || true)
    [ -n "${ip}" ] && method="hostname_I_ipv4"
  fi
  if [ -z "${ip}" ]; then
    ip=$(
      srun --nodes=1 --ntasks=1 \
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

export NODE_IP="$(resolve_host_ip "${NODE}")"
if [ -z "${NODE_IP}" ]; then
  echo "Error: could not resolve an IPv4 address for node ${NODE}." >&2
  exit 1
fi
echo "NODE_IP=${NODE_IP}"
export VLLM_HOST_IP="${NODE_IP}"
echo "VLLM_HOST_IP=${VLLM_HOST_IP}"
configure_socket_ifnames "${NODE_IP}" 0

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno,ens,enp,eth,ib}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL,P2P,TUNING}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"

export RAY_PORT="${RAY_PORT:-6378}"
export RAY_ADDRESS="${NODE_IP}:${RAY_PORT}"
echo "RAY_PORT=${RAY_PORT} RAY_ADDRESS=${RAY_ADDRESS}"

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

TRACE_BASE="/data/engs-glass/catz0932/inference-traces/vllm/results"
TRACE_RUN_DIR="${TRACE_BASE}/${SLURM_JOB_ID}"
RAY_TMP_ROOT="${TRACE_RUN_DIR}/ray_tmp"
RAY_TMP_LINK_BASE="/tmp/vray-${SLURM_JOB_ID}"

mkdir -p "${TRACE_RUN_DIR}/nsight"
mkdir -p "${TRACE_RUN_DIR}/nccl_logs"
mkdir -p "${RAY_TMP_ROOT}"

export NSYS_ENABLE="${NSYS_ENABLE:-1}"
export NSYS_DIR="${TRACE_RUN_DIR}/nsight"
export NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cudnn,cublas}"
export NSYS_DELAY="${NSYS_DELAY:-0}"
# Worker-only Nsight via vLLM flags (no outer nsys on api_server — avoids NCCL init failures).
export NSYS_PROFILE_VLLM="${NSYS_PROFILE_VLLM:-1}"
export NSYS_PROFILE_RAY="${NSYS_PROFILE_RAY:-0}"

export NCCL_DEBUG_FILE="${TRACE_RUN_DIR}/nccl_logs/nccl_%h_%p.log"

echo "TRACE_RUN_DIR=${TRACE_RUN_DIR}"
echo "RAY_TMP_ROOT=${RAY_TMP_ROOT}"
echo "NSYS_DIR=${NSYS_DIR}"
echo "NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}"
echo "NSYS_TRACE=${NSYS_TRACE}"
echo "NSYS_DELAY=${NSYS_DELAY}"
echo "NSYS_PROFILE_VLLM=${NSYS_PROFILE_VLLM}"
echo "NSYS_PROFILE_RAY=${NSYS_PROFILE_RAY}"
echo "nsys path: $(command -v nsys || echo '<not found>')"
nsys --version || true

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

slurm_debug "PATH=${PATH}"
slurm_debug "pip install starting (cuda + build + editable vllm)..."

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

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
HOST="${HOST:-${NODE_IP}}"
PORT="${PORT:-8000}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
NUM_NODES=1
TP="${TP:-2}"
PP="${PP:-1}"
EP="${EP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}"
SERVE_SCRIPT="${REPO_ROOT}/serving_scripts/serve_ShareGPT_multi_node.sh"

echo "=== runtime knobs ==="
echo "MODEL_ID=${MODEL_ID} HOST=${HOST} PORT=${PORT} TP=${TP} PP=${PP} EP=${EP}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE} NUM_NODES=${NUM_NODES} CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "SERVE_SCRIPT=${SERVE_SCRIPT}"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set"
else
  echo "HF_TOKEN is not set"
fi

SERVER_STEP_PID=""
HEAD_RAY_PID=""

cleanup() {
  if [ -n "${SERVER_STEP_PID}" ] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    kill "${SERVER_STEP_PID}" 2>/dev/null || true
    wait "${SERVER_STEP_PID}" 2>/dev/null || true
  fi
  if [ -n "${HEAD_RAY_PID}" ] && kill -0 "${HEAD_RAY_PID}" 2>/dev/null; then
    kill "${HEAD_RAY_PID}" 2>/dev/null || true
    wait "${HEAD_RAY_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

collect_ray_logs() {
  echo "Collecting Ray logs from shared Ray temp dir..."
  local out="${TRACE_RUN_DIR}/ray_logs"
  mkdir -p "${out}"

  for node_dir in "${RAY_TMP_ROOT}"/*; do
    [ -d "${node_dir}" ] || continue

    local node_name
    node_name="$(basename "${node_dir}")"

    local session
    session="$(readlink -f "${node_dir}/session_latest" 2>/dev/null || true)"

    if [ -z "${session}" ] || [ ! -d "${session}/logs" ]; then
      echo "No Ray session logs found for ${node_name} under ${node_dir}" >&2
      continue
    fi

    echo "Archiving Ray logs for ${node_name}: ${session}/logs"

    tar \
      -C "${session}" \
      -czf "${out}/ray_logs_${node_name}.tgz" \
      logs \
      --exclude='logs/nsight/*.qdstrm' \
      --exclude='logs/nsight/*.nsys-rep' \
      --exclude='logs/events/*' \
      2>/dev/null || true
  done

  echo "Ray log archives:"
  find "${out}" -type f -printf "%p %s bytes\n" 2>/dev/null || true
}

echo "=== Ray head (local, ${GPUS_PER_NODE} GPUs) ==="
(
  unset GLOO_SOCKET_IFNAME
  export VLLM_HOST_IP="${NODE_IP}"
  configure_socket_ifnames "${NODE_IP}" 0

  rm -rf "${RAY_TMP_LINK_BASE}-${NODE}"
  mkdir -p "${RAY_TMP_ROOT}/${NODE}"
  ln -sfn "${RAY_TMP_ROOT}/${NODE}" "${RAY_TMP_LINK_BASE}-${NODE}"
  echo "Ray temp dir link: ${RAY_TMP_LINK_BASE}-${NODE} -> ${RAY_TMP_ROOT}/${NODE}"

  if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_RAY}" = "1" ]; then
    echo "Profiling Ray head with Nsight Systems"
    echo "Nsight output: ${NSYS_DIR}/ray_head_${NODE}.nsys-rep"
    nsys profile \
      --force-overwrite=true \
      --trace="${NSYS_TRACE}" \
      --sample=none \
      --delay="${NSYS_DELAY}" \
      --output="${NSYS_DIR}/ray_head_${NODE}" \
      "${RAY_BIN}" start --block \
        --head \
        --node-ip-address="${NODE_IP}" \
        --port="${RAY_PORT}" \
        --num-gpus="${GPUS_PER_NODE}" \
        --num-cpus="${CPUS_PER_TASK}" \
        --temp-dir="${RAY_TMP_LINK_BASE}-${NODE}"
  else
    "${RAY_BIN}" start --block \
      --head \
      --node-ip-address="${NODE_IP}" \
      --port="${RAY_PORT}" \
      --num-gpus="${GPUS_PER_NODE}" \
      --num-cpus="${CPUS_PER_TASK}" \
      --temp-dir="${RAY_TMP_LINK_BASE}-${NODE}"
  fi
) &
HEAD_RAY_PID=$!
echo "HEAD_RAY_PID=${HEAD_RAY_PID}"
sleep 20

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
VLLM_TRACE_FLAGS=()
if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_VLLM}" = "1" ]; then
  VLLM_TRACE_FLAGS+=(
    --ray-workers-use-nsight
    --enable-layerwise-nvtx-tracing
    --enable-mfu-metrics
    --kv-cache-metrics
    --kv-cache-metrics-sample 1.0
    --enable-logging-iteration-details
  )
fi

if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_VLLM}" = "1" ]; then
  echo "Starting vLLM server with Ray worker Nsight profiling enabled"
  echo "Ray worker Nsight reports should appear under ${RAY_TMP_ROOT}/${NODE}/session_*/logs/nsight"
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --distributed-executor-backend ray \
    --tensor-parallel-size "${TP}" \
    --pipeline-parallel-size "${PP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enforce-eager \
    "${VLLM_TRACE_FLAGS[@]}" \
    --disable-custom-all-reduce &
else
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --distributed-executor-backend ray \
    --tensor-parallel-size "${TP}" \
    --pipeline-parallel-size "${PP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enforce-eager \
    --disable-custom-all-reduce &
fi

SERVER_STEP_PID=$!
echo "Started vLLM server process (pid=${SERVER_STEP_PID}). Waiting for /health ..."

_health_wait_n=0
until curl -fsS "http://${NODE_IP}:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    echo "Server process exited before becoming ready." >&2
    wait "${SERVER_STEP_PID}" || true
    exit 1
  fi
  _health_wait_n=$((_health_wait_n + 1))
  if [ "${DEBUG_SLURM_SCRIPT}" = "1" ] || [ "$((_health_wait_n % 12))" -eq 0 ]; then
    echo "Still waiting for http://${NODE_IP}:${PORT}/health (attempt ${_health_wait_n}) ..."
  fi
  sleep 5
done
unset _health_wait_n

echo "Server is healthy. Running ${SERVE_SCRIPT} ..."
echo "SP=${SP} SD=${SD}"
HOST="${NODE_IP}" PORT="${PORT}" MODEL_ID="${MODEL_ID}" \
  SP="${SP}" SD="${SD}" \
  HEAD_NODE_IP="${NODE_IP}" \
  GPUS_PER_NODE="${GPUS_PER_NODE}" CPUS_PER_TASK="${CPUS_PER_TASK}" \
  RAY_PORT="${RAY_PORT}" bash "${SERVE_SCRIPT}" "${SLURM_JOB_ID}" "${NODE}"

echo "Workload finished. Stopping vLLM/Nsight server process cleanly..."
if [ -n "${SERVER_STEP_PID}" ] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
  kill -INT "${SERVER_STEP_PID}" 2>/dev/null || true
  wait "${SERVER_STEP_PID}" 2>/dev/null || true
  SERVER_STEP_PID=""
fi

echo "Stopping Ray so Nsight worker reports finalize..."
if [ -n "${HEAD_RAY_PID}" ] && kill -0 "${HEAD_RAY_PID}" 2>/dev/null; then
  kill -TERM "${HEAD_RAY_PID}" 2>/dev/null || true
  wait "${HEAD_RAY_PID}" 2>/dev/null || true
fi
HEAD_RAY_PID=""

echo "Waiting briefly for Ray/Nsight files to flush..."
sleep 30

collect_ray_logs

echo "Collecting Ray worker Nsight reports from shared Ray temp dir..."
mkdir -p "${TRACE_RUN_DIR}/ray_worker_nsight"

echo "Looking for Nsight reports under ${RAY_TMP_ROOT}:"
find "${RAY_TMP_ROOT}" \
  -type f -name "*.nsys-rep" \
  -printf "%p %s bytes\n" 2>/dev/null || true

find "${RAY_TMP_ROOT}" \
  -type f -name "*.nsys-rep" -print0 2>/dev/null \
  | while IFS= read -r -d '' report; do
      rel="${report#${RAY_TMP_ROOT}/}"
      node_name="${rel%%/*}"
      mkdir -p "${TRACE_RUN_DIR}/ray_worker_nsight/${node_name}"
      cp -v "${report}" "${TRACE_RUN_DIR}/ray_worker_nsight/${node_name}/" || true
    done

echo "Trace files:"
find "${TRACE_RUN_DIR}" -maxdepth 5 -type f -printf "%p %s bytes\n" 2>/dev/null || true
