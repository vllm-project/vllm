#!/usr/bin/env bash
#SBATCH --nodelist=htc-g[053,054,055,059,060]
#SBATCH --job-name=vllm-host-qwen3-30b
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=01:00:10
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err
#SBATCH --mail-user=jason.miller@eng.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=engs-glass
#SBATCH --qos=priority

set -euo pipefail

SCRIPT_VERSION="arc-ray-qwen3-30b-a3b-instruct"

# Set DEBUG_SLURM_SCRIPT=1 for extra diagnostics (DNS probes, PATH, ray location).
# Set NSYS_COPY_DEBUG=1 for verbose Nsight copy traces (per-node ls/stat/find).
DEBUG_SLURM_SCRIPT="${DEBUG_SLURM_SCRIPT:-0}"
NSYS_COPY_DEBUG="${NSYS_COPY_DEBUG:-0}"

slurm_debug() {
  if [ "${DEBUG_SLURM_SCRIPT}" = "1" ]; then
    echo "[slurm-debug] $*" >&2
  fi
}

nsight_debug() {
  if [ "${NSYS_COPY_DEBUG}" = "1" ] || [ "${DEBUG_SLURM_SCRIPT}" = "1" ]; then
    echo "[nsight-debug] $*"
  fi
}

# Always printed during Nsight copy/shutdown (lightweight progress markers).
nsight_copy_msg() {
  echo "[nsight-copy] $*"
}

# Audit Nsight artifacts on a compute node: hunt beyond session_latest so we can
# tell "file exists elsewhere" vs "never written" vs "copy path wrong".
nsight_audit_on_node() {
  local node="$1"
  local label="${2:-audit}"
  local job_id="${SLURM_JOB_ID:-unknown}"

  nsight_copy_msg "${label}: full Nsight hunt on ${node} (job ${job_id})..."
  srun \
    --overlap \
    --nodelist "${node}" \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --mem=4G \
    bash -lc "
      set +e
      COPY_GLOB='/tmp/ray/session_latest/logs/nsight/*.nsys-rep'
      COPY_DIR='/tmp/ray/session_latest/logs/nsight'

      echo '[nsight-copy] === ${label} audit host='\$(hostname)' expected_node=${node} job=${job_id} ==='
      echo '[nsight-copy] copy script ONLY reads: '\${COPY_GLOB}

      echo '[nsight-copy] session_latest symlink:'
      ls -la /tmp/ray/session_latest 2>/dev/null || echo '  MISSING /tmp/ray/session_latest'
      echo '[nsight-copy] session_latest resolved ->' \$(readlink -f /tmp/ray/session_latest 2>/dev/null || echo MISSING)

      if [ -d \"\${COPY_DIR}\" ]; then
        echo '[nsight-copy] COPY_DIR listing (what cp uses):'
        ls -la \"\${COPY_DIR}/\" 2>/dev/null || true
        for f in \"\${COPY_DIR}\"/*; do
          [ -f \"\$f\" ] && stat -c '[nsight-copy]   copy-target %n %s bytes' \"\$f\" 2>/dev/null || true
        done
      else
        echo '[nsight-copy] COPY_DIR missing: '\${COPY_DIR}
      fi

      echo '[nsight-copy] --- HUNT: every *.nsys-rep under /tmp/ray ---'
      ray_rep_count=\$(find /tmp/ray -type f -name '*.nsys-rep' 2>/dev/null | wc -l | tr -d ' ')
      echo '[nsight-copy] count='\${ray_rep_count}
      find /tmp/ray -type f -name '*.nsys-rep' -printf '[nsight-copy]   %p %s bytes\n' 2>/dev/null | sort || true

      echo '[nsight-copy] --- HUNT: every *.qdstrm (unfinished reports) under /tmp/ray ---'
      find /tmp/ray -type f -name '*.qdstrm' -printf '[nsight-copy]   %p %s bytes\n' 2>/dev/null | sort || true

      echo '[nsight-copy] --- HUNT: worker_process_* under /tmp (any path) ---'
      find /tmp -maxdepth 8 -type f -name 'worker_process*.nsys-rep' \
        -printf '[nsight-copy]   %p %s bytes\n' 2>/dev/null | sort || true

      echo '[nsight-copy] --- HUNT: /tmp/vray-${job_id}* (custom Ray temp dirs) ---'
      for vroot in /tmp/vray-${job_id}-* /tmp/vray-${job_id}; do
        [ -e \"\${vroot}\" ] || continue
        echo '[nsight-copy]   scanning '\${vroot}
        find \"\${vroot}\" -type f \\( -name '*.nsys-rep' -o -name '*.qdstrm' \\) \
          -printf '[nsight-copy]     %p %s bytes\n' 2>/dev/null | sort || true
      done

      echo '[nsight-copy] --- HUNT: shared TRACE dir on this node (if visible) ---'
      find '${TRACE_RUN_DIR}' -maxdepth 4 -type f \\( -name '*.nsys-rep' -o -name '*.qdstrm' \\) \
        -printf '[nsight-copy]   %p %s bytes\n' 2>/dev/null | sort || true

      echo '[nsight-copy] --- HUNT: API-server nsys under ${NSYS_DIR} (head only) ---'
      find '${NSYS_DIR}' -maxdepth 2 -type f -name '*.nsys-rep' \
        -printf '[nsight-copy]   %p %s bytes\n' 2>/dev/null | sort || true

      copy_n=\$(find \"\${COPY_DIR}\" -type f -name '*.nsys-rep' 2>/dev/null | wc -l | tr -d ' ')
      ray_n=\$(find /tmp/ray -type f -name '*.nsys-rep' 2>/dev/null | wc -l | tr -d ' ')
      tmp_n=\$(find /tmp -maxdepth 8 -type f -name '*.nsys-rep' 2>/dev/null | wc -l | tr -d ' ')
      echo '[nsight-copy] SUMMARY on '\$(hostname)': copy_dir='\${copy_n}' /tmp/ray='\${ray_n}' /tmp(all)='\${tmp_n}'

      if [ \"\${ray_n}\" -gt \"\${copy_n}\" ]; then
        echo '[nsight-copy] WARNING: .nsys-rep exist under /tmp/ray OUTSIDE copy_dir — cp may miss them:'
        find /tmp/ray -type f -name '*.nsys-rep' ! -path \"\${COPY_DIR}/*\" \
          -printf '[nsight-copy]   MISSED %p %s bytes\n' 2>/dev/null | sort || true
      fi
      if [ \"\${copy_n}\" -eq 0 ] && [ \"\${ray_n}\" -gt 0 ]; then
        echo '[nsight-copy] WARNING: copy_dir empty but other /tmp/ray reports exist (wrong session_latest?)'
      fi
      if [ \"\${copy_n}\" -eq 0 ] && [ \"\${tmp_n}\" -eq 0 ]; then
        echo '[nsight-copy] WARNING: no .nsys-rep anywhere under /tmp on this node (not written or already deleted)'
      fi
      zero_n=\$(find /tmp/ray -type f -name '*.nsys-rep' -size 0 2>/dev/null | wc -l | tr -d ' ')
      if [ \"\${zero_n}\" -gt 0 ]; then
        echo '[nsight-copy] WARNING: zero-byte .nsys-rep present (finalize failed):'
        find /tmp/ray -type f -name '*.nsys-rep' -size 0 \
          -printf '[nsight-copy]   ZERO %p\n' 2>/dev/null || true
      fi

      if [ \"${NSYS_COPY_DEBUG}\" = \"1\" ] || [ \"${DEBUG_SLURM_SCRIPT}\" = \"1\" ]; then
        echo '[nsight-debug] all Ray sessions by mtime on '\$(hostname)':'
        ls -lt /tmp/ray 2>/dev/null | head -12 || true
        echo '[nsight-debug] nsight dirs under every session:'
        find /tmp/ray -type d -path '*/logs/nsight' 2>/dev/null | while read -r d; do
          echo \"[nsight-debug]   \${d}:\"
          ls -la \"\${d}\" 2>/dev/null | sed 's/^/[nsight-debug]     /' || true
        done
      fi
    " || nsight_copy_msg "${label}: srun audit failed on ${node} (exit $?)"
}

nsight_peek_on_node() {
  local node="$1"
  local label="${2:-peek}"
  nsight_audit_on_node "${node}" "${label}"
}

nsight_summarize_dest() {
  local node="$1"
  local dest="${TRACE_RUN_DIR}/ray_worker_nsight/${node}"

  nsight_copy_msg "dest summary for ${node}: ${dest}"
  if [ -d "${dest}" ]; then
    ls -la "${dest}/" 2>/dev/null || true
    find "${dest}" -type f \( -name '*.nsys-rep' -o -name '*.qdstrm' \) \
      -exec stat -c '[nsight-copy]   %n %s bytes' {} \; 2>/dev/null || true
  else
    nsight_copy_msg "  (no dest directory yet)"
  fi
}

# SP = prompt / prefill token bucket
# SD = decode / output tokens per request
SP="${SP:-128}"
SD="${SD:-128}"
export NSYS_ENABLE="${NSYS_ENABLE:-1}"

export HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export WORKER_NODES=$(scontrol show hostnames $SLURM_NODELIST | tail -n+2)

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

# ARC/some clusters return link-local IPv6 records for Slurm hostnames.
# Ray/vLLM need a routable node address here, so resolve an IPv4 address.
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

resolve_node_ib_ip() {
  local nodename="$1"
  local ip=""

  ip=$(
    srun --nodelist="${nodename}" --nodes=1 --ntasks=1 \
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
      bash -lc '
        ip -o -4 addr show scope global up \
          | awk '"'"'$2 ~ /^ib/ {split($4,a,"/"); print a[1]; exit}'"'"'
      ' 2>/dev/null || true
  )

  slurm_debug "resolve_node_ib_ip(${nodename}) -> ${ip:-<empty>}"
  printf "%s" "${ip}"
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

export HEAD_NODE_IP="$(resolve_host_ip "${HEAD_NODE}")"
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
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL,P2P,TUNING}"

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"

export RAY_PORT="${RAY_PORT:-6378}"
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
echo "RAY_PORT=${RAY_PORT} RAY_ADDRESS=${RAY_ADDRESS}"

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

# === Trace output directory ===
TRACE_BASE="/data/engs-glass/catz0932/inference-traces/vllm/results"
TRACE_RUN_DIR="${TRACE_BASE}/${SLURM_JOB_ID}"

mkdir -p "${TRACE_RUN_DIR}/nsight"
mkdir -p "${TRACE_RUN_DIR}/nccl_logs"

# === Nsight Systems ===
export NSYS_ENABLE="${NSYS_ENABLE:-1}"
export NSYS_DIR="${TRACE_RUN_DIR}/nsight"
export NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cudnn,cublas}"
export NSYS_DELAY="${NSYS_DELAY:-0}"
export NSYS_PROFILE_SERVER="${NSYS_PROFILE_SERVER:-1}"
export NSYS_PROFILE_RAY="${NSYS_PROFILE_RAY:-0}"

# === NCCL logs ===
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_FILE="${TRACE_RUN_DIR}/nccl_logs/nccl_%h_%p.log"

echo "TRACE_RUN_DIR=${TRACE_RUN_DIR}"
echo "NSYS_DIR=${NSYS_DIR}"
echo "NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}"
echo "NSYS_TRACE=${NSYS_TRACE}"
echo "NSYS_DELAY=${NSYS_DELAY}"
echo "NSYS_PROFILE_SERVER=${NSYS_PROFILE_SERVER}"
echo "NSYS_PROFILE_RAY=${NSYS_PROFILE_RAY}"
echo "NSYS_COPY_DEBUG=${NSYS_COPY_DEBUG} (set to 1 with DEBUG_SLURM_SCRIPT=1 for verbose Nsight copy logs)"
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

echo "After venv: python=$(command -v python) ray=$(command -v ray 2>/dev/null || echo '<not on PATH>')"
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
  echo "Error: ray binary not found at ${RAY_BIN}. Install ray into this venv." >&2
  echo "Hint: source \"${VENV_DIR}/bin/activate\" && python -m pip install ray" >&2
  exit 1
fi
echo "Using RAY_BIN=${RAY_BIN}"

export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
HOST="${HOST:-${HEAD_NODE_IP}}"
PORT="${PORT:-8000}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-2}}"
TP="${TP:-1}"
PP="${PP:-${NUM_NODES}}"
EP="${EP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-1}}"
SERVE_SCRIPT="${REPO_ROOT}/serving_scripts/serve_ShareGPT_multi_node.sh"

echo "=== runtime knobs ==="
echo "MODEL_ID=${MODEL_ID} HOST=${HOST} PORT=${PORT} TP=${TP} PP=${PP} EP=${EP}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE} NUM_NODES=${NUM_NODES} CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE} NCCL_NET=${NCCL_NET} NCCL_IB_HCA=${NCCL_IB_HCA}"
echo "NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "NCCL_DEBUG=${NCCL_DEBUG} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
echo "SERVE_SCRIPT=${SERVE_SCRIPT}"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set"
else
  echo "HF_TOKEN is not set"
fi
slurm_debug "VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE} VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM}"

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
echo "Starting head node ${HEAD_NODE} (HEAD_RAY_PID will be set)..."
RAY_HEAD_CMD="$(
  declare -f interface_for_ip
  declare -f interface_has_ip
  declare -f configure_socket_ifnames
)
source \"${VENV_DIR}/bin/activate\"
unset GLOO_SOCKET_IFNAME
export VLLM_HOST_IP=${HEAD_NODE_IP}
configure_socket_ifnames \"${HEAD_NODE_IP}\" 0

if [ \"${NSYS_ENABLE}\" = \"1\" ] && [ \"${NSYS_PROFILE_RAY}\" = \"1\" ]; then
  echo \"Profiling Ray head with Nsight Systems\"
  echo \"Nsight output: ${NSYS_DIR}/ray_head_${HEAD_NODE}.nsys-rep\"

  nsys profile \\
    --force-overwrite=true \\
    --trace=\"${NSYS_TRACE}\" \\
    --sample=none \\
    --delay=\"${NSYS_DELAY}\" \\
    --output=\"${NSYS_DIR}/ray_head_${HEAD_NODE}\" \\
    \"${RAY_BIN}\" start --block \\
      --head \\
      --node-ip-address=${HEAD_NODE_IP} \\
      --port=${RAY_PORT} \\
      --num-gpus=${GPUS_PER_NODE} \\
      --num-cpus=${CPUS_PER_TASK}
else
  \"${RAY_BIN}\" start --block \\
    --head \\
    --node-ip-address=${HEAD_NODE_IP} \\
    --port=${RAY_PORT} \\
    --num-gpus=${GPUS_PER_NODE} \\
    --num-cpus=${CPUS_PER_TASK}
fi"
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
  echo "Starting worker nodes..."
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

if [ \"${NSYS_ENABLE}\" = \"1\" ] && [ \"${NSYS_PROFILE_RAY}\" = \"1\" ]; then
  echo \"Profiling Ray worker ${WORKER} with Nsight Systems\"
  echo \"Nsight output: ${NSYS_DIR}/ray_worker_${WORKER}.nsys-rep\"

  nsys profile \\
    --force-overwrite=true \\
    --trace=\"${NSYS_TRACE}\" \\
    --sample=none \\
    --delay=\"${NSYS_DELAY}\" \\
    --output=\"${NSYS_DIR}/ray_worker_${WORKER}\" \\
    \"${RAY_BIN}\" start --block \\
      --address=${HEAD_NODE_IP}:${RAY_PORT} \\
      --node-ip-address=${WORKER_IP} \\
      --num-gpus=${GPUS_PER_NODE} \\
      --num-cpus=${CPUS_PER_TASK}
else
  \"${RAY_BIN}\" start --block \\
    --address=${HEAD_NODE_IP}:${RAY_PORT} \\
    --node-ip-address=${WORKER_IP} \\
    --num-gpus=${GPUS_PER_NODE} \\
    --num-cpus=${CPUS_PER_TASK}
fi"
    srun \
      --nodelist "${WORKER}" \
      --nodes=1 \
      --ntasks=1 \
      --ntasks-per-node=1 \
      --gpus-per-task="${GPUS_PER_NODE}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      bash -lc "${RAY_WORKER_CMD}" &
    WORKER_RAY_PIDS="${WORKER_RAY_PIDS} $!"
    echo "Worker Ray step pid: $! (WORKER_RAY_PIDS=${WORKER_RAY_PIDS})"
  done
  sleep 20
fi

echo "=== ray status ==="
echo "Checking cluster status..."
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
echo "Starting vLLM server on head node..."

VLLM_TRACE_FLAGS=()
if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_SERVER}" = "1" ]; then
  VLLM_TRACE_FLAGS+=(
    --ray-workers-use-nsight
    --enable-layerwise-nvtx-tracing
    --enable-logging-iteration-details
  )
fi

if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_SERVER}" = "1" ]; then
  echo "Profiling vLLM server and Ray workers with Nsight Systems"
  echo "API-server Nsight output: ${NSYS_DIR}/vllm_api_server_${HEAD_NODE}.nsys-rep"
  nsight_copy_msg "worker traces expected on each node: /tmp/ray/session_latest/logs/nsight/worker_process_*.nsys-rep"
  nsight_copy_msg "PP rank 0 -> ${HEAD_NODE}, PP rank 1 -> ${WORKER_NODES:-<none>}"
  nsight_debug "NSYS_PROFILE_SERVER=1: outer nsys wraps api_server on ${HEAD_NODE}; --ray-workers-use-nsight on Ray actors"

  nsys profile \
    --force-overwrite=true \
    --trace="${NSYS_TRACE}" \
    --sample=none \
    --delay="${NSYS_DELAY}" \
    --output="${NSYS_DIR}/vllm_api_server_${HEAD_NODE}" \
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
until curl -fsS "http://${HEAD_NODE_IP}:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    echo "Server process exited before becoming ready." >&2
    wait "${SERVER_STEP_PID}" || true
    exit 1
  fi
  _health_wait_n=$((_health_wait_n + 1))
  # Every ~60s by default; every loop if DEBUG_SLURM_SCRIPT=1
  if [ "${DEBUG_SLURM_SCRIPT}" = "1" ] || [ "$((_health_wait_n % 12))" -eq 0 ]; then
    echo "Still waiting for http://${HEAD_NODE_IP}:${PORT}/health (attempt ${_health_wait_n}) ..."
  fi
  sleep 5
done
unset _health_wait_n
echo "Server is healthy. Running ${SERVE_SCRIPT} ..."
echo "SP=${SP} SD=${SD}"
HOST="${HEAD_NODE_IP}" PORT="${PORT}" MODEL_ID="${MODEL_ID}" \
  SP="${SP}" SD="${SD}" \
  HEAD_NODE_IP="${HEAD_NODE_IP}" \
  GPUS_PER_NODE="${GPUS_PER_NODE}" CPUS_PER_TASK="${CPUS_PER_TASK}" \
  RAY_PORT="${RAY_PORT}" bash "${SERVE_SCRIPT}" "${SLURM_JOB_ID}" "${HEAD_NODE}"
  
echo "Workload finished. Stopping vLLM/Nsight server process cleanly..."
nsight_copy_msg "pre-shutdown: API server nsight (if any) under ${NSYS_DIR}"
ls -la "${NSYS_DIR}/" 2>/dev/null || nsight_copy_msg "  (no ${NSYS_DIR} yet)"
nsight_peek_on_node "${HEAD_NODE}" "pre-shutdown"
for WORKER in ${WORKER_NODES}; do
  nsight_peek_on_node "${WORKER}" "pre-shutdown"
done

if [ -n "${SERVER_STEP_PID}" ] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
  kill -INT "${SERVER_STEP_PID}" 2>/dev/null || true
  wait "${SERVER_STEP_PID}" 2>/dev/null || true
  SERVER_STEP_PID=""
fi

echo "Stopping Ray background srun steps before copying Nsight reports..."

if [ -n "${HEAD_RAY_PID}" ] && kill -0 "${HEAD_RAY_PID}" 2>/dev/null; then
  kill -TERM "${HEAD_RAY_PID}" 2>/dev/null || true
  wait "${HEAD_RAY_PID}" 2>/dev/null || true
fi
HEAD_RAY_PID=""

for pid in ${WORKER_RAY_PIDS}; do
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
done
WORKER_RAY_PIDS=""

nsight_copy_msg "post-ray-stop: peek nodes before flush sleep"
nsight_peek_on_node "${HEAD_NODE}" "post-ray-stop"
for WORKER in ${WORKER_NODES}; do
  nsight_peek_on_node "${WORKER}" "post-ray-stop"
done

echo "Waiting briefly for Ray/Nsight files to flush..."
sleep 10

nsight_copy_msg "post-flush: peek nodes before copy"
nsight_peek_on_node "${HEAD_NODE}" "post-flush"
for WORKER in ${WORKER_NODES}; do
  nsight_peek_on_node "${WORKER}" "post-flush"
done

echo "Copying Ray worker Nsight reports..."
mkdir -p "${TRACE_RUN_DIR}/ray_worker_nsight"

copy_ray_nsight_from_node() {
  local NODE="$1"
  local dest="${TRACE_RUN_DIR}/ray_worker_nsight/${NODE}"

  nsight_copy_msg "=== copy start: ${NODE} -> ${dest} ==="
  nsight_peek_on_node "${NODE}" "immediately-before-cp"

  srun \
    --overlap \
    --nodelist "${NODE}" \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --mem=1G \
    bash -lc "
      set +e
      echo '[nsight-copy] on '\$(hostname)' (expected ${NODE})'
      echo '[nsight-copy] dest=${dest}'
      echo '[nsight-copy] session_latest ->' \$(readlink -f /tmp/ray/session_latest 2>/dev/null || echo MISSING)
      mkdir -p '${dest}'
      if [ -d /tmp/ray/session_latest/logs/nsight ]; then
        echo '[nsight-copy] source dir listing:'
        ls -la /tmp/ray/session_latest/logs/nsight/ 2>/dev/null || true
        for f in /tmp/ray/session_latest/logs/nsight/*; do
          [ -f \"\$f\" ] && stat -c '[nsight-copy] src %n %s bytes' \"\$f\" 2>/dev/null || true
        done
        echo '[nsight-copy] running cp -v ...'
        cp -v /tmp/ray/session_latest/logs/nsight/*.nsys-rep '${dest}/' 2>&1
        cp_status=\$?
        echo '[nsight-copy] cp exit status='\$cp_status
        echo '[nsight-copy] dest after cp:'
        ls -la '${dest}/' 2>/dev/null || true
        for f in '${dest}'/*; do
          [ -f \"\$f\" ] && stat -c '[nsight-copy] dest %n %s bytes' \"\$f\" 2>/dev/null || true
        done
      else
        echo '[nsight-copy] No /tmp/ray/session_latest/logs/nsight on '\$(hostname)' (slurm node ${NODE})'
        echo '[nsight-copy] /tmp/ray top-level:'
        ls -la /tmp/ray/ 2>/dev/null | head -10 || true
        echo '[nsight-copy] HUNT after failed copy path (may be elsewhere on disk):'
        find /tmp/ray -type f -name '*.nsys-rep' -printf '[nsight-copy]   found %p %s bytes\n' 2>/dev/null | sort || true
        find /tmp -maxdepth 8 -type f -name 'worker_process*.nsys-rep' \
          -printf '[nsight-copy]   found %p %s bytes\n' 2>/dev/null | sort || true
        find /tmp -type f -name '*.qdstrm' -printf '[nsight-copy]   partial %p %s bytes\n' 2>/dev/null | sort || true
      fi
    "
  local srun_status=$?
  nsight_copy_msg "=== copy end: ${NODE} (srun exit ${srun_status}) ==="
  nsight_audit_on_node "${NODE}" "after-copy"
  nsight_summarize_dest "${NODE}"
}

copy_ray_nsight_from_node "${HEAD_NODE}"

for WORKER in ${WORKER_NODES}; do
  copy_ray_nsight_from_node "${WORKER}"
done

nsight_copy_msg "=== final Nsight artifact summary ==="
nsight_copy_msg "API server trace dir: ${NSYS_DIR}"
ls -la "${NSYS_DIR}/" 2>/dev/null || true
for NODE in ${HEAD_NODE} ${WORKER_NODES}; do
  nsight_summarize_dest "${NODE}"
done

echo "Trace files:"
find "${TRACE_RUN_DIR}" -maxdepth 5 -type f -printf "%p %s bytes\n" 2>/dev/null || true